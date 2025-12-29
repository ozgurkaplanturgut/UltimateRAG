from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import AsyncIterator, Optional

from aiokafka import AIOKafkaProducer
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.infrastructure.kafka.dispatcher import ResponseDispatcher
from app.infrastructure.mongodb.rag_repository import create_rag_request_log
from app.infrastructure.redis.doc_state import get_doc_state, DocState
from app.services.rag.schemas import RagRequest

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/rag", tags=["rag"])


def _sse(data: dict) -> str:
    """Format data as Server-Sent Event (SSE)."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_ping() -> str:
    return ": ping\n\n"


@router.post("/documents/{doc_id}/upload")
async def upload_document(
    request: Request,
    doc_id: str,
    source_url: str = Query(...),
    filename: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
) -> dict:
    """Endpoint to upload a document for RAG processing."""
    producer: AIOKafkaProducer = getattr(request.app.state, "producer", None)
    if producer is None:
        raise HTTPException(status_code=500, detail="Kafka producer not ready")

    request_id = str(uuid.uuid4())
    job = RagRequest(
        request_id=request_id,
        action="upload",
        user_id=user_id,
        doc_id=doc_id,
        source_url=source_url,
        filename=filename,
        created_at=datetime.utcnow(),
    )

    await create_rag_request_log(job)

    await producer.send_and_wait(
        settings.rag_request_topic,
        key=request_id,
        value=job.model_dump(mode="json"),
    )

    return {"request_id": request_id, "status": "queued"}


@router.delete("/documents/{doc_id}/delete")
async def delete_document(
    request: Request,
    doc_id: str,
    user_id: Optional[str] = Query(default=None),
) -> dict:
    """Endpoint to delete a document for RAG processing."""
    producer: AIOKafkaProducer = getattr(request.app.state, "producer", None)
    if producer is None:
        raise HTTPException(status_code=500, detail="Kafka producer not ready")

    request_id = str(uuid.uuid4())
    job = RagRequest(
        request_id=request_id,
        action="delete",
        user_id=user_id,
        doc_id=doc_id,
        created_at=datetime.utcnow(),
    )

    await create_rag_request_log(job)

    await producer.send_and_wait(
        settings.rag_request_topic,
        key=request_id,
        value=job.model_dump(mode="json"),
    )

    return {"request_id": request_id, "status": "queued"}


@router.get("/stream/{doc_id}")
async def rag_stream(
    request: Request,
    doc_id: str,
    prompt: str = Query(...),
    session_id: str = Query(...),
    user_id: Optional[str] = Query(default=None),
) -> StreamingResponse:
    """Endpoint to handle RAG query streaming via Server-Sent Events (SSE)."""
    producer: AIOKafkaProducer = getattr(request.app.state, "producer", None)
    dispatcher: ResponseDispatcher = getattr(request.app.state, "dispatcher", None)

    if producer is None or dispatcher is None:
        raise HTTPException(status_code=500, detail="Kafka/dispatcher not ready")

    # 🔐 REDIS STATE GATE
    state = await get_doc_state(user_id, doc_id)
    if state in (DocState.UPLOADING.value, DocState.DELETING.value):
        raise HTTPException(
            status_code=409,
            detail=f"Document is currently {state.lower()}, please try again later.",
        )

    request_id = str(uuid.uuid4())
    q = await dispatcher.open_stream(request_id)

    job = RagRequest(
        request_id=request_id,
        action="query",
        user_id=user_id,
        session_id=session_id,
        doc_id=doc_id,
        prompt=prompt,
        created_at=datetime.utcnow(),
    )

    await create_rag_request_log(job)

    await producer.send_and_wait(
        settings.rag_request_topic,
        key=request_id,
        value=job.model_dump(mode="json"),
    )

    heartbeat_s = float(settings.sse_heartbeat_s)
    max_idle_s = float(settings.sse_max_idle_s)
    max_total_s = float(settings.sse_max_total_s)

    async def event_generator() -> AsyncIterator[str]:
        started = time.monotonic()
        last_activity = started

        try:
            yield _sse(
                {
                    "request_id": request_id,
                    "event_type": "started",
                    "content": "",
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

            while True:
                if await request.is_disconnected():
                    break

                now = time.monotonic()
                if now - started > max_total_s:
                    yield _sse(
                        {
                            "request_id": request_id,
                            "event_type": "error",
                            "content": "stream_timeout",
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    )
                    yield _sse(
                        {
                            "request_id": request_id,
                            "event_type": "end",
                            "content": "",
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    )
                    break

                try:
                    payload = await asyncio.wait_for(q.get(), timeout=heartbeat_s)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_activity > max_idle_s:
                        yield _sse(
                            {
                                "request_id": request_id,
                                "event_type": "error",
                                "content": f"stream_idle_timeout_after_{max_idle_s}s",
                                "created_at": datetime.utcnow().isoformat(),
                            }
                        )
                        yield _sse(
                            {
                                "request_id": request_id,
                                "event_type": "end",
                                "content": "",
                                "created_at": datetime.utcnow().isoformat(),
                            }
                        )
                        break
                    yield _sse_ping()
                    continue

                last_activity = time.monotonic()
                yield _sse(payload)

                if payload.get("event_type") in ("end", "error"):
                    break

        finally:
            await dispatcher.close_stream(request_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
