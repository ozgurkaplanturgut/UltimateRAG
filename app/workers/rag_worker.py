from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Optional, Set

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from httpx import Timeout
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.mongodb.client import close_mongo, init_mongo
from app.infrastructure.mongodb.rag_repository import (
    mark_request_error,
    mark_request_processing,
    mark_request_success,
)
from app.infrastructure.qdrant.client import delete_doc_points, ensure_collection, get_qdrant, doc_meta_point_id
from app.infrastructure.redis.client import get_redis
from app.services.document.service import build_chunks, download_text, upsert_chunks, upsert_doc_meta
from app.services.rag.schemas import RagRequest, RagResponseEvent
from app.services.rag.service import process_query, send_end, send_event, send_error_and_mark

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

RETRY_HEADER = b"x-retry"


def get_retry_count(msg) -> int:
    """Read retry count from Kafka message headers."""
    try:
        for k, v in (msg.headers or []):
            if k == RETRY_HEADER and v is not None:
                return int(v.decode("utf-8"))
    except Exception:
        pass
    return 0


def with_retry_header(headers, retry_count: int):
    """Return headers with updated x-retry."""
    h = list(headers or [])
    h = [(k, v) for (k, v) in h if k != RETRY_HEADER]
    h.append((RETRY_HEADER, str(retry_count).encode("utf-8")))
    return h


def backoff_seconds(retry_count: int) -> float:
    """Exponential backoff + jitter (same as original)."""
    base = float(settings.retry_base_seconds)
    cap = float(settings.retry_max_seconds)
    delay = min(cap, base * (2 ** max(0, retry_count - 1)))
    jitter = (time.time() % 1.0) * 0.25 * delay
    return delay + jitter


async def commit(consumer: AIOKafkaConsumer, msg) -> None:
    """Manual commit (enable_auto_commit=False)."""
    tp = TopicPartition(msg.topic, msg.partition)
    offsets = {tp: OffsetAndMetadata(msg.offset + 1, "")}
    await consumer.commit(offsets=offsets)


async def send_dlq(producer: AIOKafkaProducer, original_msg, err: str) -> None:
    """Send message to DLQ with metadata (behavior preserved)."""
    payload = {
        "topic": original_msg.topic,
        "partition": original_msg.partition,
        "offset": original_msg.offset,
        "timestamp_ms": int(time.time() * 1000),
        "error": err,
        "headers": [
            (
                k.decode("utf-8") if isinstance(k,
                                                (bytes, bytearray)) else str(k),
                v.decode("utf-8") if isinstance(v, (bytes, bytearray)
                                                ) else (None if v is None else str(v)),
            )
            for (k, v) in (original_msg.headers or [])
        ],
        "value": original_msg.value,
    }
    await producer.send_and_wait(settings.rag_dlq_topic, key=original_msg.key or "", value=payload)


async def idempotency_acquire(request_id: str) -> bool:
    """Acquire idempotency lock in Redis (SET NX EX)."""
    redis = get_redis()
    key = f"rag:done:{request_id}"
    ok = await redis.set(key, "1", ex=int(settings.idempotency_ttl_seconds), nx=True)
    return bool(ok)


async def idempotency_release_on_fail(request_id: str) -> None:
    """Release idempotency lock to allow retry."""
    redis = get_redis()
    with suppress(Exception):
        await redis.delete(f"rag:done:{request_id}")


async def handle_upload(req: RagRequest, qdrant, producer: AIOKafkaProducer, client: AsyncOpenAI) -> None:
    """Upload pipeline (preserves original steps)."""
    await asyncio.to_thread(ensure_collection, qdrant)
    await asyncio.to_thread(delete_doc_points, qdrant, req.user_id, req.doc_id)

    if not req.source_url:
        raise RuntimeError("source_url missing")

    text = await download_text(req.source_url)
    chunks = await build_chunks(text)

    meta_id = doc_meta_point_id(req.user_id or "", req.doc_id)
    await upsert_doc_meta(
        qdrant,
        user_id=req.user_id,
        doc_id=req.doc_id,
        meta_id=meta_id,
        client=client,
        text=text,
    )

    count = await upsert_chunks(qdrant, user_id=req.user_id, doc_id=req.doc_id, chunks=chunks, client=client)

    await send_event(producer, RagResponseEvent(request_id=req.request_id, event_type="chunk", content="uploaded"))
    await mark_request_success(req.request_id, extra={"chunks": count})


async def handle_delete(req: RagRequest, qdrant, producer: AIOKafkaProducer) -> None:
    """Delete pipeline (preserves original steps)."""
    await asyncio.to_thread(ensure_collection, qdrant)
    await asyncio.to_thread(delete_doc_points, qdrant, req.user_id, req.doc_id)

    await send_event(producer, RagResponseEvent(request_id=req.request_id, event_type="chunk", content="deleted"))
    await mark_request_success(req.request_id)


async def process_rag_request(
    req: RagRequest,
    producer: AIOKafkaProducer,
    client: AsyncOpenAI,
    cross: CrossEncoder,
) -> None:
    """
    Process a single RagRequest.
    Preserves:
    - mark processing
    - query handles its own end/error; upload/delete wrapper guarantees end
    """
    qdrant = get_qdrant()

    await mark_request_processing(req.request_id)

    if req.action == "query":
        # Query pipeline guarantees end inside process_query
        await process_query(req=req, producer=producer, client=client, cross=cross, qdrant=qdrant)
        return

    err: Optional[str] = None
    try:
        if req.action == "upload":
            await handle_upload(req, qdrant, producer, client)
        elif req.action == "delete":
            await handle_delete(req, qdrant, producer)
        else:
            raise RuntimeError(f"unknown action: {req.action}")
    except Exception as e:
        err = str(e)
        logger.exception("RAG action failed", extra={
                         "request_id": req.request_id, "action": req.action, "error": err})
        await send_error_and_mark(producer, req.request_id, err)
        with suppress(Exception):
            await mark_request_error(req.request_id, err)
    finally:
        await send_end(producer, req.request_id)

    if err is None:
        await mark_request_success(req.request_id)


async def worker_main() -> None:
    """Main worker loop: consume -> idempotency -> process -> retry/DLQ -> commit."""
    logger.info("Starting RAG worker...")

    await init_mongo()

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=Timeout(
            connect=10.0, read=settings.stream_hard_timeout, write=10.0, pool=10.0),
    )

    cross = CrossEncoder(settings.rag_cross_encoder_model,
                         device=settings.rag_cross_encoder_device)

    consumer = AIOKafkaConsumer(
        settings.rag_request_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.rag_worker_group_id,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        session_timeout_ms=30000,
        heartbeat_interval_ms=3000,
        max_poll_interval_ms=10 * 60 * 1000,
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        key_serializer=lambda v: v.encode(
            "utf-8") if isinstance(v, str) else v,
        value_serializer=lambda v: json.dumps(
            v, ensure_ascii=False, default=str).encode("utf-8"),
        acks="all",
        linger_ms=10,
    )

    await consumer.start()
    await producer.start()

    semaphore = asyncio.Semaphore(settings.worker_max_concurrency)
    in_flight: Set[asyncio.Task] = set()

    async def _handle_message(msg) -> None:
        try:
            try:
                req = RagRequest.model_validate(msg.value)
            except Exception as e:
                with suppress(Exception):
                    await send_dlq(producer, msg, f"validation_error: {e}")
                await commit(consumer, msg)
                return

            request_id = req.request_id
            retry_count = get_retry_count(msg)

            first = await idempotency_acquire(request_id)
            if not first:
                await commit(consumer, msg)
                return

            try:
                await process_rag_request(
                    req,
                    producer=producer,
                    client=client,
                    cross=cross,
                )
                await commit(consumer, msg)
                return
            except Exception as e:
                logger.exception(
                    "RAG request failed request_id=%s", request_id)
                await idempotency_release_on_fail(request_id)

                if retry_count < int(settings.max_retries):
                    delay = backoff_seconds(retry_count + 1)
                    await asyncio.sleep(delay)

                    headers = with_retry_header(msg.headers, retry_count + 1)
                    await producer.send_and_wait(
                        settings.rag_request_topic,
                        key=request_id,
                        value=msg.value,
                        headers=headers,
                    )
                    await commit(consumer, msg)
                    return

                with suppress(Exception):
                    await send_dlq(producer, msg, f"max_retries_exceeded: {e}")
                await commit(consumer, msg)

        finally:
            semaphore.release()

    try:
        async for msg in consumer:
            await semaphore.acquire()
            task = asyncio.create_task(_handle_message(msg))
            in_flight.add(task)
            task.add_done_callback(lambda t: in_flight.discard(t))
    finally:
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

        with suppress(KafkaError):
            await consumer.stop()
        with suppress(KafkaError):
            await producer.stop()

        await close_mongo()
        logger.info("RAG worker stopped")


if __name__ == "__main__":
    asyncio.run(worker_main())
