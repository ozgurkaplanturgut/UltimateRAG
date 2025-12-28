from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import suppress
from functools import partial
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.stream_buffer import StreamBuffer
from app.infrastructure.mongodb.client import get_database
from app.infrastructure.mongodb.rag_repository import (
    get_last_messages_for_session,
    mark_request_error,
    mark_request_processing,
    mark_request_success,
    save_message,
)
from app.infrastructure.qdrant.client import doc_meta_point_id, ensure_collection, filter_doc
from app.services.rag.answer import stream_answer
from app.services.rag.rerank import rerank_windows
from app.services.rag.retrieve import hybrid_retrieve_with_rewrites, prf_snippets
from app.services.rag.rewrite import rewrite_queries
from app.services.rag.router import route_rag_query
from app.services.rag.schemas import RagRequest, RagResponseEvent

settings = get_settings()
logger = logging.getLogger(__name__)


async def run_blocking(fn, *args, **kwargs):
    """Run sync work in a thread."""
    return await asyncio.to_thread(partial(fn, *args, **kwargs))


def backoff_seconds(retry_count: int) -> float:
    """Exponential backoff with jitter (preserves original behavior)."""
    base = float(settings.retry_base_seconds)
    cap = float(settings.retry_max_seconds)
    delay = min(cap, base * (2 ** max(0, retry_count - 1)))
    jitter = random.random() * 0.25 * delay
    return delay + jitter


async def send_event(producer, ev: RagResponseEvent) -> None:
    """Send response event to Kafka (same topic/key/value behavior)."""
    await producer.send_and_wait(
        settings.rag_response_topic,
        key=ev.request_id,
        value=ev.model_dump(mode="json"),
    )


async def send_error_and_mark(producer, request_id: str, err: str) -> None:
    """Send error event + mark request error (best-effort)."""
    with suppress(Exception):
        await send_event(
            producer,
            RagResponseEvent(request_id=request_id,
                             event_type="error", content=err),
        )
    with suppress(Exception):
        await mark_request_error(request_id, err)


async def send_end(producer, request_id: str) -> None:
    """Always send end event (best-effort)."""
    with suppress(Exception):
        await send_event(
            producer,
            RagResponseEvent(request_id=request_id,
                             event_type="end", content=""),
        )


async def build_conversation_text(history_docs: List[Dict[str, Any]]) -> str:
    """Build compact conversation text from last 8 turns."""
    conv_turns: List[str] = []
    for m in (history_docs or [])[-8:]:
        role = (m.get("role") or "").upper()
        content = (m.get("content") or "").strip()
        if role and content:
            conv_turns.append(f"{role}: {content}")
    return "\n".join(conv_turns).strip()


async def get_followup_cached_contexts(
    user_id: Optional[str],
    session_id: str,
    doc_id: str,
) -> List[str]:
    """
    FOLLOWUP mode: reuse last assistant message meta.support as contexts.
    Preserves original behavior: query Mongo directly for last assistant message.
    """
    try:
        db = get_database()
        last_assistant = await db.rag_messages.find_one(
            {"user_id": user_id, "session_id": session_id,
                "doc_id": doc_id, "role": "assistant"},
            sort=[("created_at", -1)],
            projection={"meta.support": 1},
        )
        cached = None
        if last_assistant:
            cached = (last_assistant.get("meta") or {}).get("support")

        if isinstance(cached, list) and cached:
            return [(s.get("quote") or "") for s in cached if isinstance(s, dict) and s.get("quote")]
        return []
    except Exception:
        return []


async def process_query(
    *,
    req: RagRequest,
    producer,
    client: AsyncOpenAI,
    cross: CrossEncoder,
    qdrant: QdrantClient,
) -> None:
    """
    Query pipeline (preserves original behavior):
    - Load history
    - Router (FOLLOWUP/RETRIEVE)
    - Save user message
    - If RETRIEVE: doc_meta -> PRF -> rewrite -> hybrid retrieve -> window rerank
    - If FOLLOWUP: reuse cached support quotes
    - Stream answer with StreamBuffer and watchdog timeouts
    - Always send END
    - On success save assistant message + mark success with extra stats
    """
    if not req.prompt or not req.session_id:
        raise RuntimeError("prompt/session_id missing")

    await run_blocking(ensure_collection, qdrant)

    history_docs = await get_last_messages_for_session(
        req.user_id, req.session_id, req.doc_id, max_pairs=20
    )
    conversation_text = await build_conversation_text(history_docs)

    decision = await route_rag_query(client, req.prompt, conversation_text)
    logger.info(
        "RAG router route=%s conf=%.2f reason=%s request_id=%s doc_id=%s user_id=%s",
        decision.route,
        float(decision.confidence),
        decision.reason,
        req.request_id,
        req.doc_id,
        req.user_id,
    )

    await save_message(req.user_id, req.session_id, req.doc_id, "user", req.prompt)

    contexts: List[str] = []
    support: List[dict] = []

    if decision.route == "RETRIEVE":
        meta_id = doc_meta_point_id(req.user_id or "", req.doc_id)
        meta = await run_blocking(
            qdrant.retrieve,
            collection_name=settings.qdrant_collection,
            ids=[meta_id],
            with_payload=True,
        )

        doc_profile: Dict[str, Any] = {}
        if meta and meta[0] and meta[0].payload:
            doc_profile = meta[0].payload.get("doc_profile") or {}

        flt: Filter = filter_doc(req.user_id, req.doc_id)

        snippets = await prf_snippets(client, qdrant, req.prompt, flt, limit=settings.rag_prf_limit)
        rewrites = await rewrite_queries(
            client,
            req.prompt,
            doc_profile,
            snippets,
            history_docs if settings.rag_rewrite_use_history else None,
        )
        rewrite_pack = [req.prompt] + rewrites

        candidates = await hybrid_retrieve_with_rewrites(client, qdrant, rewrite_pack, flt)
        reranked = await run_blocking(rerank_windows, cross, candidates, rewrite_pack)

        contexts = [wtext for (_pid, _score, wtext, _payload) in reranked]
        support = [{"pid": pid, "quote": (wtext or "")[:400]} for (
            pid, _score, wtext, _payload) in reranked[:5]]
    else:
        contexts = await get_followup_cached_contexts(req.user_id, req.session_id, req.doc_id)

    logger.info(
        "RAG contexts=%d route=%s request_id=%s doc_id=%s user_id=%s",
        len(contexts),
        decision.route,
        req.request_id,
        req.doc_id,
        req.user_id,
    )

    buffer = StreamBuffer(
        lambda chunk: send_event(
            producer,
            RagResponseEvent(request_id=req.request_id,
                             event_type="chunk", content=chunk),
        ),
        max_chars=256,
        max_interval_s=0.05,
    )

    full: List[str] = []
    error_msg: Optional[str] = None

    start_ts = time.monotonic()
    last_token_ts = start_ts
    got_first_token = False

    async def stream_llm() -> None:
        nonlocal last_token_ts, got_first_token
        async for delta in stream_answer(client, req.prompt, contexts, conversation=conversation_text):
            if not delta:
                continue
            got_first_token = True
            last_token_ts = time.monotonic()
            full.append(delta)
            await buffer.add(delta)

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if not got_first_token and (now - start_ts > settings.stream_first_token_timeout):
                raise TimeoutError(
                    f"first_token_timeout_after_{settings.stream_first_token_timeout}s")
            if got_first_token and (now - last_token_ts > settings.stream_idle_timeout):
                raise TimeoutError(
                    f"stream_idle_timeout_after_{settings.stream_idle_timeout}s")

    stream_task = asyncio.create_task(stream_llm())
    watchdog_task = asyncio.create_task(watchdog())

    try:
        await stream_task
    except Exception as e:
        error_msg = str(e)
        logger.exception("RAG streaming failed", extra={
                         "request_id": req.request_id, "error": error_msg})
        await send_error_and_mark(producer, req.request_id, error_msg)
    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task
        with suppress(Exception):
            await buffer.close()
        await send_end(producer, req.request_id)

    if error_msg is None:
        ans = "".join(full).strip()
        msg_meta: Dict[str, Any] = {
            "route": decision.route, "router_conf": float(decision.confidence)}
        if decision.route == "RETRIEVE" and support:
            msg_meta["support"] = support

        await save_message(req.user_id, req.session_id, req.doc_id, "assistant", ans, meta=msg_meta)

        await mark_request_success(
            req.request_id,
            extra={
                "route": decision.route,
                "router_confidence": float(decision.confidence),
                "contexts": len(contexts),
                "support": len(support) if support else 0,
            },
        )


async def process_non_query_action(
    *,
    req: RagRequest,
    producer,
    action_fn,
) -> None:
    """
    Wrapper for upload/delete:
    - mark processing
    - run action
    - on error: error event + mark_error
    - always: end event
    """
    await mark_request_processing(req.request_id)

    err: Optional[str] = None
    try:
        await action_fn()
    except Exception as e:
        err = str(e)
        logger.exception("RAG action failed", extra={
                         "request_id": req.request_id, "action": req.action, "error": err})
        await send_error_and_mark(producer, req.request_id, err)
    finally:
        await send_end(producer, req.request_id)

    if err is None:
        await mark_request_success(req.request_id)
