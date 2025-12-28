from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.infrastructure.mongodb.client import get_database
from app.services.rag.schemas import RagRequest

REQUESTS_COLLECTION = "rag_requests"
MESSAGES_COLLECTION = "rag_messages"


async def create_rag_request_log(req: RagRequest) -> None:
    """Insert initial request log entry with status=queued."""
    db = get_database()
    await db[REQUESTS_COLLECTION].insert_one(
        {
            "request_id": req.request_id,
            "action": req.action,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "doc_id": req.doc_id,
            "source_url": req.source_url,
            "created_at": req.created_at,
            "status": "queued",
        }
    )


async def mark_request_processing(request_id: str) -> None:
    """Mark request as processing."""
    db = get_database()
    await db[REQUESTS_COLLECTION].update_one(
        {"request_id": request_id},
        {"$set": {"status": "processing", "started_at": datetime.utcnow()}},
    )


async def mark_request_success(request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Mark request as success, optionally storing extra fields."""
    db = get_database()
    upd: Dict[str, Any] = {"status": "success",
                           "finished_at": datetime.utcnow()}
    if extra:
        upd.update(extra)
    await db[REQUESTS_COLLECTION].update_one({"request_id": request_id}, {"$set": upd})


async def mark_request_error(request_id: str, error: str) -> None:
    """Mark request as error with error message."""
    db = get_database()
    await db[REQUESTS_COLLECTION].update_one(
        {"request_id": request_id},
        {"$set": {"status": "error", "error": error, "finished_at": datetime.utcnow()}},
    )


async def save_message(
    user_id: Optional[str],
    session_id: str,
    doc_id: str,
    role: str,
    content: str,
    meta: Optional[dict] = None,
) -> None:
    """Append a chat message."""
    db = get_database()
    await db[MESSAGES_COLLECTION].insert_one(
        {
            "user_id": user_id,
            "session_id": session_id,
            "doc_id": doc_id,
            "role": role,
            "content": content,
            "meta": meta,
            "created_at": datetime.utcnow(),
        }
    )


async def get_last_messages_for_session(
    user_id: Optional[str],
    session_id: str,
    doc_id: str,
    max_pairs: int = 20,
) -> List[Dict[str, Any]]:
    """Fetch last N user/assistant message pairs for a session/doc."""
    db = get_database()
    q: Dict[str, Any] = {"session_id": session_id, "doc_id": doc_id}
    if user_id is not None:
        q["user_id"] = user_id

    cursor = db[MESSAGES_COLLECTION].find(q).sort(
        "created_at", -1).limit(max_pairs * 2)
    docs = await cursor.to_list(length=max_pairs * 2)
    docs.reverse()
    return docs
