from __future__ import annotations

from enum import Enum
from typing import Optional

from app.infrastructure.redis.client import get_redis


class DocState(str, Enum):
    """Document processing state."""
    UPLOADING = "UPLOADING"
    READY = "READY"
    DELETING = "DELETING"


def _key(user_id: Optional[str], doc_id: str) -> str:
    """Generate Redis key for document state."""
    uid = user_id or "anonymous"
    return f"rag:doc:{uid}:{doc_id}:status"


async def set_doc_state(user_id: Optional[str], doc_id: str, state: DocState) -> None:
    """Set the document processing state in Redis."""
    redis = get_redis()
    await redis.set(_key(user_id, doc_id), state.value)


async def get_doc_state(user_id: Optional[str], doc_id: str) -> Optional[str]:
    """Get the document processing state from Redis."""
    redis = get_redis()
    return await redis.get(_key(user_id, doc_id))


async def clear_doc_state(user_id: Optional[str], doc_id: str) -> None:
    """Clear the document processing state from Redis."""
    redis = get_redis()
    await redis.delete(_key(user_id, doc_id))
