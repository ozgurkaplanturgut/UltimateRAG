from __future__ import annotations

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def init_mongo() -> None:
    """
    Initialize a single MongoDB client and database handle.
    Creates required indexes for chat history retrieval.
    """
    global _client, _db
    if _client is not None:
        return

    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongo_dsn)
    _db = _client[settings.mongo_db_name]

    await _db["rag_messages"].create_index(
        [("user_id", 1), ("session_id", 1), ("doc_id", 1), ("created_at", -1)],
        background=True,
    )


async def close_mongo() -> None:
    """Close Mongo client."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


def get_database() -> AsyncIOMotorDatabase:
    """Get initialized database handle or fail-fast."""
    if _db is None:
        raise RuntimeError("MongoDB is not initialized")
    return _db
