from __future__ import annotations

from typing import Optional

import redis.asyncio as redis

from app.core.config import get_settings

_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Return singleton async Redis client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client
