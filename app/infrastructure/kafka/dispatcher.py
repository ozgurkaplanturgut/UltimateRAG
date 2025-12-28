from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional

from aiokafka import AIOKafkaConsumer

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ResponseDispatcher:
    """
    Kafka consumer used by API service.
    Routes response events to per-request asyncio.Queue objects in memory.

    Behavior preserved:
    - per request_id queue
    - if queue is full, drop oldest message
    """

    def __init__(self, topics: List[str], group_id: str, max_queue_size: int = 2000) -> None:
        self._topics = topics
        self._group_id = group_id
        self._max_queue_size = max_queue_size

        self._consumer: Optional[AIOKafkaConsumer] = None
        self._task: Optional[asyncio.Task] = None

        self._streams: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._running = asyncio.Event()

    async def start(self) -> None:
        """Start consumer and dispatch loop."""
        if self._consumer is not None:
            return

        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=self._group_id,
            enable_auto_commit=True,
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
        )
        await self._consumer.start()
        self._running.set()
        self._task = asyncio.create_task(self._run())
        logger.info("ResponseDispatcher started topics=%s group_id=%s",
                    self._topics, self._group_id)

    async def stop(self) -> None:
        """Stop consumer and clear streams."""
        self._running.clear()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

        if self._consumer:
            await self._consumer.stop()

        self._consumer = None
        self._task = None

        async with self._lock:
            self._streams.clear()

        logger.info("ResponseDispatcher stopped")

    async def open_stream(self, request_id: str) -> asyncio.Queue:
        """Create and register a queue for request_id."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        async with self._lock:
            self._streams[request_id] = q
        return q

    async def close_stream(self, request_id: str) -> None:
        """Unregister a queue for request_id."""
        async with self._lock:
            self._streams.pop(request_id, None)

    async def _deliver(self, request_id: str, payload: dict) -> None:
        """Deliver payload to request queue if exists."""
        async with self._lock:
            q = self._streams.get(request_id)

        if not q:
            return

        if q.full():
            try:
                _ = q.get_nowait()
            except Exception:
                pass

        try:
            q.put_nowait(payload)
        except Exception:
            pass

    async def _run(self) -> None:
        """Main consume loop."""
        assert self._consumer is not None

        while self._running.is_set():
            try:
                msg = await self._consumer.getone()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Dispatcher consumer error")
                await asyncio.sleep(0.2)
                continue

            payload = msg.value
            request_id = payload.get("request_id")
            if not request_id:
                continue

            await self._deliver(request_id, payload)
