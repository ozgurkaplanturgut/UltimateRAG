from __future__ import annotations

import time
from typing import Awaitable, Callable, List

FlushCallback = Callable[[str], Awaitable[None]]


class StreamBuffer:
    """
    Buffer small text chunks and flush them by:
    - max_chars threshold
    - max_interval_s since last flush

    Used to reduce Kafka message volume while streaming LLM tokens.
    """

    def __init__(
        self,
        on_flush: FlushCallback,
        *,
        max_chars: int = 256,
        max_interval_s: float = 0.15,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if max_interval_s <= 0:
            raise ValueError("max_interval_s must be > 0")

        self._on_flush = on_flush
        self._max_chars = max_chars
        self._max_interval_s = max_interval_s

        self._buf: List[str] = []
        self._buf_chars = 0
        self._last_flush = time.monotonic()
        self._closed = False

    async def add(self, text: str) -> None:
        """Add text and flush if thresholds exceeded."""
        if self._closed or not text:
            return

        self._buf.append(text)
        self._buf_chars += len(text)

        now = time.monotonic()
        if self._buf_chars >= self._max_chars or (now - self._last_flush) >= self._max_interval_s:
            await self.flush()

    async def flush(self) -> None:
        """Flush buffered text using callback."""
        if self._closed:
            return

        if not self._buf:
            self._last_flush = time.monotonic()
            return

        payload = "".join(self._buf)
        self._buf.clear()
        self._buf_chars = 0
        self._last_flush = time.monotonic()
        await self._on_flush(payload)

    async def close(self) -> None:
        """Flush and prevent further adds."""
        if self._closed:
            return
        await self.flush()
        self._closed = True
