"""Async rate limiter — at most one call per `min_interval` seconds, in order.

Process-local; arXiv asks for max one request every 3 seconds, and our CLI/MCP
runs as a single process, so a per-instance lock is enough. Not appropriate
for distributed deployments.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
