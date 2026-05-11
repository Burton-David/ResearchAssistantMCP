"""Async rate limiter — at most one call per `min_interval` seconds, in order.

Process-local; arXiv asks for max one request every 3 seconds, and our CLI/MCP
runs as a single process, so a per-instance lock is enough. Not appropriate
for distributed deployments.

`RateLimiter` is the simple fixed-interval limiter — fine when the upstream's
documented rate is what they actually enforce (arXiv, OpenAlex). For services
where the documented rate is aspirational (Semantic Scholar's "1 RPS" tier
empirically 429s on 4-5 sequential requests at 1.1s spacing), use
`AdaptiveRateLimiter`: it doubles its interval on each 429 and decays back
toward the baseline on success.
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


class AdaptiveRateLimiter:
    """Rate limiter that learns from 429s.

    Starts at `base_interval`. On `record_failure()` (called from the
    HTTP-response path when we see a 429), doubles the current interval
    up to `max_interval`. On `record_success()`, multiplies the current
    interval by `decay` (0.7 = ~30% recovery per success) toward the
    baseline, never below it.

    Why this exists: Semantic Scholar's documented 1-RPS authenticated
    tier actually 429s after ~4 quick requests; we discovered this by
    probing the API directly with our key. A static interval can't
    distinguish "this upstream is being strict today" from "we're
    asking too fast" — adaptive backs off in response to actual
    feedback and recovers when conditions improve.

    The two thresholds:
      - base_interval: the rate we're sure is safe per the docs (1.0s for S2)
      - max_interval: hard ceiling so a sustained outage doesn't pin
        every call at exponentially-long waits (30.0s default)
    """

    def __init__(
        self,
        base_interval_seconds: float,
        *,
        max_interval_seconds: float = 30.0,
        decay: float = 0.7,
    ) -> None:
        if base_interval_seconds <= 0:
            raise ValueError("base_interval_seconds must be positive")
        if max_interval_seconds < base_interval_seconds:
            raise ValueError("max_interval_seconds must be >= base_interval_seconds")
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self._base = base_interval_seconds
        self._max = max_interval_seconds
        self._decay = decay
        self._current = base_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._current - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    def record_failure(self) -> None:
        """Tell the limiter we got rate-limited. Doubles the interval
        (capped at max_interval). Cheap — called from the HTTP path,
        no async needed."""
        self._current = min(self._current * 2, self._max)

    def record_success(self) -> None:
        """Tell the limiter the upstream is healthy. Decays the
        interval geometrically back toward the baseline."""
        if self._current > self._base:
            self._current = max(self._current * self._decay, self._base)

    @property
    def current_interval(self) -> float:
        """Read-only view for diagnostics / tests."""
        return self._current
