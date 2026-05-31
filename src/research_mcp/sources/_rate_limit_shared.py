"""Cross-process variant of :class:`AdaptiveRateLimiter`.

``AdaptiveRateLimiter`` is process-local: each research-mcp process keeps its
own interval and last-call timestamp. When several processes share one API key
— a ``research-mcp repl`` next to Claude Desktop's ``research-mcp serve`` plus
an ad-hoc script — they each respect the limit individually but collectively
exceed it, and the user sees more 429s than they should.

``SharedAdaptiveRateLimiter`` coordinates through a small JSON sidecar guarded
by an ``fcntl`` file lock, so every process on one machine serializes against a
shared minimum interval *and* a shared adaptive backoff (a 429 seen by one
process slows the others too). It is a drop-in for ``AdaptiveRateLimiter``:
same ``acquire`` / ``record_failure`` / ``record_success`` / ``current_interval``
surface.

POSIX only — ``fcntl.flock`` covers macOS and Linux; Windows is deferred, and
the env opt-in that selects this limiter is simply a no-op there. Opt in for
Semantic Scholar with ``RESEARCH_MCP_S2_SHARED_RATELIMIT=1``; it is off by
default so the test suite never touches the shared ``~/.cache`` state.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path.home() / ".cache" / "research-mcp" / "rate-limits"


class SharedAdaptiveRateLimiter:
    """Adaptive rate limiter coordinated across processes via a file lock.

    State lives in ``<state_dir>/<source>.json`` as ``{last_call, interval}``,
    read-updated-written under an exclusive ``fcntl`` lock so concurrent
    processes serialize. ``acquire`` reserves its slot — it writes the wall-clock
    time it *will* fire at while holding the lock — then sleeps outside the lock;
    the blocking lock + IO runs in a worker thread so a sibling process holding
    the lock can't stall the event loop.

    Wall-clock ``time.time()`` is used (not ``time.monotonic`` like the local
    limiter) because timestamps have to be comparable across processes. For a
    politeness limiter the worst case under a clock adjustment is one early or
    one extra-delayed call, which is benign.
    """

    def __init__(
        self,
        base_interval_seconds: float,
        *,
        source: str,
        state_dir: str | os.PathLike[str] | None = None,
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
        self._lock = asyncio.Lock()  # serializes coroutines within this process
        directory = Path(state_dir) if state_dir is not None else _DEFAULT_STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / f"{source}.json"
        self._lock_path = directory / f"{source}.lock"

    async def acquire(self) -> None:
        async with self._lock:
            wait = await asyncio.to_thread(self._reserve_slot)
        if wait > 0:
            await asyncio.sleep(wait)

    def record_failure(self) -> None:
        """Double the shared interval (capped at max) after a 429."""
        self._mutate_interval(lambda i: min(i * 2, self._max))

    def record_success(self) -> None:
        """Decay the shared interval geometrically back toward the baseline."""
        self._mutate_interval(
            lambda i: max(i * self._decay, self._base) if i > self._base else i
        )

    @property
    def current_interval(self) -> float:
        """Read-only view for diagnostics / tests; best-effort, unlocked."""
        return self._read_state()["interval"]

    def _reserve_slot(self) -> float:
        """Blocking. Under an exclusive lock: read state, compute the wait,
        write the reserved ``last_call`` so the next process waits relative to
        us, release. Returns the seconds the caller should sleep."""
        with open(self._lock_path, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                state = self._read_state()
                now = time.time()
                interval = state["interval"]
                wait = max(interval - (now - state["last_call"]), 0.0)
                self._write_state(last_call=now + wait, interval=interval)
                return wait
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)

    def _mutate_interval(self, fn: Callable[[float], float]) -> None:
        # record_* are invoked synchronously as backoff callbacks, so this is a
        # brief blocking lock + IO on the event-loop thread (sub-millisecond —
        # no sleep is ever held under the lock). Best-effort: a lock/IO failure
        # just means the next call reuses the last good interval; never raise
        # into the request path.
        try:
            with open(self._lock_path, "w") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try:
                    state = self._read_state()
                    self._write_state(
                        last_call=state["last_call"],
                        interval=fn(state["interval"]),
                    )
                finally:
                    fcntl.flock(lockf, fcntl.LOCK_UN)
        except OSError as exc:
            _log.warning("shared rate-limit interval update failed (ignored): %s", exc)

    def _read_state(self) -> dict[str, float]:
        # A missing or corrupt sidecar means "first call" — recover to the
        # baseline rather than propagating an error. tmp+rename writes guarantee
        # a reader sees either the old or the new complete file, never a partial.
        try:
            data = json.loads(self._path.read_text())
            return {
                "last_call": float(data["last_call"]),
                "interval": float(data["interval"]),
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return {"last_call": 0.0, "interval": self._base}

    def _write_state(self, *, last_call: float, interval: float) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps({"last_call": last_call, "interval": interval}))
        os.replace(tmp, self._path)
