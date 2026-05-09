"""Tiny disk cache for HTTP responses keyed by query hash.

Used by Source adapters to avoid hammering arXiv / Semantic Scholar across
repeated CLI invocations or REPL sessions. Entries older than `ttl_seconds`
are treated as misses.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


class DiskCache:
    """Filesystem-backed key/value cache. Values are bytes, keys are arbitrary strings."""

    def __init__(self, directory: str | os.PathLike[str], ttl_seconds: int) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    def get(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        if (time.time() - path.stat().st_mtime) > self._ttl:
            return None
        return path.read_bytes()

    def set(self, key: str, value: bytes) -> None:
        path = self._path_for(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(value)
        os.replace(tmp, path)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.bin"
