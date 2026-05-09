"""Deterministic in-process Embedder.

Used by the REPL default wiring and by tests. Same input text always produces
the same vector, so test assertions over similarity scores are stable.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


class FakeEmbedder:
    """Hash-based embedder. Pure CPU, no network, no API keys.

    The vector is derived from SHA-256 of the input text, expanded to
    `dimension` floats and L2-normalized. Identical inputs produce identical
    outputs; cosine similarity between unrelated strings is near zero, between
    a string and itself is exactly 1.0.
    """

    dimension: int

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        floats: list[float] = []
        i = 0
        while len(floats) < self.dimension:
            chunk = seed[i % len(seed) : (i % len(seed)) + 4] or seed[:4]
            i += 4
            seed = hashlib.sha256(seed).digest()
            value = int.from_bytes(chunk, "big", signed=False) / 2**32
            floats.append(value * 2.0 - 1.0)
        norm = math.sqrt(sum(f * f for f in floats)) or 1.0
        return [f / norm for f in floats]
