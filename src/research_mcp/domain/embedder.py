"""Embedder protocol: convert text to fixed-dimension vectors."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """An embedder.

    `dimension` is the output vector size. Must be a class attribute or property
    so the Index can validate compatibility at wire-up time.

    `embed` is batched — implementations should pack inputs into a single API call
    where the underlying provider supports it.
    """

    dimension: int

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding per input text, in the same order.

        `len(returned) == len(texts)` must hold. Each returned embedding has
        length `self.dimension`.
        """
        ...
