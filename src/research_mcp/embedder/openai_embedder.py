"""OpenAI Embeddings API adapter."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Final

from openai import AsyncOpenAI

# OpenAI's documented per-request batch ceiling is 2048 inputs.
_MAX_BATCH: Final = 2048

_DIMENSIONS: Final[dict[str, int]] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """OpenAI embeddings.

    Defaults to `text-embedding-3-small` (1536-dim). Pass `model="text-embedding-3-large"`
    for the 3072-dim variant. Reads `OPENAI_API_KEY` from the environment unless `api_key`
    is passed explicitly.
    """

    dimension: int

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if model not in _DIMENSIONS:
            raise ValueError(
                f"unknown embedding model {model!r}; "
                f"supported: {sorted(_DIMENSIONS)}"
            )
        self.model = model
        self.dimension = _DIMENSIONS[model]
        self._client = client or AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = list(texts[start : start + _MAX_BATCH])
            response = await self._client.embeddings.create(model=self.model, input=batch)
            out.extend(item.embedding for item in response.data)
        return out
