"""OpenAI Embeddings API adapter.

Truncates oversized inputs locally with tiktoken before they reach the API.
text-embedding-3-* caps at 8192 tokens; we truncate to 8000 to leave a
margin for tokenization quirks. A truncation-then-decode roundtrip is
mildly lossy on rare BPE boundaries but keeps the embedding quality
indistinguishable from the unmodified text for the truncated portion —
the meaningful loss is the dropped tail, which the user can avoid by
chunking the document themselves before ingest.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Final

import tiktoken
from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

# OpenAI's documented per-request batch ceiling is 2048 inputs.
_MAX_BATCH: Final = 2048

# text-embedding-3-* hard limit is 8192 tokens; we leave a small margin
# in case the local tokenizer disagrees with the server one on edge cases.
_MAX_TOKENS_PER_INPUT: Final = 8000

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
        self._encoder = tiktoken.encoding_for_model(model)

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        truncated = [self._truncate(t) for t in texts]
        # OpenAI rejects empty strings. If a caller hands us all-whitespace
        # input, the truncated form is "" — substitute a single space so the
        # batch indices line up with the input order.
        sanitized = [t if t else " " for t in truncated]
        out: list[list[float]] = []
        for start in range(0, len(sanitized), _MAX_BATCH):
            batch = sanitized[start : start + _MAX_BATCH]
            response = await self._client.embeddings.create(
                model=self.model, input=batch
            )
            out.extend(item.embedding for item in response.data)
        return out

    def _truncate(self, text: str) -> str:
        tokens = self._encoder.encode(text, disallowed_special=())
        if len(tokens) <= _MAX_TOKENS_PER_INPUT:
            return text
        _log.info(
            "truncating input from %d to %d tokens for embedding",
            len(tokens),
            _MAX_TOKENS_PER_INPUT,
        )
        return self._encoder.decode(tokens[:_MAX_TOKENS_PER_INPUT])
