"""Sentence-transformers embedding adapter.

A drop-in alternative to `OpenAIEmbedder` for users who don't want to
depend on the OpenAI API. Wraps any HuggingFace SentenceTransformers
model — the default is `BAAI/bge-base-en-v1.5` (768d), which scores
near the top of MTEB retrieval benchmarks for English academic text.

Lazy-loads the model on first `embed()` call: HuggingFace model loading
takes 2-5 seconds on first use and downloads ~440 MB the very first
time. Doing it eagerly in `__init__` would block the MCP server
handshake; deferring it to first use keeps startup snappy at the cost
of a one-time pause on the first ingest. A threading.Lock prevents two
concurrent first-use embeds from each loading the model.

Optional. Install via `pip install research-mcp[sentence-transformers]`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any, Final

_log = logging.getLogger(__name__)

# Known dimensions for popular default models. Used so `dimension` can be
# read before the model is loaded — the wiring layer needs it to build
# the Index. For unknown model names we resolve by loading eagerly.
_KNOWN_DIMENSIONS: Final[dict[str, int]] = {
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
}

DEFAULT_MODEL: Final = "BAAI/bge-base-en-v1.5"


class SentenceTransformersEmbedder:
    """An `Embedder` backed by a HuggingFace `SentenceTransformer` model.

    Construction is cheap; the model itself loads on the first `embed()`
    call. Pass `device="cpu"` / `"mps"` / `"cuda"` to override the
    default device probe. Read `RESEARCH_MCP_EMBEDDER_DEVICE` from the
    environment if no explicit device is passed.
    """

    dimension: int

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model
        self._device = device or os.environ.get("RESEARCH_MCP_EMBEDDER_DEVICE")
        self._batch_size = int(
            os.environ.get("RESEARCH_MCP_ST_BATCH_SIZE", batch_size)
        )
        self._model: Any = None
        self._load_lock = threading.Lock()
        if model in _KNOWN_DIMENSIONS:
            self.dimension = _KNOWN_DIMENSIONS[model]
        else:
            # Unknown model — load eagerly so we can ask it for its dim.
            loaded = self._load_model()
            self.dimension = int(loaded.get_sentence_embedding_dimension())

    def _load_model(self) -> Any:
        """Idempotent, thread-safe model loader. Returns the loaded model."""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import (  # type: ignore[import-not-found]  # optional extra
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "SentenceTransformersEmbedder requires the optional "
                    "'sentence-transformers' extra. Install with:\n"
                    "    pip install 'research-mcp[sentence-transformers]'"
                ) from exc
            _log.info("loading sentence-transformers model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
            return self._model

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts))

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        # `convert_to_numpy=True` returns an ndarray; tolist() yields plain
        # Python lists which match the protocol's Sequence[Sequence[float]]
        # return type and serialize cleanly through the index sidecar.
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        result: list[list[float]] = embeddings.tolist()
        return result
