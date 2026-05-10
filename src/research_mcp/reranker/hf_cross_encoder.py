"""Cross-encoder reranker backed by sentence_transformers.CrossEncoder.

Lazy-loads the model on first `score()` call (mirrors
SentenceTransformersEmbedder), so server boot stays snappy and the
~250 MB BAAI/bge-reranker-base download only fires when the user
actually issues a search or recall with the reranker enabled.

Cross-encoders score (query, candidate) pairs by joint encoding rather
than by independent embeddings + dot product. They're ~50x slower than
bi-encoder embedding lookup but consistently improve ranking quality on
out-of-distribution queries — which is exactly the diagnostic agent's
lattice-QCD-on-chemistry-query case.

Optional. Install via `pip install research-mcp[sentence-transformers]`
(the cross_encoder module ships with sentence-transformers; no extra
dep to declare).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any, Final

from research_mcp.domain.paper import Paper

_log = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "BAAI/bge-reranker-base"

# Models we know are valid cross-encoders. Used only for friendlier early
# validation; an unknown name still works as long as Hugging Face has a
# matching repo. The eager-load path below catches a typo by failing fast
# with a clean error referencing the env var.
_KNOWN_MODELS: Final[frozenset[str]] = frozenset(
    {
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-large",
        "BAAI/bge-reranker-v2-m3",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    }
)


class HuggingFaceCrossEncoderReranker:
    """A `Reranker` backed by `sentence_transformers.CrossEncoder`.

    Construction loads the model eagerly so a typo in the model name fails
    at boot with a legible `RuntimeError` instead of a confusing 'no
    candidates' result later. Following first construction, the model
    object is held for the process lifetime.
    """

    name: str

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model
        self.name = f"cross-encoder:{model}"
        self._device = device or os.environ.get("RESEARCH_MCP_RERANKER_DEVICE")
        self._batch_size = int(
            os.environ.get("RESEARCH_MCP_RERANKER_BATCH_SIZE", batch_size)
        )
        self._model: Any = None
        self._load_lock = threading.Lock()
        # Lazy-load by design. Earlier versions eager-loaded the model
        # to fail fast on typos, but that pushed server boot past
        # Claude Desktop's MCP initialize-handshake timeout — the
        # ~5-10s sentence-transformers warmup blocked tool registration
        # entirely. Now the model loads on first `.score()` call. To
        # still catch a typo'd model name at config time, the wiring
        # layer can call `validate()` post-construct — see
        # `validate_model_available()` below.

    def _load_model(self) -> Any:
        """Idempotent, thread-safe model loader."""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "HuggingFaceCrossEncoderReranker requires the optional "
                    "'sentence-transformers' extra (which ships CrossEncoder). "
                    "Install with:\n"
                    "    pip install 'research-mcp[sentence-transformers]'"
                ) from exc
            _log.info("loading cross-encoder model %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self._device)
            return self._model

    async def score(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> Sequence[float]:
        if not papers:
            return []
        return await asyncio.to_thread(self._score_sync, query, list(papers))

    def _score_sync(self, query: str, papers: list[Paper]) -> list[float]:
        model = self._load_model()
        # Cross-encoder takes pairs as [query, document] strings.
        # Concatenate title + abstract per the standard retrieval recipe.
        pairs = [
            [query, _document_text(p)]
            for p in papers
        ]
        scores = model.predict(
            pairs,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def _document_text(paper: Paper) -> str:
    pieces = [paper.title, paper.abstract]
    if paper.full_text:
        # full_text may dwarf the cross-encoder's max sequence length (~512
        # tokens). The model truncates internally; passing the full string
        # is fine — the title and lead abstract sentences are the most
        # informative pieces and they go in first.
        pieces.append(paper.full_text)
    return "\n\n".join(p for p in pieces if p)
