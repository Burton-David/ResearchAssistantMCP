"""Index protocol: a vector store for Papers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_mcp.domain.paper import Paper


@runtime_checkable
class Index(Protocol):
    """A vector index of Papers.

    Implementations must be safe to call concurrently. Persistence (or lack of it)
    is an implementation detail — `FaissIndex` writes to disk, `MemoryIndex` doesn't.
    """

    async def upsert(
        self,
        papers: Sequence[Paper],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update papers + their embeddings.

        `len(papers) == len(embeddings)` must hold. If a paper with the same `id`
        already exists, replace it. Embedding dimension must match the index's
        configured dimension.
        """
        ...

    async def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
    ) -> Sequence[tuple[Paper, float]]:
        """Return top-k (paper, similarity_score) tuples for the query embedding.

        Score semantics depend on the implementation but higher always = more
        similar. Returned in descending score order.
        """
        ...

    async def delete(self, paper_ids: Sequence[str]) -> None:
        """Remove papers by canonical id. Silently ignore ids not in the index."""
        ...

    async def count(self) -> int:
        """Number of papers currently in the index."""
        ...
