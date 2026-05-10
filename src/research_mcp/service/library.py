"""LibraryService — ingest, recall, delete, count over a local Index.

Composes Index + Embedder + one or more ingest Sources. Each Source's
`fetch` is asked in turn until one resolves the id; this is how the same
service handles `arxiv:1706.03762`, `doi:10.1038/...`, and `s2:abc123`
without a separate prefix table at the wiring layer.
"""

from __future__ import annotations

from collections.abc import Sequence

from research_mcp.domain.embedder import Embedder
from research_mcp.domain.index import Index
from research_mcp.domain.paper import Paper
from research_mcp.domain.source import Source


class PaperNotFoundError(LookupError):
    """Raised when an ingest is attempted for an id no source can resolve."""


class LibraryService:
    def __init__(
        self,
        *,
        index: Index,
        embedder: Embedder,
        ingest_sources: Sequence[Source],
    ) -> None:
        if not ingest_sources:
            raise ValueError("LibraryService requires at least one ingest Source")
        self._index = index
        self._embedder = embedder
        self._sources = tuple(ingest_sources)

    @property
    def index(self) -> Index:
        return self._index

    @property
    def ingest_sources(self) -> tuple[Source, ...]:
        return self._sources

    async def fetch(self, paper_id: str) -> Paper | None:
        """Resolve a paper id against the configured Sources in order.

        Returns the first Source's hit. Returns None if no Source recognized
        the id — per the Source protocol contract, an implementation must
        return None (not raise) for ids it does not own.
        """
        for source in self._sources:
            paper = await source.fetch(paper_id)
            if paper is not None:
                return paper
        return None

    async def ingest(self, paper_id: str) -> Paper:
        paper = await self.fetch(paper_id)
        if paper is None:
            names = ", ".join(s.name for s in self._sources)
            raise PaperNotFoundError(
                f"no configured source could resolve paper id {paper_id!r} "
                f"(tried: {names})"
            )
        return await self.ingest_paper(paper)

    async def ingest_paper(self, paper: Paper) -> Paper:
        text = _embedding_text(paper)
        [vector] = await self._embedder.embed([text])
        await self._index.upsert([paper], [vector])
        return paper

    async def recall(self, query: str, k: int = 10) -> Sequence[tuple[Paper, float]]:
        [vector] = await self._embedder.embed([query])
        return await self._index.search(vector, k=k)

    async def delete(self, paper_id: str) -> None:
        await self._index.delete([paper_id])

    async def count(self) -> int:
        return await self._index.count()


def _embedding_text(paper: Paper) -> str:
    pieces = [paper.title, paper.abstract]
    if paper.full_text:
        pieces.append(paper.full_text)
    return "\n\n".join(p for p in pieces if p)
