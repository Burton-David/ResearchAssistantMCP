"""LibraryService — ingest, recall, delete, count over a local Index.

Composes Index + Embedder + ingest Source. The ingest Source supplies the
paper's metadata when ingesting by id; the Embedder turns title + abstract
(plus full text if present) into a vector; the Index stores both.
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
        ingest_source: Source,
    ) -> None:
        if index.dimension != embedder.dimension:
            raise ValueError(
                f"index dim {index.dimension} != embedder dim {embedder.dimension}"
            )
        self._index = index
        self._embedder = embedder
        self._ingest_source = ingest_source

    @property
    def index(self) -> Index:
        return self._index

    async def ingest(self, paper_id: str) -> Paper:
        paper = await self._ingest_source.fetch(paper_id)
        if paper is None:
            raise PaperNotFoundError(paper_id)
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
