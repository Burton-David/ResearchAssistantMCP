"""LibraryService — ingest, recall, delete, count over a local Index.

Composes Index + Embedder + one or more ingest Sources. Each Source's
`fetch` is asked in turn until one resolves the id; this is how the same
service handles `arxiv:1706.03762`, `doi:10.1038/...`, and `s2:abc123`
without a separate prefix table at the wiring layer.

The id-resolution loop is also exposed as the free `fetch_from_sources`
function so callers that don't need an embedder (cite_paper, get_paper)
can resolve ids without depending on a fully constructed LibraryService.
"""

from __future__ import annotations

from collections.abc import Sequence

from research_mcp.domain.embedder import Embedder
from research_mcp.domain.index import Index
from research_mcp.domain.paper import Paper
from research_mcp.domain.source import Source
from research_mcp.errors import SourceUnavailable


async def fetch_from_sources(
    sources: Sequence[Source], paper_id: str
) -> Paper | None:
    """Walk Sources in order, return the first hit.

    Three outcomes:
      - A Source returned a Paper: that Paper.
      - Every Source returned None (id not owned by any of them): None.
      - At least one Source raised `SourceUnavailable` and no Source
        returned a Paper: re-raises the first `SourceUnavailable`.
    """
    first_unavailable: SourceUnavailable | None = None
    for source in sources:
        try:
            paper = await source.fetch(paper_id)
        except SourceUnavailable as exc:
            if first_unavailable is None:
                first_unavailable = exc
            continue
        if paper is not None:
            return paper
    if first_unavailable is not None:
        raise first_unavailable
    return None


class PaperNotFoundError(LookupError):
    """Raised when an ingest is attempted for an id no source can resolve.

    Distinct from `SourceUnavailable`: this means every Source we asked
    confirmed it does not own the id. Use this only when we have a
    definitive miss — if a Source was unavailable and we never got a
    confirmed answer, propagate `SourceUnavailable` instead.
    """


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

        Thin wrapper around the free `fetch_from_sources`; both call sites
        share the same routing logic.
        """
        return await fetch_from_sources(self._sources, paper_id)

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
