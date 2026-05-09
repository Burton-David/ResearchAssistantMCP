"""Source protocol: anything that can return Papers given a SearchQuery is a Source."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery


@runtime_checkable
class Source(Protocol):
    """A source of papers.

    Implementations must be safe to call concurrently — the orchestrator
    fans out a SearchQuery across all configured sources in parallel.
    """

    name: str

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        """Return papers matching the query. Empty sequence on no results.

        Implementations should not raise on transient network errors — log
        and return an empty sequence so partial results from other sources
        still flow through.
        """
        ...

    async def fetch(self, paper_id: str) -> Paper | None:
        """Fetch a single paper by canonical id. None if not found.

        `paper_id` is expected to carry the source prefix (`arxiv:2401.12345`).
        Implementations should return None if the id doesn't belong to this source.
        """
        ...
