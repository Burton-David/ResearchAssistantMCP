"""Source protocol: anything that can return Papers given a SearchQuery is a Source."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery


@runtime_checkable
class Source(Protocol):
    """A source of papers.

    Implementations must be safe to call concurrently — the orchestrator
    fans out a SearchQuery across all configured sources in parallel.
    """

    name: str
    """Human-readable adapter name. Used in error messages, library_status,
    search-result provenance ("source": "arxiv" / "semantic_scholar" /
    "arxiv+semantic_scholar"). Must be stable across versions — clients
    persist it in indexes and expect it not to change."""

    id_prefixes: tuple[str, ...]
    """Canonical-id prefixes (the part before ':') that this Source owns.

    arXiv emits 'arxiv:2401.12345', so ArxivSource.id_prefixes = ('arxiv',).
    Semantic Scholar accepts S2 corpus ids and DOIs, so its id_prefixes =
    ('s2', 'doi'). The wiring layer uses this to derive provenance and
    route ids without hard-coding adapter names — adding a new Source
    (PubMed, OpenAlex) is now a self-contained change.

    Prefixes must be lower-case and not contain ':'. Two Sources may
    declare overlapping prefixes (e.g., both PubMed and S2 might own
    'doi'); in that case the wiring layer's Source-list order decides
    which one resolves first."""

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
