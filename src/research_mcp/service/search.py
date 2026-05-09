"""SearchService — fan out a query across all configured Sources, dedup, return.

Sources run concurrently via asyncio.gather. Per the Source protocol contract,
implementations must not raise on transient errors — they return empty
sequences instead — so a single dead source never poisons the merged result.
We use return_exceptions=True as a belt-and-suspenders for misbehaving
implementations.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.source import Source

_log = logging.getLogger(__name__)


class SearchService:
    def __init__(self, sources: Sequence[Source]) -> None:
        if not sources:
            raise ValueError("SearchService requires at least one Source")
        self._sources = tuple(sources)

    @property
    def sources(self) -> tuple[Source, ...]:
        return self._sources

    async def search(self, query: SearchQuery) -> list[Paper]:
        results = await asyncio.gather(
            *(s.search(query) for s in self._sources),
            return_exceptions=True,
        )
        merged: list[Paper] = []
        seen: set[str] = set()
        for source, outcome in zip(self._sources, results, strict=True):
            if isinstance(outcome, BaseException):
                _log.warning("source %r raised %s; ignoring", source.name, outcome)
                continue
            for paper in outcome:
                if any(key in seen for key in _merge_keys(paper)):
                    continue
                merged.append(paper)
                seen.update(_merge_keys(paper))
        return merged


def _merge_keys(paper: Paper) -> set[str]:
    """Identifiers that should collapse the same paper across different sources."""
    keys = {paper.id}
    if paper.arxiv_id:
        keys.add(f"arxiv:{paper.arxiv_id}")
    if paper.doi:
        keys.add(f"doi:{paper.doi.lower()}")
    if paper.semantic_scholar_id:
        keys.add(f"s2:{paper.semantic_scholar_id}")
    return keys
