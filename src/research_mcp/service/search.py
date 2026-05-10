"""SearchService — fan out a query across all configured Sources, dedup, return.

Sources run concurrently via asyncio.gather. Per the Source protocol contract,
implementations must not raise on transient errors — they return empty
sequences instead — so a single dead source never poisons the merged result.
We use return_exceptions=True as a belt-and-suspenders for misbehaving
implementations.

Merge strategy: round-robin across sources so the top hit from each gets a
fair shot at the top of the merged list, dedup by canonical id / arxiv id /
DOI / S2 id / normalized title, then truncate to `query.max_results`. Without
the truncation step the caller's `max_results=N` request returns up to
`N * len(sources)` rows in practice — a real bug we surface in tests.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Sequence

from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.source import Source

_log = logging.getLogger(__name__)

# Words that should not influence title-based dedup. Kept tiny on purpose;
# we want to collapse "Attention Is All You Need" and "Attention is all you need"
# without merging two genuinely different papers that happen to share a stopword.
_TITLE_STOPWORDS = frozenset({"a", "an", "the"})

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


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
        per_source: list[list[Paper]] = []
        for source, outcome in zip(self._sources, results, strict=True):
            if isinstance(outcome, BaseException):
                _log.warning("source %r raised %s; ignoring", source.name, outcome)
                per_source.append([])
                continue
            per_source.append(list(outcome))

        merged: list[Paper] = []
        seen: set[str] = set()
        # Round-robin: take the i-th paper from each source in turn. This
        # interleaves results so the top hit from a single source can't
        # monopolize the merged output.
        max_depth = max((len(papers) for papers in per_source), default=0)
        for depth in range(max_depth):
            if len(merged) >= query.max_results:
                break
            for papers in per_source:
                if depth >= len(papers):
                    continue
                paper = papers[depth]
                keys = _merge_keys(paper)
                if any(key in seen for key in keys):
                    continue
                merged.append(paper)
                seen.update(keys)
                if len(merged) >= query.max_results:
                    break
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
    title_key = _title_key(paper)
    if title_key:
        keys.add(title_key)
    return keys


def _title_key(paper: Paper) -> str:
    """Normalized title + first-author-surname key for cross-source dedup.

    arXiv and Semantic Scholar return the same paper with slightly different
    casing and punctuation in the title — and arXiv often sets `arxiv_id` while
    S2 sometimes does not, so id-only dedup misses these. We pair the
    normalized title with the first author's normalized surname so two
    same-titled-but-different papers don't collide.
    """
    title = (paper.title or "").strip()
    if not title:
        return ""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    tokens = [t for t in _NON_ALNUM_RE.split(folded.lower()) if t and t not in _TITLE_STOPWORDS]
    if not tokens:
        return ""
    title_part = "-".join(tokens)
    surname = ""
    if paper.authors:
        first = unicodedata.normalize("NFKD", paper.authors[0].name).encode("ascii", "ignore").decode()
        surname_tokens = [t for t in _NON_ALNUM_RE.split(first.lower()) if t]
        surname = surname_tokens[-1] if surname_tokens else ""
    return f"title:{title_part}|{surname}"
