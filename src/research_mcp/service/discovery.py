"""Title-based paper discovery.

The intended workflow: a user has a paper title (and maybe authors) but no
canonical id, and wants the system to surface candidate hits. Bare
`search_papers` works for this in theory, but in practice arXiv's relevance
ranking buries the canonical paper under recent derivative work that
shares the title pattern (the integration test for `arxiv:1706.03762`
already documents this).

`DiscoveryService` composes the existing `SearchService` and re-ranks the
top hits by title-token Jaccard similarity, with an author-surname bonus
that breaks ties between same-titled papers by different authors. Returns
at most three candidates with `confidence` ∈ [0, 1] sorted descending.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.service.search import SearchService

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TITLE_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "in", "on", "for", "with", "and", "or", "to"}
)

# How many candidates we ask SearchService for before re-ranking. More =
# better recall, more API calls. 12 is enough to cover the usual "right
# paper at rank 6" case without being expensive.
_PROBE_DEPTH = 12

# How many candidates we hand back. Three is enough for the LLM to disambiguate.
_RESULT_LIMIT = 3

# Author-surname-match bonus added to the Jaccard score when at least one
# requested author matches the candidate's author list.
_AUTHOR_MATCH_BONUS = 0.15


@dataclass(frozen=True, slots=True)
class DiscoveryHit:
    paper: Paper
    sources: tuple[str, ...]
    confidence: float


class DiscoveryService:
    def __init__(self, search: SearchService) -> None:
        self._search = search

    async def find_paper(
        self,
        title: str,
        authors: tuple[str, ...] = (),
    ) -> list[DiscoveryHit]:
        title = title.strip()
        if not title:
            return []

        # Build the upstream query as bare title tokens plus author surnames.
        # arXiv's full-text search splits on whitespace and ranks by overall
        # match; quoting forces a phrase that often misses recent papers
        # whose titles include extra subtitle text. We rely on the Jaccard
        # re-rank below to elevate the canonical hit.
        query_parts = [title]
        for author in authors:
            surname = _surname(author)
            if surname:
                query_parts.append(surname)
        query_text = " ".join(query_parts)

        results = await self._search.search(
            SearchQuery(text=query_text, max_results=_PROBE_DEPTH)
        )

        target_tokens = _title_tokens(title)
        target_surnames = {_surname(a) for a in authors if a}
        target_surnames.discard("")

        scored: list[DiscoveryHit] = []
        for result in results:
            hit_tokens = _title_tokens(result.paper.title)
            jaccard = _jaccard(target_tokens, hit_tokens)
            bonus = 0.0
            if target_surnames:
                hit_surnames = {_surname(a.name) for a in result.paper.authors}
                hit_surnames.discard("")
                if target_surnames & hit_surnames:
                    bonus = _AUTHOR_MATCH_BONUS
            # Don't clamp at 1.0: a perfect title match deserves to outrank
            # a perfect title match with no author corroboration when the
            # caller passed authors. Scores in [0, 1.15].
            score = jaccard + bonus
            scored.append(
                DiscoveryHit(
                    paper=result.paper,
                    sources=result.sources,
                    confidence=score,
                )
            )

        scored.sort(key=lambda h: h.confidence, reverse=True)
        return [h for h in scored[:_RESULT_LIMIT] if h.confidence > 0.0]


def _title_tokens(title: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return {
        t
        for t in _NON_ALNUM_RE.split(folded.lower())
        if t and t not in _TITLE_STOPWORDS
    }


def has_significant_tokens(title: str) -> bool:
    """Public predicate: does this title produce at least one non-stopword token?

    Used by the MCP layer to surface a "your title was all stopwords" note
    when find_paper returns empty for that structural reason rather than
    upstream relevance failure.
    """
    return bool(_title_tokens(title))


def _surname(author_name: str | Author) -> str:
    raw = author_name.name if isinstance(author_name, Author) else author_name
    raw = raw.strip()
    if not raw:
        return ""
    if "," in raw:
        return _surname(raw.partition(",")[0])
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    parts = [t for t in _NON_ALNUM_RE.split(folded.lower()) if t]
    return parts[-1] if parts else ""


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    s_a, s_b = set(a), set(b)
    if not s_a and not s_b:
        return 0.0
    return len(s_a & s_b) / len(s_a | s_b)
