"""SearchService — fan out a query across all configured Sources, merge, return.

Sources run concurrently via asyncio.gather. Per the Source protocol contract,
implementations must not raise on transient errors — they return empty
sequences instead — so a single dead source never poisons the merged result.
We use return_exceptions=True as a belt-and-suspenders for misbehaving
implementations.

Merge strategy:
  1. Round-robin across sources so the top hit from each gets a fair shot
     at the top of the merged list.
  2. Group by id / arxiv id / DOI / S2 id / normalized title — same paper
     under different ids in different sources collapses to one row.
  3. **Field-by-field enrichment** when a group spans multiple sources:
     when arXiv has no DOI but Semantic Scholar does, the merged record
     keeps both. The previous "first paper wins" merge silently dropped
     the second source's data.
  4. Truncate to `query.max_results`. Without this, the caller's `N`
     request returned up to `N * len(sources)` rows.

Each merged record carries the set of source names that contributed to
it (`SearchResult.sources`), which the MCP layer surfaces as the `source`
field on `PaperSummary`. A user — or an LLM — can see at a glance whether
a record is arxiv-only, s2-only, or enriched across both.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.reranker import Reranker
from research_mcp.domain.source import Source
from research_mcp.errors import SourceUnavailable

# When a reranker is configured, fetch this many times the user-requested
# max_results from each Source before reranking. The standard recipe is
# bi-encoder→top-50, cross-encoder→top-K. With max_results=10 and 5x
# widening we ask each Source for 50; after dedup that's typically ~50-100
# unique candidates to rerank.
_RERANK_POOL_FACTOR = 5

# Hard cap on the per-source request size; arXiv and S2 both cap their own
# responses around 100. Going past doesn't help quality and risks 429s.
_MAX_RERANK_POOL = 50

_log = logging.getLogger(__name__)

# Words that should not influence title-based dedup. Kept tiny on purpose;
# we want to collapse "Attention Is All You Need" and "Attention is all you need"
# without merging two genuinely different papers that happen to share a stopword.
_TITLE_STOPWORDS = frozenset({"a", "an", "the"})

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Order of preference when picking the canonical id for a merged record.
# arXiv ids never change across versions (we strip the version suffix on
# parse), so they're the most stable. DOIs are next-best — durable but only
# present after publication. Semantic Scholar's corpusId is stable but not
# externally meaningful, so it's last.
_ID_PREFIX_RANK = ("arxiv", "doi", "s2")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One merged hit from `SearchService.search`.

    `sources` is the set of adapter names that contributed metadata —
    `("arxiv",)`, `("semantic_scholar",)`, or both when enrichment merged
    across sources.
    """

    paper: Paper
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Outcome of a `SearchService.search` call.

    `partial_failures` carries human-readable per-source failure reasons
    (e.g., `"arxiv: HTTP 429"`) so the LLM caller can distinguish
    "no papers match" (empty `results`, empty `partial_failures`) from
    "every source was rate-limited" (empty `results`, populated
    `partial_failures`). Callers should generally retry on the latter.
    """

    results: list[SearchResult]
    partial_failures: tuple[str, ...] = ()


class SearchService:
    def __init__(
        self,
        sources: Sequence[Source],
        *,
        reranker: Reranker | None = None,
    ) -> None:
        if not sources:
            raise ValueError("SearchService requires at least one Source")
        self._sources = tuple(sources)
        self._reranker = reranker

    @property
    def sources(self) -> tuple[Source, ...]:
        return self._sources

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    async def search(self, query: SearchQuery) -> SearchOutcome:
        # When a reranker is configured, widen the per-source request so the
        # cross-encoder has more candidates to choose from. The merged
        # candidate set is then reranked and truncated to max_results.
        if self._reranker is None:
            upstream_query = query
        else:
            widened = min(
                query.max_results * _RERANK_POOL_FACTOR, _MAX_RERANK_POOL
            )
            upstream_query = SearchQuery(
                text=query.text,
                max_results=max(widened, query.max_results),
                year_min=query.year_min,
                year_max=query.year_max,
                authors=query.authors,
            )
        outcomes = await asyncio.gather(
            *(s.search(upstream_query) for s in self._sources),
            return_exceptions=True,
        )
        per_source: list[list[Paper]] = []
        failures: list[str] = []
        for source, outcome in zip(self._sources, outcomes, strict=True):
            if isinstance(outcome, SourceUnavailable):
                failures.append(f"{outcome.source_name}: {outcome.short_reason()}")
                _log.warning("source %r unavailable: %s", source.name, outcome.short_reason())
                per_source.append([])
                continue
            if isinstance(outcome, BaseException):
                _log.warning("source %r raised %s; ignoring", source.name, outcome)
                failures.append(f"{source.name}: unexpected {type(outcome).__name__}")
                per_source.append([])
                continue
            per_source.append(list(outcome))

        merged: list[Paper] = []
        contributors: list[set[str]] = []
        # Maps every key (id / arxiv_id / doi / s2_id / title-key) to the
        # index of the merged record that owns it. Used both to detect
        # duplicates and to extend the lookup table after a merge swaps in
        # an enriched record with potentially new ids.
        key_to_index: dict[str, int] = {}

        # When a reranker is set, keep the full deduped pool (so the
        # cross-encoder gets the broadest candidate set possible) and
        # truncate at the end. Without a reranker, stop merging once we
        # have max_results — same as before.
        merge_cap = (
            upstream_query.max_results if self._reranker is not None
            else query.max_results
        )
        max_depth = max((len(papers) for papers in per_source), default=0)
        for depth in range(max_depth):
            if len(merged) >= merge_cap:
                break
            for source_index, papers in enumerate(per_source):
                if depth >= len(papers):
                    continue
                paper = papers[depth]
                source_name = self._sources[source_index].name
                keys = _merge_keys(paper)
                existing = next(
                    (key_to_index[k] for k in keys if k in key_to_index), None
                )
                if existing is not None:
                    merged[existing] = _merge_records(merged[existing], paper)
                    contributors[existing].add(source_name)
                    for k in _merge_keys(merged[existing]):
                        key_to_index[k] = existing
                    continue
                merged.append(paper)
                contributors.append({source_name})
                idx = len(merged) - 1
                for k in keys:
                    key_to_index[k] = idx
                if len(merged) >= merge_cap:
                    break

        # Rerank if configured and we have candidates. On failure, fall back
        # to the bi-encoder ordering and surface the failure as a partial.
        if self._reranker is not None and merged:
            try:
                scores = await self._reranker.score(query.text, merged)
            except Exception as exc:
                failures.append(
                    f"reranker:{self._reranker.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                _log.warning(
                    "reranker %r failed: %s; falling back to bi-encoder order",
                    self._reranker.name, exc,
                )
            else:
                order = sorted(
                    range(len(merged)), key=lambda i: scores[i], reverse=True
                )
                merged = [merged[i] for i in order]
                contributors = [contributors[i] for i in order]

        merged = merged[: query.max_results]
        contributors = contributors[: query.max_results]

        return SearchOutcome(
            results=[
                SearchResult(paper=p, sources=tuple(sorted(s)))
                for p, s in zip(merged, contributors, strict=True)
            ],
            partial_failures=tuple(failures),
        )


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


def _merge_records(a: Paper, b: Paper) -> Paper:
    """Field-by-field merge of two `Paper` records that represent the same paper.

    The first arg is treated as the existing record (older, has incumbency);
    the second is the newcomer. Either may carry richer data than the other,
    so we never assume one source is uniformly better — we go field by field.
    """
    return Paper(
        id=_pick_id(a.id, b.id),
        title=_pick_longer(a.title, b.title),
        abstract=_pick_longer(a.abstract, b.abstract),
        authors=_pick_authors(a.authors, b.authors),
        published=_pick_date(a.published, b.published),
        url=_pick_url(a, b),
        venue=_pick_first_non_none(a.venue, b.venue),
        doi=_pick_first_non_none(a.doi, b.doi),
        arxiv_id=_pick_first_non_none(a.arxiv_id, b.arxiv_id),
        semantic_scholar_id=_pick_first_non_none(a.semantic_scholar_id, b.semantic_scholar_id),
        pdf_url=_pick_first_non_none(a.pdf_url, b.pdf_url),
        full_text=_pick_first_non_none(a.full_text, b.full_text),
        # Pick the higher count when both sources report one — different
        # adapters' count snapshots aren't synchronized, so 'higher' is
        # the closest proxy for 'fresher' available without a timestamp.
        citation_count=_pick_higher_int(a.citation_count, b.citation_count),
        metadata=_merge_metadata(a.metadata, b.metadata),
    )


def _pick_id(a_id: str, b_id: str) -> str:
    a_prefix = a_id.split(":", 1)[0]
    b_prefix = b_id.split(":", 1)[0]
    a_rank = _ID_PREFIX_RANK.index(a_prefix) if a_prefix in _ID_PREFIX_RANK else len(_ID_PREFIX_RANK)
    b_rank = _ID_PREFIX_RANK.index(b_prefix) if b_prefix in _ID_PREFIX_RANK else len(_ID_PREFIX_RANK)
    return a_id if a_rank <= b_rank else b_id


def _pick_longer(a: str, b: str) -> str:
    """Prefer the longer non-empty string; fall back to whichever is non-empty."""
    if a and not b:
        return a
    if b and not a:
        return b
    return a if len(a) >= len(b) else b


def _pick_authors(a: tuple[Author, ...], b: tuple[Author, ...]) -> tuple[Author, ...]:
    """Prefer the longer author list. S2 enriches affiliations more reliably
    than arXiv, but neither is uniformly better — length is a serviceable
    proxy for completeness."""
    if not a:
        return b
    if not b:
        return a
    return a if len(a) >= len(b) else b


def _pick_date(a: date | None, b: date | None) -> date | None:
    """Prefer non-None; among two non-None dates, prefer the one with a
    non-default month/day (i.e., not 1/1, which is what we substitute for
    year-only metadata from S2)."""
    if a is None:
        return b
    if b is None:
        return a
    a_default = a.month == 1 and a.day == 1
    b_default = b.month == 1 and b.day == 1
    if a_default and not b_default:
        return b
    return a


def _pick_url(a: Paper, b: Paper) -> str | None:
    """Prefer the canonical-source URL: arxiv abs page if either has an arXiv
    id, else the first non-None URL."""
    arxiv_id = a.arxiv_id or b.arxiv_id
    if arxiv_id:
        for paper in (a, b):
            if paper.url and "arxiv.org" in paper.url:
                return paper.url
        return f"https://arxiv.org/abs/{arxiv_id}"
    return a.url or b.url


def _pick_first_non_none(a: str | None, b: str | None) -> str | None:
    return a if a is not None else b


def _pick_higher_int(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _merge_metadata(
    a: Mapping[str, str], b: Mapping[str, str]
) -> Mapping[str, str]:
    if not a and not b:
        return MappingProxyType({})
    merged = dict(a)
    merged.update(b)
    return MappingProxyType(merged)
