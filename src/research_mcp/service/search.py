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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.reranker import Reranker
from research_mcp.domain.source import Source
from research_mcp.errors import SourceUnavailable, redact_secrets
from research_mcp.service._merge import merge_records
from research_mcp.service._tokens import tokenize

# When a reranker is configured, fetch this many times the user-requested
# max_results from each Source before reranking. The standard recipe is
# bi-encoder→top-50, cross-encoder→top-K. With max_results=10 and 5x
# widening we ask each Source for 50; after dedup that's typically ~50-100
# unique candidates to rerank.
_RERANK_POOL_FACTOR = 5

# Hard cap on the per-source request size; arXiv and S2 both cap their own
# responses around 100. Going past doesn't help quality and risks 429s.
_MAX_RERANK_POOL = 50

# Per-source wall-clock budget inside a parallel fan-out. With backoff
# retries, a 429ing source could spend up to 4x30s + 7s of backoff =
# ~127s before surfacing failure. asyncio.gather waits for ALL tasks,
# so one slow source pinned the whole search at 127s — which broke
# assist_draft inside its 180s tool budget. Capping each source at 25s
# means a slow upstream contributes a partial_failure and the merge
# proceeds with what came back from the responsive sources.
_PER_SOURCE_TIMEOUT_SECONDS = 25.0

_log = logging.getLogger(__name__)

# Words that should not influence title-based dedup. Kept tiny on purpose;
# we want to collapse "Attention Is All You Need" and "Attention is all you need"
# without merging two genuinely different papers that happen to share a stopword.
_TITLE_STOPWORDS = frozenset({"a", "an", "the"})


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

    `source_contributions` maps each configured source name to the
    number of UNIQUE papers it contributed to the merged result set
    (post-dedup). A `{"pubmed": 0}` entry on a biomedical query means
    PubMed returned nothing — distinct from PubMed erroring (which
    appears in `partial_failures`). Surfaces "silent recall miss" vs
    "transient failure" without forcing the caller to scrape result
    attribution.
    """

    results: list[SearchResult]
    partial_failures: tuple[str, ...] = ()
    source_contributions: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )


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
        # Wrap each source.search in its own wait_for so one slow source
        # (S2 rate-limited, arxiv 5xx, etc.) doesn't stall the merge.
        # asyncio.gather still waits for all to complete, but each task
        # is bounded — a hung source returns TimeoutError which gets
        # captured as a partial_failure below.
        async def _bounded(source: Source) -> Sequence[Paper]:
            try:
                return await asyncio.wait_for(
                    source.search(upstream_query),
                    timeout=_PER_SOURCE_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                raise SourceUnavailable(
                    source.name,
                    f"per-source budget exceeded ({_PER_SOURCE_TIMEOUT_SECONDS:.0f}s)",
                ) from exc

        outcomes = await asyncio.gather(
            *(_bounded(s) for s in self._sources),
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
                _log.warning(
                    "source %r raised %s; ignoring",
                    source.name, redact_secrets(repr(outcome)),
                )
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
                    merged[existing] = merge_records(merged[existing], paper)
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

        # Count post-dedup contributions per source so callers can
        # diagnose "PubMed returned 0 results for a textbook PubMed
        # query" without scraping result attribution. Every configured
        # source appears in the dict, even those that contributed
        # zero — explicit absence vs implicit absence.
        contribution_counts: dict[str, int] = {s.name: 0 for s in self._sources}
        for contributor_set in contributors:
            for name in contributor_set:
                contribution_counts[name] = contribution_counts.get(name, 0) + 1

        return SearchOutcome(
            results=[
                SearchResult(paper=p, sources=tuple(sorted(s)))
                for p, s in zip(merged, contributors, strict=True)
            ],
            partial_failures=tuple(failures),
            source_contributions=MappingProxyType(contribution_counts),
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
    tokens = tokenize(title, stopwords=_TITLE_STOPWORDS)
    if not tokens:
        return ""
    title_part = "-".join(tokens)
    surname = ""
    if paper.authors:
        surname_tokens = tokenize(paper.authors[0].name)
        surname = surname_tokens[-1] if surname_tokens else ""
    return f"title:{title_part}|{surname}"
