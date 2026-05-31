"""Field-by-field merge of `Paper` records from different sources.

Cross-source enrichment needs to fold two records that represent the same
paper into one, taking the richer value per field. Both `SearchService`
(in-search dedup) and `LibraryService` (ingest-time enrichment) need this,
so it lives in its own module rather than as a private symbol one service
reaches into the other for.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from types import MappingProxyType

from research_mcp.domain.paper import Author, Paper

# Order of preference when picking the canonical id for a merged record.
# arXiv ids never change across versions (we strip the version suffix on
# parse), so they're the most stable. DOIs are next-best — durable but only
# present after publication. Semantic Scholar's corpusId is stable but not
# externally meaningful, so it's last.
_ID_PREFIX_RANK = ("arxiv", "doi", "s2")


def merge_records(a: Paper, b: Paper) -> Paper:
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


def _merge_metadata(a: Mapping[str, str], b: Mapping[str, str]) -> Mapping[str, str]:
    if not a and not b:
        return MappingProxyType({})
    merged = dict(a)
    merged.update(b)
    return MappingProxyType(merged)
