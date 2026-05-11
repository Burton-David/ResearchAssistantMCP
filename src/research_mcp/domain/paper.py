"""Core paper and author types — immutable, hashable, source-agnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Author:
    """An author of a paper.

    `affiliation` and `orcid` are optional because many sources don't reliably provide them.

    `s2_id` is Semantic Scholar's stable per-author identifier (digit-string)
    when the parsing source knows it. Used to fetch h-index via S2's
    `/author/{id}` endpoint without paying a name-disambiguation round-trip;
    other sources (arXiv, OpenAlex, PubMed) currently don't populate it.
    """

    name: str
    affiliation: str | None = None
    orcid: str | None = None
    s2_id: str | None = None


@dataclass(frozen=True, slots=True)
class Paper:
    """A paper with metadata, source-agnostic.

    `id` is the canonical identifier. The prefix tells you which adapter produced it:
    `arxiv:2401.12345`, `doi:10.1038/...`, or `s2:abc123`. Always include the prefix
    so dedup across sources works.

    Optional fields default to None / empty; sources fill in what they have.
    `full_text` is populated only after a deliberate ingest — search results don't
    include it.
    """

    id: str
    title: str
    abstract: str
    authors: tuple[Author, ...]
    published: date | None = None
    url: str | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    pdf_url: str | None = None
    full_text: str | None = None
    citation_count: int | None = None
    """How many other works have cited this paper, per the source that
    populated it. None means 'unknown' — distinct from 0 ('cited zero
    times'); the scorer treats those differently. S2 and OpenAlex
    populate this; arXiv doesn't and PubMed doesn't directly."""
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
