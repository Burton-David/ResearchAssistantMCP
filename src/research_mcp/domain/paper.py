"""Core paper and author types — immutable, hashable, source-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Author:
    """An author of a paper.

    `affiliation` and `orcid` are optional because many sources don't reliably provide them.
    """

    name: str
    affiliation: str | None = None
    orcid: str | None = None


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
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
