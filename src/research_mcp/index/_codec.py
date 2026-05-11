"""Paper <-> dict serialization for index sidecar storage.

Kept private to the index package — Paper is the public dataclass, and the
on-disk representation is an implementation detail of where the index lives.
Avoids pickle (which is fragile across Python versions and a security risk
if an index file is ever shared).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from types import MappingProxyType
from typing import Any

from research_mcp.domain.paper import Author, Paper


def paper_to_dict(paper: Paper) -> dict[str, Any]:
    return {
        "id": paper.id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": [asdict(a) for a in paper.authors],
        "published": paper.published.isoformat() if paper.published else None,
        "url": paper.url,
        "venue": paper.venue,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "semantic_scholar_id": paper.semantic_scholar_id,
        "pdf_url": paper.pdf_url,
        "full_text": paper.full_text,
        "citation_count": paper.citation_count,
        "metadata": dict(paper.metadata),
    }


def paper_from_dict(d: dict[str, Any]) -> Paper:
    authors = tuple(Author(**a) for a in d.get("authors", ()))
    published_value = d.get("published")
    published: date | None = (
        date.fromisoformat(published_value) if isinstance(published_value, str) else None
    )
    metadata = MappingProxyType(dict(d.get("metadata") or {}))
    return Paper(
        id=d["id"],
        title=d["title"],
        abstract=d["abstract"],
        authors=authors,
        published=published,
        url=d.get("url"),
        venue=d.get("venue"),
        doi=d.get("doi"),
        arxiv_id=d.get("arxiv_id"),
        semantic_scholar_id=d.get("semantic_scholar_id"),
        pdf_url=d.get("pdf_url"),
        full_text=d.get("full_text"),
        citation_count=d.get("citation_count"),
        metadata=metadata,
    )
