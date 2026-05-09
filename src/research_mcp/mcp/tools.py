"""Pydantic models for MCP tool inputs and outputs.

Schemas are intentionally strict — we ban unknown fields so a buggy LLM call
fails loud rather than silently dropping arguments. Output models are exported
for tests; the server returns plain dicts via `model_dump`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper

CitationFormatLiteral = Literal["ama", "apa", "mla", "chicago", "bibtex"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchPapersInput(_Strict):
    query: str = Field(..., min_length=1, description="Free-form search text.")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return.")
    year_min: int | None = Field(None, description="Earliest publication year (inclusive).")
    year_max: int | None = Field(None, description="Latest publication year (inclusive).")


class IngestPaperInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description="Canonical paper id with source prefix, e.g. 'arxiv:1706.03762'.",
    )


class LibrarySearchInput(_Strict):
    query: str = Field(..., min_length=1, description="Free-form recall text.")
    k: int = Field(10, ge=1, le=100, description="How many neighbors to return.")


class CitePaperInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description="Canonical paper id, must already be in the local library.",
    )
    format: CitationFormatLiteral = Field(
        "ama", description="Citation format. Defaults to AMA."
    )


class PaperSummary(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    venue: str | None
    url: str | None
    pdf_url: str | None
    doi: str | None


class SearchPapersOutput(BaseModel):
    results: list[PaperSummary]


class LibrarySearchOutput(BaseModel):
    results: list[tuple[PaperSummary, float]]


class IngestPaperOutput(BaseModel):
    paper: PaperSummary
    library_count: int


class CitePaperOutput(BaseModel):
    citation: str
    format: CitationFormatLiteral


def paper_to_summary(paper: Paper) -> PaperSummary:
    return PaperSummary(
        id=paper.id,
        title=paper.title,
        abstract=paper.abstract,
        authors=[a.name for a in paper.authors],
        year=paper.published.year if paper.published else None,
        venue=paper.venue,
        url=paper.url,
        pdf_url=paper.pdf_url,
        doi=paper.doi,
    )


def to_citation_format(value: str) -> CitationFormat:
    return CitationFormat(value)
