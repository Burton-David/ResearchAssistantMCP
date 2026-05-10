"""Pydantic models for MCP tool inputs and outputs.

Schemas are intentionally strict — we ban unknown fields so a buggy LLM call
fails loud rather than silently dropping arguments. Output models are exported
for tests; the server returns plain dicts via `model_dump`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper

CitationFormatLiteral = Literal["ama", "apa", "mla", "chicago", "bibtex"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace characters")
    return value


# Reusable validated query string: rejects "", "   ", "\n\n" up front so the
# user gets a real error instead of a silent empty result list.
NonBlankStr = Annotated[str, Field(..., min_length=1)]


class SearchPapersInput(_Strict):
    query: NonBlankStr = Field(..., description="Free-form search text.")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return.")
    year_min: int | None = Field(None, description="Earliest publication year (inclusive).")
    year_max: int | None = Field(None, description="Latest publication year (inclusive).")

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        return _reject_blank(value)

    @model_validator(mode="after")
    def _year_range_well_ordered(self) -> SearchPapersInput:
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            raise ValueError(
                f"year_min ({self.year_min}) must not exceed year_max ({self.year_max})"
            )
        return self


class IngestPaperInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description="Canonical paper id with source prefix, e.g. 'arxiv:1706.03762'.",
    )


class LibrarySearchInput(_Strict):
    query: NonBlankStr = Field(..., description="Free-form recall text.")
    k: int = Field(10, ge=1, le=100, description="How many neighbors to return.")

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        return _reject_blank(value)


class CitePaperInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Canonical paper id with source prefix (e.g. 'arxiv:1706.03762', "
            "'doi:10.1038/...', 's2:abc...'). Fetches metadata from the "
            "originating source on demand — the paper does NOT need to be "
            "in the local library."
        ),
    )
    format: CitationFormatLiteral = Field(
        "ama", description="Citation format. Defaults to AMA."
    )


class LibraryStatusInput(_Strict):
    """No arguments. Reports library size without requiring an ingest call."""


# Past this many authors, large-collaboration HEP/ML papers (BESIII, ATLAS,
# CMS, …) blow tens of KB of context for one search result. We truncate
# explicitly with a `+ N more` count so the LLM still knows the full
# authorship is larger than what it sees.
_MAX_AUTHORS_IN_SUMMARY = 20


class PaperSummary(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str]
    authors_truncated: bool = False
    authors_total: int = 0
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


class LibraryStatusOutput(BaseModel):
    count: int


def paper_to_summary(paper: Paper) -> PaperSummary:
    all_authors = [a.name for a in paper.authors]
    truncated = len(all_authors) > _MAX_AUTHORS_IN_SUMMARY
    return PaperSummary(
        id=paper.id,
        title=paper.title,
        abstract=paper.abstract,
        authors=all_authors[:_MAX_AUTHORS_IN_SUMMARY],
        authors_truncated=truncated,
        authors_total=len(all_authors),
        year=paper.published.year if paper.published else None,
        venue=paper.venue,
        url=paper.url,
        pdf_url=paper.pdf_url,
        doi=paper.doi,
    )


def to_citation_format(value: str) -> CitationFormat:
    return CitationFormat(value)
