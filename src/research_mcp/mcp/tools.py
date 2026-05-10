"""Pydantic models for MCP tool inputs and outputs.

Schemas are intentionally strict — we ban unknown fields so a buggy LLM call
fails loud rather than silently dropping arguments. Output models are exported
for tests; the server returns plain dicts via `model_dump`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper
from research_mcp.domain.source import Source

CitationFormatLiteral = Literal["ama", "apa", "mla", "chicago", "bibtex"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace characters")
    return value


# Reusable validated query string: rejects "", "   ", "\n\n" up front so the
# user gets a real error instead of a silent empty result list. Cap at 500
# chars so an over-long query doesn't get rejected by upstream APIs (arXiv's
# URL length cap, S2's body validation) only to come back as a confusing
# transient-error with no signal that the input was the problem.
_MAX_QUERY_CHARS = 500
NonBlankStr = Annotated[str, Field(..., min_length=1, max_length=_MAX_QUERY_CHARS)]


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


class GetPaperInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Canonical paper id with source prefix (e.g. 'arxiv:1706.03762', "
            "'doi:10.1038/...', 's2:abc...'). Returns full Paper metadata "
            "without ingesting; use ingest_paper if you want it in the "
            "local library."
        ),
    )


class FindPaperInput(_Strict):
    title: NonBlankStr = Field(
        ..., description="Paper title (or close approximation)."
    )
    authors: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Optional author names; surnames are extracted automatically and "
            "used to break ties between same-titled papers by different authors."
        ),
    )

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        return _reject_blank(value)


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
    source: str = ""
    """Adapters that contributed metadata, joined with '+' if multiple
    (e.g. 'arxiv', 'semantic_scholar', 'arxiv+semantic_scholar')."""


class SearchPapersOutput(BaseModel):
    results: list[PaperSummary]
    partial_failures: list[str] = []
    """Per-source transient failures (e.g. ['arxiv: HTTP 429']). When this
    is non-empty AND results is empty, the caller should retry rather than
    treat it as 'no papers exist'."""


class LibrarySearchHit(BaseModel):
    """One ranked recall result.

    `score` is cosine similarity in [-1, 1]; for OpenAI text-embedding-3-*
    on a normalized FAISS index, in-domain hits typically score 0.4-0.8 and
    near-duplicates score >0.9.
    """

    paper: PaperSummary
    score: float


class LibrarySearchOutput(BaseModel):
    results: list[LibrarySearchHit]


class IngestPaperOutput(BaseModel):
    paper: PaperSummary
    library_count: int


class CitePaperOutput(BaseModel):
    citation: str
    format: CitationFormatLiteral


class LibraryStatusOutput(BaseModel):
    count: int
    embedder: str | None = None
    """Embedder selection string ('openai:...', 'sentence-transformers:...').
    None when no embedder is configured — in which case `count` is 0 and
    ingest/recall tools will refuse with a helpful message."""
    reranker: str | None = None
    """Reranker selection string ('cross-encoder:BAAI/bge-reranker-base').
    None when no reranker is configured. Reranking is opt-in because it
    adds 200-1000ms per search/recall."""
    note: str | None = None
    """Human-readable hint shown when the library isn't fully configured."""


class GetPaperOutput(BaseModel):
    paper: PaperSummary


class FindPaperHit(BaseModel):
    paper: PaperSummary
    confidence: float
    """Title-token Jaccard similarity plus a small author-match bonus when
    the caller passed authors. Range [0, 1.15]. Above ~0.7 indicates a
    confident match; below ~0.3 means the discovered paper is probably
    not what was asked for; >1.0 implies a perfect title match with
    author corroboration."""


class FindPaperOutput(BaseModel):
    results: list[FindPaperHit]
    note: str | None = None
    """Hint surfaced when the result is empty for a structural reason —
    e.g., the title was all stopwords and produced no Jaccard tokens."""
    partial_failures: list[str] = []
    """Per-source transient failures, same shape as SearchPapersOutput."""


class ChunkPaperInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Canonical paper id with source prefix. The paper is fetched "
            "from the originating source (no ingest required), then "
            "section-aware chunking is applied to title + abstract "
            "(+ full_text when present)."
        ),
    )


class TextChunkSummary(BaseModel):
    chunk_id: str
    paper_id: str
    section: str | None
    text: str
    char_count: int
    start_char: int
    end_char: int


class ChunkPaperOutput(BaseModel):
    chunks: list[TextChunkSummary]
    chunker: str
    """Active chunker name (e.g., 'section-aware', 'simple') so the LLM
    caller knows which chunking semantics produced these chunks."""


def paper_to_summary(paper: Paper, *, source: str = "") -> PaperSummary:
    """Project a `Paper` into the LLM-friendly summary view.

    `source` names the adapter(s) that produced the metadata. For records
    coming out of `SearchService`, pass the joined `SearchResult.sources`
    string. For ingest / cite / get_paper outputs, derive it from the
    canonical id prefix via `source_from_id`.
    """
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
        source=source,
    )


def source_from_id(paper_id: str, sources: Sequence[Source]) -> str:
    """Find which Source's `id_prefixes` claim this paper id, return its `name`.

    Used when a Paper isn't carrying provenance through SearchService —
    e.g., responses from `ingest_paper`, `cite_paper`, `get_paper` that
    came out of `LibraryService.fetch`. The first Source in `sources`
    whose `id_prefixes` contains the id's prefix wins.

    Falls back to the bare prefix string when no Source claims it. That
    keeps the response well-formed (LLM still sees a non-empty `source`)
    without lying about provenance.
    """
    prefix = paper_id.split(":", 1)[0]
    for source in sources:
        if prefix in source.id_prefixes:
            return source.name
    return prefix


def to_citation_format(value: str) -> CitationFormat:
    return CitationFormat(value)
