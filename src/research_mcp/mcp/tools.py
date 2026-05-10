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
    source_contributions: dict[str, int] = {}
    """Post-dedup paper count per configured source — e.g.,
    {"arxiv": 3, "semantic_scholar": 4, "pubmed": 0, "openalex": 2}.
    A zero entry means the source returned no results that survived
    dedup (vs missing from the dict, which means the source isn't
    configured). Use this to diagnose silent recall misses without
    scraping per-result attribution."""


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
    sources: list[str] = []
    """Names of configured search Sources, in the order they are tried
    for cross-source merge. Lets a caller verify whether PubMed /
    OpenAlex are wired without scraping result attribution."""
    claim_extractor: str | None = None
    """Active claim extractor name ('spacy', 'llm:openai:gpt-4o-mini').
    None means extract_claims and assist_draft will refuse."""
    paper_analyzer: str | None = None
    """Active analyzer name ('openai:gpt-4o-mini', 'anthropic:...').
    None means analyze_paper will refuse."""
    citation_scorer: str | None = None
    """Active citation scorer name ('heuristic'). Citation tools refuse
    when None."""
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


# ---- extract_claims ----

# Cap input length so a 50KB paste doesn't lock the spaCy pipeline. 20K
# chars is roughly an 8-page draft which is the upper end of "paragraph
# the user wants citations for" — anything bigger should be chunked
# upstream first.
_MAX_DRAFT_CHARS = 20_000

ClaimTypeLiteral = Literal[
    "statistical",
    "methodological",
    "comparative",
    "theoretical",
    "causal",
    "factual",
    "evaluative",
]


class ExtractClaimsInput(_Strict):
    text: NonBlankStr = Field(
        ...,
        max_length=_MAX_DRAFT_CHARS,
        description=(
            "Draft text (paragraph or short section) to scan for claims "
            "that need citations. Returns one claim per detected pattern, "
            "ordered by position in the text."
        ),
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _reject_blank(value)


class ClaimSummary(BaseModel):
    text: str
    type: ClaimTypeLiteral
    confidence: float
    context: str
    suggested_search_terms: list[str]
    start_char: int
    end_char: int


class ExtractClaimsOutput(BaseModel):
    claims: list[ClaimSummary]
    extractor: str
    """Active extractor name ('spacy' / 'fake' / future LLM extractors)
    so the LLM caller knows the precision tier of these claims."""


# ---- find_citations / score_citation / explain_citation ----


class _ClaimDescriptor(_Strict):
    """Inline claim description for citation tools that take a claim
    directly. Mirrors the `Claim` domain object's surface, minus the
    char offsets which only matter when the claim was extracted from
    text the caller is showing back to the user."""

    text: NonBlankStr = Field(..., description="The verbatim claim text.")
    type: ClaimTypeLiteral = Field(
        "factual",
        description=(
            "Claim type: drives downstream search-term emphasis "
            "and explanation phrasing."
        ),
    )
    context: str = Field(
        "",
        max_length=_MAX_DRAFT_CHARS,
        description="Surrounding sentence/paragraph for disambiguation.",
    )
    suggested_search_terms: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Keywords the citation finder uses as the upstream query.",
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _reject_blank(value)


class FindCitationsInput(_Strict):
    claim: _ClaimDescriptor = Field(
        ...,
        description=(
            "Claim to find citations for. Pass the output of "
            "extract_claims's claims[i] verbatim, or hand-craft a claim "
            "object when working from a single sentence."
        ),
    )
    k: int = Field(
        5, ge=1, le=20,
        description="How many candidate citations to return.",
    )


class ScoreCitationInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description="Canonical paper id (e.g. 'arxiv:1706.03762').",
    )


class ExplainCitationInput(_Strict):
    paper_id: str = Field(
        ...,
        min_length=1,
        description="Canonical paper id (e.g. 'arxiv:1706.03762').",
    )
    claim: _ClaimDescriptor = Field(
        ..., description="The claim this paper would be cited for."
    )


class CitationQualityScoreSummary(BaseModel):
    total: float
    """0-100 headline score; bands are <45 weak, 45-65 moderate, >=65 strong."""

    venue: float
    impact: float
    author: float
    recency: float
    factors: dict[str, str]
    warnings: list[str]


class CitationCandidateSummary(BaseModel):
    paper: PaperSummary
    score: CitationQualityScoreSummary


class FindCitationsOutput(BaseModel):
    candidates: list[CitationCandidateSummary]
    scorer: str
    """Active scorer name ('heuristic' / 'fake' / future LLM scorer)."""


class ScoreCitationOutput(BaseModel):
    paper: PaperSummary
    score: CitationQualityScoreSummary


class ExplainCitationOutput(BaseModel):
    explanation: str
    score: CitationQualityScoreSummary
    paper: PaperSummary


# ---- analyze_paper ----

AnalysisKindLiteral = Literal[
    "summary",
    "contributions",
    "methodology",
    "limitations",
    "future_work",
    "datasets",
    "metrics",
    "baselines",
]


class AnalyzePaperInput(_Strict):
    paper_id: str = Field(
        ..., min_length=1,
        description="Canonical paper id (e.g. 'arxiv:1706.03762').",
    )
    kinds: list[AnalysisKindLiteral] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Which analysis kinds to extract. Empty = all kinds. Listing "
            "fewer kinds reduces output tokens but doesn't shrink the "
            "input prompt."
        ),
    )


class PaperAnalysisSummary(BaseModel):
    paper_id: str
    summary: str | None
    key_contributions: list[str]
    methodology: str | None
    technical_approach: str | None
    limitations: list[str]
    future_directions: list[str]
    datasets_used: list[str]
    metrics_reported: dict[str, float]
    baselines_compared: list[str]
    confidence: float
    model: str
    """Model identifier ('openai:gpt-4o-mini' / 'anthropic:claude-...' /
    'fake:stub'). The caller can decide whether to trust the result."""


class AnalyzePaperOutput(BaseModel):
    paper: PaperSummary
    analysis: PaperAnalysisSummary


# ---- bulk_ingest ----


class BulkIngestInput(_Strict):
    query: NonBlankStr = Field(..., description="Search query.")
    max_papers: int = Field(
        20, ge=1, le=100,
        description="Cap on how many papers to ingest from the search results.",
    )
    year_min: int | None = Field(None, description="Earliest publication year.")
    year_max: int | None = Field(None, description="Latest publication year.")

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        return _reject_blank(value)


class BulkIngestOutput(BaseModel):
    ingested_count: int
    library_count: int
    papers: list[PaperSummary]
    partial_failures: list[str] = []


# ---- assist_draft ----


class AssistDraftInput(_Strict):
    text: NonBlankStr = Field(
        ..., max_length=_MAX_DRAFT_CHARS,
        description=(
            "Draft text (paragraph or short section). The pipeline "
            "extracts claims, finds candidate citations across all "
            "configured sources, and returns ranked, explained "
            "recommendations per claim."
        ),
    )
    k_per_claim: int = Field(
        3, ge=1, le=10,
        description="How many candidate citations to return per claim.",
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _reject_blank(value)


class CitationRecommendationCandidateSummary(BaseModel):
    paper: PaperSummary
    score_total: float
    score_warnings: list[str]
    explanation: str


class CitationRecommendationSummary(BaseModel):
    claim: ClaimSummary
    candidates: list[CitationRecommendationCandidateSummary]


class AssistDraftOutput(BaseModel):
    recommendations: list[CitationRecommendationSummary]
    extractor: str
    scorer: str


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
