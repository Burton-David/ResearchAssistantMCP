"""MCP stdio server.

Wires sources, embedder, index, and the citation registry into the six
tool handlers. Default wiring uses real arXiv + Semantic Scholar; the
embedder is selected by `RESEARCH_MCP_EMBEDDER` (or auto-falls-back to
OpenAI if `OPENAI_API_KEY` is set). When no embedder is configured, the
server still serves search / cite / get_paper / library_status — only
the embedder-using tools (ingest_paper, library_search) refuse, with
a clear error message naming the env vars to set.

`run_in_memory` exists for the e2e harness; selected by
`RESEARCH_MCP_TEST_MODE=1` so the test can boot a real subprocess
without needing API keys or a writable index path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final, Literal

import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from research_mcp import __version__
from research_mcp.citation import RENDERERS
from research_mcp.citation_scorer import HeuristicCitationScorer
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.citation_scorer import CitationScorer
from research_mcp.domain.claim import ClaimExtractor
from research_mcp.domain.embedder import Embedder
from research_mcp.domain.paper import Paper
from research_mcp.domain.paper_analyzer import AnalysisKind, PaperAnalyzer
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.reranker import Reranker
from research_mcp.domain.source import Source
from research_mcp.embedder import (
    FakeEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
)
from research_mcp.errors import SourceUnavailable
from research_mcp.index import FaissIndex, MemoryIndex
from research_mcp.mcp.tools import (
    AnalyzePaperInput,
    AnalyzePaperOutput,
    AssistDraftInput,
    AssistDraftOutput,
    CitationCandidateSummary,
    CitationQualityScoreSummary,
    CitationRecommendationCandidateSummary,
    CitationRecommendationSummary,
    CitePaperInput,
    CitePaperOutput,
    ClaimSummary,
    ExplainCitationInput,
    ExplainCitationOutput,
    ExtractClaimsInput,
    ExtractClaimsOutput,
    FindCitationsInput,
    FindCitationsOutput,
    FindPaperHit,
    FindPaperInput,
    FindPaperOutput,
    FindReferencedByInput,
    FindReferencedByOutput,
    FindRelatedInput,
    FindRelatedOutput,
    GetPaperInput,
    GetPaperOutput,
    IngestPaperInput,
    IngestPaperOutput,
    LibrarySearchHit,
    LibrarySearchInput,
    LibrarySearchOutput,
    LibraryStatusInput,
    LibraryStatusOutput,
    PaperAnalysisSummary,
    SearchPapersInput,
    SearchPapersOutput,
    paper_to_summary,
    source_from_id,
)
from research_mcp.reranker import HuggingFaceCrossEncoderReranker
from research_mcp.service import DiscoveryService, LibraryService, SearchService
from research_mcp.service.analysis import AnalysisService
from research_mcp.service.citation import CitationService
from research_mcp.service.draft import DraftService
from research_mcp.service.library import fetch_with_enrichment
from research_mcp.sources import (
    ArxivSource,
    OpenAlexSource,
    PubMedSource,
    SemanticScholarSource,
)

_log = logging.getLogger(__name__)

# Per-tool timeout ceilings. Claude Desktop hard-kills tool calls at
# ~4 minutes (240s); we surface our own TIMEOUT error well before that
# so the user sees a clean failure rather than an opaque Desktop drop.
# Budgets are tuned to per-tool worst-case latency:
#   - simple metadata fetches: 60s (one source x ~30s + retries)
#   - multi-source merges: 90s (4 sources in parallel + reranker)
#   - LLM-backed tools: 150s (single LLM call + downstream search)
#   - corpus-write tools: 180s (search + N ingests; legitimately slow)
_DEFAULT_TOOL_TIMEOUT: Final = 90.0
_TOOL_TIMEOUTS: Final[dict[str, float]] = {
    "search_papers": 90.0,
    "find_paper": 90.0,
    "cite_paper": 60.0,
    "get_paper": 60.0,
    "library_status": 10.0,
    "library_search": 60.0,
    # ingest_paper now covers both single-id (~5s) and query-mode (search
    # + N parallel embed-and-upsert, up to ~60s). 180s covers the query
    # form's worst case; single-id calls return well inside the budget.
    "ingest_paper": 180.0,
    "extract_claims": 90.0,
    "find_citations": 150.0,
    "explain_citation": 60.0,
    "analyze_paper": 90.0,
    "assist_draft": 180.0,
    # Citation-graph tools: 1 parent fetch + N parallel-ish fan-out fetches
    # against OpenAlex's polite pool (0.1s min interval per request). Cold
    # cache for 10 refs is ~1-3s typical, comfortably inside this budget;
    # the headroom absorbs one or two slow OpenAlex responses without a
    # spurious tool-timeout error.
    "find_referenced_by": 60.0,
    "find_related": 60.0,
}

_CITATION_UNAVAILABLE_HINT = (
    "citation tools unavailable: no citation service configured. The "
    "server constructs one automatically when a CitationScorer is "
    "available; check server logs for boot warnings."
)
_NO_EMBEDDER_HINT = (
    "no embedder is configured. Set RESEARCH_MCP_EMBEDDER to "
    "'openai:text-embedding-3-small' (requires OPENAI_API_KEY) or "
    "'sentence-transformers:BAAI/bge-base-en-v1.5' (requires "
    "`pip install research-mcp[sentence-transformers]`)."
)


def _resolve_faiss_index_type() -> Literal["flat", "hnsw"]:
    """Parse RESEARCH_MCP_FAISS_INDEX_TYPE; default 'flat'.

    `flat` preserves the pre-#7 behavior and is the right call for libraries
    below ~100K papers. `hnsw` swaps in IndexHNSWFlat for sublinear search
    with a small recall tradeoff — appropriate once the library is large
    enough that flat search latency dominates.

    Sidecar wins on conflict (see FaissIndex docstring): asking for `hnsw`
    against an existing flat library logs a warning and keeps the flat
    data — the user must rebuild explicitly to convert.
    """
    spec = os.environ.get("RESEARCH_MCP_FAISS_INDEX_TYPE", "").strip().lower()
    if not spec or spec == "flat":
        return "flat"
    if spec == "hnsw":
        return "hnsw"
    raise RuntimeError(
        f"RESEARCH_MCP_FAISS_INDEX_TYPE={spec!r} not understood. "
        "Use 'flat' (default) or 'hnsw'."
    )


def _select_embedder() -> tuple[Embedder | None, str | None]:
    """Resolve the embedder selection from the environment.

    Returns (embedder, label) where label is the wire-level selection
    string for telemetry / library_status. (None, None) when nothing is
    configured — in which case the embedder-using tools degrade with a
    clear error message.
    """
    spec = os.environ.get("RESEARCH_MCP_EMBEDDER", "").strip()
    if spec:
        kind, _, model = spec.partition(":")
        kind = kind.strip().lower()
        model = model.strip()
        if kind == "openai":
            return OpenAIEmbedder(model or "text-embedding-3-small"), spec
        if kind in {"sentence-transformers", "st"}:
            return (
                SentenceTransformersEmbedder(
                    model or "BAAI/bge-base-en-v1.5"
                ),
                spec,
            )
        raise RuntimeError(
            f"RESEARCH_MCP_EMBEDDER={spec!r} not understood. "
            "Use 'openai:<model>' or 'sentence-transformers:<model>'."
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder(), "openai:text-embedding-3-small"
    return None, None


def _select_reranker() -> tuple[Reranker | None, str | None]:
    """Resolve the reranker selection from `RESEARCH_MCP_RERANKER`.

    Returns (reranker, label) where label is the wire-level selection
    string for library_status. (None, None) when unset — reranking is
    off by default because it adds 200-1000ms per search/recall and
    the user should opt in deliberately.
    """
    spec = os.environ.get("RESEARCH_MCP_RERANKER", "").strip()
    if not spec:
        return None, None
    kind, _, model = spec.partition(":")
    kind = kind.strip().lower()
    model = model.strip()
    if kind in {"cross-encoder", "hf-cross-encoder"}:
        return (
            HuggingFaceCrossEncoderReranker(
                model or "BAAI/bge-reranker-base"
            ),
            spec,
        )
    raise RuntimeError(
        f"RESEARCH_MCP_RERANKER={spec!r} not understood. "
        "Use 'cross-encoder:<model>'."
    )


def _claim_from_args(args: Any) -> Any:
    """Build a domain `Claim` from the flat claim_* fields on an input.

    `find_citations` and `explain_citation` accept the claim as flat
    parameters (claim_text, claim_type, claim_context, claim_search_
    terms) rather than a nested object — nested objects surface as
    $ref-style JSON schemas that some MCP clients (Claude Desktop
    among them) don't resolve, leaving the LLM to send a bare string.
    Flat is the LLM-tractable shape; this helper rebuilds the domain
    Claim from the parsed input model.
    """
    from research_mcp.domain.claim import Claim, ClaimType

    return Claim(
        text=args.claim_text,
        type=ClaimType(args.claim_type),
        confidence=0.85,  # synthetic; user-supplied claims aren't extractor-scored
        context=args.claim_context,
        suggested_search_terms=tuple(args.claim_search_terms),
    )


def _score_to_summary(score: Any) -> CitationQualityScoreSummary:
    return CitationQualityScoreSummary(
        total=score.total,
        venue=score.venue,
        impact=score.impact,
        author=score.author,
        recency=score.recency,
        factors=dict(score.factors),
        warnings=list(score.warnings),
    )




def _maybe_progress_callback(
    server: Server[Any, Any],
) -> Callable[[int, int, str], Awaitable[None]] | None:
    """Return a progress-notification callback bound to the active request.

    MCP clients opt into progress reporting by passing a `progressToken`
    in the request's `_meta`. If absent, we return None and the caller
    runs silently. If present, we return a callable that emits one
    `notifications/progress` message per call — Claude Desktop / other
    MCP clients render these as a live progress indicator on the tool
    call. Best-effort: a closed session or transport error swallows
    silently so a notification failure can't break the tool itself.
    """
    try:
        ctx = server.request_context
    except LookupError:
        return None
    meta = getattr(ctx, "meta", None)
    progress_token = getattr(meta, "progressToken", None)
    if progress_token is None:
        return None
    session = ctx.session

    async def _emit(progress: int, total: int, message: str) -> None:
        try:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=float(progress),
                total=float(total),
                message=message,
            )
        except Exception as exc:
            _log.debug("progress notification failed (ignored): %s", exc)

    return _emit


async def _resolve_paper(paper_id: str, lookup: Callable[[str], Awaitable[Paper | None]]) -> Paper:
    """Fetch a paper or raise a clean ValueError; shared by score/explain handlers."""
    try:
        paper = await lookup(paper_id)
    except SourceUnavailable as exc:
        raise ValueError(
            f"could not resolve {paper_id!r}: "
            f"{exc.source_name} is unavailable ({exc.short_reason()}). "
            "This is usually transient — try again."
        ) from exc
    if paper is None:
        raise ValueError(
            f"no configured source recognizes paper id {paper_id!r}. "
            "Use a prefixed id like 'arxiv:1706.03762', 'doi:10.1038/...', "
            "'s2:abc123', 'pmid:12345', or 'openalex:W123'."
        )
    return paper


def _fake_analyzer_for_test_mode() -> PaperAnalyzer:
    """Test-mode shortcut so e2e tests get an analysis_service without
    needing OpenAI/Anthropic credentials. Lives here to keep the
    `run_in_memory` wiring readable; production code never imports
    FakePaperAnalyzer through this path."""
    from research_mcp.paper_analyzer import FakePaperAnalyzer

    return FakePaperAnalyzer()


def _select_paper_analyzer() -> PaperAnalyzer | None:
    """Resolve `RESEARCH_MCP_ANALYSIS_MODEL` to a PaperAnalyzer instance.

    Format: `openai:<model>` or `anthropic:<model>`. Unset → None
    (analyze_paper refuses with a clear hint). Construction failures
    (missing API key, missing extra) are caught and logged so the
    server still boots.
    """
    spec = os.environ.get("RESEARCH_MCP_ANALYSIS_MODEL", "").strip()
    if not spec:
        return None
    kind, _, model = spec.partition(":")
    kind = kind.strip().lower()
    model = model.strip()
    try:
        if kind == "openai":
            from research_mcp.paper_analyzer import OpenAILLMPaperAnalyzer

            return OpenAILLMPaperAnalyzer(model=model or "gpt-4o-mini")
        if kind == "anthropic":
            from research_mcp.paper_analyzer import AnthropicLLMPaperAnalyzer

            return AnthropicLLMPaperAnalyzer(
                model=model or "claude-haiku-4-5-20251001"
            )
    except Exception as exc:  # pragma: no cover — env-specific failure
        _log.warning("paper analyzer construction failed: %s", exc)
        return None
    raise RuntimeError(
        f"RESEARCH_MCP_ANALYSIS_MODEL={spec!r} not understood. "
        "Use 'openai:<model>' or 'anthropic:<model>'."
    )


def _select_citation_scorer() -> CitationScorer:
    """Resolve `RESEARCH_MCP_CITATION_SCORER` to a CitationScorer instance.

    Formats:
      * unset / "field_aware"  → FieldAwareCitationScorer(Heuristic)  (default)
      * "heuristic"            → HeuristicCitationScorer (escape hatch — bare
                                 heuristic, no field overrides; useful for
                                 regression checks or callers that want the
                                 pre-#2 scoring behavior verbatim)
      * "llm:openai:<model>"   → OpenAILLMCitationScorer wrapping the field-
                                 aware default (composition order is
                                 inner=field, outer=LLM: the LLM relevance
                                 multiplier acts on a field-adjusted base
                                 score, not the bare heuristic)
      * "llm:anthropic:<model>" → AnthropicLLMCitationScorer wrapping the
                                  field-aware default (same shape)

    Construction failures (missing API key, missing extra) degrade to the
    bare heuristic with a warning so the server still boots.
    """
    spec = os.environ.get("RESEARCH_MCP_CITATION_SCORER", "").strip()
    if not spec or spec == "field_aware":
        from research_mcp.citation_scorer import FieldAwareCitationScorer

        return FieldAwareCitationScorer(HeuristicCitationScorer())
    if spec == "heuristic":
        return HeuristicCitationScorer()
    if spec.startswith("llm:"):
        return _build_llm_citation_scorer(spec[len("llm:"):])
    raise RuntimeError(
        f"RESEARCH_MCP_CITATION_SCORER={spec!r} not understood. "
        "Use 'field_aware' (default), 'heuristic' (escape hatch), "
        "'llm:openai:<model>', or 'llm:anthropic:<model>'."
    )


def _build_llm_citation_scorer(spec: str) -> CitationScorer:
    """Parse 'openai:<model>' or 'anthropic:<model>' into the right adapter.

    LLM scorers wrap the field-aware default (not the bare heuristic) so
    the LLM relevance multiplier composes with field-aware recency +
    impact rather than overriding them. To bypass field-awareness in an
    LLM-scored setup, set RESEARCH_MCP_CITATION_SCORER=heuristic and
    construct the LLM scorer programmatically with base_scorer=
    HeuristicCitationScorer().
    """
    from research_mcp.citation_scorer import FieldAwareCitationScorer

    base = FieldAwareCitationScorer(HeuristicCitationScorer())
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    try:
        if provider == "openai":
            from research_mcp.citation_scorer import OpenAILLMCitationScorer

            return OpenAILLMCitationScorer(
                model=model or "gpt-4o-mini",
                base_scorer=base,
            )
        if provider == "anthropic":
            from research_mcp.citation_scorer import AnthropicLLMCitationScorer

            return AnthropicLLMCitationScorer(
                model=model or "claude-haiku-4-5-20251001",
                base_scorer=base,
            )
    except Exception as exc:  # pragma: no cover — env-specific
        _log.warning(
            "LLM citation scorer construction failed (%s); falling back to heuristic",
            exc,
        )
        return HeuristicCitationScorer()
    _log.warning(
        "RESEARCH_MCP_CITATION_SCORER=llm:%r not understood; "
        "supported providers: openai, anthropic. Falling back to heuristic.",
        spec,
    )
    return HeuristicCitationScorer()


def _select_claim_extractor() -> ClaimExtractor | None:
    """Resolve the claim-extractor selection from the environment.

    `RESEARCH_MCP_CLAIM_EXTRACTOR` formats:
      * unset / "spacy"       → SpacyClaimExtractor (Round 1; ships ~80% precision)
      * "llm:openai:<model>"  → OpenAILLMClaimExtractor (Round 2; ~95% precision)
      * "llm:anthropic:<model>" → AnthropicLLMClaimExtractor (Round 2)

    Construction failures degrade to None — extract_claims and
    assist_draft refuse with a clear hint, the eight other tools stay
    available. We do not silently fall back to spaCy when the user
    asked for an LLM extractor: the precision difference is the whole
    point of the env var, so a failure should be visible.
    """
    spec = os.environ.get("RESEARCH_MCP_CLAIM_EXTRACTOR", "").strip().lower()
    if not spec or spec == "spacy":
        return _build_spacy_extractor()
    if spec.startswith("llm:"):
        return _build_llm_extractor(spec[len("llm:"):])
    _log.warning(
        "RESEARCH_MCP_CLAIM_EXTRACTOR=%r not understood; "
        "use 'spacy' or 'llm:openai:<model>' or 'llm:anthropic:<model>'.",
        spec,
    )
    return None


def _build_spacy_extractor() -> ClaimExtractor | None:
    try:
        from research_mcp.claim_extractor import SpacyClaimExtractor
    except ImportError:  # pragma: no cover — defensive
        return None
    try:
        return SpacyClaimExtractor()
    except RuntimeError as exc:
        _log.warning("spacy claim extractor unavailable: %s", exc)
        return None


def _build_llm_extractor(spec: str) -> ClaimExtractor | None:
    """Parse 'openai:<model>' or 'anthropic:<model>' into the right adapter."""
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    try:
        if provider == "openai":
            from research_mcp.claim_extractor import OpenAILLMClaimExtractor

            return OpenAILLMClaimExtractor(model=model or "gpt-4o-mini")
        if provider == "anthropic":
            from research_mcp.claim_extractor import AnthropicLLMClaimExtractor

            return AnthropicLLMClaimExtractor(
                model=model or "claude-haiku-4-5-20251001"
            )
    except Exception as exc:  # pragma: no cover — env-specific
        _log.warning("LLM claim extractor construction failed: %s", exc)
        return None
    _log.warning(
        "RESEARCH_MCP_CLAIM_EXTRACTOR=llm:%r not understood; "
        "supported providers: openai, anthropic.",
        spec,
    )
    return None


def build_server(
    *,
    search: SearchService,
    discovery: DiscoveryService,
    paper_lookup: Callable[[str], Awaitable[Paper | None]],
    library: LibraryService | None,
    embedder_label: str | None,
    reranker_label: str | None = None,
    index_type_label: str | None = None,
    claim_extractor: ClaimExtractor | None = None,
    citation_service: CitationService | None = None,
    analysis_service: AnalysisService | None = None,
    draft_service: DraftService | None = None,
    openalex: OpenAlexSource | None = None,
) -> Server[Any, Any]:
    """Construct an MCP `Server` with the six research tools registered.

    `library` may be None if no embedder is configured. In that mode the
    server still serves search/cite/get_paper/library_status; only
    ingest_paper and library_search refuse.
    """
    server: Server[Any, Any] = Server("research-mcp", version=__version__)

    # ---- MCP prompt: the one canonical entry point researchers want ----
    # We ship a single prompt template rather than a wall of them. The
    # research showed prompts are underused (~4% of public MCP servers
    # ship any) and the high-impact one for this server is "review my
    # draft for citation gaps" — bundles the right framing for
    # assist_draft and gives Claude Desktop's prompt menu an obvious
    # entry point a researcher will click before they'd type the tool
    # call by hand.
    @server.list_prompts()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_prompts() -> list[mcp_types.Prompt]:
        return [
            mcp_types.Prompt(
                name="review_draft_for_citations",
                description=(
                    "Review a draft paragraph for citation gaps. The model "
                    "will extract claims, find candidate citations across "
                    "all configured sources, and report ranked "
                    "recommendations with reasoning."
                ),
                arguments=[
                    mcp_types.PromptArgument(
                        name="draft",
                        description=(
                            "The draft text to review. A paragraph or short "
                            "section works best (under ~20K characters)."
                        ),
                        required=True,
                    ),
                ],
            ),
        ]

    @server.get_prompt()  # type: ignore[no-untyped-call,untyped-decorator]
    async def get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> mcp_types.GetPromptResult:
        if name != "review_draft_for_citations":
            raise ValueError(f"unknown prompt: {name}")
        draft = (arguments or {}).get("draft", "")
        if not draft.strip():
            raise ValueError(
                "review_draft_for_citations requires non-empty `draft`."
            )
        return mcp_types.GetPromptResult(
            description=(
                "Review the user's draft for citation gaps via the "
                "research-mcp assist_draft pipeline."
            ),
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(
                        type="text",
                        text=(
                            "Review the following draft for citation "
                            "gaps. Call the `assist_draft` tool with the "
                            "draft text below, then summarize the top "
                            "recommendation for each detected claim "
                            "(strong/moderate/weak verdict + the venue "
                            "and key reasoning). If any claim has no "
                            "strong candidate, say so explicitly rather "
                            "than recommending a weak one.\n\n"
                            f"Draft:\n{draft}"
                        ),
                    ),
                ),
            ],
        )

    # mcp SDK ships its decorators as untyped at the moment.
    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="search_papers",
                description=(
                    "Search arXiv and Semantic Scholar in parallel and return "
                    "deduplicated, cross-source-enriched metadata for each "
                    "paper. Each result carries a `source` field naming which "
                    "adapter(s) contributed."
                ),
                inputSchema=SearchPapersInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="ingest_paper",
                description=(
                    "Add papers to the local FAISS-backed library so they "
                    "can be recalled by similarity. Two modes: pass "
                    "`paper_id` to ingest one specific paper, or pass "
                    "`query` (with optional `max_papers`, `year_min`, "
                    "`year_max`) to search all configured sources and "
                    "bulk-ingest the top-N. Requires an embedder; see "
                    "library_status if unsure whether the server is "
                    "configured for ingest."
                ),
                inputSchema=IngestPaperInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="library_search",
                description=(
                    "Semantic search across the local library; returns the top-k "
                    "ingested papers with similarity scores. Requires an "
                    "embedder."
                ),
                inputSchema=LibrarySearchInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="cite_paper",
                description=(
                    "Render a citation for a paper id. Fetches metadata from "
                    "the originating source on demand — does not require the "
                    "paper to be ingested first. Defaults to AMA; supports "
                    "APA, MLA, Chicago, and BibTeX."
                ),
                inputSchema=CitePaperInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="library_status",
                description=(
                    "Report library state: paper count, configured embedder, "
                    "any setup hints. Use to verify the server is wired for "
                    "ingest before attempting one."
                ),
                inputSchema=LibraryStatusInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="get_paper",
                description=(
                    "Fetch full Paper metadata for an id without ingesting. "
                    "Useful as a preview step before deciding whether to "
                    "commit to embedding the paper into the local library."
                ),
                inputSchema=GetPaperInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="find_paper",
                description=(
                    "Find a paper by title (and optional author names) when "
                    "you don't have a canonical id. Returns at most three "
                    "candidates ranked by title-token similarity with a "
                    "confidence score. Use this to bridge from a citation "
                    "you've read about to an id you can ingest or cite."
                ),
                inputSchema=FindPaperInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="extract_claims",
                description=(
                    "Scan draft text and identify claims that need citations: "
                    "statistical (percentages, p-values, sample sizes), "
                    "methodological (techniques, algorithms), comparative "
                    "(outperforms / better than), causal, and theoretical. "
                    "Each claim carries its type, a confidence score, the "
                    "surrounding context, and suggested search terms — feed "
                    "those into search_papers / find_citations to find the "
                    "papers worth citing."
                ),
                inputSchema=ExtractClaimsInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="find_citations",
                description=(
                    "Given a Claim (typically from extract_claims), search "
                    "all configured sources, score each candidate by venue + "
                    "impact + recency, and return the top-k recommended "
                    "citations. Each candidate carries its full quality "
                    "breakdown, not just a total — so the user can see WHY "
                    "a paper ranked where it did."
                ),
                inputSchema=FindCitationsInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="explain_citation",
                description=(
                    "Produce a human-readable recommendation for citing a "
                    "specific paper as evidence for a specific claim. "
                    "Returns a strong/moderate/weak verdict plus the "
                    "venue + impact + recency reasoning the user can show "
                    "to a co-author or reviewer."
                ),
                inputSchema=ExplainCitationInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="analyze_paper",
                description=(
                    "Use an LLM to extract structured analysis of a paper: "
                    "summary, key contributions, methodology, technical "
                    "approach, limitations, future directions, datasets, "
                    "metrics, and baselines. Pass `kinds` to limit which "
                    "fields are extracted (saves output tokens). Backed "
                    "by OpenAI gpt-4o-mini or Anthropic claude-haiku, "
                    "selected via RESEARCH_MCP_ANALYSIS_MODEL."
                ),
                inputSchema=AnalyzePaperInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="assist_draft",
                description=(
                    "End-to-end citation assistant: paste a draft paragraph, "
                    "get a list of recommended citations per claim. The "
                    "pipeline extracts typed claims, finds candidate papers "
                    "across all configured sources (arXiv, Semantic Scholar, "
                    "PubMed, OpenAlex), scores each by venue + impact + "
                    "recency, and returns ranked recommendations with "
                    "human-readable explanations. Streams progress "
                    "notifications when the client supplies a "
                    "progressToken — the LLM sees 'claim 3/8 done' "
                    "messages as the pipeline runs."
                ),
                inputSchema=AssistDraftInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="find_referenced_by",
                description=(
                    "Walk OpenAlex's outgoing citation graph: return up to "
                    "`max_results` papers that the given paper cites. The "
                    "source paper id must be OpenAlex- or DOI-prefixed "
                    "(e.g. 'openalex:W2741809807', 'doi:10.1038/nature12373') "
                    "because referenced_works is an OpenAlex-only signal — "
                    "arXiv- and S2-only ids aren't supported. Requires "
                    "RESEARCH_MCP_OPENALEX_EMAIL to be set; the tool refuses "
                    "with a hint otherwise."
                ),
                inputSchema=FindReferencedByInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="find_related",
                description=(
                    "Return OpenAlex's similarity-neighborhood for the given "
                    "paper. Unlike `find_referenced_by`, this isn't a "
                    "deterministic citation graph — `related_works` is "
                    "computed by OpenAlex from topic-vector similarity, so "
                    "treat results as 'papers OpenAlex thinks are adjacent' "
                    "rather than 'papers this one cites'. Same prefix rules "
                    "and email requirement as find_referenced_by."
                ),
                inputSchema=FindRelatedInput.model_json_schema(),
            ),
        ]

    async def _do_search(arguments: dict[str, Any]) -> dict[str, Any]:
        args = SearchPapersInput.model_validate(arguments)
        outcome = await search.search(
            SearchQuery(
                text=args.query,
                max_results=args.max_results,
                year_min=args.year_min,
                year_max=args.year_max,
            )
        )
        return SearchPapersOutput(
            results=[
                paper_to_summary(r.paper, source="+".join(r.sources))
                for r in outcome.results
            ],
            partial_failures=list(outcome.partial_failures),
            source_contributions=dict(outcome.source_contributions),
        ).model_dump()

    async def _do_ingest(arguments: dict[str, Any]) -> dict[str, Any]:
        if library is None:
            raise ValueError(f"ingest_paper unavailable: {_NO_EMBEDDER_HINT}")
        args = IngestPaperInput.model_validate(arguments)
        # Step-level timing logs help diagnose hangs without logging
        # payloads. Two modes share one tool: single-id (one paper) vs
        # query (top-N from search + batched embed/upsert).
        t_start = time.monotonic()
        partial_failures: list[str] = []
        # For single-id mode we know the id up front and can pre-check
        # presence. For query mode we don't know the ids until after
        # the search, so the pre-check happens between search and bulk
        # ingest. Either way, capturing the pre-state lets us report
        # how many records were *new* (vs upserts of already-present
        # papers), which is the signal the user actually cares about.
        if args.paper_id is not None:
            already_present_pre = await library.contains([args.paper_id])
            paper = await library.ingest(args.paper_id)
            ingested: list[Paper] = [paper]
        else:
            assert args.query is not None  # enforced by IngestPaperInput validator
            outcome = await search.search(
                SearchQuery(
                    text=args.query,
                    max_results=args.max_papers,
                    year_min=args.year_min,
                    year_max=args.year_max,
                )
            )
            partial_failures = list(outcome.partial_failures)
            result_ids = [r.paper.id for r in outcome.results]
            already_present_pre = await library.contains(result_ids)
            ingested = list(
                await library.bulk_ingest([r.paper for r in outcome.results])
            )
        t_after_ingest = time.monotonic()
        newly_added = sum(1 for p in ingested if p.id not in already_present_pre)
        count = await library.count()
        t_after_count = time.monotonic()
        _log.info(
            "ingest_paper mode=%s n=%d new=%d ingest_ms=%.0f count_ms=%.0f",
            "id" if args.paper_id else "query",
            len(ingested),
            newly_added,
            (t_after_ingest - t_start) * 1000,
            (t_after_count - t_after_ingest) * 1000,
        )
        return IngestPaperOutput(
            ingested=[
                paper_to_summary(
                    p, source=source_from_id(p.id, search.sources)
                )
                for p in ingested
            ],
            newly_added=newly_added,
            library_count=count,
            partial_failures=partial_failures,
        ).model_dump()

    async def _do_recall(arguments: dict[str, Any]) -> dict[str, Any]:
        if library is None:
            raise ValueError(f"library_search unavailable: {_NO_EMBEDDER_HINT}")
        args = LibrarySearchInput.model_validate(arguments)
        results = await library.recall(args.query, k=args.k)
        return LibrarySearchOutput(
            results=[
                LibrarySearchHit(
                    paper=paper_to_summary(p, source=source_from_id(p.id, search.sources)),
                    score=score,
                )
                for p, score in results
            ]
        ).model_dump()

    async def _do_cite(arguments: dict[str, Any]) -> dict[str, Any]:
        args = CitePaperInput.model_validate(arguments)
        try:
            paper = await paper_lookup(args.paper_id)
        except SourceUnavailable as exc:
            raise ValueError(
                f"could not resolve {args.paper_id!r}: "
                f"{exc.source_name} is unavailable ({exc.short_reason()}). "
                "This is usually transient — try again. Valid id prefixes: "
                "'arxiv:1706.03762', 'doi:10.1038/...', 's2:abc123'."
            ) from exc
        if paper is None:
            raise ValueError(
                f"no configured source recognizes paper id {args.paper_id!r}. "
                "Use a prefixed id like 'arxiv:1706.03762', 'doi:10.1038/...', "
                "or 's2:abc123'."
            )
        renderer = RENDERERS[CitationFormat(args.format)]
        return CitePaperOutput(
            citation=renderer.render(paper),
            format=args.format,
        ).model_dump()

    async def _do_status(arguments: dict[str, Any]) -> dict[str, Any]:
        LibraryStatusInput.model_validate(arguments)
        source_names = [s.name for s in search.sources]
        extractor_name = (
            claim_extractor.name if claim_extractor is not None else None
        )
        analyzer_name = (
            analysis_service.analyzer.name
            if analysis_service is not None
            else None
        )
        scorer_name = (
            citation_service.scorer.name
            if citation_service is not None
            else None
        )
        if library is None:
            return LibraryStatusOutput(
                count=0,
                embedder=None,
                reranker=reranker_label,
                index_type=None,
                sources=source_names,
                claim_extractor=extractor_name,
                paper_analyzer=analyzer_name,
                citation_scorer=scorer_name,
                note=_NO_EMBEDDER_HINT,
            ).model_dump()
        return LibraryStatusOutput(
            count=await library.count(),
            embedder=embedder_label,
            reranker=reranker_label,
            index_type=index_type_label,
            sources=source_names,
            claim_extractor=extractor_name,
            paper_analyzer=analyzer_name,
            citation_scorer=scorer_name,
            note=None,
        ).model_dump()

    async def _do_extract_claims(arguments: dict[str, Any]) -> dict[str, Any]:
        if claim_extractor is None:
            raise ValueError(
                "extract_claims unavailable: no claim extractor configured. "
                "Install the optional [claim-extraction] extra (spaCy + "
                "en_core_web_sm)."
            )
        args = ExtractClaimsInput.model_validate(arguments)
        claims = await claim_extractor.extract(args.text)
        return ExtractClaimsOutput(
            claims=[
                ClaimSummary(
                    text=c.text,
                    type=c.type.value,
                    confidence=c.confidence,
                    context=c.context,
                    suggested_search_terms=list(c.suggested_search_terms),
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
                for c in claims
            ],
            extractor=claim_extractor.name,
        ).model_dump()

    async def _do_find_citations(arguments: dict[str, Any]) -> dict[str, Any]:
        if citation_service is None:
            raise ValueError(_CITATION_UNAVAILABLE_HINT)
        args = FindCitationsInput.model_validate(arguments)
        claim = _claim_from_args(args)
        candidates = await citation_service.find_citations(claim, k=args.k)
        return FindCitationsOutput(
            candidates=[
                CitationCandidateSummary(
                    paper=paper_to_summary(c.paper, source="+".join(c.sources)),
                    score=_score_to_summary(c.score),
                )
                for c in candidates
            ],
            scorer=citation_service.scorer.name,
        ).model_dump()

    async def _do_explain_citation(arguments: dict[str, Any]) -> dict[str, Any]:
        if citation_service is None:
            raise ValueError(_CITATION_UNAVAILABLE_HINT)
        args = ExplainCitationInput.model_validate(arguments)
        paper = await _resolve_paper(args.paper_id, paper_lookup)
        claim = _claim_from_args(args)
        explanation = await citation_service.explain_citation(paper, claim)
        score = await citation_service.score_citation(paper, claim)
        return ExplainCitationOutput(
            explanation=explanation,
            score=_score_to_summary(score),
            paper=paper_to_summary(paper, source=source_from_id(paper.id, search.sources)),
        ).model_dump()

    async def _do_assist_draft(arguments: dict[str, Any]) -> dict[str, Any]:
        if draft_service is None:
            raise ValueError(
                "assist_draft unavailable: needs a ClaimExtractor and a "
                "CitationService. Install [claim-extraction] (spaCy + "
                "en_core_web_sm); a citation_service is wired by default."
            )
        args = AssistDraftInput.model_validate(arguments)
        progress_cb = _maybe_progress_callback(server)
        recommendations = await draft_service.assist(
            args.text,
            k_per_claim=args.k_per_claim,
            progress=progress_cb,
        )
        return AssistDraftOutput(
            recommendations=[
                CitationRecommendationSummary(
                    claim=ClaimSummary(
                        text=rec.claim.text,
                        type=rec.claim.type.value,
                        confidence=rec.claim.confidence,
                        context=rec.claim.context,
                        suggested_search_terms=list(
                            rec.claim.suggested_search_terms
                        ),
                        start_char=rec.claim.start_char,
                        end_char=rec.claim.end_char,
                    ),
                    candidates=[
                        CitationRecommendationCandidateSummary(
                            paper=paper_to_summary(
                                c.paper,
                                source="+".join(c.sources),
                            ),
                            score_total=c.score_total,
                            score_warnings=list(c.score_warnings),
                            explanation=c.explanation,
                        )
                        for c in rec.candidates
                    ],
                )
                for rec in recommendations
            ],
            extractor=draft_service.extractor.name,
            scorer=draft_service.citation.scorer.name,
        ).model_dump()

    async def _do_analyze_paper(arguments: dict[str, Any]) -> dict[str, Any]:
        if analysis_service is None:
            raise ValueError(
                "analyze_paper unavailable: no analysis service configured. "
                "Set RESEARCH_MCP_ANALYSIS_MODEL to "
                "'openai:gpt-4o-mini' (requires OPENAI_API_KEY) or "
                "'anthropic:claude-haiku-4-5-20251001' (requires "
                "ANTHROPIC_API_KEY)."
            )
        args = AnalyzePaperInput.model_validate(arguments)
        # Same diagnostic motivation as ingest_paper: split the work
        # into measurable phases (paper resolution vs LLM call) so a
        # next-time hang shows which step is responsible.
        t_start = time.monotonic()
        paper = await _resolve_paper(args.paper_id, paper_lookup)
        t_after_resolve = time.monotonic()
        kinds = tuple(AnalysisKind(k) for k in args.kinds)
        analysis = await analysis_service.analyze(paper, kinds)
        t_after_analyze = time.monotonic()
        _log.info(
            "analyze_paper id=%s resolve_ms=%.0f analyze_ms=%.0f model=%s",
            args.paper_id,
            (t_after_resolve - t_start) * 1000,
            (t_after_analyze - t_after_resolve) * 1000,
            analysis.model,
        )
        return AnalyzePaperOutput(
            paper=paper_to_summary(paper, source=source_from_id(paper.id, search.sources)),
            analysis=PaperAnalysisSummary(
                paper_id=analysis.paper_id,
                summary=analysis.summary,
                key_contributions=list(analysis.key_contributions),
                methodology=analysis.methodology,
                technical_approach=analysis.technical_approach,
                limitations=list(analysis.limitations),
                future_directions=list(analysis.future_directions),
                datasets_used=list(analysis.datasets_used),
                metrics_reported=dict(analysis.metrics_reported),
                baselines_compared=list(analysis.baselines_compared),
                confidence=analysis.confidence,
                model=analysis.model,
            ),
        ).model_dump()

    async def _do_find_paper(arguments: dict[str, Any]) -> dict[str, Any]:
        args = FindPaperInput.model_validate(arguments)
        outcome = await discovery.find_paper(
            title=args.title, authors=tuple(args.authors)
        )
        note: str | None = None
        if not outcome.hits:
            from research_mcp.service.discovery import has_significant_tokens

            if not has_significant_tokens(args.title):
                note = (
                    "title contained only stopwords; no significant tokens "
                    "to match against. Try a more specific title."
                )
        return FindPaperOutput(
            results=[
                FindPaperHit(
                    paper=paper_to_summary(h.paper, source="+".join(h.sources)),
                    confidence=h.confidence,
                )
                for h in outcome.hits
            ],
            note=note,
            partial_failures=list(outcome.partial_failures),
        ).model_dump()

    async def _do_get_paper(arguments: dict[str, Any]) -> dict[str, Any]:
        args = GetPaperInput.model_validate(arguments)
        try:
            paper = await paper_lookup(args.paper_id)
        except SourceUnavailable as exc:
            raise ValueError(
                f"could not resolve {args.paper_id!r}: "
                f"{exc.source_name} is unavailable ({exc.short_reason()}). "
                "This is usually transient — try again. Valid id prefixes: "
                "'arxiv:1706.03762', 'doi:10.1038/...', 's2:abc123'."
            ) from exc
        if paper is None:
            raise ValueError(
                f"no configured source recognizes paper id {args.paper_id!r}. "
                "Use a prefixed id like 'arxiv:1706.03762', 'doi:10.1038/...', "
                "or 's2:abc123'."
            )
        return GetPaperOutput(
            paper=paper_to_summary(paper, source=source_from_id(paper.id, search.sources))
        ).model_dump()

    async def _do_find_referenced_by(arguments: dict[str, Any]) -> dict[str, Any]:
        if openalex is None:
            raise ValueError(
                "find_referenced_by unavailable: OpenAlex isn't configured. "
                "Set RESEARCH_MCP_OPENALEX_EMAIL to enable it. The "
                "referenced_works signal is OpenAlex-only — other sources "
                "(arXiv, Semantic Scholar, PubMed) don't expose outgoing-"
                "citation graphs through their public APIs."
            )
        args = FindReferencedByInput.model_validate(arguments)
        try:
            referenced = await openalex.fetch_referenced(
                args.paper_id, limit=args.max_results
            )
        except SourceUnavailable as exc:
            raise ValueError(
                f"could not walk references for {args.paper_id!r}: "
                f"{exc.source_name} is unavailable ({exc.short_reason()}). "
                "This is usually transient — try again."
            ) from exc
        return FindReferencedByOutput(
            results=[
                paper_to_summary(p, source=source_from_id(p.id, search.sources))
                for p in referenced
            ],
        ).model_dump()

    async def _do_find_related(arguments: dict[str, Any]) -> dict[str, Any]:
        if openalex is None:
            raise ValueError(
                "find_related unavailable: OpenAlex isn't configured. "
                "Set RESEARCH_MCP_OPENALEX_EMAIL to enable it. The "
                "related_works signal is OpenAlex-only."
            )
        args = FindRelatedInput.model_validate(arguments)
        try:
            related = await openalex.fetch_related(
                args.paper_id, limit=args.max_results
            )
        except SourceUnavailable as exc:
            raise ValueError(
                f"could not find related works for {args.paper_id!r}: "
                f"{exc.source_name} is unavailable ({exc.short_reason()}). "
                "This is usually transient — try again."
            ) from exc
        return FindRelatedOutput(
            results=[
                paper_to_summary(p, source=source_from_id(p.id, search.sources))
                for p in related
            ],
        ).model_dump()

    handlers: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
        "search_papers": _do_search,
        "ingest_paper": _do_ingest,
        "library_search": _do_recall,
        "cite_paper": _do_cite,
        "library_status": _do_status,
        "get_paper": _do_get_paper,
        "find_paper": _do_find_paper,
        "extract_claims": _do_extract_claims,
        "find_citations": _do_find_citations,
        "explain_citation": _do_explain_citation,
        "analyze_paper": _do_analyze_paper,
        "assist_draft": _do_assist_draft,
        "find_referenced_by": _do_find_referenced_by,
        "find_related": _do_find_related,
    }

    # validate_input=False bypasses the mcp SDK's strict jsonschema check so
    # pydantic — which is doing the same job inside each handler — gets first
    # crack at the arguments. The motivation is concrete: model clients
    # frequently serialize numeric tool args as JSON strings ("2018" instead
    # of 2018). jsonschema rejects those as type-mismatched; pydantic's
    # default lax mode coerces them. extra="forbid" on each Input model still
    # bounces hallucinated unknown keys, so we don't lose schema strictness.
    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]  # mcp SDK decorators are untyped
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        # Log every call: name, arg keys (not values — args may be large
        # queries or paper bodies), elapsed ms, result-shape hint. The next
        # agent debugging an issue should be able to grep the Claude Desktop
        # log for tool=cite_paper and reconstruct the call sequence.
        arg_keys = ",".join(sorted(arguments.keys())) or "-"
        start = time.monotonic()
        budget = _TOOL_TIMEOUTS.get(name, _DEFAULT_TOOL_TIMEOUT)
        try:
            result = await asyncio.wait_for(handler(arguments), timeout=budget)
        except TimeoutError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            _log.warning(
                "tool=%s args=%s elapsed=%.0fms timeout_after=%.0fs",
                name, arg_keys, elapsed_ms, budget,
            )
            raise ValueError(
                f"{name} timed out after {budget:.0f}s. This is usually "
                "an upstream rate limit or LLM API slowdown — try again "
                "in a moment, or simplify the query."
            ) from exc
        except ValidationError as exc:
            # Pydantic's default __str__ embeds a docs URL
            # (https://errors.pydantic.dev/...) in the message. That leaks
            # framework noise to the LLM client — same class of leakage we
            # cleaned out of the SourceUnavailable path. Reformat to a
            # clean "field: message" line and drop the URL.
            elapsed_ms = (time.monotonic() - start) * 1000
            _log.warning(
                "tool=%s args=%s elapsed=%.0fms validation_error",
                name, arg_keys, elapsed_ms,
            )
            raise ValueError(_format_validation_error(name, exc)) from exc
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            _log.warning(
                "tool=%s args=%s elapsed=%.0fms error=%s: %s",
                name, arg_keys, elapsed_ms, type(exc).__name__, exc,
            )
            raise
        elapsed_ms = (time.monotonic() - start) * 1000
        _log.info(
            "tool=%s args=%s elapsed=%.0fms results=%s",
            name, arg_keys, elapsed_ms, _result_hint(name, result),
        )
        return result

    return server


def _format_validation_error(tool_name: str, exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a clean, URL-free message.

    Pydantic v2's default str(exc) includes a 'For further information visit
    https://errors.pydantic.dev/...' line that should not reach the LLM.
    """
    parts: list[str] = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "validation failed")
        parts.append(f"{field}: {msg}" if field else msg)
    joined = "; ".join(parts) if parts else "invalid input"
    return f"invalid arguments for {tool_name}: {joined}"


def _result_hint(tool_name: str, result: dict[str, Any]) -> str:
    """A compact, log-friendly summary of a tool result's shape.

    Keeps the log line short — the full payload may run hundreds of KB
    for a search response — while preserving enough signal for grep
    queries like "find tool calls that returned zero results".
    """
    if isinstance(result.get("results"), list):
        return f"n={len(result['results'])}"
    if tool_name == "library_status":
        return f"count={result.get('count')}"
    if tool_name == "ingest_paper":
        return (
            f"ingested={len(result.get('ingested', []))} "
            f"library_count={result.get('library_count')}"
        )
    if "paper" in result:
        return "paper=1"
    if "citation" in result:
        return f"format={result.get('format')}"
    return "-"


def _configure_logging() -> None:
    """Set up `research_mcp.*` logging on stderr at INFO.

    Scoped to our namespace so we don't override user / library logging
    config. Idempotent — re-runs in the same process don't duplicate
    handlers. Only the entry points (`run_default`, `run_in_memory`,
    CLI `main`) call this; importing the module is side-effect free.
    """
    pkg_log = logging.getLogger("research_mcp")
    if pkg_log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    pkg_log.addHandler(handler)
    pkg_log.setLevel(logging.INFO)


async def run_default() -> None:
    """Production wiring: real arXiv + S2; embedder selected from env.

    With no embedder configured, the server still boots in degraded mode
    (search/cite/get_paper work; ingest/recall refuse with a clear hint).
    """
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    # PubMed is enabled by default — research-mcp's primary medical-use case
    # depends on it. `RESEARCH_MCP_DISABLE_PUBMED=1` opts out for users who
    # want a leaner source list (e.g. CS-only workflows).
    pubmed: PubMedSource | None = None
    sources_list: list[Source] = [arxiv, s2]
    if os.environ.get("RESEARCH_MCP_DISABLE_PUBMED") != "1":
        pubmed = PubMedSource()
        sources_list.append(pubmed)
    # OpenAlex is opt-in: their polite-pool guidance requires a `mailto`,
    # and rather than ship a placeholder we treat the email as the
    # opt-in signal. Set RESEARCH_MCP_OPENALEX_EMAIL to enable.
    openalex: OpenAlexSource | None = None
    openalex_email = os.environ.get("RESEARCH_MCP_OPENALEX_EMAIL")
    if openalex_email:
        openalex = OpenAlexSource(email=openalex_email)
        sources_list.append(openalex)
    sources: tuple[Source, ...] = tuple(sources_list)
    reranker, reranker_label = _select_reranker()
    search = SearchService(sources, reranker=reranker)
    discovery = DiscoveryService(search)

    embedder, label = _select_embedder()
    library: LibraryService | None = None
    index_to_close: FaissIndex | None = None
    index_type_label: str | None = None
    if embedder is not None:
        index_path = os.environ.get("RESEARCH_MCP_INDEX_PATH")
        if not index_path:
            raise RuntimeError(
                "RESEARCH_MCP_INDEX_PATH is required when an embedder is "
                "configured. Set it to a writable directory; FAISS files "
                "will live there."
            )
        index_type = _resolve_faiss_index_type()
        index = FaissIndex(
            index_path, dimension=embedder.dimension, index_type=index_type
        )
        index_to_close = index
        index_type_label = index.index_type
        library = LibraryService(
            index=index,
            embedder=embedder,
            ingest_sources=sources,
            reranker=reranker,
        )
    else:
        _log.warning("no embedder configured: %s", _NO_EMBEDDER_HINT)

    async def paper_lookup(paper_id: str) -> Paper | None:
        # See `fetch_with_enrichment` for why we go through it instead
        # of `fetch_from_sources` directly: enrichment is what gets
        # venue + citation_count onto arxiv-only records.
        return await fetch_with_enrichment(sources, paper_id)

    paper_analyzer = _select_paper_analyzer()
    claim_extractor = _select_claim_extractor()
    citation_svc = CitationService(search=search, scorer=_select_citation_scorer())
    draft_svc = (
        DraftService(extractor=claim_extractor, citation=citation_svc)
        if claim_extractor is not None
        else None
    )
    server = build_server(
        search=search,
        discovery=discovery,
        paper_lookup=paper_lookup,
        library=library,
        embedder_label=label,
        reranker_label=reranker_label,
        index_type_label=index_type_label,
        claim_extractor=claim_extractor,
        citation_service=citation_svc,
        analysis_service=(
            AnalysisService(analyzer=paper_analyzer)
            if paper_analyzer is not None
            else None
        ),
        draft_service=draft_svc,
        openalex=openalex,
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await arxiv.aclose()
        await s2.aclose()
        if pubmed is not None:
            await pubmed.aclose()
        if openalex is not None:
            await openalex.aclose()
        if index_to_close is not None:
            index_to_close.close()


async def run_in_memory() -> None:
    """In-memory wiring used by the e2e harness — no API keys required.

    Selected when `RESEARCH_MCP_TEST_MODE=1` so the e2e test can boot a real
    server subprocess without needing OpenAI or a writable index path.
    """
    arxiv = ArxivSource()
    sources: tuple[Source, ...] = (arxiv,)
    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    library = LibraryService(index=index, embedder=embedder, ingest_sources=sources)
    search = SearchService(sources)
    discovery = DiscoveryService(search)
    # run_in_memory ignores RESEARCH_MCP_RERANKER on purpose — the e2e
    # test mode exists precisely to avoid pulling 250 MB of cross-encoder
    # weights every test run.

    async def paper_lookup(paper_id: str) -> Paper | None:
        # See `fetch_with_enrichment` for why we go through it instead
        # of `fetch_from_sources` directly: enrichment is what gets
        # venue + citation_count onto arxiv-only records.
        return await fetch_with_enrichment(sources, paper_id)

    test_mode_extractor = _select_claim_extractor()
    test_mode_citation = CitationService(
        search=search, scorer=HeuristicCitationScorer()
    )
    server = build_server(
        search=search,
        discovery=discovery,
        paper_lookup=paper_lookup,
        library=library,
        embedder_label="fake:test-mode",
        claim_extractor=test_mode_extractor,
        citation_service=test_mode_citation,
        # run_in_memory ignores RESEARCH_MCP_ANALYSIS_MODEL on purpose;
        # the e2e harness uses FakePaperAnalyzer to avoid burning API
        # credits on every test run.
        analysis_service=AnalysisService(analyzer=_fake_analyzer_for_test_mode()),
        draft_service=(
            DraftService(extractor=test_mode_extractor, citation=test_mode_citation)
            if test_mode_extractor is not None
            else None
        ),
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await arxiv.aclose()


async def main() -> None:
    from research_mcp._env import load_dotenv

    load_dotenv()
    _configure_logging()
    if os.environ.get("RESEARCH_MCP_TEST_MODE") == "1":
        await run_in_memory()
    else:
        await run_default()
