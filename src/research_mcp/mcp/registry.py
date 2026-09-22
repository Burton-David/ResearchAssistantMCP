"""One declarative entry per tool, used for both advertising and dispatch.

`list_tools` and the dispatch table were separate literals keyed by tool name.
A tool present in one and missing from the other produced either a tool clients
could call but never discover, or one they discovered and could not call, and
neither shape fails a type check or a test that does not exercise that tool.

A `ToolSpec` carries the advertised description, the input model the schema is
derived from, and the handler, so the two views cannot disagree. Adding a tool
is one entry.

Per-tool timeouts stay in `server._TOOL_TIMEOUTS` rather than moving here: they
are read at dispatch time so a test can widen or narrow a budget on a server
that is already running.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from typing import Any

import mcp.types as mcp_types
from pydantic import BaseModel

from research_mcp.mcp.tools import (
    AnalyzePaperInput,
    AssistDraftInput,
    CitePaperInput,
    ExplainCitationInput,
    ExtractClaimsInput,
    FindCitationsInput,
    FindPaperInput,
    FindReferencedByInput,
    FindRelatedInput,
    GetPaperInput,
    IngestPaperInput,
    LibrarySearchInput,
    LibraryStatusInput,
    SearchPapersInput,
)

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool's advertised surface and the coroutine that serves it."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler

    def to_mcp_tool(self) -> mcp_types.Tool:
        """Render the protocol-level tool description sent to clients.

        The schema is derived from `input_model` rather than written out, so
        the advertised contract cannot drift from what the handler validates.
        """
        return mcp_types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_model.model_json_schema(),
        )


def advertise(specs: list[ToolSpec]) -> list[mcp_types.Tool]:
    """Render every spec for a `list_tools` response."""
    return [spec.to_mcp_tool() for spec in specs]


def dispatch_table(specs: list[ToolSpec]) -> dict[str, ToolHandler]:
    """Index handlers by tool name for `call_tool`.

    Raises `ValueError` on a duplicate name. Two specs sharing a name would
    otherwise advertise both and silently serve whichever came last.
    """
    table: dict[str, ToolHandler] = {}
    for spec in specs:
        if spec.name in table:
            raise ValueError(f"duplicate tool name in registry: {spec.name!r}")
        table[spec.name] = spec.handler
    return table


@dataclass(frozen=True, slots=True)
class ToolHandlers:
    """The coroutine serving each tool, one field per tool name.

    `build_server` closes over its injected services to make these, so they
    cannot live at module scope. Naming a field per tool means omitting one
    is a type error at the call site rather than a missing key discovered
    when a client calls the tool.
    """

    search_papers: ToolHandler
    ingest_paper: ToolHandler
    library_search: ToolHandler
    cite_paper: ToolHandler
    library_status: ToolHandler
    get_paper: ToolHandler
    find_paper: ToolHandler
    extract_claims: ToolHandler
    find_citations: ToolHandler
    explain_citation: ToolHandler
    analyze_paper: ToolHandler
    assist_draft: ToolHandler
    find_referenced_by: ToolHandler
    find_related: ToolHandler


def build_specs(handlers: ToolHandlers) -> list[ToolSpec]:
    """The full tool table: what each tool is called, does, accepts, and runs.

    Descriptions are written for the model reading them, not for a human
    skimming source, so they name concrete ids, env vars, and sibling tools.
    """
    specs = [
        ToolSpec(
            name="search_papers",
            description=(
                "Search arXiv and Semantic Scholar in parallel and return "
                "deduplicated, cross-source-enriched metadata for each "
                "paper. Each result carries a `source` field naming which "
                "adapter(s) contributed."
            ),
            input_model=SearchPapersInput,
            handler=handlers.search_papers,
        ),
        ToolSpec(
            name="ingest_paper",
            description=(
                "Add papers to the local FAISS-backed library so they "
                "can be recalled by similarity. Two modes: pass "
                "`paper_id` to ingest one specific paper, or pass "
                "`query` (with optional `max_papers`, `year_min`, "
                "`year_max`) to search all configured sources and "
                "bulk-ingest the top-N. Query mode streams progress "
                "notifications when the client supplies a progressToken — "
                "the LLM sees 'embedding 20 papers' / 'indexing' updates "
                "as the ingest runs. Requires an embedder; see "
                "library_status if unsure whether the server is "
                "configured for ingest."
            ),
            input_model=IngestPaperInput,
            handler=handlers.ingest_paper,
        ),
        ToolSpec(
            name="library_search",
            description=(
                "Semantic search across the local library; returns the top-k "
                "ingested papers with similarity scores. Requires an "
                "embedder."
            ),
            input_model=LibrarySearchInput,
            handler=handlers.library_search,
        ),
        ToolSpec(
            name="cite_paper",
            description=(
                "Render a citation for a paper id. Fetches metadata from "
                "the originating source on demand — does not require the "
                "paper to be ingested first. Defaults to AMA; supports "
                "APA, MLA, Chicago, and BibTeX."
            ),
            input_model=CitePaperInput,
            handler=handlers.cite_paper,
        ),
        ToolSpec(
            name="library_status",
            description=(
                "Report library state: paper count, configured embedder, "
                "any setup hints. Use to verify the server is wired for "
                "ingest before attempting one."
            ),
            input_model=LibraryStatusInput,
            handler=handlers.library_status,
        ),
        ToolSpec(
            name="get_paper",
            description=(
                "Fetch full Paper metadata for an id without ingesting. "
                "Useful as a preview step before deciding whether to "
                "commit to embedding the paper into the local library."
            ),
            input_model=GetPaperInput,
            handler=handlers.get_paper,
        ),
        ToolSpec(
            name="find_paper",
            description=(
                "Find a paper by title (and optional author names) when "
                "you don't have a canonical id. Returns at most three "
                "candidates ranked by title-token similarity with a "
                "confidence score. Use this to bridge from a citation "
                "you've read about to an id you can ingest or cite."
            ),
            input_model=FindPaperInput,
            handler=handlers.find_paper,
        ),
        ToolSpec(
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
            input_model=ExtractClaimsInput,
            handler=handlers.extract_claims,
        ),
        ToolSpec(
            name="find_citations",
            description=(
                "Given a Claim (typically from extract_claims), search "
                "all configured sources, score each candidate by venue + "
                "impact + recency, and return the top-k recommended "
                "citations. Each candidate carries its full quality "
                "breakdown, not just a total — so the user can see WHY "
                "a paper ranked where it did."
            ),
            input_model=FindCitationsInput,
            handler=handlers.find_citations,
        ),
        ToolSpec(
            name="explain_citation",
            description=(
                "Produce a human-readable recommendation for citing a "
                "specific paper as evidence for a specific claim. "
                "Returns a strong/moderate/weak verdict plus the "
                "venue + impact + recency reasoning the user can show "
                "to a co-author or reviewer."
            ),
            input_model=ExplainCitationInput,
            handler=handlers.explain_citation,
        ),
        ToolSpec(
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
            input_model=AnalyzePaperInput,
            handler=handlers.analyze_paper,
        ),
        ToolSpec(
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
            input_model=AssistDraftInput,
            handler=handlers.assist_draft,
        ),
        ToolSpec(
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
            input_model=FindReferencedByInput,
            handler=handlers.find_referenced_by,
        ),
        ToolSpec(
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
            input_model=FindRelatedInput,
            handler=handlers.find_related,
        ),
    ]
    # Guards the other half of the drift this module exists to prevent: a
    # handler field added above with no matching spec would be wired and
    # never advertised. mypy catches a missing handler; only this catches a
    # handler nobody registered.
    declared = {f.name for f in fields(ToolHandlers)}
    registered = {spec.name for spec in specs}
    if declared != registered:
        raise ValueError(
            f"tool table out of sync: handlers-without-specs="
            f"{sorted(declared - registered)}, "
            f"specs-without-handlers={sorted(registered - declared)}"
        )
    return specs
