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

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from research_mcp import __version__
from research_mcp.citation import RENDERERS
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.embedder import Embedder
from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.source import Source
from research_mcp.embedder import (
    FakeEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
)
from research_mcp.errors import SourceUnavailable
from research_mcp.index import FaissIndex, MemoryIndex
from research_mcp.mcp.tools import (
    CitePaperInput,
    CitePaperOutput,
    FindPaperHit,
    FindPaperInput,
    FindPaperOutput,
    GetPaperInput,
    GetPaperOutput,
    IngestPaperInput,
    IngestPaperOutput,
    LibrarySearchHit,
    LibrarySearchInput,
    LibrarySearchOutput,
    LibraryStatusInput,
    LibraryStatusOutput,
    SearchPapersInput,
    SearchPapersOutput,
    paper_to_summary,
    source_from_id,
)
from research_mcp.service import DiscoveryService, LibraryService, SearchService
from research_mcp.service.library import fetch_from_sources
from research_mcp.sources import ArxivSource, SemanticScholarSource

_log = logging.getLogger(__name__)

_NO_EMBEDDER_HINT = (
    "no embedder is configured. Set RESEARCH_MCP_EMBEDDER to "
    "'openai:text-embedding-3-small' (requires OPENAI_API_KEY) or "
    "'sentence-transformers:BAAI/bge-base-en-v1.5' (requires "
    "`pip install research-mcp[sentence-transformers]`)."
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


def build_server(
    *,
    search: SearchService,
    discovery: DiscoveryService,
    paper_lookup: Callable[[str], Awaitable[Paper | None]],
    library: LibraryService | None,
    embedder_label: str | None,
) -> Server[Any, Any]:
    """Construct an MCP `Server` with the six research tools registered.

    `library` may be None if no embedder is configured. In that mode the
    server still serves search/cite/get_paper/library_status; only
    ingest_paper and library_search refuse.
    """
    server: Server[Any, Any] = Server("research-mcp", version=__version__)

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
                    "Fetch a paper by canonical id and add it to the local "
                    "FAISS-backed library so it can be recalled by similarity. "
                    "Requires an embedder; see library_status if unsure whether "
                    "the server is configured for ingest."
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
        ]

    async def _do_search(arguments: dict[str, Any]) -> dict[str, Any]:
        args = SearchPapersInput.model_validate(arguments)
        results = await search.search(
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
                for r in results
            ]
        ).model_dump()

    async def _do_ingest(arguments: dict[str, Any]) -> dict[str, Any]:
        if library is None:
            raise ValueError(f"ingest_paper unavailable: {_NO_EMBEDDER_HINT}")
        args = IngestPaperInput.model_validate(arguments)
        paper = await library.ingest(args.paper_id)
        return IngestPaperOutput(
            paper=paper_to_summary(paper, source=source_from_id(paper.id)),
            library_count=await library.count(),
        ).model_dump()

    async def _do_recall(arguments: dict[str, Any]) -> dict[str, Any]:
        if library is None:
            raise ValueError(f"library_search unavailable: {_NO_EMBEDDER_HINT}")
        args = LibrarySearchInput.model_validate(arguments)
        results = await library.recall(args.query, k=args.k)
        return LibrarySearchOutput(
            results=[
                LibrarySearchHit(
                    paper=paper_to_summary(p, source=source_from_id(p.id)),
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
                f"could not resolve {args.paper_id!r}: source {exc.source_name!r} "
                f"is unavailable ({exc.reason}). This is usually transient — try again."
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
        if library is None:
            return LibraryStatusOutput(
                count=0,
                embedder=None,
                note=_NO_EMBEDDER_HINT,
            ).model_dump()
        return LibraryStatusOutput(
            count=await library.count(),
            embedder=embedder_label,
            note=None,
        ).model_dump()

    async def _do_find_paper(arguments: dict[str, Any]) -> dict[str, Any]:
        args = FindPaperInput.model_validate(arguments)
        hits = await discovery.find_paper(
            title=args.title, authors=tuple(args.authors)
        )
        return FindPaperOutput(
            results=[
                FindPaperHit(
                    paper=paper_to_summary(h.paper, source="+".join(h.sources)),
                    confidence=h.confidence,
                )
                for h in hits
            ]
        ).model_dump()

    async def _do_get_paper(arguments: dict[str, Any]) -> dict[str, Any]:
        args = GetPaperInput.model_validate(arguments)
        try:
            paper = await paper_lookup(args.paper_id)
        except SourceUnavailable as exc:
            raise ValueError(
                f"could not resolve {args.paper_id!r}: source {exc.source_name!r} "
                f"is unavailable ({exc.reason}). This is usually transient — try again."
            ) from exc
        if paper is None:
            raise ValueError(
                f"no configured source recognizes paper id {args.paper_id!r}. "
                "Use a prefixed id like 'arxiv:1706.03762', 'doi:10.1038/...', "
                "or 's2:abc123'."
            )
        return GetPaperOutput(
            paper=paper_to_summary(paper, source=source_from_id(paper.id))
        ).model_dump()

    handlers: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
        "search_papers": _do_search,
        "ingest_paper": _do_ingest,
        "library_search": _do_recall,
        "cite_paper": _do_cite,
        "library_status": _do_status,
        "get_paper": _do_get_paper,
        "find_paper": _do_find_paper,
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
        return await handler(arguments)

    return server


async def run_default() -> None:
    """Production wiring: real arXiv + S2; embedder selected from env.

    With no embedder configured, the server still boots in degraded mode
    (search/cite/get_paper work; ingest/recall refuse with a clear hint).
    """
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    sources: tuple[Source, ...] = (arxiv, s2)
    search = SearchService(sources)
    discovery = DiscoveryService(search)

    embedder, label = _select_embedder()
    library: LibraryService | None = None
    index_to_close: FaissIndex | None = None
    if embedder is not None:
        index_path = os.environ.get("RESEARCH_MCP_INDEX_PATH")
        if not index_path:
            raise RuntimeError(
                "RESEARCH_MCP_INDEX_PATH is required when an embedder is "
                "configured. Set it to a writable directory; FAISS files "
                "will live there."
            )
        index = FaissIndex(index_path, dimension=embedder.dimension)
        index_to_close = index
        library = LibraryService(
            index=index, embedder=embedder, ingest_sources=sources
        )
    else:
        _log.warning("no embedder configured: %s", _NO_EMBEDDER_HINT)

    async def paper_lookup(paper_id: str) -> Paper | None:
        return await fetch_from_sources(sources, paper_id)

    server = build_server(
        search=search,
        discovery=discovery,
        paper_lookup=paper_lookup,
        library=library,
        embedder_label=label,
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

    async def paper_lookup(paper_id: str) -> Paper | None:
        return await fetch_from_sources(sources, paper_id)

    server = build_server(
        search=search,
        discovery=discovery,
        paper_lookup=paper_lookup,
        library=library,
        embedder_label="fake:test-mode",
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
    if os.environ.get("RESEARCH_MCP_TEST_MODE") == "1":
        await run_in_memory()
    else:
        await run_default()
