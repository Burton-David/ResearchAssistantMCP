"""MCP stdio server.

Wires sources, embedder, index, and the citation registry into the four
tool handlers. Default wiring uses real arXiv + Semantic Scholar + OpenAI
embeddings + FAISS on disk; a `build_test_server` helper swaps in
FakeEmbedder + MemoryIndex for the e2e harness.
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
from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.embedder import FakeEmbedder, OpenAIEmbedder
from research_mcp.errors import SourceUnavailable
from research_mcp.index import FaissIndex, MemoryIndex
from research_mcp.mcp.tools import (
    CitePaperInput,
    CitePaperOutput,
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
from research_mcp.service import LibraryService, SearchService
from research_mcp.sources import ArxivSource, SemanticScholarSource

_log = logging.getLogger(__name__)


def build_server(
    *,
    search: SearchService,
    library: LibraryService,
    paper_lookup: Callable[[str], Awaitable[Paper | None]],
) -> Server[Any, Any]:
    """Construct an MCP `Server` with the four research tools registered."""
    server: Server[Any, Any] = Server("research-mcp", version=__version__)

    # mcp SDK ships its decorators as untyped at the moment.
    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="search_papers",
                description=(
                    "Search arXiv and Semantic Scholar in parallel and return "
                    "deduplicated metadata for each paper."
                ),
                inputSchema=SearchPapersInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="ingest_paper",
                description=(
                    "Fetch a paper by canonical id and add it to the local "
                    "FAISS-backed library so it can be recalled by similarity."
                ),
                inputSchema=IngestPaperInput.model_json_schema(),
            ),
            mcp_types.Tool(
                name="library_search",
                description=(
                    "Semantic search across the local library; returns the top-k "
                    "ingested papers with similarity scores."
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
                    "Report the number of papers currently in the local "
                    "library. Useful for verifying ingest state without "
                    "needing to ingest another paper."
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
        args = IngestPaperInput.model_validate(arguments)
        paper = await library.ingest(args.paper_id)
        return IngestPaperOutput(
            paper=paper_to_summary(paper, source=source_from_id(paper.id)),
            library_count=await library.count(),
        ).model_dump()

    async def _do_recall(arguments: dict[str, Any]) -> dict[str, Any]:
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
        return LibraryStatusOutput(count=await library.count()).model_dump()

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
    """Production wiring: real APIs, OpenAI embeddings, FAISS on disk."""
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    embedder = OpenAIEmbedder()
    index_path = os.environ.get("RESEARCH_MCP_INDEX_PATH")
    if not index_path:
        raise RuntimeError(
            "RESEARCH_MCP_INDEX_PATH is required when running the MCP server. "
            "Set it to a writable directory; FAISS files will live there."
        )
    index = FaissIndex(index_path, dimension=embedder.dimension)
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[arxiv, s2]
    )
    search = SearchService([arxiv, s2])

    server = build_server(search=search, library=library, paper_lookup=library.fetch)
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
        index.close()


async def run_in_memory() -> None:
    """In-memory wiring used by the e2e harness — no API keys required.

    Selected when `RESEARCH_MCP_TEST_MODE=1` so the e2e test can boot a real
    server subprocess without needing OpenAI or a writable index path.
    """
    arxiv = ArxivSource()
    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    library = LibraryService(index=index, embedder=embedder, ingest_sources=[arxiv])
    search = SearchService([arxiv])

    server = build_server(search=search, library=library, paper_lookup=library.fetch)
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
