"""Command-line entry point for research-mcp.

`research-mcp` (no subcommand) runs the MCP stdio server. Other subcommands
are convenience wrappers around the services so the CLI is useful by itself,
without an MCP client in the loop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import click

from research_mcp.citation import RENDERERS
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.embedder import Embedder
from research_mcp.domain.index import Index
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.source import Source
from research_mcp.embedder import FakeEmbedder, OpenAIEmbedder
from research_mcp.index import FaissIndex, MemoryIndex
from research_mcp.service import LibraryService, SearchService
from research_mcp.sources import ArxivSource, SemanticScholarSource


@dataclass
class _CliLibrary:
    """Bundles the disposables a CLI subcommand uses, so cleanup is one place."""

    arxiv: ArxivSource
    s2: SemanticScholarSource
    library: LibraryService
    index_close: Callable[[], None] | None


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """research-mcp — MCP server and CLI for research workflows."""
    from research_mcp._env import load_dotenv

    load_dotenv()
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@main.command()
def serve() -> None:
    """Run the MCP stdio server (default subcommand)."""
    from research_mcp.mcp.server import main as server_main

    asyncio.run(server_main())


@main.command()
@click.argument("repl_args", nargs=-1, type=click.UNPROCESSED)
def repl(repl_args: tuple[str, ...]) -> None:
    """Drop into IPython with the four protocols and a wired LibraryService."""
    from research_mcp.repl import main as repl_main

    repl_main(list(repl_args))


@main.command()
@click.argument("query")
@click.option("--max", "max_results", type=int, default=20, show_default=True)
@click.option("--source", type=click.Choice(["arxiv", "s2", "all"]), default="all", show_default=True)
def search(query: str, max_results: int, source: str) -> None:
    """Search arXiv and/or Semantic Scholar across configured sources."""
    asyncio.run(_search(query, max_results, source))


async def _search(query: str, max_results: int, source: str) -> None:
    sources: list[Source] = []
    arxiv: ArxivSource | None = None
    s2: SemanticScholarSource | None = None
    if source in {"arxiv", "all"}:
        arxiv = ArxivSource()
        sources.append(arxiv)
    if source in {"s2", "all"}:
        s2 = SemanticScholarSource()
        sources.append(s2)
    try:
        svc = SearchService(sources)
        results = await svc.search(SearchQuery(text=query, max_results=max_results))
        if not results:
            click.echo("No results.", err=True)
            return
        for hit in results:
            paper = hit.paper
            source_label = "+".join(hit.sources)
            click.echo(f"{paper.id}\t[{source_label}]\t{paper.title}")
            snippet = paper.abstract[:200].replace("\n", " ")
            if snippet:
                click.echo(f"    {snippet}{'...' if len(paper.abstract) > 200 else ''}")
    finally:
        if arxiv is not None:
            await arxiv.aclose()
        if s2 is not None:
            await s2.aclose()


@main.command()
@click.argument("paper_id")
def ingest(paper_id: str) -> None:
    """Pull a paper into the local FAISS-backed library."""
    asyncio.run(_ingest(paper_id))


async def _ingest(paper_id: str) -> None:
    cli_lib = _build_library_for_cli()
    try:
        paper = await cli_lib.library.ingest(paper_id)
        click.echo(f"Ingested: {paper.id}\t{paper.title}")
        click.echo(f"Library now has {await cli_lib.library.count()} paper(s).")
    finally:
        await cli_lib.arxiv.aclose()
        await cli_lib.s2.aclose()
        if cli_lib.index_close is not None:
            cli_lib.index_close()


@main.command()
@click.argument("query")
@click.option("--k", type=int, default=10, show_default=True)
def recall(query: str, k: int) -> None:
    """Semantic search over the local library."""
    asyncio.run(_recall(query, k))


async def _recall(query: str, k: int) -> None:
    cli_lib = _build_library_for_cli()
    try:
        results = await cli_lib.library.recall(query, k=k)
        if not results:
            click.echo("Library is empty or no matches.", err=True)
            return
        for paper, score in results:
            click.echo(f"{score:.4f}\t{paper.id}\t{paper.title}")
    finally:
        await cli_lib.arxiv.aclose()
        await cli_lib.s2.aclose()
        if cli_lib.index_close is not None:
            cli_lib.index_close()


@main.command()
@click.argument("paper_id")
@click.option(
    "--format",
    "fmt",
    type=click.Choice([f.value for f in CitationFormat]),
    default=CitationFormat.AMA.value,
    show_default=True,
)
def cite(paper_id: str, fmt: str) -> None:
    """Render a citation for a paper. Fetches metadata from the appropriate source."""
    asyncio.run(_cite(paper_id, fmt))


async def _cite(paper_id: str, fmt: str) -> None:
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    try:
        paper = None
        prefix = paper_id.split(":", 1)[0]
        if prefix == "arxiv":
            paper = await arxiv.fetch(paper_id)
        elif prefix in {"s2", "doi"}:
            paper = await s2.fetch(paper_id)
        if paper is None:
            click.echo(f"Paper not found: {paper_id}", err=True)
            sys.exit(1)
        click.echo(RENDERERS[CitationFormat(fmt)].render(paper))
    finally:
        await arxiv.aclose()
        await s2.aclose()


def _build_library_for_cli() -> _CliLibrary:
    """Wire a LibraryService for CLI use.

    With both `OPENAI_API_KEY` and `RESEARCH_MCP_INDEX_PATH` set, uses real
    OpenAI embeddings + persistent FAISS so `ingest`/`recall` cumulate across
    invocations. Otherwise falls back to FakeEmbedder + MemoryIndex so the
    CLI still works with no API keys (results don't persist between runs).
    """
    arxiv = ArxivSource()
    embedder: Embedder
    index: Index
    index_close: Callable[[], None] | None
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("RESEARCH_MCP_INDEX_PATH"):
        oai = OpenAIEmbedder()
        faiss = FaissIndex.from_env(oai.dimension)
        embedder = oai
        index = faiss
        index_close = faiss.close
    else:
        embedder = FakeEmbedder(64)
        index = MemoryIndex(embedder.dimension)
        index_close = None
    s2 = SemanticScholarSource()
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[arxiv, s2]
    )
    return _CliLibrary(arxiv=arxiv, s2=s2, library=library, index_close=index_close)
