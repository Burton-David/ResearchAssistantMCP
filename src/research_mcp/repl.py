"""Drop into IPython with the four protocols, the dataclasses, and a wired
LibraryService already imported and bound.

Embedder selection follows the same precedence as the MCP server:

  1. RESEARCH_MCP_EMBEDDER=openai:<model> or
     RESEARCH_MCP_EMBEDDER=sentence-transformers:<model> (explicit pick)
  2. OPENAI_API_KEY set → openai:text-embedding-3-small
  3. Otherwise → FakeEmbedder(64) so the REPL still loads with no config

Pass `--fake` to force the FakeEmbedder regardless of env (useful for
offline experimentation without burning API calls).

Convenience helpers exposed at top level:

    await q("attention is all you need")     # arXiv + S2 search → list[Paper]
    await library.ingest("arxiv:1706.03762") # ingest into the wired index
    await library.recall("transformers")     # search the local library
    cite(paper, "ama")                       # render any format
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from research_mcp.citation import RENDERERS
from research_mcp.domain import (
    Author,
    CitationFormat,
    CitationRenderer,
    Embedder,
    Index,
    Paper,
    SearchQuery,
    Source,
)
from research_mcp.embedder import FakeEmbedder
from research_mcp.index import MemoryIndex
from research_mcp.service import DiscoveryService, LibraryService, SearchService
from research_mcp.sources import ArxivSource, SemanticScholarSource

_log = logging.getLogger(__name__)

_BANNER_TEMPLATE = """\
research-mcp REPL
─────────────────
  arxiv      : ArxivSource()
  s2         : SemanticScholarSource()
  embedder   : {embedder_label}
  index      : MemoryIndex({dim})
  library    : LibraryService(index, embedder, [arxiv, s2])
  search     : SearchService([arxiv, s2])
  discovery  : DiscoveryService(search)

  await q("attention is all you need")          → search arXiv + S2
  await library.ingest("arxiv:1706.03762")      → add to local index
  await library.recall("transformers")          → search local library
  await discovery.find_paper("Attention Is...") → title-based lookup
  cite(paper, "ama")                            → render any format

  RESEARCH_MCP_EMBEDDER={embedder_env_hint}
  Pass --fake to force FakeEmbedder regardless of env.
"""


def build_namespace(*, force_fake: bool = False) -> dict[str, Any]:
    """Construct the REPL namespace. Tested independently of IPython."""
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    embedder: Embedder
    embedder_label: str
    if force_fake:
        embedder = FakeEmbedder(64)
        embedder_label = "FakeEmbedder(64) — forced via --fake"
    else:
        from research_mcp.mcp.server import _select_embedder

        selected, label = _select_embedder()
        if selected is not None and label is not None:
            embedder = selected
            embedder_label = label
        else:
            embedder = FakeEmbedder(64)
            embedder_label = (
                "FakeEmbedder(64) — set RESEARCH_MCP_EMBEDDER for a real one"
            )
    index = MemoryIndex(embedder.dimension)
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[arxiv, s2]
    )
    search = SearchService([arxiv, s2])
    discovery = DiscoveryService(search)

    async def q(text: str, max_results: int = 10) -> list[Paper]:
        outcome = await search.search(
            SearchQuery(text=text, max_results=max_results)
        )
        if outcome.partial_failures:
            print(f"⚠ partial failures: {outcome.partial_failures}")
        return [r.paper for r in outcome.results]

    def cite(paper: Paper, fmt: str = "ama") -> str:
        return RENDERERS[CitationFormat(fmt)].render(paper)

    return {
        "Author": Author,
        "CitationFormat": CitationFormat,
        "CitationRenderer": CitationRenderer,
        "Embedder": Embedder,
        "Index": Index,
        "Paper": Paper,
        "SearchQuery": SearchQuery,
        "Source": Source,
        "arxiv": arxiv,
        "s2": s2,
        "embedder": embedder,
        "embedder_label": embedder_label,
        "index": index,
        "library": library,
        "search": search,
        "discovery": discovery,
        "q": q,
        "cite": cite,
        "asyncio": asyncio,
    }


def main(argv: list[str] | None = None) -> None:
    try:
        from IPython import start_ipython
        from traitlets.config import Config
    except ImportError as exc:
        raise SystemExit(
            "research-mcp repl requires IPython. Install dev deps:\n"
            "    pip install 'research-mcp[dev]'\n"
            "or just `pip install ipython`."
        ) from exc

    args = list(argv if argv is not None else sys.argv[1:])
    force_fake = "--fake" in args
    if force_fake:
        args.remove("--fake")
    # Legacy alias — the prior `--openai` flag now means "do not override env"
    # since the env var is the canonical selector. Drop it silently for back-
    # compat with any muscle memory.
    while "--openai" in args:
        args.remove("--openai")

    namespace = build_namespace(force_fake=force_fake)
    banner = _BANNER_TEMPLATE.format(
        embedder_label=namespace["embedder_label"],
        dim=namespace["embedder"].dimension,
        embedder_env_hint="(unset → FakeEmbedder)"
        if "FakeEmbedder" in namespace["embedder_label"]
        else "(set)",
    )

    config = Config()
    config.InteractiveShellApp.exec_lines = ["%autoawait asyncio"]
    config.TerminalInteractiveShell.banner2 = banner
    # IPython's public entry point lacks complete type stubs.
    start_ipython(argv=args, user_ns=namespace, config=config)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
