"""Drop into IPython with the four protocols, the dataclasses, and a wired
LibraryService already imported and bound. Default wiring uses MemoryIndex +
FakeEmbedder so the REPL works with no API keys.

Convenience helpers exposed at top level:

    await q("attention is all you need")     # arXiv search → list[Paper]
    await library.ingest("arxiv:1706.03762") # ingest into MemoryIndex
    await library.recall("transformers")     # search the local library
    cite(paper, "ama")                       # render any format
"""

from __future__ import annotations

import asyncio
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
from research_mcp.service import LibraryService, SearchService
from research_mcp.sources import ArxivSource, SemanticScholarSource

_BANNER = """\
research-mcp REPL
─────────────────
  arxiv      : ArxivSource()
  s2         : SemanticScholarSource()
  embedder   : FakeEmbedder(64)        # deterministic, no API keys
  index      : MemoryIndex(64)
  library    : LibraryService(index, embedder, [arxiv, s2])
  search     : SearchService([arxiv, s2])

  await q("attention is all you need")     → search arXiv + S2
  await library.ingest("arxiv:1706.03762") → add to local index (also doi:/s2:)
  await library.recall("transformers")     → search local library
  cite(paper, "ama")                       → render any format

  Pass --openai or set OPENAI_API_KEY for the real OpenAI embedder.
"""


def build_namespace(*, use_openai: bool = False) -> dict[str, Any]:
    """Construct the REPL namespace. Tested independently of IPython."""
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    embedder: Embedder
    if use_openai:
        from research_mcp.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
    else:
        embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    library = LibraryService(index=index, embedder=embedder, ingest_sources=[arxiv, s2])
    search = SearchService([arxiv, s2])

    async def q(text: str, max_results: int = 10) -> list[Paper]:
        results = await search.search(SearchQuery(text=text, max_results=max_results))
        return [r.paper for r in results]

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
        "index": index,
        "library": library,
        "search": search,
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
    use_openai = "--openai" in args
    if use_openai:
        args.remove("--openai")
    namespace = build_namespace(use_openai=use_openai)
    config = Config()
    config.InteractiveShellApp.exec_lines = ["%autoawait asyncio"]
    config.TerminalInteractiveShell.banner2 = _BANNER
    # IPython's public entry point lacks complete type stubs.
    start_ipython(argv=args, user_ns=namespace, config=config)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
