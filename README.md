# research-mcp

An MCP server for research workflows. Search arXiv and Semantic Scholar, build a local FAISS-backed library, and generate citations — from inside Claude Desktop, Claude Code, or any MCP-compatible client.

## Why

LLMs are good at synthesis and bad at sourcing. Off-the-shelf web search returns ranked-for-clicks results, not papers. This server gives the model a clean tool surface for actual literature work: search by query, ingest into a local index, retrieve by semantic similarity, format citations.

## What it does

- **Search** — arXiv and Semantic Scholar, with optional year and author filters.
- **Ingest** — pull a paper's metadata (and full text where available) into a local FAISS index keyed by canonical IDs.
- **Recall** — semantic search over your local library, returning top-k papers with similarity scores.
- **Cite** — render any paper as AMA, APA, MLA, Chicago, or BibTeX.

## Quick start

```bash
git clone https://github.com/burton-david/research-mcp
cd research-mcp
uv sync                                    # or: pip install -e ".[dev]"

export OPENAI_API_KEY=sk-...
export RESEARCH_MCP_INDEX_PATH=~/research_index
export SEMANTIC_SCHOLAR_API_KEY=...        # optional, for higher rate limits

research-mcp serve                         # stdio MCP server
research-mcp repl                          # IPython with abstractions wired
research-mcp search "transformer scaling laws" --max 10
```

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research": {
      "command": "research-mcp",
      "args": ["serve"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "RESEARCH_MCP_INDEX_PATH": "/Users/you/research_index"
      }
    }
  }
}
```

Restart Claude Desktop. New tools appear: `search_papers`, `ingest_paper`, `library_search`, `cite_paper`.

## Architecture

The codebase is built around four orthogonal protocols in `src/research_mcp/domain/`:

```
Source            where papers come from         (arxiv, semantic_scholar, ...)
Index             where ingested papers live     (faiss, in-memory, pluggable)
Embedder          how text becomes vectors       (openai, sentence-transformers)
CitationRenderer  how a paper becomes a string   (ama, apa, bibtex, mla, chicago)
```

Implementations live in sibling packages. The MCP server composes them but knows nothing about specific implementations. Swapping FAISS for Chroma is a one-line change in the wiring module.

See [docs/architecture.md](docs/architecture.md) and [docs/adr/0001-protocol-based-abstractions.md](docs/adr/0001-protocol-based-abstractions.md).

## Development

```bash
uv sync --dev
pytest                       # full test run
pytest -m unit               # fast, no network
ruff check                   # lint
mypy src                     # type-check (strict)
research-mcp repl            # interactive REPL
```

The project follows REPL-first development: prove a pattern works in `research-mcp repl` before refactoring it into a module. The four core protocols are stable; new abstractions get added only when usage demands them.

## Status

Alpha. Interfaces may change before 1.0. PRs welcome.

## License

MIT — see [LICENSE](LICENSE).
