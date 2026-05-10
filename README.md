# research-mcp

An MCP server for research workflows. Search arXiv and Semantic Scholar, build a local FAISS-backed library, and generate citations — from inside Claude Desktop, Claude Code, or any MCP-compatible client.

## Why

LLMs are good at synthesis and bad at sourcing. Off-the-shelf web search returns ranked-for-clicks results, not papers. This server gives the model a clean tool surface for actual literature work: search by query, ingest into a local index, retrieve by semantic similarity, format citations.

## What it does

Seven MCP tools the LLM calls directly:

- **`search_papers`** — arXiv and Semantic Scholar in parallel, with cross-source enrichment (DOI from S2 doesn't get thrown away when arXiv is the first source) and provenance tracking (each result tells you which adapter(s) contributed).
- **`find_paper`** — title-and-author lookup that re-ranks search hits by Jaccard similarity. Use this when you have a citation but no canonical id.
- **`ingest_paper`** — pull a paper's metadata into a local FAISS index keyed by canonical id (`arxiv:`, `doi:`, `s2:`).
- **`library_search`** — semantic recall over your local library, top-k with similarity scores.
- **`cite_paper`** — render any paper as AMA, APA, MLA, Chicago, or BibTeX. Fetches metadata on demand; ingest not required.
- **`get_paper`** — preview full Paper metadata for an id without ingesting.
- **`library_status`** — paper count, configured embedder, setup hints if anything's missing.

## Quick start

```bash
git clone https://github.com/burton-david/research-mcp
cd research-mcp
uv sync                                    # or: pip install -e ".[dev]"

# search, find, cite, get_paper work with no env vars at all:
research-mcp search "transformer scaling laws" --max 10
research-mcp cite arxiv:1706.03762 --format ama

# ingesting and recalling needs an embedder + a place to put the index.
# Pick ONE embedder by setting RESEARCH_MCP_EMBEDDER:
export RESEARCH_MCP_EMBEDDER="openai:text-embedding-3-small"
export OPENAI_API_KEY=sk-...

# ...or run fully offline with sentence-transformers (one-time ~440MB download):
pip install 'research-mcp[sentence-transformers]'
export RESEARCH_MCP_EMBEDDER="sentence-transformers:BAAI/bge-base-en-v1.5"

# Either way, set where the FAISS index lives:
export RESEARCH_MCP_INDEX_PATH=~/research_index
export SEMANTIC_SCHOLAR_API_KEY=...        # optional, raises S2 rate limit

# Optional: enable a cross-encoder reranker for better non-CS-domain
# search quality (chemistry, biology, etc. — arXiv's keyword ranker
# struggles there). Adds 200-1000ms per search/recall; off by default.
export RESEARCH_MCP_RERANKER="cross-encoder:BAAI/bge-reranker-base"

research-mcp serve                         # stdio MCP server
research-mcp repl                          # IPython with abstractions wired
```

If you skip the embedder env var, the server still starts in degraded
mode: `search_papers`, `find_paper`, `cite_paper`, `get_paper`, and
`library_status` work; `ingest_paper` and `library_search` refuse with a
hint pointing at the env vars to set.

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-mcp": {
      "command": "/absolute/path/to/research-mcp/.venv/bin/research-mcp",
      "args": ["serve"]
    }
  }
}
```

Use the absolute path to the `research-mcp` binary — Claude Desktop won't have your venv on `$PATH`. The server auto-loads `.env` from the project root, so you don't need an `env` block in this file as long as `OPENAI_API_KEY` and `RESEARCH_MCP_INDEX_PATH` live there. Prefer that over inlining secrets into the desktop config.

Restart Claude Desktop (⌘Q, not just close the window). Seven tools appear: `search_papers`, `find_paper`, `ingest_paper`, `library_search`, `cite_paper`, `get_paper`, `library_status`.

## Architecture

The codebase is built around five orthogonal protocols in `src/research_mcp/domain/`:

```
Source            where papers come from         (arxiv, semantic_scholar, ...)
Index             where ingested papers live     (faiss, in-memory, pluggable)
Embedder          how text becomes vectors       (openai, sentence-transformers)
CitationRenderer  how a paper becomes a string   (ama, apa, bibtex, mla, chicago)
Reranker          how candidates get rescored    (hf-cross-encoder, optional)
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

## Comparison

**vs. raw arXiv API access.** A direct arXiv call gives you XML you have to parse on every script, no rate-limit handling, no caching, no Semantic Scholar fallback, no local recall. `research-mcp` adds adapters that already speak `Paper`, a process-local rate limiter, a 24-hour disk cache, and protocol-based extension points so adding IEEE / OpenAlex / a local PDF folder is a single new class.

**vs. LangChain's research tools.** LangChain bundles retrieval, prompting, and memory under a deep class hierarchy that you opt into wholesale. `research-mcp` is the retrieval half *only*, exposed as MCP tools — the LLM does the prompting and orchestration in whatever client you prefer (Claude Desktop, Claude Code, Cursor, Continue). Four `typing.Protocol` abstractions instead of LangChain's BaseRetriever / BaseEmbeddings / VectorStore inheritance trees; a third-party `Source` is a single class with no registration step.

**vs. running an LLM agent against the web.** Web search is ranked for clicks. arXiv and Semantic Scholar are ranked for relevance against scientific literature; ingest-then-recall lets the model build a stable, semantically-indexed corpus across sessions instead of re-searching every time.

## Status

Alpha. Interfaces may change before 1.0. PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
