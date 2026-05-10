# research-mcp

A citation-finding research assistant for scientists, exposed as an MCP server. Paste a draft paragraph; it identifies the claims that need citations, finds candidate papers across arXiv, Semantic Scholar, PubMed, and OpenAlex, scores their quality, and explains each recommendation — all from inside Claude Desktop, Claude Code, or any MCP-compatible client.

## The killer demo

```
assist_draft text="""
Recent transformer models have outperformed LSTMs on machine translation
tasks, and self-attention enables parallel computation across positions.
The proposed architecture achieves 28.4 BLEU on WMT 2014 EN-DE.
"""
```

→ extracts three claims (one comparative, one methodological, one statistical), runs each through search across all configured sources, scores candidates by venue + impact + recency, and returns ranked recommendations with the reasoning a researcher could show to a co-author. One MCP call, full pipeline.

## Why

LLMs are good at synthesis and bad at sourcing. Off-the-shelf web search returns ranked-for-clicks results, not papers. A general "search and summarize" agent doesn't know which claims actually need citations or which venues are reputable for the field at hand. `research-mcp` gives the model a clean tool surface for actual literature work: claim extraction, multi-source citation finding, quality scoring, structured paper analysis, semantic recall across a local FAISS-backed library.

## What it does

Fourteen MCP tools the LLM calls directly, organized by stage of the research workflow:

**Citation assistance**
- **`assist_draft`** — the killer demo: draft → claims → ranked, explained citation recommendations.
- **`extract_claims`** — scan draft text and tag claims (statistical, methodological, comparative, causal, theoretical) with confidence + suggested search terms.
- **`find_citations`** — given a claim, return top-k candidate papers with quality scores.
- **`score_citation`** — quality breakdown for a paper: venue tier, citation impact, recency, warnings (predatory venue, retracted).
- **`explain_citation`** — strong/moderate/weak verdict for citing a specific paper as evidence for a specific claim.

**Paper analysis**
- **`analyze_paper`** — LLM-driven structured extraction (summary, contributions, methodology, datasets, metrics, baselines). Backed by OpenAI gpt-4o-mini or Anthropic claude-haiku, selected via env.
- **`chunk_paper`** — section-aware chunking (abstract, introduction, methodology, …) of full text.

**Search and corpus**
- **`search_papers`** — arXiv + Semantic Scholar + PubMed + OpenAlex in parallel, with cross-source enrichment (DOI from S2 doesn't get thrown away when arXiv is the first source) and provenance tracking.
- **`find_paper`** — title-and-author lookup with Jaccard re-ranking, for when you have a citation but no canonical id.
- **`ingest_paper`** / **`bulk_ingest`** — pull metadata into a local FAISS index, one paper at a time or N from a search query.
- **`library_search`** — semantic recall over your local library, top-k with similarity scores.
- **`cite_paper`** — render any paper as AMA, APA, MLA, Chicago, or BibTeX. Fetches metadata on demand; ingest not required.
- **`get_paper`** — preview full Paper metadata for an id without ingesting.
- **`library_status`** — paper count, configured embedder/reranker/extractor/scorer, setup hints if anything's missing.

## Quick start

```bash
git clone https://github.com/Burton-David/ResearchAssistantMCP
cd ResearchAssistantMCP
uv sync                                    # or: pip install -e ".[dev]"

# Citation extraction (spaCy + en_core_web_sm) is required for assist_draft
# and extract_claims. Other tools work without it.
pip install -e ".[claim-extraction]"
python -m spacy download en_core_web_sm

# Search, find, cite, get_paper, score_citation, find_citations all work
# with no env vars at all:
research-mcp search "transformer scaling laws" --max 10
research-mcp cite arxiv:1706.03762 --format ama

# Ingest, recall, bulk_ingest need an embedder + a place to put the index.
# Pick ONE embedder by setting RESEARCH_MCP_EMBEDDER:
export RESEARCH_MCP_EMBEDDER="openai:text-embedding-3-small"
export OPENAI_API_KEY=sk-...

# ...or run fully offline with sentence-transformers (one-time ~440MB download):
pip install -e ".[sentence-transformers]"
export RESEARCH_MCP_EMBEDDER="sentence-transformers:BAAI/bge-base-en-v1.5"

# Either way, set where the FAISS index lives:
export RESEARCH_MCP_INDEX_PATH=~/research_index

# Optional source / quality knobs:
export SEMANTIC_SCHOLAR_API_KEY=...           # raises S2 rate limit
export NCBI_API_KEY=...                        # raises PubMed rate limit (3 → 10/sec)
export RESEARCH_MCP_OPENALEX_EMAIL=you@lab.edu # opt in to OpenAlex
export RESEARCH_MCP_DISABLE_PUBMED=1           # opt OUT of PubMed (default: on)

# Optional: enable a cross-encoder reranker for better non-CS-domain
# search quality. Adds 200-1000ms per search/recall; off by default.
export RESEARCH_MCP_RERANKER="cross-encoder:BAAI/bge-reranker-base"

# Optional: pick the LLM that powers analyze_paper. Otherwise analyze_paper
# refuses with a clear hint.
export RESEARCH_MCP_ANALYSIS_MODEL="openai:gpt-4o-mini"
# or: export RESEARCH_MCP_ANALYSIS_MODEL="anthropic:claude-haiku-4-5-20251001"

research-mcp serve                         # stdio MCP server
research-mcp repl                          # IPython with abstractions wired
```

If you skip the optional knobs, the server still starts in degraded mode: tools that need a missing dependency refuse with a hint pointing at the env var or extra to install.

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

Restart Claude Desktop (⌘Q, not just close the window). The fourteen tools appear in the model's tool list.

## Architecture

Nine orthogonal protocols in `src/research_mcp/domain/`:

```
Source            where papers come from         (arxiv, s2, pubmed, openalex)
Index             where ingested papers live     (faiss, in-memory, pluggable)
Embedder          how text becomes vectors       (openai, sentence-transformers)
CitationRenderer  how a paper becomes a string   (ama, apa, bibtex, mla, chicago)
Reranker          how candidates get rescored    (hf-cross-encoder, optional)
ClaimExtractor    draft text → typed claims      (spacy, fake; LLM extractor planned)
Chunker           paper → section-aware chunks   (section-aware, simple, fake)
CitationScorer    paper → quality breakdown      (heuristic; LLM-based planned)
PaperAnalyzer     paper → structured analysis    (openai, anthropic, fake)
```

Six services compose them: `SearchService`, `LibraryService`, `DiscoveryService`, `CitationService`, `AnalysisService`, `DraftService`. The MCP server wires services into tools but knows nothing about specific implementations — swapping FAISS for Chroma, OpenAI for a local LLM, or spaCy for a transformer-based extractor is a one-line change in the wiring module.

See [docs/architecture.md](docs/architecture.md) and the ADRs at [docs/adr/](docs/adr/).

## Development

```bash
uv sync --dev
pytest                       # full test run
pytest -m unit               # fast, no network
ruff check                   # lint
mypy src                     # type-check (strict)
research-mcp repl            # interactive REPL
```

The project follows REPL-first development: prove a pattern works in `research-mcp repl` before refactoring it into a module. Test with real implementations of protocols we own (no mocks of `Source` / `ClaimExtractor` / `CitationScorer` etc.); only third-party SDKs (httpx, OpenAI, Anthropic) get stubbed in unit tests.

## Comparison

**vs. raw arXiv / PubMed API access.** Direct API calls give you XML or JSON you have to parse on every script, no rate-limit handling, no cross-source enrichment, no claim-aware citation ranking, no local recall. `research-mcp` adds adapters that already speak `Paper`, process-local rate limiters with exponential backoff and Retry-After honoring, a 24-hour disk cache, citation quality scoring, and protocol-based extension points so adding IEEE / a local PDF folder / a custom scorer is a single new class.

**vs. LangChain's research tools.** LangChain bundles retrieval, prompting, and memory under a deep class hierarchy that you opt into wholesale. `research-mcp` is the retrieval and citation-quality half *only*, exposed as MCP tools — the LLM does the prompting and orchestration in whatever client you prefer (Claude Desktop, Claude Code, Cursor, Continue). Nine `typing.Protocol` abstractions instead of LangChain's BaseRetriever / BaseEmbeddings / VectorStore inheritance trees; a third-party `Source` or `CitationScorer` is a single class with no registration step.

**vs. running an LLM agent against the web.** Web search is ranked for clicks. arXiv, Semantic Scholar, PubMed, and OpenAlex are ranked for relevance against scientific literature, with citation counts and venue metadata that drive the quality scorer. The model can't do that math from web snippets.

## Status

Alpha. Interfaces may change before 1.0. PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
