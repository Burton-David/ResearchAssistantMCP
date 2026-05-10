# Architecture

`research-mcp` is built around five orthogonal protocols. All behavior is composed from these — there is no inheritance hierarchy, no plugin registry, no framework code beyond the protocols themselves.

## The five protocols

All five live in `src/research_mcp/domain/`. They are `typing.Protocol` declarations, not abstract base classes — see [ADR-0001](adr/0001-protocol-based-abstractions.md) for why.

### Source

```python
async def search(self, query: SearchQuery) -> Sequence[Paper]: ...
async def fetch(self, paper_id: str) -> Paper | None: ...
```

Anything that returns `Paper`s given a `SearchQuery` is a Source. Adapters: `arxiv`, `semantic_scholar`. Adding a new database (IEEE, ACM, OpenAlex) means writing one Source implementation; the rest of the system requires no changes.

### Index

```python
async def upsert(papers, embeddings) -> None: ...
async def search(embedding, k) -> Sequence[tuple[Paper, float]]: ...
async def delete(paper_ids) -> None: ...
async def count() -> int: ...
```

A vector store for ingested papers. Implementations: `FaissIndex` (default, on-disk), `MemoryIndex` (tests, no FAISS dependency).

### Embedder

```python
dimension: int
async def embed(texts) -> Sequence[Sequence[float]]: ...
```

Converts text to fixed-dimension vectors. Implementations: `OpenAIEmbedder` (default), optionally `SentenceTransformersEmbedder` for local inference.

### CitationRenderer

```python
format: CitationFormat
def render(paper) -> str: ...
```

Turns a `Paper` into a formatted citation string. One renderer per supported format.

### Reranker

```python
async def score(query: str, papers: Sequence[Paper]) -> Sequence[float]: ...
```

Optional. Rescores `(query, paper)` pairs for relevance, sitting one layer above bi-encoder retrieval. The motivating use case is non-CS-domain queries where keyword-driven upstream rankers (arXiv) put share-a-word papers atop the list. Implementations: `HuggingFaceCrossEncoderReranker` (wraps `sentence-transformers` CrossEncoder; default `BAAI/bge-reranker-base`); `FakeReranker` for tests. Selected via `RESEARCH_MCP_RERANKER=cross-encoder:<model>`; off by default because cross-encoder inference adds 200-1000ms per call.

## Composition

Two services compose the protocols:

- **`SearchService`** — wraps a list of `Source`s + an optional `Reranker`. Dispatches a query to all sources in parallel, merges results, dedups by canonical id, optionally reranks the merged pool, then truncates to the user's `max_results`.
- **`LibraryService`** — wraps `Index` + `Embedder` + ingest `Source`s + an optional `Reranker`. Methods: `ingest(paper_id)`, `recall(query, k)`, `delete(paper_id)`, `count()`. With a reranker, `recall` pulls a wider candidate set from the index, rescores, and returns top-k by reranker score.
- **`DiscoveryService`** — wraps `SearchService`. `find_paper(title, authors)` for title-and-author lookup with Jaccard re-ranking.

The MCP server in `src/research_mcp/mcp/` defines tool schemas (`search_papers`, `find_paper`, `ingest_paper`, `library_search`, `cite_paper`, `get_paper`, `library_status`) and wires services to tool handlers. The server depends on the services; the services depend on the protocols. Concrete adapters (`arxiv`, `faiss_index`, `openai_embedder`, `ama_renderer`, `hf_cross_encoder`) are injected at the wiring layer in `cli.py` and `mcp/server.py`.

## Why async

I/O bound everywhere — HTTP to arXiv and Semantic Scholar, calls to the OpenAI embeddings API, optional disk I/O on FAISS. `asyncio` is the lowest-friction concurrency model. No CPU-bound work lives in the abstractions. Both the CLI and the MCP server already run an event loop, so there is no sync/async impedance.

## Test pyramid

- **Unit tests** (`tests/unit/`) use `MemoryIndex` and a deterministic fake `Embedder`. No network. Sub-second per file. Aim for high coverage on `domain/` and `service/`.
- **Integration tests** (`tests/integration/`) hit real arXiv and Semantic Scholar. Gated by `RESEARCH_MCP_INTEGRATION=1`. Expect rate-limit retries.
- **End-to-end** (`tests/e2e/`) boots the MCP server in stdio mode as a subprocess, sends a tool call, asserts response shape.

## What we deliberately do not have

- **A plugin registry.** Adapters are composed by the wiring layer, not registered by name. If you want to add a new Source, write the class and pass it to `SearchService`.
- **A workflow engine.** The services are thin. Anything more complex is the LLM's job — that is the entire point of MCP.
- **A "Manager" or "Orchestrator" class.** Composition is just calling functions on services. There is no orchestrator god object.
- **Sync wrappers.** Everything is async end-to-end. If you need to call from sync code, use `asyncio.run`.
