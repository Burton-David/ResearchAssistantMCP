# Roadmap

This is what's deferred from the v0.1 line, ordered by what would deliver the most value to a researcher using the citation-assistant flow. None of it is committed; this is the running shape of "what we'd build next if someone wanted to contribute." All items have corresponding GitHub issues with acceptance criteria — see the issue tracker filtered by [`roadmap`](https://github.com/Burton-David/ResearchAssistantMCP/issues?q=label%3Aroadmap).

Contributions welcome. Issues tagged [`good first issue`](https://github.com/Burton-David/ResearchAssistantMCP/issues?q=label%3A%22good+first+issue%22) are scoped for a single afternoon and don't require deep familiarity with the architecture.

## Sources and coverage

- **PDF full-text ingestion.** The `Chunker` protocol is already wired but only sees title + abstract today. A future batch fetches PDFs (arXiv direct, OpenAlex open-access, S2 `openAccessPdf`), extracts via pyMuPDF / pdfplumber, feeds `Chunker`. Unlocks fine-grained recall and meaningful `analyze_paper.methodology`/`limitations` extraction. Big feature.

## Infrastructure

- **Cross-process rate-limit coordination.** Today each process has its own `RateLimiter` / `AdaptiveRateLimiter`. Two research-mcp processes against the same API key can collectively exceed quota. File-lock or shared-state (sqlite, redis) would coordinate. Matters for power users running REPL + Claude Desktop simultaneously.
- **Token-bucket rate limiting.** `AdaptiveRateLimiter` is minimum-interval-based — no burst tolerance. A token bucket with sliding-window enforcement matches what some upstreams actually enforce. Likely overkill for our throughput; flagged for completeness.

## Code organization

These are refactors, not features. They're small and ship-friendly for a first PR.

- **Extract shared `service/_merge.py`.** `service/library.py` reaches into `service/search.py:_merge_records` for cross-source merging. Service-shared infra should be its own module, not a cross-service private symbol.

## Out of scope (intentional)

- **Multi-library / `library_id` parameter.** Single library per server keeps the model simple. A user who needs multiple libraries can run multiple MCP server instances against different index paths.
- **MCP Skills (Anthropic).** Skills are workflow templates *on top of* tools. research-mcp *is* the tool layer; shipping our own Skills competes with that pitch. See the README "Comparison" section.
- **MCP Resources for ingested papers.** Resources are for always-load reference data; papers are query-dependent. Wrong abstraction.
