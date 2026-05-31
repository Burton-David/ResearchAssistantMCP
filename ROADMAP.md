# Roadmap

What's on the wish list beyond the current release. None of it is committed — this is the running shape of "what we'd build next if someone wanted to contribute." Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## Infrastructure

- **Token-bucket rate limiting.** `AdaptiveRateLimiter` is minimum-interval-based — no burst tolerance. A token bucket with sliding-window enforcement matches what some upstreams actually enforce. Likely overkill for our throughput; flagged for completeness.

## Out of scope (intentional)

- **Multi-library / `library_id` parameter.** Single library per server keeps the model simple. A user who needs multiple libraries can run multiple MCP server instances against different index paths.
- **MCP Skills (Anthropic).** Skills are workflow templates *on top of* tools. research-mcp *is* the tool layer; shipping our own Skills competes with that pitch. See the README "Comparison" section.
- **MCP Resources for ingested papers.** Resources are for always-load reference data; papers are query-dependent. Wrong abstraction.
