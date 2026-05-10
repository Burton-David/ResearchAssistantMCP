"""Reference orchestrator: end-to-end assist_draft pipeline in pure Python.

Shows how research-mcp's services COMPOSE outside the MCP layer. The
MCP server is the production-facing surface; this script is what a
downstream agent / SaaS product would write if it wanted to embed the
same pipeline in its own Python code without going through MCP.

Demonstrates:
  * The 9 protocols are usable directly — no MCP wrapper required.
  * The services compose: ClaimExtractor + SearchService + Sources +
    CitationService → DraftService runs the full extract → search →
    score → explain pipeline.
  * Streaming progress works without an MCP client: pass a plain
    async callback and you'll see "claim 3/5 done" updates as the
    pipeline runs.

Run:
    pip install -e ".[dev,claim-extraction]"
    python -m spacy download en_core_web_sm
    export OPENAI_API_KEY=sk-...
    python examples/orchestrator_demo.py

Equivalent agent-SDK wrapper (≈10 lines, omitted to avoid adding a
dependency on the Claude Agent SDK):

    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    async with ClaudeSDKClient(
        options=ClaudeAgentOptions(
            mcp_servers={"research": {"command": "research-mcp", "args": ["serve"]}},
            allowed_tools=["mcp__research__assist_draft"],
        )
    ) as client:
        await client.query(
            "Use assist_draft to review this paragraph for citation gaps: ..."
        )
"""

from __future__ import annotations

import asyncio

from research_mcp.citation_scorer import HeuristicCitationScorer
from research_mcp.claim_extractor import SpacyClaimExtractor
from research_mcp.service.citation import CitationService
from research_mcp.service.draft import DraftService
from research_mcp.service.search import SearchService
from research_mcp.sources import ArxivSource, SemanticScholarSource

DRAFT = (
    "Recent transformer models have outperformed LSTMs by 23% on "
    "machine translation tasks. Self-attention enables parallel "
    "computation across positions in a sequence."
)


async def main() -> None:
    arxiv = ArxivSource()
    s2 = SemanticScholarSource()
    sources = [arxiv, s2]

    extractor = SpacyClaimExtractor()
    citation = CitationService(
        search=SearchService(sources),
        scorer=HeuristicCitationScorer(),
    )
    draft = DraftService(extractor=extractor, citation=citation)

    async def on_progress(done: int, total: int, msg: str) -> None:
        print(f"  [{done}/{total}] {msg}")

    print(f"Reviewing draft ({len(DRAFT)} chars)...")
    try:
        recommendations = await draft.assist(
            DRAFT, k_per_claim=3, progress=on_progress
        )
    finally:
        await arxiv.aclose()
        await s2.aclose()

    print(f"\n{len(recommendations)} claims, recommendations below:\n")
    for rec in recommendations:
        print(f"CLAIM [{rec.claim.type.value}] {rec.claim.text!r}")
        for cand in rec.candidates:
            print(f"  - [{cand.score_total:.0f}/100] {cand.paper.title[:70]}")
            if cand.score_warnings:
                print(f"    warnings: {list(cand.score_warnings)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
