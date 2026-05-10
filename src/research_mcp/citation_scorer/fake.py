"""Constant-output `CitationScorer` for tests."""

from __future__ import annotations

from research_mcp.domain.citation_scorer import CitationQualityScore
from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper


class FakeCitationScorer:
    """Returns a fixed mid-band score regardless of input.

    Tests that need a `CitationScorer` to plug into a service test but
    don't care about the math grab this. Real scoring is exercised in
    `test_citation_scorer.py`.
    """

    name: str = "fake"

    async def score(
        self,
        paper: Paper,
        claim: Claim | None = None,
    ) -> CitationQualityScore:
        del paper, claim  # constant output regardless of input
        return CitationQualityScore(
            total=50.0,
            venue=12.5,
            impact=12.5,
            author=10.0,
            recency=7.5,
        )
