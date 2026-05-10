"""DraftService — the killer-demo orchestrator.

Pipeline:

  1. `ClaimExtractor` turns the user's draft text into a sequence of
     typed claims with confidence and search terms.
  2. For each claim, `CitationService.find_citations` searches the
     configured Sources, scores each candidate, returns top-k.
  3. For each candidate, `CitationService.explain_citation` produces
     a human-readable recommendation.
  4. The whole thing comes back as `CitationRecommendation[]` —
     one per claim, each bundling the claim, the ranked candidates,
     and per-candidate explanations.

The service is small on purpose; the intelligence lives in the
extractor, search, scorer, and explainer. DraftService just wires
the pipeline so a single `assist_draft` MCP call can run the full
flow.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from research_mcp.domain.claim import Claim, ClaimExtractor
from research_mcp.domain.paper import Paper
from research_mcp.service.citation import CitationCandidate, CitationService

# Callback shape: (completed, total, message) → awaitable. Called once
# per claim as recommendations finish (in completion order, not input
# order). `None` means progress reporting is disabled.
ProgressCallback = Callable[[int, int, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CitationRecommendationCandidate:
    """One ranked candidate inside a `CitationRecommendation`.

    Combines `CitationCandidate` (paper + score + sources) with the
    `explain_citation` reasoning string. We keep the explanation
    alongside the candidate so a UI can show the recommendation in
    one block per candidate without a second async round-trip.
    """

    paper: Paper
    score_total: float
    score_warnings: tuple[str, ...]
    sources: tuple[str, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class CitationRecommendation:
    """A claim plus its top-ranked, explained citation candidates."""

    claim: Claim
    candidates: tuple[CitationRecommendationCandidate, ...]


class DraftService:
    def __init__(
        self,
        *,
        extractor: ClaimExtractor,
        citation: CitationService,
    ) -> None:
        self._extractor = extractor
        self._citation = citation

    async def assist(
        self,
        text: str,
        *,
        k_per_claim: int = 3,
        progress: ProgressCallback | None = None,
    ) -> tuple[CitationRecommendation, ...]:
        """Run the full draft → claims → citations → explanations pipeline.

        When `progress` is provided, it's invoked after each claim's
        recommendation completes (in completion order, not document
        order). Total is reported as len(claims) + 1 to count the
        initial claim-extraction step as 0/N+1 → 1/N+1 → … → N+1/N+1.
        Used by the MCP layer to stream progress notifications when
        the client passed a progressToken with the call.
        """
        if not text or not text.strip():
            return ()
        if progress is not None:
            await progress(0, 1, "extracting claims...")
        claims = await self._extractor.extract(text)
        if not claims:
            if progress is not None:
                await progress(1, 1, "no claims found")
            return ()

        total = len(claims) + 1
        if progress is not None:
            await progress(1, total, f"found {len(claims)} claims")

        completed = 1
        completed_lock = asyncio.Lock()

        async def _track(claim: Claim) -> CitationRecommendation:
            rec = await self._recommend(claim, k_per_claim)
            if progress is not None:
                nonlocal completed
                async with completed_lock:
                    completed += 1
                    snapshot = completed
                await progress(snapshot, total, f"claim {snapshot - 1} of {len(claims)} done")
            return rec

        recommendations = await asyncio.gather(*(_track(c) for c in claims))
        return tuple(recommendations)

    async def _recommend(
        self, claim: Claim, k: int
    ) -> CitationRecommendation:
        candidates = await self._citation.find_citations(claim, k=k)
        # Build explanations in parallel; one LLM/heuristic call per
        # candidate. With heuristic scoring this is sub-millisecond per
        # candidate, but `explain_citation` may eventually delegate to
        # an LLM-backed scorer where async parallelism matters.
        explained = await asyncio.gather(
            *(self._build_candidate(claim, c) for c in candidates)
        )
        return CitationRecommendation(claim=claim, candidates=tuple(explained))

    async def _build_candidate(
        self, claim: Claim, candidate: CitationCandidate
    ) -> CitationRecommendationCandidate:
        explanation = await self._citation.explain_citation(candidate.paper, claim)
        return CitationRecommendationCandidate(
            paper=candidate.paper,
            score_total=candidate.score.total,
            score_warnings=candidate.score.warnings,
            sources=candidate.sources,
            explanation=explanation,
        )

    @property
    def extractor(self) -> ClaimExtractor:
        return self._extractor

    @property
    def citation(self) -> CitationService:
        return self._citation
