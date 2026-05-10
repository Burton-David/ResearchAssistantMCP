"""CitationService — find / score / explain citations for a Claim.

Composes a `SearchService` and a `CitationScorer`. Three operations:

  1. `find_citations(claim, k)` — turn a Claim into a query, search the
     configured Sources, score each candidate, return top-k.
  2. `score_citation(paper)` — direct quality breakdown for a paper
     the caller already has.
  3. `explain_citation(paper, claim)` — produce a human-readable
     recommendation including strength, key factors, and warnings.

The service is thin on purpose: real intelligence lives in the
scorer (`HeuristicCitationScorer` today, `LLMCitationScorer` tomorrow)
and in the upstream sources. The service just wires them together,
which is exactly the protocol-composition pattern this project earns
its keep on.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_mcp.domain.citation_scorer import (
    CitationQualityScore,
    CitationScorer,
)
from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.service.search import SearchService

# Strength bands for explain_citation output. The user wants a quick
# read across many candidates; these labels show up in the first
# sentence of the explanation.
_STRONG_THRESHOLD = 65.0
_MODERATE_THRESHOLD = 45.0


@dataclass(frozen=True, slots=True)
class CitationCandidate:
    """A scored candidate citation for a `Claim`.

    `paper` is the candidate; `score` is its `CitationQualityScore` (the
    full breakdown, not just the total — UI / agent should be able to
    show why each candidate ranked where it did). `sources` records the
    upstream adapter(s) that contributed metadata, same shape as
    `SearchResult.sources`.
    """

    paper: Paper
    score: CitationQualityScore
    sources: tuple[str, ...]


class CitationService:
    def __init__(
        self,
        *,
        search: SearchService,
        scorer: CitationScorer,
    ) -> None:
        self._search = search
        self._scorer = scorer

    @property
    def scorer(self) -> CitationScorer:
        """The configured CitationScorer; surfaced so the MCP layer can
        report its name in tool outputs without reaching through `_scorer`."""
        return self._scorer

    async def find_citations(
        self, claim: Claim, *, k: int = 5
    ) -> tuple[CitationCandidate, ...]:
        """Search for candidate citations and return the top-k by score."""
        query_text = _query_from_claim(claim)
        if not query_text:
            return ()

        outcome = await self._search.search(
            SearchQuery(text=query_text, max_results=max(k * 2, 10))
        )
        if not outcome.results:
            return ()

        # Score each candidate under the claim. The scorer's `claim`
        # parameter is documented as optional — heuristic ignores it,
        # but a future LLM scorer reads it.
        scored: list[CitationCandidate] = []
        for result in outcome.results:
            quality = await self._scorer.score(result.paper, claim)
            scored.append(
                CitationCandidate(
                    paper=result.paper,
                    score=quality,
                    sources=result.sources,
                )
            )
        scored.sort(key=lambda c: c.score.total, reverse=True)
        return tuple(scored[:k])

    async def score_citation(
        self, paper: Paper, claim: Claim | None = None
    ) -> CitationQualityScore:
        """Direct scorer call for a paper the caller already has."""
        return await self._scorer.score(paper, claim)

    async def explain_citation(self, paper: Paper, claim: Claim) -> str:
        """Compose a human-readable recommendation with strength + factors."""
        score = await self._scorer.score(paper, claim)
        strength = _strength_label(score.total)
        venue = paper.venue or "an unspecified venue"

        lines: list[str] = []
        lines.append(
            f"{strength} recommendation for citing this paper as a "
            f"{claim.type.value} claim: total score {score.total:.0f}/100."
        )
        lines.append(f"Published in {venue}.")
        for factor_name in ("venue", "impact", "recency"):
            note = score.factors.get(factor_name)
            if note:
                lines.append(f"- {factor_name.title()}: {note}")
        if score.warnings:
            lines.append("Warnings: " + "; ".join(score.warnings) + ".")
        return "\n".join(lines)


def _query_from_claim(claim: Claim) -> str:
    """Build the upstream query text from a claim's signals.

    Prefer the claim's `suggested_search_terms` (typically noun chunks),
    falling back to the claim's surrounding context, falling back to
    the literal claim text. The emptier the claim, the more we lean on
    context to produce a non-trivial query.
    """
    if claim.suggested_search_terms:
        return " ".join(claim.suggested_search_terms)
    if claim.context and claim.context.strip():
        return claim.context.strip()
    return claim.text.strip()


def _strength_label(total: float) -> str:
    if total >= _STRONG_THRESHOLD:
        return "Strong"
    if total >= _MODERATE_THRESHOLD:
        return "Moderate"
    return "Weak"
