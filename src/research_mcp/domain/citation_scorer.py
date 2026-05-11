"""CitationScorer protocol: paper → quality score with breakdown.

Citation quality is multi-factor and field-dependent. The original
ResearchAssistantAgent baked this into a single 533-line module; this
protocol abstracts it so other strategies can plug in (e.g., an
LLM-based "would this paper be appropriate to cite for this claim"
scorer that looks at semantic fit rather than venue prestige).

The default `HeuristicCitationScorer` (Batch 6) implements the
field-aware scoring from the original:

  * Venue tier (NeurIPS / ICML / Nature / predatory list)
  * Citation impact + velocity (citations per year since publication)
  * Author h-index when available (S2/OpenAlex provide it)
  * Recency, with field-specific decay (CS values fresh, math values
    stable, medicine values clinical-trial-aged-into-evidence)
  * Warnings: self-citation, predatory, retracted, very recent so
    citation velocity is not yet meaningful

The score is a single 0-100 number with the per-factor breakdown and
human-readable factor explanations. Callers display the breakdown so a
researcher can see WHY a paper scored high or low.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper


@dataclass(frozen=True, slots=True)
class CitationQualityScore:
    """Quality breakdown for a candidate citation.

    `total` is the headline score in [0, 100]. `venue`, `impact`,
    `author`, and `recency` are the four constituent dimensions in
    their own [0, X] ranges (the implementation chooses the weights
    that sum to ~100; e.g., venue=25, impact=25, author=20,
    recency=15, with 15 reserved for cross-source consensus or other
    bonuses).

    `factors` is a free-form mapping of factor name → human-readable
    explanation, used by `explain_citation` to show the user why the
    paper scored what it scored.

    `warnings` flags concerns: self-citation, predatory venue,
    retracted, suspiciously few citations for the paper's age.
    """

    total: float
    venue: float
    impact: float
    author: float
    recency: float
    factors: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    warnings: tuple[str, ...] = ()


@runtime_checkable
class CitationScorer(Protocol):
    """A citation quality scorer.

    `claim` is optional: scorers that operate purely on paper metadata
    (venue / author / recency) ignore it; scorers that adapt to claim
    type (a methodological claim wants a methods paper; a comparative
    claim wants a paper that includes both compared methods) use it.
    """

    name: str

    async def score(
        self,
        paper: Paper,
        claim: Claim | None = None,
    ) -> CitationQualityScore:
        """Return a quality score for `paper` as a citation.

        Pure of side effects; no network calls in the scorer itself —
        any data the scorer needs (citation count, h-index) must be
        on the `Paper` already (populated by the source adapter or
        cross-source enrichment).
        """
        ...


