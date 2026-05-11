"""Field-aware `CitationScorer` — overrides recency + impact per detected field.

Composes on top of a base scorer (defaulting to `HeuristicCitationScorer`)
in the same wrap-and-augment pattern as the LLM scorers:

    FieldAwareCitationScorer(HeuristicCitationScorer())   # the default
    OpenAILLMCitationScorer(
        base_scorer=FieldAwareCitationScorer(HeuristicCitationScorer())
    )  # inner=field, outer=LLM (see plan: relevance multiplier on a
       # field-adjusted base score)

What it changes vs base:
  * `recency` uses a per-field half-life (CS 4y, medicine 8y, math 15y,
    physics 6y, default 5y).
  * `impact` uses a per-field citation-velocity baseline (CS 5, medicine
    10, math 2, physics 8, default 5).
  * `factors["field"]` reports the detected field for telemetry.

What it leaves untouched:
  * `venue` (heuristic's venue lists already cover top-tier venues
    across fields — NEJM, Annals of Math, PRL, NeurIPS are all in
    `_venues.py`).
  * `author` (placeholder, becomes field-aware in #1).
  * `warnings` (heuristic's "very recent" / "predatory venue" /
    "missing metadata" warnings all still fire correctly).

When the detector returns `Field.DEFAULT`, this scorer returns the base
score unchanged — no factor dict mutation, no rounding-error in `total`.
That preserves regression behavior for callers that explicitly want the
old scoring path via `RESEARCH_MCP_CITATION_SCORER=heuristic`.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final

from research_mcp.citation_scorer._field import Field, detect_field
from research_mcp.citation_scorer._field_params import (
    EXPECTED_CITATION_VELOCITY,
    RECENCY_HALF_LIFE_YEARS,
)
from research_mcp.citation_scorer.heuristic import HeuristicCitationScorer
from research_mcp.domain.citation_scorer import (
    CitationQualityScore,
    CitationScorer,
)
from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper

# Constants mirrored from heuristic.py — these define what "100%" means
# for each dimension. Keeping them here avoids importing from a private-
# style module name (`heuristic._RECENCY_MAX`); if the heuristic's max
# changes, both modules need updating, but that's a one-line edit.
_RECENCY_MAX: Final = 25.0
_IMPACT_MAX: Final = 30.0
# Below this many days, citation count is too noisy to score on.
# Mirrors heuristic.py's threshold so the "fresh but unproven" penalty
# kicks in at the same age across both scorers.
_VERY_RECENT_DAYS: Final = 180


class FieldAwareCitationScorer:
    """Wraps a base CitationScorer; rescales recency + impact per field."""

    name: str = "field_aware"

    def __init__(
        self,
        base: CitationScorer | None = None,
        *,
        now: _dt.date | None = None,
    ) -> None:
        self._base = base or HeuristicCitationScorer()
        # `now` is overridable for deterministic tests; in production,
        # `_dt.date.today()` is called once at construction and reused
        # for the scorer's lifetime — fine for short-lived MCP handler
        # calls, and we don't want each `.score()` call to re-read the
        # clock and risk same-paper drift across a single request.
        self._now = now or _dt.date.today()

    async def score(
        self,
        paper: Paper,
        claim: Claim | None = None,
    ) -> CitationQualityScore:
        base = await self._base.score(paper, claim)
        field = detect_field(paper)
        if field is Field.DEFAULT:
            # No field hint detected — return the base score verbatim.
            # This is the regression-safety path: papers without
            # `arxiv_primary_category` or `openalex_field` metadata
            # score identically to the pre-field-aware behavior.
            return base
        new_recency, recency_factor = _score_recency_field_aware(
            paper, field, self._now
        )
        new_impact, impact_factor = _score_impact_field_aware(
            paper, field, self._now
        )
        # `total` must equal `venue + impact + author + recency` per the
        # heuristic's reconciliation invariant (chaos-test 67.5≠60 bug).
        # Venue + author come straight from base — we only redistribute
        # the two field-sensitive dimensions.
        new_total = base.venue + new_impact + base.author + new_recency
        new_factors = dict(base.factors)
        new_factors["recency"] = recency_factor
        new_factors["impact"] = impact_factor
        new_factors["field"] = f"detected as {field.value}"
        return CitationQualityScore(
            total=round(new_total, 2),
            venue=base.venue,
            impact=round(new_impact, 2),
            author=base.author,
            recency=round(new_recency, 2),
            factors=new_factors,
            # Base warnings already cover the same edge cases (retracted,
            # predatory, very-recent, missing-metadata); the field-aware
            # recompute uses different half-lives but the SAME thresholds,
            # so emitting new warnings would duplicate the base's.
            warnings=base.warnings,
        )


def _score_recency_field_aware(
    paper: Paper, field: Field, now: _dt.date,
) -> tuple[float, str]:
    """Per-field linear-decay recency score.

    Mirrors the structure of `heuristic._score_recency` but with a
    field-specific half-life. The "fresh but unproven" penalty (drop
    to 30% when citation_count is None/0 AND paper is under the
    half-life) still applies — that check protects against new papers
    free-riding to the top of recommendations regardless of field.
    """
    if not paper.published:
        return 0.0, "Publication date unknown."
    half_life = RECENCY_HALF_LIFE_YEARS[field]
    years = max((now - paper.published).days / 365.25, 0.0)
    factor = max(0.0, 1.0 - years / half_life)
    score = _RECENCY_MAX * factor

    impact_unknown = paper.citation_count is None or paper.citation_count == 0
    if impact_unknown and years < half_life:
        penalty = 0.3
        score *= penalty
        return (
            score,
            f"Published {years:.1f} years ago; recency contribution "
            f"capped at {penalty * 100:.0f}% because citation impact "
            f"is unverified (field={field.value}, "
            f"half_life={half_life:.0f}y).",
        )
    return (
        score,
        f"Published {years:.1f} years ago "
        f"(field={field.value}, half_life={half_life:.0f}y).",
    )


def _score_impact_field_aware(
    paper: Paper, field: Field, now: _dt.date,
) -> tuple[float, str]:
    """Per-field citation-velocity impact score.

    Same shape as `heuristic._score_impact`: citations/year divided by
    the field's expected velocity, capped at 3x. The cap matters for
    math papers where a single high-impact result might run at 50x the
    field's baseline velocity — without the cap one paper would zero
    out the dimension's resolution for everyone else.
    """
    if paper.citation_count is None:
        # Same baseline as heuristic — don't zero arXiv-only records.
        return (
            _IMPACT_MAX * 0.4,
            "Citation count unavailable from this source.",
        )
    days_since = (now - paper.published).days if paper.published else 365 * 5
    if days_since <= 0:
        days_since = 1
    years = max(days_since / 365.25, 0.1)
    velocity = paper.citation_count / years
    expected = EXPECTED_CITATION_VELOCITY[field]
    normalized = min(velocity / expected, 3.0) / 3.0
    return (
        _IMPACT_MAX * normalized,
        (
            f"Citations: {paper.citation_count}, "
            f"velocity: {velocity:.1f}/yr "
            f"(expected ~{expected:.0f}/yr for {field.value})."
        ),
    )
