"""Heuristic citation-quality scorer — venue + impact + author + recency.

Inputs come from `Paper` (already populated by the source adapter):
no LLM calls, no network. Tradeoff: misses semantic appropriateness
("does this paper actually say what we want to cite it for?"). The
LLM-based scorer fills that gap; this one anchors on metadata that
survives offline.

Score breakdown (the four `CitationQualityScore` dimensions, summing
to 100):

  * `venue` (0-30) — top venue → 30, predatory → 0, unknown → ~9.
    This is the peer-review proxy: top conferences/journals get full
    credit, arXiv-only preprints get partial, predatory patterns zero.
  * `impact` (0-35) — citations / years-since-pub vs. an expected
    rate. arXiv-only papers (citation_count=None) get a baseline so
    they're not zeroed.
  * `author` (0-25) — h-index of strongest author, field-aware tier
    table. Falls back to a placeholder when no S2 author id is known.
  * `recency` (0-10) — logarithmic decay, **never zero**. A 50-year-
    old paper still gets a small contribution. Papers with citation
    count above `_FOUNDATIONAL_CITATION_THRESHOLD` skip the decay
    entirely — they've proven foundational regardless of age.

Recency dropped from 25 → 10 points in the 2026-05-13 revamp because
the previous weight + linear-decay-to-zero structurally penalized
foundational papers (Vaswani at 8.9y → 0 recency under CS half-life,
losing a quarter of the total score against newer-but-less-canonical
candidates). The 15 reclaimed points went to venue (+5) / impact (+5)
/ author (+5).

Warnings can zero the entire score:
  * `is_retracted` flag in `Paper.metadata` → total = 0
  * predatory-pattern venue → venue = 0 + warning
  * very recent (< 6 months) → impact-warn, but don't zero anything
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from research_mcp.citation_scorer._author import HIndexLookup, score_authors
from research_mcp.citation_scorer._field import Field
from research_mcp.citation_scorer._venues import (
    ARXIV_CATEGORY_BOOST,
    PREDATORY_PATTERNS,
    TOP_VENUE_EXACT_PATTERNS,
    TOP_VENUE_PARTIAL_PATTERNS,
)
from research_mcp.domain.citation_scorer import CitationQualityScore
from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper

# Maximum points each dimension can contribute, summing to 100 so
# `total` exactly equals venue + impact + author + recency.
#
# Weight history:
#   v1: 25 / 25 / 10 / 25, reserving 15 for a never-implemented
#       cross-source-consensus dimension. Total didn't reconcile.
#   v2: 25 / 30 / 20 / 25 (post-#15 / #17), residual eliminated,
#       redistributed into impact / author / recency.
#   v3 (this file): 30 / 35 / 25 / 10. Recency dropped from 25 to 10
#       because the previous weight + linear-decay-to-zero structurally
#       penalized foundational papers. Reclaimed points went to the
#       three signals that *don't* decay with age.
_VENUE_MAX: Final = 30.0
_IMPACT_MAX: Final = 35.0
_AUTHOR_MAX: Final = 25.0
_RECENCY_MAX: Final = 10.0

# Used by the recency dimension as the half-life of the logarithmic
# decay (NOT a hard cutoff — see `_score_recency`). Field-aware
# overrides this per detected discipline.
_RECENCY_HALF_LIFE_YEARS: Final = 5.0

# Papers with citation count at or above this threshold skip the
# recency decay entirely. The intuition: a paper with this many
# citations has *proven* foundational, and ranking it below a newer
# adjacent paper for being old is a clear scorer failure. 10K is a
# defensible field-agnostic floor — across CS, medicine, math, and
# physics, anything north of 10K cites is universally canonical.
_FOUNDATIONAL_CITATION_THRESHOLD: Final = 10_000

# Citation-velocity expectation (citations / year). 5 is a defensible
# field-agnostic baseline — CS papers cite faster, math much slower —
# but we don't yet detect field, so a single number stands in.
_EXPECTED_CITATION_VELOCITY: Final = 5.0

# Below this many days, citation count is too noisy to score on.
_VERY_RECENT_DAYS: Final = 180


class HeuristicCitationScorer:
    """Computes a `CitationQualityScore` from `Paper` metadata alone."""

    name: str = "heuristic"

    def __init__(
        self,
        *,
        now: _dt.date | None = None,
        h_index_lookup: HIndexLookup | None = None,
    ) -> None:
        # `now` is overridable for deterministic tests; production
        # constructs the scorer once at boot, so a fixed `now` would go
        # stale, but per-call freshness isn't worth a parameter.
        self._now = now or _dt.date.today()
        # `h_index_lookup` is the bound S2 `fetch_h_index` method (or a
        # test fake); None disables the h-index dim and falls back to the
        # placeholder. The heuristic always uses Field.DEFAULT tiers —
        # field-aware tier selection is the FieldAwareCitationScorer's job.
        self._h_index_lookup = h_index_lookup

    async def score(
        self,
        paper: Paper,
        claim: Claim | None = None,
    ) -> CitationQualityScore:
        del claim  # heuristic scorer is paper-only; LLM scorer will use the claim
        warnings: list[str] = []

        # Retraction is a blocking flag — a retracted paper isn't safe to
        # cite regardless of how good its venue / impact look.
        if _is_retracted(paper):
            return CitationQualityScore(
                total=0.0,
                venue=0.0,
                impact=0.0,
                author=0.0,
                recency=0.0,
                factors=MappingProxyType(
                    {"retraction": "Paper is marked retracted; do not cite."}
                ),
                warnings=("retracted",),
            )

        venue, venue_factor, venue_warns = _score_venue(paper)
        warnings.extend(venue_warns)
        impact, impact_factor, impact_warns = _score_impact(paper, self._now)
        warnings.extend(impact_warns)
        author, author_factor = await score_authors(
            paper, Field.DEFAULT, self._h_index_lookup
        )
        recency, recency_factor = _score_recency(paper, self._now)

        # Surface a warning when KEY metadata is missing so the user
        # knows the score is uncertain rather than silently low. The
        # chaos test caught Vaswani scoring 35/100 because arxiv-only
        # records lack citation_count + venue; without this warning,
        # the user can't tell that the score reflects missing data
        # rather than a poor citation.
        missing = []
        if paper.citation_count is None:
            missing.append("citation count")
        if not paper.venue:
            missing.append("venue")
        if missing:
            warnings.append(
                f"score reflects missing metadata: {', '.join(missing)} "
                "unavailable; cross-source enrichment may not have resolved"
            )

        total = venue + impact + author + recency

        factors: Mapping[str, str] = MappingProxyType(
            {
                "venue": venue_factor,
                "impact": impact_factor,
                "author": author_factor,
                "recency": recency_factor,
            }
        )
        return CitationQualityScore(
            total=round(total, 2),
            venue=round(venue, 2),
            impact=round(impact, 2),
            author=round(author, 2),
            recency=round(recency, 2),
            factors=factors,
            warnings=tuple(warnings),
        )


_PARTIAL_VENUE_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{re.escape(p)}\b") for p in TOP_VENUE_PARTIAL_PATTERNS
)


def _matches_top_venue(venue_lower: str) -> bool:
    """True for top-tier venues, False for everything else.

    Two pattern sets feed this. Partial patterns (`physical review letters`,
    `neurips`) are distinctive enough to use as substring/word-boundary
    matches without false positives. Exact patterns (`science`, `cell`,
    `acl`) collide with common English words and must equal the whole
    stripped venue string. The check runs both: if the venue is exactly
    one of the exact patterns OR any partial pattern matches, it's top-tier.
    """
    if venue_lower in TOP_VENUE_EXACT_PATTERNS:
        return True
    return any(rx.search(venue_lower) for rx in _PARTIAL_VENUE_REGEXES)


def _is_retracted(paper: Paper) -> bool:
    flag = (paper.metadata.get("is_retracted") or "").strip().lower()
    return flag in {"1", "true", "yes"}


def _score_venue(paper: Paper) -> tuple[float, str, list[str]]:
    venue = (paper.venue or "").lower().strip()
    if not venue:
        return _VENUE_MAX * 0.3, "Venue unknown", []

    # Predatory check first: a match overrides any positive signal.
    for pattern in PREDATORY_PATTERNS:
        if pattern in venue:
            return (
                0.0,
                f"Predatory-style venue name detected: '{paper.venue}'.",
                ["predatory venue suspected"],
            )

    # Top venue match: regex with word boundaries so "science" matches the
    # journal Science but not "Computer Science", and "cell" matches Cell
    # but not "Wireless Cell" or "Neural Cell Networks". TOP_VENUE_PATTERNS
    # entries with multiple words are anchored at both ends.
    if _matches_top_venue(venue):
        return _VENUE_MAX, f"Top-tier venue: '{paper.venue}'.", []

    # arXiv with a strong-category metadata hint
    if "arxiv" in venue:
        arxiv_cat = (paper.metadata.get("arxiv_primary_category") or "").lower()
        if arxiv_cat in ARXIV_CATEGORY_BOOST:
            return (
                _VENUE_MAX * 0.6,
                f"arXiv preprint in {arxiv_cat}; common venue for top ML papers.",
                [],
            )
        return _VENUE_MAX * 0.4, "arXiv preprint (peer-review status unknown).", []

    # Generic conference / journal hint
    if any(t in venue for t in ("conference", "proceedings", "journal", "transactions")):
        return _VENUE_MAX * 0.5, f"Standard venue: '{paper.venue}'.", []

    return _VENUE_MAX * 0.3, f"Unrecognized venue: '{paper.venue}'.", []


def _score_impact(paper: Paper, now: _dt.date) -> tuple[float, str, list[str]]:
    """Score citation impact, with a baseline when count is unknown.

    Citations-per-year normalized against a 5-cites/yr expectation.
    Cap at 3x the expectation (everything beyond that is rounding error
    in a 25-point dimension)."""
    if paper.citation_count is None:
        # Don't zero arXiv-only papers; surface the missing-data hint.
        return (
            _IMPACT_MAX * 0.4,
            "Citation count unavailable from this source.",
            [],
        )
    days_since = (now - paper.published).days if paper.published else 365 * 5
    if days_since <= 0:
        days_since = 1
    years = max(days_since / 365.25, 0.1)
    velocity = paper.citation_count / years
    normalized = min(velocity / _EXPECTED_CITATION_VELOCITY, 3.0) / 3.0

    warnings: list[str] = []
    if days_since < _VERY_RECENT_DAYS:
        warnings.append(
            "very recent — citation velocity not yet meaningful"
        )

    return (
        _IMPACT_MAX * normalized,
        (
            f"Citations: {paper.citation_count}, "
            f"velocity: {velocity:.1f}/yr "
            f"(expected ~{_EXPECTED_CITATION_VELOCITY:.0f})."
        ),
        warnings,
    )


def _score_recency(paper: Paper, now: _dt.date) -> tuple[float, str]:
    """Logarithmic-decay recency score that never zeros out foundational papers.

    Decay shape: ``1 / (1 + years / half_life)`` — at the half-life,
    factor is 0.5; never reaches 0 for any finite age. A 50-year-old
    paper still contributes ~7% of `_RECENCY_MAX`. This was changed
    from a linear-decay-to-zero in 2026-05-13 because the previous
    behavior penalized foundational papers (Vaswani at 8.9y under the
    CS half-life of 4y → 0 recency, costing 25 points it never had
    a chance to recover).

    Two exceptions to the smooth decay:

    1. **Foundational-citation exemption** — papers with citation
       count ≥ ``_FOUNDATIONAL_CITATION_THRESHOLD`` skip the decay
       entirely and get full recency credit. A 175K-cite paper has
       *proven* canonical; demoting it for age is a scorer failure.

    2. **Fresh-but-unproven penalty** — kept from the earlier
       version. When `citation_count` is None or 0 AND the paper is
       under the half-life, drop the contribution by 70%. Stops
       un-cited brand-new papers from free-riding to the top.
    """
    if not paper.published:
        return 0.0, "Publication date unknown."
    years = max((now - paper.published).days / 365.25, 0.0)

    if (
        paper.citation_count is not None
        and paper.citation_count >= _FOUNDATIONAL_CITATION_THRESHOLD
    ):
        return (
            _RECENCY_MAX,
            f"Published {years:.1f} years ago; recency uncapped because "
            f"citation count ({paper.citation_count:,}) is at or above "
            f"the foundational threshold ({_FOUNDATIONAL_CITATION_THRESHOLD:,}).",
        )

    factor = 1.0 / (1.0 + years / _RECENCY_HALF_LIFE_YEARS)
    score = _RECENCY_MAX * factor

    impact_unknown = paper.citation_count is None or paper.citation_count == 0
    if impact_unknown and years < _RECENCY_HALF_LIFE_YEARS:
        penalty = 0.3
        score *= penalty
        return (
            score,
            f"Published {years:.1f} years ago; recency contribution "
            f"capped at {penalty * 100:.0f}% because citation impact "
            "is unverified.",
        )
    return score, f"Published {years:.1f} years ago."
