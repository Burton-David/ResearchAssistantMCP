"""Author-dimension scoring with field-aware h-index tiers.

Replaces the placeholder `0.5 * _AUTHOR_MAX` baseline in the heuristic
scorer with a real signal: max h-index across the paper's authors,
mapped through a per-field tier table.

The "strongest author" heuristic (max instead of mean) reflects how
credibility actually works for citations: a lab's senior researcher
carries the same weight whether they're first author (early career)
or last author (PI), and a senior co-author lifts a paper that an
unknown first author wouldn't carry alone. Mean would dilute that
signal across collaborator counts that vary wildly by field (HEP
papers with 3,000 authors vs. a 2-author math paper).

Falls back to the heuristic placeholder when h-index data isn't
available — the score should never zero an unknown-author paper to
"definitely not credible," only mark it as "uncertain."
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Final

from research_mcp.citation_scorer._field import Field
from research_mcp.domain.paper import Paper

_log = logging.getLogger(__name__)

# Public type alias for the lookup callable. Source adapters expose a
# method matching this signature; the wiring layer passes the bound
# method directly so the scorer doesn't need to know which Source it
# came from.
HIndexLookup = Callable[[str], Awaitable[int | None]]

_AUTHOR_MAX: Final = 20.0

# Multiplier when h-index data is unavailable (no authors, no lookup,
# all lookups returned None). Same as the old heuristic placeholder so
# the regression behavior is preserved for arXiv-only / pre-#1 papers.
_PLACEHOLDER_MULTIPLIER: Final = 0.5

# Per-field h-index tier table. Each tier is `(h_threshold, multiplier)`
# with thresholds sorted ascending. The scoring function walks the list
# and returns the highest multiplier whose threshold the author's
# h-index meets. Field-specific thresholds reflect h-index inflation
# across disciplines:
#   * Medicine runs highest — heavily-cited multi-author papers and
#     long reference lists; an h=40 medicine researcher is mid-career.
#   * Math runs lowest — small communities, slow citation accumulation;
#     an h=20 math researcher is well-established.
#   * CS / physics sit between. Used as the DEFAULT tier so unmapped
#     fields don't get the math discount or the medicine premium.
_TIERS: dict[Field, tuple[tuple[int, float], ...]] = {
    Field.CS: ((0, 0.0), (8, 0.4), (15, 0.6), (25, 0.8), (50, 1.0)),
    Field.MEDICINE: ((0, 0.0), (10, 0.4), (20, 0.6), (40, 0.8), (80, 1.0)),
    Field.MATH: ((0, 0.0), (5, 0.4), (10, 0.6), (20, 0.8), (40, 1.0)),
    Field.PHYSICS: ((0, 0.0), (8, 0.4), (15, 0.6), (25, 0.8), (50, 1.0)),
    Field.DEFAULT: ((0, 0.0), (8, 0.4), (15, 0.6), (25, 0.8), (50, 1.0)),
}


async def score_authors(
    paper: Paper,
    field: Field,
    lookup: HIndexLookup | None,
) -> tuple[float, str]:
    """Score the author dimension using h-index when available.

    Strategy: take MAX h-index across authors with a known `s2_id`,
    map through `_TIERS[field]`, return `(score, factor_text)`. Falls
    back to the heuristic placeholder (10/20) when no signal can be
    resolved — the placeholder is "we don't know" not "definitely not
    credible," and zeroing an unknown-author paper would systematically
    bias against junior researchers and underrepresented venues.
    """
    if not paper.authors:
        return (
            _AUTHOR_MAX * _PLACEHOLDER_MULTIPLIER,
            "No author metadata available.",
        )
    if lookup is None:
        return (
            _AUTHOR_MAX * _PLACEHOLDER_MULTIPLIER,
            f"{len(paper.authors)} author(s); h-index lookup not configured.",
        )
    h_indexes: list[int] = []
    for author in paper.authors:
        if not author.s2_id:
            continue
        try:
            h = await lookup(author.s2_id)
        except Exception as exc:  # lookup failures are non-fatal — fall back below
            # A flaky S2 call shouldn't poison the whole batch — log
            # for diagnosability and skip the author. The placeholder
            # fallback below kicks in if no authors resolve.
            _log.warning(
                "h-index lookup failed for author %s: %s",
                author.s2_id, exc,
            )
            continue
        if h is not None:
            h_indexes.append(h)
    if not h_indexes:
        return (
            _AUTHOR_MAX * _PLACEHOLDER_MULTIPLIER,
            f"{len(paper.authors)} author(s); h-index unavailable "
            "(no S2 author ids resolved or endpoint returned no data).",
        )
    max_h = max(h_indexes)
    multiplier = _tier_multiplier(max_h, _TIERS[field])
    return (
        _AUTHOR_MAX * multiplier,
        (
            f"Strongest author h-index: {max_h} "
            f"({len(h_indexes)}/{len(paper.authors)} authors resolved, "
            f"field={field.value} multiplier={multiplier:.2f})."
        ),
    )


def _tier_multiplier(h: int, tiers: tuple[tuple[int, float], ...]) -> float:
    """Walk ascending tiers; return the highest multiplier whose threshold
    the h-index meets. Tiers must be sorted by threshold ascending."""
    multiplier = 0.0
    for threshold, mult in tiers:
        if h >= threshold:
            multiplier = mult
        else:
            break
    return multiplier
