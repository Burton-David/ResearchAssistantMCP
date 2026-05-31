"""Per-field parameters that override the heuristic scorer's defaults.

This module holds the two parameter tables `FieldAwareCitationScorer`
reads: **recency** half-lives (how fast a paper ages out of relevance)
and **impact** citation-velocity baselines (what a paper's velocity is
compared against). Venue stays field-agnostic — the venue lists in
`_venues.py` already span top-tier venues across all five fields (NEJM,
Annals of Math, PRL, NeurIPS, etc.). The author dimension is also
field-aware, but its per-field h-index tiers live in `_author.py`.

Numbers are picked to bracket realistic field differences rather than
chase precision; a math paper from 2010 SHOULD score about as well as
a CS paper from 2022 on recency. The exact half-lives are educated
guesses backed by typical citation-half-life surveys for each field.
"""

from __future__ import annotations

from research_mcp.citation_scorer._field import Field

# Years until recency contribution decays to zero. CS papers age out
# fastest; math papers age slowest (a 1990s proof is still the proof).
# Medicine sits in the middle — clinical trials need years to mature
# into citations but five-year-old guidelines are still current.
RECENCY_HALF_LIFE_YEARS: dict[Field, float] = {
    Field.CS: 4.0,
    Field.MEDICINE: 8.0,
    Field.MATH: 15.0,
    Field.PHYSICS: 6.0,
    Field.DEFAULT: 5.0,
}

# Expected citations per year for the impact normalization. The actual
# number doesn't matter much — what matters is the RATIO of an
# individual paper's velocity to its field's baseline. Math at 2/yr
# means a math paper with 10 cites/yr scores in the top tier; the same
# velocity in medicine (baseline 10) is merely average.
EXPECTED_CITATION_VELOCITY: dict[Field, float] = {
    Field.CS: 5.0,
    Field.MEDICINE: 10.0,
    Field.MATH: 2.0,
    Field.PHYSICS: 8.0,
    Field.DEFAULT: 5.0,
}
