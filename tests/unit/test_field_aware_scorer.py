"""FieldAwareCitationScorer + field detection unit tests."""

from __future__ import annotations

from datetime import date
from types import MappingProxyType

import pytest

from research_mcp.citation_scorer._field import Field, detect_field
from research_mcp.citation_scorer.field_aware import FieldAwareCitationScorer
from research_mcp.citation_scorer.heuristic import HeuristicCitationScorer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


# ---- field detection ----


def _paper(**metadata: str) -> Paper:
    """Minimal Paper just to test detect_field — title/abstract irrelevant."""
    return Paper(
        id="test:1",
        title="t",
        abstract="",
        authors=(Author("a"),),
        metadata=MappingProxyType(metadata),
    )


def test_detect_field_returns_cs_for_arxiv_cs_category() -> None:
    """`cs.LG`, `cs.AI`, `cs.CL` → CS. The category code is split on `.` and
    the prefix maps to the field; the suffix (sub-discipline) doesn't
    matter for our coarse buckets."""
    assert detect_field(_paper(arxiv_primary_category="cs.LG")) == Field.CS
    assert detect_field(_paper(arxiv_primary_category="cs.CL")) == Field.CS
    assert detect_field(_paper(arxiv_primary_category="cs.CV")) == Field.CS


def test_detect_field_routes_stat_to_cs() -> None:
    """`stat.ML` papers are statistical-ML methods used heavily in CS;
    treating them as CS gives the right recency/velocity expectations."""
    assert detect_field(_paper(arxiv_primary_category="stat.ML")) == Field.CS
    assert detect_field(_paper(arxiv_primary_category="stat.AP")) == Field.CS


def test_detect_field_returns_math_for_arxiv_math_category() -> None:
    assert detect_field(_paper(arxiv_primary_category="math.AG")) == Field.MATH
    assert detect_field(_paper(arxiv_primary_category="math.NT")) == Field.MATH


def test_detect_field_returns_physics_for_arxiv_physics_variants() -> None:
    """arXiv has many physics-adjacent top-level categories; condensed
    matter / astro-ph / hep-* / quant-ph all behave like physics for
    citation purposes (slow citation accumulation, top venues are
    Phys Rev Letters etc.). They should all collapse to Field.PHYSICS."""
    assert detect_field(_paper(arxiv_primary_category="physics.flu-dyn")) == Field.PHYSICS
    assert detect_field(_paper(arxiv_primary_category="cond-mat.str-el")) == Field.PHYSICS
    assert detect_field(_paper(arxiv_primary_category="astro-ph.HE")) == Field.PHYSICS
    assert detect_field(_paper(arxiv_primary_category="hep-th")) == Field.PHYSICS
    assert detect_field(_paper(arxiv_primary_category="quant-ph")) == Field.PHYSICS


def test_detect_field_maps_q_bio_to_medicine() -> None:
    """Quantitative biology is the closest arXiv bucket to medicine —
    biomarker discovery, computational genomics, etc."""
    assert detect_field(_paper(arxiv_primary_category="q-bio.GN")) == Field.MEDICINE


def test_detect_field_returns_default_for_unknown_arxiv_prefix() -> None:
    """A real arXiv category we don't map (`econ.GN`, `eess.SY`)
    shouldn't crash — return DEFAULT and let the heuristic scorer
    handle it with its default 5-year recency."""
    assert detect_field(_paper(arxiv_primary_category="econ.GN")) == Field.DEFAULT
    assert detect_field(_paper(arxiv_primary_category="eess.SY")) == Field.DEFAULT


def test_detect_field_falls_back_to_openalex_field_when_arxiv_missing() -> None:
    """A paper from OpenAlex (no arxiv_primary_category) should still be
    detectable via openalex_field."""
    assert detect_field(_paper(openalex_field="Computer Science")) == Field.CS
    assert detect_field(_paper(openalex_field="Medicine")) == Field.MEDICINE
    assert detect_field(_paper(openalex_field="Mathematics")) == Field.MATH


def test_detect_field_handles_openalex_aliases_for_medicine() -> None:
    """OpenAlex's `field.display_name` for clinical/biomedical work has
    several variants — we collapse them all to MEDICINE for scoring
    purposes (same citation half-life and velocity)."""
    for alias in (
        "Medicine",
        "Nursing",
        "Dentistry",
        "Pharmacology, Toxicology and Pharmaceutics",
        "Biochemistry, Genetics and Molecular Biology",
    ):
        assert detect_field(_paper(openalex_field=alias)) == Field.MEDICINE


def test_detect_field_arxiv_wins_over_openalex_when_both_present() -> None:
    """When a paper carries both signals (cross-source enrichment), the
    finer-grained arXiv category wins — `cs.LG` is more specific than
    OpenAlex's coarse "Computer Science"."""
    paper = _paper(
        arxiv_primary_category="math.AG",
        openalex_field="Computer Science",  # Disagree intentionally
    )
    assert detect_field(paper) == Field.MATH


def test_detect_field_returns_default_when_both_metadata_keys_missing() -> None:
    """No metadata at all (e.g., a paper from a source that doesn't
    populate field hints) returns DEFAULT — the heuristic scorer's
    field-agnostic defaults apply."""
    assert detect_field(_paper()) == Field.DEFAULT


def test_detect_field_returns_default_for_empty_string_values() -> None:
    """Defensive: `arxiv_primary_category=""` shouldn't crash trying to
    split on `.` (it would yield `[""]` and then look up `""` in the
    prefix map, which isn't there, so DEFAULT is returned. But the
    test locks this in)."""
    assert detect_field(_paper(arxiv_primary_category="")) == Field.DEFAULT
    assert detect_field(_paper(openalex_field="")) == Field.DEFAULT


# ---- FieldAwareCitationScorer ----


_FIXED_NOW = date(2026, 5, 11)


def _math_paper_old(citations: int = 50) -> Paper:
    """A 2015 math paper — 11 years old. Under CS half-life (4y) this is
    far past its prime; under math half-life (15y) it's still relevant."""
    return Paper(
        id="arxiv:1503.00001",
        title="A New Construction in Algebraic Geometry",
        abstract="...",
        authors=(Author("Mathematician"),),
        published=date(2015, 5, 1),
        venue=None,
        citation_count=citations,
        arxiv_id="1503.00001",
        metadata=MappingProxyType({"arxiv_primary_category": "math.AG"}),
    )


def _cs_paper_old(citations: int = 50) -> Paper:
    """A 2015 CS paper — same age as the math one, different field."""
    return Paper(
        id="arxiv:1503.00002",
        title="Old Attention Mechanism",
        abstract="...",
        authors=(Author("Researcher"),),
        published=date(2015, 5, 1),
        venue=None,
        citation_count=citations,
        arxiv_id="1503.00002",
        metadata=MappingProxyType({"arxiv_primary_category": "cs.LG"}),
    )


def _no_metadata_paper(citations: int = 50) -> Paper:
    """Identical to the CS paper above but missing the field-hint metadata,
    so detect_field returns DEFAULT."""
    return Paper(
        id="arxiv:1503.00003",
        title="Paper Without Field Hints",
        abstract="...",
        authors=(Author("Anonymous"),),
        published=date(2015, 5, 1),
        venue=None,
        citation_count=citations,
        arxiv_id="1503.00003",
    )


async def test_field_aware_returns_base_unchanged_for_default_field() -> None:
    """When `detect_field` returns DEFAULT (no metadata hints), the
    field-aware scorer is a pass-through. This is the regression-safety
    contract: callers that don't carry field metadata get the exact same
    score they always have."""
    scorer = FieldAwareCitationScorer(
        HeuristicCitationScorer(now=_FIXED_NOW), now=_FIXED_NOW
    )
    paper = _no_metadata_paper()
    base = await HeuristicCitationScorer(now=_FIXED_NOW).score(paper)
    field_aware = await scorer.score(paper)
    assert field_aware.total == base.total
    assert field_aware.recency == base.recency
    assert field_aware.impact == base.impact
    # And the factor dict is identical — no "field" key inserted.
    assert "field" not in field_aware.factors


async def test_field_aware_extends_recency_for_old_math_paper() -> None:
    """An 11-year-old math paper under CS half-life (4y) would score 0
    on recency. Under math half-life (15y) it should retain ~4/15 ≈ 27%
    of the recency dimension — visibly higher than the CS treatment."""
    cs_scorer = FieldAwareCitationScorer(
        HeuristicCitationScorer(now=_FIXED_NOW), now=_FIXED_NOW
    )
    cs_score = await cs_scorer.score(_cs_paper_old())
    math_score = await cs_scorer.score(_math_paper_old())
    # CS recency for an 11-year-old paper is 0 (past 4y half-life).
    assert cs_score.recency == 0.0
    # Math recency for the same age is positive (still under 15y half-life).
    assert math_score.recency > 0.0
    # And the math paper's total should reflect the recency uplift.
    assert math_score.total > cs_score.total


async def test_field_aware_rescales_impact_for_math_baseline() -> None:
    """50 citations over 11 years = ~4.5 citations/year. Against CS's
    5/year baseline that's average (~0.9x). Against math's 2/year
    baseline it's well above (~2.25x). Math impact should outscore CS
    impact for the identical citation history."""
    scorer = FieldAwareCitationScorer(
        HeuristicCitationScorer(now=_FIXED_NOW), now=_FIXED_NOW
    )
    cs = await scorer.score(_cs_paper_old(citations=50))
    math = await scorer.score(_math_paper_old(citations=50))
    assert math.impact > cs.impact


async def test_field_aware_records_detected_field_in_factors() -> None:
    """The factors dict reports the detected field for telemetry — lets a
    debugger see why scoring diverged from base without having to re-run
    detection. Format is loose, just contains the field name."""
    scorer = FieldAwareCitationScorer(
        HeuristicCitationScorer(now=_FIXED_NOW), now=_FIXED_NOW
    )
    score = await scorer.score(_math_paper_old())
    assert "field" in score.factors
    assert "math" in score.factors["field"].lower()


async def test_field_aware_preserves_venue_and_author_from_base() -> None:
    """Field-aware overrides recency + impact; venue + author come
    straight from base. Verify by comparing dim-by-dim against the
    base scorer for a non-DEFAULT-field paper."""
    base_scorer = HeuristicCitationScorer(now=_FIXED_NOW)
    paper = _cs_paper_old()
    base = await base_scorer.score(paper)
    field_aware = await FieldAwareCitationScorer(
        base_scorer, now=_FIXED_NOW,
    ).score(paper)
    assert field_aware.venue == base.venue
    assert field_aware.author == base.author


async def test_field_aware_total_reconciles_with_dimensions() -> None:
    """Locks down the reconciliation invariant the chaos test caught
    (total ≠ sum-of-dims). For field-aware output, `total` must equal
    `venue + impact + author + recency` (within rounding)."""
    scorer = FieldAwareCitationScorer(
        HeuristicCitationScorer(now=_FIXED_NOW), now=_FIXED_NOW
    )
    score = await scorer.score(_math_paper_old())
    expected = score.venue + score.impact + score.author + score.recency
    assert abs(score.total - expected) < 0.05


async def test_field_aware_preserves_base_warnings() -> None:
    """The heuristic scorer emits warnings for missing metadata, very-
    recent papers, predatory venues, retracted papers. Field-aware
    composition mustn't drop those — the warnings carry actionable
    info for the LLM caller."""
    scorer = FieldAwareCitationScorer(
        HeuristicCitationScorer(now=_FIXED_NOW), now=_FIXED_NOW
    )
    # arXiv-only paper with no venue → base emits a "missing metadata" warning.
    score = await scorer.score(_cs_paper_old())
    assert any("missing metadata" in w for w in score.warnings)


async def test_field_aware_name_is_stable() -> None:
    """Name is read by library_status / telemetry — lock it down."""
    scorer = FieldAwareCitationScorer()
    assert scorer.name == "field_aware"
