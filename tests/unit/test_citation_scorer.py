"""HeuristicCitationScorer tests — venue tiers, impact, recency, warnings."""

from __future__ import annotations

from datetime import date

import pytest

from research_mcp.citation_scorer import (
    FakeCitationScorer,
    HeuristicCitationScorer,
)
from research_mcp.domain.citation_scorer import CitationScorer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def _paper(
    *,
    venue: str | None = None,
    year: int | None = 2020,
    citation_count: int | None = None,
    authors: tuple[Author, ...] = (Author("X"),),
    metadata: dict[str, str] | None = None,
    paper_id: str = "x:1",
) -> Paper:
    from types import MappingProxyType

    return Paper(
        id=paper_id,
        title="Test paper",
        abstract="",
        authors=authors,
        published=date(year, 1, 1) if year is not None else None,
        venue=venue,
        citation_count=citation_count,
        metadata=MappingProxyType(metadata or {}),
    )


# ---- protocol conformance ----


def test_heuristic_satisfies_protocol() -> None:
    s = HeuristicCitationScorer()
    assert isinstance(s, CitationScorer)
    assert s.name == "heuristic"


def test_fake_satisfies_protocol() -> None:
    s = FakeCitationScorer()
    assert isinstance(s, CitationScorer)
    assert s.name == "fake"


# ---- score is 0..100 ----


async def test_total_score_is_in_zero_hundred() -> None:
    """Across many shapes of paper, total stays bounded."""
    scorer = HeuristicCitationScorer()
    for paper in [
        _paper(venue="NeurIPS", year=2020, citation_count=5000),
        _paper(venue=None, year=None, citation_count=None),
        _paper(venue="Sketchy Predatory Journal", year=2024, citation_count=0),
    ]:
        result = await scorer.score(paper)
        assert 0.0 <= result.total <= 100.0


# ---- venue tier ----


async def test_top_venue_scores_higher_than_unknown() -> None:
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    top = await scorer.score(
        _paper(venue="NeurIPS", year=2020, citation_count=5000)
    )
    weak = await scorer.score(
        _paper(venue=None, year=2020, citation_count=5000)
    )
    assert top.venue > weak.venue


async def test_recognizes_top_journals_case_insensitive() -> None:
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    nature = await scorer.score(_paper(venue="Nature", year=2020))
    nature_lower = await scorer.score(_paper(venue="nature medicine", year=2020))
    # Both should score higher than an unknown venue
    unknown = await scorer.score(_paper(venue="Local Workshop on X", year=2020))
    assert nature.venue > unknown.venue
    assert nature_lower.venue > unknown.venue


async def test_predatory_venue_flagged_and_penalized() -> None:
    """Names containing predatory-list patterns must produce a warning AND
    drag the venue score to zero (not just shave a bit). False-flagging a
    real venue is bad, but our predatory list is conservative — a clear
    pattern match is the user's signal to double-check."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(
        _paper(venue="Open Access Journal of Multidisciplinary Studies", year=2024)
    )
    assert any("predatory" in w.lower() for w in result.warnings)
    assert result.venue == 0.0


async def test_single_word_journal_match_is_exact_only() -> None:
    """`Science` (the journal) must match; `Computer Science` and
    `Procedia Computer Science` must NOT — they aren't the journal. This
    was a real false-positive caught at REPL test."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    journal = await scorer.score(_paper(venue="Science", year=2020))
    not_journal = await scorer.score(
        _paper(venue="Procedia Computer Science", year=2020)
    )
    assert journal.venue == 25.0  # exact match
    assert not_journal.venue < 25.0  # generic conference / unknown


async def test_acronym_match_does_not_fire_inside_longer_words() -> None:
    """`acl` must match the conference but not 'manacled' or 'oracle'."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    acl = await scorer.score(_paper(venue="ACL", year=2020))
    nope = await scorer.score(_paper(venue="Oracle Database Conference", year=2020))
    assert acl.venue == 25.0
    assert nope.venue < 25.0


async def test_warns_when_citation_count_missing() -> None:
    """The chaos test caught Vaswani scoring 35/100 because its arxiv-only
    record had no citation_count. Without a warning, the user couldn't
    tell whether the score reflected a poor citation or missing data."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(
        _paper(venue="NeurIPS", year=2017, citation_count=None)
    )
    assert any(
        "missing metadata" in w.lower() and "citation count" in w.lower()
        for w in result.warnings
    )


async def test_warns_when_venue_missing() -> None:
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(_paper(venue=None, year=2017, citation_count=10000))
    assert any(
        "missing metadata" in w.lower() and "venue" in w.lower()
        for w in result.warnings
    )


async def test_warns_about_both_when_both_missing() -> None:
    """The arxiv-only-record case: no citation_count AND no venue. One
    consolidated warning, both fields named."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(_paper(venue=None, year=2017, citation_count=None))
    metadata_warnings = [w for w in result.warnings if "missing metadata" in w.lower()]
    assert metadata_warnings
    msg = metadata_warnings[0]
    assert "citation count" in msg.lower()
    assert "venue" in msg.lower()


async def test_no_metadata_warning_when_fully_enriched() -> None:
    """A fully-enriched paper (venue + citation_count + year) should not
    raise the missing-metadata warning."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(
        _paper(venue="NeurIPS", year=2017, citation_count=70000)
    )
    assert not any("missing metadata" in w.lower() for w in result.warnings)


# ---- impact / citation count ----


async def test_high_citation_count_boosts_impact() -> None:
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    high = await scorer.score(
        _paper(venue="NeurIPS", year=2017, citation_count=50000)
    )
    low = await scorer.score(
        _paper(venue="NeurIPS", year=2017, citation_count=2)
    )
    assert high.impact > low.impact


async def test_unknown_citation_count_does_not_zero_impact() -> None:
    """A paper with citation_count=None (e.g., from arXiv) should still
    get a non-zero impact baseline, otherwise arXiv-only results all
    collapse to zero impact and never make recommendations."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(_paper(venue="arXiv", year=2024, citation_count=None))
    assert result.impact > 0.0


# ---- recency ----


async def test_recent_paper_scores_higher_recency_than_old_one() -> None:
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    recent = await scorer.score(_paper(venue="NeurIPS", year=2024))
    old = await scorer.score(_paper(venue="NeurIPS", year=1995))
    assert recent.recency > old.recency


async def test_recency_is_zero_when_year_unknown() -> None:
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(_paper(venue="NeurIPS", year=None))
    assert result.recency == 0.0


# ---- retraction / very-recent warnings ----


async def test_retracted_metadata_produces_warning_and_zeroes_score() -> None:
    """A retracted paper is unsafe to cite regardless of venue/impact."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(
        _paper(
            venue="NeurIPS",
            year=2018,
            citation_count=2000,
            metadata={"is_retracted": "true"},
        )
    )
    assert any("retract" in w.lower() for w in result.warnings)
    assert result.total == 0.0


async def test_very_recent_paper_warned_about_low_citation_signal() -> None:
    """Papers under 6 months old don't have meaningful citation velocity;
    flag so the user knows the impact dimension is unreliable."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(
        _paper(venue="NeurIPS", year=2026, citation_count=0)
    )
    assert any("recent" in w.lower() or "velocity" in w.lower() for w in result.warnings)


# ---- factors map (used by explain_citation) ----


async def test_factors_contains_human_readable_explanations() -> None:
    """`factors` is the dict that explain_citation reads. Every key
    populated implies a corresponding sentence we'd show the user."""
    scorer = HeuristicCitationScorer(now=date(2026, 5, 10))
    result = await scorer.score(
        _paper(venue="NeurIPS", year=2017, citation_count=5000)
    )
    assert "venue" in result.factors
    assert "recency" in result.factors
    assert "impact" in result.factors
    # Each factor is a non-empty sentence.
    for value in result.factors.values():
        assert len(value) > 5


# ---- FakeCitationScorer ----


async def test_fake_returns_constant_50() -> None:
    """Predictable for tests; can be checked without computing weights."""
    scorer = FakeCitationScorer()
    p = _paper(venue="NeurIPS", year=2020)
    result = await scorer.score(p)
    assert result.total == 50.0
    assert result.warnings == ()
