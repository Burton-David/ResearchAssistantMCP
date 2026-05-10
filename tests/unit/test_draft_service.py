"""DraftService tests — claim extraction → per-claim citations → recommendations."""

from __future__ import annotations

from datetime import date

import pytest

from research_mcp.citation_scorer import HeuristicCitationScorer
from research_mcp.claim_extractor import FakeClaimExtractor
from research_mcp.domain.paper import Author, Paper
from research_mcp.service.citation import CitationService
from research_mcp.service.draft import (
    CitationRecommendation,
    DraftService,
)
from research_mcp.service.search import SearchService
from tests.conftest import ReturnAllSource

pytestmark = pytest.mark.unit


def _paper(
    *,
    paper_id: str = "arxiv:1",
    title: str = "Attention is all you need",
    venue: str = "NeurIPS",
    year: int = 2017,
    citations: int = 5000,
) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract=f"Abstract for {title}",
        authors=(Author("X"),),
        published=date(year, 1, 1),
        venue=venue,
        citation_count=citations,
    )


def _service(papers: list[Paper]) -> DraftService:
    return DraftService(
        extractor=FakeClaimExtractor(),
        citation=CitationService(
            search=SearchService([ReturnAllSource("arxiv", papers)]),
            scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
        ),
    )


# ---- ordering / shape ----


async def test_assist_returns_one_recommendation_per_claim() -> None:
    """FakeClaimExtractor emits one STATISTICAL claim per digit-bearing
    sentence; assist should return one CitationRecommendation per claim."""
    svc = _service([_paper()])
    text = "The model improved by 23%. We saw 47 papers. No claims here."
    recs = await svc.assist(text, k_per_claim=2)
    assert len(recs) == 2
    for r in recs:
        assert isinstance(r, CitationRecommendation)


async def test_assist_returns_empty_for_text_with_no_claims() -> None:
    svc = _service([_paper()])
    recs = await svc.assist("Pure description without numbers.", k_per_claim=2)
    assert recs == ()


async def test_assist_returns_empty_for_empty_text() -> None:
    svc = _service([_paper()])
    assert await svc.assist("", k_per_claim=2) == ()


async def test_assist_caps_candidates_per_claim() -> None:
    """k_per_claim bounds the candidate count per recommendation."""
    papers = [_paper(paper_id=f"arxiv:{i}", title=f"paper {i}") for i in range(10)]
    svc = _service(papers)
    text = "We achieved 90% accuracy."
    recs = await svc.assist(text, k_per_claim=3)
    assert recs
    for r in recs:
        assert len(r.candidates) <= 3


async def test_assist_orders_recommendations_by_claim_position() -> None:
    """Recommendations should track the order claims appear in the text."""
    svc = _service([_paper()])
    text = "First sentence with 1. Second with 2. Third with 3."
    recs = await svc.assist(text, k_per_claim=1)
    starts = [r.claim.start_char for r in recs]
    assert starts == sorted(starts)


# ---- recommendation shape ----


async def test_recommendation_carries_explanation_per_candidate() -> None:
    """Each candidate should have a non-empty explanation; the user sees
    that string when the agent renders the recommendation."""
    svc = _service([_paper(citations=2000)])
    text = "Our method improved by 30%."
    recs = await svc.assist(text, k_per_claim=1)
    assert recs
    rec = recs[0]
    assert rec.candidates
    for c in rec.candidates:
        assert c.explanation
        # A non-trivial sentence; substring check is loose to avoid
        # locking down exact phrasing.
        assert len(c.explanation) > 20


# ---- progress callback ----


async def test_assist_emits_progress_at_extraction_and_per_claim() -> None:
    """The MCP layer wires a progress callback bound to the client's
    progressToken; DraftService must invoke it at extraction start and
    after each claim completes so the user sees live progress."""
    svc = _service([_paper(citations=2000)])
    events: list[tuple[int, int, str]] = []

    async def progress(p: int, total: int, msg: str) -> None:
        events.append((p, total, msg))

    # Three sentences with digits → FakeClaimExtractor yields 3 claims.
    text = "First with 10. Second with 20. Third with 30."
    recs = await svc.assist(text, k_per_claim=1, progress=progress)
    assert len(recs) == 3

    # Expected: one start event, one "claims found" event, then one
    # per-claim completion. Total stays len(claims) + 1.
    assert events[0] == (0, 1, "extracting claims...")
    assert events[1][0] == 1  # one step done after extraction
    assert events[1][1] == 4  # 3 claims + 1 extraction step
    assert len(events) == 5  # start + found + 3 per-claim


async def test_assist_short_circuit_emits_zero_then_one_when_no_claims() -> None:
    """Pure-description text → no claims. The user still needs progress
    closure so the UI doesn't think the call is still running."""
    svc = _service([_paper(citations=2000)])
    events: list[tuple[int, int, str]] = []

    async def progress(p: int, total: int, msg: str) -> None:
        events.append((p, total, msg))

    text = "No numbers here. Pure description."
    recs = await svc.assist(text, k_per_claim=1, progress=progress)
    assert recs == ()
    # Two events: extraction-starting and no-claims-found closure.
    assert events == [(0, 1, "extracting claims..."), (1, 1, "no claims found")]


async def test_assist_no_progress_callback_runs_silently() -> None:
    """When progress=None, behavior matches the pre-progress baseline
    (no callback invocations, same return shape)."""
    svc = _service([_paper(citations=2000)])
    text = "First with 10. Second with 20."
    recs = await svc.assist(text, k_per_claim=1, progress=None)
    assert len(recs) == 2
