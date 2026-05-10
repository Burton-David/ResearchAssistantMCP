"""CitationService tests — query generation, scoring, ranking, explanations."""

from __future__ import annotations

from datetime import date

import pytest

from research_mcp.citation_scorer import (
    FakeCitationScorer,
    HeuristicCitationScorer,
)
from research_mcp.domain.claim import Claim, ClaimType
from research_mcp.domain.paper import Author, Paper
from research_mcp.service.citation import CitationService
from research_mcp.service.search import SearchService
from tests.conftest import ReturnAllSource

pytestmark = pytest.mark.unit


def _claim(text: str = "outperforms", *, search_terms: tuple[str, ...] = ("attention",)) -> Claim:
    return Claim(
        text=text,
        type=ClaimType.COMPARATIVE,
        confidence=0.85,
        context=text,
        suggested_search_terms=search_terms,
    )


def _paper(
    *,
    paper_id: str = "arxiv:1706.03762",
    title: str = "Attention paper",
    venue: str | None = "NeurIPS",
    year: int | None = 2017,
    citations: int | None = 5000,
    metadata: dict[str, str] | None = None,
) -> Paper:
    from types import MappingProxyType

    return Paper(
        id=paper_id,
        title=title,
        abstract=f"Abstract for {title}",
        authors=(Author("X"),),
        published=date(year, 1, 1) if year else None,
        venue=venue,
        citation_count=citations,
        metadata=MappingProxyType(metadata or {}),
    )


# ---- find_citations ----


async def test_find_citations_uses_claim_search_terms_for_query() -> None:
    """The query handed to SearchService should be derived from the claim's
    suggested_search_terms, not the literal claim text — search terms are
    the noun chunks; the literal text might be one word like 'outperforms'."""

    # Capture the query text seen by the source so the test asserts the
    # actual contract: search_terms drove the query.
    class _CapturingSource:
        name = "arxiv"
        id_prefixes = ("arxiv",)

        def __init__(self, paper: Paper) -> None:
            self._paper = paper
            self.last_query: str | None = None

        async def search(self, query):  # type: ignore[no-untyped-def]
            self.last_query = query.text
            return [self._paper]

        async def fetch(self, paper_id):  # type: ignore[no-untyped-def]
            return self._paper if paper_id == self._paper.id else None

    paper = _paper()
    source = _CapturingSource(paper)
    svc = CitationService(
        search=SearchService([source]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    claim = _claim(text="outperforms", search_terms=("attention", "transformer"))
    result = await svc.find_citations(claim, k=5)
    # Both search_terms appeared in the query.
    assert source.last_query is not None
    assert "attention" in source.last_query
    assert "transformer" in source.last_query
    assert any(c.paper.id == paper.id for c in result)


async def test_find_citations_returns_top_k_by_total_score() -> None:
    """High-impact paper should rank above a weaker one for the same claim."""
    high = _paper(
        paper_id="arxiv:1",
        title="attention paper one",
        venue="NeurIPS",
        year=2017,
        citations=10000,
    )
    low = _paper(
        paper_id="arxiv:2",
        title="attention paper two",
        venue=None,
        year=2017,
        citations=2,
    )
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [high, low])]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    result = await svc.find_citations(_claim(), k=2)
    assert [c.paper.id for c in result] == [high.id, low.id]


async def test_find_citations_respects_k() -> None:
    papers = [_paper(paper_id=f"arxiv:{i}", title=f"attention {i}") for i in range(10)]
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", papers)]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    result = await svc.find_citations(_claim(), k=3)
    assert len(result) == 3


async def test_find_citations_falls_back_to_claim_text_when_no_search_terms() -> None:
    """A claim with empty search_terms falls back to context, then text."""
    paper = _paper(title="outperforms benchmark", citations=100)
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [paper])]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    claim = Claim(
        text="outperforms",
        type=ClaimType.COMPARATIVE,
        confidence=0.85,
        context="model outperforms baselines",
        suggested_search_terms=(),
    )
    result = await svc.find_citations(claim, k=5)
    assert any(c.paper.id == paper.id for c in result)


async def test_find_citations_returns_empty_for_no_search_terms_and_blank_claim() -> None:
    """A claim with no signal at all yields no citations rather than crashing."""
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [_paper()])]),
        scorer=HeuristicCitationScorer(),
    )
    claim = Claim(
        text="",
        type=ClaimType.COMPARATIVE,
        confidence=0.85,
        context="",
        suggested_search_terms=(),
    )
    result = await svc.find_citations(claim, k=3)
    assert result == ()


# ---- score_citation ----


async def test_score_citation_delegates_to_scorer() -> None:
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [])]),
        scorer=FakeCitationScorer(),
    )
    score = await svc.score_citation(_paper())
    assert score.total == 50.0


# ---- explain_citation ----


async def test_explain_citation_includes_claim_type_and_factors() -> None:
    """Explanation reads as a recommendation: it names the claim it would be
    citing for, the venue, and any warnings. The exact wording is flexible."""
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [])]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    paper = _paper(venue="NeurIPS", year=2017, citations=5000)
    claim = _claim(text="outperforms")
    explanation = await svc.explain_citation(paper, claim)
    # The explanation must mention the venue and the claim type for the user
    # to understand the recommendation.
    assert "NeurIPS" in explanation
    assert "comparative" in explanation.lower() or "claim" in explanation.lower()


async def test_explain_citation_mentions_warnings_when_present() -> None:
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [])]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    retracted_paper = _paper(metadata={"is_retracted": "true"})
    explanation = await svc.explain_citation(retracted_paper, _claim())
    assert "retract" in explanation.lower()


async def test_explain_citation_classifies_strength() -> None:
    """Output should classify the recommendation as strong / moderate /
    weak so a user scanning multiple recommendations can prioritize."""
    svc = CitationService(
        search=SearchService([ReturnAllSource("arxiv", [])]),
        scorer=HeuristicCitationScorer(now=date(2026, 5, 10)),
    )
    strong = await svc.explain_citation(
        _paper(venue="NeurIPS", year=2024, citations=2000), _claim()
    )
    weak = await svc.explain_citation(
        _paper(venue=None, year=1980, citations=None), _claim()
    )
    assert any(word in strong.lower() for word in ("strong", "moderate"))
    assert "weak" in weak.lower() or "low" in weak.lower()
