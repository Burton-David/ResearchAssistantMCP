"""DiscoveryService tests — title-based ranking, author tie-break, threshold."""

from __future__ import annotations

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.service import DiscoveryService, SearchService
from tests.conftest import ReturnAllSource

pytestmark = pytest.mark.unit


def _vaswani() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="seq2seq transduction",
        authors=(Author("Ashish Vaswani"), Author("Noam Shazeer")),
        arxiv_id="1706.03762",
    )


def _decoy(title: str) -> Paper:
    return Paper(
        id=f"x:{abs(hash(title)) % 10000}",
        title=title,
        abstract="",
        authors=(Author("Anonymous"),),
    )


async def test_finds_canonical_paper_in_top_position() -> None:
    src = ReturnAllSource(
        "arxiv",
        [
            _decoy("Do You Even Need Attention"),
            _vaswani(),
            _decoy("Attention Mechanisms in Vision"),
        ],
    )
    discovery = DiscoveryService(SearchService([src]))
    outcome = await discovery.find_paper("Attention Is All You Need")
    assert outcome.hits, "expected at least one candidate"
    assert outcome.hits[0].paper.id == _vaswani().id
    assert outcome.hits[0].confidence > 0.7
    assert outcome.partial_failures == ()


async def test_author_match_breaks_tie_between_same_titled_papers() -> None:
    smith_paper = Paper(
        id="x:smith",
        title="A Survey",
        abstract="",
        authors=(Author("Alice Smith"),),
    )
    jones_paper = Paper(
        id="x:jones",
        title="A Survey",
        abstract="",
        authors=(Author("Bob Jones"),),
    )
    src = ReturnAllSource("arxiv", [jones_paper, smith_paper])
    discovery = DiscoveryService(SearchService([src]))
    outcome = await discovery.find_paper("A Survey", authors=("Alice Smith",))
    assert outcome.hits[0].paper.id == "x:smith"
    assert outcome.hits[0].confidence > outcome.hits[1].confidence


async def test_returns_empty_for_blank_title() -> None:
    src = ReturnAllSource("arxiv", [_vaswani()])
    discovery = DiscoveryService(SearchService([src]))
    outcome = await discovery.find_paper("   ")
    assert outcome.hits == []


async def test_caps_at_three_results() -> None:
    decoys = [_decoy(f"Attention Variant {i}") for i in range(10)]
    src = ReturnAllSource("arxiv", decoys)
    discovery = DiscoveryService(SearchService([src]))
    outcome = await discovery.find_paper("Attention Variant")
    assert len(outcome.hits) <= 3


async def test_zero_confidence_results_filtered_out() -> None:
    src = ReturnAllSource("arxiv", [_decoy("Quantum Cryptography in Bovine Lactation")])
    discovery = DiscoveryService(SearchService([src]))
    outcome = await discovery.find_paper("Attention Is All You Need")
    assert outcome.hits == []


async def test_empty_authors_list_works() -> None:
    src = ReturnAllSource("arxiv", [_vaswani(), _decoy("Vision Transformers")])
    discovery = DiscoveryService(SearchService([src]))
    outcome = await discovery.find_paper("Attention Is All You Need", authors=())
    assert outcome.hits, "expected at least one hit"
    assert outcome.hits[0].paper.id == _vaswani().id


async def test_partial_failures_pass_through_from_search() -> None:
    """A find_paper call against a partially-down source-set surfaces the
    upstream failures so the LLM client can decide between retry and
    'no such paper'."""
    from tests.conftest import UnavailableSource

    src_a = UnavailableSource("arxiv", "HTTP 429")
    src_b = ReturnAllSource("semantic_scholar", [_vaswani()])
    discovery = DiscoveryService(SearchService([src_a, src_b]))
    outcome = await discovery.find_paper("Attention Is All You Need")
    # Hit still surfaces from the live source.
    assert outcome.hits[0].paper.id == _vaswani().id
    # Failure from the dead source flows through.
    assert any("arxiv" in f and "429" in f for f in outcome.partial_failures)


def test_has_significant_tokens_predicate() -> None:
    """The MCP layer uses this to surface a 'all-stopwords' note."""
    from research_mcp.service.discovery import has_significant_tokens

    assert has_significant_tokens("Attention Is All You Need") is True
    assert has_significant_tokens("a the of and an") is False
    assert has_significant_tokens("") is False
    assert has_significant_tokens("   ") is False
