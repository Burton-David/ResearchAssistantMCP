"""DiscoveryService tests — title-based ranking, author tie-break, threshold."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.service import DiscoveryService, SearchService

pytestmark = pytest.mark.unit


class _ReturnAllSource:
    """Test source that returns its full paper list for every query.

    `StaticSource` uses substring matching, which doesn't compose well with
    Discovery's multi-word query (title + author surnames) — the substring
    rarely matches. Real search APIs do token-level matching, which is
    closer to what `_ReturnAllSource` does here. The Discovery service is
    responsible for re-ranking what the source returns; this source's job
    in the test is just to surface the candidates."""

    def __init__(self, name: str, papers: Sequence[Paper]) -> None:
        self.name = name
        self._papers = list(papers)

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        return self._papers[: query.max_results]

    async def fetch(self, paper_id: str) -> Paper | None:
        for p in self._papers:
            if p.id == paper_id:
                return p
        return None


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
    src = _ReturnAllSource(
        "arxiv",
        [
            _decoy("Do You Even Need Attention"),
            _vaswani(),
            _decoy("Attention Mechanisms in Vision"),
        ],
    )
    discovery = DiscoveryService(SearchService([src]))
    hits = await discovery.find_paper("Attention Is All You Need")
    assert hits, "expected at least one candidate"
    assert hits[0].paper.id == _vaswani().id
    # Exact-title match should clear an obvious confidence threshold.
    assert hits[0].confidence > 0.7


async def test_author_match_breaks_tie_between_same_titled_papers() -> None:
    """Two papers titled 'A Survey' by different first authors. The user
    asks for the Smith one — author bonus pushes Smith above Jones."""
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
    src = _ReturnAllSource("arxiv", [jones_paper, smith_paper])
    discovery = DiscoveryService(SearchService([src]))
    hits = await discovery.find_paper("A Survey", authors=("Alice Smith",))
    assert hits[0].paper.id == "x:smith"
    assert hits[0].confidence > hits[1].confidence


async def test_returns_empty_for_blank_title() -> None:
    src = _ReturnAllSource("arxiv", [_vaswani()])
    discovery = DiscoveryService(SearchService([src]))
    assert await discovery.find_paper("   ") == []


async def test_caps_at_three_results() -> None:
    """The service returns at most 3 candidates regardless of upstream count."""
    decoys = [_decoy(f"Attention Variant {i}") for i in range(10)]
    src = _ReturnAllSource("arxiv", decoys)
    discovery = DiscoveryService(SearchService([src]))
    hits = await discovery.find_paper("Attention Variant")
    assert len(hits) <= 3


async def test_zero_confidence_results_filtered_out() -> None:
    """When nothing in the search results overlaps the title even minimally,
    we don't surface noise."""
    src = _ReturnAllSource("arxiv", [_decoy("Quantum Cryptography in Bovine Lactation")])
    discovery = DiscoveryService(SearchService([src]))
    # Search will return the decoy since StaticSource matches by substring;
    # but the title-token Jaccard is 0, so it should be filtered.
    hits = await discovery.find_paper("Attention Is All You Need")
    assert hits == []
