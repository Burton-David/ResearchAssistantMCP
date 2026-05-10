"""SearchService behavior tests — concurrency, dedup, graceful failure."""

from __future__ import annotations

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.service import SearchService
from tests.conftest import RaisingSource, StaticSource

pytestmark = pytest.mark.unit


async def test_returns_results_from_single_source(vaswani_paper: Paper) -> None:
    src = StaticSource("arxiv", [vaswani_paper])
    svc = SearchService([src])
    results = await svc.search(SearchQuery(text="attention"))
    assert [p.id for p in results] == [vaswani_paper.id]


async def test_dedups_across_sources_by_arxiv_id(vaswani_paper: Paper) -> None:
    s2_copy = Paper(
        id="s2:abc",
        title=vaswani_paper.title,
        abstract=vaswani_paper.abstract,
        authors=vaswani_paper.authors,
        arxiv_id=vaswani_paper.arxiv_id,
    )
    src_a = StaticSource("arxiv", [vaswani_paper])
    src_b = StaticSource("s2", [s2_copy])
    svc = SearchService([src_a, src_b])
    results = await svc.search(SearchQuery(text="attention"))
    # both sources returned the same arXiv paper; merged result has one entry.
    assert len(results) == 1
    assert results[0].id == vaswani_paper.id  # first source wins


async def test_dedups_across_sources_by_doi() -> None:
    a = Paper(id="src1:1", title="t", abstract="a", authors=(), doi="10.1/X")
    b = Paper(id="src2:2", title="t", abstract="a", authors=(), doi="10.1/X")
    svc = SearchService([StaticSource("a", [a]), StaticSource("b", [b])])
    results = await svc.search(SearchQuery(text="t"))
    assert len(results) == 1


async def test_dedups_by_normalized_title_when_ids_disagree() -> None:
    """The same paper from two sources with different ids and casing."""
    a = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="seq2seq transduction",
        authors=(Author("Ashish Vaswani"),),
    )
    b = Paper(
        id="s2:204e3073",
        title="attention is all you need",  # title-cased differently, no arxiv_id
        abstract="seq2seq transduction",
        authors=(Author("Ashish Vaswani"),),
    )
    svc = SearchService([StaticSource("a", [a]), StaticSource("b", [b])])
    results = await svc.search(SearchQuery(text="attention"))
    assert len(results) == 1
    assert results[0].id == "arxiv:1706.03762"


async def test_title_dedup_does_not_collapse_distinct_papers() -> None:
    """Same title with different first authors must stay separate."""
    a = Paper(id="x:1", title="A Survey", abstract="...", authors=(Author("Alice Smith"),))
    b = Paper(id="x:2", title="A Survey", abstract="...", authors=(Author("Bob Jones"),))
    svc = SearchService([StaticSource("a", [a, b])])
    results = await svc.search(SearchQuery(text="survey"))
    assert {p.id for p in results} == {"x:1", "x:2"}


async def test_failing_source_does_not_kill_merge(vaswani_paper: Paper) -> None:
    svc = SearchService([RaisingSource(), StaticSource("arxiv", [vaswani_paper])])
    results = await svc.search(SearchQuery(text="attention"))
    assert [p.id for p in results] == [vaswani_paper.id]


def test_construction_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError):
        SearchService([])


async def test_max_results_caps_total_output() -> None:
    """Without this cap, max_results=N returned up to N * len(sources) rows."""
    a1 = Paper(id="src1:1", title="paper one", abstract="alpha", authors=())
    a2 = Paper(id="src1:2", title="paper two", abstract="alpha", authors=())
    b1 = Paper(id="src2:1", title="paper three", abstract="alpha", authors=())
    b2 = Paper(id="src2:2", title="paper four", abstract="alpha", authors=())
    svc = SearchService([StaticSource("a", [a1, a2]), StaticSource("b", [b1, b2])])
    results = await svc.search(SearchQuery(text="paper", max_results=2))
    assert len(results) == 2


async def test_round_robin_interleaves_sources() -> None:
    """First slot from first source, second slot from second source."""
    a1 = Paper(id="src1:1", title="paper a1", abstract="alpha", authors=())
    a2 = Paper(id="src1:2", title="paper a2", abstract="alpha", authors=())
    b1 = Paper(id="src2:1", title="paper b1", abstract="alpha", authors=())
    b2 = Paper(id="src2:2", title="paper b2", abstract="alpha", authors=())
    svc = SearchService([StaticSource("a", [a1, a2]), StaticSource("b", [b1, b2])])
    results = await svc.search(SearchQuery(text="paper", max_results=4))
    assert [p.id for p in results] == ["src1:1", "src2:1", "src1:2", "src2:2"]
