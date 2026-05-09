"""SearchService behavior tests — concurrency, dedup, graceful failure."""

from __future__ import annotations

import pytest

from research_mcp.domain.paper import Paper
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


async def test_failing_source_does_not_kill_merge(vaswani_paper: Paper) -> None:
    svc = SearchService([RaisingSource(), StaticSource("arxiv", [vaswani_paper])])
    results = await svc.search(SearchQuery(text="attention"))
    assert [p.id for p in results] == [vaswani_paper.id]


def test_construction_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError):
        SearchService([])
