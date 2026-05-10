"""SearchService behavior tests — concurrency, dedup, graceful failure,
cross-source enrichment, provenance tracking."""

from __future__ import annotations

from datetime import date

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
    assert [r.paper.id for r in results] == [vaswani_paper.id]
    assert results[0].sources == ("arxiv",)


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
    # arXiv prefix beats s2: prefix in canonical-id selection
    assert results[0].paper.id == vaswani_paper.id
    # provenance: both sources contributed
    assert results[0].sources == ("arxiv", "s2")


async def test_dedups_across_sources_by_doi() -> None:
    a = Paper(id="src1:1", title="t", abstract="a", authors=(), doi="10.1/X")
    b = Paper(id="src2:2", title="t", abstract="a", authors=(), doi="10.1/X")
    svc = SearchService([StaticSource("a", [a]), StaticSource("b", [b])])
    results = await svc.search(SearchQuery(text="t"))
    assert len(results) == 1
    assert results[0].sources == ("a", "b")


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
    assert results[0].paper.id == "arxiv:1706.03762"


async def test_title_dedup_does_not_collapse_distinct_papers() -> None:
    """Same title with different first authors must stay separate."""
    a = Paper(id="x:1", title="A Survey", abstract="...", authors=(Author("Alice Smith"),))
    b = Paper(id="x:2", title="A Survey", abstract="...", authors=(Author("Bob Jones"),))
    svc = SearchService([StaticSource("a", [a, b])])
    results = await svc.search(SearchQuery(text="survey"))
    assert {r.paper.id for r in results} == {"x:1", "x:2"}


async def test_failing_source_does_not_kill_merge(vaswani_paper: Paper) -> None:
    svc = SearchService([RaisingSource(), StaticSource("arxiv", [vaswani_paper])])
    results = await svc.search(SearchQuery(text="attention"))
    assert [r.paper.id for r in results] == [vaswani_paper.id]


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
    assert [r.paper.id for r in results] == ["src1:1", "src2:1", "src1:2", "src2:2"]


# ---- enrichment tests (M2) -------------------------------------------------


async def test_enrichment_keeps_doi_from_s2_when_arxiv_missing_it() -> None:
    """The audit's biggest win: DOI from S2 stops being thrown away when
    arXiv is the first source and has no DOI of its own."""
    arxiv_record = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="...",
        authors=(Author("Ashish Vaswani"),),
        arxiv_id="1706.03762",
        doi=None,
        venue=None,
    )
    s2_record = Paper(
        id="s2:204e3073",
        title="Attention Is All You Need",
        abstract="...",
        authors=(Author("Ashish Vaswani"),),
        arxiv_id="1706.03762",  # links to the arxiv paper
        doi="10.48550/arXiv.1706.03762",
        venue="NeurIPS",
        semantic_scholar_id="204e3073",
    )
    svc = SearchService(
        [
            StaticSource("arxiv", [arxiv_record]),
            StaticSource("semantic_scholar", [s2_record]),
        ]
    )
    [result] = await svc.search(SearchQuery(text="attention"))
    assert result.paper.id == "arxiv:1706.03762"  # arxiv id stays canonical
    assert result.paper.doi == "10.48550/arXiv.1706.03762"  # ← from S2
    assert result.paper.venue == "NeurIPS"  # ← from S2
    assert result.paper.semantic_scholar_id == "204e3073"  # ← from S2
    assert result.paper.arxiv_id == "1706.03762"  # kept from arxiv
    assert result.sources == ("arxiv", "semantic_scholar")


async def test_enrichment_prefers_longer_title_and_abstract() -> None:
    """If one source truncates the abstract, the merged record should keep
    the fuller text from the other."""
    short = Paper(
        id="arxiv:1",
        title="Attention Is All",  # truncated upstream
        abstract="seq2seq...",  # also truncated
        authors=(Author("Vaswani"),),
        arxiv_id="1",
    )
    full = Paper(
        id="s2:abc",
        title="Attention Is All You Need",
        abstract=(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks..."
        ),
        authors=(Author("Vaswani"),),
        arxiv_id="1",
    )
    svc = SearchService(
        [StaticSource("arxiv", [short]), StaticSource("s2", [full])]
    )
    [result] = await svc.search(SearchQuery(text="attention"))
    assert result.paper.title == "Attention Is All You Need"
    assert "transduction" in result.paper.abstract


async def test_enrichment_prefers_longer_author_list() -> None:
    short_authors = Paper(
        id="arxiv:1",
        title="t",
        abstract="a",
        authors=(Author("Vaswani"),),
        arxiv_id="1",
    )
    full_authors = Paper(
        id="s2:abc",
        title="t",
        abstract="a",
        authors=tuple(Author(n) for n in ["Vaswani", "Shazeer", "Parmar"]),
        arxiv_id="1",
    )
    svc = SearchService(
        [StaticSource("arxiv", [short_authors]), StaticSource("s2", [full_authors])]
    )
    [result] = await svc.search(SearchQuery(text="t"))
    assert len(result.paper.authors) == 3


async def test_enrichment_prefers_precise_published_date() -> None:
    """S2 sometimes returns year-only data with month/day=1; arXiv has the real date."""
    year_only = Paper(
        id="s2:abc",
        title="t",
        abstract="a",
        authors=(),
        published=date(2017, 1, 1),  # year-only sentinel
        doi="10.1/X",
    )
    full_date = Paper(
        id="arxiv:1",
        title="t",
        abstract="a",
        authors=(),
        published=date(2017, 6, 12),  # real publication date
        doi="10.1/X",
    )
    svc = SearchService(
        [StaticSource("s2", [year_only]), StaticSource("arxiv", [full_date])]
    )
    [result] = await svc.search(SearchQuery(text="t"))
    assert result.paper.published == date(2017, 6, 12)


async def test_provenance_single_source() -> None:
    p = Paper(id="arxiv:1", title="t", abstract="a", authors=(), arxiv_id="1")
    svc = SearchService([StaticSource("arxiv", [p])])
    [result] = await svc.search(SearchQuery(text="t"))
    assert result.sources == ("arxiv",)
