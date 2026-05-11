"""Domain dataclass round-trip and hashability tests."""

from __future__ import annotations

from datetime import date

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.index._codec import paper_from_dict, paper_to_dict

pytestmark = pytest.mark.unit


def test_author_is_hashable_and_equal() -> None:
    a = Author("Vaswani", affiliation="Google")
    b = Author("Vaswani", affiliation="Google")
    assert hash(a) == hash(b)
    assert a == b
    assert {a, b} == {a}


def test_paper_is_frozen() -> None:
    p = Paper(id="arxiv:1", title="t", abstract="a", authors=(Author("X"),))
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises FrozenInstanceError
        p.title = "other"  # type: ignore[misc]  # we are deliberately violating frozen for the test


def test_search_query_defaults() -> None:
    q = SearchQuery(text="anything")
    assert q.max_results == 20
    assert q.year_min is None
    assert q.year_max is None
    assert q.authors == ()


def test_paper_codec_roundtrip(vaswani_paper: Paper) -> None:
    encoded = paper_to_dict(vaswani_paper)
    decoded = paper_from_dict(encoded)
    assert decoded == vaswani_paper


def test_paper_codec_handles_none_published() -> None:
    p = Paper(id="x:1", title="t", abstract="a", authors=(Author("Y"),))
    decoded = paper_from_dict(paper_to_dict(p))
    assert decoded.published is None


def test_paper_codec_handles_metadata() -> None:
    from types import MappingProxyType

    p = Paper(
        id="x:1",
        title="t",
        abstract="a",
        authors=(),
        metadata=MappingProxyType({"k": "v"}),
        published=date(2020, 1, 1),
    )
    decoded = paper_from_dict(paper_to_dict(p))
    assert dict(decoded.metadata) == {"k": "v"}


def test_paper_codec_preserves_citation_count() -> None:
    """citation_count drives the heuristic scorer's impact dimension; if
    the codec drops it, papers reloaded from FAISS lose enrichment and
    score 35/100 instead of 70+. This was a real bug caught by the
    code-quality review pass."""
    p = Paper(
        id="arxiv:1706.03762",
        title="Attention",
        abstract="...",
        authors=(Author("Vaswani"),),
        published=date(2017, 6, 12),
        citation_count=175574,
        venue="NeurIPS",
    )
    decoded = paper_from_dict(paper_to_dict(p))
    assert decoded.citation_count == 175574
    assert decoded.venue == "NeurIPS"


def test_paper_codec_preserves_none_citation_count() -> None:
    """The None case is distinct from 0 — Paper.citation_count=None
    means 'this source didn't report it' (the scorer warns), while 0
    means 'cited zero times' (the scorer dings impact). The codec
    must keep them distinguishable across the roundtrip."""
    p = Paper(
        id="arxiv:9", title="t", abstract="", authors=(Author("X"),),
        citation_count=None,
    )
    decoded = paper_from_dict(paper_to_dict(p))
    assert decoded.citation_count is None
