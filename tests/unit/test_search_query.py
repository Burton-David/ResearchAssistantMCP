"""Quick coverage for SearchQuery edges + arxiv source string builder."""

from __future__ import annotations

import pytest

from research_mcp.domain.query import SearchQuery
from research_mcp.sources.arxiv import _build_search_string

pytestmark = pytest.mark.unit


def test_empty_query_falls_back_to_wildcard() -> None:
    s = _build_search_string(SearchQuery(text=""))
    assert s == "all:*"


def test_year_range_emits_submitted_date_filter() -> None:
    s = _build_search_string(SearchQuery(text="attention", year_min=2020, year_max=2023))
    assert "submittedDate:[" in s
    assert "all:attention" in s


def test_author_filter_uses_au_field() -> None:
    s = _build_search_string(SearchQuery(text="x", authors=("Vaswani", "Shazeer")))
    assert 'au:"Vaswani"' in s
    assert 'au:"Shazeer"' in s
