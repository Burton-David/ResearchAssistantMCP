"""MLA renderer tests."""

from __future__ import annotations

import pytest

from research_mcp.citation.mla import MLARenderer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def test_vaswani_golden(vaswani_paper: Paper) -> None:
    expected = (
        'Vaswani, Ashish, et al. "Attention Is All You Need." '
        "*arXiv*, 2017, https://arxiv.org/abs/1706.03762."
    )
    assert MLARenderer().render(vaswani_paper) == expected


def test_two_authors_use_and() -> None:
    p = Paper(
        id="x:1",
        title="T",
        abstract="",
        authors=(Author("Alice Smith"), Author("Bob Jones")),
    )
    assert "Smith, Alice and Jones, Bob" in MLARenderer().render(p)


def test_renders_empty_paper_without_raising() -> None:
    MLARenderer().render(Paper(id="x:1", title="", abstract="", authors=()))
