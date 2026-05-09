"""AMA renderer golden-file tests.

The Vaswani output is the anchor — if this assertion changes, the change is
visible in the diff and warrants a deliberate review of the AMA format choice.
"""

from __future__ import annotations

import pytest

from research_mcp.citation.ama import AMARenderer
from research_mcp.domain.citation import CitationFormat, CitationRenderer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def test_protocol_conformance() -> None:
    assert isinstance(AMARenderer(), CitationRenderer)
    assert AMARenderer().format == CitationFormat.AMA


def test_vaswani_golden(vaswani_paper: Paper) -> None:
    expected = (
        "Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. "
        "Attention Is All You Need. *arXiv*. 2017. arXiv:1706.03762. "
        "https://arxiv.org/abs/1706.03762"
    )
    assert AMARenderer().render(vaswani_paper) == expected


def test_renders_with_doi_when_present() -> None:
    paper = Paper(
        id="doi:10.1/X",
        title="Title.",
        abstract="",
        authors=(Author("Smith"),),
        doi="10.1/X",
    )
    out = AMARenderer().render(paper)
    assert "doi:10.1/X" in out
    assert "arXiv" not in out


def test_renders_empty_paper_without_raising() -> None:
    p = Paper(id="x:1", title="", abstract="", authors=())
    assert AMARenderer().render(p) == ""


def test_more_than_six_authors_collapses_to_et_al() -> None:
    authors = tuple(Author(f"X Author{i}") for i in range(8))
    p = Paper(id="x:1", title="T", abstract="", authors=authors)
    out = AMARenderer().render(p)
    assert "et al" in out
    assert out.count(",") <= 6  # 6 authors max, then "et al"
