"""APA renderer tests."""

from __future__ import annotations

import pytest

from research_mcp.citation.apa import APARenderer
from research_mcp.domain.citation import CitationFormat, CitationRenderer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def test_protocol_conformance() -> None:
    assert isinstance(APARenderer(), CitationRenderer)
    assert APARenderer().format == CitationFormat.APA


def test_vaswani_golden(vaswani_paper: Paper) -> None:
    expected = (
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., "
        "Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). "
        "Attention Is All You Need. *arXiv*. https://arxiv.org/abs/1706.03762"
    )
    assert APARenderer().render(vaswani_paper) == expected


def test_no_date_falls_back_to_n_d() -> None:
    p = Paper(id="x:1", title="T", abstract="", authors=(Author("Smith"),))
    assert "(n.d)." in APARenderer().render(p)


def test_renders_empty_paper_without_raising() -> None:
    p = Paper(id="x:1", title="", abstract="", authors=())
    APARenderer().render(p)
