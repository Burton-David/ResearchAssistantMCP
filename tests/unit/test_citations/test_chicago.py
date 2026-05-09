"""Chicago renderer tests."""

from __future__ import annotations

import pytest

from research_mcp.citation.chicago import ChicagoRenderer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def test_vaswani_golden(vaswani_paper: Paper) -> None:
    out = ChicagoRenderer().render(vaswani_paper)
    assert out.startswith("Vaswani, Ashish")
    assert " 2017. " in out
    assert '"Attention Is All You Need."' in out
    assert "*arXiv*" in out


def test_no_date_falls_back_to_n_d() -> None:
    p = Paper(id="x:1", title="T", abstract="", authors=(Author("Smith"),))
    assert "n.d." in ChicagoRenderer().render(p)


def test_renders_empty_paper_without_raising() -> None:
    ChicagoRenderer().render(Paper(id="x:1", title="", abstract="", authors=()))
