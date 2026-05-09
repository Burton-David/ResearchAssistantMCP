"""BibTeX renderer tests."""

from __future__ import annotations

import pytest

from research_mcp.citation.bibtex import BibtexRenderer
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def test_arxiv_paper_emits_misc_with_eprint(vaswani_paper: Paper) -> None:
    out = BibtexRenderer().render(vaswani_paper)
    assert out.startswith("@misc{")
    assert "vaswani2017" in out
    assert "eprint = {1706.03762}" in out
    assert "archivePrefix = {arXiv}" in out


def test_journal_paper_emits_article() -> None:
    p = Paper(
        id="doi:1",
        title="Title",
        abstract="",
        authors=(Author("Curie Marie"),),
        venue="Nature",
        doi="10.1/X",
    )
    out = BibtexRenderer().render(p)
    assert out.startswith("@article{")
    assert "journal = {Nature}" in out
    assert "doi = {10.1/X}" in out


def test_special_characters_escaped() -> None:
    p = Paper(id="x:1", title="A & B {percent} %", abstract="", authors=())
    out = BibtexRenderer().render(p)
    # Each special char appears in escaped form, not raw inside the braces.
    assert r"A \& B" in out
    assert r"\%" in out


def test_renders_empty_paper_without_raising() -> None:
    BibtexRenderer().render(Paper(id="x:1", title="", abstract="", authors=()))
