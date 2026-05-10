"""Chunker tests — section detection, sliding window, fallback paths."""

from __future__ import annotations

import pytest

from research_mcp.chunker import FakeChunker, SectionAwareChunker, SimpleChunker
from research_mcp.chunker._text import paper_text, sliding_windows
from research_mcp.domain import Chunker
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


def _paper(
    *,
    title: str = "t",
    abstract: str = "a",
    full_text: str | None = None,
) -> Paper:
    return Paper(
        id="x:1",
        title=title,
        abstract=abstract,
        authors=(Author("X"),),
        full_text=full_text,
    )


# ---- protocol conformance ----


def test_section_aware_satisfies_protocol() -> None:
    assert isinstance(SectionAwareChunker(), Chunker)


def test_simple_satisfies_protocol() -> None:
    assert isinstance(SimpleChunker(), Chunker)


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeChunker(), Chunker)


# ---- paper_text helper ----


def test_paper_text_prefers_full_text() -> None:
    p = _paper(title="t", abstract="a", full_text="this is the body")
    assert paper_text(p) == "this is the body"


def test_paper_text_falls_back_to_title_abstract() -> None:
    p = _paper(title="hello", abstract="world")
    assert paper_text(p) == "hello\n\nworld"


def test_paper_text_empty_paper_returns_empty() -> None:
    p = _paper(title="", abstract="")
    assert paper_text(p) == ""


# ---- sliding_windows ----


def test_sliding_windows_yields_chunks_under_size_in_one_piece() -> None:
    text = "Short text."
    chunks = list(sliding_windows(text, chunk_chars=100, overlap_chars=10))
    assert len(chunks) == 1
    assert chunks[0] == ("Short text.", 0, len(text))


def test_sliding_windows_splits_long_text() -> None:
    text = ("one. " * 200).strip()  # ~1000 chars
    chunks = list(sliding_windows(text, chunk_chars=300, overlap_chars=50))
    assert len(chunks) >= 3
    # Each chunk's text should match the slice indicated by start/end.
    for chunk_text, start, end in chunks:
        assert text[start:end] == chunk_text


def test_sliding_windows_overlap_repeats_tail() -> None:
    from itertools import pairwise

    text = "abcdefghij" * 50  # 500 chars, no sentence boundaries
    chunks = list(sliding_windows(text, chunk_chars=100, overlap_chars=20))
    assert len(chunks) > 2
    # Adjacent chunks should share their overlap region (with the caveat
    # that sliding_windows tries to break at sentence boundaries; with no
    # sentence terminators in this input, the break is just the hard size).
    for prev, cur in pairwise(chunks):
        _, _, prev_end = prev
        _, cur_start, _ = cur
        assert cur_start < prev_end


def test_sliding_windows_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        list(sliding_windows("abc", chunk_chars=10, overlap_chars=10))
    with pytest.raises(ValueError):
        list(sliding_windows("abc", chunk_chars=10, overlap_chars=-1))


def test_sliding_windows_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        list(sliding_windows("abc", chunk_chars=0, overlap_chars=0))


# ---- SectionAwareChunker ----


async def test_section_aware_falls_back_to_abstract_for_title_only() -> None:
    p = _paper(title="Attention Is All You Need", abstract="seq2seq transduction")
    chunks = list(await SectionAwareChunker().chunk(p))
    assert len(chunks) == 1
    assert chunks[0].section == "abstract"
    assert "Attention" in chunks[0].text


async def test_section_aware_returns_empty_for_empty_paper() -> None:
    p = _paper(title="", abstract="")
    chunks = list(await SectionAwareChunker().chunk(p))
    assert chunks == []


async def test_section_aware_detects_markdown_headers() -> None:
    full_text = (
        "## Abstract\nWe propose a new transformer architecture.\n\n"
        "## Introduction\nThe dominant sequence transduction models...\n\n"
        "## Methodology\nWe use scaled dot-product attention.\n\n"
        "## Results\nBLEU score of 28.4 on WMT 2014 EN-DE.\n"
    )
    p = _paper(full_text=full_text)
    chunks = list(await SectionAwareChunker().chunk(p))
    sections = {c.section for c in chunks}
    assert "abstract" in sections
    assert "introduction" in sections
    assert "methodology" in sections
    assert "results" in sections


async def test_section_aware_detects_numbered_headers() -> None:
    full_text = (
        "1. Introduction\nThis paper introduces a new method.\n\n"
        "2. Methodology\nWe build on prior work in attention.\n\n"
        "3. Results\nState-of-the-art on three benchmarks.\n"
    )
    p = _paper(full_text=full_text)
    chunks = list(await SectionAwareChunker().chunk(p))
    sections = {c.section for c in chunks}
    assert {"introduction", "methodology", "results"} <= sections


async def test_section_aware_unmarked_full_text_falls_back_to_body() -> None:
    """Full-text without recognizable section headers chunks as 'body'."""
    p = _paper(full_text="This is a 5000-character body. " * 200)
    chunks = list(await SectionAwareChunker().chunk(p))
    assert len(chunks) >= 1
    assert all(c.section == "body" for c in chunks)


async def test_section_aware_splits_long_section_via_sliding_window() -> None:
    long_methodology = "We use attention. " * 500  # ~9000 chars
    full_text = f"## Methodology\n{long_methodology}"
    p = _paper(full_text=full_text)
    chunks = list(await SectionAwareChunker(max_chunk_chars=2000).chunk(p))
    methodology_chunks = [c for c in chunks if c.section == "methodology"]
    assert len(methodology_chunks) >= 4  # ~9000 / 2000


async def test_section_aware_chunk_id_format() -> None:
    p = _paper(full_text="## Abstract\nbrief.\n\n## Methodology\nstuff.\n")
    chunks = list(await SectionAwareChunker().chunk(p))
    for c in chunks:
        # paper_id#section#index
        assert c.chunk_id.startswith(f"{p.id}#")
        assert c.section in c.chunk_id


# ---- SimpleChunker ----


async def test_simple_one_chunk_for_short_text() -> None:
    p = _paper(title="t", abstract="a")
    chunks = list(await SimpleChunker(max_chunk_chars=2000).chunk(p))
    assert len(chunks) == 1
    assert chunks[0].section is None


async def test_simple_multiple_chunks_for_long_text() -> None:
    p = _paper(full_text="abc " * 1500)  # ~6000 chars
    chunks = list(await SimpleChunker(max_chunk_chars=1500, overlap_chars=100).chunk(p))
    assert len(chunks) >= 3


# ---- FakeChunker ----


async def test_fake_returns_one_chunk_with_full_text() -> None:
    p = _paper(title="hello", abstract="world")
    chunks = list(await FakeChunker().chunk(p))
    assert len(chunks) == 1
    assert chunks[0].section == "abstract"
    assert "hello" in chunks[0].text
