"""MCP tool input model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_mcp.mcp.tools import (
    CitePaperInput,
    IngestPaperInput,
    LibrarySearchInput,
    SearchPapersInput,
    paper_to_summary,
)

pytestmark = pytest.mark.unit


def test_search_papers_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": "x", "rogue": 1})


def test_search_papers_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": ""})


def test_max_results_capped_at_100() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": "x", "max_results": 9999})


def test_ingest_paper_input_requires_id() -> None:
    with pytest.raises(ValidationError):
        IngestPaperInput.model_validate({})


def test_library_search_default_k() -> None:
    parsed = LibrarySearchInput.model_validate({"query": "x"})
    assert parsed.k == 10


def test_cite_paper_default_format() -> None:
    parsed = CitePaperInput.model_validate({"paper_id": "arxiv:1"})
    assert parsed.format == "ama"


def test_cite_paper_unknown_format_rejected() -> None:
    with pytest.raises(ValidationError):
        CitePaperInput.model_validate({"paper_id": "arxiv:1", "format": "vancouver"})


def test_paper_to_summary_handles_minimal_paper(vaswani_paper) -> None:  # type: ignore[no-untyped-def]
    summary = paper_to_summary(vaswani_paper)
    assert summary.id == vaswani_paper.id
    assert summary.year == 2017
    assert summary.authors[0] == "Ashish Vaswani"
