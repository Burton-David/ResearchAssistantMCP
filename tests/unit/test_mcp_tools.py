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


def test_search_papers_rejects_whitespace_only_query() -> None:
    """Pure whitespace passes min_length=1 — we want to reject it explicitly."""
    with pytest.raises(ValidationError) as exc:
        SearchPapersInput.model_validate({"query": "   \n\t  "})
    assert "non-whitespace" in str(exc.value)


def test_library_search_rejects_whitespace_only_query() -> None:
    from research_mcp.mcp.tools import LibrarySearchInput

    with pytest.raises(ValidationError):
        LibrarySearchInput.model_validate({"query": "   "})


def test_year_max_below_year_min_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        SearchPapersInput.model_validate(
            {"query": "x", "year_min": 2024, "year_max": 2018}
        )
    assert "year_min" in str(exc.value)


def test_year_equal_min_max_accepted() -> None:
    parsed = SearchPapersInput.model_validate(
        {"query": "x", "year_min": 2020, "year_max": 2020}
    )
    assert parsed.year_min == 2020 and parsed.year_max == 2020


def test_max_results_capped_at_100() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": "x", "max_results": 9999})


def test_year_min_accepts_string_int() -> None:
    """Models often serialize numeric tool args as strings — coerce, don't reject.

    The pydantic default lax mode handles this; we disable the mcp SDK's
    jsonschema layer (which doesn't coerce) so this path actually runs.
    """
    parsed = SearchPapersInput.model_validate(
        {"query": "BERT", "year_min": "2018", "year_max": "2020"}
    )
    assert parsed.year_min == 2018
    assert parsed.year_max == 2020


def test_year_min_accepts_real_int() -> None:
    parsed = SearchPapersInput.model_validate({"query": "BERT", "year_min": 2018})
    assert parsed.year_min == 2018


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


def test_library_status_input_takes_no_args() -> None:
    from research_mcp.mcp.tools import LibraryStatusInput

    parsed = LibraryStatusInput.model_validate({})
    assert parsed is not None


def test_library_status_input_rejects_unknown_field() -> None:
    from research_mcp.mcp.tools import LibraryStatusInput

    with pytest.raises(ValidationError):
        LibraryStatusInput.model_validate({"unexpected": "x"})
