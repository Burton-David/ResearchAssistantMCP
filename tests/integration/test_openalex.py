"""Integration tests against the live OpenAlex API.

Gated by `RESEARCH_MCP_INTEGRATION=1`. Picks up
`RESEARCH_MCP_OPENALEX_EMAIL` if set; falls back to a generic test email
so the gated suite still runs on a fresh checkout.
"""

from __future__ import annotations

import os

import pytest

from research_mcp.domain.query import SearchQuery
from research_mcp.sources.openalex import OpenAlexSource

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RESEARCH_MCP_INTEGRATION") != "1",
        reason="set RESEARCH_MCP_INTEGRATION=1 to run integration tests",
    ),
]


_TEST_EMAIL = os.environ.get("RESEARCH_MCP_OPENALEX_EMAIL", "research-mcp-test@example.com")


async def test_openalex_search_returns_results_for_attention_query(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = OpenAlexSource(email=_TEST_EMAIL, cache_dir=tmp_path / "openalex")
    try:
        results = await src.search(
            SearchQuery(text="attention is all you need transformer", max_results=5),
        )
    finally:
        await src.aclose()
    assert results, "expected non-empty OpenAlex search results"
    assert all(p.id.startswith("openalex:") for p in results)


async def test_openalex_fetch_by_known_openalex_id(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """W2626778328 is OpenAlex's record for `Attention Is All You Need`."""
    src = OpenAlexSource(email=_TEST_EMAIL, cache_dir=tmp_path / "openalex")
    try:
        paper = await src.fetch("openalex:W2626778328")
    finally:
        await src.aclose()
    assert paper is not None
    assert paper.id == "openalex:W2626778328"
    assert "Attention" in paper.title


async def test_openalex_fetch_unknown_id_returns_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """OpenAlex returns 404 for nonexistent ids; we surface that as None,
    not as a transient SourceUnavailable."""
    src = OpenAlexSource(email=_TEST_EMAIL, cache_dir=tmp_path / "openalex")
    try:
        paper = await src.fetch("openalex:W9999999999")
    finally:
        await src.aclose()
    assert paper is None
