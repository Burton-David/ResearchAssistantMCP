"""Tests for `_select_openalex`, the env-driven OpenAlex wiring."""

from __future__ import annotations

import pytest

from research_mcp.mcp.server import _select_openalex

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_openalex_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_MCP_OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_MCP_OPENALEX_EMAIL", raising=False)


async def test_no_settings_still_builds_source_for_citation_graph_tools() -> None:
    """find_referenced_by and find_related only do single-work lookups, which
    cost zero OpenAlex credits, so they work with nothing configured. Search
    stays opt-in because every search page costs credits."""
    source, in_search = _select_openalex()
    try:
        assert in_search is False
        assert source._headers == {}
        assert source._email is None
    finally:
        await source.aclose()


async def test_api_key_setting_becomes_bearer_header_and_enables_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_MCP_OPENALEX_API_KEY", " oa-key-123 ")
    source, in_search = _select_openalex()
    try:
        assert in_search is True
        assert source._headers == {"Authorization": "Bearer oa-key-123"}
    finally:
        await source.aclose()


async def test_legacy_email_setting_still_enables_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configs written before the polite pool was retired only set the email.
    Those users had OpenAlex in search and should keep it."""
    monkeypatch.setenv("RESEARCH_MCP_OPENALEX_EMAIL", "you@lab.edu")
    source, in_search = _select_openalex()
    try:
        assert in_search is True
        assert source._email == "you@lab.edu"
        assert source._headers == {}
    finally:
        await source.aclose()
