"""End-to-end MCP stdio test.

Boots `research-mcp serve` in test mode (`RESEARCH_MCP_TEST_MODE=1` so the
server uses FakeEmbedder + MemoryIndex and needs no API keys), opens an MCP
client session over stdio, and round-trips a representative subset of the
seven tools. Asserts the response shapes — the network may return zero
results, but every tool must produce a well-formed structured-content
payload, never raise on the wire.
"""

from __future__ import annotations

import os
import sys

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.e2e


def _server_params() -> StdioServerParameters:
    venv_bin = os.path.dirname(sys.executable)
    server_cmd = os.path.join(venv_bin, "research-mcp")
    if not os.path.exists(server_cmd):
        pytest.skip(f"research-mcp not installed at {server_cmd}")
    return StdioServerParameters(
        command=server_cmd,
        args=["serve"],
        env={**os.environ, "RESEARCH_MCP_TEST_MODE": "1"},
    )


async def test_lists_all_seven_tools() -> None:
    async with stdio_client(_server_params()) as (read, write), ClientSession(
        read, write
    ) as session:
        await session.initialize()
        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert {
            "search_papers",
            "ingest_paper",
            "library_search",
            "cite_paper",
            "library_status",
            "get_paper",
            "find_paper",
        } <= tool_names


async def test_search_papers_round_trip() -> None:
    async with stdio_client(_server_params()) as (read, write), ClientSession(
        read, write
    ) as session:
        await session.initialize()
        result = await session.call_tool(
            "search_papers",
            {"query": "transformer attention", "max_results": 2},
        )
        assert not result.isError, f"tool call errored: {result}"
        structured = result.structuredContent
        assert isinstance(structured, dict)
        assert "results" in structured
        assert isinstance(structured["results"], list)
        # Each result carries a source provenance string.
        for hit in structured["results"]:
            assert "source" in hit
            assert isinstance(hit["source"], str)


async def test_get_paper_round_trip() -> None:
    """get_paper resolves arXiv ids without ingesting; works in test mode
    because the in-memory wiring keeps the arXiv source live."""
    async with stdio_client(_server_params()) as (read, write), ClientSession(
        read, write
    ) as session:
        await session.initialize()
        result = await session.call_tool(
            "get_paper", {"paper_id": "arxiv:1706.03762"}
        )
        assert not result.isError, f"tool call errored: {result}"
        structured = result.structuredContent
        assert isinstance(structured, dict)
        assert structured["paper"]["id"] == "arxiv:1706.03762"
        assert structured["paper"]["source"] == "arxiv"


async def test_library_status_reports_test_mode_embedder() -> None:
    async with stdio_client(_server_params()) as (read, write), ClientSession(
        read, write
    ) as session:
        await session.initialize()
        result = await session.call_tool("library_status", {})
        assert not result.isError
        structured = result.structuredContent
        assert isinstance(structured, dict)
        assert structured["count"] == 0  # fresh in-memory library
        assert structured["embedder"] == "fake:test-mode"
