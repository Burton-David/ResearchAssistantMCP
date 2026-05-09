"""End-to-end MCP stdio test.

Boots `research-mcp serve` in test mode (`RESEARCH_MCP_TEST_MODE=1` so the
server uses FakeEmbedder + MemoryIndex and needs no API keys), opens an MCP
client session over stdio, lists tools, and calls `search_papers`. Asserts
the response shape — the network may return zero arXiv hits, but the tool
must produce a well-formed structured-content payload either way.
"""

from __future__ import annotations

import os
import sys

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.e2e


async def test_search_papers_round_trip() -> None:
    venv_bin = os.path.dirname(sys.executable)
    server_cmd = os.path.join(venv_bin, "research-mcp")
    if not os.path.exists(server_cmd):
        pytest.skip(f"research-mcp not installed at {server_cmd}")

    params = StdioServerParameters(
        command=server_cmd,
        args=["serve"],
        env={**os.environ, "RESEARCH_MCP_TEST_MODE": "1"},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert {"search_papers", "ingest_paper", "library_search", "cite_paper"} <= tool_names

        result = await session.call_tool(
            "search_papers",
            {"query": "transformer attention", "max_results": 2},
        )
        assert not result.isError, f"tool call errored: {result}"
        structured = result.structuredContent
        assert isinstance(structured, dict)
        assert "results" in structured
        assert isinstance(structured["results"], list)
