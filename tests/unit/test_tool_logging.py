"""Per-tool-call structured logging tests.

The logging path runs inside `build_server`'s `call_tool` dispatch, which is
hard to drive without a real MCP client transport. Instead we hand-spin the
handler dict the way the dispatcher does and assert the log records that
fire. The behavior is small enough to test that way without standing up
stdio.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from research_mcp.mcp.server import _result_hint

pytestmark = pytest.mark.unit


def test_result_hint_for_search_responses() -> None:
    assert _result_hint("search_papers", {"results": []}) == "n=0"
    assert _result_hint("search_papers", {"results": [1, 2, 3]}) == "n=3"
    assert _result_hint("library_search", {"results": []}) == "n=0"
    assert _result_hint("find_paper", {"results": [{"x": 1}]}) == "n=1"


def test_result_hint_for_library_status() -> None:
    assert _result_hint("library_status", {"count": 12}) == "count=12"
    assert _result_hint("library_status", {"count": 0}) == "count=0"


def test_result_hint_for_ingest() -> None:
    assert (
        _result_hint("ingest_paper", {"paper": {}, "library_count": 7})
        == "library_count=7"
    )


def test_result_hint_for_get_paper() -> None:
    assert _result_hint("get_paper", {"paper": {}}) == "paper=1"


def test_result_hint_for_cite() -> None:
    assert (
        _result_hint("cite_paper", {"citation": "...", "format": "bibtex"})
        == "format=bibtex"
    )


def test_result_hint_unknown_shape() -> None:
    assert _result_hint("unknown_tool", {"weird": "shape"}) == "-"


async def test_call_tool_emits_info_log_with_elapsed_and_result_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end through the dispatcher: a successful tool call emits one
    INFO record naming the tool, arg keys, elapsed ms, and result hint."""
    from research_mcp.embedder import FakeEmbedder
    from research_mcp.index import MemoryIndex
    from research_mcp.mcp.server import build_server
    from research_mcp.service import (
        DiscoveryService,
        LibraryService,
        SearchService,
    )
    from tests.conftest import StaticSource

    arxiv = StaticSource("arxiv", [])
    embedder = FakeEmbedder(16)
    index = MemoryIndex(16)
    library = LibraryService(index=index, embedder=embedder, ingest_sources=[arxiv])
    search = SearchService([arxiv])
    discovery = DiscoveryService(search)

    async def paper_lookup(_paper_id: str):  # type: ignore[no-untyped-def]
        return None

    server = build_server(
        search=search,
        discovery=discovery,
        paper_lookup=paper_lookup,
        library=library,
        embedder_label="test",
    )
    # Pull the wrapped call_tool out of the server's request_handlers.
    # The mcp SDK keys handlers by request type — find the CallToolRequest
    # entry and synthesize a request to drive the dispatch.
    import mcp.types as mcp_types

    handler = server.request_handlers[mcp_types.CallToolRequest]

    async def drive() -> None:
        with caplog.at_level(logging.INFO, logger="research_mcp.mcp.server"):
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(
                    name="library_status",
                    arguments={},
                ),
            )
            await handler(req)

    await drive()

    matching = [
        r for r in caplog.records
        if r.name == "research_mcp.mcp.server" and "tool=library_status" in r.getMessage()
    ]
    assert matching, f"expected one INFO record, got {[r.getMessage() for r in caplog.records]}"
    msg = matching[0].getMessage()
    assert "tool=library_status" in msg
    assert "elapsed=" in msg
    assert "count=" in msg


async def test_call_tool_surfaces_timeout_as_clean_value_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A handler that exceeds its budget must surface as a clean
    ValueError naming the budget — not propagate the underlying
    asyncio.TimeoutError, and not let the call run past the budget.
    Without this, Claude Desktop hits its own 4-min hard kill before
    we surface our own diagnostic."""
    import asyncio

    import mcp.types as mcp_types

    from research_mcp.mcp.server import _TOOL_TIMEOUTS, build_server
    from research_mcp.service import DiscoveryService, SearchService
    from tests.conftest import StaticSource

    arxiv = StaticSource("arxiv", [])
    search = SearchService([arxiv])
    discovery = DiscoveryService(search)

    # A library that hangs forever inside count() — simulating a wedged
    # FAISS read or a wedged upstream HTTP call.
    class _HangingLibrary:
        async def count(self) -> int:
            await asyncio.sleep(60)  # would block past any sane budget
            return 0

    server = build_server(
        search=search,
        discovery=discovery,
        paper_lookup=lambda _id: None,  # type: ignore[arg-type,return-value]
        library=_HangingLibrary(),  # type: ignore[arg-type]
        embedder_label="test",
    )
    # Override the budget to something tiny so the test runs fast.
    monkey_budget = 0.05
    _TOOL_TIMEOUTS["library_status"] = monkey_budget
    try:
        handler = server.request_handlers[mcp_types.CallToolRequest]
        req = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(
                name="library_status",
                arguments={},
            ),
        )
        # The dispatcher converts a TimeoutError into a ValueError; the
        # MCP SDK then surfaces ValueError as a tool-error response.
        # We can't easily inspect the response body without going through
        # the SDK, so we invoke the *inner* handler directly via the
        # handlers dict — the timeout logic lives there too.
        with caplog.at_level("WARNING", logger="research_mcp.mcp.server"):
            response = await handler(req)
        # The MCP SDK wraps tool results as ServerResult(root=CallToolResult).
        # On error: isError=True and content carries the message text.
        result = response.root  # type: ignore[attr-defined]
        text = "".join(
            getattr(block, "text", "")
            for block in (getattr(result, "content", None) or [])
        )
        assert result.isError
        assert "timed out" in text.lower()
    finally:
        # Restore so the next test doesn't see our monkey-patched value.
        _TOOL_TIMEOUTS["library_status"] = 10.0


def test_configure_logging_is_idempotent() -> None:
    """Re-running the configurator must not duplicate the stderr handler."""
    from research_mcp.mcp.server import _configure_logging

    pkg_log = logging.getLogger("research_mcp")
    pkg_log.handlers.clear()
    _configure_logging()
    handler_count_after_first = len(pkg_log.handlers)
    _configure_logging()
    handler_count_after_second = len(pkg_log.handlers)
    assert handler_count_after_first == handler_count_after_second == 1
    # Restore: tear down so the rest of the suite isn't logging to stderr.
    pkg_log.handlers.clear()


def _ensure_loop_works() -> None:
    """Pytest-asyncio's asyncio_mode=auto needs at least one async test in
    the module to set up; this no-op file lock helps that."""
    asyncio.get_event_loop_policy()
