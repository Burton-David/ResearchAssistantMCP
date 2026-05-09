"""Integration tests against the live arXiv API.

Gated by `RESEARCH_MCP_INTEGRATION=1` because they hit the network and are
subject to upstream rate limits / outages.
"""

from __future__ import annotations

import os

import pytest

from research_mcp.domain.query import SearchQuery
from research_mcp.sources.arxiv import ArxivSource

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RESEARCH_MCP_INTEGRATION") != "1",
        reason="set RESEARCH_MCP_INTEGRATION=1 to run integration tests",
    ),
]


async def test_arxiv_returns_vaswani_for_attention_query(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # arXiv's free-text relevance ranking does not always surface the canonical
    # paper for the bare "attention is all you need" query (recent derivative
    # papers with the same title pattern crowd it out), so we anchor the query
    # with author names to exercise the multi-field search path.
    src = ArxivSource(cache_dir=tmp_path / "arxiv")
    try:
        results = await src.search(
            SearchQuery(
                text="Attention Is All You Need Vaswani Shazeer Parmar",
                max_results=5,
            )
        )
    finally:
        await src.aclose()
    assert results, "expected non-empty search results"
    titles = " | ".join(p.title for p in results)
    assert any(
        p.arxiv_id == "1706.03762" for p in results
    ), f"expected Vaswani et al. (1706.03762) in results; got {titles}"


async def test_arxiv_fetch_by_id_returns_canonical_paper(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = ArxivSource(cache_dir=tmp_path / "arxiv")
    try:
        paper = await src.fetch("arxiv:1706.03762")
    finally:
        await src.aclose()
    assert paper is not None
    assert paper.id == "arxiv:1706.03762"
    assert paper.arxiv_id == "1706.03762"
    assert "Attention" in paper.title
