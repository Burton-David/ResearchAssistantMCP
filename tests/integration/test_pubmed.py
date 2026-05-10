"""Integration tests against the live NCBI E-utilities API.

Gated by `RESEARCH_MCP_INTEGRATION=1`. Picks up `NCBI_API_KEY` and
`NCBI_EMAIL` from the environment when present; works without either,
just slower.
"""

from __future__ import annotations

import os

import pytest

from research_mcp.domain.query import SearchQuery
from research_mcp.sources.pubmed import PubMedSource

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RESEARCH_MCP_INTEGRATION") != "1",
        reason="set RESEARCH_MCP_INTEGRATION=1 to run integration tests",
    ),
]


async def test_pubmed_returns_results_for_diabetes_query(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = PubMedSource(cache_dir=tmp_path / "pubmed")
    try:
        results = await src.search(
            SearchQuery(text="metformin type 2 diabetes", max_results=5),
        )
    finally:
        await src.aclose()
    assert results, "expected non-empty PubMed results for metformin/T2D"
    assert all(p.id.startswith("pmid:") for p in results)
    assert all(p.title for p in results)


async def test_pubmed_fetch_canonical_brca1_paper(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """PMID 7545954 — the original BRCA1 paper (Miki et al., 1994). Stable
    record; safe to anchor an integration test on."""
    src = PubMedSource(cache_dir=tmp_path / "pubmed")
    try:
        paper = await src.fetch("pmid:7545954")
    finally:
        await src.aclose()
    assert paper is not None
    assert paper.id == "pmid:7545954"
    assert "BRCA1" in paper.title or "breast" in paper.title.lower()
    assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/7545954/"


async def test_pubmed_fetch_returns_none_for_arxiv_id(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A non-PubMed id must return None without an API call wasted."""
    src = PubMedSource(cache_dir=tmp_path / "pubmed")
    try:
        paper = await src.fetch("arxiv:1706.03762")
    finally:
        await src.aclose()
    assert paper is None
