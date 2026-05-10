"""HuggingFaceCrossEncoderReranker tests against the real model.

Gated by `RESEARCH_MCP_RERANKER_TESTS=1` because:
  - First run downloads ~250 MB of BAAI/bge-reranker-base weights.
  - Each test invocation runs cross-encoder inference (CPU: ~100ms).
  - sentence-transformers + faiss in the same process can SIGABRT on
    macOS without `KMP_DUPLICATE_LIB_OK=TRUE` (we set this at package
    import in research_mcp/__init__.py — but the gate also keeps CI
    runs cheap).

Skip on every default unit run; opt in deliberately when verifying
the model behavior end-to-end.
"""

from __future__ import annotations

import os

import pytest

from research_mcp.domain.paper import Author, Paper

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        os.environ.get("RESEARCH_MCP_RERANKER_TESTS") != "1",
        reason=(
            "set RESEARCH_MCP_RERANKER_TESTS=1 to download "
            "BAAI/bge-reranker-base and exercise real cross-encoder scoring "
            "(~250 MB on first run)"
        ),
    ),
]


async def test_real_cross_encoder_separates_relevant_from_off_topic() -> None:
    """The smallest end-to-end claim: feed the real bge-reranker-base a
    query plus two candidate Papers — one obviously relevant, one
    obviously not — and assert the relevant paper scores higher.

    No specific score thresholds — cross-encoders return raw logits whose
    range varies between models; we just sort and check the ordering.
    """
    from research_mcp.reranker import HuggingFaceCrossEncoderReranker

    relevant = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract=(
            "We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence "
            "and convolutions entirely."
        ),
        authors=(Author("Ashish Vaswani"),),
    )
    off_topic = Paper(
        id="x:housing",
        title="Linear regression for housing price prediction",
        abstract=(
            "We benchmark linear regression baselines on the Boston housing "
            "dataset and discuss feature engineering for tabular data."
        ),
        authors=(Author("Anonymous"),),
    )
    reranker = HuggingFaceCrossEncoderReranker()
    scores = await reranker.score(
        "transformer architecture for sequence modeling",
        [off_topic, relevant],
    )
    assert scores[1] > scores[0], (
        f"expected the transformer paper to outscore housing-prices; "
        f"got off_topic={scores[0]:.3f}, relevant={scores[1]:.3f}"
    )
