"""Boot-time regression guard: reranker must lazy-load its model.

Lives in its own file because `test_hf_cross_encoder_reranker.py` has
a module-level skipif that gates inference tests behind
RESEARCH_MCP_RERANKER_TESTS=1. This test runs UNCONDITIONALLY — it
doesn't touch the model, it just verifies the construction-time
contract.

History: a previous version of `HuggingFaceCrossEncoderReranker.__init__`
eager-loaded the model to fail fast on typos. With
RESEARCH_MCP_RERANKER=cross-encoder:BAAI/bge-reranker-base in the
environment, that load took 5-10 seconds (cached) or longer (cold
download). Claude Desktop's MCP initialize-handshake timeout was
shorter than that, so the server got killed before tool registration
completed — research-mcp wouldn't appear in the tool list even though
the binary, config, and code were all correct. Diagnosed during a QA
session by booting the binary with a minimal env and timing it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_reranker_construction_does_not_load_model() -> None:
    """Construction must NOT trigger sentence-transformers model load.
    `__init__` only stores config; `_load_model` runs on first
    `.score()` call."""
    from research_mcp.reranker import HuggingFaceCrossEncoderReranker

    rr = HuggingFaceCrossEncoderReranker("BAAI/bge-reranker-base")
    assert rr._model is None, (
        "HuggingFaceCrossEncoderReranker.__init__ must NOT load the "
        "model — eager-load blocks server boot past Claude Desktop's "
        "MCP handshake timeout. Lazy-load on first .score() instead."
    )
