"""Integration tests for the LLM claim extractors against real APIs.

Each test is gated by both `RESEARCH_MCP_INTEGRATION=1` and the
provider-specific API key. Costs <$0.001 per run on cheap default
models (gpt-4o-mini / claude-haiku).
"""

from __future__ import annotations

import os

import pytest

from research_mcp.claim_extractor import (
    AnthropicLLMClaimExtractor,
    OpenAILLMClaimExtractor,
)
from research_mcp.domain.claim import ClaimType

pytestmark = pytest.mark.integration


_TEST_TEXT = (
    "Recent transformer models have outperformed LSTMs by 23% on machine "
    "translation tasks. The proposed approach achieves a BLEU score of "
    "28.4 on WMT 2014 EN-DE."
)


@pytest.mark.skipif(
    os.environ.get("RESEARCH_MCP_INTEGRATION") != "1"
    or not os.environ.get("OPENAI_API_KEY"),
    reason=(
        "set RESEARCH_MCP_INTEGRATION=1 + OPENAI_API_KEY to run real-API "
        "claim-extractor tests"
    ),
)
async def test_openai_extractor_finds_comparative_and_statistical_claims() -> None:
    extractor = OpenAILLMClaimExtractor()
    claims = list(await extractor.extract(_TEST_TEXT))
    assert claims, "expected non-empty claim list"
    types = {c.type for c in claims}
    # 4o-mini reliably tags the outperforms-by-23% sentence comparative-
    # or-statistical (or both, as overlapping spans). At minimum, one of
    # the two claim types should be present.
    assert ClaimType.COMPARATIVE in types or ClaimType.STATISTICAL in types
    # Span anchoring: every claim with a non-default offset must
    # round-trip to its text in the input.
    for c in claims:
        if c.start_char != 0 or c.end_char != 0:
            assert _TEST_TEXT[c.start_char : c.end_char] == c.text


@pytest.mark.skipif(
    os.environ.get("RESEARCH_MCP_INTEGRATION") != "1"
    or not os.environ.get("ANTHROPIC_API_KEY"),
    reason=(
        "set RESEARCH_MCP_INTEGRATION=1 + ANTHROPIC_API_KEY to run real-API "
        "claim-extractor tests"
    ),
)
async def test_anthropic_extractor_finds_comparative_and_statistical_claims() -> None:
    extractor = AnthropicLLMClaimExtractor()
    claims = list(await extractor.extract(_TEST_TEXT))
    assert claims
    types = {c.type for c in claims}
    assert ClaimType.COMPARATIVE in types or ClaimType.STATISTICAL in types
    for c in claims:
        if c.start_char != 0 or c.end_char != 0:
            assert _TEST_TEXT[c.start_char : c.end_char] == c.text
