"""Integration tests for the LLM citation scorers against real APIs.

Each test is gated by both `RESEARCH_MCP_INTEGRATION=1` and the
provider-specific API key. Costs <$0.001 per run on cheap default
models (gpt-4o-mini / claude-haiku).
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from research_mcp.citation_scorer import (
    AnthropicLLMCitationScorer,
    HeuristicCitationScorer,
    OpenAILLMCitationScorer,
)
from research_mcp.domain.claim import Claim, ClaimType
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.integration


def _vaswani() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract=(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks that include an "
            "encoder and a decoder. The best performing models also connect "
            "the encoder and decoder through an attention mechanism. We "
            "propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence "
            "and convolutions entirely."
        ),
        authors=(Author("Ashish Vaswani"), Author("Noam Shazeer")),
        published=date(2017, 6, 12),
        venue="NeurIPS",
        citation_count=80000,
        arxiv_id="1706.03762",
    )


def _claim(text: str) -> Claim:
    return Claim(
        text=text,
        type=ClaimType.METHODOLOGICAL,
        confidence=0.9,
        context=text,
        suggested_search_terms=("transformer", "attention"),
    )


@pytest.mark.skipif(
    os.environ.get("RESEARCH_MCP_INTEGRATION") != "1"
    or not os.environ.get("OPENAI_API_KEY"),
    reason=(
        "set RESEARCH_MCP_INTEGRATION=1 + OPENAI_API_KEY to run real-API "
        "citation-scorer tests"
    ),
)
async def test_openai_distinguishes_relevant_from_unrelated_claim() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    scorer = OpenAILLMCitationScorer(base_scorer=base)

    relevant = await scorer.score(
        _vaswani(), _claim("the Transformer relies entirely on attention")
    )
    unrelated = await scorer.score(
        _vaswani(), _claim("metformin reduces HbA1c in type 2 diabetes")
    )

    assert relevant.total > unrelated.total
    assert "low semantic relevance to claim" in unrelated.warnings


@pytest.mark.skipif(
    os.environ.get("RESEARCH_MCP_INTEGRATION") != "1"
    or not os.environ.get("ANTHROPIC_API_KEY"),
    reason=(
        "set RESEARCH_MCP_INTEGRATION=1 + ANTHROPIC_API_KEY to run real-API "
        "citation-scorer tests"
    ),
)
async def test_anthropic_distinguishes_relevant_from_unrelated_claim() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    scorer = AnthropicLLMCitationScorer(base_scorer=base)

    relevant = await scorer.score(
        _vaswani(), _claim("the Transformer relies entirely on attention")
    )
    unrelated = await scorer.score(
        _vaswani(), _claim("metformin reduces HbA1c in type 2 diabetes")
    )

    assert relevant.total > unrelated.total
    assert "low semantic relevance to claim" in unrelated.warnings
