"""LLM-based CitationScorer tests — OpenAI and Anthropic implementations.

The scorers wrap a base CitationScorer and adjust the total by an
LLM-judged relevance signal. We test by injecting stub clients (the
OpenAI / Anthropic SDKs are third-party — fair game to stub at the
boundary) and a real `HeuristicCitationScorer` as the base. No mocks
of CitationScorer (the protocol we own).
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from research_mcp.citation_scorer import (
    AnthropicLLMCitationScorer,
    HeuristicCitationScorer,
    OpenAILLMCitationScorer,
)
from research_mcp.citation_scorer._llm_schema import (
    RELEVANCE_SCHEMA,
    RELEVANCE_TOOL_NAME,
    payload_to_relevance,
    user_prompt,
)
from research_mcp.domain.citation_scorer import CitationScorer
from research_mcp.domain.claim import Claim, ClaimType
from research_mcp.domain.paper import Author, Paper

pytestmark = pytest.mark.unit


# ---- protocol conformance ----


def test_openai_satisfies_protocol() -> None:
    s = OpenAILLMCitationScorer(client=_StubOpenAIClient(payload=_HIGH_PAYLOAD))
    assert isinstance(s, CitationScorer)
    assert s.name.startswith("llm:openai:")


def test_anthropic_satisfies_protocol() -> None:
    s = AnthropicLLMCitationScorer(client=_StubAnthropicClient(payload=_HIGH_PAYLOAD))
    assert isinstance(s, CitationScorer)
    assert s.name.startswith("llm:anthropic:")


# ---- compose-on-base behavior ----


def _vaswani() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract=(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. We propose a new "
            "simple network architecture, the Transformer, based solely on "
            "attention mechanisms."
        ),
        authors=(Author("Ashish Vaswani"), Author("Noam Shazeer")),
        published=date(2017, 6, 12),
        venue="NeurIPS",
        citation_count=80000,
        arxiv_id="1706.03762",
    )


def _claim(text: str, claim_type: ClaimType = ClaimType.METHODOLOGICAL) -> Claim:
    return Claim(
        text=text,
        type=claim_type,
        confidence=0.9,
        context=text,
        suggested_search_terms=("transformer", "attention"),
    )


_HIGH_PAYLOAD = {"relevance": 0.9, "reasoning": "Paper introduces the Transformer."}
_LOW_PAYLOAD = {"relevance": 0.1, "reasoning": "Unrelated to this medical claim."}


async def test_no_claim_returns_base_score_unchanged() -> None:
    """The LLM scorer cannot judge relevance without a claim; it must
    fall back to the base score with no LLM call and no relevance
    factor or warning added."""
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    stub = _StubOpenAIClient(payload=_HIGH_PAYLOAD)
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)

    base_score = await base.score(_vaswani())
    llm_score = await scorer.score(_vaswani())

    assert llm_score.total == base_score.total
    assert "relevance" not in llm_score.factors
    assert stub.calls == 0


async def test_high_relevance_preserves_most_of_base_total() -> None:
    """Relevance modulates within a 0.5-1.0 band: at relevance=0.9 the
    total stays at 95% of base, well above the floor and above any
    plausibly low-relevance result on the same paper."""
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("attention is all you need"))

    stub = _StubOpenAIClient(payload=_HIGH_PAYLOAD)
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("attention is all you need"))

    assert llm_score.total == pytest.approx(base_score.total * 0.95, rel=1e-3)
    assert llm_score.total <= base_score.total
    assert "relevance" in llm_score.factors
    assert "Semantic fit: 0.90" in llm_score.factors["relevance"]
    assert "low semantic relevance to claim" not in llm_score.warnings


async def test_high_relevance_total_exceeds_low_relevance_total() -> None:
    """Same paper, two claims with different relevance: the higher-
    relevance result must rank above the lower-relevance one. This is
    the property find_citations actually uses (sort by total desc)."""
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    high_stub = _StubOpenAIClient(payload=_HIGH_PAYLOAD)
    low_stub = _StubOpenAIClient(payload=_LOW_PAYLOAD)
    high_scorer = OpenAILLMCitationScorer(base_scorer=base, client=high_stub)
    low_scorer = OpenAILLMCitationScorer(base_scorer=base, client=low_stub)

    high = await high_scorer.score(_vaswani(), _claim("transformers"))
    low = await low_scorer.score(_vaswani(), _claim("transformers"))
    assert high.total > low.total


async def test_low_relevance_drops_total_and_warns() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("metformin reduces HbA1c"))

    stub = _StubOpenAIClient(payload=_LOW_PAYLOAD)
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("metformin reduces HbA1c"))

    assert llm_score.total < base_score.total
    assert llm_score.total >= 0.0
    assert "low semantic relevance to claim" in llm_score.warnings


async def test_base_dimensions_are_preserved() -> None:
    """The four heuristic dimensions stay data-grounded; only the total
    is modulated. UI shows the breakdown so the user can see the
    venue/impact/author/recency reasoning unchanged."""
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubOpenAIClient(payload=_HIGH_PAYLOAD)
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.venue == base_score.venue
    assert llm_score.impact == base_score.impact
    assert llm_score.author == base_score.author
    assert llm_score.recency == base_score.recency


async def test_total_clamped_to_zero_hundred_band() -> None:
    """Even at relevance 1.0 the total cannot exceed 100; even at
    relevance 0.0 the total cannot fall below zero."""
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    scorer = OpenAILLMCitationScorer(
        base_scorer=base,
        client=_StubOpenAIClient(payload={"relevance": 1.0, "reasoning": "perfect"}),
    )
    score = await scorer.score(_vaswani(), _claim("transformers"))
    assert 0.0 <= score.total <= 100.0


# ---- failure modes: SDK exception, malformed JSON, no tool_use ----


async def test_openai_sdk_exception_falls_back_to_base() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubOpenAIClient(raise_exc=RuntimeError("503 upstream"))
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.total == base_score.total
    assert "relevance" not in llm_score.factors


async def test_openai_malformed_json_falls_back_to_base() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubOpenAIClient(raw_content="this is not json")
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.total == base_score.total
    assert "relevance" not in llm_score.factors


async def test_openai_missing_relevance_falls_back_to_base() -> None:
    """LLM returned valid JSON but without the relevance number — fall
    back rather than guessing a value."""
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubOpenAIClient(payload={"reasoning": "no relevance field"})
    scorer = OpenAILLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.total == base_score.total


async def test_anthropic_sdk_exception_falls_back_to_base() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubAnthropicClient(raise_exc=RuntimeError("503"))
    scorer = AnthropicLLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.total == base_score.total


async def test_anthropic_no_tool_use_block_falls_back_to_base() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubAnthropicClient(content=[])
    scorer = AnthropicLLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.total == base_score.total


async def test_anthropic_high_relevance_preserves_most_of_base_total() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("transformers"))

    stub = _StubAnthropicClient(payload=_HIGH_PAYLOAD)
    scorer = AnthropicLLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("transformers"))

    assert llm_score.total == pytest.approx(base_score.total * 0.95, rel=1e-3)
    assert llm_score.total <= base_score.total


async def test_anthropic_low_relevance_drops_total_and_warns() -> None:
    base = HeuristicCitationScorer(now=date(2026, 5, 10))
    base_score = await base.score(_vaswani(), _claim("metformin"))

    stub = _StubAnthropicClient(payload=_LOW_PAYLOAD)
    scorer = AnthropicLLMCitationScorer(base_scorer=base, client=stub)
    llm_score = await scorer.score(_vaswani(), _claim("metformin"))

    assert llm_score.total < base_score.total
    assert "low semantic relevance to claim" in llm_score.warnings


# ---- one API call per score ----


async def test_one_api_call_per_score() -> None:
    """Lock down: one score call → exactly one LLM call. A future
    'let's also call for self-critique' change should be deliberate."""
    stub = _StubOpenAIClient(payload=_HIGH_PAYLOAD)
    scorer = OpenAILLMCitationScorer(client=stub)
    await scorer.score(_vaswani(), _claim("transformers"))
    assert stub.calls == 1


# ---- prompt + schema invariants ----


def test_user_prompt_includes_paper_title_and_claim_text() -> None:
    rendered = user_prompt(_vaswani(), _claim("attention is all you need"))
    assert "Attention Is All You Need" in rendered
    assert "attention is all you need" in rendered
    assert "methodological" in rendered


def test_relevance_schema_is_strict_and_minimal() -> None:
    assert RELEVANCE_SCHEMA["additionalProperties"] is False
    assert set(RELEVANCE_SCHEMA["required"]) == {"relevance", "reasoning"}
    rel = RELEVANCE_SCHEMA["properties"]["relevance"]
    assert rel["type"] == "number"
    assert rel["minimum"] == 0.0
    assert rel["maximum"] == 1.0


def test_payload_to_relevance_handles_none() -> None:
    assert payload_to_relevance(None) is None


def test_payload_to_relevance_clamps_out_of_band_values() -> None:
    """LLM returned 1.4 — clamp to 1.0 rather than dropping the judgment."""
    judgment = payload_to_relevance({"relevance": 1.4, "reasoning": "x"})
    assert judgment is not None
    assert judgment.relevance == 1.0
    judgment = payload_to_relevance({"relevance": -0.3, "reasoning": "x"})
    assert judgment is not None
    assert judgment.relevance == 0.0


def test_payload_to_relevance_returns_none_for_non_numeric_relevance() -> None:
    assert payload_to_relevance({"relevance": "high", "reasoning": "x"}) is None


# ---- max_retries plumbed through to SDK on default construction ----


def test_openai_passes_max_retries_to_client() -> None:
    from openai import AsyncOpenAI

    s = OpenAILLMCitationScorer(api_key="sk-test", max_retries=7)
    assert isinstance(s._client, AsyncOpenAI)
    assert s._client.max_retries == 7


def test_anthropic_passes_max_retries_to_client() -> None:
    from anthropic import AsyncAnthropic

    s = AnthropicLLMCitationScorer(api_key="sk-test", max_retries=7)
    assert isinstance(s._client, AsyncAnthropic)
    assert s._client.max_retries == 7


def test_default_base_scorer_is_heuristic() -> None:
    """Issue #3 design decision: compose on top of HeuristicCitationScorer
    by default; do not replace it."""
    s = OpenAILLMCitationScorer(api_key="sk-test")
    assert isinstance(s._base, HeuristicCitationScorer)


# ---- stubs ----


class _StubOpenAIChoice:
    def __init__(self, content: str | None) -> None:
        self.message = SimpleNamespace(content=content)


class _StubOpenAIResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_StubOpenAIChoice(content)]


class _StubOpenAIClient:
    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        raw_content: str | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._raise = raise_exc
        if payload is not None:
            self._content: str | None = json.dumps(payload)
        else:
            self._content = raw_content
        self.calls = 0
        self.max_retries = 4
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs: Any) -> _StubOpenAIResponse:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return _StubOpenAIResponse(self._content)


class _StubAnthropicToolBlock:
    type = "tool_use"

    def __init__(self, name: str, payload: dict[str, Any]) -> None:
        self.name = name
        self.input = payload


class _StubAnthropicResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _StubAnthropicClient:
    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        content: list[Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._raise = raise_exc
        if content is not None:
            self._content = content
        elif payload is not None:
            self._content = [_StubAnthropicToolBlock(RELEVANCE_TOOL_NAME, payload)]
        else:
            self._content = []
        self.calls = 0
        self.max_retries = 4
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> _StubAnthropicResponse:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return _StubAnthropicResponse(self._content)
