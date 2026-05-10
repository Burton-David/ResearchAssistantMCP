"""LLM-based ClaimExtractor tests — OpenAI and Anthropic implementations.

The extractors call out to OpenAI / Anthropic SDKs; we test by injecting
a stub client that returns canned JSON. Real-API exercise lives in the
gated integration tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from research_mcp.claim_extractor import (
    AnthropicLLMClaimExtractor,
    OpenAILLMClaimExtractor,
)
from research_mcp.claim_extractor._llm_schema import (
    CLAIM_EXTRACTION_SCHEMA,
    CLAIM_EXTRACTION_TOOL_NAME,
    payload_to_claims,
    user_prompt,
)
from research_mcp.domain import ClaimExtractor, ClaimType

pytestmark = pytest.mark.unit


# ---- protocol conformance ----


def test_openai_satisfies_protocol() -> None:
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(payload={"claims": []}))
    assert isinstance(e, ClaimExtractor)
    assert e.name.startswith("llm:openai:")


def test_anthropic_satisfies_protocol() -> None:
    e = AnthropicLLMClaimExtractor(client=_StubAnthropicClient(payload={"claims": []}))
    assert isinstance(e, ClaimExtractor)
    assert e.name.startswith("llm:anthropic:")


# ---- short-circuit on empty input ----


async def test_openai_short_circuits_on_blank_input() -> None:
    stub = _StubOpenAIClient(payload={"claims": []})
    e = OpenAILLMClaimExtractor(client=stub)
    assert list(await e.extract("")) == []
    assert list(await e.extract("   \n\t  ")) == []
    assert stub.calls == 0  # no API hit for blank input


async def test_anthropic_short_circuits_on_blank_input() -> None:
    stub = _StubAnthropicClient(payload={"claims": []})
    e = AnthropicLLMClaimExtractor(client=stub)
    assert list(await e.extract("")) == []
    assert stub.calls == 0


# ---- happy path ----


_TEXT = (
    "Recent transformer models have outperformed LSTMs by 23% on "
    "machine translation tasks. The proposed approach achieves a BLEU "
    "score of 28.4 on WMT 2014."
)
_PAYLOAD = {
    "claims": [
        {
            "text": "Recent transformer models have outperformed LSTMs by 23%",
            "type": "comparative",
            "confidence": 0.92,
            "context": (
                "Recent transformer models have outperformed LSTMs by 23% "
                "on machine translation tasks."
            ),
            "suggested_search_terms": [
                "transformer", "LSTM", "machine translation",
            ],
        },
        {
            "text": "BLEU score of 28.4",
            "type": "statistical",
            "confidence": 0.88,
            "context": (
                "The proposed approach achieves a BLEU score of 28.4 on WMT 2014."
            ),
            "suggested_search_terms": ["BLEU", "WMT 2014", "machine translation"],
        },
    ]
}


async def test_openai_lifts_payload_into_ordered_claims() -> None:
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(payload=_PAYLOAD))
    claims = list(await e.extract(_TEXT))
    assert len(claims) == 2
    # Document order: comparative first, then statistical.
    assert claims[0].type == ClaimType.COMPARATIVE
    assert claims[1].type == ClaimType.STATISTICAL
    assert claims[0].start_char < claims[1].start_char


async def test_anthropic_lifts_payload_into_ordered_claims() -> None:
    e = AnthropicLLMClaimExtractor(client=_StubAnthropicClient(payload=_PAYLOAD))
    claims = list(await e.extract(_TEXT))
    assert len(claims) == 2
    assert claims[0].type == ClaimType.COMPARATIVE
    assert claims[1].type == ClaimType.STATISTICAL


# ---- span anchoring ----


async def test_anchoring_finds_claim_text_in_input() -> None:
    """start_char/end_char round-trip to the original text, so a UI
    can highlight the claim in place."""
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(payload=_PAYLOAD))
    claims = list(await e.extract(_TEXT))
    for c in claims:
        assert _TEXT[c.start_char : c.end_char] == c.text


async def test_anchoring_falls_back_to_zero_zero_for_hallucinated_text() -> None:
    """If the LLM rewords the claim instead of copying it, we still
    return the claim — start/end just default to (0, 0). Drop-on-miss
    would lose information; a UI can degrade gracefully when the anchor
    is absent."""
    payload = {
        "claims": [
            {
                "text": "this string is NOT in the input verbatim",
                "type": "factual",
                "confidence": 0.7,
                "context": "context",
                "suggested_search_terms": ["x"],
            }
        ]
    }
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(payload=payload))
    claims = list(await e.extract("Some unrelated input text."))
    assert len(claims) == 1
    assert claims[0].start_char == 0
    assert claims[0].end_char == 0


# ---- robust to malformed responses ----


async def test_openai_returns_empty_for_missing_claims_array() -> None:
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(payload={}))
    assert list(await e.extract(_TEXT)) == []


async def test_openai_returns_empty_for_invalid_json() -> None:
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(raw_content="not json"))
    assert list(await e.extract(_TEXT)) == []


async def test_openai_returns_empty_when_sdk_raises() -> None:
    """SDK retries are exhausted → log + return []. assist_draft must
    not crash because OpenAI is briefly down."""
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(raise_exc=RuntimeError("503")))
    assert list(await e.extract(_TEXT)) == []


async def test_anthropic_returns_empty_when_sdk_raises() -> None:
    e = AnthropicLLMClaimExtractor(
        client=_StubAnthropicClient(raise_exc=RuntimeError("503"))
    )
    assert list(await e.extract(_TEXT)) == []


async def test_anthropic_returns_empty_for_no_tool_use_block() -> None:
    e = AnthropicLLMClaimExtractor(client=_StubAnthropicClient(content=[]))
    assert list(await e.extract(_TEXT)) == []


# ---- unknown claim types ----


async def test_unknown_claim_type_falls_back_to_factual() -> None:
    """LLM hallucinated 'unknown' as a type — keep the claim, classify
    as FACTUAL. Dropping would lose the claim entirely."""
    payload = {
        "claims": [
            {
                "text": "Recent transformer models have outperformed LSTMs by 23%",
                "type": "unknown_type",
                "confidence": 0.9,
                "context": "ctx",
                "suggested_search_terms": [],
            }
        ]
    }
    e = OpenAILLMClaimExtractor(client=_StubOpenAIClient(payload=payload))
    claims = list(await e.extract(_TEXT))
    assert len(claims) == 1
    assert claims[0].type == ClaimType.FACTUAL


async def test_extractor_metadata_records_model_name() -> None:
    """Caller can tell which extractor produced the claim — useful when
    the server is hot-swapped between spaCy and LLM extractors."""
    e = OpenAILLMClaimExtractor(
        model="gpt-4o-mini", client=_StubOpenAIClient(payload=_PAYLOAD)
    )
    claims = list(await e.extract(_TEXT))
    assert all(c.metadata.get("extractor") == "llm:openai:gpt-4o-mini" for c in claims)


# ---- vectorization: one call per extract regardless of claim count ----


async def test_one_api_call_per_extract_regardless_of_claim_count() -> None:
    """The whole point of the LLM extractor: a single call returns ALL
    claims in the input, not one call per sentence. Lock that down so
    a future "let's split per-paragraph" change has to be deliberate."""
    stub = _StubOpenAIClient(payload=_PAYLOAD)
    e = OpenAILLMClaimExtractor(client=stub)
    long_input = (_TEXT + "\n\n") * 10  # 10x repeat → many claims expected
    await e.extract(long_input)
    assert stub.calls == 1


# ---- prompt + schema invariants ----


def test_prompt_includes_input_text_verbatim() -> None:
    rendered = user_prompt("My specific input.")
    assert "My specific input." in rendered


def test_schema_is_strict_and_self_consistent() -> None:
    assert CLAIM_EXTRACTION_SCHEMA["additionalProperties"] is False
    claim_props = CLAIM_EXTRACTION_SCHEMA["properties"]["claims"]["items"][
        "properties"
    ]
    # Every claim type from the enum must be a legal value
    valid_types = set(claim_props["type"]["enum"])
    for ct in ClaimType:
        assert ct.value in valid_types


def test_payload_to_claims_handles_none_payload() -> None:
    assert payload_to_claims(None, text="x", model_name="test") == ()


def test_payload_to_claims_handles_non_list_claims() -> None:
    assert payload_to_claims(
        {"claims": "not a list"}, text="x", model_name="test"
    ) == ()


def test_payload_to_claims_skips_non_dict_items() -> None:
    """A list element that isn't a dict (LLM glitched) gets skipped, others survive."""
    payload = {
        "claims": [
            "not a dict",
            {
                "text": "real claim",
                "type": "factual",
                "confidence": 0.5,
                "context": "ctx",
                "suggested_search_terms": [],
            },
        ]
    }
    claims = payload_to_claims(payload, text="real claim is here", model_name="x")
    assert len(claims) == 1


# ---- max_retries plumbed through to SDK ----


def test_openai_extractor_passes_max_retries_to_client_when_constructed() -> None:
    """Verify the SDK gets the bumped retry count on default construction.

    Since we accept a pre-built client too, this only checks the
    default-build path — a custom client overrides retry behavior at
    its own discretion."""
    from openai import AsyncOpenAI

    e = OpenAILLMClaimExtractor(api_key="sk-test", max_retries=7)
    assert isinstance(e._client, AsyncOpenAI)
    assert e._client.max_retries == 7


def test_anthropic_extractor_passes_max_retries_to_client_when_constructed() -> None:
    from anthropic import AsyncAnthropic

    e = AnthropicLLMClaimExtractor(api_key="sk-test", max_retries=7)
    assert isinstance(e._client, AsyncAnthropic)
    assert e._client.max_retries == 7


# ---- stubs ----


class _StubOpenAIChoice:
    def __init__(self, content: str | None) -> None:
        self.message = SimpleNamespace(content=content)


class _StubOpenAIResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_StubOpenAIChoice(content)]


class _StubOpenAIClient:
    """Minimal AsyncOpenAI lookalike: only the surface OpenAILLMClaimExtractor touches."""

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
            self._content = [
                _StubAnthropicToolBlock(CLAIM_EXTRACTION_TOOL_NAME, payload)
            ]
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
