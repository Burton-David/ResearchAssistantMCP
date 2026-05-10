"""PaperAnalyzer tests — fake, OpenAI, and Anthropic implementations."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.paper_analyzer import (
    AnalysisKind,
    PaperAnalysis,
    PaperAnalyzer,
)
from research_mcp.paper_analyzer import (
    AnthropicLLMPaperAnalyzer,
    FakePaperAnalyzer,
    OpenAILLMPaperAnalyzer,
)
from research_mcp.paper_analyzer._schema import ANALYSIS_SCHEMA, user_prompt

pytestmark = pytest.mark.unit


def _paper(
    *,
    title: str = "Attention Is All You Need",
    abstract: str = "We propose a new transformer architecture.",
    full_text: str | None = None,
) -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title=title,
        abstract=abstract,
        authors=(Author("A. Vaswani"),),
        published=date(2017, 6, 12),
        full_text=full_text,
    )


# ---- protocol conformance ----


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakePaperAnalyzer(), PaperAnalyzer)
    assert FakePaperAnalyzer().name == "fake"


def test_openai_satisfies_protocol() -> None:
    """Construct without making any API calls — just a shape check."""
    a = OpenAILLMPaperAnalyzer(client=_StubOpenAIClient(payload={}))
    assert isinstance(a, PaperAnalyzer)
    assert a.name.startswith("openai:")


def test_anthropic_satisfies_protocol() -> None:
    a = AnthropicLLMPaperAnalyzer(client=_StubAnthropicClient(payload={}))
    assert isinstance(a, PaperAnalyzer)
    assert a.name.startswith("anthropic:")


# ---- empty input ----


async def test_fake_returns_empty_for_blank_paper() -> None:
    blank = Paper(id="x:1", title="", abstract="", authors=())
    result = await FakePaperAnalyzer().analyze(blank)
    assert isinstance(result, PaperAnalysis)
    assert result.confidence == 0.0
    assert result.summary is None


async def test_openai_returns_empty_for_blank_paper() -> None:
    blank = Paper(id="x:1", title="", abstract="", authors=())
    stub = _StubOpenAIClient(payload={"summary": "should not see this"})
    a = OpenAILLMPaperAnalyzer(client=stub)
    result = await a.analyze(blank)
    assert result.confidence == 0.0
    assert result.summary is None
    # No API call happened — short-circuited.
    assert stub.calls == 0


async def test_anthropic_returns_empty_for_blank_paper() -> None:
    blank = Paper(id="x:1", title="", abstract="", authors=())
    stub = _StubAnthropicClient(payload={})
    a = AnthropicLLMPaperAnalyzer(client=stub)
    result = await a.analyze(blank)
    assert result.confidence == 0.0
    assert stub.calls == 0


# ---- happy path ----


async def test_fake_returns_first_sentence_summary() -> None:
    """The fake builds its summary from `text_for_paper(paper)`, which
    concatenates title + abstract with a blank line between them. The
    first sentence boundary is the title, since titles don't end in '.'.
    Lock that behavior down so test_assist_draft and similar callers
    can rely on it."""
    paper = Paper(
        id="x:1",
        title="",
        abstract="A new transformer. With attention. Beats LSTMs.",
        authors=(Author("X"),),
    )
    result = await FakePaperAnalyzer().analyze(paper)
    assert result.summary == "A new transformer."
    assert result.confidence == 0.5
    assert result.model == "fake:stub"


async def test_openai_lifts_payload_into_analysis() -> None:
    payload = {
        "summary": "Transformers replace RNNs for sequence transduction.",
        "key_contributions": [
            "Self-attention", "Multi-head attention", "Positional encoding",
        ],
        "methodology": "Encoder-decoder stacks of self-attention layers.",
        "technical_approach": "Scaled dot-product attention.",
        "limitations": ["Quadratic in sequence length"],
        "future_directions": ["Apply beyond NLP"],
        "datasets_used": ["WMT 2014 EN-DE", "WMT 2014 EN-FR"],
        "metrics_reported": {"bleu_en_de": 28.4, "bleu_en_fr": 41.8},
        "baselines_compared": ["GNMT"],
        "confidence": 0.92,
    }
    a = OpenAILLMPaperAnalyzer(
        model="gpt-4o-mini",
        client=_StubOpenAIClient(payload=payload),
    )
    result = await a.analyze(_paper())
    assert result.summary == payload["summary"]
    assert result.key_contributions == tuple(payload["key_contributions"])
    assert result.methodology == payload["methodology"]
    assert result.datasets_used == tuple(payload["datasets_used"])
    assert result.metrics_reported["bleu_en_de"] == 28.4
    assert result.confidence == 0.92
    assert result.model == "openai:gpt-4o-mini"


async def test_anthropic_lifts_payload_into_analysis() -> None:
    payload = {
        "summary": "Transformer architecture.",
        "key_contributions": ["Self-attention"],
        "methodology": "Stacked attention layers.",
        "technical_approach": "Scaled dot-product attention.",
        "limitations": ["Quadratic memory"],
        "future_directions": [],
        "datasets_used": ["WMT14"],
        "metrics_reported": {"bleu": 28.4},
        "baselines_compared": [],
        "confidence": 0.88,
    }
    a = AnthropicLLMPaperAnalyzer(
        model="claude-haiku-4-5-20251001",
        client=_StubAnthropicClient(payload=payload),
    )
    result = await a.analyze(_paper())
    assert result.summary == "Transformer architecture."
    assert result.metrics_reported["bleu"] == 28.4
    assert result.confidence == 0.88
    assert result.model == "anthropic:claude-haiku-4-5-20251001"


# ---- malformed responses ----


async def test_openai_handles_invalid_json_content() -> None:
    """If structured outputs ever ship malformed JSON (e.g. server-side
    fallback to plaintext), we degrade to confidence=0 instead of raising."""
    a = OpenAILLMPaperAnalyzer(client=_StubOpenAIClient(raw_content="not json"))
    result = await a.analyze(_paper())
    assert result.confidence == 0.0
    assert result.summary is None


async def test_openai_handles_empty_content() -> None:
    a = OpenAILLMPaperAnalyzer(client=_StubOpenAIClient(raw_content=""))
    result = await a.analyze(_paper())
    assert result.confidence == 0.0


async def test_anthropic_handles_response_with_no_tool_use_block() -> None:
    """If the model fails to call the tool (rare with tool_choice forced,
    but not impossible), we should return blank rather than crash."""
    a = AnthropicLLMPaperAnalyzer(client=_StubAnthropicClient(content=[]))
    result = await a.analyze(_paper())
    assert result.confidence == 0.0


# ---- prompt shape ----


def test_user_prompt_includes_title_abstract_and_kinds() -> None:
    """Lock down the prompt so a future change can't silently drop fields."""
    paper = _paper(title="Some Paper", abstract="Some abstract.")
    rendered = user_prompt(paper, [AnalysisKind.METHODOLOGY])
    assert "Some Paper" in rendered
    assert "Some abstract." in rendered
    assert "methodology" in rendered.lower()


def test_user_prompt_truncates_huge_full_text() -> None:
    """Don't blow the context budget on a 500K-char paper."""
    paper = _paper(full_text="x" * 200_000)
    rendered = user_prompt(paper, [])
    assert "[...truncated...]" in rendered
    # Bounded: not the full 200K + overhead.
    assert len(rendered) < 100_000


# ---- schema ----


def test_schema_is_strict_and_self_consistent() -> None:
    """The OpenAI structured-output mode rejects schemas that aren't
    additionalProperties:false. Lock that down."""
    assert ANALYSIS_SCHEMA["additionalProperties"] is False
    # Every required field must be defined in properties.
    for required in ANALYSIS_SCHEMA["required"]:
        assert required in ANALYSIS_SCHEMA["properties"]


# ---- stubs ----


class _StubOpenAIChoice:
    def __init__(self, content: str | None) -> None:
        self.message = SimpleNamespace(content=content)


class _StubOpenAIResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_StubOpenAIChoice(content)]


class _StubOpenAIClient:
    """Minimal AsyncOpenAI lookalike: only the surface OpenAILLMPaperAnalyzer touches.

    Pass `payload` to have the stub serialize it as JSON content; pass
    `raw_content` to override with a literal string (for invalid-JSON
    test cases).
    """

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        raw_content: str | None = None,
    ) -> None:
        if payload is not None:
            self._content: str | None = json.dumps(payload)
        else:
            self._content = raw_content
        self.calls = 0
        # Mimic the SDK's nested attribute layout.
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs: Any) -> _StubOpenAIResponse:
        self.calls += 1
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
    """Minimal AsyncAnthropic lookalike: returns a tool_use block with the
    payload (or a custom content list, for malformed-response tests)."""

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        content: list[Any] | None = None,
    ) -> None:
        from research_mcp.paper_analyzer._schema import ANALYSIS_TOOL_NAME

        if content is not None:
            self._content = content
        elif payload is not None:
            self._content = [_StubAnthropicToolBlock(ANALYSIS_TOOL_NAME, payload)]
        else:
            self._content = []
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> _StubAnthropicResponse:
        self.calls += 1
        return _StubAnthropicResponse(self._content)
