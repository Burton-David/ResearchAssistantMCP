"""Anthropic-backed `PaperAnalyzer` using tool use for structured output.

Model default: `claude-haiku-4-5-20251001` — cheap and fast; sufficient
for paper analysis. Override via the `model` constructor argument or
`RESEARCH_MCP_ANALYSIS_MODEL=anthropic:claude-sonnet-4-6`.

Anthropic doesn't have a direct "structured output" feature like
OpenAI's response_format; instead we send a tool whose input_schema
matches the analysis schema, then force the model to call it via
`tool_choice={"type": "tool", "name": "submit_paper_analysis"}`.
The tool's input becomes the structured analysis.

The point of having both implementations side by side: shows the
protocol composes with any LLM provider. Selection is a runtime
env-var; tests cover both paths.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Final

from anthropic import AsyncAnthropic

from research_mcp.domain.paper import Paper
from research_mcp.domain.paper_analyzer import AnalysisKind, PaperAnalysis
from research_mcp.paper_analyzer._schema import (
    ANALYSIS_SCHEMA,
    ANALYSIS_TOOL_DESCRIPTION,
    ANALYSIS_TOOL_NAME,
    system_prompt,
    text_for_paper,
    user_prompt,
)

_log = logging.getLogger(__name__)

_DEFAULT_MODEL: Final = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_RETRIES: Final = 4
# See OpenAILLMPaperAnalyzer for the rationale: 90s explicit per-attempt
# timeout vs the SDK default of 600s, since the tool call has to stay
# under Claude Desktop's 4-min hard kill.
_DEFAULT_TIMEOUT_SECONDS: Final = 90.0
_DEFAULT_MAX_TOKENS: Final = 4096


class AnthropicLLMPaperAnalyzer:
    """Tool-use-based paper analyzer over the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self._max_tokens = max_tokens
        self._client = client or AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
            timeout=timeout,
        )

    async def analyze(
        self,
        paper: Paper,
        kinds: Sequence[AnalysisKind] = (),
    ) -> PaperAnalysis:
        if not text_for_paper(paper):
            return PaperAnalysis(paper_id=paper.id, model=self.name)

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system_prompt(),
            tools=[
                {
                    "name": ANALYSIS_TOOL_NAME,
                    "description": ANALYSIS_TOOL_DESCRIPTION,
                    "input_schema": ANALYSIS_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": ANALYSIS_TOOL_NAME},
            messages=[{"role": "user", "content": user_prompt(paper, kinds)}],
        )
        payload = _extract_tool_payload(response)
        if payload is None:
            _log.warning("anthropic analyzer returned no tool_use block")
            return PaperAnalysis(paper_id=paper.id, model=self.name)
        return _payload_to_analysis(payload, paper_id=paper.id, model_name=self.name)


def _extract_tool_payload(response: Any) -> dict[str, Any] | None:
    """Pull the tool_use input out of the Anthropic response.

    The response's `content` is a list of blocks; we want the first one
    with type 'tool_use' and matching name. tool_choice forces a tool
    call, so this should always succeed — the None return is defense
    against API contract drift.
    """
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == ANALYSIS_TOOL_NAME:
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    return None


def _payload_to_analysis(
    payload: dict[str, Any], *, paper_id: str, model_name: str
) -> PaperAnalysis:
    metrics = payload.get("metrics_reported") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    cleaned_metrics = {
        str(k): float(v) for k, v in metrics.items() if isinstance(v, int | float)
    }
    return PaperAnalysis(
        paper_id=paper_id,
        summary=_pick_str(payload.get("summary")),
        key_contributions=_pick_str_tuple(payload.get("key_contributions")),
        methodology=_pick_str(payload.get("methodology")),
        technical_approach=_pick_str(payload.get("technical_approach")),
        limitations=_pick_str_tuple(payload.get("limitations")),
        future_directions=_pick_str_tuple(payload.get("future_directions")),
        datasets_used=_pick_str_tuple(payload.get("datasets_used")),
        metrics_reported=MappingProxyType(cleaned_metrics),
        baselines_compared=_pick_str_tuple(payload.get("baselines_compared")),
        confidence=_pick_confidence(payload.get("confidence")),
        model=model_name,
    )


def _pick_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _pick_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(v.strip() for v in value if isinstance(v, str) and v.strip())


def _pick_confidence(value: Any) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return 0.0
