"""Anthropic-backed `CitationScorer` that adjusts a base score by semantic relevance.

Same compose-on-base shape as `OpenAILLMCitationScorer`. Anthropic
doesn't expose a direct response_format JSON-schema feature; instead
we register the relevance schema as a tool's input_schema and force
the model to call it via `tool_choice`. The tool input is the
relevance judgment.

Default model: `claude-haiku-4-5-20251001`. Same retry / timeout
rationale as the OpenAI scorer — short prompt, short output, 30s is
generous.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Final

from anthropic import AsyncAnthropic

from research_mcp.citation_scorer._llm_schema import (
    RELEVANCE_SCHEMA,
    RELEVANCE_TOOL_DESCRIPTION,
    RELEVANCE_TOOL_NAME,
    RelevanceJudgment,
    payload_to_relevance,
    system_prompt,
    user_prompt,
)
from research_mcp.citation_scorer.heuristic import HeuristicCitationScorer
from research_mcp.domain.citation_scorer import (
    CitationQualityScore,
    CitationScorer,
)
from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper

_log = logging.getLogger(__name__)

_DEFAULT_MODEL: Final = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_RETRIES: Final = 4
_DEFAULT_TIMEOUT_SECONDS: Final = 30.0
# Output is at most a number plus a one-sentence reasoning; 512 tokens
# is comfortable headroom for the tool call envelope.
_DEFAULT_MAX_TOKENS: Final = 512

_RELEVANCE_FLOOR: Final = 0.5
_LOW_RELEVANCE_WARN_THRESHOLD: Final = 0.4


class AnthropicLLMCitationScorer:
    """Wraps a base CitationScorer with an LLM-driven relevance adjustment."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        base_scorer: CitationScorer | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.name = f"llm:anthropic:{model}"
        self._max_tokens = max_tokens
        self._base = base_scorer or HeuristicCitationScorer()
        self._client = client or AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
            timeout=timeout,
        )

    async def score(
        self,
        paper: Paper,
        claim: Claim | None = None,
    ) -> CitationQualityScore:
        base = await self._base.score(paper, claim)
        if claim is None:
            return base

        judgment = await self._judge_relevance(paper, claim)
        if judgment is None:
            return base

        modulator = _RELEVANCE_FLOOR + (1.0 - _RELEVANCE_FLOOR) * judgment.relevance
        adjusted = max(0.0, min(100.0, base.total * modulator))
        new_factors = dict(base.factors)
        new_factors["relevance"] = (
            f"Semantic fit: {judgment.relevance:.2f}. {judgment.reasoning}".strip()
        )
        new_warnings = list(base.warnings)
        if judgment.relevance < _LOW_RELEVANCE_WARN_THRESHOLD:
            new_warnings.append("low semantic relevance to claim")
        return CitationQualityScore(
            total=round(adjusted, 2),
            venue=base.venue,
            impact=base.impact,
            author=base.author,
            recency=base.recency,
            factors=new_factors,
            warnings=tuple(new_warnings),
        )

    async def _judge_relevance(
        self, paper: Paper, claim: Claim
    ) -> RelevanceJudgment | None:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system_prompt(),
                tools=[
                    {
                        "name": RELEVANCE_TOOL_NAME,
                        "description": RELEVANCE_TOOL_DESCRIPTION,
                        "input_schema": RELEVANCE_SCHEMA,
                    }
                ],
                tool_choice={"type": "tool", "name": RELEVANCE_TOOL_NAME},
                messages=[{"role": "user", "content": user_prompt(paper, claim)}],
            )
        except Exception as exc:
            _log.warning("anthropic citation scorer call failed: %s", exc)
            return None

        payload = _extract_tool_payload(response)
        if payload is None:
            _log.warning("anthropic citation scorer returned no tool_use block")
            return None
        return payload_to_relevance(payload)


def _extract_tool_payload(response: Any) -> dict[str, Any] | None:
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == RELEVANCE_TOOL_NAME
        ):
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    return None
