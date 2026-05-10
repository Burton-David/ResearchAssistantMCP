"""OpenAI-backed `PaperAnalyzer` using structured outputs.

Model default: `gpt-4o-mini` — cheap (~$0.15/1M input tokens) and
sufficient for paper-summary extraction. Override via the `model`
constructor argument or `RESEARCH_MCP_ANALYSIS_MODEL=openai:gpt-4o`.

Structured outputs keep the response on-schema: the SDK enforces the
JSON schema the analyzer hands it, so we never see hallucinated keys
or shape drift. One `client.chat.completions.create` call per
`analyze` — no per-kind multiplication, since the schema covers all
analysis kinds at once and `kinds` only constrains which fields the
prompt asks for.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Final

from openai import AsyncOpenAI

from research_mcp.domain.paper import Paper
from research_mcp.domain.paper_analyzer import AnalysisKind, PaperAnalysis
from research_mcp.paper_analyzer._schema import (
    ANALYSIS_SCHEMA,
    system_prompt,
    text_for_paper,
    user_prompt,
)

_log = logging.getLogger(__name__)

_DEFAULT_MODEL: Final = "gpt-4o-mini"
_DEFAULT_MAX_RETRIES: Final = 4
# response_format requires a name on the schema; OpenAI rejects names with
# spaces or other invalid identifiers.
_SCHEMA_NAME: Final = "paper_analysis"


class OpenAILLMPaperAnalyzer:
    """Structured-output paper analyzer over the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        # SDK default is 2 retries; we bump to 4 for the user's hot path
        # (analyze_paper / assist_draft are interactive). The SDK does
        # exponential backoff and honors Retry-After.
        self._client = client or AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            max_retries=max_retries,
        )

    async def analyze(
        self,
        paper: Paper,
        kinds: Sequence[AnalysisKind] = (),
    ) -> PaperAnalysis:
        if not text_for_paper(paper):
            return PaperAnalysis(paper_id=paper.id, model=self.name)

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(paper, kinds)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": _SCHEMA_NAME,
                    "schema": ANALYSIS_SCHEMA,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            _log.warning("openai paper analyzer returned empty content")
            return PaperAnalysis(paper_id=paper.id, model=self.name)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            _log.exception("openai paper analyzer returned non-JSON content")
            return PaperAnalysis(paper_id=paper.id, model=self.name)
        return _payload_to_analysis(parsed, paper_id=paper.id, model_name=self.name)


def _payload_to_analysis(
    payload: dict[str, Any], *, paper_id: str, model_name: str
) -> PaperAnalysis:
    """Lift the LLM's JSON output into the immutable domain object."""
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
