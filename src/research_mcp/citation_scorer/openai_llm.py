"""OpenAI-backed `CitationScorer` that adjusts a base score by semantic relevance.

Composes on top of a base scorer (HeuristicCitationScorer by default).
The four data-grounded dimensions (venue / impact / author / recency)
come from the base; the LLM only judges whether the paper actually
says what the claim wants to cite it for, and modulates the total
within a 0.5-1.0 band so the heuristic signal is never erased.

Default model: `gpt-4o-mini`. Override per-instance via the `model`
kwarg or globally via `RESEARCH_MCP_CITATION_SCORER=llm:openai:<model>`.
Backoff: SDK exponential schedule, `max_retries=4` (vs. SDK default 2)
because the scorer is on the find_citations / explain_citation hot
path. Timeout 30s — the prompt is short (<500 tokens) and the output
is a single number plus a sentence; the only legitimate slow path is
a transient API stall.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Final

from openai import AsyncOpenAI

from research_mcp.citation_scorer._llm_schema import (
    RELEVANCE_SCHEMA,
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

_DEFAULT_MODEL: Final = "gpt-4o-mini"
_DEFAULT_MAX_RETRIES: Final = 4
_DEFAULT_TIMEOUT_SECONDS: Final = 30.0
_SCHEMA_NAME: Final = "citation_relevance"

# Lower bound for the relevance modulator. With this band, a relevance
# of 0.0 still preserves 50% of the heuristic signal — the LLM never
# fully overrides venue/impact/author/recency, only nudges them.
_RELEVANCE_FLOOR: Final = 0.5
# A relevance below this triggers a warning surfaced through the score's
# warnings tuple. 0.4 catches "wrong subfield" without false-positiving
# on borderline cases (relevance 0.5-0.6 is "topically related").
_LOW_RELEVANCE_WARN_THRESHOLD: Final = 0.4


class OpenAILLMCitationScorer:
    """Wraps a base CitationScorer with an LLM-driven relevance adjustment."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        base_scorer: CitationScorer | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.name = f"llm:openai:{model}"
        self._base = base_scorer or HeuristicCitationScorer()
        self._client = client or AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
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
            # API failed or returned malformed output: fall back to the
            # base score so a transient LLM outage doesn't break the
            # scoring pipeline.
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
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(paper, claim)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": _SCHEMA_NAME,
                        "schema": RELEVANCE_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:
            _log.warning("openai citation scorer call failed: %s", exc)
            return None

        content = response.choices[0].message.content
        if not content:
            _log.warning("openai citation scorer returned empty content")
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            _log.exception("openai citation scorer returned non-JSON content")
            return None
        return payload_to_relevance(parsed)
