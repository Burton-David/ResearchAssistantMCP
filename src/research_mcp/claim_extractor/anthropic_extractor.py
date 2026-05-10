"""Anthropic-backed `ClaimExtractor` using tool use.

Same one-call-per-extract pattern as the OpenAI extractor. Anthropic
doesn't have a direct response_format for JSON schema; instead we
register the schema as a tool's input_schema and force the model to
call it via `tool_choice`. The tool input is the structured claim list.

Default model: `claude-haiku-4-5-20251001`. Same backoff rationale as
the OpenAI extractor — `max_retries=4` (SDK default 2) since a
transient 429 should not break `assist_draft`. The SDK's exponential
schedule honors Retry-After.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Final

from anthropic import AsyncAnthropic

from research_mcp.claim_extractor._llm_schema import (
    CLAIM_EXTRACTION_SCHEMA,
    CLAIM_EXTRACTION_TOOL_DESCRIPTION,
    CLAIM_EXTRACTION_TOOL_NAME,
    payload_to_claims,
    system_prompt,
    user_prompt,
)
from research_mcp.domain.claim import Claim

_log = logging.getLogger(__name__)

_DEFAULT_MODEL: Final = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_RETRIES: Final = 4
# See OpenAILLMClaimExtractor for the rationale: Anthropic SDK default
# is 600s x 4 retries; that's enough to wedge an MCP call past Claude
# Desktop's hard kill. 60s is generous for claim extraction.
_DEFAULT_TIMEOUT_SECONDS: Final = 60.0
# Claim lists for a 20K-char paragraph rarely exceed 50 entries. Cap on
# tokens generously so the model has room for verbose context fields.
_DEFAULT_MAX_TOKENS: Final = 4096


class AnthropicLLMClaimExtractor:
    """Tool-use-based claim extractor over the Anthropic Messages API."""

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
        self.name = f"llm:anthropic:{model}"
        self._max_tokens = max_tokens
        self._client = client or AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
            timeout=timeout,
        )

    async def extract(self, text: str) -> Sequence[Claim]:
        if not text or not text.strip():
            return ()
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system_prompt(),
                tools=[
                    {
                        "name": CLAIM_EXTRACTION_TOOL_NAME,
                        "description": CLAIM_EXTRACTION_TOOL_DESCRIPTION,
                        "input_schema": CLAIM_EXTRACTION_SCHEMA,
                    }
                ],
                tool_choice={"type": "tool", "name": CLAIM_EXTRACTION_TOOL_NAME},
                messages=[{"role": "user", "content": user_prompt(text)}],
            )
        except Exception as exc:
            _log.warning("anthropic claim extractor call failed: %s", exc)
            return ()

        payload = _extract_tool_payload(response)
        if payload is None:
            _log.warning("anthropic claim extractor returned no tool_use block")
            return ()
        return payload_to_claims(payload, text=text, model_name=self.name)


def _extract_tool_payload(response: Any) -> dict[str, Any] | None:
    """Pull the tool_use input out of the Anthropic response.

    The response's `content` is a list of blocks; we want the one with
    type 'tool_use' and matching name. tool_choice forces a tool call,
    so this should always succeed — None return defends against API
    contract drift.
    """
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == CLAIM_EXTRACTION_TOOL_NAME
        ):
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    return None
