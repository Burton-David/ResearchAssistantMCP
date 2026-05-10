"""OpenAI-backed `ClaimExtractor` using structured outputs.

One LLM call per `extract(text)` — vectorized at the level the user
cares about: extract every claim from the input in a single pass
rather than scanning per-sentence. The model sees the full context,
so cross-sentence references ("These results suggest...") classify
correctly.

Default model: `gpt-4o-mini`. Override per-instance via the `model`
kwarg or globally via `RESEARCH_MCP_CLAIM_EXTRACTOR=llm:openai:<model>`.

Backoff: delegated to the OpenAI SDK's built-in exponential retry
schedule. Default `max_retries=4` (vs. the SDK default of 2) since
claim extraction is on the user's hot path and a transient 429 should
not surface as a failed `assist_draft` call. The SDK uses an
exponential schedule (~0.5s, 1s, 2s, 4s) plus Retry-After honoring,
which matches what we built for the HTTP source adapters.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Final

from openai import AsyncOpenAI

from research_mcp.claim_extractor._llm_schema import (
    CLAIM_EXTRACTION_SCHEMA,
    payload_to_claims,
    system_prompt,
    user_prompt,
)
from research_mcp.domain.claim import Claim

_log = logging.getLogger(__name__)

_DEFAULT_MODEL: Final = "gpt-4o-mini"
_DEFAULT_MAX_RETRIES: Final = 4
# Explicit per-attempt timeout. The OpenAI SDK default is 600s, which
# gives 4 retries x 600s = 40 minutes worst case before surfacing a
# failure — long enough to wedge an MCP tool call past Claude
# Desktop's ~4-min hard kill. 60s comfortably covers a normal
# claim-extraction call (~2-5s) plus headroom for tail latency.
_DEFAULT_TIMEOUT_SECONDS: Final = 60.0
_SCHEMA_NAME: Final = "claim_extraction"


class OpenAILLMClaimExtractor:
    """Structured-output claim extractor over the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.name = f"llm:openai:{model}"
        self._client = client or AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            max_retries=max_retries,
            timeout=timeout,
        )

    async def extract(self, text: str) -> Sequence[Claim]:
        if not text or not text.strip():
            return ()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(text)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": _SCHEMA_NAME,
                        "schema": CLAIM_EXTRACTION_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:
            # The SDK has already exhausted retries; surface as empty
            # rather than letting an LLM-API outage break assist_draft.
            # Log loudly so an operator notices.
            _log.warning("openai claim extractor call failed: %s", exc)
            return ()

        content = response.choices[0].message.content
        if not content:
            _log.warning("openai claim extractor returned empty content")
            return ()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            _log.exception("openai claim extractor returned non-JSON content")
            return ()
        return payload_to_claims(parsed, text=text, model_name=self.name)
