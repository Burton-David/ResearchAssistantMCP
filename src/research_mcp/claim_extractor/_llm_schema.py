"""Shared schema, prompt, and payload-lifting for the LLM claim extractors.

Both `OpenAILLMClaimExtractor` and `AnthropicLLMClaimExtractor` ask the
model for the same JSON shape, with the same prompt, and convert the
response to `Claim[]` the same way. The LLM-call mechanics differ (one
uses response_format, the other tool_use); the surrounding work is
identical and lives here so any prompt change is one-shot.

Span anchoring: the model returns each claim's verbatim text; we
compute `start_char` / `end_char` post-hoc with `str.find` against the
original input. LLMs hallucinate offsets; they don't hallucinate the
exact text they were just asked to copy. Falling back to (0, 0) when
find fails preserves the claim rather than dropping it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from research_mcp.domain.claim import Claim, ClaimType

_log = logging.getLogger(__name__)

# Per-claim object schema. Required keys are the user-facing fields a
# downstream caller (CitationService, find_citations) actually uses;
# optional fields like keywords would just add slop without helping
# precision.
_CLAIM_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {
            "type": "string",
            "description": (
                "The claim, verbatim from the input — copy character-for-"
                "character. The downstream pipeline finds this string in "
                "the original text to compute offsets, so paraphrasing "
                "breaks the highlight."
            ),
        },
        "type": {
            "type": "string",
            "enum": [t.value for t in ClaimType],
            "description": (
                "Statistical (numbers/percentages/p-values), "
                "methodological (techniques/algorithms), comparative "
                "(outperforms / better than), causal (X causes Y), "
                "theoretical (suggests/implies), factual (named "
                "entities, definitions), evaluative (importance / SOTA)."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "0-1 confidence that this claim genuinely needs a "
                "citation. Pure description without an empirical "
                "claim should not appear; if you're unsure whether a "
                "sentence makes a citable claim, lower the confidence "
                "rather than dropping it."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "The surrounding sentence or short paragraph the claim "
                "lives in, copied verbatim from the input. Used to "
                "disambiguate which paper the citation should point at."
            ),
        },
        "suggested_search_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "2-5 keywords that would surface relevant papers in a "
                "literature search. For comparative claims include both "
                "compared methods; for methodological claims include "
                "the technique name; for statistical claims include "
                "the metric and domain."
            ),
        },
    },
    "required": [
        "text",
        "type",
        "confidence",
        "context",
        "suggested_search_terms",
    ],
}

# Top-level schema. `claims: []` is a legal output (no citations needed
# in pure description). additionalProperties:false is required for
# OpenAI structured outputs.
CLAIM_EXTRACTION_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {"type": "array", "items": _CLAIM_SCHEMA},
    },
    "required": ["claims"],
}

CLAIM_EXTRACTION_TOOL_NAME: Final = "submit_claims"
CLAIM_EXTRACTION_TOOL_DESCRIPTION: Final = (
    "Submit the list of claims extracted from the input text. Each "
    "claim's `text` MUST be copied verbatim from the input. Return an "
    "empty array when the input contains no claims that need citations."
)


_SYSTEM_PROMPT: Final = (
    "You are an expert at identifying claims in scientific writing that "
    "need citations. Read the user-supplied text and return a list of "
    "claims, each with: a verbatim copy of the claim from the input, a "
    "type, confidence in [0, 1], the surrounding sentence as context, "
    "and 2-5 search terms.\n"
    "\n"
    "What counts as a claim that needs a citation:\n"
    "- Statistical claims: percentages, p-values, sample sizes, "
    "correlations, effect sizes.\n"
    "- Methodological claims: 'we used X', 'based on the Y method'.\n"
    "- Comparative claims: 'outperforms Z', 'is more accurate than W'.\n"
    "- Causal claims: 'X causes Y', 'leads to', 'results in'.\n"
    "- Theoretical claims: 'suggests that', 'evidence for', 'consistent "
    "with the X hypothesis'.\n"
    "- Factual claims: specific named results, datasets, prior findings.\n"
    "\n"
    "What does NOT count: pure description ('the sky is blue'), the "
    "author's own ongoing thinking ('we now turn to...'), or "
    "purely-rhetorical asides. When uncertain, include the claim with "
    "lower confidence rather than dropping it.\n"
    "\n"
    "Return text spans EXACTLY as they appear in the input. The "
    "downstream pipeline anchors highlights and search-term extraction "
    "by string match against the original text — paraphrased spans get "
    "dropped to (0, 0) offsets and lose their UI anchor."
)


def system_prompt() -> str:
    return _SYSTEM_PROMPT


def user_prompt(text: str) -> str:
    """The user-facing prompt is just the input text plus a closing
    instruction. Keep the prompt boring; the system message carries
    the schema and rules."""
    return (
        f"Input text:\n{text}\n\n"
        "Return your extraction via the structured-output tool/JSON "
        "shape provided."
    )


@dataclass(frozen=True, slots=True)
class _RawClaim:
    """Pre-anchoring projection of a single LLM-returned claim."""

    text: str
    type: ClaimType
    confidence: float
    context: str
    suggested_search_terms: tuple[str, ...]


def payload_to_claims(
    payload: Mapping[str, Any] | None,
    *,
    text: str,
    model_name: str,
) -> tuple[Claim, ...]:
    """Lift the LLM JSON output into ordered, anchored `Claim` objects.

    Steps:
      1. Validate top-level shape; missing/garbled → empty tuple.
      2. Coerce each item to a `_RawClaim`.
      3. Anchor each claim's `text` to the original input via
         `str.find`. Misses fall back to (0, 0) so a hallucinated
         span doesn't lose the claim.
      4. Sort by start_char so callers see document order.

    `model_name` flows into Claim.metadata so a downstream consumer
    can tell which extractor produced the claim — useful when the
    server is hot-swapped between spaCy and an LLM extractor.
    """
    if not isinstance(payload, Mapping):
        return ()
    items = payload.get("claims") or []
    if not isinstance(items, list):
        return ()

    raw_claims: list[_RawClaim] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        coerced = _coerce_raw(item)
        if coerced is None:
            continue
        raw_claims.append(coerced)

    metadata: Mapping[str, str] = MappingProxyType({"extractor": model_name})
    anchored: list[Claim] = []
    for raw in raw_claims:
        start, end = _anchor_span(text, raw.text)
        anchored.append(
            Claim(
                text=raw.text,
                type=raw.type,
                confidence=raw.confidence,
                context=raw.context or raw.text,
                suggested_search_terms=raw.suggested_search_terms,
                start_char=start,
                end_char=end,
                metadata=metadata,
            )
        )
    anchored.sort(key=lambda c: (c.start_char, c.end_char))
    return tuple(anchored)


def _coerce_raw(item: Mapping[str, Any]) -> _RawClaim | None:
    text = _pick_str(item.get("text"))
    if not text:
        return None
    raw_type = _pick_str(item.get("type")) or ""
    try:
        claim_type = ClaimType(raw_type)
    except ValueError:
        # Unknown enum value: fall back to FACTUAL rather than drop the
        # claim. The LLM identified something worth citing; the type
        # mislabel is recoverable downstream.
        _log.debug("LLM returned unknown claim type %r; falling back to FACTUAL", raw_type)
        claim_type = ClaimType.FACTUAL
    confidence = _pick_confidence(item.get("confidence"))
    context = _pick_str(item.get("context")) or ""
    terms = _pick_str_tuple(item.get("suggested_search_terms"))
    return _RawClaim(
        text=text,
        type=claim_type,
        confidence=confidence,
        context=context,
        suggested_search_terms=terms,
    )


def _anchor_span(text: str, claim_text: str) -> tuple[int, int]:
    """Find `claim_text` in `text` and return (start, end). Hallucination
    fallback: (0, 0) so the claim survives but has no UI anchor."""
    if not claim_text:
        return 0, 0
    idx = text.find(claim_text)
    if idx < 0:
        # Try a less-strict match: collapse whitespace runs, since the
        # LLM sometimes normalizes them. If that also fails, give up.
        normalized_input = " ".join(text.split())
        normalized_claim = " ".join(claim_text.split())
        idx = normalized_input.find(normalized_claim)
        if idx < 0:
            return 0, 0
        # Indices in normalized space don't translate back; the claim
        # carries the LLM's text but we can't anchor it in the original.
        return 0, 0
    return idx, idx + len(claim_text)


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
