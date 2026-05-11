"""Shared JSON schema + prompt for the LLM-backed paper analyzers.

Both `OpenAILLMPaperAnalyzer` and `AnthropicLLMPaperAnalyzer` request
the same structured output shape; this module owns the schema so any
field rename / addition is one-shot. The schema is intentionally
shaped to mirror `PaperAnalysis` so the analyzer's job is just
"fill these fields, return JSON."
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Final

from research_mcp.domain.paper import Paper
from research_mcp.domain.paper_analyzer import (
    ALL_ANALYSIS_KINDS,
    AnalysisKind,
    PaperAnalysis,
)

# The schema that the LLM is asked to fill. `additionalProperties: false`
# locks the output shape; both OpenAI's structured outputs and Anthropic's
# tool use respect this.
ANALYSIS_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-3 sentence overview of the paper.",
        },
        "key_contributions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet-style list of distinct contributions.",
        },
        "methodology": {
            "type": "string",
            "description": "How the work was done (experimental design, "
            "approach, analytical framework).",
        },
        "technical_approach": {
            "type": "string",
            "description": "Specific techniques, algorithms, or models used.",
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Limitations the authors acknowledge or that are "
            "evident from the paper.",
        },
        "future_directions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Future-work suggestions explicitly mentioned.",
        },
        "datasets_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Named datasets used; preserve the exact names.",
        },
        "metrics_reported": {
            # OpenAI's structured-output strict mode REJECTS schemas with
            # open-ended `additionalProperties: <type>`; only `additional
            # Properties: false` is allowed. We model the metrics as a
            # list of (name, value) pairs and convert to a dict in the
            # parser. Same expressive power, conformant to strict mode.
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["name", "value"],
            },
            "description": (
                "Headline numeric metrics as name/value pairs (e.g., "
                "[{name: 'accuracy', value: 0.873}, {name: 'bleu', "
                "value: 28.4}])."
            ),
        },
        "baselines_compared": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Named baselines/competitors compared against.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Self-assessed confidence: how much of the "
            "structure was actually present in the input vs. guessed.",
        },
    },
    "required": [
        "summary",
        "key_contributions",
        "methodology",
        "technical_approach",
        "limitations",
        "future_directions",
        "datasets_used",
        "metrics_reported",
        "baselines_compared",
        "confidence",
    ],
}

ANALYSIS_TOOL_NAME: Final = "submit_paper_analysis"
ANALYSIS_TOOL_DESCRIPTION: Final = (
    "Submit a structured analysis of the supplied research paper. "
    "Fill every field; for fields you cannot determine from the input, "
    "use an empty array / empty object / empty string and lower the "
    "confidence accordingly. Do not invent results."
)

# Truncate paper text to keep prompts under a reasonable token budget.
# At ~4 chars/token, 60K chars ≈ 15K tokens — cheap on 4o-mini /
# claude-haiku and well within their context windows.
_MAX_PAPER_CHARS: Final = 60_000


_SYSTEM_PROMPT: Final = (
    "You are a careful research-paper analyst. Read the paper and "
    "fill in a structured analysis using ONLY information present "
    "in the paper text. Do not invent datasets, metrics, or baselines "
    "the paper does not name. When the paper omits a field (e.g., no "
    "limitations section), return an empty value for that field and "
    "lower the confidence. Be specific over comprehensive."
)


def system_prompt() -> str:
    return _SYSTEM_PROMPT


def user_prompt(paper: Paper, kinds: Sequence[AnalysisKind]) -> str:
    """Render the user-facing prompt with the paper text and target kinds.

    `kinds` empty → all kinds. Listing the requested kinds in the
    prompt nudges the model to omit non-asked-for fields.
    """
    requested = list(kinds) if kinds else list(ALL_ANALYSIS_KINDS)
    parts: list[str] = []
    parts.append(f"Title: {paper.title.strip()}")
    if paper.authors:
        author_names = ", ".join(a.name for a in paper.authors[:6])
        suffix = "" if len(paper.authors) <= 6 else f" + {len(paper.authors) - 6} more"
        parts.append(f"Authors: {author_names}{suffix}")
    if paper.venue:
        parts.append(f"Venue: {paper.venue}")
    if paper.published:
        parts.append(f"Published: {paper.published.isoformat()}")
    if paper.abstract:
        parts.append(f"\nAbstract:\n{paper.abstract.strip()}")
    if paper.full_text:
        truncated = paper.full_text[:_MAX_PAPER_CHARS]
        if len(paper.full_text) > _MAX_PAPER_CHARS:
            truncated += "\n[...truncated...]"
        parts.append(f"\nFull text:\n{truncated}")

    parts.append(
        "\nRequested analysis fields: " + ", ".join(k.value for k in requested) + "."
    )
    parts.append(
        "Return your analysis via the structured-output tool/JSON shape provided."
    )
    return "\n".join(parts)


def text_for_paper(paper: Paper) -> str:
    """Concatenated representation of paper text for empty-input checks.

    `analyze` returns confidence=0.0 with all fields blank when this is
    empty — the LLM has nothing to chew on.
    """
    bits: list[str] = []
    if paper.title:
        bits.append(paper.title.strip())
    if paper.abstract:
        bits.append(paper.abstract.strip())
    if paper.full_text:
        bits.append(paper.full_text.strip())
    return "\n\n".join(b for b in bits if b)


def payload_to_analysis(
    payload: dict[str, Any] | None,
    *,
    paper_id: str,
    model_name: str,
) -> PaperAnalysis:
    """Lift the LLM's JSON output into the immutable domain object.

    Shared by both `OpenAILLMPaperAnalyzer` and `AnthropicLLMPaperAnalyzer`
    — the SDK call mechanics differ (response_format vs tool_use), but
    the JSON shape is identical and the conversion is the same.
    `metrics_reported` is a list of `{name, value}` pairs in the schema
    (strict mode rejects open-ended additionalProperties); the parser
    converts to a dict for the domain object.

    Returns a blank-but-valid `PaperAnalysis` for any malformed input —
    callers convert SDK errors / non-JSON content into this shape rather
    than raising.
    """
    if not isinstance(payload, dict):
        return PaperAnalysis(paper_id=paper_id, model=model_name)
    cleaned_metrics = _metrics_from_pairs(payload.get("metrics_reported"))
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


def _metrics_from_pairs(value: Any) -> dict[str, float]:
    """Convert `[{name, value}, ...]` from the LLM back to `{name: value}`.

    The schema models metrics as a list of pairs because OpenAI strict
    mode rejects `additionalProperties: <type>`; this is the inverse
    conversion run on the LLM response payload.
    """
    if not isinstance(value, list):
        return {}
    out: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        val = item.get("value")
        if isinstance(name, str) and name.strip() and isinstance(val, int | float):
            out[name.strip()] = float(val)
    return out
