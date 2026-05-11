"""Shared JSON schema, prompt, and payload-lifting for the LLM citation scorers.

Both `OpenAILLMCitationScorer` and `AnthropicLLMCitationScorer` ask the
model the same question: given a paper's title and abstract and a
specific claim, how relevant is the paper as a citation for that claim?
The schema is intentionally tiny — one float in [0, 1] plus a short
reasoning string — because the LLM's job here is judgment, not
extraction. The four heuristic dimensions (venue / impact / author /
recency) stay data-grounded; the LLM only modulates the total to
reflect semantic fit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from research_mcp.domain.claim import Claim
from research_mcp.domain.paper import Paper

RELEVANCE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevance": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "0-1 score for how well the paper's title + abstract "
                "support the specific claim. 1.0 = paper directly "
                "introduces or evidences the claim; 0.5 = topically "
                "related but not directly supporting; 0.0 = unrelated. "
                "Judge semantic fit only — venue prestige, author "
                "fame, and recency are scored separately."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One short sentence (under 200 characters) naming the "
                "specific overlap or gap between the paper and the "
                "claim. Concrete: cite the matching concept, dataset, "
                "or method — not 'this paper is relevant'."
            ),
        },
    },
    "required": ["relevance", "reasoning"],
}

RELEVANCE_TOOL_NAME: Final = "submit_citation_relevance"
RELEVANCE_TOOL_DESCRIPTION: Final = (
    "Submit a relevance judgment for the supplied paper as a candidate "
    "citation for the supplied claim. Return a relevance score in [0, 1] "
    "and a one-sentence reasoning that names the specific concept, "
    "method, or finding that does (or does not) support the claim."
)

# Abstract-only is enough for relevance judgment; we don't need full
# text for "does this paper say what we want to cite it for?". Keeping
# the prompt small also keeps cost in the cents-per-thousand-calls range.
_MAX_ABSTRACT_CHARS: Final = 4_000


_SYSTEM_PROMPT: Final = (
    "You judge whether a research paper is an appropriate citation for "
    "a specific claim. You are given the paper's title and abstract, "
    "and the claim with its type and surrounding context. Return a "
    "relevance score in [0, 1] reflecting semantic fit:\n"
    "\n"
    "- 0.9-1.0: paper directly introduces, defines, or empirically "
    "evidences the claim.\n"
    "- 0.6-0.8: paper substantively addresses the claim's topic and "
    "supports it indirectly (e.g., a survey covering the claim's area).\n"
    "- 0.3-0.5: same broad field but the paper does not address the "
    "specific claim.\n"
    "- 0.0-0.2: unrelated topic, wrong subfield, or actively contradicts "
    "the claim without being a critique citation.\n"
    "\n"
    "Judge ONLY semantic appropriateness. Venue prestige, author fame, "
    "citation count, and publication date are scored separately and "
    "must NOT enter your relevance judgment. A predatory-venue paper "
    "that exactly evidences the claim still gets a high relevance "
    "score — the heuristic dimensions will down-weight the total."
)


def system_prompt() -> str:
    return _SYSTEM_PROMPT


def user_prompt(paper: Paper, claim: Claim) -> str:
    parts: list[str] = []
    parts.append(f"Paper title: {paper.title.strip()}")
    if paper.venue:
        parts.append(f"Venue: {paper.venue}")
    if paper.published:
        parts.append(f"Published: {paper.published.isoformat()}")
    abstract = (paper.abstract or "").strip()
    if abstract:
        truncated = abstract[:_MAX_ABSTRACT_CHARS]
        if len(abstract) > _MAX_ABSTRACT_CHARS:
            truncated += " [...truncated...]"
        parts.append(f"\nAbstract:\n{truncated}")
    else:
        parts.append("\nAbstract: (not available)")

    parts.append(f"\nClaim type: {claim.type.value}")
    parts.append(f"Claim text: {claim.text.strip()}")
    claim_context = (claim.context or "").strip()
    if claim_context and claim_context != claim.text.strip():
        parts.append(f"Claim context: {claim_context}")
    if claim.suggested_search_terms:
        terms = ", ".join(t for t in claim.suggested_search_terms if t)
        if terms:
            parts.append(f"Search terms tried: {terms}")

    parts.append(
        "\nReturn your judgment via the structured-output tool/JSON shape provided."
    )
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """Lifted relevance signal: a score in [0, 1] plus reasoning."""

    relevance: float
    reasoning: str


def payload_to_relevance(payload: Mapping[str, Any] | None) -> RelevanceJudgment | None:
    """Lift the LLM JSON output into a `RelevanceJudgment`.

    Returns None when the payload is malformed or missing the relevance
    number — callers fall back to the base score and skip the warning.
    """
    if not isinstance(payload, Mapping):
        return None
    raw_relevance = payload.get("relevance")
    if not isinstance(raw_relevance, int | float):
        return None
    relevance = max(0.0, min(1.0, float(raw_relevance)))
    raw_reasoning = payload.get("reasoning")
    reasoning = raw_reasoning.strip() if isinstance(raw_reasoning, str) else ""
    return RelevanceJudgment(relevance=relevance, reasoning=reasoning)
