"""Regex patterns for claim detection, by type.

Each `<TYPE>_PATTERNS` constant is a tuple of regex strings, all
case-insensitive at match time. Patterns lean conservative — false
positives on academic prose dilute citation suggestions, and a missed
claim is recoverable (the user can re-prompt) while a wrong-typed
claim leads the citation finder to bad candidates.

Patterns that fire on the same span are deduplicated downstream by
`_dedup_claims`. Order within a tuple doesn't matter for correctness
but matters for confidence ranking when two same-type matches overlap
— more specific patterns sit before more general ones so the dedup
keeps the high-precision one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from research_mcp.domain.claim import ClaimType

# ---- raw patterns, by type ----

STATISTICAL_PATTERNS: tuple[str, ...] = (
    # Percent change with direction word
    r"\b\d+(?:\.\d+)?%\s+(?:increase|decrease|improvement|reduction|growth|decline)",
    r"(?:increased|decreased|improved|reduced|grew|declined)\s+by\s+\d+(?:\.\d+)?%",
    # Correlation strength
    r"(?:correlation|correlates?)\s+(?:of|between|with)\s+[rR]?\s*=?\s*[-+]?\d*\.?\d+",
    r"(?:significant|strong|weak|moderate)\s+correlation",
    # Statistical significance
    r"\b[pP]\s*[<>=]\s*0?\.\d+",
    r"statistically\s+significant",
    r"confidence\s+interval",
    # Effect sizes
    r"effect\s+size\s+(?:of|=)\s*\d*\.?\d+",
    r"Cohen'?s\s+d\s*=\s*\d*\.?\d+",
    # Sample sizes
    r"\b[nN]\s*=\s*\d+",
    r"sample\s+(?:size|of)\s+\d+",
)

METHODOLOGICAL_PATTERNS: tuple[str, ...] = (
    r"(?:we|this\s+study)\s+(?:used?|employ(?:ed)?|implement(?:ed)?|applied?|develop(?:ed)?)",
    r"(?:using|employing|implementing|applying)\s+(?:the\s+)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:method|technique|approach|algorithm)",
    r"(?:novel|new|proposed|modified)\s+(?:method|technique|approach|algorithm)",
    r"based\s+on\s+(?:the\s+)?[A-Z][A-Za-z]+",
)

COMPARATIVE_PATTERNS: tuple[str, ...] = (
    r"(?:outperform(?:s|ed)?|exceed(?:s|ed)?)",
    r"(?:better|superior|inferior)\s+(?:than|to)",
    r"(?:more|less)\s+(?:accurate|efficient|effective|robust)\s+than",
    r"(?:compared\s+to|in\s+comparison\s+with|versus|vs\.?)\s+[A-Za-z]+",
    r"(?:highest|lowest|best|worst)\s+(?:performance|accuracy|results?)",
    r"(?:state-of-the-art|SOTA|baseline)\s+(?:performance|results?)",
)

THEORETICAL_PATTERNS: tuple[str, ...] = (
    r"(?:suggest(?:s|ed)?|indicate(?:s|d)?|imply|implies|demonstrate(?:s|d)?)\s+that",
    r"(?:evidence|results?)\s+(?:for|of|supporting)",
    r"(?:consistent|inconsistent)\s+with\s+(?:the\s+)?[A-Za-z]+\s+(?:theory|hypothesis|model)",
    r"(?:support(?:s|ed)?|contradict(?:s|ed)?|confirm(?:s|ed)?)\s+(?:the\s+)?(?:hypothesis|theory)",
    r"(?:mechanism|process|phenomenon)\s+(?:of|for|behind)",
)

CAUSAL_PATTERNS: tuple[str, ...] = (
    r"(?:caus(?:es?|ed|ing)|leads?\s+to|results?\s+in|due\s+to|because\s+of)",
    r"(?:effect|impact|influence)\s+(?:of|on)",
    r"(?:induces?|triggers?|promotes?|inhibits?)",
)


# ---- per-type confidence floors ----

# Confidence per type reflects how reliable the regexes are on academic
# prose: percent-change patterns rarely false-fire (0.9), comparative
# triggers like "outperforms" sometimes appear in unrelated contexts
# (0.85), and theoretical/causal triggers are softer (0.8).
_TYPE_CONFIDENCE: dict[ClaimType, float] = {
    ClaimType.STATISTICAL: 0.9,
    ClaimType.METHODOLOGICAL: 0.85,
    ClaimType.COMPARATIVE: 0.85,
    ClaimType.THEORETICAL: 0.8,
    ClaimType.CAUSAL: 0.8,
}

# Maps each type to its compiled pattern set; populated lazily.
_PATTERNS_BY_TYPE: dict[ClaimType, tuple[str, ...]] = {
    ClaimType.STATISTICAL: STATISTICAL_PATTERNS,
    ClaimType.METHODOLOGICAL: METHODOLOGICAL_PATTERNS,
    ClaimType.COMPARATIVE: COMPARATIVE_PATTERNS,
    ClaimType.THEORETICAL: THEORETICAL_PATTERNS,
    ClaimType.CAUSAL: CAUSAL_PATTERNS,
}


_compiled_cache: dict[ClaimType, tuple[re.Pattern[str], ...]] = {}


def _compile(claim_type: ClaimType) -> tuple[re.Pattern[str], ...]:
    if claim_type not in _compiled_cache:
        _compiled_cache[claim_type] = tuple(
            re.compile(p, re.IGNORECASE) for p in _PATTERNS_BY_TYPE[claim_type]
        )
    return _compiled_cache[claim_type]


def confidence_for(claim_type: ClaimType) -> float:
    """Return the per-type confidence floor for pattern-derived claims."""
    return _TYPE_CONFIDENCE.get(claim_type, 0.7)


def iter_pattern_matches(
    text: str,
) -> Iterator[tuple[ClaimType, re.Match[str]]]:
    """Yield `(type, match)` for every pattern hit in `text`, all types."""
    for claim_type in _PATTERNS_BY_TYPE:
        for pattern in _compile(claim_type):
            for match in pattern.finditer(text):
                yield claim_type, match
