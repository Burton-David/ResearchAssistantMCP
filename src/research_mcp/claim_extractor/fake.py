"""Deterministic `ClaimExtractor` for tests.

Behavior: split text on sentence-ending punctuation, emit one
`STATISTICAL` claim per sentence containing a digit. Search terms are
the alphanumeric word tokens in the sentence. Confidence is fixed at
0.5 — high enough to not be filtered out by hypothetical thresholds,
low enough to never be confused with a real extractor's output.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from research_mcp.domain.claim import Claim, ClaimType

_SENTENCE_RE: Final = re.compile(r"[^.!?]+[.!?]+")
_HAS_DIGIT_RE: Final = re.compile(r"\d")
_WORD_RE: Final = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


class FakeClaimExtractor:
    """Returns one STATISTICAL claim per digit-bearing sentence."""

    name: str = "fake"

    async def extract(self, text: str) -> Sequence[Claim]:
        if not text or not text.strip():
            return ()
        claims: list[Claim] = []
        for match in _SENTENCE_RE.finditer(text):
            sentence = match.group().strip()
            if not _HAS_DIGIT_RE.search(sentence):
                continue
            words = tuple(_WORD_RE.findall(sentence))[:5]
            claims.append(
                Claim(
                    text=sentence,
                    type=ClaimType.STATISTICAL,
                    confidence=0.5,
                    context=sentence,
                    suggested_search_terms=words,
                    start_char=match.start() + (len(match.group()) - len(match.group().lstrip())),
                    end_char=match.end(),
                )
            )
        return tuple(claims)
