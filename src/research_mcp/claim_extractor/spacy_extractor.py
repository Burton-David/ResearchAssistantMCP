"""spaCy-backed `ClaimExtractor` — pattern matching + noun-chunk search terms.

The pipeline:

1. Run `text` through spaCy once for tokenization + POS + sentence
   segmentation + NER. Cheap on the small `en_core_web_sm` model
   (~5ms / KB).
2. Apply the regex patterns from `_patterns.py` and emit a `Claim` per
   match, dropping confidence-by-type.
3. For each claim, derive `suggested_search_terms` from the spaCy
   noun chunks that fall within the claim's *context window* (not
   just the claim span — a one-word claim like "outperforms" has
   nothing to search, but its surrounding noun chunks do).
4. Deduplicate by overlapping spans, keeping higher confidence.
5. Sort by `start_char` so callers see claims in document order.

`extract` runs the synchronous spaCy pipeline inside `asyncio.to_thread`
so a slow upstream document doesn't block the event loop. spaCy's
`Doc.__init__` releases the GIL during its rust-backed work, so this
is genuinely parallelizable.

Lazy import of spacy: a fresh checkout without the `[claim-extraction]`
extra installed will still import the rest of the package; the error
is raised at constructor time with a clear hint.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final

from research_mcp.claim_extractor._patterns import (
    confidence_for,
    iter_pattern_matches,
)
from research_mcp.domain.claim import Claim, ClaimType

if TYPE_CHECKING:  # pragma: no cover
    from spacy.language import Language
    from spacy.tokens import Doc

_DEFAULT_MODEL: Final = "en_core_web_sm"
_DEFAULT_CONTEXT_CHARS: Final = 100
_MAX_SEARCH_TERMS: Final = 5
_HOW_TO_INSTALL: Final = (
    "Install the optional extra: pip install 'research-mcp[claim-extraction]' "
    "and run: python -m spacy download en_core_web_sm"
)


class SpacyClaimExtractor:
    """Pattern + spaCy claim extractor."""

    name: str = "spacy"

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        nlp: Language | None = None,
        context_chars: int = _DEFAULT_CONTEXT_CHARS,
    ) -> None:
        if nlp is None:
            nlp = _load_spacy(model_name)
        self._nlp = nlp
        self._context = context_chars

    async def extract(self, text: str) -> Sequence[Claim]:
        if not text or not text.strip():
            return ()
        # spaCy's pipeline is synchronous and CPU-bound; offload so a
        # 10K-char document doesn't stall the event loop.
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> tuple[Claim, ...]:
        doc = self._nlp(text)
        raw_claims: list[Claim] = []
        for claim_type, match in iter_pattern_matches(text):
            start, end = match.start(), match.end()
            context = _make_context(text, start, end, self._context)
            claim = Claim(
                text=match.group(),
                type=claim_type,
                confidence=confidence_for(claim_type),
                context=context,
                start_char=start,
                end_char=end,
                suggested_search_terms=_search_terms_for(
                    doc, claim_type, start, end, self._context
                ),
            )
            raw_claims.append(claim)
        deduped = _dedup_claims(raw_claims)
        return tuple(sorted(deduped, key=lambda c: c.start_char))


def _load_spacy(model_name: str) -> Language:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover — environment-specific
        raise RuntimeError(
            f"spaCy is required for SpacyClaimExtractor. {_HOW_TO_INSTALL}"
        ) from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. {_HOW_TO_INSTALL}"
        ) from exc


def _make_context(text: str, start: int, end: int, window: int) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return text[lo:hi]


def _search_terms_for(
    doc: Doc, claim_type: ClaimType, start: int, end: int, window: int
) -> tuple[str, ...]:
    """Pull up to `_MAX_SEARCH_TERMS` noun-chunk-derived search terms.

    For STATISTICAL claims we additionally include any percentages /
    sample-size tokens that sit inside the claim span (those make
    distinctive query refinements). For METHODOLOGICAL claims we
    favor proper-noun chunks (method names like "Adam" or "BERT").
    """
    lo = max(0, start - window)
    hi = min(len(doc.text), end + window)
    seen: dict[str, None] = {}

    # Noun chunks sitting in the context window
    for chunk in doc.noun_chunks:
        if chunk.start_char >= hi or chunk.end_char <= lo:
            continue
        clean = _clean_chunk(chunk.text)
        if clean and clean.lower() not in _STOP_CHUNKS:
            seen.setdefault(clean, None)

    # Proper-noun emphasis for methodological claims
    if claim_type == ClaimType.METHODOLOGICAL:
        for token in doc:
            if token.idx < lo or token.idx + len(token.text) > hi:
                continue
            if token.pos_ == "PROPN" and token.is_alpha:
                seen.setdefault(token.text, None)

    return tuple(list(seen.keys())[:_MAX_SEARCH_TERMS])


def _clean_chunk(text: str) -> str:
    """Trim leading determiners ('the dominant model' → 'dominant model')."""
    parts = text.split()
    while parts and parts[0].lower() in _LEADING_DETERMINERS:
        parts.pop(0)
    return " ".join(parts).strip()


_LEADING_DETERMINERS: frozenset[str] = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "our", "their", "its"}
)
_STOP_CHUNKS: frozenset[str] = frozenset(
    {"we", "i", "they", "it", "this", "that", "these", "those", "you"}
)


def _dedup_claims(claims: Iterable[Claim]) -> list[Claim]:
    """Sort by (start_char, -confidence), drop spans overlapping a kept one.

    Two patterns can fire on the same text — e.g., "improved by 23%"
    matches both the percent-change and the directional-change rules.
    Without this, the user sees duplicate suggestions for one underlying
    statement. Keeping the higher-confidence one first means we always
    drop the weaker variant on overlap.
    """
    ordered = sorted(claims, key=lambda c: (c.start_char, -c.confidence))
    kept: list[Claim] = []
    last_end = -1
    for claim in ordered:
        if claim.start_char < last_end:
            continue
        kept.append(claim)
        last_end = claim.end_char
    return kept
