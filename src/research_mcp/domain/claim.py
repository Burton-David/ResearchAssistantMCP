"""ClaimExtractor protocol: draft text → typed claims that need citations.

The citation-assistant mission starts here. A user pastes a draft paragraph;
the extractor identifies sentences that make verifiable claims (statistics,
methodology, comparisons), classifies each, and emits suggested search
terms that downstream services use to find candidate citations.

Implementations differ in how they detect claims:
  * `SpacyClaimExtractor` — pattern-matching + spaCy NER. Fast, no API
    cost, ~80% precision on academic prose.
  * `FakeClaimExtractor` — deterministic stub for tests.
  * Future: an LLM-based extractor with higher precision but real cost
    per call.

The protocol stays small on purpose. The intelligence (which patterns,
which NER model, which prompt) lives in implementations; the protocol
just promises "string in, list of typed claims out."
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable


class ClaimType(StrEnum):
    """The kind of evidence a claim asserts.

    Different types want different citations: a STATISTICAL claim cites
    a study; a METHODOLOGICAL claim cites the paper that introduced the
    method; a COMPARATIVE claim cites both compared baselines. The
    citation finder uses this to rank candidates differently per type.
    """

    STATISTICAL = "statistical"
    """Numeric claims: percentages, correlations, p-values, sample sizes."""

    METHODOLOGICAL = "methodological"
    """Claims about techniques, algorithms, or experimental approaches."""

    COMPARATIVE = "comparative"
    """X outperforms / is better than / matches Y."""

    THEORETICAL = "theoretical"
    """Claims about mechanisms, hypotheses, or theoretical implications."""

    CAUSAL = "causal"
    """X causes / leads to / results in Y."""

    FACTUAL = "factual"
    """Empirical claims that don't fit the above (named entities,
    historical facts, definitions)."""

    EVALUATIVE = "evaluative"
    """Quality / importance claims ('a major advance', 'state of the
    art'). Hardest to cite well."""


@dataclass(frozen=True, slots=True)
class Claim:
    """A claim extracted from draft text that may need a citation.

    `text` is the verbatim claim. `context` is the surrounding sentence
    or paragraph for disambiguation. `suggested_search_terms` are the
    keywords (typically noun chunks) the extractor thinks would surface
    relevant papers; the citation finder uses them as the upstream
    query.

    `start_char` and `end_char` are character offsets into the original
    draft, so a UI / agent can highlight the claim in place.
    """

    text: str
    type: ClaimType
    confidence: float
    """In [0, 1]. Pattern-based extractors set this from pattern strength;
    LLM-based extractors can use logprobs."""

    context: str
    suggested_search_terms: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    start_char: int = 0
    end_char: int = 0
    metadata: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@runtime_checkable
class ClaimExtractor(Protocol):
    """A claim extractor.

    Implementations must be safe to call concurrently — the citation
    assistant may extract claims from multiple drafts in parallel.
    """

    name: str
    """Human-readable identifier for telemetry / library_status."""

    async def extract(self, text: str) -> Sequence[Claim]:
        """Return claims extracted from `text`.

        Order is significant: claims should be returned in the order they
        appear in `text` so a calling LLM can map them back to source
        positions without reading `start_char`.

        Empty list is valid: short texts, lists of references, or
        purely descriptive prose may have no claims that need citation.
        """
        ...
