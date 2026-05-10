"""ClaimExtractor tests — pattern coverage, ordering, dedup, search-term shape."""

from __future__ import annotations

import pytest

from research_mcp.claim_extractor import FakeClaimExtractor, SpacyClaimExtractor
from research_mcp.domain import Claim, ClaimExtractor, ClaimType

pytestmark = pytest.mark.unit


# Loading en_core_web_sm takes ~500ms; do it once per session and pass the
# pre-loaded pipeline into every constructor. Real-spaCy tests instead of
# mocks per the project's no-mocks-of-owned-protocols rule (spaCy is third
# party, but the *patterns* are ours and we want to exercise them on the
# real tagger output).
@pytest.fixture(scope="session")
def _nlp():  # type: ignore[no-untyped-def]
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # pragma: no cover — env-specific failure mode
        pytest.skip(f"en_core_web_sm not installed: {exc}")


@pytest.fixture
def extractor(_nlp):  # type: ignore[no-untyped-def]
    return SpacyClaimExtractor(nlp=_nlp)


# ---- protocol conformance ----


def test_spacy_satisfies_protocol(_nlp) -> None:  # type: ignore[no-untyped-def]
    src = SpacyClaimExtractor(nlp=_nlp)
    assert isinstance(src, ClaimExtractor)
    assert src.name == "spacy"


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeClaimExtractor(), ClaimExtractor)
    assert FakeClaimExtractor().name == "fake"


# ---- statistical claims ----


async def test_extracts_percent_change_as_statistical(extractor: SpacyClaimExtractor) -> None:
    text = "The new method achieved a 12% increase in accuracy."
    claims = list(await extractor.extract(text))
    stat = [c for c in claims if c.type == ClaimType.STATISTICAL]
    assert stat, f"expected a STATISTICAL claim; got {[c.type for c in claims]}"
    assert any("12%" in c.text for c in stat)


async def test_extracts_p_value_as_statistical(extractor: SpacyClaimExtractor) -> None:
    text = "The treatment effect was statistically significant (p < 0.001)."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.STATISTICAL for c in claims)


async def test_extracts_correlation_as_statistical(
    extractor: SpacyClaimExtractor,
) -> None:
    text = "We found a strong correlation of r = 0.85 between dosage and response."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.STATISTICAL for c in claims)


# ---- methodological claims ----


async def test_extracts_methodological_claim(extractor: SpacyClaimExtractor) -> None:
    text = "We employed a transformer architecture with multi-head attention."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.METHODOLOGICAL for c in claims)


# ---- comparative claims ----


async def test_extracts_comparative_outperforms(
    extractor: SpacyClaimExtractor,
) -> None:
    text = "Our model outperforms BERT on three benchmarks."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.COMPARATIVE for c in claims)


async def test_extracts_comparative_better_than(
    extractor: SpacyClaimExtractor,
) -> None:
    text = "The proposed approach is more accurate than existing baselines."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.COMPARATIVE for c in claims)


# ---- causal claims ----


async def test_extracts_causal_leads_to(extractor: SpacyClaimExtractor) -> None:
    text = "Sleep deprivation leads to reduced cognitive performance."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.CAUSAL for c in claims)


# ---- theoretical claims ----


async def test_extracts_theoretical_suggests_that(
    extractor: SpacyClaimExtractor,
) -> None:
    text = "These results suggest that attention mechanisms are universal."
    claims = list(await extractor.extract(text))
    assert any(c.type == ClaimType.THEORETICAL for c in claims)


# ---- ordering, dedup, and shape ----


async def test_claims_returned_in_document_order(
    extractor: SpacyClaimExtractor,
) -> None:
    """A calling LLM expects claim[0] to be earlier in the text than claim[1]."""
    text = (
        "We propose a novel attention mechanism. "
        "It outperforms LSTMs by 23%. "
        "These findings suggest that self-attention is universally effective."
    )
    claims = list(await extractor.extract(text))
    starts = [c.start_char for c in claims]
    assert starts == sorted(starts)


async def test_overlapping_matches_are_deduplicated(
    extractor: SpacyClaimExtractor,
) -> None:
    """Multiple regexes can fire on the same span — return one claim."""
    # "improved by 30%" matches both the percentage-change pattern and the
    # generic statistical-significance pattern; we want a single Claim.
    from itertools import pairwise

    text = "Accuracy improved by 30%."
    claims = list(await extractor.extract(text))
    stat_spans = [
        (c.start_char, c.end_char)
        for c in claims
        if c.type == ClaimType.STATISTICAL
    ]
    # Adjacent windows should not overlap; if they do, dedup failed.
    for (a_start, a_end), (b_start, b_end) in pairwise(stat_spans):
        assert a_end <= b_start, f"overlap: ({a_start},{a_end}) and ({b_start},{b_end})"


async def test_extract_returns_empty_for_empty_text(
    extractor: SpacyClaimExtractor,
) -> None:
    assert list(await extractor.extract("")) == []


async def test_extract_returns_empty_for_pure_descriptive_prose(
    extractor: SpacyClaimExtractor,
) -> None:
    """Sentences without claim-trigger patterns should yield zero claims."""
    text = "The sky was blue. The trees swayed gently in the breeze. A bird flew by."
    claims = list(await extractor.extract(text))
    assert claims == []


async def test_each_claim_has_context_window(
    extractor: SpacyClaimExtractor,
) -> None:
    """Context should be a non-empty substring around the claim."""
    text = "The treatment achieved a 47% improvement in remission rates."
    claims = list(await extractor.extract(text))
    assert claims
    for c in claims:
        assert c.context  # non-empty
        assert len(c.context) >= len(c.text)


async def test_each_claim_has_search_terms(extractor: SpacyClaimExtractor) -> None:
    """Search terms drive the citation finder; must be non-empty for non-trivial claims."""
    text = "Multi-head attention outperforms LSTMs on machine translation tasks."
    claims = list(await extractor.extract(text))
    assert claims
    # At least one claim should yield search terms — they're how the
    # citation finder hits an upstream search.
    assert any(c.suggested_search_terms for c in claims)


async def test_start_end_chars_round_trip_to_original_text(
    extractor: SpacyClaimExtractor,
) -> None:
    """start_char/end_char must index into the *original* text, not the
    cleaned/normalized version, so a UI can highlight the claim in place."""
    text = "We achieved a 12% improvement using gradient boosting."
    claims = list(await extractor.extract(text))
    for c in claims:
        assert text[c.start_char : c.end_char] == c.text


async def test_confidence_is_in_unit_interval(
    extractor: SpacyClaimExtractor,
) -> None:
    text = (
        "The drug significantly reduced symptoms (p < 0.01) and outperformed placebo."
    )
    for c in await extractor.extract(text):
        assert 0.0 <= c.confidence <= 1.0


# ---- FakeClaimExtractor ----


async def test_fake_emits_one_statistical_claim_per_sentence_with_a_number() -> None:
    """FakeClaimExtractor is for tests that need a ClaimExtractor but don't
    want to spin up spaCy. Behavior: one STATISTICAL claim per
    sentence containing a digit."""
    fake = FakeClaimExtractor()
    text = "The first sentence has 100 in it. This one has none. Another with 42 here."
    claims = list(await fake.extract(text))
    assert len(claims) == 2
    assert all(c.type == ClaimType.STATISTICAL for c in claims)
    assert claims[0].start_char < claims[1].start_char


async def test_fake_returns_empty_for_text_with_no_numbers() -> None:
    fake = FakeClaimExtractor()
    assert list(await fake.extract("No numbers in this text.")) == []


async def test_fake_returns_empty_for_empty_text() -> None:
    assert list(await FakeClaimExtractor().extract("")) == []


# ---- Claim shape ----


def test_claim_is_immutable() -> None:
    """Claim is a frozen dataclass — mutation must raise."""
    c = Claim(text="x", type=ClaimType.STATISTICAL, confidence=0.5, context="x")
    with pytest.raises((AttributeError, TypeError)):
        c.text = "y"  # type: ignore[misc]
