"""H-index plumbing: SemanticScholarSource.fetch_h_index + score_authors helper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from research_mcp.citation_scorer._author import (
    _AUTHOR_MAX,
    _TIERS,
    _tier_multiplier,
    score_authors,
)
from research_mcp.citation_scorer._field import Field
from research_mcp.domain.paper import Author, Paper
from research_mcp.errors import SourceUnavailable
from research_mcp.sources.semantic_scholar import SemanticScholarSource

pytestmark = pytest.mark.unit


# ---- SemanticScholarSource.fetch_h_index ----


def _build_source(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
) -> SemanticScholarSource:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return SemanticScholarSource(
        cache_dir=tmp_path / "cache",
        # Speed: the no-key tier has a 1.0s default interval; override
        # to a near-zero positive value (the rate limiter requires
        # strictly positive intervals) so the test isn't blocked.
        min_interval_seconds=0.001,
        client=client,
    )


async def test_fetch_h_index_returns_int(tmp_path: Path) -> None:
    """Happy path: S2 returns `{ "hIndex": 42 }`; we return 42."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graph/v1/author/12345"
        assert dict(request.url.params).get("fields") == "hIndex"
        return httpx.Response(200, json={"authorId": "12345", "hIndex": 42})

    src = _build_source(tmp_path, handler)
    try:
        h = await src.fetch_h_index("12345")
    finally:
        await src.aclose()
    assert h == 42


async def test_fetch_h_index_returns_none_for_404(tmp_path: Path) -> None:
    """Unknown author id → 404 → None, NOT SourceUnavailable. Matches
    `fetch()` semantics: 404 is "this id doesn't exist," not a transient
    failure to surface to the user."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Author not found"})

    src = _build_source(tmp_path, handler)
    try:
        h = await src.fetch_h_index("nonexistent_id")
    finally:
        await src.aclose()
    assert h is None


async def test_fetch_h_index_returns_none_for_missing_field(tmp_path: Path) -> None:
    """S2 occasionally returns an author record with no `hIndex` field
    (very new accounts, deleted profiles). Treat as no signal, not error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"authorId": "12345"})  # no hIndex

    src = _build_source(tmp_path, handler)
    try:
        h = await src.fetch_h_index("12345")
    finally:
        await src.aclose()
    assert h is None


async def test_fetch_h_index_returns_none_for_empty_id(tmp_path: Path) -> None:
    """Blank id shouldn't hit the network — short-circuit to None."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    src = _build_source(tmp_path, handler)
    try:
        assert await src.fetch_h_index("") is None
        assert await src.fetch_h_index("   ") is None
    finally:
        await src.aclose()
    assert calls == 0


async def test_fetch_h_index_propagates_5xx_as_source_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 5xx is a transient outage — surface SourceUnavailable
    so the caller can warn the user vs. silently dropping the signal."""
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep",
        _no_sleep,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream busy")

    src = _build_source(tmp_path, handler)
    try:
        with pytest.raises(SourceUnavailable) as exc_info:
            await src.fetch_h_index("12345")
    finally:
        await src.aclose()
    assert exc_info.value.source_name == "semantic_scholar"


async def test_fetch_h_index_is_cached(tmp_path: Path) -> None:
    """Repeated lookups for the same author shouldn't keep hitting S2 —
    the same disk cache that backs paper lookups handles this."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"hIndex": 30})

    src = _build_source(tmp_path, handler)
    try:
        first = await src.fetch_h_index("12345")
        second = await src.fetch_h_index("12345")
    finally:
        await src.aclose()
    assert first == second == 30
    assert calls == 1  # second call served from disk cache


# ---- score_authors helper ----


def _paper(authors: tuple[Author, ...]) -> Paper:
    return Paper(id="t:1", title="t", abstract="", authors=authors)


async def test_score_authors_placeholder_when_no_authors() -> None:
    score, factor = await score_authors(_paper(()), Field.CS, lookup=None)
    assert score == _AUTHOR_MAX * 0.5  # placeholder
    assert "No author metadata" in factor


async def test_score_authors_placeholder_when_lookup_is_none() -> None:
    paper = _paper((Author("A", s2_id="1"),))
    score, factor = await score_authors(paper, Field.CS, lookup=None)
    assert score == _AUTHOR_MAX * 0.5
    assert "not configured" in factor


async def test_score_authors_placeholder_when_authors_have_no_s2_id() -> None:
    """ArXiv-only papers come through without `s2_id` populated.
    score_authors should fall back to the placeholder rather than
    treating "no signal" as "h-index zero."""""
    paper = _paper((Author("A"), Author("B")))
    score, _ = await score_authors(paper, Field.CS, lookup=_stub_lookup({}))
    assert score == _AUTHOR_MAX * 0.5


async def test_score_authors_placeholder_when_all_lookups_return_none() -> None:
    """All authors have s2_id but the endpoint returns None for each
    (very-new accounts, removed profiles). Treat as no signal."""
    paper = _paper((Author("A", s2_id="1"), Author("B", s2_id="2")))
    score, factor = await score_authors(
        paper, Field.CS, lookup=_stub_lookup({"1": None, "2": None})
    )
    assert score == _AUTHOR_MAX * 0.5
    assert "h-index unavailable" in factor


async def test_score_authors_uses_max_h_index_across_authors() -> None:
    """The "strongest author" signal — a paper's senior researcher's
    h-index dominates. Authors A (h=5), B (h=30), C (h=12) → use 30."""
    paper = _paper(
        (
            Author("A", s2_id="1"),
            Author("B", s2_id="2"),
            Author("C", s2_id="3"),
        )
    )
    lookup = _stub_lookup({"1": 5, "2": 30, "3": 12})
    score, factor = await score_authors(paper, Field.CS, lookup=lookup)
    # CS tiers: 30 ≥ 25 → multiplier 0.8 → 16
    assert score == _AUTHOR_MAX * 0.8
    assert "30" in factor  # max h-index appears in the explanation


async def test_score_authors_skips_individual_lookup_failures() -> None:
    """One author's lookup raising (timeout, malformed response) shouldn't
    zero out the dimension when other authors have good data."""
    paper = _paper(
        (Author("A", s2_id="1"), Author("B", s2_id="2"))
    )

    async def lookup(s2_id: str) -> int | None:
        if s2_id == "1":
            raise RuntimeError("simulated lookup failure")
        return 60  # B comes through fine

    score, _ = await score_authors(paper, Field.CS, lookup=lookup)
    # 60 ≥ 50 → CS tier 1.0
    assert score == _AUTHOR_MAX * 1.0


async def test_score_authors_uses_field_aware_tiers_for_math() -> None:
    """Same h-index reads differently per field — math thresholds are
    lower because the community is smaller. h=20 is mid-tier in CS
    (multiplier 0.6) but top-tier in math (multiplier 0.8)."""
    paper = _paper((Author("A", s2_id="1"),))
    lookup = _stub_lookup({"1": 20})
    cs_score, _ = await score_authors(paper, Field.CS, lookup=lookup)
    math_score, _ = await score_authors(paper, Field.MATH, lookup=lookup)
    # CS: h=20 → tier (15, 0.6) → 12
    # Math: h=20 → tier (20, 0.8) → 16
    assert cs_score < math_score


async def test_score_authors_uses_field_aware_tiers_for_medicine() -> None:
    """Medicine thresholds run higher than CS. h=30 is CS top-tier
    (0.8 mult, score 16) but mid-tier in medicine (0.6 mult, score 12)."""
    paper = _paper((Author("A", s2_id="1"),))
    lookup = _stub_lookup({"1": 30})
    cs_score, _ = await score_authors(paper, Field.CS, lookup=lookup)
    med_score, _ = await score_authors(paper, Field.MEDICINE, lookup=lookup)
    assert cs_score > med_score


# ---- _tier_multiplier ----


def test_tier_multiplier_returns_highest_threshold_met() -> None:
    """The function walks ascending tiers; an h-index that meets multiple
    thresholds picks up the highest multiplier."""
    cs_tiers = _TIERS[Field.CS]
    # CS tiers: (0, 0.0), (8, 0.4), (15, 0.6), (25, 0.8), (50, 1.0)
    assert _tier_multiplier(7, cs_tiers) == 0.0
    assert _tier_multiplier(8, cs_tiers) == 0.4
    assert _tier_multiplier(20, cs_tiers) == 0.6
    assert _tier_multiplier(50, cs_tiers) == 1.0
    assert _tier_multiplier(200, cs_tiers) == 1.0  # caps at top tier


# ---- helpers ----


def _stub_lookup(
    table: dict[str, int | None],
) -> Callable[[str], object]:
    """Build an async lookup that returns table[s2_id] when present."""

    async def lookup(s2_id: str) -> int | None:
        return table.get(s2_id)

    return lookup


async def _no_sleep(seconds: float) -> None:
    return None
