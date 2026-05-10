"""OpenAlexSource unit tests — abstract reconstruction, prefix routing, search shape."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import httpx
import pytest

from research_mcp.domain.paper import Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.source import Source
from research_mcp.errors import SourceUnavailable
from research_mcp.sources.openalex import (
    OpenAlexSource,
    _parse_work,
    _reconstruct_abstract,
    _strip_doi_url,
    _strip_openalex_id_url,
)

pytestmark = pytest.mark.unit


# ---- protocol conformance ----


def test_openalex_satisfies_source_protocol() -> None:
    src = OpenAlexSource(email="t@example.com", cache_dir=Path("/tmp/_unused"))
    assert isinstance(src, Source)
    assert src.name == "openalex"
    # Claims openalex:; also claims doi: so the wiring layer can route DOI
    # lookups here when openalex is listed before s2.
    assert "openalex" in src.id_prefixes
    assert "doi" in src.id_prefixes


def test_openalex_requires_email_at_construction() -> None:
    """Per the OpenAlex polite-pool guidance, every request must carry mailto.
    A blank email would silently degrade to the 'common pool' (slower, less
    reliable) — better to refuse construction so the user fixes it now."""
    with pytest.raises(ValueError, match="email"):
        OpenAlexSource(email="", cache_dir=Path("/tmp/_unused"))


# ---- url-stripping helpers ----


def test_strip_openalex_id_url_extracts_w_id() -> None:
    assert _strip_openalex_id_url("https://openalex.org/W2626778328") == "W2626778328"


def test_strip_openalex_id_url_returns_none_for_garbage() -> None:
    assert _strip_openalex_id_url("not-an-openalex-url") is None
    assert _strip_openalex_id_url("") is None
    assert _strip_openalex_id_url(None) is None


def test_strip_doi_url_extracts_bare_doi() -> None:
    assert _strip_doi_url("https://doi.org/10.1038/nature12373") == "10.1038/nature12373"
    # Some sources include `http://` instead of https
    assert _strip_doi_url("http://doi.org/10.x/y") == "10.x/y"


def test_strip_doi_url_returns_none_for_no_doi() -> None:
    assert _strip_doi_url(None) is None
    assert _strip_doi_url("") is None


# ---- abstract inverted-index reconstruction ----


def test_reconstruct_abstract_orders_words_by_position() -> None:
    """OpenAlex stores abstracts as `{word: [positions]}`; the parser
    reverses to a flat string."""
    inverted = {
        "The": [0, 4],
        "quick": [1],
        "brown": [2],
        "fox": [3],
        "rest": [5],
    }
    assert _reconstruct_abstract(inverted) == "The quick brown fox The rest"


def test_reconstruct_abstract_handles_none() -> None:
    """About 5% of OpenAlex records have no abstract (per their docs).
    A None inverted index must round-trip as empty string."""
    assert _reconstruct_abstract(None) == ""


def test_reconstruct_abstract_handles_empty_dict() -> None:
    assert _reconstruct_abstract({}) == ""


# ---- _parse_work ----


# Trimmed-down real OpenAlex response captured from a live API call.
_VASWANI_WORK: dict = {
    "id": "https://openalex.org/W2626778328",
    "doi": "https://doi.org/10.65215/2q58a426",
    "title": "Attention Is All You Need",
    "publication_year": 2017,
    "publication_date": "2017-06-12",
    "cited_by_count": 6536,
    "is_retracted": False,
    "primary_location": {
        "id": "doi:10.65215/2q58a426",
        "landing_page_url": "https://doi.org/10.65215/2q58a426",
        "pdf_url": None,
        "source": {"display_name": "NeurIPS", "type": "conference"},
    },
    "best_oa_location": {
        "pdf_url": "https://example.org/vaswani.pdf",
    },
    "open_access": {"is_oa": True, "oa_url": "https://example.org/vaswani.pdf"},
    "authorships": [
        {"author_position": "first",
         "author": {"display_name": "Ashish Vaswani", "orcid": "https://orcid.org/0000-0002-7794-2085"}},
        {"author_position": "middle",
         "author": {"display_name": "Noam Shazeer", "orcid": None}},
    ],
    "abstract_inverted_index": {
        "The": [0],
        "dominant": [1],
        "sequence": [2],
        "transduction": [3],
        "models": [4],
        "are": [5],
        "RNNs.": [6],
    },
}


def test_parse_work_extracts_canonical_paper() -> None:
    p = _parse_work(_VASWANI_WORK)
    assert p is not None
    assert p.id == "openalex:W2626778328"
    assert p.title == "Attention Is All You Need"
    assert p.doi == "10.65215/2q58a426"
    assert p.published == date(2017, 6, 12)


def test_parse_work_reconstructs_abstract() -> None:
    p = _parse_work(_VASWANI_WORK)
    assert p is not None
    assert p.abstract == "The dominant sequence transduction models are RNNs."


def test_parse_work_extracts_authors_with_orcids() -> None:
    p = _parse_work(_VASWANI_WORK)
    assert p is not None
    assert len(p.authors) == 2
    assert p.authors[0].name == "Ashish Vaswani"
    assert p.authors[0].orcid == "https://orcid.org/0000-0002-7794-2085"
    assert p.authors[1].name == "Noam Shazeer"
    assert p.authors[1].orcid is None


def test_parse_work_extracts_venue_from_primary_location_source() -> None:
    p = _parse_work(_VASWANI_WORK)
    assert p is not None
    assert p.venue == "NeurIPS"


def test_parse_work_uses_best_oa_pdf_url_when_available() -> None:
    p = _parse_work(_VASWANI_WORK)
    assert p is not None
    assert p.pdf_url == "https://example.org/vaswani.pdf"


def test_parse_work_falls_back_to_publication_year_only_when_date_missing() -> None:
    work = dict(_VASWANI_WORK)
    work["publication_date"] = None
    p = _parse_work(work)
    assert p is not None
    assert p.published == date(2017, 1, 1)


def test_parse_work_returns_none_when_id_missing() -> None:
    work = dict(_VASWANI_WORK)
    work["id"] = None
    assert _parse_work(work) is None


def test_parse_work_returns_none_when_id_unparseable() -> None:
    work = dict(_VASWANI_WORK)
    work["id"] = "not-an-openalex-url"
    assert _parse_work(work) is None


def test_parse_work_handles_missing_doi() -> None:
    """Many OpenAlex records lack a DOI (preprints, dissertations, etc.)."""
    work = dict(_VASWANI_WORK)
    work["doi"] = None
    p = _parse_work(work)
    assert p is not None
    assert p.doi is None


def test_parse_work_handles_null_primary_location_source() -> None:
    work = dict(_VASWANI_WORK)
    work["primary_location"] = {"source": None, "pdf_url": None}
    p = _parse_work(work)
    assert p is not None
    assert p.venue is None


def test_parse_work_handles_null_open_access_block() -> None:
    work = dict(_VASWANI_WORK)
    work["best_oa_location"] = None
    work["open_access"] = None
    p = _parse_work(work)
    assert p is not None
    assert p.pdf_url is None


def test_parse_work_handles_no_abstract() -> None:
    """Around 5% of OpenAlex records carry no abstract."""
    work = dict(_VASWANI_WORK)
    work["abstract_inverted_index"] = None
    p = _parse_work(work)
    assert p is not None
    assert p.abstract == ""


# ---- end-to-end search/fetch via MockTransport ----


def _build_source(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    email: str = "test@example.com",
) -> OpenAlexSource:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return OpenAlexSource(
        email=email,
        cache_dir=tmp_path / "cache",
        client=client,
    )


def _search_response(*works: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"meta": {"count": len(works)}, "results": list(works)}
    )


def _work_response(work: dict) -> httpx.Response:
    return httpx.Response(200, json=work)


async def test_search_calls_works_endpoint_with_search_param(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response(_VASWANI_WORK)

    src = _build_source(tmp_path, handler)
    try:
        results = await src.search(SearchQuery(text="attention transformer", max_results=5))
    finally:
        await src.aclose()
    assert len(results) == 1
    assert results[0].id == "openalex:W2626778328"
    assert captured[0].path == "/works"
    params = dict(captured[0].params)
    assert params["search"] == "attention transformer"
    assert params["per-page"] == "5"
    # Polite pool: every request carries mailto=
    assert params["mailto"] == "test@example.com"


async def test_search_passes_year_filter(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response()

    src = _build_source(tmp_path, handler)
    try:
        await src.search(SearchQuery(text="x", year_min=2018, year_max=2022, max_results=5))
    finally:
        await src.aclose()
    params = dict(captured[0].params)
    # OpenAlex filter syntax: filter=publication_year:2018-2022
    assert "publication_year:2018-2022" in params.get("filter", "")


async def test_search_caps_per_page_at_200(tmp_path: Path) -> None:
    """OpenAlex `/works` rejects per-page > 200; cap on our side so a
    user passing max_results=500 still gets a clean response."""
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response()

    src = _build_source(tmp_path, handler)
    try:
        await src.search(SearchQuery(text="x", max_results=500))
    finally:
        await src.aclose()
    assert dict(captured[0].params)["per-page"] == "200"


async def test_fetch_by_openalex_id_uses_works_path(tmp_path: Path) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return _work_response(_VASWANI_WORK)

    src = _build_source(tmp_path, handler)
    try:
        paper = await src.fetch("openalex:W2626778328")
    finally:
        await src.aclose()
    assert paper is not None
    assert paper.id == "openalex:W2626778328"
    assert captured == ["/works/W2626778328"]


async def test_fetch_by_doi_uses_works_doi_path(tmp_path: Path) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return _work_response(_VASWANI_WORK)

    src = _build_source(tmp_path, handler)
    try:
        paper = await src.fetch("doi:10.65215/2q58a426")
    finally:
        await src.aclose()
    assert paper is not None
    assert captured == ["/works/doi:10.65215/2q58a426"]


async def test_fetch_with_unknown_prefix_returns_none_without_request(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    src = _build_source(tmp_path, handler)
    try:
        assert await src.fetch("arxiv:1706.03762") is None
        assert await src.fetch("pmid:12345") is None
        assert await src.fetch("malformed-id") is None
    finally:
        await src.aclose()
    assert calls == 0


async def test_fetch_returns_none_on_404(tmp_path: Path) -> None:
    """OpenAlex returns 404 for unknown ids; that's not an outage."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Not found."})

    src = _build_source(tmp_path, handler)
    try:
        paper = await src.fetch("openalex:W9999999999")
    finally:
        await src.aclose()
    assert paper is None


async def test_search_raises_source_unavailable_on_persistent_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep",
        _no_sleep,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream busy")

    src = _build_source(tmp_path, handler)
    try:
        with pytest.raises(SourceUnavailable) as exc_info:
            await src.search(SearchQuery(text="x", max_results=5))
    finally:
        await src.aclose()
    assert exc_info.value.source_name == "openalex"


async def test_search_raises_source_unavailable_on_invalid_json(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    src = _build_source(tmp_path, handler)
    try:
        with pytest.raises(SourceUnavailable):
            await src.search(SearchQuery(text="x", max_results=5))
    finally:
        await src.aclose()


async def test_response_is_cached_so_repeat_calls_skip_http(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _search_response(_VASWANI_WORK)

    src = _build_source(tmp_path, handler)
    try:
        first: Sequence[Paper] = await src.search(SearchQuery(text="cached", max_results=5))
        second: Sequence[Paper] = await src.search(SearchQuery(text="cached", max_results=5))
    finally:
        await src.aclose()
    assert [p.id for p in first] == [p.id for p in second]
    assert calls == 1  # second call served from disk cache


async def test_mailto_param_attached_to_every_request(tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if "/works/" in request.url.path:
            return _work_response(_VASWANI_WORK)
        return _search_response(_VASWANI_WORK)

    src = _build_source(tmp_path, handler, email="user@lab.edu")
    try:
        await src.search(SearchQuery(text="x", max_results=5))
        await src.fetch("openalex:W2626778328")
    finally:
        await src.aclose()
    for params in seen:
        assert params["mailto"] == "user@lab.edu"


# ---- input round-trip ----


def test_parse_search_payload_skips_unparseable_works() -> None:
    """If one record in the results array is malformed, the others should
    still come through. Mirrors how arxiv parser tolerates broken entries."""
    from research_mcp.sources.openalex import _parse_search_payload

    body = json.dumps({
        "results": [
            _VASWANI_WORK,
            {"id": None, "title": "broken record"},
            _VASWANI_WORK,
        ]
    }).encode()
    papers = _parse_search_payload(body, source_name="openalex")
    assert len(papers) == 2
    assert all(p.id == "openalex:W2626778328" for p in papers)


async def _no_sleep(seconds: float) -> None:
    return None
