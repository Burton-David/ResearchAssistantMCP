"""PubMedSource unit tests — XML parsing, esearch routing, prefix discipline."""

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
from research_mcp.sources.pubmed import (
    PubMedSource,
    _build_term,
    _coerce_month,
    _parse_pubmed_xml,
)

pytestmark = pytest.mark.unit


# ---- protocol conformance ----


def test_pubmed_satisfies_source_protocol() -> None:
    src = PubMedSource(cache_dir=Path("/tmp/_unused"))
    assert isinstance(src, Source)
    assert src.name == "pubmed"
    assert src.id_prefixes == ("pmid", "pmc")


# ---- _build_term ----


def test_build_term_passes_text_through_unmodified() -> None:
    """PubMed users may write native PubMed grammar; don't munge it."""
    q = SearchQuery(text="asthma[MeSH] AND children[Title]", max_results=5)
    assert _build_term(q) == "asthma[MeSH] AND children[Title]"


def test_build_term_folds_authors_into_author_clauses() -> None:
    q = SearchQuery(text="diabetes", authors=("Smith J", "Doe A"), max_results=5)
    assert _build_term(q) == 'diabetes AND "Smith J"[Author] AND "Doe A"[Author]'


def test_build_term_falls_back_to_wildcard_for_empty_query() -> None:
    """Empty query is unusual but legal; PubMed accepts `*` as a wildcard."""
    q = SearchQuery(text="", max_results=5)
    assert _build_term(q) == "*"


# ---- _coerce_month ----


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Jan", 1), ("January", 1), ("01", 1), ("1", 1),
        ("Dec", 12), ("DECEMBER", 12), ("12", 12),
        ("", None), ("Foo", None), ("13", None), ("0", None),
    ],
)
def test_coerce_month(value: str, expected: int | None) -> None:
    assert _coerce_month(value) == expected


# ---- _parse_pubmed_xml ----


_SAMPLE_PUBMED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345678</PMID>
      <Article PubModel="Print">
        <Journal>
          <ISSN>1234-5678</ISSN>
          <JournalIssue><PubDate><Year>2023</Year><Month>Jul</Month><Day>15</Day></PubDate></JournalIssue>
          <Title>The New England Journal of Medicine</Title>
          <ISOAbbreviation>N Engl J Med</ISOAbbreviation>
        </Journal>
        <ArticleTitle>Effectiveness of an Intervention for Type 2 Diabetes.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background sentence.</AbstractText>
          <AbstractText Label="METHODS">Methods sentence.</AbstractText>
          <AbstractText Label="RESULTS">Results sentence.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
          <Author><LastName>Doe</LastName><ForeName>Alex</ForeName></Author>
          <Author><CollectiveName>The DIABETES Study Group</CollectiveName></Author>
        </AuthorList>
        <ArticleDate DateType="Electronic"><Year>2023</Year><Month>06</Month><Day>30</Day></ArticleDate>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName MajorTopicYN="Y">Diabetes Mellitus, Type 2</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName MajorTopicYN="N">Humans</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1056/NEJMoa1234567</ArticleId>
        <ArticleId IdType="pmc">PMC11111111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_xml_extracts_canonical_paper() -> None:
    papers = _parse_pubmed_xml(_SAMPLE_PUBMED_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p.id == "pmid:12345678"
    assert p.title.startswith("Effectiveness of an Intervention")
    assert p.doi == "10.1056/NEJMoa1234567"
    assert p.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert p.venue == "The New England Journal of Medicine"


def test_parse_pubmed_xml_keeps_structured_abstract_labels() -> None:
    p = _parse_pubmed_xml(_SAMPLE_PUBMED_XML)[0]
    # Labels are preserved inline so the LLM sees the structure.
    assert "BACKGROUND: Background sentence." in p.abstract
    assert "METHODS: Methods sentence." in p.abstract
    assert "RESULTS: Results sentence." in p.abstract


def test_parse_pubmed_xml_extracts_authors_in_forename_lastname_order() -> None:
    p = _parse_pubmed_xml(_SAMPLE_PUBMED_XML)[0]
    names = [a.name for a in p.authors]
    assert names == ["Jane Smith", "Alex Doe", "The DIABETES Study Group"]


def test_parse_pubmed_xml_prefers_electronic_pub_date() -> None:
    """ArticleDate[@DateType='Electronic'] is more accurate than PubDate."""
    p = _parse_pubmed_xml(_SAMPLE_PUBMED_XML)[0]
    assert p.published == date(2023, 6, 30)


def test_parse_pubmed_xml_carries_pmc_and_mesh_in_metadata() -> None:
    p = _parse_pubmed_xml(_SAMPLE_PUBMED_XML)[0]
    assert p.metadata["pmc_id"] == "PMC11111111"
    assert "Diabetes Mellitus, Type 2" in p.metadata["mesh_terms"]
    assert "Humans" in p.metadata["mesh_terms"]


def test_parse_pubmed_xml_skips_articles_without_pmid() -> None:
    xml = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <Article><ArticleTitle>No id</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
    assert _parse_pubmed_xml(xml) == []


def test_parse_pubmed_xml_returns_empty_on_malformed_xml() -> None:
    """A garbled body shouldn't crash; the calling _fetch raised already if
    the body itself was an HTTP error."""
    assert _parse_pubmed_xml(b"<not valid xml") == []


def test_parse_pubmed_xml_handles_empty_body() -> None:
    assert _parse_pubmed_xml(b"") == []


# ---- end-to-end search/fetch via MockTransport ----


def _build_source(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = None,
) -> PubMedSource:
    """Construct a PubMedSource whose HTTP traffic is satisfied by `handler`."""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return PubMedSource(
        api_key=api_key,
        email="test@example.com",
        cache_dir=tmp_path / "cache",
        client=client,
    )


def _esearch_response(*pmids: str) -> httpx.Response:
    body = {"esearchresult": {"idlist": list(pmids), "count": str(len(pmids))}}
    return httpx.Response(200, json=body)


def _efetch_response(xml: bytes = _SAMPLE_PUBMED_XML) -> httpx.Response:
    return httpx.Response(200, content=xml, headers={"content-type": "application/xml"})


async def test_search_does_esearch_then_efetch(tmp_path: Path) -> None:
    """Two-step flow: esearch returns PMIDs, efetch returns XML records."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("esearch.fcgi"):
            return _esearch_response("12345678")
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        results = await src.search(SearchQuery(text="diabetes type 2", max_results=5))
    finally:
        await src.aclose()
    assert [c.split("/")[-1] for c in calls] == ["esearch.fcgi", "efetch.fcgi"]
    assert len(results) == 1
    assert results[0].id == "pmid:12345678"


async def test_search_short_circuits_when_esearch_returns_no_pmids(
    tmp_path: Path,
) -> None:
    """Empty PMID list means no efetch — saves an API call."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _esearch_response()  # empty

    src = _build_source(tmp_path, handler)
    try:
        results = await src.search(SearchQuery(text="nothing matches this", max_results=5))
    finally:
        await src.aclose()
    assert results == []
    assert len(calls) == 1
    assert calls[0].endswith("esearch.fcgi")


async def test_search_passes_year_range_as_pdat(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        if request.url.path.endswith("esearch.fcgi"):
            return _esearch_response()
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        await src.search(SearchQuery(text="x", year_min=2020, year_max=2023, max_results=5))
    finally:
        await src.aclose()
    params = dict(captured[0].params)
    assert params["datetype"] == "pdat"
    assert params["mindate"] == "2020/01/01"
    assert params["maxdate"] == "2023/12/31"


async def test_fetch_pmid_efetches_directly(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        paper = await src.fetch("pmid:12345678")
    finally:
        await src.aclose()
    assert paper is not None
    assert paper.id == "pmid:12345678"
    # No esearch call — pmid: routes straight to efetch.
    assert all(c.endswith("efetch.fcgi") for c in calls)


async def test_fetch_pmc_does_esearch_conversion_then_efetch(tmp_path: Path) -> None:
    """PMC id needs a PMID conversion first; verify both steps run."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        calls.append((path, params.get("term", params.get("id", ""))))
        if path.endswith("esearch.fcgi"):
            return _esearch_response("12345678")
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        paper = await src.fetch("pmc:11111111")
    finally:
        await src.aclose()
    assert paper is not None
    assert paper.id == "pmid:12345678"
    # First call: esearch with `PMC<id>[PMC]`. Second: efetch with the resolved pmid.
    assert calls[0][0].endswith("esearch.fcgi")
    assert "PMC11111111[PMC]" in calls[0][1]
    assert calls[1][0].endswith("efetch.fcgi")
    assert calls[1][1] == "12345678"


async def test_fetch_pmc_returns_none_when_conversion_fails(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # esearch returns no idlist — PMC id couldn't be resolved.
        return _esearch_response()

    src = _build_source(tmp_path, handler)
    try:
        paper = await src.fetch("pmc:99999999")
    finally:
        await src.aclose()
    assert paper is None


async def test_fetch_with_unknown_prefix_returns_none_without_request(
    tmp_path: Path,
) -> None:
    """Per the Source contract, prefix routing is owned by the wiring layer;
    a Source must not silently field requests for prefixes it doesn't claim."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, content=b"")

    src = _build_source(tmp_path, handler)
    try:
        assert await src.fetch("arxiv:1706.03762") is None
        assert await src.fetch("doi:10.1234/x") is None
        assert await src.fetch("malformed-id") is None
    finally:
        await src.aclose()
    assert calls == 0


async def test_fetch_with_empty_id_after_prefix_returns_none(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        assert await src.fetch("pmid:") is None
        assert await src.fetch("pmc:") is None
    finally:
        await src.aclose()
    assert calls == 0


async def test_search_raises_source_unavailable_on_persistent_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backoff retries 5xx; after the schedule exhausts, the wrapper raises
    SourceUnavailable so SearchService can report partial_failures."""
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
    assert exc_info.value.source_name == "pubmed"


async def test_search_raises_source_unavailable_on_invalid_json(
    tmp_path: Path,
) -> None:
    """A 200 with non-JSON body shouldn't be confused with 'no results' —
    treat as transient so the user retries instead of seeing an empty list."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, content=b"not json")
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        with pytest.raises(SourceUnavailable):
            await src.search(SearchQuery(text="x", max_results=5))
    finally:
        await src.aclose()


async def test_api_key_changes_rate_limit_interval(tmp_path: Path) -> None:
    """Authenticated tier: 10 req/sec → 0.1s. Unauthenticated: 3 req/sec → 0.34s."""
    src_auth = _build_source(tmp_path / "a", _ok_handler, api_key="ncbi-test-key")
    src_noauth = _build_source(tmp_path / "b", _ok_handler)
    try:
        # Reach into the rate limiter to assert configured interval.
        # Fragile but the shape is stable and the assertion is the whole
        # point of the test; faking time would be more code for the same
        # signal.
        assert src_auth._rate._interval == pytest.approx(0.10)
        assert src_noauth._rate._interval == pytest.approx(0.34)
    finally:
        await src_auth.aclose()
        await src_noauth.aclose()


async def test_email_and_tool_attached_to_every_request(tmp_path: Path) -> None:
    """NCBI fair-use policy expects identification on every request."""
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if request.url.path.endswith("esearch.fcgi"):
            return _esearch_response("12345678")
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        await src.search(SearchQuery(text="x", max_results=5))
    finally:
        await src.aclose()
    for params in seen:
        assert params["email"] == "test@example.com"
        assert params["tool"] == "research-mcp"


async def test_api_key_added_to_request_when_set(tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if request.url.path.endswith("esearch.fcgi"):
            return _esearch_response()
        return _efetch_response()

    src = _build_source(tmp_path, handler, api_key="key-abc")
    try:
        await src.search(SearchQuery(text="x", max_results=5))
    finally:
        await src.aclose()
    assert seen[0]["api_key"] == "key-abc"


async def test_response_is_cached_so_repeat_calls_skip_http(tmp_path: Path) -> None:
    """DiskCache short-circuits the second call; verify by counting requests."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path.endswith("esearch.fcgi"):
            return _esearch_response("12345678")
        return _efetch_response()

    src = _build_source(tmp_path, handler)
    try:
        first: Sequence[Paper] = await src.search(SearchQuery(text="cached", max_results=5))
        second: Sequence[Paper] = await src.search(SearchQuery(text="cached", max_results=5))
    finally:
        await src.aclose()
    assert [p.id for p in first] == [p.id for p in second]
    assert calls == 2  # first esearch + first efetch; both subsequent calls served from cache


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("esearch.fcgi"):
        return _esearch_response()
    return _efetch_response()


async def _no_sleep(seconds: float) -> None:
    return None


# ---- input round-trip ----


def test_detect_pubmed_error_finds_backend_failure() -> None:
    """The actual NCBI 200-OK-with-ERROR body that broke metformin
    search for ~12 hours during a chaos test session, frozen as a
    test fixture so this regression can't return undetected."""
    from research_mcp.sources.pubmed import _detect_pubmed_error

    body = (
        b'{"header":{"type":"esearch","version":"0.3"},'
        b'"esearchresult":{"ERROR":"Search Backend failed: Couldn\'t '
        b'resolve #pmquerysrv-mz?dbaf=pubmed, the address table is empty."}}'
    )
    error = _detect_pubmed_error(body)
    assert error is not None
    assert "Search Backend failed" in error


def test_detect_pubmed_error_finds_database_not_supported() -> None:
    """Second NCBI failure mode caught in the same cache audit."""
    from research_mcp.sources.pubmed import _detect_pubmed_error

    body = (
        b'{"header":{"type":"esearch","version":"0.3"},'
        b'"esearchresult":{"ERROR":"Database is not supported: pubmed"}}'
    )
    assert _detect_pubmed_error(body) == "Database is not supported: pubmed"


def test_detect_pubmed_error_passes_normal_response() -> None:
    """Healthy esearch JSON with an idlist must not be flagged as
    an error — false positives would break successful searches."""
    from research_mcp.sources.pubmed import _detect_pubmed_error

    body = (
        b'{"header":{"type":"esearch","version":"0.3"},'
        b'"esearchresult":{"count":"3","retmax":"3","retstart":"0",'
        b'"idlist":["12345","67890","11111"]}}'
    )
    assert _detect_pubmed_error(body) is None


def test_detect_pubmed_error_ignores_xml_efetch_bodies() -> None:
    """efetch returns XML; the ERROR detector must not waste time
    trying to parse it as JSON."""
    from research_mcp.sources.pubmed import _detect_pubmed_error

    body = b'<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
    assert _detect_pubmed_error(body) is None


def test_detect_pubmed_error_handles_empty_or_garbage_body() -> None:
    from research_mcp.sources.pubmed import _detect_pubmed_error

    assert _detect_pubmed_error(b"") is None
    assert _detect_pubmed_error(b"not json") is None


async def test_search_raises_source_unavailable_on_ncbi_backend_error(
    tmp_path: Path,
) -> None:
    """End-to-end regression: NCBI returns 200 OK with ERROR JSON;
    PubMedSource raises SourceUnavailable instead of silently
    returning [] and caching the failure. Mirrors the chaos-test bug
    where metformin search showed 0 PubMed results with empty
    partial_failures for 12+ hours after a transient NCBI outage."""
    error_body = (
        b'{"header":{"type":"esearch"},'
        b'"esearchresult":{"ERROR":"Search Backend failed: '
        b'address table is empty."}}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=error_body)

    src = _build_source(tmp_path, handler)
    try:
        with pytest.raises(SourceUnavailable, match="upstream backend error"):
            await src.search(SearchQuery(text="metformin", max_results=5))
    finally:
        await src.aclose()


async def test_error_response_is_not_cached(tmp_path: Path) -> None:
    """The bug's persistence vector: a 200-with-ERROR response was
    cached for 24 hours, so the failure outlived the upstream's
    recovery. Verify that error responses bypass the cache write."""
    calls = 0
    error_body = (
        b'{"esearchresult":{"ERROR":"Search Backend failed"}}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=error_body)

    src = _build_source(tmp_path, handler)
    try:
        # First call raises and does NOT cache the error.
        with pytest.raises(SourceUnavailable):
            await src.search(SearchQuery(text="metformin", max_results=5))
        # Second call must hit the network again (not return cached error).
        with pytest.raises(SourceUnavailable):
            await src.search(SearchQuery(text="metformin", max_results=5))
    finally:
        await src.aclose()
    assert calls == 2  # both calls actually went out — error wasn't cached


async def test_stale_cached_error_self_heals_on_next_call(
    tmp_path: Path,
) -> None:
    """If an OLDER server version cached an error body before this fix
    shipped, the next call must treat the cache hit as a miss and
    re-fetch. Otherwise users running the fix still see 12-hour-old
    silent failures until the cache TTL expires."""
    from research_mcp.sources._cache import DiskCache

    error_body = (
        b'{"esearchresult":{"ERROR":"Search Backend failed"}}'
    )
    good_body = (
        b'{"esearchresult":{"idlist":["12345"]}}'
    )

    # Pre-populate the cache with the stale error response.
    cache_dir = tmp_path / "cache"
    cache = DiskCache(cache_dir, ttl_seconds=24 * 60 * 60)
    # Cache key shape: see PubMedSource._fetch.
    key = (
        "/esearch.fcgi?"
        "db=pubmed&email=test@example.com&retmax=5&retmode=json&"
        "sort=relevance&term=metformin&tool=research-mcp"
    )
    cache.set(key, error_body)

    sequence = iter([good_body, _SAMPLE_PUBMED_XML])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, content=next(sequence))
        return httpx.Response(200, content=_SAMPLE_PUBMED_XML)

    src = _build_source(tmp_path, handler)
    try:
        results = await src.search(SearchQuery(text="metformin", max_results=5))
    finally:
        await src.aclose()
    # The stale ERROR cache entry was bypassed; the fresh esearch +
    # efetch came through normally.
    assert len(results) == 1


def test_esearch_idlist_parser_handles_extra_fields() -> None:
    """NCBI sometimes returns idlist alongside warning/error blocks; the
    parser should pick out the ids and ignore the rest."""
    payload = json.dumps(
        {
            "header": {"type": "esearch"},
            "esearchresult": {
                "count": "2",
                "idlist": ["111", "222"],
                "warninglist": {"phrasesignored": [], "phrasesnotfound": ["foo"]},
            },
        }
    ).encode()
    from research_mcp.sources.pubmed import _parse_esearch_idlist

    assert _parse_esearch_idlist(payload, source_name="pubmed") == ["111", "222"]
