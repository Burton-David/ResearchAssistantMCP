"""NCBI E-utilities adapter for PubMed.

Endpoint: https://eutils.ncbi.nlm.nih.gov/entrez/eutils

Two-step flow: `esearch.fcgi` returns matching PMIDs (JSON);
`efetch.fcgi` returns PubMed XML records for those PMIDs. Rate-limited
per NCBI guidance — 3 req/sec without an API key, 10 req/sec with
`NCBI_API_KEY` set. The `email` query parameter is required for
identification under their fair-use policy; we read `NCBI_EMAIL` and
fall back to a generic placeholder so a user without the env set still
gets results.

Two `id_prefixes`: `pmid:` (the canonical PubMed id we emit) and
`pmc:` (PubMed Central full-text id). PMC fetches are routed via
esearch with `<id>[PMC]` to convert to a PMID, then efetch the PMID
record — two API calls, but it lets a user paste either id form into
`get_paper`/`cite_paper` and reach the same record.

Reference: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final
from xml.etree import ElementTree as ET

import httpx

# defusedxml shadows fromstring with a hardened parser that rejects
# billion-laughs / quadratic-blowup payloads. We keep ET for the
# Element type alias and use defusedxml for the parse call itself.
from defusedxml import ElementTree as safe_xml  # noqa: N813

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.errors import SourceUnavailable, redact_secrets
from research_mcp.sources._backoff import with_backoff
from research_mcp.sources._cache import DiskCache
from research_mcp.sources._rate_limit import RateLimiter

_log = logging.getLogger(__name__)

_API_BASE: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_TIMEOUT: Final = 30.0
# NCBI guidance: 3 req/sec without an API key, 10 req/sec with one.
# Convert to per-request minimum intervals.
_INTERVAL_NOAUTH: Final = 0.34
_INTERVAL_AUTH: Final = 0.10
# NCBI requires an email for identification. Falling back to a generic
# value still gets through; the `tool` parameter joins it for full
# attribution per their guidelines.
_DEFAULT_EMAIL: Final = "research-mcp@example.com"
_TOOL_NAME: Final = "research-mcp"

# Lowercase month-prefix → numeric month, used to normalize PubMed's
# free-form Month elements (e.g. "Jan", "January", "01" all map to 1).
_MONTHS: Final = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class PubMedSource:
    """A `Source` that fronts NCBI E-utilities for PubMed records."""

    name: str = "pubmed"
    id_prefixes: tuple[str, ...] = ("pmid", "pmc")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        )
        self._email = email if email is not None else os.environ.get(
            "NCBI_EMAIL", _DEFAULT_EMAIL
        )
        cache_path = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "research-mcp" / "pubmed"
        )
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        interval = _INTERVAL_AUTH if self._api_key else _INTERVAL_NOAUTH
        self._rate = RateLimiter(interval)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        pmids = await self._esearch(query)
        if not pmids:
            return []
        return await self._efetch(pmids)

    async def fetch(self, paper_id: str) -> Paper | None:
        # Route by prefix; per the Source contract, prefixes we don't claim
        # return None without an API call.
        prefix, _, raw = paper_id.partition(":")
        if prefix not in self.id_prefixes or not raw:
            return None
        if prefix == "pmid":
            papers = await self._efetch([raw])
            return papers[0] if papers else None
        # PMC: esearch the `<id>[PMC]` term to map PMC → PMID, then efetch.
        # Costs an extra API call vs pmid lookup, but lets users paste either
        # canonical form into cite_paper / get_paper interchangeably.
        pmid = await self._pmc_to_pmid(raw)
        if pmid is None:
            return None
        papers = await self._efetch([pmid])
        return papers[0] if papers else None

    async def _esearch(self, query: SearchQuery) -> list[str]:
        params = self._common_params() | {
            "db": "pubmed",
            "term": _build_term(query),
            "retmax": str(min(query.max_results, 100)),
            "retmode": "json",
            "sort": "relevance",
        }
        if query.year_min is not None or query.year_max is not None:
            params["datetype"] = "pdat"
            if query.year_min is not None:
                params["mindate"] = f"{query.year_min}/01/01"
            if query.year_max is not None:
                params["maxdate"] = f"{query.year_max}/12/31"
        body = await self._fetch("/esearch.fcgi", params)
        return _parse_esearch_idlist(body, source_name=self.name)

    async def _efetch(self, pmids: Sequence[str]) -> list[Paper]:
        params = self._common_params() | {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        body = await self._fetch("/efetch.fcgi", params)
        return _parse_pubmed_xml(body)

    async def _pmc_to_pmid(self, pmc_raw: str) -> str | None:
        bare = pmc_raw.removeprefix("PMC").removeprefix("pmc")
        if not bare:
            return None
        params = self._common_params() | {
            "db": "pubmed",
            "term": f"PMC{bare}[PMC]",
            "retmode": "json",
            "retmax": "1",
        }
        body = await self._fetch("/esearch.fcgi", params)
        idlist = _parse_esearch_idlist(body, source_name=self.name)
        return idlist[0] if idlist else None

    def _common_params(self) -> dict[str, str]:
        params = {"email": self._email, "tool": _TOOL_NAME}
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def _fetch(self, path: str, params: dict[str, str]) -> bytes:
        cache_key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Re-validate cached bodies. NCBI sometimes returns 200 OK
            # with `esearchresult.ERROR` set (backend outage, throttled,
            # malformed query). The previous version cached those for
            # 24 hours and returned silent empty results forever after.
            # Hit-time validation lets a stale failure self-heal: an
            # invalid cached body is treated as a miss and re-fetched.
            error = _detect_pubmed_error(cached)
            if error is None:
                return cached
            _log.info("pubmed cache hit had ERROR (%s); refetching", error)
        await self._rate.acquire()

        async def do_request() -> httpx.Response:
            return await self._client.get(_API_BASE + path, params=params)

        try:
            response = await with_backoff(do_request, source_name=self.name)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Scrub api_key= from logged URL — see errors.redact_secrets.
            scrubbed = redact_secrets(str(exc))
            _log.warning("pubmed request failed for %s: %s", path, scrubbed)
            raise SourceUnavailable(self.name, scrubbed) from exc
        body = response.content
        # E-utilities returns 200 OK + ERROR JSON for backend failures
        # instead of a proper 5xx. Surface it as a transient SourceUnavailable
        # (so it lands in partial_failures and the user sees it), and do
        # NOT cache — the 24-hour TTL would otherwise persist the failure
        # past the upstream recovery.
        error = _detect_pubmed_error(body)
        if error is not None:
            _log.warning("pubmed returned 200 OK with ERROR body: %s", error)
            raise SourceUnavailable(
                self.name, f"upstream backend error: {error}"
            )
        self._cache.set(cache_key, body)
        return body


def _build_term(query: SearchQuery) -> str:
    """Compose a PubMed `term=` value from a SearchQuery.

    PubMed's term grammar is its own thing — `<text>[Field]` clauses
    AND-joined. We pass `text` through unmodified so users can write
    native PubMed queries (`asthma[MeSH] AND children[Title]`) and they
    just work, but also fold each `authors=` entry into a `[Author]`
    clause for parity with the other Sources.
    """
    parts: list[str] = []
    if query.text:
        parts.append(query.text)
    for author in query.authors:
        parts.append(f'"{author}"[Author]')
    return " AND ".join(parts) if parts else "*"


def _detect_pubmed_error(body: bytes) -> str | None:
    """Inspect an esearch JSON body for NCBI's 200-OK-with-ERROR shape.

    NCBI returns a 200 OK + JSON body with `esearchresult.ERROR` set
    when the backend service is unavailable, the query is malformed,
    or the database is misconfigured. Examples seen in the wild:
        "Search Backend failed: Couldn't resolve #pmquerysrv-mz..."
        "Database is not supported: pubmed"
    Returns the error message if present, else None. Non-JSON or
    non-esearch bodies (efetch XML, etc.) return None — efetch's
    failure mode is HTTP-level and handled by the HTTPError branch.
    """
    if not body:
        return None
    # efetch returns XML; quick prefix check avoids parsing it as JSON.
    if body.lstrip().startswith(b"<"):
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        return None
    error = result.get("ERROR")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return None


def _parse_esearch_idlist(body: bytes, *, source_name: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        _log.exception("pubmed esearch response not JSON")
        raise SourceUnavailable(source_name, "esearch response was not JSON") from exc
    result = payload.get("esearchresult") or {}
    idlist = result.get("idlist") or []
    return [str(p) for p in idlist if p]


def _parse_pubmed_xml(body: bytes) -> list[Paper]:
    if not body:
        return []
    try:
        root = safe_xml.fromstring(body)
    except ET.ParseError:
        _log.exception("pubmed XML parse error")
        return []
    return [
        paper
        for entry in root.findall(".//PubmedArticle")
        if (paper := _parse_article(entry))
    ]


def _parse_article(article: ET.Element) -> Paper | None:
    pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
    if not pmid:
        return None
    article_elem = article.find(".//Article")
    if article_elem is None:
        return None

    title = " ".join((article_elem.findtext("ArticleTitle") or "").split())
    abstract = _extract_abstract(article_elem)
    authors = _extract_authors(article_elem)
    venue = _extract_venue(article_elem)
    published = _extract_date(article_elem)
    doi = _extract_id(article, "doi")
    pmc_id = _extract_id(article, "pmc")
    mesh_terms = _extract_mesh(article)

    metadata: dict[str, str] = {}
    if pmc_id:
        metadata["pmc_id"] = pmc_id
    if mesh_terms:
        # Joined for transport; downstream consumers split on `; ` if needed.
        metadata["mesh_terms"] = "; ".join(mesh_terms)

    return Paper(
        id=f"pmid:{pmid}",
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        venue=venue,
        doi=doi,
        pdf_url=None,
        metadata=MappingProxyType(metadata),
    )


def _extract_abstract(article_elem: ET.Element) -> str:
    """Concatenate PubMed's structured abstract sections into a single string.

    PubMed marks structured abstracts with `<AbstractText Label="BACKGROUND">`,
    `<AbstractText Label="METHODS">`, etc. Preserve the labels inline so the
    LLM sees the structure that PubMed publishers intentionally encoded.
    """
    abstract_elem = article_elem.find("Abstract")
    if abstract_elem is None:
        return ""
    parts: list[str] = []
    for text_elem in abstract_elem.findall("AbstractText"):
        text = " ".join((text_elem.text or "").split())
        if not text:
            continue
        label = text_elem.get("Label") or ""
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _extract_authors(article_elem: ET.Element) -> tuple[Author, ...]:
    author_list = article_elem.find("AuthorList")
    if author_list is None:
        return ()
    authors: list[Author] = []
    for author_elem in author_list.findall("Author"):
        last = (author_elem.findtext("LastName") or "").strip()
        fore = (author_elem.findtext("ForeName") or "").strip()
        if last and fore:
            authors.append(Author(name=f"{fore} {last}"))
        elif last:
            authors.append(Author(name=last))
        else:
            collective = (author_elem.findtext("CollectiveName") or "").strip()
            if collective:
                authors.append(Author(name=collective))
    return tuple(authors)


def _extract_venue(article_elem: ET.Element) -> str | None:
    journal = article_elem.find("Journal")
    if journal is None:
        return None
    title = (journal.findtext("Title") or "").strip()
    if title:
        return title
    abbrev = (journal.findtext("ISOAbbreviation") or "").strip()
    return abbrev or None


def _extract_id(article: ET.Element, id_type: str) -> str | None:
    """Pull a typed ArticleId — PubMed records carry DOI, PMC, PII, …
    in a single ArticleIdList; we just pluck the one we want."""
    for id_elem in article.findall(".//ArticleIdList/ArticleId"):
        if (id_elem.get("IdType") or "").lower() == id_type.lower():
            text = (id_elem.text or "").strip()
            return text or None
    return None


def _extract_mesh(article: ET.Element) -> tuple[str, ...]:
    mesh_list = article.find(".//MeshHeadingList")
    if mesh_list is None:
        return ()
    terms: list[str] = []
    for mesh in mesh_list.findall("MeshHeading"):
        descriptor = (mesh.findtext("DescriptorName") or "").strip()
        if descriptor:
            terms.append(descriptor)
    return tuple(terms)


def _extract_date(article_elem: ET.Element) -> date | None:
    """Pull a publication date from PubMed's article element.

    Prefers `ArticleDate[@DateType='Electronic']` (the e-pub date, which
    is more reliable than the print PubDate for newer records). Falls
    back to the journal's PubDate. Both can have free-form Month strings
    ('Jan', 'January', '01'); we normalize via `_MONTHS`.
    """
    el = article_elem.find(".//ArticleDate")
    if el is not None:
        result = _parse_date_element(el)
        if result is not None:
            return result
    pub_date = article_elem.find(".//Journal/JournalIssue/PubDate")
    if pub_date is not None:
        result = _parse_date_element(pub_date)
        if result is not None:
            return result
    return None


def _parse_date_element(el: ET.Element) -> date | None:
    year_str = (el.findtext("Year") or "").strip()
    if not year_str:
        return None
    try:
        year = int(year_str)
    except ValueError:
        return None
    month_str = (el.findtext("Month") or "").strip()
    day_str = (el.findtext("Day") or "").strip()
    month = _coerce_month(month_str) or 1
    day = _coerce_day(day_str) or 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _coerce_month(value: str) -> int | None:
    if not value:
        return None
    lower = value[:3].lower()
    if lower in _MONTHS:
        return _MONTHS[lower]
    try:
        n = int(value)
    except ValueError:
        return None
    return n if 1 <= n <= 12 else None


def _coerce_day(value: str) -> int | None:
    if not value:
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if 1 <= n <= 31 else None
