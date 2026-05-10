"""arXiv Atom API adapter.

Endpoint: https://export.arxiv.org/api/query

Honors arXiv's published guidance of ~1 request every 3 seconds via a process-local
RateLimiter. Disk-caches responses by query hash for 24 hours so repeated CLI runs
and REPL sessions don't pound the API.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

import httpx

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.errors import SourceUnavailable
from research_mcp.sources._backoff import with_backoff
from research_mcp.sources._cache import DiskCache
from research_mcp.sources._rate_limit import RateLimiter

_log = logging.getLogger(__name__)

_API_URL: Final = "https://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_MIN_INTERVAL: Final = 3.0
_DEFAULT_TIMEOUT: Final = 30.0


class ArxivSource:
    """A `Source` that fronts the arXiv Atom API."""

    name: str = "arxiv"
    id_prefixes: tuple[str, ...] = ("arxiv",)

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        min_interval_seconds: float = _DEFAULT_MIN_INTERVAL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cache_path = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "research-mcp" / "arxiv"
        )
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        self._rate = RateLimiter(min_interval_seconds)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        search_str = _build_search_string(query)
        params = {
            "search_query": search_str,
            "start": "0",
            "max_results": str(query.max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        # Lets SourceUnavailable propagate per the updated Source contract.
        # SearchService catches per-source so a 429 here doesn't kill the
        # merged result, and surfaces the failure via partial_failures.
        body = await self._fetch(params)
        return _parse_feed(body)

    async def fetch(self, paper_id: str) -> Paper | None:
        if not paper_id.startswith("arxiv:"):
            return None
        bare = paper_id.removeprefix("arxiv:")
        params = {"id_list": bare, "max_results": "1"}
        body = await self._fetch(params)
        papers = _parse_feed(body)
        return papers[0] if papers else None

    async def _fetch(self, params: dict[str, str]) -> bytes:
        cache_key = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        await self._rate.acquire()

        async def do_request() -> httpx.Response:
            return await self._client.get(_API_URL, params=params)

        try:
            response = await with_backoff(do_request, source_name=self.name)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("arxiv request failed for %s: %s", params, exc)
            raise SourceUnavailable(self.name, str(exc)) from exc
        body = response.content
        self._cache.set(cache_key, body)
        return body


def _build_search_string(query: SearchQuery) -> str:
    parts: list[str] = []
    if query.text:
        parts.append(f"all:{query.text}")
    for author in query.authors:
        parts.append(f'au:"{author}"')
    if query.year_min is not None or query.year_max is not None:
        lo = f"{query.year_min}01010000" if query.year_min else "00000101000"
        hi = f"{query.year_max}12312359" if query.year_max else "99991231235"
        parts.append(f"submittedDate:[{lo} TO {hi}]")
    return " AND ".join(parts) if parts else "all:*"


def _parse_feed(body: bytes) -> list[Paper]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        _log.exception("arxiv feed parse error")
        return []
    return [p for entry in root.findall("atom:entry", _NS) if (p := _parse_entry(entry))]


def _parse_entry(entry: ET.Element) -> Paper | None:
    id_text = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
    arxiv_id = _arxiv_id_from_url(id_text)
    if not arxiv_id:
        return None
    title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
    abstract = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
    authors = tuple(
        Author(name=name)
        for el in entry.findall("atom:author", _NS)
        if (name := (el.findtext("atom:name", default="", namespaces=_NS) or "").strip())
    )
    published = _parse_date(entry.findtext("atom:published", default="", namespaces=_NS))
    pdf_url: str | None = None
    abs_url: str | None = None
    for link in entry.findall("atom:link", _NS):
        href = link.get("href")
        if not href:
            continue
        if link.get("type") == "application/pdf":
            pdf_url = href
        elif link.get("rel") == "alternate":
            abs_url = href
    doi = entry.findtext("arxiv:doi", default=None, namespaces=_NS) or None
    journal = entry.findtext("arxiv:journal_ref", default=None, namespaces=_NS) or None
    return Paper(
        id=f"arxiv:{arxiv_id}",
        title=" ".join(title.split()),
        abstract=" ".join(abstract.split()),
        authors=authors,
        published=published,
        url=abs_url,
        venue=journal,
        doi=doi,
        arxiv_id=arxiv_id,
        pdf_url=pdf_url,
    )


def _arxiv_id_from_url(id_url: str) -> str | None:
    # id_url looks like "http://arxiv.org/abs/2401.12345v2"
    if "/abs/" not in id_url:
        return None
    raw = id_url.rsplit("/abs/", 1)[-1]
    # strip version suffix so "2401.12345v2" canonicalizes to "2401.12345"
    if "v" in raw and raw.split("v")[-1].isdigit():
        raw = raw.rsplit("v", 1)[0]
    return raw or None


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None
