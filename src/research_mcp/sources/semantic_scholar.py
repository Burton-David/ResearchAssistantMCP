"""Semantic Scholar Graph API adapter.

Endpoint: https://api.semanticscholar.org/graph/v1

Reads `SEMANTIC_SCHOLAR_API_KEY` from the environment for higher rate limits.
Works without a key, but the public tier is heavily throttled (100 req / 5 min).
Caches responses and rate-limits requests so the REPL/CLI stay polite.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import httpx

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.errors import SourceUnavailable
from research_mcp.sources._cache import DiskCache
from research_mcp.sources._rate_limit import RateLimiter

_log = logging.getLogger(__name__)

_API_BASE: Final = "https://api.semanticscholar.org/graph/v1"
_FIELDS: Final = (
    "paperId,title,abstract,authors,year,publicationDate,externalIds,"
    "openAccessPdf,venue,journal,url"
)
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_TIMEOUT: Final = 30.0
# Public tier: ~1 req/sec is the de-facto polite ceiling. Authenticated tier
# permits more, but a single shared client doing serial calls is fine for now.
_DEFAULT_MIN_INTERVAL_NOAUTH: Final = 1.0
_DEFAULT_MIN_INTERVAL_AUTH: Final = 0.1


class SemanticScholarSource:
    """A `Source` that fronts the Semantic Scholar Graph API."""

    name: str = "semantic_scholar"
    id_prefixes: tuple[str, ...] = ("s2", "doi")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        cache_path = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "research-mcp" / "semantic_scholar"
        )
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        interval = _DEFAULT_MIN_INTERVAL_AUTH if self._api_key else _DEFAULT_MIN_INTERVAL_NOAUTH
        self._rate = RateLimiter(interval)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        params: dict[str, str] = {
            "query": query.text,
            "limit": str(min(query.max_results, 100)),
            "fields": _FIELDS,
        }
        if query.year_min is not None or query.year_max is not None:
            lo = str(query.year_min) if query.year_min is not None else ""
            hi = str(query.year_max) if query.year_max is not None else ""
            params["year"] = f"{lo}-{hi}"
        # Lets SourceUnavailable propagate per the updated Source contract.
        body = await self._fetch("/paper/search", params)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            _log.exception("semantic scholar response not JSON")
            raise SourceUnavailable(self.name, "response was not JSON") from exc
        data = payload.get("data") or []
        return [paper for raw in data if (paper := _parse_paper(raw))]

    async def fetch(self, paper_id: str) -> Paper | None:
        # Honor `id_prefixes`: the wiring layer routes ids by prefix, and a
        # Source must not silently handle prefixes it doesn't claim. Without
        # this guard, fetch("arxiv:9999.99999") would burn an API call here
        # and bubble up S2's 404/429 even though ArxivSource is the rightful
        # owner of the `arxiv:` prefix and already returned None for the id.
        prefix = paper_id.split(":", 1)[0]
        if prefix not in self.id_prefixes:
            return None
        s2_id = _strip_prefix(paper_id)
        if s2_id is None:
            return None
        body = await self._fetch(f"/paper/{s2_id}", {"fields": _FIELDS})
        try:
            return _parse_paper(json.loads(body))
        except json.JSONDecodeError as exc:
            _log.exception("semantic scholar fetch response not JSON")
            # A garbled body from S2 isn't an "id unknown"; treat as transient.
            raise SourceUnavailable(self.name, "response was not JSON") from exc

    async def _fetch(self, path: str, params: dict[str, str]) -> bytes:
        cache_key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        await self._rate.acquire()
        headers = {"x-api-key": self._api_key} if self._api_key else {}
        try:
            response = await self._client.get(_API_BASE + path, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("semantic scholar request failed for %s: %s", path, exc)
            raise SourceUnavailable(self.name, str(exc)) from exc
        body = response.content
        self._cache.set(cache_key, body)
        return body


def _strip_prefix(paper_id: str) -> str | None:
    """Convert our canonical ids back to S2's accepted id forms.

    Returns:
        - For `s2:abc` → `abc`
        - For `arxiv:1234.5678` → `arXiv:1234.5678`
        - For `doi:10.x/y` → `DOI:10.x/y`
    """
    if paper_id.startswith("s2:"):
        return paper_id.removeprefix("s2:")
    if paper_id.startswith("arxiv:"):
        return f"arXiv:{paper_id.removeprefix('arxiv:')}"
    if paper_id.startswith("doi:"):
        return f"DOI:{paper_id.removeprefix('doi:')}"
    return None


def _parse_paper(raw: dict[str, Any]) -> Paper | None:
    s2_id = raw.get("paperId")
    if not s2_id:
        return None
    external = raw.get("externalIds") or {}
    arxiv_id = external.get("ArXiv")
    doi = external.get("DOI")
    pdf_block = raw.get("openAccessPdf") or {}
    pdf_url = pdf_block.get("url") if isinstance(pdf_block, dict) else None
    venue = raw.get("venue") or (
        (raw.get("journal") or {}).get("name") if isinstance(raw.get("journal"), dict) else None
    )
    authors = tuple(
        Author(name=a["name"])
        for a in raw.get("authors") or []
        if isinstance(a, dict) and a.get("name")
    )
    published = _parse_date(raw.get("publicationDate"), raw.get("year"))
    return Paper(
        id=f"s2:{s2_id}",
        title=raw.get("title") or "",
        abstract=raw.get("abstract") or "",
        authors=authors,
        published=published,
        url=raw.get("url"),
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=s2_id,
        pdf_url=pdf_url,
    )


def _parse_date(iso_value: Any, year_value: Any) -> date | None:
    if isinstance(iso_value, str):
        try:
            return date.fromisoformat(iso_value)
        except ValueError:
            pass
    if isinstance(year_value, int):
        try:
            return date(year_value, 1, 1)
        except ValueError:
            return None
    return None
