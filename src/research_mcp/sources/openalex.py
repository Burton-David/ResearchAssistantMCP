"""OpenAlex `/works` adapter.

Endpoint: https://api.openalex.org/works

OpenAlex is free and unauthenticated; politeness is via a `mailto`
query parameter that puts requests in the polite pool (faster, more
reliable than the common pool). We make `email` a required constructor
argument — running without one would silently drop us into the slow
pool, so it's better to refuse construction and surface the misconfig
at boot. The wiring layer reads `RESEARCH_MCP_OPENALEX_EMAIL`; if it's
unset, OpenAlex isn't added to the Source list (no error, just one
fewer source).

Two `id_prefixes`: `openalex:` (the canonical form, e.g. `openalex:W2626778328`)
and `doi:`, since OpenAlex's `/works/doi:<doi>` endpoint resolves DOIs
directly. The `doi:` prefix overlaps with `SemanticScholarSource`; the
wiring layer's source-list ordering decides who wins on collisions.

OpenAlex stores abstracts as inverted indices (`{word: [positions]}`)
rather than plain text — see `_reconstruct_abstract`. This is a
quirk of their copyright-friendly storage model.
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
from research_mcp.sources._backoff import with_backoff
from research_mcp.sources._cache import DiskCache
from research_mcp.sources._rate_limit import RateLimiter

_log = logging.getLogger(__name__)

_API_BASE: Final = "https://api.openalex.org"
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_TIMEOUT: Final = 30.0
# OpenAlex's polite pool tolerates ~10 req/sec without complaint.
_DEFAULT_MIN_INTERVAL: Final = 0.1
# /works rejects per-page > 200; cap on our side so a SearchQuery with
# max_results=500 doesn't get a 400 back.
_MAX_PER_PAGE: Final = 200


class OpenAlexSource:
    """A `Source` that fronts the OpenAlex `/works` API."""

    name: str = "openalex"
    id_prefixes: tuple[str, ...] = ("openalex", "doi")

    def __init__(
        self,
        *,
        email: str,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        min_interval_seconds: float = _DEFAULT_MIN_INTERVAL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not email or not email.strip():
            raise ValueError(
                "OpenAlexSource requires a non-empty email — OpenAlex's polite pool "
                "uses ?mailto= for identification, and an empty value silently degrades "
                "to the slow common pool. Set RESEARCH_MCP_OPENALEX_EMAIL."
            )
        self._email = email.strip()
        cache_path = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "research-mcp" / "openalex"
        )
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        self._rate = RateLimiter(min_interval_seconds)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        params: dict[str, str] = {
            "search": query.text,
            "per-page": str(min(query.max_results, _MAX_PER_PAGE)),
        }
        if query.year_min is not None or query.year_max is not None:
            lo = str(query.year_min) if query.year_min is not None else ""
            hi = str(query.year_max) if query.year_max is not None else ""
            params["filter"] = f"publication_year:{lo}-{hi}"
        body = await self._fetch("/works", params)
        return _parse_search_payload(body, source_name=self.name)

    async def fetch(self, paper_id: str) -> Paper | None:
        prefix, _, raw = paper_id.partition(":")
        if prefix not in self.id_prefixes or not raw:
            return None
        # Both routes hit /works/<id>; OpenAlex accepts `/works/W123` and
        # `/works/doi:10.x/y` interchangeably.
        path = f"/works/{raw}" if prefix == "openalex" else f"/works/doi:{raw}"
        body = await self._fetch_or_none(path)
        if body is None:
            return None
        try:
            return _parse_work(json.loads(body))
        except json.JSONDecodeError as exc:
            _log.exception("openalex fetch response not JSON")
            raise SourceUnavailable(self.name, "response was not JSON") from exc

    async def _fetch(self, path: str, params: dict[str, str] | None = None) -> bytes:
        return await self._fetch_inner(path, params, allow_404=False) or b""

    async def _fetch_or_none(self, path: str) -> bytes | None:
        return await self._fetch_inner(path, None, allow_404=True)

    async def _fetch_inner(
        self,
        path: str,
        params: dict[str, str] | None,
        *,
        allow_404: bool,
    ) -> bytes | None:
        # Polite-pool param appears on every request.
        merged_params = {"mailto": self._email}
        if params:
            merged_params.update(params)
        cache_key = path + "?" + "&".join(
            f"{k}={v}" for k, v in sorted(merged_params.items())
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        await self._rate.acquire()

        async def do_request() -> httpx.Response:
            return await self._client.get(_API_BASE + path, params=merged_params)

        try:
            response = await with_backoff(do_request, source_name=self.name)
            # 404 on /works/<id> means "id unknown to OpenAlex" — that's
            # legitimately "not in this source," not a transient failure.
            if allow_404 and response.status_code == 404:
                return None
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("openalex request failed for %s: %s", path, exc)
            raise SourceUnavailable(self.name, str(exc)) from exc
        body = response.content
        self._cache.set(cache_key, body)
        return body


def _parse_search_payload(body: bytes, *, source_name: str) -> list[Paper]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        _log.exception("openalex search response not JSON")
        raise SourceUnavailable(source_name, "response was not JSON") from exc
    raw_results = payload.get("results") or []
    return [paper for raw in raw_results if (paper := _parse_work(raw))]


def _parse_work(raw: dict[str, Any] | None) -> Paper | None:
    if not raw:
        return None
    openalex_id = _strip_openalex_id_url(raw.get("id"))
    if not openalex_id:
        return None

    title = (raw.get("title") or raw.get("display_name") or "").strip()
    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
    published = _parse_publication_date(raw.get("publication_date"), raw.get("publication_year"))
    doi = _strip_doi_url(raw.get("doi"))
    venue = _extract_venue(raw)
    pdf_url = _extract_pdf_url(raw)
    authors = _extract_authors(raw)

    return Paper(
        id=f"openalex:{openalex_id}",
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        url=raw.get("id"),
        venue=venue,
        doi=doi,
        pdf_url=pdf_url,
    )


def _strip_openalex_id_url(value: Any) -> str | None:
    """`https://openalex.org/W123` → `W123`. Anything else → None.

    OpenAlex always returns its own ids as full URLs; some downstream
    endpoints accept the bare W-id, so we canonicalize on the bare form.
    """
    if not isinstance(value, str) or not value:
        return None
    marker = "openalex.org/"
    if marker not in value:
        return None
    bare = value.split(marker, 1)[1]
    return bare or None


def _strip_doi_url(value: Any) -> str | None:
    """`https://doi.org/10.x/y` → `10.x/y`. Handles http and https forms."""
    if not isinstance(value, str) or not value:
        return None
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if value.startswith(prefix):
            return value.removeprefix(prefix) or None
    return value or None


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Flatten OpenAlex's `{word: [positions]}` index back to a string.

    Each word can appear at multiple positions; we sort by position and
    join with spaces. Quirky but preserves word order without storing
    the abstract verbatim (which is OpenAlex's copyright workaround).
    """
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, posns in inverted.items():
        for pos in posns:
            positions.append((pos, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _parse_publication_date(date_str: Any, year: Any) -> date | None:
    if isinstance(date_str, str) and date_str:
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            pass
    if isinstance(year, int):
        try:
            return date(year, 1, 1)
        except ValueError:
            return None
    return None


def _extract_venue(raw: dict[str, Any]) -> str | None:
    primary = raw.get("primary_location") or {}
    source = primary.get("source") if isinstance(primary, dict) else None
    if isinstance(source, dict):
        name = (source.get("display_name") or "").strip()
        if name:
            return name
    return None


def _extract_pdf_url(raw: dict[str, Any]) -> str | None:
    """Prefer `best_oa_location.pdf_url`; fall back to `open_access.oa_url`.

    Both fields can be present; best_oa_location is a curated open-access
    landing, oa_url is whatever URL OpenAlex has cached. Either is more
    likely to actually serve a PDF than primary_location.pdf_url, which
    is often null even when an OA copy exists elsewhere.
    """
    best = raw.get("best_oa_location") or {}
    if isinstance(best, dict):
        url = best.get("pdf_url")
        if isinstance(url, str) and url:
            return url
    oa = raw.get("open_access") or {}
    if isinstance(oa, dict):
        url = oa.get("oa_url")
        if isinstance(url, str) and url:
            return url
    return None


def _extract_authors(raw: dict[str, Any]) -> tuple[Author, ...]:
    authors: list[Author] = []
    for entry in raw.get("authorships") or []:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        if not isinstance(author, dict):
            continue
        name = (author.get("display_name") or "").strip()
        if not name:
            continue
        orcid = author.get("orcid")
        authors.append(Author(name=name, orcid=orcid if isinstance(orcid, str) else None))
    return tuple(authors)
