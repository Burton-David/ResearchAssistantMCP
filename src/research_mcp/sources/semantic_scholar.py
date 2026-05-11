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
from research_mcp.sources._backoff import with_backoff
from research_mcp.sources._cache import DiskCache
from research_mcp.sources._rate_limit import AdaptiveRateLimiter

_log = logging.getLogger(__name__)

_API_BASE: Final = "https://api.semanticscholar.org/graph/v1"
_FIELDS: Final = (
    "paperId,title,abstract,authors,year,publicationDate,externalIds,"
    "openAccessPdf,venue,journal,url,citationCount"
)
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_TIMEOUT: Final = 30.0
# Per S2's 2025 release notes (https://github.com/allenai/s2-folks/
# blob/main/API_RELEASE_NOTES.md):
#   - Unauthenticated: 5,000 requests / 5 minutes, SHARED across all
#     anonymous users. The shared-pool semantics mean even sub-RPS
#     traffic can 429 if other users burn through the pool, so we
#     hold to 1 RPS as a polite ceiling.
#   - Authenticated (default new-key tier): 1 RPS on all endpoints.
#     This is what the user lands in after key approval; bursting
#     past it is the cause of the cascading 429s the chaos tests
#     surfaced in prior rounds (we were running at 10 RPS — 10x
#     the documented limit).
# Higher tiers exist by application; the constructor accepts a
# `min_interval_seconds` override for callers who've negotiated one.
_DEFAULT_MIN_INTERVAL_NOAUTH: Final = 1.0
_DEFAULT_MIN_INTERVAL_AUTH: Final = 1.0


class SemanticScholarSource:
    """A `Source` that fronts the Semantic Scholar Graph API."""

    name: str = "semantic_scholar"
    # arxiv: is included so cross-source enrichment can reach S2 with an
    # arxiv-id-only paper (e.g., a Vaswani lookup via arxiv:1706.03762
    # gets the canonical NeurIPS venue + citation_count from S2). S2's
    # /paper/<id> endpoint accepts arXiv:NNNN.NNNNN natively; the guard
    # was the only thing blocking it. Wiring-layer order keeps arxiv
    # itself listed first, so arxiv-prefixed user fetches still hit
    # ArxivSource — S2 only contributes during enrichment fan-out.
    id_prefixes: tuple[str, ...] = ("s2", "doi", "arxiv")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        min_interval_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        cache_path = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "research-mcp" / "semantic_scholar"
        )
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        # Resolve interval in priority order:
        #   1. explicit constructor arg (programmatic override)
        #   2. RESEARCH_MCP_S2_MIN_INTERVAL env var (for users on
        #      negotiated higher tiers who don't want to fork code)
        #   3. tier-default (1.0s for both with/without key — see
        #      docstring rationale)
        env_override = os.environ.get("RESEARCH_MCP_S2_MIN_INTERVAL")
        if min_interval_seconds is not None:
            interval = min_interval_seconds
        elif env_override:
            try:
                interval = float(env_override)
            except ValueError:
                _log.warning(
                    "RESEARCH_MCP_S2_MIN_INTERVAL=%r is not a float; "
                    "falling back to default", env_override,
                )
                interval = (
                    _DEFAULT_MIN_INTERVAL_AUTH if self._api_key
                    else _DEFAULT_MIN_INTERVAL_NOAUTH
                )
        elif self._api_key:
            interval = _DEFAULT_MIN_INTERVAL_AUTH
        else:
            interval = _DEFAULT_MIN_INTERVAL_NOAUTH
        # AdaptiveRateLimiter learns from 429s. Empirically S2's free-
        # tier "1 RPS" enforcement is tighter than documented: 4
        # quick requests at 1.1s spacing produce a 429 on the 5th.
        # Static interval can't recover from that; adaptive doubles
        # on each 429 and decays back on success — same baseline (1.0s),
        # but headroom when S2 is being strict.
        self._rate = AdaptiveRateLimiter(interval)
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
        # 404 here means "this id doesn't exist in S2's index" — return
        # None per the Source contract, NOT SourceUnavailable. Earlier
        # we surfaced 404s as "semantic_scholar is unavailable, usually
        # transient" which was misleading: a typo'd arxiv id would tell
        # the user to retry instead of "no such paper."
        body = await self._fetch_or_none(f"/paper/{s2_id}", {"fields": _FIELDS})
        if body is None:
            return None
        try:
            return _parse_paper(json.loads(body))
        except json.JSONDecodeError as exc:
            _log.exception("semantic scholar fetch response not JSON")
            # A garbled body from S2 isn't an "id unknown"; treat as transient.
            raise SourceUnavailable(self.name, "response was not JSON") from exc

    async def _fetch_or_none(
        self, path: str, params: dict[str, str]
    ) -> bytes | None:
        """Like `_fetch` but returns None on 404 instead of raising."""
        return await self._fetch_inner(path, params, allow_404=True)

    async def _fetch(self, path: str, params: dict[str, str]) -> bytes:
        body = await self._fetch_inner(path, params, allow_404=False)
        assert body is not None  # allow_404=False makes None impossible
        return body

    async def _fetch_inner(
        self,
        path: str,
        params: dict[str, str],
        *,
        allow_404: bool,
    ) -> bytes | None:
        cache_key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        await self._rate.acquire()
        headers = {"x-api-key": self._api_key} if self._api_key else {}

        async def do_request() -> httpx.Response:
            return await self._client.get(
                _API_BASE + path, params=params, headers=headers
            )

        try:
            response = await with_backoff(
                do_request,
                source_name=self.name,
                on_throttled=self._rate.record_failure,
                on_success=self._rate.record_success,
            )
            # 404 = id unknown; let `fetch()` surface as None. Anything
            # else (5xx, 429 after retries, network error) is transient.
            if allow_404 and response.status_code == 404:
                return None
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
    raw_count = raw.get("citationCount")
    citation_count = raw_count if isinstance(raw_count, int) else None
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
        citation_count=citation_count,
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
