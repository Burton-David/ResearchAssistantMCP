"""PDF fetching + text extraction for full-text ingest.

Search results carry only title + abstract. During ingest, `LibraryService`
calls a `PdfFetcher` to populate `Paper.full_text` from the paper's `pdf_url`
when one is available, so the section-aware chunker and the analyzer see real
body text instead of inferring from the abstract.

Extraction is best-effort: a PDF that's missing, unreachable, not actually a
PDF, larger than the byte cap, or scanned-image-only yields `None`, and ingest
proceeds with `full_text` unset. The fetcher never raises for a PDF-side
problem.

pdfplumber (MIT, on top of pdfminer.six) does the extraction. It's an optional
extra — ``pip install research-mcp[pdf]``; when it isn't installed every fetch
returns `None` with a one-time hint.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import httpx

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: Final = 30.0
# Download ceiling. A 200-page paper is ~2-5 MiB; 25 MiB leaves room for
# figure-heavy PDFs while bounding memory and refusing pathological files.
_DEFAULT_MAX_BYTES: Final = 25 * 1024 * 1024
# Extract-time character cap (~160 pages at ~3k chars/page). Set well above the
# analyzer's 60k prompt truncation (paper_analyzer/_schema.py) because the
# chunker wants the *whole* body to produce section-tagged chunks — capping at
# 60k would silently drop results/conclusion sections. Bounds the embedder
# input and the FAISS sidecar size.
_DEFAULT_MAX_CHARS: Final = 500_000
# Below this many extracted characters, treat the PDF as scanned/image-only and
# decline it — OCR is out of scope.
_MIN_EXTRACTABLE_CHARS: Final = 200

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "research-mcp" / "pdf_text"


@runtime_checkable
class PdfFetcher(Protocol):
    async def fetch_text(self, paper_id: str, pdf_url: str) -> str | None:
        """Return extracted body text for `pdf_url`, or `None`.

        `None` means the PDF is unavailable, unfetchable, not a PDF, over the
        size cap, or scanned-image-only. The method never raises for a PDF-side
        problem — `None` is the graceful-degrade signal the caller relies on.
        `paper_id` is the cache key.
        """
        ...


class PdfTextCache:
    """Disk cache for extracted PDF text, keyed by `paper.id`.

    No TTL — a paper's PDF text doesn't change and re-extraction is expensive,
    so this is deliberately separate from the 24h API-response `DiskCache`. An
    empty cached string is a valid entry: it records "we fetched this and found
    no extractable text" so a scanned PDF isn't re-downloaded on every ingest.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, paper_id: str) -> str | None:
        path = self._path_for(paper_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def set(self, paper_id: str, text: str) -> None:
        path = self._path_for(paper_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _path_for(self, paper_id: str) -> Path:
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.txt"


class HttpPdfFetcher:
    """`PdfFetcher` that downloads over httpx and extracts text with pdfplumber."""

    def __init__(
        self,
        *,
        cache: PdfTextCache | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> None:
        if cache is not None:
            self._cache = cache
        elif cache_dir is not None:
            self._cache = PdfTextCache(cache_dir)
        else:
            self._cache = PdfTextCache(_DEFAULT_CACHE_DIR)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_chars = max_chars

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_text(self, paper_id: str, pdf_url: str) -> str | None:
        cached = self._cache.get(paper_id)
        if cached is not None:
            return cached or None  # a cached "" marks a scanned/no-text PDF
        data = await self._download(pdf_url)
        if data is None:
            return None
        text = await asyncio.to_thread(self._extract_sync, data)
        if text is None:
            # pdfplumber isn't installed — don't cache, so a later install retries.
            return None
        self._cache.set(paper_id, text)
        return text or None

    async def _download(self, pdf_url: str) -> bytes | None:
        try:
            async with self._client.stream(
                "GET", pdf_url, timeout=self._timeout, follow_redirects=True
            ) as resp:
                if resp.status_code != 200:
                    _log.info("PDF fetch %s -> HTTP %d; skipping", pdf_url, resp.status_code)
                    return None
                ctype = resp.headers.get("content-type", "").lower()
                if "html" in ctype:
                    _log.info("PDF url %s served HTML (%s); skipping", pdf_url, ctype)
                    return None
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > self._max_bytes:
                        _log.info(
                            "PDF %s exceeds %d-byte cap; skipping", pdf_url, self._max_bytes
                        )
                        return None
        except (httpx.HTTPError, OSError) as exc:
            _log.warning("PDF fetch failed for %s (ignored): %s", pdf_url, exc)
            return None
        if not bytes(buf[:8]).startswith(b"%PDF"):
            _log.info("PDF url %s did not return PDF bytes; skipping", pdf_url)
            return None
        return bytes(buf)

    def _extract_sync(self, data: bytes) -> str | None:
        try:
            import pdfplumber
        except ImportError:
            _log.warning(
                "pdfplumber not installed; install research-mcp[pdf] for full-text ingest"
            )
            return None
        parts: list[str] = []
        total = 0
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    parts.append(page_text)
                    total += len(page_text)
                    if total >= self._max_chars:
                        break
        except Exception as exc:
            # pdfminer raises a grab-bag of exceptions on malformed PDFs; none
            # should break ingest. Cache as "no text" so we don't refetch.
            _log.warning("PDF text extraction failed (ignored): %s", exc)
            return ""
        text = "\n\n".join(parts).strip()[: self._max_chars]
        if len(text) < _MIN_EXTRACTABLE_CHARS:
            return ""  # scanned/image-only or near-empty
        return text
