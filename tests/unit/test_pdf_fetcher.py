"""HttpPdfFetcher tests — offline, no real network.

httpx is stubbed with `MockTransport` (the pattern from `test_h_index.py`); a
real but tiny PDF is built in-process with reportlab so pdfplumber has genuine
bytes to extract, and nothing binary is committed to the repo.
"""

from __future__ import annotations

import io
import sys

import httpx
import pytest
from reportlab.pdfgen import canvas

from research_mcp.pdf import HttpPdfFetcher

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session")
def text_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Introduction")
    c.drawString(72, 700, "This is the body text of a test paper. " * 20)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def scanned_pdf_bytes() -> bytes:
    # A page with only a drawn shape — no text layer, like a scanned image.
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.rect(72, 700, 200, 50, fill=0)
    c.showPage()
    c.save()
    return buf.getvalue()


def _fetcher(handler, tmp_path, **kwargs) -> HttpPdfFetcher:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return HttpPdfFetcher(cache_dir=tmp_path, client=client, **kwargs)


def _pdf_handler(body: bytes, content_type: str = "application/pdf"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    return handler


async def test_fetch_text_extracts_body(text_pdf_bytes: bytes, tmp_path) -> None:
    fetcher = _fetcher(_pdf_handler(text_pdf_bytes), tmp_path)
    text = await fetcher.fetch_text("arxiv:1", "https://example.org/1.pdf")
    assert text is not None
    assert "Introduction" in text
    assert "body text of a test paper" in text


async def test_fetch_text_caches_to_disk(text_pdf_bytes: bytes, tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, content=text_pdf_bytes, headers={"content-type": "application/pdf"}
        )

    fetcher = _fetcher(handler, tmp_path)
    first = await fetcher.fetch_text("arxiv:1", "https://example.org/1.pdf")
    second = await fetcher.fetch_text("arxiv:1", "https://example.org/1.pdf")
    assert first == second
    assert calls == 1  # second call served from the disk cache
    assert list(tmp_path.glob("*.txt"))


async def test_fetch_text_returns_none_on_http_error(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    fetcher = _fetcher(handler, tmp_path)
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/missing.pdf") is None


async def test_fetch_text_returns_none_for_html_landing_page(tmp_path) -> None:
    fetcher = _fetcher(
        _pdf_handler(b"<html><body>Not a PDF</body></html>", content_type="text/html"),
        tmp_path,
    )
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/landing") is None


async def test_fetch_text_returns_none_for_non_pdf_bytes(tmp_path) -> None:
    # Correct content-type but the body isn't a PDF — the magic-byte guard catches it.
    fetcher = _fetcher(_pdf_handler(b"this is not a pdf at all"), tmp_path)
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/fake.pdf") is None


async def test_fetch_text_respects_max_bytes(text_pdf_bytes: bytes, tmp_path) -> None:
    fetcher = _fetcher(_pdf_handler(text_pdf_bytes), tmp_path, max_bytes=10)
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/big.pdf") is None


async def test_fetch_text_caps_at_max_chars(text_pdf_bytes: bytes, tmp_path) -> None:
    fetcher = _fetcher(_pdf_handler(text_pdf_bytes), tmp_path, max_chars=300)
    text = await fetcher.fetch_text("arxiv:1", "https://example.org/1.pdf")
    assert text is not None
    assert len(text) <= 300


async def test_scanned_pdf_returns_none_and_caches(scanned_pdf_bytes: bytes, tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, content=scanned_pdf_bytes, headers={"content-type": "application/pdf"}
        )

    fetcher = _fetcher(handler, tmp_path)
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/scan.pdf") is None
    # Cached as empty so a scanned PDF isn't re-downloaded next ingest.
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/scan.pdf") is None
    assert calls == 1


async def test_missing_pdfplumber_returns_none(
    text_pdf_bytes: bytes, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate the optional extra not being installed: `import pdfplumber`
    # raises ImportError inside _extract_sync, which degrades to None.
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    fetcher = _fetcher(_pdf_handler(text_pdf_bytes), tmp_path)
    assert await fetcher.fetch_text("arxiv:1", "https://example.org/1.pdf") is None
