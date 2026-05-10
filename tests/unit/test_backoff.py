"""HTTP retry with exponential backoff for Source adapters."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from research_mcp.sources._backoff import with_backoff

pytestmark = pytest.mark.unit


def _make_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers=headers or {},
        content=b"",
        request=httpx.Request("GET", "https://example.test/"),
    )


def _scripted_responses(
    statuses: list[int],
    headers_per_status: dict[int, dict[str, str]] | None = None,
) -> Iterator[httpx.Response]:
    """Yield responses with the given statuses in order."""
    headers_per_status = headers_per_status or {}
    for status in statuses:
        yield _make_response(status, headers_per_status.get(status))


async def test_returns_immediately_on_2xx() -> None:
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        return _make_response(200)

    response = await with_backoff(do_request, source_name="test")
    assert response.status_code == 200
    assert calls == 1


async def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demonstrates the core behavior: 429 → wait → retry → success."""
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep",
        _no_sleep,
    )
    iterator = _scripted_responses([429, 429, 200])

    async def do_request() -> httpx.Response:
        return next(iterator)

    response = await with_backoff(do_request, source_name="test")
    assert response.status_code == 200


async def test_retries_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", _no_sleep
    )
    iterator = _scripted_responses([503, 502, 200])

    async def do_request() -> httpx.Response:
        return next(iterator)

    response = await with_backoff(do_request, source_name="test")
    assert response.status_code == 200


async def test_does_not_retry_on_4xx_other_than_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 means the resource doesn't exist; retrying won't help and just
    delays the user."""
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", _no_sleep
    )
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        return _make_response(404)

    response = await with_backoff(do_request, source_name="test")
    assert response.status_code == 404
    assert calls == 1


async def test_returns_final_failure_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four attempts (initial + 3 retries by default) return 429.
    `with_backoff` returns the final response — caller decides how to
    surface it."""
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", _no_sleep
    )
    iterator = _scripted_responses([429, 429, 429, 429])

    async def do_request() -> httpx.Response:
        return next(iterator)

    response = await with_backoff(do_request, source_name="test")
    assert response.status_code == 429


async def test_honors_retry_after_header_when_longer_than_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the server says 'wait 5s' and our schedule says 1s, sleep 5s."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", fake_sleep
    )
    iterator = _scripted_responses(
        [429, 200],
        headers_per_status={429: {"retry-after": "5"}},
    )

    async def do_request() -> httpx.Response:
        return next(iterator)

    await with_backoff(do_request, source_name="test")
    # First (and only) sleep should honor Retry-After=5 over our 1s schedule.
    assert sleeps == [5.0]


async def test_caps_retry_after_to_30_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving server claiming 'Retry-After: 600' shouldn't pin us."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", fake_sleep
    )
    iterator = _scripted_responses(
        [429, 200],
        headers_per_status={429: {"retry-after": "600"}},
    )

    async def do_request() -> httpx.Response:
        return next(iterator)

    await with_backoff(do_request, source_name="test")
    assert sleeps == [30.0]  # capped


async def test_retries_on_network_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", _no_sleep
    )
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("simulated network glitch")
        return _make_response(200)

    response = await with_backoff(do_request, source_name="test")
    assert response.status_code == 200
    assert calls == 3


async def test_propagates_network_error_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_mcp.sources._backoff.asyncio.sleep", _no_sleep
    )

    async def do_request() -> httpx.Response:
        raise httpx.ConnectError("network is down for everyone")

    with pytest.raises(httpx.ConnectError):
        await with_backoff(do_request, source_name="test")


async def _no_sleep(seconds: float) -> None:
    """Sleep replacement so tests run instantly."""
    return None
