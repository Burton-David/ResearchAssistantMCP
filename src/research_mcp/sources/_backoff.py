"""HTTP retry with exponential backoff for Source adapters.

Both arXiv and Semantic Scholar return 429 (rate-limited) and 5xx (server
busy) under load. The Source contract used to swallow these as None;
after the polish-batch contract update they propagate as
`SourceUnavailable`. Both behaviors fail one request — but most of these
failures are transient, and the upstream's own Retry-After header tells
us when the burst has cleared.

`with_backoff` runs the request, retries on retryable statuses (429 +
5xx) and network errors with a 1s/2s/4s schedule (caps at three retries
to bound total wait at ~7s before giving up), and honors Retry-After
when the server suggests a longer wait. Final failure propagates as
the underlying httpx exception so the calling adapter can wrap it as
`SourceUnavailable` with a normal short-reason summary.

Both source adapters now route through this helper. Required behavior
per the Semantic Scholar API key application checkbox ("I will apply
exponential backoff…") and good citizenship more broadly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

from research_mcp.errors import redact_secrets

_log = logging.getLogger(__name__)

# Retry on rate-limited and transient server errors. 4xx other than 429
# are deliberately not retried — they indicate a malformed request that
# won't get better on the next attempt.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Exponential backoff: 1s, 2s, 4s. Three retries plus the initial attempt
# means a total of four tries and a worst-case ~7s wait. Higher caps don't
# help — the 429 cooldown for S2 unauth is on the order of minutes, so
# retrying for longer just delays the user without succeeding.
_DEFAULT_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)

# Retry-After can suggest very long waits (S2 sometimes asks for 60s+).
# Cap so a misbehaving upstream can't pin a single request for minutes.
_MAX_RETRY_AFTER_SECONDS = 30.0


async def with_backoff(
    do_request: Callable[[], Awaitable[httpx.Response]],
    *,
    source_name: str,
    delays: tuple[float, ...] = _DEFAULT_DELAYS,
    on_throttled: Callable[[], None] | None = None,
    on_success: Callable[[], None] | None = None,
) -> httpx.Response:
    """Run `do_request()` with exponential backoff on retryable failures.

    Returns the final httpx.Response (which the caller should then pass
    through `raise_for_status` for a clean error if non-2xx). On a
    network-level failure (httpx.HTTPError raised before a response
    arrives) all attempts are exhausted before the exception propagates.

    `delays` is the sequence of inter-attempt sleeps; total attempts =
    `len(delays) + 1`.

    `on_throttled` / `on_success` are optional callbacks for an
    AdaptiveRateLimiter to learn from the response. They fire when we
    see a 429 (throttled — slow down) and when a request returns 2xx
    or any non-retryable status (success — relax back toward baseline).
    """
    last_network_error: httpx.HTTPError | None = None
    last_response: httpx.Response | None = None
    for attempt in range(len(delays) + 1):
        if attempt > 0:
            sleep_for = delays[attempt - 1]
            # If the previous attempt was a real Response with Retry-After,
            # honor that suggestion (capped) instead of our schedule.
            if last_response is not None:
                hint = _parse_retry_after(
                    last_response.headers.get("retry-after")
                )
                if hint is not None:
                    sleep_for = max(sleep_for, min(hint, _MAX_RETRY_AFTER_SECONDS))
            _log.info(
                "%s: backing off %.1fs before retry %d/%d",
                source_name, sleep_for, attempt, len(delays),
            )
            await asyncio.sleep(sleep_for)
        try:
            response = await do_request()
        except httpx.HTTPError as exc:
            last_network_error = exc
            last_response = None
            if attempt == len(delays):
                raise
            _log.warning(
                "%s: network error on attempt %d/%d (%s); retrying",
                source_name, attempt + 1, len(delays) + 1,
                redact_secrets(str(exc)),
            )
            continue
        if response.status_code not in _RETRYABLE_STATUS:
            if on_success is not None:
                on_success()
            return response
        if response.status_code == 429 and on_throttled is not None:
            on_throttled()
        last_response = response
        if attempt == len(delays):
            return response  # caller's raise_for_status will produce the error
        _log.warning(
            "%s: HTTP %d on attempt %d/%d; retrying",
            source_name, response.status_code, attempt + 1, len(delays) + 1,
        )
    # Unreachable — the loop returns or raises on every path. Keeping the
    # statement here so mypy infers the return type cleanly.
    if last_response is not None:
        return last_response
    assert last_network_error is not None
    raise last_network_error


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. Spec allows seconds-as-int OR HTTP-date.

    We honor seconds. HTTP-date is rare enough and computing the delta is
    error-prone (timezone, clock skew); we'd rather fall back to our
    scheduled delay than parse it wrong.
    """
    if value is None:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
