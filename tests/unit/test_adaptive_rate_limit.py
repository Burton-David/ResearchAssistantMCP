"""AdaptiveRateLimiter tests."""

from __future__ import annotations

import time

import pytest

from research_mcp.sources._rate_limit import AdaptiveRateLimiter

pytestmark = pytest.mark.unit


def test_constructor_validates_arguments() -> None:
    with pytest.raises(ValueError):
        AdaptiveRateLimiter(0)
    with pytest.raises(ValueError):
        AdaptiveRateLimiter(1.0, max_interval_seconds=0.5)
    with pytest.raises(ValueError):
        AdaptiveRateLimiter(1.0, decay=0)
    with pytest.raises(ValueError):
        AdaptiveRateLimiter(1.0, decay=1.0)


def test_starts_at_base_interval() -> None:
    rl = AdaptiveRateLimiter(1.0)
    assert rl.current_interval == 1.0


def test_record_failure_doubles_interval_up_to_max() -> None:
    rl = AdaptiveRateLimiter(1.0, max_interval_seconds=10.0)
    rl.record_failure()
    assert rl.current_interval == 2.0
    rl.record_failure()
    assert rl.current_interval == 4.0
    rl.record_failure()
    assert rl.current_interval == 8.0
    rl.record_failure()  # would go to 16, capped at 10
    assert rl.current_interval == 10.0
    rl.record_failure()
    assert rl.current_interval == 10.0


def test_record_success_decays_back_toward_baseline() -> None:
    rl = AdaptiveRateLimiter(1.0, decay=0.5)
    rl.record_failure()
    rl.record_failure()
    rl.record_failure()
    assert rl.current_interval == 8.0
    rl.record_success()
    assert rl.current_interval == 4.0
    rl.record_success()
    assert rl.current_interval == 2.0
    rl.record_success()
    assert rl.current_interval == 1.0
    rl.record_success()
    assert rl.current_interval == 1.0  # floored at base


def test_success_at_baseline_is_a_noop() -> None:
    rl = AdaptiveRateLimiter(1.0)
    rl.record_success()
    assert rl.current_interval == 1.0


async def test_acquire_respects_current_interval() -> None:
    """Smoke test that acquire() honors the (possibly-expanded) interval.
    Uses a small base so the test runs in <1s."""
    rl = AdaptiveRateLimiter(0.05)
    t0 = time.monotonic()
    await rl.acquire()
    await rl.acquire()  # waits ~0.05s
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.04  # allow 10% slack for scheduling jitter

    # After a failure, interval doubles to 0.1s.
    rl.record_failure()
    t1 = time.monotonic()
    await rl.acquire()
    elapsed_after = time.monotonic() - t1
    assert elapsed_after >= 0.08
