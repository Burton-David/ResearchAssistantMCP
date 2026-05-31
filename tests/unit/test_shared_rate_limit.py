"""SharedAdaptiveRateLimiter tests.

All state goes through `state_dir=tmp_path`; nothing here touches the real
`~/.cache`. POSIX-only — `fcntl` (and therefore the module under test) is not
available on Windows, which the limiter explicitly defers.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from research_mcp.sources._rate_limit_shared import SharedAdaptiveRateLimiter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl is POSIX-only; Windows deferred"
    ),
]


def test_constructor_validates_arguments(tmp_path) -> None:
    with pytest.raises(ValueError):
        SharedAdaptiveRateLimiter(0, source="s2", state_dir=tmp_path)
    with pytest.raises(ValueError):
        SharedAdaptiveRateLimiter(
            1.0, source="s2", state_dir=tmp_path, max_interval_seconds=0.5
        )
    with pytest.raises(ValueError):
        SharedAdaptiveRateLimiter(1.0, source="s2", state_dir=tmp_path, decay=0)
    with pytest.raises(ValueError):
        SharedAdaptiveRateLimiter(1.0, source="s2", state_dir=tmp_path, decay=1.0)


def test_starts_at_base_interval(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(1.0, source="s2", state_dir=tmp_path)
    assert rl.current_interval == 1.0


def test_record_failure_doubles_up_to_max(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(
        1.0, source="s2", state_dir=tmp_path, max_interval_seconds=10.0
    )
    rl.record_failure()
    assert rl.current_interval == 2.0
    rl.record_failure()
    assert rl.current_interval == 4.0
    rl.record_failure()
    assert rl.current_interval == 8.0
    rl.record_failure()  # 16 → capped at 10
    assert rl.current_interval == 10.0
    rl.record_failure()
    assert rl.current_interval == 10.0


def test_record_success_decays_to_base(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(1.0, source="s2", state_dir=tmp_path, decay=0.5)
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


def test_two_instances_share_interval_via_sidecar(tmp_path) -> None:
    a = SharedAdaptiveRateLimiter(1.0, source="s2", state_dir=tmp_path)
    a.record_failure()
    # A separate instance (simulating a second process) reads the same file.
    b = SharedAdaptiveRateLimiter(1.0, source="s2", state_dir=tmp_path)
    assert b.current_interval == 2.0


async def test_two_instances_serialize_acquire(tmp_path) -> None:
    # The issue-#8 acceptance criterion: two limiters over the same backing
    # file must serialize. The second acquire waits ~base after the first.
    a = SharedAdaptiveRateLimiter(0.1, source="s2", state_dir=tmp_path)
    b = SharedAdaptiveRateLimiter(0.1, source="s2", state_dir=tmp_path)
    t0 = time.monotonic()
    await a.acquire()  # fresh sidecar → fires immediately, reserves its slot
    await b.acquire()  # different instance → waits ~0.1 via the shared file
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.08  # 10% slack for scheduling jitter


async def test_first_acquire_does_not_wait(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(5.0, source="s2", state_dir=tmp_path)
    t0 = time.monotonic()
    await rl.acquire()
    # No prior call recorded → no wait, despite the 5s interval.
    assert time.monotonic() - t0 < 1.0


async def test_record_failure_lengthens_next_wait(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(0.05, source="s2", state_dir=tmp_path)
    await rl.acquire()
    rl.record_failure()  # interval 0.05 → 0.1
    t0 = time.monotonic()
    await rl.acquire()
    assert time.monotonic() - t0 >= 0.08


async def test_corrupt_sidecar_recovers_to_baseline(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(0.05, source="s2", state_dir=tmp_path)
    (tmp_path / "s2.json").write_text("{not valid json")
    assert rl.current_interval == 0.05  # recovers to base, no raise
    await rl.acquire()  # also does not raise


async def test_acquire_writes_atomically_no_tmp_left(tmp_path) -> None:
    rl = SharedAdaptiveRateLimiter(0.01, source="s2", state_dir=tmp_path)
    await rl.acquire()
    data = json.loads((tmp_path / "s2.json").read_text())
    assert set(data) == {"last_call", "interval"}
    assert not list(tmp_path.glob("*.tmp"))
