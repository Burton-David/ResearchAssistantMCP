"""FakeEmbedder behavior tests."""

from __future__ import annotations

import math

import pytest

from research_mcp.domain.embedder import Embedder
from research_mcp.embedder import FakeEmbedder

pytestmark = pytest.mark.unit


def test_protocol_conformance() -> None:
    assert isinstance(FakeEmbedder(16), Embedder)


async def test_dimension_matches_output() -> None:
    e = FakeEmbedder(32)
    [v] = await e.embed(["hello"])
    assert len(v) == 32


async def test_deterministic() -> None:
    e = FakeEmbedder(64)
    [a] = await e.embed(["repeatable"])
    [b] = await e.embed(["repeatable"])
    assert list(a) == list(b)


async def test_normalized_to_unit_length() -> None:
    e = FakeEmbedder(64)
    [v] = await e.embed(["any text"])
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_rejects_zero_dimension() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(0)
