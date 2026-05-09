"""FaissIndex behavior tests — persistence and atomic-write checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from research_mcp.domain.index import Index
from research_mcp.domain.paper import Author, Paper
from research_mcp.embedder import FakeEmbedder
from research_mcp.index import FaissIndex

pytestmark = pytest.mark.unit


def test_protocol_conformance(tmp_path: Path) -> None:
    idx = FaissIndex(tmp_path, dimension=8)
    try:
        assert isinstance(idx, Index)
    finally:
        idx.close()


async def test_upsert_persists_across_reload(tmp_path: Path) -> None:
    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    p = Paper(id="arxiv:1", title="t", abstract="a", authors=(Author("X"),))
    [v] = await e.embed(["x"])
    await idx.upsert([p], [v])
    idx.close()
    idx2 = FaissIndex(tmp_path, 16)
    try:
        assert await idx2.count() == 1
        [(found, _)] = await idx2.search(v, k=1)
        assert found.id == p.id
        assert found.title == "t"
    finally:
        idx2.close()


async def test_atomic_write_leaves_no_tmp_files(tmp_path: Path) -> None:
    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    try:
        p = Paper(id="x:1", title="t", abstract="a", authors=())
        [v] = await e.embed(["v"])
        await idx.upsert([p], [v])
        assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    finally:
        idx.close()


async def test_dimension_mismatch_on_reload_raises(tmp_path: Path) -> None:
    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    p = Paper(id="x:1", title="t", abstract="a", authors=())
    [v] = await e.embed(["v"])
    await idx.upsert([p], [v])
    idx.close()
    with pytest.raises(ValueError):
        FaissIndex(tmp_path, 32)


async def test_delete_then_recall_does_not_return_deleted(tmp_path: Path) -> None:
    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    try:
        p1 = Paper(id="x:1", title="t1", abstract="a", authors=())
        p2 = Paper(id="x:2", title="t2", abstract="a", authors=())
        embs = await e.embed(["a", "b"])
        await idx.upsert([p1, p2], embs)
        await idx.delete(["x:1"])
        results = await idx.search(embs[0], k=2)
        assert all(p.id != "x:1" for p, _ in results)
    finally:
        idx.close()
