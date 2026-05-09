"""MemoryIndex behavior tests."""

from __future__ import annotations

import pytest

from research_mcp.domain.index import Index
from research_mcp.domain.paper import Author, Paper
from research_mcp.embedder import FakeEmbedder
from research_mcp.index import MemoryIndex

pytestmark = pytest.mark.unit


def test_protocol_conformance() -> None:
    assert isinstance(MemoryIndex(8), Index)


async def test_upsert_and_count() -> None:
    e = FakeEmbedder(16)
    idx = MemoryIndex(16)
    p1 = Paper(id="arxiv:1", title="t1", abstract="a", authors=(Author("X"),))
    p2 = Paper(id="arxiv:2", title="t2", abstract="b", authors=(Author("Y"),))
    embs = await e.embed(["one", "two"])
    await idx.upsert([p1, p2], embs)
    assert await idx.count() == 2


async def test_upsert_replaces_by_id() -> None:
    e = FakeEmbedder(16)
    idx = MemoryIndex(16)
    p = Paper(id="arxiv:1", title="t1", abstract="a", authors=())
    [v1] = await e.embed(["v1"])
    await idx.upsert([p], [v1])
    p2 = Paper(id="arxiv:1", title="t1-v2", abstract="a", authors=())
    [v2] = await e.embed(["v2"])
    await idx.upsert([p2], [v2])
    assert await idx.count() == 1
    [(found, _)] = await idx.search(v2, k=1)
    assert found.title == "t1-v2"


async def test_search_returns_top_k_in_descending_order() -> None:
    e = FakeEmbedder(64)
    idx = MemoryIndex(64)
    papers = [Paper(id=f"x:{i}", title=f"t{i}", abstract="a", authors=()) for i in range(5)]
    embs = await e.embed([f"text-{i}" for i in range(5)])
    await idx.upsert(papers, embs)
    [query] = await e.embed(["text-2"])
    results = await idx.search(query, k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # the exact-match paper should rank highest
    assert results[0][0].id == "x:2"


async def test_search_self_score_is_one() -> None:
    e = FakeEmbedder(64)
    idx = MemoryIndex(64)
    p = Paper(id="x:1", title="t", abstract="a", authors=())
    [v] = await e.embed(["only"])
    await idx.upsert([p], [v])
    [(_, score)] = await idx.search(v, k=1)
    assert abs(score - 1.0) < 1e-5


async def test_delete_removes_papers() -> None:
    e = FakeEmbedder(16)
    idx = MemoryIndex(16)
    p1 = Paper(id="x:1", title="t1", abstract="a", authors=())
    p2 = Paper(id="x:2", title="t2", abstract="a", authors=())
    embs = await e.embed(["a", "b"])
    await idx.upsert([p1, p2], embs)
    await idx.delete(["x:1"])
    assert await idx.count() == 1


async def test_search_on_empty_index_returns_empty() -> None:
    idx = MemoryIndex(8)
    assert list(await idx.search([0.0] * 8, k=5)) == []


async def test_dimension_mismatch_raises() -> None:
    idx = MemoryIndex(4)
    p = Paper(id="x:1", title="t", abstract="a", authors=())
    with pytest.raises(ValueError):
        await idx.upsert([p], [[0.1, 0.2, 0.3]])
