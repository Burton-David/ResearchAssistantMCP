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


async def test_search_uses_one_sqlite_query_for_top_k(tmp_path: Path) -> None:
    """Previous implementation issued one SELECT per result (N+1). The IN-clause
    rewrite should produce a single SELECT regardless of k."""
    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    try:
        papers = [Paper(id=f"x:{i}", title=f"t{i}", abstract="a", authors=()) for i in range(5)]
        embs = await e.embed([f"text-{i}" for i in range(5)])
        await idx.upsert(papers, embs)

        # Use sqlite3's trace callback to count metadata SELECTs during search.
        # The callback fires for every executed statement; we filter by the
        # specific query the search path issues.
        select_count = 0

        def trace(stmt: str) -> None:
            nonlocal select_count
            if "SELECT faiss_id, body FROM papers WHERE faiss_id IN" in stmt:
                select_count += 1

        # The search path runs in a worker thread (asyncio.to_thread), so we
        # have to install the trace on the connection that thread will use.
        # Easiest: trigger a no-op search first to materialize the worker
        # thread's connection, then install the trace on it.
        # Actually simpler: install on the main-thread connection and also
        # call the search path synchronously through _search_sync, which uses
        # whatever thread we're on — the lock is asyncio-level and we're not
        # contending here.
        idx._conn().set_trace_callback(trace)
        idx._search_sync(embs[0], 5)
        assert select_count == 1, f"expected 1 metadata SELECT, got {select_count}"
    finally:
        idx.close()


async def test_concurrent_upsert_and_search_does_not_race(tmp_path: Path) -> None:
    """The asyncio.Lock around upsert/search prevents the 'search-during-upsert'
    race that would corrupt FAISS state. We can't reliably observe the bug
    without the lock, but we can at least verify both paths complete cleanly
    under contention without exceptions or missing rows."""
    import asyncio as aio

    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    try:
        papers = [Paper(id=f"x:{i}", title=f"t{i}", abstract="a", authors=()) for i in range(50)]
        embs = await e.embed([f"text-{i}" for i in range(50)])

        async def insert(i: int) -> None:
            await idx.upsert([papers[i]], [embs[i]])

        async def query(i: int) -> None:
            await idx.search(embs[i % 50], k=3)

        # Interleave 50 inserts with 50 searches.
        tasks = [insert(i) for i in range(50)] + [query(i) for i in range(50)]
        await aio.gather(*tasks)
        assert await idx.count() == 50
    finally:
        idx.close()


async def test_reconcile_drops_orphan_sqlite_rows(tmp_path: Path) -> None:
    """Simulate a crash between SQLite commit and FAISS rename: a SQLite row
    exists for a faiss_id that the loaded FAISS doesn't know about. On
    next __init__, the orphan must be dropped so count() and search lookups
    stay consistent."""
    import json
    import sqlite3

    e = FakeEmbedder(16)
    idx = FaissIndex(tmp_path, 16)
    p = Paper(id="x:1", title="t", abstract="a", authors=())
    [v] = await e.embed(["v"])
    await idx.upsert([p], [v])
    idx.close()

    # Inject a phantom row that no faiss_id maps to.
    conn = sqlite3.connect(tmp_path / "papers.sqlite")
    conn.execute(
        "INSERT INTO papers (faiss_id, paper_id, body) VALUES (?, ?, ?)",
        (
            999,
            "x:phantom",
            json.dumps({"id": "x:phantom", "title": "phantom", "abstract": "",
                        "authors": [], "published": None, "url": None, "venue": None,
                        "doi": None, "arxiv_id": None, "semantic_scholar_id": None,
                        "pdf_url": None, "full_text": None, "metadata": {}}),
        ),
    )
    conn.commit()
    conn.close()

    # Re-open the index; reconcile should drop the orphan.
    idx2 = FaissIndex(tmp_path, 16)
    try:
        assert await idx2.count() == 1  # only the original row
        rows = idx2._conn().execute(
            "SELECT paper_id FROM papers"
        ).fetchall()
        assert {r["paper_id"] for r in rows} == {"x:1"}
    finally:
        idx2.close()
