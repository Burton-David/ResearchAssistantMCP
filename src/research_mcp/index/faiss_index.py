"""On-disk FAISS index with a SQLite metadata sidecar.

Layout under `RESEARCH_MCP_INDEX_PATH`:

    {path}/
      vectors.faiss      # the FAISS IndexIDMap2(IndexFlatIP)
      papers.sqlite      # rowid (== faiss id) -> JSON-encoded Paper

Embeddings are L2-normalized on insert so inner-product search ranks by cosine.

Concurrency: a single asyncio.Lock serializes upsert/delete/search. FAISS
IndexFlatIP is not thread-safe under mixed reads and writes, and the
SQLite sidecar reads can race against in-flight commits, so we serialize.
The cost is throughput on parallel reads — fine for a single-user research
tool, would need a reader/writer split for a multi-tenant deployment.

Per-thread SQLite connections: even with check_same_thread=False, sharing
one connection across threads is undefined per the SQLite docs. Each
worker thread gets its own connection via threading.local; the connection
opens lazily on first use.

Crash safety: writes happen as
  1. Build the new in-memory FAISS state.
  2. Write `vectors.faiss.tmp` atomically (still on disk under the same
     filename until rename).
  3. SQLite executemany + commit.
  4. Atomic rename of `vectors.faiss.tmp` to `vectors.faiss`.

If the process dies between steps 3 and 4, SQLite holds rows whose
faiss_id isn't in the loaded `vectors.faiss`. On next startup, the
reconcile step deletes those orphan SQLite rows so `count()` and
search-result lookups stay consistent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import faiss  # type: ignore[import-untyped]  # faiss-cpu has no type stubs
import numpy as np

from research_mcp.domain.paper import Paper
from research_mcp.index._codec import paper_from_dict, paper_to_dict

_log = logging.getLogger(__name__)


class FaissIndex:
    """A persistent FAISS-backed `Index`.

    `path` is a directory; created on demand. Pass via constructor or set
    `RESEARCH_MCP_INDEX_PATH` and call `FaissIndex.from_env(dimension)`.
    """

    def __init__(self, path: str | os.PathLike[str], dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._faiss_path = self._dir / "vectors.faiss"
        self._sqlite_path = self._dir / "papers.sqlite"
        self._lock = asyncio.Lock()
        self._index = self._load_or_create_index()

        # Per-thread connection store. Each thread that runs upsert/search/
        # delete via asyncio.to_thread() gets its own connection on first
        # use. Closing the FaissIndex closes the bookkeeping connection
        # used here in __init__; the per-thread ones close when the worker
        # threads exit.
        self._tls = threading.local()
        self._init_schema()

        self._reconcile_orphans()

    @classmethod
    def from_env(cls, dimension: int) -> FaissIndex:
        path = os.environ.get("RESEARCH_MCP_INDEX_PATH")
        if not path:
            raise RuntimeError("RESEARCH_MCP_INDEX_PATH is not set")
        return cls(path, dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _conn(self) -> sqlite3.Connection:
        """Return the calling thread's SQLite connection, creating it on
        first use. SQLite forbids cross-thread connection use; this gives
        each worker thread its own."""
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._sqlite_path)
            conn.row_factory = sqlite3.Row
            self._tls.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS papers ("
            " faiss_id INTEGER PRIMARY KEY,"
            " paper_id TEXT NOT NULL UNIQUE,"
            " body TEXT NOT NULL"
            ")"
        )
        conn.commit()

    def _reconcile_orphans(self) -> None:
        """Drop SQLite rows whose `faiss_id` isn't in the loaded FAISS index.

        Recovers from a crash that landed between SQLite commit and the FAISS
        rename: SQLite is ahead, FAISS is behind, the orphan rows can never
        be reached by search anyway. Deleting them makes `count()` honest.
        """
        live_ids = self._collect_live_faiss_ids()
        conn = self._conn()
        cur = conn.execute("SELECT faiss_id FROM papers")
        sqlite_ids = [int(r["faiss_id"]) for r in cur.fetchall()]
        orphans = [fid for fid in sqlite_ids if fid not in live_ids]
        if not orphans:
            return
        _log.warning(
            "FaissIndex: dropping %d SQLite rows orphaned from FAISS "
            "(likely from a crash between SQLite commit and FAISS rename)",
            len(orphans),
        )
        conn.executemany("DELETE FROM papers WHERE faiss_id = ?", [(fid,) for fid in orphans])
        conn.commit()

    def _collect_live_faiss_ids(self) -> set[int]:
        # IndexIDMap2 keeps an internal id_map array; faiss exposes it as
        # an int64 view. Extract via reconstruct loop is expensive — instead,
        # we rely on a search over a zero vector to enumerate, which is
        # cheap-ish for IndexFlatIP since k can equal ntotal. For real-world
        # sizes this is fine; for >100k papers, switch to walking the C++
        # id_map directly.
        if self._index.ntotal == 0:
            return set()
        zero = np.zeros((1, self._dimension), dtype=np.float32)
        _, ids = self._index.search(zero, self._index.ntotal)
        return {int(i) for i in ids[0] if i != -1}

    def _load_or_create_index(self) -> faiss.Index:
        if self._faiss_path.exists():
            loaded = faiss.read_index(str(self._faiss_path))
            if loaded.d != self._dimension:
                raise ValueError(
                    f"index on disk has dim {loaded.d}, expected {self._dimension}"
                )
            return loaded
        base = faiss.IndexFlatIP(self._dimension)
        return faiss.IndexIDMap2(base)

    def _persist_vectors(self) -> None:
        tmp = self._faiss_path.with_suffix(self._faiss_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp))
        os.replace(tmp, self._faiss_path)

    async def upsert(
        self,
        papers: Sequence[Paper],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(papers) != len(embeddings):
            raise ValueError("papers and embeddings must have equal length")
        if not papers:
            return
        async with self._lock:
            await asyncio.to_thread(self._upsert_sync, list(papers), list(embeddings))

    def _upsert_sync(
        self, papers: list[Paper], embeddings: list[Sequence[float]]
    ) -> None:
        conn = self._conn()
        existing_rows = conn.execute(
            f"SELECT faiss_id, paper_id FROM papers WHERE paper_id IN ({_placeholders(len(papers))})",
            [p.id for p in papers],
        ).fetchall()
        existing = {row["paper_id"]: row["faiss_id"] for row in existing_rows}
        if existing:
            self._index.remove_ids(np.array(list(existing.values()), dtype=np.int64))

        max_row = conn.execute("SELECT MAX(faiss_id) FROM papers").fetchone()[0] or 0
        next_id = max_row + 1
        new_ids: list[int] = []
        for paper in papers:
            if paper.id in existing:
                new_ids.append(existing[paper.id])
            else:
                new_ids.append(next_id)
                next_id += 1

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.shape != (len(papers), self._dimension):
            raise ValueError(
                f"embeddings shape {matrix.shape} != ({len(papers)}, {self._dimension})"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        self._index.add_with_ids(matrix, np.asarray(new_ids, dtype=np.int64))

        # New crash-safe order:
        # 1. Build the new FAISS state in-memory (done above via remove + add).
        # 2. Write vectors.faiss.tmp to disk.
        # 3. SQLite commit.
        # 4. Atomic rename.
        # If we die between (3) and (4), SQLite holds rows pointing at
        # faiss_ids that aren't in the disk FAISS yet. _reconcile_orphans
        # on next startup deletes those rows.
        tmp = self._faiss_path.with_suffix(self._faiss_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp))
        try:
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO papers (faiss_id, paper_id, body) VALUES (?, ?, ?)",
                    [
                        (fid, p.id, json.dumps(paper_to_dict(p)))
                        for fid, p in zip(new_ids, papers, strict=True)
                    ],
                )
        except Exception:
            # SQLite commit failed; leave the .tmp in place but don't rename.
            # Next startup will see only the old vectors.faiss; reconcile is
            # a no-op since SQLite never advanced.
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, self._faiss_path)

    async def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
    ) -> Sequence[tuple[Paper, float]]:
        async with self._lock:
            if self._index.ntotal == 0:
                return []
            return await asyncio.to_thread(self._search_sync, embedding, k)

    def _search_sync(
        self, embedding: Sequence[float], k: int
    ) -> list[tuple[Paper, float]]:
        query = np.asarray([embedding], dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        k = min(k, self._index.ntotal)
        scores, ids = self._index.search(query, k)
        # One IN-clause query for all k rows, then re-sort in Python by FAISS
        # rank. Replaces the previous N+1 SELECT loop.
        ranked_ids = [int(i) for i in ids[0] if i != -1]
        if not ranked_ids:
            return []
        rows = self._conn().execute(
            f"SELECT faiss_id, body FROM papers WHERE faiss_id IN ({_placeholders(len(ranked_ids))})",
            ranked_ids,
        ).fetchall()
        body_by_id = {int(r["faiss_id"]): r["body"] for r in rows}
        results: list[tuple[Paper, float]] = []
        for fid, score in zip(ids[0], scores[0], strict=True):
            if fid == -1:
                continue
            body = body_by_id.get(int(fid))
            if body is None:  # post-reconcile, this should never happen
                continue
            results.append((paper_from_dict(json.loads(body)), float(score)))
        return results

    async def delete(self, paper_ids: Sequence[str]) -> None:
        if not paper_ids:
            return
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, list(paper_ids))

    def _delete_sync(self, paper_ids: list[str]) -> None:
        conn = self._conn()
        rows = conn.execute(
            f"SELECT faiss_id FROM papers WHERE paper_id IN ({_placeholders(len(paper_ids))})",
            paper_ids,
        ).fetchall()
        faiss_ids = [int(r["faiss_id"]) for r in rows]
        if not faiss_ids:
            return
        self._index.remove_ids(np.asarray(faiss_ids, dtype=np.int64))
        # Same crash-safe order as upsert.
        tmp = self._faiss_path.with_suffix(self._faiss_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp))
        try:
            with conn:
                conn.execute(
                    f"DELETE FROM papers WHERE paper_id IN ({_placeholders(len(paper_ids))})",
                    paper_ids,
                )
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, self._faiss_path)

    async def count(self) -> int:
        async with self._lock:
            return int(self._index.ntotal)

    def close(self) -> None:
        # Close the bookkeeping connection used in __init__ / _reconcile_orphans.
        # Other threads' connections are closed by the GC when those threads exit.
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)
