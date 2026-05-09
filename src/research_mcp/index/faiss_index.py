"""On-disk FAISS index with a SQLite metadata sidecar.

Layout under `RESEARCH_MCP_INDEX_PATH`:

    {path}/
      vectors.faiss      # the FAISS IndexIDMap2(IndexFlatIP)
      papers.sqlite      # rowid (== faiss id) -> JSON-encoded Paper

Embeddings are L2-normalized on insert so inner-product search ranks by cosine.
Writes are atomic: vectors are persisted to `vectors.faiss.tmp` then renamed.
SQLite is left to its own ACID guarantees.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import faiss  # type: ignore[import-untyped]  # faiss-cpu has no type stubs
import numpy as np

from research_mcp.domain.paper import Paper
from research_mcp.index._codec import paper_from_dict, paper_to_dict


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
        self._db = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS papers ("
            " faiss_id INTEGER PRIMARY KEY,"
            " paper_id TEXT NOT NULL UNIQUE,"
            " body TEXT NOT NULL"
            ")"
        )
        self._db.commit()

    @classmethod
    def from_env(cls, dimension: int) -> FaissIndex:
        path = os.environ.get("RESEARCH_MCP_INDEX_PATH")
        if not path:
            raise RuntimeError("RESEARCH_MCP_INDEX_PATH is not set")
        return cls(path, dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

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
        existing_rows = self._db.execute(
            f"SELECT faiss_id, paper_id FROM papers WHERE paper_id IN ({_placeholders(len(papers))})",
            [p.id for p in papers],
        ).fetchall()
        existing = {row["paper_id"]: row["faiss_id"] for row in existing_rows}
        if existing:
            self._index.remove_ids(np.array(list(existing.values()), dtype=np.int64))

        max_row = self._db.execute("SELECT MAX(faiss_id) FROM papers").fetchone()[0] or 0
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

        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO papers (faiss_id, paper_id, body) VALUES (?, ?, ?)",
                [
                    (fid, p.id, json.dumps(paper_to_dict(p)))
                    for fid, p in zip(new_ids, papers, strict=True)
                ],
            )
        self._persist_vectors()

    async def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
    ) -> Sequence[tuple[Paper, float]]:
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
        results: list[tuple[Paper, float]] = []
        for fid, score in zip(ids[0], scores[0], strict=True):
            if fid == -1:
                continue
            row = self._db.execute(
                "SELECT body FROM papers WHERE faiss_id = ?", (int(fid),)
            ).fetchone()
            if row is None:
                continue
            results.append((paper_from_dict(json.loads(row["body"])), float(score)))
        return results

    async def delete(self, paper_ids: Sequence[str]) -> None:
        if not paper_ids:
            return
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, list(paper_ids))

    def _delete_sync(self, paper_ids: list[str]) -> None:
        rows = self._db.execute(
            f"SELECT faiss_id FROM papers WHERE paper_id IN ({_placeholders(len(paper_ids))})",
            paper_ids,
        ).fetchall()
        faiss_ids = [int(r["faiss_id"]) for r in rows]
        if not faiss_ids:
            return
        self._index.remove_ids(np.asarray(faiss_ids, dtype=np.int64))
        with self._db:
            self._db.execute(
                f"DELETE FROM papers WHERE paper_id IN ({_placeholders(len(paper_ids))})",
                paper_ids,
            )
        self._persist_vectors()

    async def count(self) -> int:
        return int(self._index.ntotal)

    def close(self) -> None:
        self._db.close()


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)
