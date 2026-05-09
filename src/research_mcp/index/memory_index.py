"""In-memory vector index. Uses numpy for cosine similarity.

Used by the test suite and the REPL default. No persistence, no FAISS dependency.
Embeddings are L2-normalized on insert so search is a single matrix-vector dot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from research_mcp.domain.paper import Paper


class MemoryIndex:
    """A purely in-memory `Index`.

    Storage layout: a parallel list of paper ids, a list of `Paper` objects,
    and a 2D `(n, dim)` float32 numpy array of L2-normalized embeddings. Lookup
    is by id (O(n) scan) since we don't expect huge in-memory libraries.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._ids: list[str] = []
        self._papers: list[Paper] = []
        self._matrix: npt.NDArray[np.float32] = np.zeros((0, dimension), dtype=np.float32)
        self._lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

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
            for paper, emb in zip(papers, embeddings, strict=True):
                vec = _normalize(np.asarray(emb, dtype=np.float32))
                if vec.shape != (self._dimension,):
                    raise ValueError(
                        f"embedding dim {vec.shape[0]} != index dim {self._dimension}"
                    )
                if paper.id in self._ids:
                    row = self._ids.index(paper.id)
                    self._papers[row] = paper
                    self._matrix[row] = vec
                else:
                    self._ids.append(paper.id)
                    self._papers.append(paper)
                    self._matrix = np.vstack([self._matrix, vec[np.newaxis, :]])

    async def search(
        self,
        embedding: Sequence[float],
        k: int = 10,
    ) -> Sequence[tuple[Paper, float]]:
        if not self._papers:
            return []
        query = _normalize(np.asarray(embedding, dtype=np.float32))
        if query.shape != (self._dimension,):
            raise ValueError(
                f"query dim {query.shape[0]} != index dim {self._dimension}"
            )
        scores = self._matrix @ query
        k = min(k, len(self._papers))
        top = np.argpartition(-scores, k - 1)[:k]
        top_sorted = top[np.argsort(-scores[top])]
        return [(self._papers[i], float(scores[i])) for i in top_sorted]

    async def delete(self, paper_ids: Sequence[str]) -> None:
        if not paper_ids:
            return
        async with self._lock:
            keep = [i for i, pid in enumerate(self._ids) if pid not in set(paper_ids)]
            self._ids = [self._ids[i] for i in keep]
            self._papers = [self._papers[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros(
                (0, self._dimension), dtype=np.float32
            )

    async def count(self) -> int:
        return len(self._papers)


def _normalize(vec: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm
