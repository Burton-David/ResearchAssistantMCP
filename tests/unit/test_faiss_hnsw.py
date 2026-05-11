"""HNSW-mode FaissIndex tests.

Three coverage points the design (#7) committed to:
  1. Round-trip — build HNSW, save, reload from a new instance, search the
     same query and recover identical top-10 results. Locks in the
     persistence path (vectors.faiss + index.meta.json sidecar).
  2. Recall — on 10K synthetic vectors, HNSW recall@10 vs flat baseline is
     at least 0.95 across 20 random queries. Catches a regression where
     someone tweaks the HNSW params below the recall floor we promised.
  3. Migration safety — an existing flat library without a sidecar +
     RESEARCH_MCP_FAISS_INDEX_TYPE=hnsw should NOT silently rebuild as
     HNSW. We log a warning, preserve the flat data, and write a sidecar
     pinning the type as flat on the next save.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from research_mcp.domain.paper import Paper
from research_mcp.index import FaissIndex

pytestmark = pytest.mark.unit


def _make_papers(n: int) -> list[Paper]:
    return [
        Paper(id=f"x:{i}", title=f"t{i}", abstract="a", authors=())
        for i in range(n)
    ]


async def test_hnsw_roundtrip_identical_top10(tmp_path: Path) -> None:
    """HNSW indexes survive close+reopen with bit-identical search results."""
    rng = np.random.default_rng(0)
    n, d = 100, 64
    vectors = rng.standard_normal((n, d)).astype(np.float32)
    papers = _make_papers(n)

    idx = FaissIndex(tmp_path, dimension=d, index_type="hnsw")
    try:
        assert idx.index_type == "hnsw"
        await idx.upsert(papers, vectors.tolist())

        # Sidecar must be written on save.
        sidecar = json.loads((tmp_path / "index.meta.json").read_text())
        assert sidecar == {"type": "hnsw", "dimension": 64, "version": 1}

        query = vectors[0].tolist()
        before = [(p.id, score) for p, score in await idx.search(query, k=10)]
    finally:
        idx.close()

    idx2 = FaissIndex(tmp_path, dimension=d, index_type="hnsw")
    try:
        assert idx2.index_type == "hnsw"
        after = [(p.id, score) for p, score in await idx2.search(query, k=10)]
    finally:
        idx2.close()

    # Top-10 ids must match exactly. HNSW is deterministic for fixed
    # params + identical insert order; scores may have tiny float drift
    # across the C++ write/read boundary so we compare ids only and
    # leave the score check loose.
    assert [pid for pid, _ in before] == [pid for pid, _ in after]
    for (_, s1), (_, s2) in zip(before, after, strict=True):
        assert abs(s1 - s2) < 1e-5


async def test_hnsw_recall_at_least_0_95_vs_flat(tmp_path: Path) -> None:
    """HNSW must recover ≥95% of flat's top-10 across 20 queries on 10K vectors.

    Synthetic uniform-Gaussian vectors are an easy regime for HNSW; the
    real bar this guards against is someone lowering efSearch or M below
    the point where the index loses too much recall to be worth the
    search-speed win.
    """
    rng = np.random.default_rng(42)
    n, d = 10_000, 64
    vectors = rng.standard_normal((n, d)).astype(np.float32)
    papers = _make_papers(n)

    flat_idx = FaissIndex(tmp_path / "flat", dimension=d, index_type="flat")
    hnsw_idx = FaissIndex(tmp_path / "hnsw", dimension=d, index_type="hnsw")
    try:
        embeddings = vectors.tolist()
        await flat_idx.upsert(papers, embeddings)
        await hnsw_idx.upsert(papers, embeddings)

        query_indices = rng.choice(n, size=20, replace=False)
        recalls: list[float] = []
        for q_idx in query_indices:
            query = vectors[int(q_idx)].tolist()
            flat_ids = {p.id for p, _ in await flat_idx.search(query, k=10)}
            hnsw_ids = {p.id for p, _ in await hnsw_idx.search(query, k=10)}
            recalls.append(len(flat_ids & hnsw_ids) / 10.0)
        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= 0.95, (
            f"HNSW recall@10 = {mean_recall:.3f} < 0.95 floor "
            f"(per-query recalls: {recalls})"
        )
    finally:
        flat_idx.close()
        hnsw_idx.close()


async def test_migration_existing_flat_without_sidecar_stays_flat(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A pre-sidecar flat library + index_type='hnsw' must preserve flat data.

    Setup: build a flat index, ingest a paper, close. Delete the sidecar
    to simulate a directory created before #7 landed. Re-open with
    index_type='hnsw'. Expect:
      - the index loads as flat (existing data sacred)
      - a warning names what happened
      - on the next save, the sidecar is written pinning type='flat'
    """
    d = 16
    flat_paper = Paper(id="x:1", title="t", abstract="a", authors=())
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(d).astype(np.float32).tolist()

    idx = FaissIndex(tmp_path, dimension=d, index_type="flat")
    try:
        await idx.upsert([flat_paper], [vec])
    finally:
        idx.close()

    # Wipe the sidecar to model a pre-#7 on-disk layout.
    sidecar = tmp_path / "index.meta.json"
    assert sidecar.exists()  # sanity: the flat build wrote one
    sidecar.unlink()

    caplog.set_level(logging.WARNING, logger="research_mcp.index.faiss_index")
    idx2 = FaissIndex(tmp_path, dimension=d, index_type="hnsw")
    try:
        # Effective type is flat — sidecar absence + vectors.faiss present
        # means "trust the legacy on-disk data".
        assert idx2.index_type == "flat"
        # The flat-ingested paper must still be searchable.
        results = await idx2.search(vec, k=1)
        assert len(results) == 1 and results[0][0].id == "x:1"
        # And there must be a clear warning in the log naming the situation.
        warning_text = " ".join(r.getMessage() for r in caplog.records)
        assert "no index.meta.json sidecar" in warning_text
        assert "hnsw" in warning_text

        # Next save writes the sidecar — pinned as flat, not hnsw.
        assert not sidecar.exists()
        other = Paper(id="x:2", title="t2", abstract="a", authors=())
        await idx2.upsert([other], [vec])
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta == {"type": "flat", "dimension": d, "version": 1}
    finally:
        idx2.close()
