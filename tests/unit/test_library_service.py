"""LibraryService behavior tests."""

from __future__ import annotations

import pytest

from research_mcp.domain.paper import Paper
from research_mcp.embedder import FakeEmbedder
from research_mcp.index import MemoryIndex
from research_mcp.service.library import LibraryService, PaperNotFoundError
from tests.conftest import StaticSource

pytestmark = pytest.mark.unit


async def test_ingest_then_recall(vaswani_paper: Paper, bert_paper: Paper) -> None:
    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [vaswani_paper, bert_paper])
    library = LibraryService(index=index, embedder=embedder, ingest_source=src)
    await library.ingest(vaswani_paper.id)
    await library.ingest(bert_paper.id)
    assert await library.count() == 2
    # FakeEmbedder is hash-based; recalling with the exact same text we ingested
    # gives a perfect-similarity hit. Semantic relevance ranking is verified
    # in the integration tests against the real OpenAI embedder.
    text = f"{vaswani_paper.title}\n\n{vaswani_paper.abstract}"
    [(top, score), *_] = await library.recall(text, k=2)
    assert top.id == vaswani_paper.id
    assert abs(score - 1.0) < 1e-5


async def test_ingest_unknown_id_raises() -> None:
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [])
    library = LibraryService(index=index, embedder=embedder, ingest_source=src)
    with pytest.raises(PaperNotFoundError):
        await library.ingest("arxiv:does-not-exist")


async def test_dimension_mismatch_at_ingest(vaswani_paper: Paper) -> None:
    """Mismatched embedder/index dimensions surface at ingest time.

    The Index protocol doesn't expose dimension, so we can't validate at
    construction. The Index implementations raise a clear error on first
    upsert, which is the next-best place to fail — and still loudly enough
    that the user knows what went wrong.
    """
    embedder = FakeEmbedder(64)
    index = MemoryIndex(32)
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_source=StaticSource("arxiv", [vaswani_paper]),
    )
    with pytest.raises(ValueError):
        await library.ingest(vaswani_paper.id)


async def test_delete_removes_from_index(vaswani_paper: Paper) -> None:
    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [vaswani_paper])
    library = LibraryService(index=index, embedder=embedder, ingest_source=src)
    await library.ingest(vaswani_paper.id)
    await library.delete(vaswani_paper.id)
    assert await library.count() == 0
