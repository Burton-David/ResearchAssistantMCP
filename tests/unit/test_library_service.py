"""LibraryService behavior tests."""

from __future__ import annotations

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.embedder import FakeEmbedder
from research_mcp.errors import SourceUnavailable
from research_mcp.index import MemoryIndex
from research_mcp.service.library import LibraryService, PaperNotFoundError
from tests.conftest import StaticSource, UnavailableSource

pytestmark = pytest.mark.unit


async def test_ingest_then_recall(vaswani_paper: Paper, bert_paper: Paper) -> None:
    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [vaswani_paper, bert_paper])
    library = LibraryService(index=index, embedder=embedder, ingest_sources=[src])
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


async def test_ingest_unknown_id_raises_with_helpful_message() -> None:
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    src_a = StaticSource("arxiv", [])
    src_b = StaticSource("s2", [])
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[src_a, src_b]
    )
    with pytest.raises(PaperNotFoundError) as info:
        await library.ingest("arxiv:does-not-exist")
    msg = str(info.value)
    # The id and the source names that were tried both surface in the error.
    assert "arxiv:does-not-exist" in msg
    assert "arxiv" in msg
    assert "s2" in msg


async def test_ingest_routes_to_first_matching_source() -> None:
    """A multi-source library tries each Source in turn and ingests from
    whichever resolves the id first."""
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    paper = Paper(
        id="doi:10.1038/s41586-021-03819-2",
        title="Highly accurate protein structure prediction with AlphaFold",
        abstract="...",
        authors=(Author("John Jumper"),),
    )
    arxiv_only = StaticSource("arxiv", [])  # rejects the doi: id
    s2_with_paper = StaticSource("s2", [paper])
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[arxiv_only, s2_with_paper]
    )
    ingested = await library.ingest(paper.id)
    assert ingested.id == paper.id
    assert await library.count() == 1


async def test_fetch_returns_none_when_no_source_resolves() -> None:
    """`fetch` is the soft form of `ingest` — used for cite_paper. None means
    'no source recognized this id', not 'paper does not exist'."""
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [])
    library = LibraryService(index=index, embedder=embedder, ingest_sources=[src])
    assert await library.fetch("arxiv:nope") is None


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
        ingest_sources=[StaticSource("arxiv", [vaswani_paper])],
    )
    with pytest.raises(ValueError):
        await library.ingest(vaswani_paper.id)


async def test_delete_removes_from_index(vaswani_paper: Paper) -> None:
    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [vaswani_paper])
    library = LibraryService(index=index, embedder=embedder, ingest_sources=[src])
    await library.ingest(vaswani_paper.id)
    await library.delete(vaswani_paper.id)
    assert await library.count() == 0


# ---- reranker integration ----


async def test_recall_with_reranker_reorders_top_k() -> None:
    """Without reranker, FakeEmbedder produces near-random scores. With a
    reranker that scores by token overlap with the query, the on-topic
    paper rises to the top regardless of FAISS cosine."""
    from research_mcp.reranker import FakeReranker

    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    on_topic = Paper(
        id="x:on",
        title="cerium oxide platinum catalysis",
        abstract="single-atom Pt redox chemistry",
        authors=(Author("Datye"),),
    )
    off_topic = Paper(
        id="x:off",
        title="lattice quantum chromodynamics",
        abstract="heavy quark masses",
        authors=(Author("Quigg"),),
    )
    src = StaticSource("arxiv", [on_topic, off_topic])
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_sources=[src],
        reranker=FakeReranker(),
    )
    await library.ingest(on_topic.id)
    await library.ingest(off_topic.id)
    results = await library.recall("platinum cerium chemistry", k=2)
    # Reranker pushes the on-topic chemistry paper to the top.
    [(top, _), (second, _)] = results
    assert top.id == "x:on"
    assert second.id == "x:off"


async def test_recall_reranker_failure_falls_back_to_cosine() -> None:
    """A reranker exception during recall must not crash — fall back to
    FAISS cosine ordering, log a warning, return what we have."""

    class _BrokenReranker:
        name = "broken"

        async def score(self, query, papers):  # type: ignore[no-untyped-def]
            raise RuntimeError("model server timed out")

    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    p = Paper(id="x:1", title="t", abstract="a", authors=(Author("X"),))
    src = StaticSource("arxiv", [p])
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_sources=[src],
        reranker=_BrokenReranker(),
    )
    await library.ingest(p.id)
    results = await library.recall("t", k=1)
    # We still get a result back, just from FAISS cosine ordering.
    assert len(results) == 1
    assert results[0][0].id == p.id


async def test_recall_with_reranker_pulls_wider_pool() -> None:
    """When reranker is set, recall fetches `k * 5` candidates from FAISS
    so the cross-encoder has more to choose from. Verify by ingesting many
    papers with similar embeddings and checking that the reranker sees
    more than k candidates before truncation."""
    from research_mcp.reranker import FakeReranker

    captured_k: list[int] = []

    class _CapturingReranker:
        name = "capturing"

        async def score(self, query, papers):  # type: ignore[no-untyped-def]
            captured_k.append(len(papers))
            return await FakeReranker().score(query, papers)

    embedder = FakeEmbedder(64)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource(
        "arxiv",
        [
            Paper(id=f"x:{i}", title=f"paper {i}", abstract="x", authors=())
            for i in range(20)
        ],
    )
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_sources=[src],
        reranker=_CapturingReranker(),
    )
    for i in range(20):
        await library.ingest(f"x:{i}")
    await library.recall("paper", k=2)
    # k=2 with 5x widening = 10 candidates passed to the reranker.
    # Index has 20 so we get 10, not 20.
    assert captured_k[0] == 10


def test_construction_requires_at_least_one_ingest_source() -> None:
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    with pytest.raises(ValueError):
        LibraryService(index=index, embedder=embedder, ingest_sources=[])


async def test_fetch_propagates_source_unavailable_when_no_definite_answer() -> None:
    """If every Source either returned None or was unavailable, and at least
    one was unavailable, we propagate the unavailability — we don't have a
    definitive miss to report."""
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_sources=[
            StaticSource("arxiv", []),  # confirms id not mine
            UnavailableSource("semantic_scholar", "429 rate limited"),
        ],
    )
    with pytest.raises(SourceUnavailable) as exc:
        await library.fetch("arxiv:nope")
    assert exc.value.source_name == "semantic_scholar"
    assert "rate limited" in exc.value.reason


async def test_fetch_returns_paper_even_when_other_source_unavailable() -> None:
    """If one Source resolves the id, an unavailable peer is irrelevant."""
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    paper = Paper(
        id="arxiv:1", title="t", abstract="", authors=(Author("Smith"),), arxiv_id="1"
    )
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_sources=[UnavailableSource("flaky"), StaticSource("arxiv", [paper])],
    )
    found = await library.fetch(paper.id)
    assert found is not None
    assert found.id == paper.id


async def test_fetch_returns_none_when_every_source_says_not_mine() -> None:
    embedder = FakeEmbedder(16)
    index = MemoryIndex(embedder.dimension)
    library = LibraryService(
        index=index,
        embedder=embedder,
        ingest_sources=[StaticSource("a", []), StaticSource("b", [])],
    )
    assert await library.fetch("arxiv:nope") is None


async def test_bulk_ingest_batches_embedder_calls() -> None:
    """Single batched embed call for multiple papers — verifies the hot
    path scales to a 10-paper search-and-ingest without 10 round-trips."""

    class _CountingEmbedder:
        dimension: int = 8

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def embed(self, texts):  # type: ignore[no-untyped-def]
            self.batch_sizes.append(len(texts))
            return [tuple(float(i) for i in range(self.dimension))] * len(texts)

    embedder = _CountingEmbedder()
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [])
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[src]  # type: ignore[arg-type]
    )
    papers = [
        Paper(id=f"arxiv:{i}", title=f"t{i}", abstract="", authors=(Author("X"),))
        for i in range(5)
    ]
    ingested = await library.bulk_ingest(papers)
    assert len(ingested) == 5
    # Exactly one call to the embedder, with all five inputs in it.
    assert embedder.batch_sizes == [5]
    assert await library.count() == 5


async def test_bulk_ingest_deduplicates_by_paper_id() -> None:
    embedder = FakeEmbedder(8)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [])
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[src]
    )
    paper = Paper(id="arxiv:1", title="t", abstract="", authors=(Author("X"),))
    ingested = await library.bulk_ingest([paper, paper, paper])
    assert len(ingested) == 1
    assert await library.count() == 1


async def test_bulk_ingest_empty_input_is_noop() -> None:
    embedder = FakeEmbedder(8)
    index = MemoryIndex(embedder.dimension)
    src = StaticSource("arxiv", [])
    library = LibraryService(
        index=index, embedder=embedder, ingest_sources=[src]
    )
    assert await library.bulk_ingest([]) == ()
    assert await library.count() == 0


async def test_fetch_with_enrichment_merges_metadata_across_sources() -> None:
    """The chaos-test fix: when only arxiv claims `arxiv:1706.03762`, S2
    should still get queried for venue + citation_count and merged into
    the primary record. Without enrichment, score_citation starves on
    missing metadata and Vaswani gets 35/100."""
    from datetime import date as _date

    from research_mcp.service.library import fetch_with_enrichment

    arxiv_record = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="...",
        authors=(Author("Vaswani"),),
        published=_date(2017, 6, 12),
        arxiv_id="1706.03762",
        # Notice: no venue, no citation_count — same as a real arxiv response.
    )
    s2_record = Paper(
        id="s2:abc123",
        title="Attention Is All You Need",
        abstract="...",
        authors=(Author("Vaswani"),),
        published=_date(2017, 6, 12),
        arxiv_id="1706.03762",
        venue="NeurIPS",
        citation_count=70000,
    )

    class _ArxivFake:
        name = "arxiv"
        id_prefixes = ("arxiv",)

        async def search(self, query):  # type: ignore[no-untyped-def]
            return []

        async def fetch(self, paper_id: str) -> Paper | None:
            return arxiv_record if paper_id == "arxiv:1706.03762" else None

    class _S2Fake:
        # Mirrors the production widening: S2 now claims arxiv: too so
        # cross-source enrichment can reach it.
        name = "semantic_scholar"
        id_prefixes = ("s2", "doi", "arxiv")

        async def search(self, query):  # type: ignore[no-untyped-def]
            return []

        async def fetch(self, paper_id: str) -> Paper | None:
            if paper_id in {"arxiv:1706.03762", "s2:abc123"}:
                return s2_record
            return None

    enriched = await fetch_with_enrichment(
        [_ArxivFake(), _S2Fake()], "arxiv:1706.03762"  # type: ignore[list-item]
    )
    assert enriched is not None
    # Enrichment populated the missing fields.
    assert enriched.venue == "NeurIPS"
    assert enriched.citation_count == 70000


async def test_fetch_with_enrichment_returns_primary_when_no_other_source_matches() -> None:
    """If no other source can resolve the paper, return the primary unchanged."""
    from datetime import date as _date

    from research_mcp.service.library import fetch_with_enrichment

    primary = Paper(
        id="arxiv:1706.03762",
        title="x",
        abstract="",
        authors=(Author("A"),),
        published=_date(2017, 1, 1),
        arxiv_id="1706.03762",
    )

    class _ArxivFake:
        name = "arxiv"
        id_prefixes = ("arxiv",)

        async def search(self, query):  # type: ignore[no-untyped-def]
            return []

        async def fetch(self, paper_id: str) -> Paper | None:
            return primary if paper_id == "arxiv:1706.03762" else None

    class _NoMatch:
        name = "nomatch"
        id_prefixes = ("xyz",)

        async def search(self, query):  # type: ignore[no-untyped-def]
            return []

        async def fetch(self, paper_id: str) -> Paper | None:
            return None

    enriched = await fetch_with_enrichment(
        [_ArxivFake(), _NoMatch()], "arxiv:1706.03762"  # type: ignore[list-item]
    )
    assert enriched == primary


async def test_fetch_with_enrichment_returns_none_when_primary_missing() -> None:
    from research_mcp.service.library import fetch_with_enrichment

    class _NoneSource:
        name = "x"
        id_prefixes = ("x",)

        async def search(self, query):  # type: ignore[no-untyped-def]
            return []

        async def fetch(self, paper_id: str) -> Paper | None:
            return None

    assert await fetch_with_enrichment(
        [_NoneSource()], "arxiv:9"  # type: ignore[list-item]
    ) is None
