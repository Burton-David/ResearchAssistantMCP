"""Reranker Protocol conformance + FakeReranker behavior."""

from __future__ import annotations

import pytest

from research_mcp.domain import Reranker
from research_mcp.domain.paper import Author, Paper
from research_mcp.reranker import FakeReranker

pytestmark = pytest.mark.unit


def test_protocol_conformance() -> None:
    assert isinstance(FakeReranker(), Reranker)


async def test_empty_papers_short_circuits() -> None:
    r = FakeReranker()
    assert list(await r.score("anything", [])) == []


async def test_score_length_matches_input() -> None:
    r = FakeReranker()
    papers = [
        Paper(id=f"x:{i}", title=f"t{i}", abstract="", authors=())
        for i in range(5)
    ]
    scores = await r.score("anything", papers)
    assert len(scores) == 5


async def test_token_overlap_ranks_relevant_above_irrelevant() -> None:
    """The fixture exists so SearchService / LibraryService tests can rely on
    a deterministic 'reranker reorders' assertion. Lock the basic claim."""
    r = FakeReranker()
    on_topic = Paper(
        id="x:1",
        title="Single-atom platinum CeO2 catalysis",
        abstract="cerium oxide redox cycling supports metallic active sites",
        authors=(Author("Datye"),),
    )
    off_topic = Paper(
        id="x:2",
        title="Lattice QCD on the Fermilab grid",
        abstract="heavy quark masses computed via lattice methods",
        authors=(Author("Quigg"),),
    )
    scores = await r.score("platinum cerium catalysis chemistry", [off_topic, on_topic])
    assert scores[1] > scores[0]


async def test_query_with_no_tokens_scores_zero() -> None:
    r = FakeReranker()
    papers = [Paper(id="x:1", title="t", abstract="a", authors=())]
    assert list(await r.score("", papers)) == [0.0]
