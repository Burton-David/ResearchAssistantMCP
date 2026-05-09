"""Shared fixtures and lightweight in-test implementations of the four protocols.

Per the project working agreement, we do not mock protocols we own — we write
real, minimal implementations and use them in the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery


class StaticSource:
    """A `Source` that returns a fixed list of `Paper` and supports id lookup."""

    def __init__(self, name: str, papers: Sequence[Paper]) -> None:
        self.name = name
        self._papers = list(papers)

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        # Trivial relevance: papers whose title or abstract contains the query word.
        needle = query.text.lower()
        hits = [
            p
            for p in self._papers
            if needle in p.title.lower() or needle in p.abstract.lower()
        ]
        return hits[: query.max_results]

    async def fetch(self, paper_id: str) -> Paper | None:
        for paper in self._papers:
            if paper.id == paper_id:
                return paper
        return None


class RaisingSource:
    """A `Source` whose `search` raises — used to verify SearchService graceful merge."""

    name: str = "raising"

    async def search(self, query: SearchQuery) -> Sequence[Paper]:
        raise RuntimeError("simulated outage")

    async def fetch(self, paper_id: str) -> Paper | None:
        return None


@pytest.fixture
def vaswani_paper() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract=(
            "The dominant sequence transduction models are based on complex recurrent "
            "or convolutional neural networks..."
        ),
        authors=(
            Author("Ashish Vaswani"),
            Author("Noam Shazeer"),
            Author("Niki Parmar"),
            Author("Jakob Uszkoreit"),
            Author("Llion Jones"),
            Author("Aidan N Gomez"),
            Author("Lukasz Kaiser"),
            Author("Illia Polosukhin"),
        ),
        published=date(2017, 6, 12),
        url="https://arxiv.org/abs/1706.03762",
        venue=None,
        arxiv_id="1706.03762",
    )


@pytest.fixture
def bert_paper() -> Paper:
    return Paper(
        id="arxiv:1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="We introduce a new language representation model called BERT...",
        authors=(
            Author("Jacob Devlin"),
            Author("Ming-Wei Chang"),
            Author("Kenton Lee"),
            Author("Kristina Toutanova"),
        ),
        published=date(2018, 10, 11),
        url="https://arxiv.org/abs/1810.04805",
        arxiv_id="1810.04805",
    )
