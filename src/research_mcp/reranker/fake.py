"""Deterministic Reranker for tests.

Scores a (query, paper) pair by Jaccard token overlap between the query and
the concatenated `paper.title + paper.abstract`. Lets unit tests verify the
reranker integration shape without pulling 250 MB of cross-encoder weights
on every CI run.

The scoring isn't semantically meaningful — it's a stand-in. Tests that
assert "reranker reorders mixed-domain candidates" pass tokens designed to
match one set of papers and not another, then check the ordering.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from research_mcp.domain.paper import Paper

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class FakeReranker:
    """Pure-Python, no-dep, no-network Reranker for tests.

    Scores by Jaccard overlap between query tokens and paper title+abstract
    tokens. Identical query / paper text produces score 1.0; disjoint
    vocabulary produces 0.0.
    """

    name: str

    def __init__(self, name: str = "fake-reranker") -> None:
        self.name = name

    async def score(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> Sequence[float]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return [0.0] * len(papers)
        return [_jaccard(q_tokens, _tokens(f"{p.title} {p.abstract}")) for p in papers]


def _tokens(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return {t for t in _NON_ALNUM_RE.split(folded.lower()) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
