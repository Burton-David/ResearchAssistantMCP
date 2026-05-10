"""Reranker protocol: rescore (query, paper) pairs for relevance.

Sits one layer above bi-encoder retrieval (FAISS / arXiv-and-S2 keyword
search). The contract is small: given a query and a set of candidate
Papers, return one score per paper in input order; higher = more relevant.

The motivating use case is the diagnostic agent's lattice-QCD finding:
arXiv's keyword ranker put physics papers at the top for a chemistry
query because both shared the word 'lattice'. A cross-encoder reranker
scoring the same candidates against the full query string elevates
chemistry papers and demotes off-topic shares.

A reranker is OPTIONAL. SearchService and LibraryService both compose
with `Reranker | None` and behave identically to today when the slot is
empty. When set, they widen the bi-encoder candidate pool, rerank, and
truncate to the user's requested limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_mcp.domain.paper import Paper


@runtime_checkable
class Reranker(Protocol):
    """A relevance reranker.

    Implementations must be safe to call concurrently — services may issue
    overlapping search and recall calls that each invoke `score`. The
    underlying model load (cross-encoder weights, etc.) typically happens
    once on first use; implementations should guard model construction
    behind a lock.
    """

    name: str
    """Human-readable name. Surfaced via library_status so users can verify
    the active reranker without reading logs."""

    async def score(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> Sequence[float]:
        """Return one relevance score per Paper, in input order.

        Higher = more relevant. Score range depends on the implementation:
        cross-encoders typically return raw logits in [-12, +6]; the
        FakeReranker uses Jaccard token overlap in [0, 1]. Callers should
        sort by score (descending), not interpret absolute values.

        `len(returned) == len(papers)` must hold.
        """
        ...
