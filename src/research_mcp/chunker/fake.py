"""Deterministic Chunker for tests.

Emits one chunk per Paper, containing `paper_text(paper)` verbatim and
labeled `"abstract"`. Useful when you need a `Chunker` to plug into a
service test but don't actually want to verify chunking logic — that's
what the section-aware tests are for.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from research_mcp.chunker._text import paper_text
from research_mcp.domain.chunker import TextChunk
from research_mcp.domain.paper import Paper


class FakeChunker:
    """Returns one chunk per paper, labeled 'abstract'."""

    name: str = "fake"
    max_chunk_chars: int = 100_000

    async def chunk(self, paper: Paper) -> Sequence[TextChunk]:
        text = paper_text(paper)
        if not text:
            return []
        return [
            TextChunk(
                text=text,
                chunk_id=f"{paper.id}#fake#0",
                paper_id=paper.id,
                section="abstract",
                start_char=0,
                end_char=len(text),
                metadata=MappingProxyType({}),
            )
        ]
