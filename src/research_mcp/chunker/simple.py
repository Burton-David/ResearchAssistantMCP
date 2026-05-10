"""Sliding-window Chunker that ignores section structure.

Use when the source text is structurally unmarked (raw extracted PDF
text, unstructured notes) or when the caller wants uniform chunk sizes
regardless of section boundaries. The default `SectionAwareChunker` is
better for papers with detectable section headers; this one is the
no-frills alternative.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from research_mcp.chunker._text import paper_text, sliding_windows
from research_mcp.domain.chunker import TextChunk
from research_mcp.domain.paper import Paper


class SimpleChunker:
    """Sliding-window `Chunker` with no section detection."""

    name: str = "simple"

    def __init__(
        self,
        *,
        max_chunk_chars: int = 2000,
        overlap_chars: int = 200,
    ) -> None:
        if max_chunk_chars <= overlap_chars:
            raise ValueError("max_chunk_chars must exceed overlap_chars")
        self.max_chunk_chars = max_chunk_chars
        self._overlap = overlap_chars

    async def chunk(self, paper: Paper) -> Sequence[TextChunk]:
        text = paper_text(paper)
        if not text:
            return []
        return [
            TextChunk(
                text=chunk_text,
                chunk_id=f"{paper.id}#chunk#{i}",
                paper_id=paper.id,
                section=None,
                start_char=start,
                end_char=end,
                metadata=MappingProxyType({"chunk_index": str(i)}),
            )
            for i, (chunk_text, start, end) in enumerate(
                sliding_windows(
                    text,
                    chunk_chars=self.max_chunk_chars,
                    overlap_chars=self._overlap,
                )
            )
        ]
