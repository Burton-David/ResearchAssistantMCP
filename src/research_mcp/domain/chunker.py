"""Chunker protocol: paper → list of section-aware text chunks.

Cross-encoders, embedders, and LLM analyzers all have input length
limits. A 30-page PDF must be chunked before either embedding or
analysis. Different chunking strategies suit different goals:

  * Section-aware (`abstract` / `introduction` / `methodology` / …)
    is best when downstream consumers care about provenance — a
    citation should ideally cite a methodology paragraph rather than
    a stray sentence from the conclusion.
  * Sliding-window is fine when all that matters is keeping each
    chunk under a token cap.

The Chunker abstraction lets either kind plug in without touching the
embedder, the analyzer, or the citation finder.

Today the Chunker only consumes `Paper.title`, `Paper.abstract`, and
`Paper.full_text`. Downstream PDF download (out of scope for this
plan) will populate `full_text`, and chunkers gain useful work to do.
For now they yield 1-3 chunks per paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from research_mcp.domain.paper import Paper


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of a paper's text suitable for embedding or analysis.

    `chunk_id` is unique within the paper; combining `paper_id` and
    `chunk_id` produces a globally unique key. `section` is the named
    section the chunk came from (`"abstract"`, `"methodology"`, etc.)
    or None for sliding-window chunks that don't track structure.

    `start_char` / `end_char` are offsets into whatever the chunker was
    given (typically `paper.full_text` if present, else
    `paper.title + paper.abstract`).
    """

    text: str
    chunk_id: str
    paper_id: str
    section: str | None = None
    start_char: int = 0
    end_char: int = 0
    metadata: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@runtime_checkable
class Chunker(Protocol):
    """A chunker.

    Implementations must be safe to call concurrently. Chunking is pure
    CPU and typically inexpensive, so callers may invoke it inline or
    via `asyncio.to_thread` depending on document size.
    """

    name: str
    max_chunk_chars: int
    """Soft upper bound on `chunk.text` length. Implementations may
    exceed it if a single sentence runs longer; the embedder /
    analyzer downstream will truncate at the model boundary anyway."""

    async def chunk(self, paper: Paper) -> Sequence[TextChunk]:
        """Yield text chunks for `paper`.

        At minimum returns one chunk for `title + abstract`. If
        `paper.full_text` is populated, returns multiple chunks
        respecting `max_chunk_chars`.

        Empty input → empty result; callers shouldn't have to special-
        case a paper with no abstract.
        """
        ...
