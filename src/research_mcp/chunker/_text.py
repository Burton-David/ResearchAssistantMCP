"""Shared text helpers for chunker implementations.

Kept private to the chunker package: derives the canonical "text to chunk"
string from a Paper, and provides a sliding-window splitter both
implementations call.
"""

from __future__ import annotations

from collections.abc import Iterator

from research_mcp.domain.paper import Paper


def paper_text(paper: Paper) -> str:
    """Pick the text to chunk from a Paper.

    Prefer `full_text` if populated (will happen after PDF download lands);
    otherwise concatenate title + abstract. Empty papers (no title, no
    abstract, no full_text) return ""; chunkers must handle that without
    raising.
    """
    if paper.full_text:
        return paper.full_text
    pieces = [p for p in (paper.title, paper.abstract) if p]
    return "\n\n".join(pieces)


def sliding_windows(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
    start_offset: int = 0,
) -> Iterator[tuple[str, int, int]]:
    """Yield (chunk_text, absolute_start, absolute_end) tuples.

    `chunk_chars` is the soft target; chunks may exceed it slightly when a
    sentence ends past the boundary. `overlap_chars` ensures cross-chunk
    context — the first `overlap_chars` of each non-first chunk repeats
    the tail of the previous one. `start_offset` lets callers chunk a
    sub-region of a larger document while preserving absolute character
    offsets in the returned tuples.

    Empty input yields nothing.
    """
    if not text:
        return
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be in [0, chunk_chars)")

    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + chunk_chars, n)
        # Try to break at the next sentence end after `end - 0.1*chunk_chars`
        # so we don't bisect a sentence. Falls back to the hard end if no
        # sentence terminator is reachable.
        if end < n:
            search_from = max(end - chunk_chars // 10, pos + 1)
            window = text[search_from:end + chunk_chars // 10]
            best = -1
            for i, ch in enumerate(window):
                if ch in {".", "!", "?", "\n"} and i + search_from > pos:
                    best = search_from + i + 1
            if best != -1 and best <= n:
                end = best
        yield text[pos:end], pos + start_offset, end + start_offset
        if end >= n:
            return
        pos = max(end - overlap_chars, pos + 1)
