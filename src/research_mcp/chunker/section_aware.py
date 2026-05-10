"""Section-aware Chunker for research papers.

Detects standard paper section headers (`abstract`, `introduction`,
`methodology`, `experiments`, `results`, `discussion`, `conclusion`, …)
via regex and yields one or more chunks per section, each tagged with
the section name.

Falls back to single-chunk output when:
  * The paper only has title + abstract (no full_text yet) — chunks as
    one `"abstract"` section.
  * full_text is populated but no recognizable section headers — chunks
    as one `"body"` section using the sliding-window splitter.

Patterns lifted from the original ResearchAssistantAgent text_chunker.py.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from types import MappingProxyType

from research_mcp.chunker._text import paper_text, sliding_windows
from research_mcp.domain.chunker import TextChunk
from research_mcp.domain.paper import Paper

# Three patterns for section headers:
#   * Markdown-ish: `## Methodology`, `# Experiments`
#   * Numbered:    `3. Methodology`, `3.1 Approach`
#   * Bare:        `Methodology` / `Methodology:` at line start
_SECTION_PATTERNS = [
    r"^#+\s*(abstract|introduction|related work|background|"
    r"methodology|methods|approach|experiments|results|evaluation|"
    r"discussion|conclusion|future work|references|acknowledgments)",
    r"^\d+(?:\.\d+)*\.?\s*(introduction|related work|background|"
    r"methodology|methods|approach|experiments|results|evaluation|"
    r"discussion|conclusion|future work)",
    r"^(abstract|introduction|related work|background|"
    r"methodology|methods|approach|experiments|results|evaluation|"
    r"discussion|conclusion|future work|references|acknowledgments)\s*:?$",
]
_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)


class SectionAwareChunker:
    """A `Chunker` that respects paper section boundaries.

    `chunk_chars` is the soft target chunk size; sections smaller than
    that produce one chunk each, sections larger get split via the
    sliding window with `overlap_chars` of context overlap.
    """

    name: str = "section-aware"

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
        sections = list(_extract_sections(text))
        if not sections:
            # No headers found. If we only have title + abstract, label it
            # "abstract"; otherwise "body" since this is presumably full_text
            # without recognizable section markers.
            label = "abstract" if not paper.full_text else "body"
            return list(_chunk_section(paper.id, label, text, 0, self))
        chunks: list[TextChunk] = []
        for name, body, start in sections:
            chunks.extend(_chunk_section(paper.id, name, body, start, self))
        return chunks


def _extract_sections(text: str) -> list[tuple[str, str, int]]:
    """Walk the text and return (section_name, body, abs_start) tuples."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return []
    out: list[tuple[str, str, int]] = []
    for i, match in enumerate(matches):
        # Pull the actual section name out of whichever group fired.
        name = next((g for g in match.groups() if g), match.group()).strip().lower()
        # Strip punctuation / symbols from header text.
        name = re.sub(r"[^a-z\s]", "", name).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            out.append((name, body, body_start))
    return out


def _chunk_section(
    paper_id: str,
    section: str,
    body: str,
    body_start: int,
    chunker: SectionAwareChunker,
) -> list[TextChunk]:
    if len(body) <= chunker.max_chunk_chars:
        return [
            TextChunk(
                text=body,
                chunk_id=f"{paper_id}#{section}#0",
                paper_id=paper_id,
                section=section,
                start_char=body_start,
                end_char=body_start + len(body),
                metadata=MappingProxyType({"section": section}),
            )
        ]
    chunks: list[TextChunk] = []
    for i, (chunk_text, start, end) in enumerate(
        sliding_windows(
            body,
            chunk_chars=chunker.max_chunk_chars,
            overlap_chars=chunker._overlap,
            start_offset=body_start,
        )
    ):
        chunks.append(
            TextChunk(
                text=chunk_text,
                chunk_id=f"{paper_id}#{section}#{i}",
                paper_id=paper_id,
                section=section,
                start_char=start,
                end_char=end,
                metadata=MappingProxyType(
                    {"section": section, "section_chunk_index": str(i)}
                ),
            )
        )
    return chunks
