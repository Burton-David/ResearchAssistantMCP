"""Chicago (17th-edition, author-date) citation format.

    Surname, First Middle, and First Middle Surname. Year. "Title." *Journal* Volume (Issue): pages. https://doi.org/...

For preprints, the volume/issue block is omitted and arXiv stands in for the journal.
"""

from __future__ import annotations

from research_mcp.citation._format import (
    chicago_author_inverted,
    chicago_author_normal,
    join_with_and,
)
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper


class ChicagoRenderer:
    format: CitationFormat = CitationFormat.CHICAGO

    def render(self, paper: Paper) -> str:
        parts: list[str] = []
        author_part = _render_authors(paper)
        if author_part:
            parts.append(author_part + ".")
        year = str(paper.published.year) if paper.published else "n.d"
        parts.append(f"{year}.")
        if paper.title:
            parts.append(f'"{_ensure_period(paper.title)}"')
        venue = paper.venue or ("arXiv" if paper.arxiv_id else None)
        if venue:
            parts.append(f"*{venue}*.")
        if paper.doi:
            parts.append(f"https://doi.org/{paper.doi}.")
        elif paper.url:
            parts.append(f"{paper.url}.")
        return " ".join(parts).strip()


def _render_authors(paper: Paper) -> str:
    if not paper.authors:
        return ""
    head = chicago_author_inverted(paper.authors[0])
    tail = [chicago_author_normal(a) for a in paper.authors[1:]]
    return join_with_and([head, *tail])


def _ensure_period(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    return s if s.endswith(".") else s + "."
