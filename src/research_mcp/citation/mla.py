"""MLA (9th-edition) citation format.

    Surname, First Middle, et al. "Title of Article." *Journal*, Year, URL.

For arXiv preprints we use "arXiv" as the container.
"""

from __future__ import annotations

from research_mcp.citation._format import (
    join_with_and,
    mla_author_inverted,
    mla_author_normal,
)
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper


class MLARenderer:
    format: CitationFormat = CitationFormat.MLA

    def render(self, paper: Paper) -> str:
        parts: list[str] = []
        author_part = _render_authors(paper)
        if author_part:
            parts.append(author_part + ".")
        if paper.title:
            parts.append(f'"{_ensure_period(paper.title)}"')
        venue = paper.venue or ("arXiv" if paper.arxiv_id else None)
        if venue:
            parts.append(f"*{venue}*,")
        if paper.published:
            parts.append(f"{paper.published.year},")
        if paper.doi:
            parts.append(f"https://doi.org/{paper.doi}.")
        elif paper.url:
            parts.append(f"{paper.url}.")
        return " ".join(parts).strip().rstrip(",")


def _render_authors(paper: Paper) -> str:
    if not paper.authors:
        return ""
    head = mla_author_inverted(paper.authors[0])
    if len(paper.authors) == 1:
        return head
    if len(paper.authors) == 2:
        return join_with_and([head, mla_author_normal(paper.authors[1])])
    return f"{head}, et al"


def _ensure_period(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    return s if s.endswith(".") else s + "."
