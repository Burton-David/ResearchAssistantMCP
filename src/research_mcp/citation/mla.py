"""MLA (9th-edition) citation format.

    Surname, First Middle, et al. "Title of Article." *Journal*, Year, URL.

For arXiv preprints we use "arXiv" as the container.
"""

from __future__ import annotations

from research_mcp.citation._format import join_with_and, mla_author
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
    formatted = [mla_author(a) for a in paper.authors]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return join_with_and(formatted)
    return f"{formatted[0]}, et al"


def _ensure_period(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    return s if s.endswith(".") else s + "."
