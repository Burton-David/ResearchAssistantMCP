"""APA (7th-edition) citation format.

    Author, A. A., Author, B. B., & Author, C. C. (Year). Title of article. *Journal*. https://doi.org/...

For arXiv preprints we substitute "arXiv" as venue and append the arXiv id:

    Author, A. A., & Author, B. B. (Year). Title. *arXiv*. https://arxiv.org/abs/...
"""

from __future__ import annotations

from research_mcp.citation._format import apa_author
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper


class APARenderer:
    format: CitationFormat = CitationFormat.APA

    def render(self, paper: Paper) -> str:
        parts: list[str] = []
        author_part = _render_authors(paper)
        if author_part:
            parts.append(author_part)
        year = str(paper.published.year) if paper.published else "n.d"
        parts.append(f"({year}).")
        if paper.title:
            parts.append(_ensure_period(paper.title))
        venue = paper.venue or ("arXiv" if paper.arxiv_id else None)
        if venue:
            parts.append(f"*{venue}*.")
        if paper.doi:
            parts.append(f"https://doi.org/{paper.doi}")
        elif paper.url:
            parts.append(paper.url)
        return " ".join(parts).strip()


def _render_authors(paper: Paper) -> str:
    if not paper.authors:
        return ""
    formatted = [apa_author(a) for a in paper.authors]
    if len(formatted) == 1:
        body = formatted[0]
    elif len(formatted) == 2:
        body = f"{formatted[0]}, & {formatted[1]}"
    elif len(formatted) <= 20:
        body = ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    else:
        body = ", ".join(formatted[:19]) + ", ... " + formatted[-1]
    return body if body.endswith(".") else body + "."


def _ensure_period(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    return s if s.endswith(".") else s + "."
