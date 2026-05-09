"""AMA (American Medical Association) citation format.

AMA 11th-ed reference for journal articles:

    Author AA, Author BB, Author CC. Title of article. *Journal Name*. Year;Volume(Issue):pages. doi:10.xxxx/yyyy

For arXiv preprints we don't have volume/issue/pages, so the form collapses to:

    Author AA, Author BB. Title. *arXiv*. Year. arXiv:1234.5678. URL

Authors after the sixth are abbreviated to "et al" per AMA convention.
"""

from __future__ import annotations

from research_mcp.citation._format import ama_author
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper

_MAX_AUTHORS_BEFORE_ETAL = 6


class AMARenderer:
    format: CitationFormat = CitationFormat.AMA

    def render(self, paper: Paper) -> str:
        parts: list[str] = []
        author_part = _render_authors(paper)
        if author_part:
            parts.append(author_part + ".")
        if paper.title:
            parts.append(_ensure_period(paper.title))
        venue_part = _render_venue(paper)
        if venue_part:
            parts.append(venue_part + ".")
        date_part = _render_year(paper)
        if date_part:
            parts.append(date_part + ".")
        if paper.doi:
            parts.append(f"doi:{paper.doi}")
        elif paper.arxiv_id:
            parts.append(f"arXiv:{paper.arxiv_id}.")
        if paper.url and not paper.doi:
            parts.append(paper.url)
        return " ".join(parts).strip()


def _render_authors(paper: Paper) -> str:
    if not paper.authors:
        return ""
    formatted = [ama_author(a) for a in paper.authors]
    if len(formatted) > _MAX_AUTHORS_BEFORE_ETAL:
        formatted = formatted[:_MAX_AUTHORS_BEFORE_ETAL] + ["et al"]
    return ", ".join(formatted)


def _render_venue(paper: Paper) -> str:
    if paper.venue:
        return f"*{paper.venue}*"
    if paper.arxiv_id:
        return "*arXiv*"
    return ""


def _render_year(paper: Paper) -> str:
    if paper.published is None:
        return ""
    return str(paper.published.year)


def _ensure_period(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    return s if s.endswith(".") else s + "."
