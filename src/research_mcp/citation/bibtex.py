"""BibTeX entry generator.

Emits `@article` for everything with a venue and `@misc` for arXiv preprints
or papers with no published venue. The cite key is derived from the first
author's surname and the publication year (with the canonical id appended on
collisions, though this renderer doesn't track collisions).
"""

from __future__ import annotations

from research_mcp.citation._format import split_name
from research_mcp.domain.citation import CitationFormat
from research_mcp.domain.paper import Paper

_ESCAPE_MAP = str.maketrans({
    "{": "\\{",
    "}": "\\}",
    "%": "\\%",
    "&": "\\&",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
})


class BibtexRenderer:
    format: CitationFormat = CitationFormat.BIBTEX

    def render(self, paper: Paper) -> str:
        entry_type = "article" if paper.venue else "misc"
        key = _cite_key(paper)
        fields: list[tuple[str, str]] = []
        if paper.title:
            fields.append(("title", paper.title))
        if paper.authors:
            fields.append(("author", _format_authors(paper)))
        if paper.venue:
            fields.append(("journal", paper.venue))
        elif paper.arxiv_id:
            fields.append(("howpublished", "arXiv preprint"))
        if paper.published:
            fields.append(("year", str(paper.published.year)))
        if paper.doi:
            fields.append(("doi", paper.doi))
        if paper.arxiv_id:
            fields.append(("eprint", paper.arxiv_id))
            fields.append(("archivePrefix", "arXiv"))
        if paper.url:
            fields.append(("url", paper.url))
        body = ",\n".join(f"  {k} = {{{_escape(v)}}}" for k, v in fields)
        return f"@{entry_type}{{{key},\n{body}\n}}"


def _cite_key(paper: Paper) -> str:
    surname = ""
    if paper.authors:
        surname, _ = split_name(paper.authors[0].name)
    surname = "".join(c for c in surname.lower() if c.isalnum()) or "anon"
    year = str(paper.published.year) if paper.published else "nd"
    suffix = paper.arxiv_id or paper.id.split(":", 1)[-1]
    suffix = "".join(c for c in suffix if c.isalnum())[:8]
    return f"{surname}{year}{suffix}"


def _format_authors(paper: Paper) -> str:
    return " and ".join(a.name for a in paper.authors)


def _escape(value: str) -> str:
    return value.translate(_ESCAPE_MAP)
