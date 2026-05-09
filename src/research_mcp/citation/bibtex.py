"""BibTeX entry generator.

Emits `@article` for everything with a venue and `@misc` for arXiv preprints
or papers with no published venue. The cite key follows the conventional
`<surname><year><titleword>` pattern — e.g. `vaswani2017attention` — picking
the first non-stopword from the title so the key is meaningful in a real
`.bib` file.
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

# Conservative stopword list so titles like "Attention Is All You Need"
# resolve to "attention" rather than "is". Not exhaustive — biased toward the
# words that show up at the start of academic paper titles.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "nor", "so", "yet",
    "this", "that", "these", "those",
    "all", "any", "some", "you", "we", "i", "it", "they",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "no", "not", "more", "less", "very",
    "need", "needs", "needed",
    "toward", "towards", "via", "into", "over", "under",
    "case", "study", "studies", "approach", "approaches",
})


def _title_keyword(title: str) -> str:
    for word in title.split():
        cleaned = "".join(c for c in word if c.isalnum()).lower()
        if cleaned and cleaned not in _STOPWORDS:
            return cleaned[:16]
    return ""


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
    keyword = _title_keyword(paper.title)
    if keyword:
        return f"{surname}{year}{keyword}"
    fallback = paper.arxiv_id or paper.id.split(":", 1)[-1]
    fallback = "".join(c for c in fallback if c.isalnum())[:8]
    return f"{surname}{year}{fallback}"


def _format_authors(paper: Paper) -> str:
    return " and ".join(a.name for a in paper.authors)


def _escape(value: str) -> str:
    return value.translate(_ESCAPE_MAP)
