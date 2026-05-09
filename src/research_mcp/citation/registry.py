"""Lookup of CitationFormat -> CitationRenderer.

A single dict so `cite_paper` MCP tool dispatch and CLI `--format` flag share
the same wiring. Adding a new format is one line here.
"""

from __future__ import annotations

from research_mcp.citation.ama import AMARenderer
from research_mcp.citation.apa import APARenderer
from research_mcp.citation.bibtex import BibtexRenderer
from research_mcp.citation.chicago import ChicagoRenderer
from research_mcp.citation.mla import MLARenderer
from research_mcp.domain.citation import CitationFormat, CitationRenderer

RENDERERS: dict[CitationFormat, CitationRenderer] = {
    CitationFormat.AMA: AMARenderer(),
    CitationFormat.APA: APARenderer(),
    CitationFormat.MLA: MLARenderer(),
    CitationFormat.CHICAGO: ChicagoRenderer(),
    CitationFormat.BIBTEX: BibtexRenderer(),
}


def get_renderer(fmt: CitationFormat) -> CitationRenderer:
    return RENDERERS[fmt]
