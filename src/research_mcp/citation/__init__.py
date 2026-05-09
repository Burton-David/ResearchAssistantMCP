"""Citation renderer implementations."""

from research_mcp.citation.ama import AMARenderer
from research_mcp.citation.apa import APARenderer
from research_mcp.citation.bibtex import BibtexRenderer
from research_mcp.citation.chicago import ChicagoRenderer
from research_mcp.citation.mla import MLARenderer
from research_mcp.citation.registry import RENDERERS, get_renderer

__all__ = [
    "RENDERERS",
    "AMARenderer",
    "APARenderer",
    "BibtexRenderer",
    "ChicagoRenderer",
    "MLARenderer",
    "get_renderer",
]
