"""Core domain types and protocols. Everything else in research_mcp implements these."""

from research_mcp.domain.citation import CitationFormat, CitationRenderer
from research_mcp.domain.embedder import Embedder
from research_mcp.domain.index import Index
from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.reranker import Reranker
from research_mcp.domain.source import Source

__all__ = [
    "Author",
    "CitationFormat",
    "CitationRenderer",
    "Embedder",
    "Index",
    "Paper",
    "Reranker",
    "SearchQuery",
    "Source",
]
