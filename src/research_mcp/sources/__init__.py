"""Source implementations of the `Source` protocol."""

from research_mcp.sources.arxiv import ArxivSource
from research_mcp.sources.semantic_scholar import SemanticScholarSource

__all__ = ["ArxivSource", "SemanticScholarSource"]
