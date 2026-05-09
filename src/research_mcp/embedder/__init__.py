"""Embedder implementations of the `Embedder` protocol."""

from research_mcp.embedder.fake import FakeEmbedder
from research_mcp.embedder.openai_embedder import OpenAIEmbedder

__all__ = ["FakeEmbedder", "OpenAIEmbedder"]
