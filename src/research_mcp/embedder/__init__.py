"""Embedder implementations of the `Embedder` protocol."""

from research_mcp.embedder.fake import FakeEmbedder
from research_mcp.embedder.openai_embedder import OpenAIEmbedder
from research_mcp.embedder.sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
)

__all__ = ["FakeEmbedder", "OpenAIEmbedder", "SentenceTransformersEmbedder"]
