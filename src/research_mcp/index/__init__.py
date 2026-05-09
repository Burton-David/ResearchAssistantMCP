"""Index implementations of the `Index` protocol."""

from research_mcp.index.faiss_index import FaissIndex
from research_mcp.index.memory_index import MemoryIndex

__all__ = ["FaissIndex", "MemoryIndex"]
