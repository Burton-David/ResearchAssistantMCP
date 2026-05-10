"""Core domain types and protocols. Everything else in research_mcp implements these.

Nine protocols total. Five describe the data plane — where papers come
from (`Source`), where they're indexed (`Index`), how they're embedded
(`Embedder`), how they're rendered as citations (`CitationRenderer`),
and how candidate results are reranked (`Reranker`). Four describe the
citation-assistant intelligence — extracting claims from drafts
(`ClaimExtractor`), chunking papers for embedding/analysis (`Chunker`),
scoring citation quality (`CitationScorer`), and analyzing papers into
structured form (`PaperAnalyzer`).

The four citation-assistant protocols are additive on top of the data
plane; they compose Sources/Indexes/Embedders without replacing them.
See ADR-0002 for the case for each.
"""

from research_mcp.domain.chunker import Chunker, TextChunk
from research_mcp.domain.citation import CitationFormat, CitationRenderer
from research_mcp.domain.citation_scorer import CitationQualityScore, CitationScorer
from research_mcp.domain.claim import Claim, ClaimExtractor, ClaimType
from research_mcp.domain.embedder import Embedder
from research_mcp.domain.index import Index
from research_mcp.domain.paper import Author, Paper
from research_mcp.domain.paper_analyzer import (
    ALL_ANALYSIS_KINDS,
    AnalysisKind,
    PaperAnalysis,
    PaperAnalyzer,
)
from research_mcp.domain.query import SearchQuery
from research_mcp.domain.reranker import Reranker
from research_mcp.domain.source import Source

__all__ = [
    "ALL_ANALYSIS_KINDS",
    "AnalysisKind",
    "Author",
    "Chunker",
    "CitationFormat",
    "CitationQualityScore",
    "CitationRenderer",
    "CitationScorer",
    "Claim",
    "ClaimExtractor",
    "ClaimType",
    "Embedder",
    "Index",
    "Paper",
    "PaperAnalysis",
    "PaperAnalyzer",
    "Reranker",
    "SearchQuery",
    "Source",
    "TextChunk",
]
