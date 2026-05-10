"""AnalysisService — chunk a paper, analyze it, return structured insights.

Composes a `Chunker` (used when the paper has full_text and the
analyzer benefits from per-section breakdown) and a `PaperAnalyzer`
(does the structured extraction). For Batch 7 the chunker side is
loosely wired; the LLM analyzers handle the full paper in one call,
and per-section chunking would only help if we also did
per-section analysis. That's a future enhancement — for now the
service is a thin pass-through.
"""

from __future__ import annotations

from collections.abc import Sequence

from research_mcp.domain.chunker import Chunker
from research_mcp.domain.paper import Paper
from research_mcp.domain.paper_analyzer import (
    AnalysisKind,
    PaperAnalysis,
    PaperAnalyzer,
)


class AnalysisService:
    def __init__(
        self,
        *,
        analyzer: PaperAnalyzer,
        chunker: Chunker | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._chunker = chunker  # currently unused; reserved for per-section analysis

    @property
    def analyzer(self) -> PaperAnalyzer:
        return self._analyzer

    async def analyze(
        self,
        paper: Paper,
        kinds: Sequence[AnalysisKind] = (),
    ) -> PaperAnalysis:
        return await self._analyzer.analyze(paper, kinds)
