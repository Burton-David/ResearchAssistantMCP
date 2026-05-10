"""Deterministic `PaperAnalyzer` for tests."""

from __future__ import annotations

from collections.abc import Sequence

from research_mcp.domain.paper import Paper
from research_mcp.domain.paper_analyzer import (
    AnalysisKind,
    PaperAnalysis,
)
from research_mcp.paper_analyzer._schema import text_for_paper


class FakePaperAnalyzer:
    """Returns a stub PaperAnalysis without making any LLM calls.

    Behavior: confidence=0.5 for any non-empty input; confidence=0.0 for
    blank input. Summary is the first sentence of the input text. Other
    fields are empty.
    """

    name: str = "fake"

    async def analyze(
        self,
        paper: Paper,
        kinds: Sequence[AnalysisKind] = (),
    ) -> PaperAnalysis:
        del kinds  # fake analyzer ignores kind selection
        text = text_for_paper(paper)
        if not text:
            return PaperAnalysis(paper_id=paper.id, model="fake:stub")
        # First sentence (or first 200 chars).
        summary = text.split(".", 1)[0][:200].strip() + "."
        return PaperAnalysis(
            paper_id=paper.id,
            summary=summary,
            confidence=0.5,
            model="fake:stub",
        )
