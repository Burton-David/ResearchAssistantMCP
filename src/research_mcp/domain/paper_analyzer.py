"""PaperAnalyzer protocol: paper → structured analysis.

The "topic understanding" piece of the citation assistant. Given a
Paper (title + abstract + optional full_text), an analyzer extracts:

  * A summary
  * Key contributions
  * Methodology / technical approach
  * Limitations + future directions
  * Datasets used + metrics reported + baselines compared
  * A confidence score reflecting how much of the structure the
    analyzer was actually able to recover

Implementations differ in how they extract:

  * `OpenAILLMPaperAnalyzer` — structured JSON-output prompts to a
    GPT-class model. Default; expensive per call; high-quality.
  * `FakePaperAnalyzer` — for tests; returns canned structure.
  * Future: an Anthropic implementation, a local-LLM implementation,
    a structured-extraction-without-LLM heuristic for when budget is
    a concern.

The `kinds` parameter on `analyze` lets callers ask for only the
sections they want (saves tokens). Default is everything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from research_mcp.domain.paper import Paper


class AnalysisKind(StrEnum):
    """Which aspect of a paper to extract.

    Mostly a budgeting knob: each kind costs ~1 LLM call (in the
    default implementation). Callers that only want methodology can
    skip the rest.
    """

    SUMMARY = "summary"
    CONTRIBUTIONS = "contributions"
    METHODOLOGY = "methodology"
    LIMITATIONS = "limitations"
    FUTURE_WORK = "future_work"
    DATASETS = "datasets"
    METRICS = "metrics"
    BASELINES = "baselines"


# Default: extract everything. Implementations should accept an empty
# `kinds` argument as "give me everything you can."
ALL_ANALYSIS_KINDS: tuple[AnalysisKind, ...] = tuple(AnalysisKind)


@dataclass(frozen=True, slots=True)
class PaperAnalysis:
    """Structured analysis of a paper.

    Every field is optional — extraction can fail per-field without
    failing the whole analysis. `confidence` reports how much of the
    requested structure the analyzer was actually able to recover
    (1.0 = everything cleanly extracted; 0.0 = analyzer essentially
    gave up).
    """

    paper_id: str
    summary: str | None = None
    key_contributions: tuple[str, ...] = ()
    methodology: str | None = None
    technical_approach: str | None = None
    limitations: tuple[str, ...] = ()
    future_directions: tuple[str, ...] = ()
    datasets_used: tuple[str, ...] = ()
    metrics_reported: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    baselines_compared: tuple[str, ...] = ()
    confidence: float = 0.0
    model: str = ""
    """Model identifier used for the analysis ('openai:gpt-4o-mini',
    'anthropic:claude-...', 'fake:stub'). Surfaced so callers can
    decide whether to trust the result."""


@runtime_checkable
class PaperAnalyzer(Protocol):
    """A paper analyzer.

    Implementations must be safe to call concurrently. The default
    implementation makes LLM calls; concurrent usage may benefit from
    `asyncio.gather` or batching at the analyzer level.
    """

    name: str

    async def analyze(
        self,
        paper: Paper,
        kinds: Sequence[AnalysisKind] = (),
    ) -> PaperAnalysis:
        """Extract structured analysis of `paper`.

        `kinds` selects which aspects to extract; the empty default
        means "everything." Empty input (paper with no abstract and no
        full_text) returns a `PaperAnalysis` with `confidence=0.0` and
        every field blank — never raise on under-determined input.
        """
        ...
