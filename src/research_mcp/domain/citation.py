"""Citation rendering: turn a Paper into a formatted citation string."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from research_mcp.domain.paper import Paper


class CitationFormat(StrEnum):
    """Supported citation output formats."""

    AMA = "ama"
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    BIBTEX = "bibtex"


@runtime_checkable
class CitationRenderer(Protocol):
    """Render a Paper as a citation in a specific format.

    Renderers should never raise on incomplete papers — degrade gracefully
    (omit missing fields, or substitute "[author unknown]" / "n.d.") rather
    than crash. The point of citations is that they always render something.
    """

    format: CitationFormat

    def render(self, paper: Paper) -> str:
        """Return the formatted citation string."""
        ...
