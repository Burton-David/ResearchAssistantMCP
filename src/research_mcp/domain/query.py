"""Search queries — the input to any Source.search() call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """A search query.

    `text` is the freeform query string. The other fields narrow the result set;
    individual sources may not honor all of them (arXiv supports year filters
    via API; Semantic Scholar's range syntax differs).
    """

    text: str
    max_results: int = 20
    year_min: int | None = None
    year_max: int | None = None
    authors: tuple[str, ...] = ()
