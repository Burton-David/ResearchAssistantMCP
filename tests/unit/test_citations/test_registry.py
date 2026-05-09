"""Citation registry — ensures every CitationFormat has a registered renderer."""

from __future__ import annotations

import pytest

from research_mcp.citation import RENDERERS, get_renderer
from research_mcp.domain.citation import CitationFormat, CitationRenderer

pytestmark = pytest.mark.unit


def test_every_format_has_a_renderer() -> None:
    for fmt in CitationFormat:
        assert fmt in RENDERERS


def test_each_renderer_satisfies_protocol() -> None:
    for r in RENDERERS.values():
        assert isinstance(r, CitationRenderer)


def test_get_renderer_round_trip() -> None:
    for fmt in CitationFormat:
        assert get_renderer(fmt).format == fmt
