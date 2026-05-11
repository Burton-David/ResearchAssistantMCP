"""ArxivSource parser unit tests — focused on the Atom-feed → Paper shape.

The live API is exercised by `tests/integration/test_arxiv.py`. These tests
use a captured-and-trimmed feed body so they run offline and assert exact
field extraction.
"""

from __future__ import annotations

import pytest

from research_mcp.sources.arxiv import _parse_feed

pytestmark = pytest.mark.unit


_VASWANI_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are RNNs.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <published>2017-06-12T17:57:34Z</published>
    <link href="http://arxiv.org/abs/1706.03762v5" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf"/>
    <arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""


def test_parse_feed_extracts_canonical_paper() -> None:
    papers = _parse_feed(_VASWANI_FEED)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "1706.03762"
    assert paper.id == "arxiv:1706.03762"
    assert paper.title == "Attention Is All You Need"
    assert paper.venue == "NeurIPS 2017"
    assert paper.doi == "10.48550/arXiv.1706.03762"


def test_parse_feed_surfaces_primary_category_into_metadata() -> None:
    """`arxiv:primary_category[term]` lands in
    `metadata["arxiv_primary_category"]` so the field-aware scorer can detect
    discipline. The category code stays in its original case ("cs.CL"); the
    heuristic scorer lowercases on read."""
    papers = _parse_feed(_VASWANI_FEED)
    assert papers[0].metadata.get("arxiv_primary_category") == "cs.CL"


def test_parse_feed_returns_empty_metadata_when_primary_category_missing() -> None:
    """An entry without `<arxiv:primary_category>` shouldn't carry a
    placeholder — the absence itself is meaningful (the consumer falls back
    to other field signals)."""
    feed_no_cat = _VASWANI_FEED.replace(
        b'<arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>',
        b"",
    )
    papers = _parse_feed(feed_no_cat)
    assert papers[0].metadata == {}


def test_parse_feed_handles_primary_category_without_term_attribute() -> None:
    """Defensive: a malformed element (empty term) shouldn't insert a blank
    key into metadata."""
    feed_blank_term = _VASWANI_FEED.replace(
        b'<arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>',
        b'<arxiv:primary_category term="" scheme="http://arxiv.org/schemas/atom"/>',
    )
    papers = _parse_feed(feed_blank_term)
    assert "arxiv_primary_category" not in papers[0].metadata
