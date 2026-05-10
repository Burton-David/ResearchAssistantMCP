"""MCP tool input model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_mcp.mcp.tools import (
    CitePaperInput,
    IngestPaperInput,
    LibrarySearchInput,
    SearchPapersInput,
    paper_to_summary,
)

pytestmark = pytest.mark.unit


def test_search_papers_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": "x", "rogue": 1})


def test_search_papers_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": ""})


def test_search_papers_rejects_whitespace_only_query() -> None:
    """Pure whitespace passes min_length=1 — we want to reject it explicitly."""
    with pytest.raises(ValidationError) as exc:
        SearchPapersInput.model_validate({"query": "   \n\t  "})
    assert "non-whitespace" in str(exc.value)


def test_library_search_rejects_whitespace_only_query() -> None:
    from research_mcp.mcp.tools import LibrarySearchInput

    with pytest.raises(ValidationError):
        LibrarySearchInput.model_validate({"query": "   "})


def test_year_max_below_year_min_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        SearchPapersInput.model_validate(
            {"query": "x", "year_min": 2024, "year_max": 2018}
        )
    assert "year_min" in str(exc.value)


def test_year_equal_min_max_accepted() -> None:
    parsed = SearchPapersInput.model_validate(
        {"query": "x", "year_min": 2020, "year_max": 2020}
    )
    assert parsed.year_min == 2020 and parsed.year_max == 2020


def test_query_capped_at_500_chars() -> None:
    """Long queries hit upstream URL/body length limits; reject up front."""
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": "x" * 501})


def test_find_paper_title_capped_at_500_chars() -> None:
    from research_mcp.mcp.tools import FindPaperInput

    with pytest.raises(ValidationError):
        FindPaperInput.model_validate({"title": "x" * 501})


def test_max_results_capped_at_100() -> None:
    with pytest.raises(ValidationError):
        SearchPapersInput.model_validate({"query": "x", "max_results": 9999})


def test_year_min_accepts_string_int() -> None:
    """Models often serialize numeric tool args as strings — coerce, don't reject.

    The pydantic default lax mode handles this; we disable the mcp SDK's
    jsonschema layer (which doesn't coerce) so this path actually runs.
    """
    parsed = SearchPapersInput.model_validate(
        {"query": "BERT", "year_min": "2018", "year_max": "2020"}
    )
    assert parsed.year_min == 2018
    assert parsed.year_max == 2020


def test_year_min_accepts_real_int() -> None:
    parsed = SearchPapersInput.model_validate({"query": "BERT", "year_min": 2018})
    assert parsed.year_min == 2018


def test_ingest_paper_input_requires_id() -> None:
    with pytest.raises(ValidationError):
        IngestPaperInput.model_validate({})


def test_library_search_default_k() -> None:
    parsed = LibrarySearchInput.model_validate({"query": "x"})
    assert parsed.k == 10


def test_cite_paper_default_format() -> None:
    parsed = CitePaperInput.model_validate({"paper_id": "arxiv:1"})
    assert parsed.format == "ama"


def test_cite_paper_unknown_format_rejected() -> None:
    with pytest.raises(ValidationError):
        CitePaperInput.model_validate({"paper_id": "arxiv:1", "format": "vancouver"})


def test_paper_to_summary_handles_minimal_paper(vaswani_paper) -> None:  # type: ignore[no-untyped-def]
    summary = paper_to_summary(vaswani_paper)
    assert summary.id == vaswani_paper.id
    assert summary.year == 2017
    assert summary.authors[0] == "Ashish Vaswani"
    assert summary.authors_truncated is False
    assert summary.authors_total == 8


def test_source_from_id_routes_via_id_prefixes() -> None:
    """source_from_id must consult Source.id_prefixes, not hard-code adapter
    names. Adding a third Source (PubMed, OpenAlex) should require zero
    changes in the MCP layer."""
    from research_mcp.mcp.tools import source_from_id

    class _FakeSource:
        def __init__(self, name: str, prefixes: tuple[str, ...]) -> None:
            self.name = name
            self.id_prefixes = prefixes

        async def search(self, query):  # type: ignore[no-untyped-def]
            return []

        async def fetch(self, paper_id: str):  # type: ignore[no-untyped-def]
            return None

    arxiv = _FakeSource("arxiv", ("arxiv",))
    s2 = _FakeSource("semantic_scholar", ("s2", "doi"))
    pubmed = _FakeSource("pubmed", ("pmid",))
    sources = [arxiv, s2, pubmed]

    assert source_from_id("arxiv:1706.03762", sources) == "arxiv"
    assert source_from_id("doi:10.1038/X", sources) == "semantic_scholar"
    assert source_from_id("s2:abc123", sources) == "semantic_scholar"
    assert source_from_id("pmid:12345678", sources) == "pubmed"
    # Unknown prefix falls through to the bare prefix string.
    assert source_from_id("isbn:9780123456", sources) == "isbn"


def test_source_id_prefixes_are_correct_on_real_adapters() -> None:
    """Lock down the prefix declarations on the live adapters."""
    from research_mcp.sources import ArxivSource, SemanticScholarSource

    assert ArxivSource.id_prefixes == ("arxiv",)
    assert SemanticScholarSource.id_prefixes == ("s2", "doi")


def test_short_reason_extracts_http_status_code() -> None:
    """SourceUnavailable.short_reason() must collapse httpx's verbose error
    string ('Client error \\'429 \\' for url \\'https://...\\'\\nFor more
    information check https://...') down to a clean 'HTTP 429'."""
    from research_mcp.errors import SourceUnavailable

    raw = (
        "Client error '429 ' for url 'https://api.semanticscholar.org/...'\n"
        "For more information check: https://developer.mozilla.org/...429"
    )
    exc = SourceUnavailable("semantic_scholar", raw)
    assert exc.short_reason() == "HTTP 429"


def test_short_reason_falls_back_for_non_http_errors() -> None:
    from research_mcp.errors import SourceUnavailable

    exc = SourceUnavailable("arxiv", "DNS resolution failed")
    assert exc.short_reason() == "DNS resolution failed"


def test_format_validation_error_strips_pydantic_url() -> None:
    """call_tool wraps pydantic ValidationError so the docs URL doesn't reach
    the LLM client. The user sees a clean 'field: message' line."""
    import pydantic

    from research_mcp.mcp.server import _format_validation_error
    from research_mcp.mcp.tools import SearchPapersInput

    try:
        SearchPapersInput.model_validate({"query": "x" * 600})
    except pydantic.ValidationError as exc:
        message = _format_validation_error("search_papers", exc)
        assert "search_papers" in message
        assert "query" in message
        # The URL must NOT leak into the formatted message.
        assert "errors.pydantic.dev" not in message
        assert "https://" not in message


def test_library_status_output_carries_reranker_field() -> None:
    """Lock down the public shape: library_status surfaces both embedder
    and reranker labels so an LLM client can confirm config without
    grepping logs."""
    from research_mcp.mcp.tools import LibraryStatusOutput

    out = LibraryStatusOutput(
        count=10,
        embedder="openai:text-embedding-3-small",
        reranker="cross-encoder:BAAI/bge-reranker-base",
    )
    payload = out.model_dump()
    assert payload["embedder"] == "openai:text-embedding-3-small"
    assert payload["reranker"] == "cross-encoder:BAAI/bge-reranker-base"


def test_select_reranker_handles_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selector parses RESEARCH_MCP_RERANKER and returns (None, None)
    when unset. We don't actually instantiate a real cross-encoder here —
    that requires a network round-trip; just verify the parser branches."""
    from research_mcp.mcp.server import _select_reranker

    monkeypatch.delenv("RESEARCH_MCP_RERANKER", raising=False)
    assert _select_reranker() == (None, None)

    monkeypatch.setenv("RESEARCH_MCP_RERANKER", "magic:foo")
    with pytest.raises(RuntimeError, match=r"RESEARCH_MCP_RERANKER"):
        _select_reranker()


def test_semantic_scholar_fetch_returns_none_for_arxiv_prefix() -> None:
    """Honor id_prefixes: S2 declares ('s2', 'doi'); fetch must return None
    for an arxiv-prefixed id rather than burning an API call. This was the
    root cause of cite_paper for a typo'd arxiv id surfacing an S2 429."""
    import asyncio

    from research_mcp.sources import SemanticScholarSource

    s2 = SemanticScholarSource()

    async def run() -> None:
        result = await s2.fetch("arxiv:9999.99999")
        assert result is None
        await s2.aclose()

    asyncio.run(run())


def test_paper_to_summary_truncates_huge_author_list() -> None:
    """A 600-author HEP paper should not blow the LLM's context."""
    from research_mcp.domain.paper import Author, Paper

    big = Paper(
        id="x:1",
        title="t",
        abstract="a",
        authors=tuple(Author(f"Author {i}") for i in range(600)),
    )
    summary = paper_to_summary(big)
    assert len(summary.authors) == 20
    assert summary.authors_truncated is True
    assert summary.authors_total == 600


def test_library_status_input_takes_no_args() -> None:
    from research_mcp.mcp.tools import LibraryStatusInput

    parsed = LibraryStatusInput.model_validate({})
    assert parsed is not None


def test_library_status_input_rejects_unknown_field() -> None:
    from research_mcp.mcp.tools import LibraryStatusInput

    with pytest.raises(ValidationError):
        LibraryStatusInput.model_validate({"unexpected": "x"})


def test_get_paper_input_requires_id() -> None:
    from research_mcp.mcp.tools import GetPaperInput

    with pytest.raises(ValidationError):
        GetPaperInput.model_validate({})


def test_get_paper_input_rejects_blank_id() -> None:
    from research_mcp.mcp.tools import GetPaperInput

    with pytest.raises(ValidationError):
        GetPaperInput.model_validate({"paper_id": ""})


def test_extract_claims_input_requires_text() -> None:
    from research_mcp.mcp.tools import ExtractClaimsInput

    with pytest.raises(ValidationError):
        ExtractClaimsInput.model_validate({})


def test_extract_claims_input_rejects_blank_text() -> None:
    from research_mcp.mcp.tools import ExtractClaimsInput

    with pytest.raises(ValidationError):
        ExtractClaimsInput.model_validate({"text": "   "})


def test_extract_claims_input_caps_at_20k_chars() -> None:
    """Bound the spaCy pipeline's worst-case input — past 20K chars the
    user should chunk first instead of pasting a whole paper."""
    from research_mcp.mcp.tools import ExtractClaimsInput

    with pytest.raises(ValidationError):
        ExtractClaimsInput.model_validate({"text": "x" * 20_001})


def test_extract_claims_input_accepts_normal_paragraph() -> None:
    from research_mcp.mcp.tools import ExtractClaimsInput

    parsed = ExtractClaimsInput.model_validate(
        {"text": "Our model outperforms BERT by 12% on three benchmarks."}
    )
    assert "outperforms" in parsed.text


def test_select_claim_extractor_default_is_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env → spaCy extractor (Round 1, ships ~80% precision)."""
    from research_mcp.claim_extractor import SpacyClaimExtractor
    from research_mcp.mcp.server import _select_claim_extractor

    monkeypatch.delenv("RESEARCH_MCP_CLAIM_EXTRACTOR", raising=False)
    extractor = _select_claim_extractor()
    assert isinstance(extractor, SpacyClaimExtractor)


def test_select_claim_extractor_explicit_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_mcp.claim_extractor import SpacyClaimExtractor
    from research_mcp.mcp.server import _select_claim_extractor

    monkeypatch.setenv("RESEARCH_MCP_CLAIM_EXTRACTOR", "spacy")
    assert isinstance(_select_claim_extractor(), SpacyClaimExtractor)


def test_select_claim_extractor_llm_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`llm:openai:<model>` → OpenAILLMClaimExtractor with the right model."""
    from research_mcp.claim_extractor import OpenAILLMClaimExtractor
    from research_mcp.mcp.server import _select_claim_extractor

    monkeypatch.setenv("RESEARCH_MCP_CLAIM_EXTRACTOR", "llm:openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    extractor = _select_claim_extractor()
    assert isinstance(extractor, OpenAILLMClaimExtractor)
    assert extractor.model == "gpt-4o-mini"
    assert extractor.name == "llm:openai:gpt-4o-mini"


def test_select_claim_extractor_llm_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_mcp.claim_extractor import AnthropicLLMClaimExtractor
    from research_mcp.mcp.server import _select_claim_extractor

    monkeypatch.setenv(
        "RESEARCH_MCP_CLAIM_EXTRACTOR", "llm:anthropic:claude-haiku-4-5-20251001"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    extractor = _select_claim_extractor()
    assert isinstance(extractor, AnthropicLLMClaimExtractor)
    assert extractor.model == "claude-haiku-4-5-20251001"


def test_select_claim_extractor_garbled_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable env value should return None — no silent fallback to
    spaCy, since the user explicitly asked for something else."""
    from research_mcp.mcp.server import _select_claim_extractor

    monkeypatch.setenv("RESEARCH_MCP_CLAIM_EXTRACTOR", "frobnicate")
    assert _select_claim_extractor() is None


def test_arxiv_search_string_puts_date_first() -> None:
    """arxiv silently ignores the submittedDate filter when AND'd AFTER
    an `all:` clause. Lock the fixed ordering down so a future refactor
    can't reintroduce the bug."""
    from research_mcp.domain.query import SearchQuery
    from research_mcp.sources.arxiv import _build_search_string

    s = _build_search_string(
        SearchQuery(text="alphafold", max_results=5, year_min=2020, year_max=2022)
    )
    # Date clause must come FIRST.
    assert s.startswith("submittedDate:[")
    assert "all:alphafold" in s
    assert s.index("submittedDate") < s.index("all:")


def test_arxiv_search_string_uses_12_char_date_format() -> None:
    """arxiv's submittedDate accepts exactly YYYYMMDDHHMM (12 chars).
    Catch off-by-one length regressions."""
    from research_mcp.domain.query import SearchQuery
    from research_mcp.sources.arxiv import _build_search_string

    s = _build_search_string(SearchQuery(text="x", year_min=2020, year_max=2022))
    # Inner brackets contain "<lo> TO <hi>" — both must be 12 chars.
    assert "submittedDate:[202001010000 TO 202212312359]" in s


def test_arxiv_search_string_handles_only_year_min() -> None:
    """Year_min set, year_max not — upper bound should be a far-future
    sentinel (year 9999), not None or an empty string."""
    from research_mcp.domain.query import SearchQuery
    from research_mcp.sources.arxiv import _build_search_string

    s = _build_search_string(SearchQuery(text="x", year_min=2020))
    assert "submittedDate:[202001010000 TO 999912312359]" in s


def test_arxiv_search_string_handles_only_year_max() -> None:
    from research_mcp.domain.query import SearchQuery
    from research_mcp.sources.arxiv import _build_search_string

    s = _build_search_string(SearchQuery(text="x", year_max=2022))
    assert "submittedDate:[190001010000 TO 202212312359]" in s


def test_library_status_output_carries_source_list_and_extractor() -> None:
    """Diagnostics tool must report which sources / extractor / analyzer /
    scorer are wired so a user can verify the env config without scraping
    result attribution."""
    from research_mcp.mcp.tools import LibraryStatusOutput

    out = LibraryStatusOutput(
        count=10,
        embedder="openai:text-embedding-3-small",
        reranker="cross-encoder:BAAI/bge-reranker-base",
        sources=["arxiv", "semantic_scholar", "pubmed", "openalex"],
        claim_extractor="llm:openai:gpt-4o-mini",
        paper_analyzer="openai:gpt-4o-mini",
        citation_scorer="heuristic",
    )
    payload = out.model_dump()
    assert payload["sources"] == ["arxiv", "semantic_scholar", "pubmed", "openalex"]
    assert payload["claim_extractor"] == "llm:openai:gpt-4o-mini"
    assert payload["paper_analyzer"] == "openai:gpt-4o-mini"
    assert payload["citation_scorer"] == "heuristic"


def test_openai_extractor_passes_timeout_to_client_when_constructed() -> None:
    """Verify the explicit per-attempt timeout flows through to the SDK.
    SDK default is 600s; we override to 60s to keep tool calls under
    Claude Desktop's 4-min hard kill."""
    from openai import AsyncOpenAI

    from research_mcp.claim_extractor import OpenAILLMClaimExtractor

    e = OpenAILLMClaimExtractor(api_key="sk-test", timeout=42.0)
    assert isinstance(e._client, AsyncOpenAI)
    # The SDK normalizes a float to its `timeout` attribute. Either a
    # raw float or a Timeout object should be acceptable; both must
    # represent 42 seconds.
    timeout_attr = e._client.timeout
    if isinstance(timeout_attr, int | float):
        assert float(timeout_attr) == 42.0
    else:
        assert float(timeout_attr.read) == 42.0


def test_anthropic_extractor_passes_timeout_to_client_when_constructed() -> None:
    from anthropic import AsyncAnthropic

    from research_mcp.claim_extractor import AnthropicLLMClaimExtractor

    e = AnthropicLLMClaimExtractor(api_key="sk-test", timeout=42.0)
    assert isinstance(e._client, AsyncAnthropic)
    timeout_attr = e._client.timeout
    if isinstance(timeout_attr, int | float):
        assert float(timeout_attr) == 42.0
    else:
        assert float(timeout_attr.read) == 42.0


def test_redact_secrets_strips_api_key_query_param() -> None:
    """Pubmed (and any future source that puts a key in the query
    string) must have it scrubbed before logging or surfacing in
    SourceUnavailable.reason. NCBI's E-utilities accept api_key only
    as a query param, so this is the realistic leak path."""
    from research_mcp.errors import redact_secrets

    raw = (
        "Client error '429 ' for url 'https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi?db=pubmed&id=12345&api_key=secret123"
        "&email=test@example.com'"
    )
    scrubbed = redact_secrets(raw)
    assert "secret123" not in scrubbed
    assert "api_key=REDACTED" in scrubbed
    # Non-secret params still visible.
    assert "db=pubmed" in scrubbed
    assert "email=" in scrubbed


def test_redact_secrets_handles_first_param_position() -> None:
    """`?api_key=...` (first query param, with `?` separator) and
    `&api_key=...` (later, with `&`) must both match."""
    from research_mcp.errors import redact_secrets

    a = redact_secrets("https://x.com/path?api_key=abc123")
    b = redact_secrets("https://x.com/path?other=ok&api_key=abc123")
    assert "abc123" not in a
    assert "abc123" not in b
    assert "?api_key=REDACTED" in a
    assert "&api_key=REDACTED" in b


def test_redact_secrets_strips_token_and_secret_variants() -> None:
    """Cover other secret-shaped parameter names too — defense in depth
    for any future source that authenticates differently."""
    from research_mcp.errors import redact_secrets

    cases = [
        ("?token=abc", "?token=REDACTED"),
        ("?access_token=xyz", "?access_token=REDACTED"),
        ("?secret=hunter2", "?secret=REDACTED"),
        ("?api-key=dashed", "?api-key=REDACTED"),
    ]
    for raw, expected_fragment in cases:
        scrubbed = redact_secrets(raw)
        assert expected_fragment in scrubbed
        # The original secret value is gone.
        assert raw.split("=", 1)[1] not in scrubbed


def test_redact_secrets_is_idempotent() -> None:
    """Calling twice doesn't double-redact or break the URL."""
    from research_mcp.errors import redact_secrets

    once = redact_secrets("https://x.com/?api_key=abc")
    twice = redact_secrets(once)
    assert once == twice
