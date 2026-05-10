"""Tests for `_select_embedder` — the env-driven embedder picker."""

from __future__ import annotations

import pytest

from research_mcp.embedder import OpenAIEmbedder, SentenceTransformersEmbedder
from research_mcp.mcp.server import _select_embedder

pytestmark = pytest.mark.unit


def test_explicit_openai_spec_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_MCP_EMBEDDER", "openai:text-embedding-3-large")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    embedder, label = _select_embedder()
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == "text-embedding-3-large"
    assert label == "openai:text-embedding-3-large"


def test_explicit_sentence_transformers_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_MCP_EMBEDDER", "sentence-transformers:BAAI/bge-small-en-v1.5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder, label = _select_embedder()
    assert isinstance(embedder, SentenceTransformersEmbedder)
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert label == "sentence-transformers:BAAI/bge-small-en-v1.5"


def test_falls_back_to_openai_when_only_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_MCP_EMBEDDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    embedder, label = _select_embedder()
    assert isinstance(embedder, OpenAIEmbedder)
    assert label == "openai:text-embedding-3-small"


def test_returns_none_when_nothing_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_MCP_EMBEDDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder, label = _select_embedder()
    assert embedder is None
    assert label is None


def test_unknown_kind_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_MCP_EMBEDDER", "magic:my-special-model")
    with pytest.raises(RuntimeError, match="not understood"):
        _select_embedder()
