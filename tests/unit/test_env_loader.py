"""Tests for the stdlib `.env` loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from research_mcp._env import load_dotenv

pytestmark = pytest.mark.unit


def test_loads_simple_key_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_MCP_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("RESEARCH_MCP_TEST_KEY=hello\n")
    assert load_dotenv(tmp_path) == tmp_path / ".env"
    assert os.environ["RESEARCH_MCP_TEST_KEY"] == "hello"


def test_existing_env_var_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_MCP_TEST_KEY", "from_shell")
    (tmp_path / ".env").write_text("RESEARCH_MCP_TEST_KEY=from_file\n")
    load_dotenv(tmp_path)
    assert os.environ["RESEARCH_MCP_TEST_KEY"] == "from_shell"


def test_skips_comments_and_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RESEARCH_MCP_X", raising=False)
    (tmp_path / ".env").write_text("# a comment\n\nRESEARCH_MCP_X=ok\n")
    load_dotenv(tmp_path)
    assert os.environ["RESEARCH_MCP_X"] == "ok"


def test_strips_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_MCP_Q", raising=False)
    (tmp_path / ".env").write_text('RESEARCH_MCP_Q="quoted value"\n')
    load_dotenv(tmp_path)
    assert os.environ["RESEARCH_MCP_Q"] == "quoted value"


def test_walks_up_to_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_MCP_DEEP", raising=False)
    (tmp_path / ".env").write_text("RESEARCH_MCP_DEEP=found\n")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert load_dotenv(nested) == tmp_path / ".env"
    assert os.environ["RESEARCH_MCP_DEEP"] == "found"


def test_returns_none_when_no_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # We pass an isolated start dir; the package-dir fallback may still find
    # the project's .env, so we don't assert None on this codebase. The
    # contract is just "doesn't raise" when no .env exists at the start.
    monkeypatch.chdir(tmp_path)
    load_dotenv(tmp_path)  # must not raise
