"""Tests for the shared tokenization helpers in `service/_tokens.py`."""

from __future__ import annotations

import pytest

from research_mcp.service._tokens import NON_ALNUM_RE, normalize_unicode, tokenize

pytestmark = pytest.mark.unit


def test_normalize_unicode_folds_diacritics() -> None:
    assert normalize_unicode("Schölkopf") == "Scholkopf"
    assert normalize_unicode("Évariste Galois") == "Evariste Galois"


def test_normalize_unicode_drops_undecomposable_non_ascii() -> None:
    # CJK glyphs and arrows have no ASCII decomposition — they vanish.
    assert normalize_unicode("注意力") == ""
    assert normalize_unicode("a→b") == "ab"


def test_normalize_unicode_passes_plain_ascii_through() -> None:
    assert normalize_unicode("Attention Is All You Need") == "Attention Is All You Need"


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert tokenize("Attention Is All You Need!") == [
        "attention",
        "is",
        "all",
        "you",
        "need",
    ]


def test_tokenize_splits_numbers_and_hyphens() -> None:
    assert tokenize("GPT-4 reaches 28.4 BLEU") == [
        "gpt",
        "4",
        "reaches",
        "28",
        "4",
        "bleu",
    ]


def test_tokenize_preserves_order_and_repeats() -> None:
    assert tokenize("the the cat cat") == ["the", "the", "cat", "cat"]


def test_tokenize_empty_and_punctuation_only_inputs() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!!! --- ...") == []


def test_tokenize_filters_supplied_stopwords() -> None:
    stop = frozenset({"a", "the", "of"})
    assert tokenize("the theory of a mind", stopwords=stop) == ["theory", "mind"]


def test_tokenize_keeps_stopwords_by_default() -> None:
    assert tokenize("the theory of a mind") == ["the", "theory", "of", "a", "mind"]


def test_tokenize_folds_unicode_before_splitting() -> None:
    assert tokenize("Schölkopf & Müller") == ["scholkopf", "muller"]


def test_non_alnum_re_is_compiled_and_exported() -> None:
    assert NON_ALNUM_RE.split("a-b c") == ["a", "b", "c"]
