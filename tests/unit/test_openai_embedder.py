"""OpenAIEmbedder behavior tests.

Uses a stub `AsyncOpenAI` client so the tests are offline and deterministic.
The truncation logic is the part worth verifying — the API call itself is
trivial glue.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_mcp.embedder import OpenAIEmbedder

pytestmark = pytest.mark.unit


class _FakeEmbeddings:
    def __init__(self, dim: int) -> None:
        self.last_input: list[str] = []
        self._dim = dim

    async def create(self, *, model: str, input: list[str]) -> Any:
        del model  # not exercised in tests
        self.last_input = list(input)

        class _Item:
            def __init__(self, vec: list[float]) -> None:
                self.embedding = vec

        class _Response:
            def __init__(self, items: list[_Item]) -> None:
                self.data = items

        return _Response([_Item([0.0] * self._dim) for _ in input])


class _FakeClient:
    def __init__(self, dim: int) -> None:
        self.embeddings = _FakeEmbeddings(dim)


def test_short_inputs_pass_through_untruncated() -> None:
    client = _FakeClient(1536)
    embedder = OpenAIEmbedder(client=client)  # type: ignore[arg-type]

    import asyncio

    asyncio.run(embedder.embed(["hello world", "another short phrase"]))
    assert client.embeddings.last_input == ["hello world", "another short phrase"]


def test_long_input_is_truncated_before_send() -> None:
    client = _FakeClient(1536)
    embedder = OpenAIEmbedder(client=client)  # type: ignore[arg-type]

    # Build an input we know exceeds 8000 tokens. "transformer attention
    # mechanism " is roughly 4 tokens per repetition; 3000 repetitions
    # comfortably overshoots the cap.
    big = "transformer attention mechanism " * 3000
    assert len(embedder._encoder.encode(big)) > 8000, "test setup: input must overflow"

    import asyncio

    asyncio.run(embedder.embed([big]))
    sent = client.embeddings.last_input[0]
    # tiktoken roundtrip preserves whole tokens; the truncated form is
    # strictly shorter than the original.
    assert len(sent) < len(big)
    # Token count of what we actually sent is at or below the cap.
    assert len(embedder._encoder.encode(sent)) <= 8000


def test_empty_input_substituted_to_keep_indices_aligned() -> None:
    """If a caller passes an all-whitespace string that truncates to '',
    we substitute ' ' so the per-batch result list lines up with the input."""
    client = _FakeClient(1536)
    embedder = OpenAIEmbedder(client=client)  # type: ignore[arg-type]

    import asyncio

    asyncio.run(embedder.embed(["", "real text"]))
    assert len(client.embeddings.last_input) == 2
    assert client.embeddings.last_input[1] == "real text"


def test_empty_list_short_circuits() -> None:
    client = _FakeClient(1536)
    embedder = OpenAIEmbedder(client=client)  # type: ignore[arg-type]

    import asyncio

    out = asyncio.run(embedder.embed([]))
    assert out == []
    assert client.embeddings.last_input == []
