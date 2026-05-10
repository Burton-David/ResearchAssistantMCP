"""SentenceTransformersEmbedder tests.

Most live model interaction is gated by `RESEARCH_MCP_SENTENCE_TRANSFORMERS=1`
because the default model (`BAAI/bge-base-en-v1.5`) is ~440 MB and the unit
suite shouldn't pull it on every run. The non-gated tests verify the parts
that don't require the model to actually load.
"""

from __future__ import annotations

import os

import pytest

from research_mcp.domain.embedder import Embedder
from research_mcp.embedder import SentenceTransformersEmbedder

pytestmark = pytest.mark.unit


def test_known_model_dimension_resolves_without_loading() -> None:
    """The dim map lets us answer `embedder.dimension` before the model loads —
    important so the wiring layer can construct the Index synchronously
    without blocking on a 440 MB download."""
    e = SentenceTransformersEmbedder("BAAI/bge-base-en-v1.5")
    assert e.dimension == 768
    assert e._model is None  # confirms lazy load


def test_unknown_model_raises_clean_error_with_env_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo or genuinely unknown model name must surface a clean message
    naming RESEARCH_MCP_EMBEDDER instead of a raw HF stack trace. Unknown
    models load eagerly (they have to — the wiring layer needs `dimension`
    to construct the FAISS index at boot), but the failure mode is now
    legible.

    We mock the `_load_model` failure rather than trying to actually contact
    Hugging Face Hub. Doing the network round-trip from a unit test is both
    slow and unreliable on macOS — the cumulative state from prior numpy /
    faiss / anyio imports interacts badly with sentence-transformers'
    initialization path and can SIGABRT the test process. This mock keeps
    the test honest (verifying the wrapping logic) without that risk.
    """
    import research_mcp.embedder.sentence_transformers_embedder as st_mod

    def _broken_load(self) -> None:  # type: ignore[no-untyped-def]
        raise OSError("simulated HF Hub 404 for typo'd model name")

    monkeypatch.setattr(st_mod.SentenceTransformersEmbedder, "_load_model", _broken_load)
    with pytest.raises(RuntimeError, match=r"RESEARCH_MCP_EMBEDDER"):
        SentenceTransformersEmbedder("definitely-not-a-real/model-name")


def test_protocol_conformance() -> None:
    e = SentenceTransformersEmbedder("BAAI/bge-base-en-v1.5")
    assert isinstance(e, Embedder)


def test_small_variant_dimension() -> None:
    e = SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5")
    assert e.dimension == 384


def test_minilm_dimension() -> None:
    e = SentenceTransformersEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert e.dimension == 384


@pytest.mark.skipif(
    os.environ.get("RESEARCH_MCP_SENTENCE_TRANSFORMERS") != "1",
    reason="set RESEARCH_MCP_SENTENCE_TRANSFORMERS=1 to download bge-base "
    "and exercise real embedding (≈440 MB on first run)",
)
async def test_real_embedding_round_trip() -> None:
    e = SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5")  # smaller for the gated run
    [v1, v2] = await e.embed(["transformer attention", "transformer attention"])
    assert len(v1) == e.dimension
    # Identical inputs produce identical vectors.
    assert v1 == v2
