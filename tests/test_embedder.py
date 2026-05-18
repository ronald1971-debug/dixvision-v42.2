# ADAPTED FROM: https://github.com/UKPLab/sentence-transformers (Apache-2.0)
#
# Tests for the I-15 sentence-transformers-shape deterministic text embedder.
"""I-15 tests: hashed-token deterministic text embedder."""

from __future__ import annotations

import dataclasses
import inspect
import math
import struct

import pytest

from state.memory_tensor import embedder as emb
from state.memory_tensor.contracts import validate_embedding
from state.memory_tensor.embedder import (
    NEW_PIP_DEPENDENCIES,
    EmbedderError,
    EmbedderSpec,
    embed_batch,
    embed_text,
    enable_sentence_transformer_factory,
)


def _to_bytes(vec: tuple[float, ...]) -> bytes:
    return b"".join(struct.pack("<d", x) for x in vec)


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("sentence-transformers",)


def test_embedder_error_is_value_error_subclass() -> None:
    assert issubclass(EmbedderError, ValueError)


# ---------------------------------------------------------------------------
# EmbedderSpec validation
# ---------------------------------------------------------------------------


def test_embedder_spec_happy_path() -> None:
    spec = EmbedderSpec(dim=128)
    assert spec.dim == 128
    assert spec.normalize is True
    assert spec.ngram_range == (1, 1)


def test_embedder_spec_is_frozen() -> None:
    spec = EmbedderSpec(dim=64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.dim = 128  # type: ignore[misc]


@pytest.mark.parametrize("bad", [0, -1, 1.5, "64", True])
def test_embedder_spec_rejects_bad_dim(bad: object) -> None:
    with pytest.raises(EmbedderError):
        EmbedderSpec(dim=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [(0, 1), (1, 0), (2, 1), (1.0, 2), (1, 2, 3), [1, 2]])
def test_embedder_spec_rejects_bad_ngram_range(bad: object) -> None:
    with pytest.raises(EmbedderError):
        EmbedderSpec(dim=32, ngram_range=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# embed_text — contract compliance
# ---------------------------------------------------------------------------


def test_embed_text_returns_tuple_of_floats() -> None:
    out = embed_text("hello world", EmbedderSpec(dim=16))
    assert isinstance(out, tuple)
    assert all(isinstance(x, float) for x in out)
    assert len(out) == 16


def test_embed_text_satisfies_memory_tensor_validate_embedding() -> None:
    """Output must pass the S-08 :func:`validate_embedding` predicate."""
    out = embed_text("the quick brown fox", EmbedderSpec(dim=32))
    # validate_embedding raises on bad inputs; passing means happy path.
    validate_embedding(out, field="test")


def test_embed_text_rejects_non_str_input() -> None:
    with pytest.raises(EmbedderError):
        embed_text(123, EmbedderSpec(dim=8))  # type: ignore[arg-type]


def test_embed_text_rejects_non_spec() -> None:
    with pytest.raises(EmbedderError):
        embed_text("hi", "not-a-spec")  # type: ignore[arg-type]


def test_embed_text_empty_string_yields_zero_vector() -> None:
    out = embed_text("", EmbedderSpec(dim=8))
    assert out == (0.0,) * 8


def test_embed_text_whitespace_only_yields_zero_vector() -> None:
    out = embed_text("   \t\n  ", EmbedderSpec(dim=8))
    assert out == (0.0,) * 8


def test_embed_text_normalized_unit_norm() -> None:
    out = embed_text("alpha beta gamma delta", EmbedderSpec(dim=64))
    norm_sq = math.fsum(x * x for x in out)
    assert norm_sq == pytest.approx(1.0)


def test_embed_text_unnormalized_keeps_integer_counts() -> None:
    spec = EmbedderSpec(dim=64, normalize=False)
    out = embed_text("alpha alpha beta", spec)
    # The sum of all buckets equals the total token count.
    assert math.fsum(out) == pytest.approx(3.0)


def test_embed_text_lowercase_invariant() -> None:
    spec = EmbedderSpec(dim=64)
    a = embed_text("Hello World", spec)
    b = embed_text("hello world", spec)
    assert _to_bytes(a) == _to_bytes(b)


def test_embed_text_punctuation_stripped() -> None:
    spec = EmbedderSpec(dim=64)
    a = embed_text("hello, world!", spec)
    b = embed_text("hello world", spec)
    assert _to_bytes(a) == _to_bytes(b)


def test_embed_text_distinct_texts_yield_distinct_vectors() -> None:
    spec = EmbedderSpec(dim=64)
    a = embed_text("buy bitcoin", spec)
    b = embed_text("sell ethereum", spec)
    assert _to_bytes(a) != _to_bytes(b)


def test_embed_text_ngram_changes_output() -> None:
    unigram = embed_text("foo bar baz", EmbedderSpec(dim=64))
    bigram = embed_text("foo bar baz", EmbedderSpec(dim=64, ngram_range=(1, 2)))
    assert _to_bytes(unigram) != _to_bytes(bigram)


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------


def test_embed_batch_preserves_order() -> None:
    spec = EmbedderSpec(dim=32)
    texts = ("hello", "world", "foo")
    batch = embed_batch(texts, spec)
    assert len(batch) == 3
    for i, text in enumerate(texts):
        assert _to_bytes(batch[i]) == _to_bytes(embed_text(text, spec))


def test_embed_batch_empty_input() -> None:
    assert embed_batch((), EmbedderSpec(dim=8)) == ()


def test_embed_batch_rejects_string_input() -> None:
    with pytest.raises(EmbedderError):
        embed_batch("not a sequence", EmbedderSpec(dim=8))  # type: ignore[arg-type]


def test_embed_batch_rejects_bytes_input() -> None:
    with pytest.raises(EmbedderError):
        embed_batch(b"not a sequence", EmbedderSpec(dim=8))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def test_inv15_embed_text_three_run_byte_identical() -> None:
    spec = EmbedderSpec(dim=64)
    text = "the quick brown fox jumps over the lazy dog"
    a = embed_text(text, spec)
    b = embed_text(text, spec)
    c = embed_text(text, spec)
    assert _to_bytes(a) == _to_bytes(b) == _to_bytes(c)


def test_inv15_embed_batch_three_run_byte_identical() -> None:
    spec = EmbedderSpec(dim=32, ngram_range=(1, 2))
    texts = ("foo bar", "baz qux", "")
    a = embed_batch(texts, spec)
    b = embed_batch(texts, spec)
    c = embed_batch(texts, spec)
    assert _to_bytes(a[0]) + _to_bytes(a[1]) + _to_bytes(a[2]) == (
        _to_bytes(b[0]) + _to_bytes(b[1]) + _to_bytes(b[2])
    )
    assert _to_bytes(a[0]) + _to_bytes(a[1]) + _to_bytes(a[2]) == (
        _to_bytes(c[0]) + _to_bytes(c[1]) + _to_bytes(c[2])
    )


# ---------------------------------------------------------------------------
# sentence-transformers lazy seam
# ---------------------------------------------------------------------------


def test_sentence_transformer_factory_rejects_bad_model_name() -> None:
    with pytest.raises(EmbedderError):
        enable_sentence_transformer_factory("")
    with pytest.raises(EmbedderError):
        enable_sentence_transformer_factory(123)  # type: ignore[arg-type]


def test_sentence_transformer_factory_lazy_seam_when_missing() -> None:
    """When ``sentence_transformers`` is not installed, the factory call
    raises :class:`ModuleNotFoundError` — proving the seam stays lazy."""
    try:
        import sentence_transformers  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError):
            enable_sentence_transformer_factory("any-model")
    else:
        pytest.skip("sentence_transformers is installed; lazy-seam path unobservable")


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return inspect.getsource(emb)


def test_no_top_level_sentence_transformers_import() -> None:
    src = _module_source().splitlines()
    for line in src:
        if line.startswith(("import sentence_transformers", "from sentence_transformers")):
            raise AssertionError(
                f"sentence_transformers must be a lazy seam — no top-level import: {line!r}"
            )


def test_no_top_level_torch_import() -> None:
    src = _module_source().splitlines()
    for line in src:
        if line.startswith(("import torch", "from torch")):
            raise AssertionError(f"torch must be a lazy seam — no top-level import: {line!r}")


def test_no_top_level_numpy_import() -> None:
    src = _module_source().splitlines()
    for line in src:
        if line.startswith(("import numpy", "from numpy")):
            raise AssertionError(f"numpy must be a lazy seam — no top-level import: {line!r}")


def test_no_forbidden_top_level_imports() -> None:
    forbidden = (
        "import time",
        "from time",
        "import datetime",
        "from datetime",
        "import random",
        "from random",
        "import asyncio",
        "from asyncio",
        "import polars",
        "from polars",
        "import requests",
        "from requests",
    )
    src = _module_source().splitlines()
    for line in src:
        for bad in forbidden:
            if line.startswith(bad):
                raise AssertionError(f"forbidden top-level import: {line!r}")


def test_no_typed_event_constructors() -> None:
    forbidden_ctors = (
        "SignalEvent(",
        "ExecutionEvent(",
        "ExecutionIntent(",
        "HazardEvent(",
        "LearningUpdate(",
        "PatchProposal(",
    )
    src = _module_source()
    for ctor in forbidden_ctors:
        assert ctor not in src, f"OFFLINE_ONLY module must not construct typed events: {ctor!r}"
