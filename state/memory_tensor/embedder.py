# ADAPTED FROM: https://github.com/UKPLab/sentence-transformers (Apache-2.0)
#
# Deterministic text embedder — ``state/memory_tensor/`` is the
# vector-memory tier. The module body is pure-stdlib so it imports
# cleanly in environments without numpy, torch, or
# ``sentence_transformers``. ``sentence_transformers`` (and its torch
# stack) are lazy seams: production never imports them; the pure-Python
# hashed-token backend is the default and is INV-15-deterministic on
# every machine because it uses BLAKE2b token hashing and the canonical
# orjson-shape ``math.fsum`` summation.
#
# NEW_PIP_DEPENDENCIES = ("sentence-transformers",)
#
# Authority constraints (pinned by ``tests/test_embedder.py``):
#
#   * **OFFLINE_ONLY** — must never be imported from
#     ``execution_engine/``, ``governance_engine/``,
#     ``intelligence_engine/``, ``system_engine/``, or ``core/``.
#   * **B1** — no runtime engine imports here.
#   * **INV-15** — :func:`embed_text` and :func:`embed_batch` are pure
#     functions of their inputs; three independent calls with identical
#     arguments produce byte-identical output tuples.
#   * **B27 / B28 / INV-71** — no typed-event constructors here.
#   * No top-level imports of :mod:`sentence_transformers`,
#     :mod:`torch`, :mod:`numpy`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`asyncio`, :mod:`polars`, :mod:`requests`.
#
# The output of :func:`embed_text` satisfies the
# :class:`state.memory_tensor.contracts.validate_embedding` predicate
# byte-for-byte: a tuple of finite floats of fixed length.
"""Deterministic hashed-token text embedder (OFFLINE_ONLY)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "EmbedderError",
    "EmbedderSpec",
    "embed_text",
    "embed_batch",
    "enable_sentence_transformer_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("sentence-transformers",)


class EmbedderError(ValueError):
    """Raised by embedder helpers for malformed inputs or specs."""


# ---------------------------------------------------------------------------
# EmbedderSpec
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EmbedderSpec:
    """Frozen embedder configuration.

    Fields:
        dim: output dimension. Must be a positive ``int``.
        normalize: when true, the output is L2-normalized; when false,
            the raw bag-of-hashed-tokens count vector is returned.
        ngram_range: closed-interval token n-gram bounds. Both bounds
            inclusive; ``(1, 1)`` = unigrams only, ``(1, 2)`` = uni- +
            bi-grams.
    """

    dim: int
    normalize: bool = True
    ngram_range: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise EmbedderError(f"EmbedderSpec.dim must be int, got {type(self.dim).__name__}")
        if self.dim <= 0:
            raise EmbedderError(f"EmbedderSpec.dim must be positive, got {self.dim!r}")
        if not isinstance(self.ngram_range, tuple) or len(self.ngram_range) != 2:
            raise EmbedderError(
                f"EmbedderSpec.ngram_range must be a 2-tuple, got {self.ngram_range!r}"
            )
        lo, hi = self.ngram_range
        if not isinstance(lo, int) or not isinstance(hi, int):
            raise EmbedderError(
                "EmbedderSpec.ngram_range bounds must be int, "
                f"got ({type(lo).__name__}, {type(hi).__name__})"
            )
        if lo < 1 or hi < lo:
            raise EmbedderError(
                f"EmbedderSpec.ngram_range must satisfy 1 <= lo <= hi, got {self.ngram_range!r}"
            )


# ---------------------------------------------------------------------------
# Tokenisation + hashing
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> tuple[str, ...]:
    """Whitespace-split, lower-cased, punctuation-stripped tokens."""

    if not isinstance(text, str):
        raise EmbedderError(f"embedder input must be str, got {type(text).__name__}")
    tokens: list[str] = []
    for raw in text.split():
        cleaned = raw.strip().strip(",.;:!?'\"()[]{}").lower()
        if cleaned:
            tokens.append(cleaned)
    return tuple(tokens)


def _ngrams(tokens: tuple[str, ...], lo: int, hi: int) -> Iterable[str]:
    """Yield all n-grams with ``lo <= n <= hi`` as space-joined strings."""

    n_tokens = len(tokens)
    for n in range(lo, hi + 1):
        if n > n_tokens:
            continue
        for i in range(n_tokens - n + 1):
            yield " ".join(tokens[i : i + n])


def _bucket(token: str, dim: int) -> int:
    """Map a token to a stable bucket index in ``[0, dim)`` via BLAKE2b."""

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


# ---------------------------------------------------------------------------
# Pure-stdlib embedder
# ---------------------------------------------------------------------------


def embed_text(
    text: str,
    spec: EmbedderSpec,
) -> tuple[float, ...]:
    """Embed ``text`` into a ``spec.dim``-dimensional float tuple.

    The pure-stdlib backend hashes each token (and optionally each
    n-gram) into one of ``spec.dim`` buckets via BLAKE2b and accumulates
    a count vector. When ``spec.normalize`` is true the vector is
    L2-normalized (the empty / all-whitespace case returns a zero
    vector, which is still well-defined under cosine similarity).

    The output is INV-15-deterministic: identical ``text`` + ``spec``
    inputs always produce byte-identical output tuples on any machine.
    """

    if not isinstance(spec, EmbedderSpec):
        raise EmbedderError(f"embed_text spec must be EmbedderSpec, got {type(spec).__name__}")
    tokens = _tokenise(text)
    lo, hi = spec.ngram_range
    buckets = [0.0] * spec.dim
    for gram in _ngrams(tokens, lo, hi):
        buckets[_bucket(gram, spec.dim)] += 1.0
    if spec.normalize:
        norm_sq = math.fsum(x * x for x in buckets)
        if norm_sq > 0.0:
            inv = 1.0 / math.sqrt(norm_sq)
            buckets = [x * inv for x in buckets]
    return tuple(buckets)


def embed_batch(
    texts: Sequence[str],
    spec: EmbedderSpec,
) -> tuple[tuple[float, ...], ...]:
    """Embed a sequence of texts. Output preserves input order."""

    if isinstance(texts, (str, bytes, bytearray)):
        raise EmbedderError(
            f"embed_batch input must be a sequence of strings, got {type(texts).__name__}"
        )
    return tuple(embed_text(t, spec) for t in texts)


# ---------------------------------------------------------------------------
# Lazy ``sentence-transformers`` seam
# ---------------------------------------------------------------------------


def enable_sentence_transformer_factory(
    model_name: str,
) -> Callable[[str], tuple[float, ...]]:
    """Return a per-text embedder backed by :mod:`sentence_transformers`.

    Importing :mod:`sentence_transformers` (and its torch stack) is
    deferred to factory-call time, so production environments without
    the library installed import this module cleanly. The returned
    callable is **not** INV-15-deterministic in the general case
    (transformer inference may vary across CUDA driver versions). The
    pure-stdlib backend remains the production default; this seam is
    intended for OFFLINE_ONLY semantic-quality benchmarks against the
    hashed-token reference.
    """

    if not isinstance(model_name, str) or not model_name:
        raise EmbedderError(f"model_name must be a non-empty str, got {model_name!r}")
    from sentence_transformers import (
        SentenceTransformer,  # type: ignore[import-not-found]  # noqa: F401 - lazy seam
    )

    model: Any = SentenceTransformer(model_name)

    def _embed(text: str) -> tuple[float, ...]:
        if not isinstance(text, str):
            raise EmbedderError(
                f"sentence-transformer input must be str, got {type(text).__name__}"
            )
        vec = model.encode(text, normalize_embeddings=True)
        # ``vec`` is a numpy.ndarray; convert via ``tobytes`` round-trip
        # so we never bind numpy into the surrounding scope.
        raw = vec.tobytes()
        # numpy float32 -> tuple of Python floats (little-endian f32)
        dim = len(raw) // 4
        return tuple(struct.unpack(f"<{dim}f", raw))

    return _embed
