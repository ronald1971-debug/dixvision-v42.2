# ADAPTED FROM: https://github.com/ijl/orjson
# License: Apache-2.0 / MIT (dual licensed)
#
# Canonical orjson-shape JSON codec — OFFLINE_ONLY tier.
#
# NEW_PIP_DEPENDENCIES = ("orjson",)
#
# This module never imports :mod:`orjson` at module level. The fast-path is
# gated behind :func:`enable_orjson_factory`, which performs the lazy import
# inside the function body. The default codec is stdlib :mod:`json` configured
# to emit the **byte-identical** shape that
# ``orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)`` produces for the closed
# DIX payload alphabet (``None`` / ``bool`` / ``int`` / ``float`` / ``str`` /
# ``list`` / ``tuple`` / ``dict``).
#
# Authority constraints pinned by the test suite:
#
# * B1   — no imports from any runtime engine tier (intelligence_engine,
#          execution_engine, governance_engine, evolution_engine,
#          learning_engine).
# * INV-15 (deterministic replay) — :func:`dumps_canonical` is a pure function
#          of its input; three independent calls produce byte-identical
#          output. The byte-identity property is pinned by a 3-run replay
#          equality test.
# * B27 / B28 / INV-71 (authority symmetry) — this codec is a passive
#          transcoder; no typed events are constructed.
"""Canonical orjson-shape JSON codec (I-02)."""

from .json_codec import (
    CODEC_VERSION,
    NEW_PIP_DEPENDENCIES,
    CodecError,
    JsonCodec,
    canonical_dumps,
    canonical_loads,
    default_codec,
    enable_orjson_factory,
    stdlib_codec_factory,
)

__all__ = (
    "CODEC_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "CodecError",
    "JsonCodec",
    "canonical_dumps",
    "canonical_loads",
    "default_codec",
    "enable_orjson_factory",
    "stdlib_codec_factory",
)
