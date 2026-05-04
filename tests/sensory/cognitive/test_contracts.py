"""Unit tests for sensory.cognitive.contracts."""

from __future__ import annotations

import pytest

from sensory.cognitive.contracts import AIResponse


def _ok(**overrides: object) -> AIResponse:
    kwargs = {
        "ts_ns": 1,
        "source": "OPENAI",
        "request_id": "req-abc",
        "model": "gpt-4o",
    }
    kwargs.update(overrides)
    return AIResponse(**kwargs)  # type: ignore[arg-type]


def test_minimal_construct() -> None:
    r = _ok()
    assert r.body == ""
    assert r.finish_reason == ""
    assert r.prompt_tokens is None
    assert r.completion_tokens is None
    assert r.latency_ms is None
    assert dict(r.meta) == {}


def test_full_construct() -> None:
    r = _ok(
        body="The answer is 42.",
        finish_reason="stop",
        prompt_tokens=128,
        completion_tokens=8,
        latency_ms=412,
        meta={"safety": "ok"},
    )
    assert r.body.endswith("42.")
    assert r.prompt_tokens == 128
    assert r.completion_tokens == 8
    assert r.latency_ms == 412


def test_frozen_and_slotted() -> None:
    r = _ok()
    with pytest.raises(AttributeError):
        r.body = "hello"  # type: ignore[misc]


def test_zero_counts_allowed() -> None:
    r = _ok(prompt_tokens=0, completion_tokens=0, latency_ms=0)
    assert r.prompt_tokens == 0
    assert r.completion_tokens == 0
    assert r.latency_ms == 0


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("request_id", ""),
        ("model", ""),
        ("prompt_tokens", -1),
        ("completion_tokens", -1),
        ("latency_ms", -1),
    ],
)
def test_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok(**{field: value})
