"""Unit tests for sensory.dev.contracts."""

from __future__ import annotations

import pytest

from sensory.dev.contracts import RepoEvent


def _ok(**overrides: object) -> RepoEvent:
    kwargs = {
        "ts_ns": 1,
        "source": "GITHUB",
        "event_id": "del-abc123",
        "repo": "ronald1971-debug/dixvision-v42.2",
        "event_type": "push",
        "actor": "ronald1971-debug",
    }
    kwargs.update(overrides)
    return RepoEvent(**kwargs)  # type: ignore[arg-type]


def test_minimal_construct() -> None:
    e = _ok()
    assert e.url == ""
    assert e.occurred_ts_ns is None
    assert dict(e.meta) == {}


def test_full_construct() -> None:
    e = _ok(
        url="https://github.com/owner/repo/commit/sha",
        occurred_ts_ns=2,
        meta={"ref": "refs/heads/main", "sha": "deadbeef"},
    )
    assert e.url.startswith("https://")
    assert e.occurred_ts_ns == 2
    assert dict(e.meta)["sha"] == "deadbeef"


def test_frozen_and_slotted() -> None:
    e = _ok()
    with pytest.raises(AttributeError):
        e.event_type = "release"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("event_id", ""),
        ("repo", ""),
        ("event_type", ""),
        ("actor", ""),
        ("occurred_ts_ns", 0),
        ("occurred_ts_ns", -1),
    ],
)
def test_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok(**{field: value})
