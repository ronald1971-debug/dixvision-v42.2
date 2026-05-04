"""Unit tests for sensory.regulatory.contracts."""

from __future__ import annotations

import pytest

from sensory.regulatory.contracts import Filing


def _ok(**overrides: object) -> Filing:
    kwargs = {
        "ts_ns": 1,
        "source": "SEC_EDGAR",
        "filing_id": "0001628280-23-001234",
        "form_type": "10-K",
        "filer": "Apple Inc.",
    }
    kwargs.update(overrides)
    return Filing(**kwargs)  # type: ignore[arg-type]


def test_minimal_construct() -> None:
    f = _ok()
    assert f.url == ""
    assert f.filed_ts_ns is None
    assert dict(f.meta) == {}


def test_full_construct() -> None:
    f = _ok(
        url="https://www.sec.gov/Archives/edgar/data/...",
        filed_ts_ns=2,
        meta={"cik": "0000320193", "ticker": "AAPL"},
    )
    assert f.url.startswith("https://")
    assert f.filed_ts_ns == 2


def test_frozen_and_slotted() -> None:
    f = _ok()
    with pytest.raises(AttributeError):
        f.form_type = "8-K"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("filing_id", ""),
        ("form_type", ""),
        ("filer", ""),
        ("filed_ts_ns", 0),
        ("filed_ts_ns", -1),
    ],
)
def test_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok(**{field: value})
