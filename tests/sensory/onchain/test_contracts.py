"""Unit tests for sensory.onchain.contracts."""

from __future__ import annotations

import pytest

from sensory.onchain.contracts import OnChainMetric


def test_minimal_construct() -> None:
    m = OnChainMetric(
        ts_ns=1,
        source="GLASSNODE",
        metric="sopr",
        value=1.05,
    )
    assert m.asset == ""
    assert m.unit == ""
    assert m.observed_ts_ns is None
    assert dict(m.meta) == {}


def test_full_construct() -> None:
    m = OnChainMetric(
        ts_ns=1,
        source="DUNE",
        metric="active_addresses_24h",
        value=905_321.0,
        asset="ETH",
        unit="count",
        observed_ts_ns=2,
        meta={"chain": "ethereum"},
    )
    assert m.asset == "ETH"
    assert m.observed_ts_ns == 2
    assert dict(m.meta) == {"chain": "ethereum"}


def test_frozen() -> None:
    m = OnChainMetric(ts_ns=1, source="GLASSNODE", metric="x", value=0.0)
    with pytest.raises(AttributeError):
        m.value = 1.0  # type: ignore[misc]


def test_slotted() -> None:
    m = OnChainMetric(ts_ns=1, source="GLASSNODE", metric="x", value=0.0)
    assert not hasattr(m, "__dict__")


@pytest.mark.parametrize(
    "field, value, msg",
    [
        ("source", "", "source"),
        ("metric", "", "metric"),
        ("observed_ts_ns", 0, "observed_ts_ns"),
        ("observed_ts_ns", -1, "observed_ts_ns"),
    ],
)
def test_validation_rejects(field: str, value: object, msg: str) -> None:
    kwargs = {
        "ts_ns": 1,
        "source": "GLASSNODE",
        "metric": "x",
        "value": 0.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=msg):
        OnChainMetric(**kwargs)  # type: ignore[arg-type]
