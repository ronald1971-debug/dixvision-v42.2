"""Tests for the SignalTrust contract (Paper-S1).

Pure value-object tests; no I/O, no fixtures, no clock.
"""

from __future__ import annotations

import pytest

from core.contracts.signal_trust import (
    DEFAULT_LOW_CAP,
    DEFAULT_MED_CAP,
    SignalTrust,
    clamp_confidence,
    default_cap_for,
)


class TestDefaultCapFor:
    def test_internal_returns_none(self) -> None:
        assert default_cap_for(SignalTrust.INTERNAL) is None

    def test_external_low_returns_low_cap(self) -> None:
        assert default_cap_for(SignalTrust.EXTERNAL_LOW) == DEFAULT_LOW_CAP

    def test_external_med_returns_med_cap(self) -> None:
        assert default_cap_for(SignalTrust.EXTERNAL_MED) == DEFAULT_MED_CAP

    def test_low_cap_is_more_restrictive_than_med(self) -> None:
        assert DEFAULT_LOW_CAP <= DEFAULT_MED_CAP


class TestClampConfidence:
    def test_passthrough_when_cap_none(self) -> None:
        assert clamp_confidence(0.9, None) == 0.9

    def test_clamp_above_cap(self) -> None:
        assert clamp_confidence(0.9, 0.5) == 0.5

    def test_no_clamp_below_cap(self) -> None:
        assert clamp_confidence(0.3, 0.5) == 0.3

    def test_clamp_at_cap(self) -> None:
        assert clamp_confidence(0.5, 0.5) == 0.5

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValueError, match=r"confidence must be in"):
            clamp_confidence(-0.1, None)

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"confidence must be in"):
            clamp_confidence(1.1, None)

    def test_rejects_cap_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"cap must be in"):
            clamp_confidence(0.5, 1.1)

    def test_rejects_cap_below_zero(self) -> None:
        with pytest.raises(ValueError, match=r"cap must be in"):
            clamp_confidence(0.5, -0.1)
