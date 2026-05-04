"""Tests for the external_signal_trust.yaml loader (Paper-S1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.external_signal_trust import (
    DEFAULT_REGISTRY_PATH,
    ExternalSignalTrustRegistry,
    load_external_signal_trust,
)
from core.contracts.signal_trust import (
    DEFAULT_LOW_CAP,
    DEFAULT_MED_CAP,
    SignalTrust,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "external_signal_trust.yaml"
    p.write_text(body)
    return p


class TestLoadShippedRegistry:
    def test_default_registry_loads(self) -> None:
        reg = load_external_signal_trust(DEFAULT_REGISTRY_PATH)
        assert isinstance(reg, ExternalSignalTrustRegistry)
        assert reg.version >= 1
        # Paper-S4 added the first row — TradingView Pine alert webhook.
        # All registered rows must be EXTERNAL_LOW with a cap below the
        # built-in DEFAULT_LOW_CAP (0.5) until governance promotes them.
        for source_id, row in reg.sources.items():
            assert row.trust is SignalTrust.EXTERNAL_LOW, source_id
            assert row.cap is not None and 0.0 <= row.cap <= 0.5, source_id
        assert "SRC-SIGNAL-TRADINGVIEW-ALERT-001" in reg.sources


class TestParseRows:
    def test_parses_external_low_with_explicit_cap(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "version: 1\n"
            "sources:\n"
            "  tradingview.public:\n"
            "    trust: EXTERNAL_LOW\n"
            "    cap: 0.3\n"
            "    note: retail webhook\n",
        )
        reg = load_external_signal_trust(path)
        row = reg.sources["tradingview.public"]
        assert row.trust is SignalTrust.EXTERNAL_LOW
        assert row.cap == 0.3
        assert row.note == "retail webhook"

    def test_parses_internal_with_null_cap(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "version: 1\nsources:\n  internal.indira:\n    trust: INTERNAL\n    cap: null\n",
        )
        reg = load_external_signal_trust(path)
        row = reg.sources["internal.indira"]
        assert row.trust is SignalTrust.INTERNAL
        assert row.cap is None

    def test_internal_with_explicit_cap_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "version: 1\nsources:\n  internal.indira:\n    trust: INTERNAL\n    cap: 0.5\n",
        )
        with pytest.raises(ValueError, match="INTERNAL sources must not"):
            load_external_signal_trust(path)


class TestCapFor:
    def test_unknown_source_falls_back_to_class_default(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "version: 1\nsources: {}\n")
        reg = load_external_signal_trust(path)
        assert reg.cap_for("unknown", SignalTrust.EXTERNAL_LOW) == DEFAULT_LOW_CAP
        assert reg.cap_for("unknown", SignalTrust.EXTERNAL_MED) == DEFAULT_MED_CAP
        assert reg.cap_for("unknown", SignalTrust.INTERNAL) is None

    def test_known_row_takes_more_restrictive_of_row_and_class(self, tmp_path: Path) -> None:
        # Row says cap=0.3 (more restrictive than DEFAULT_LOW_CAP=0.5),
        # producer declares EXTERNAL_LOW. We must take 0.3.
        path = _write(
            tmp_path,
            "version: 1\nsources:\n  tv:\n    trust: EXTERNAL_LOW\n    cap: 0.3\n",
        )
        reg = load_external_signal_trust(path)
        assert reg.cap_for("tv", SignalTrust.EXTERNAL_LOW) == 0.3

    def test_class_more_restrictive_than_row_wins(self, tmp_path: Path) -> None:
        # Row says EXTERNAL_MED with cap=0.8 but the producer declares
        # EXTERNAL_LOW (DEFAULT_LOW_CAP=0.5). We must take 0.5 — fail-closed.
        path = _write(
            tmp_path,
            "version: 1\nsources:\n  qc:\n    trust: EXTERNAL_MED\n    cap: 0.8\n",
        )
        reg = load_external_signal_trust(path)
        assert reg.cap_for("qc", SignalTrust.EXTERNAL_LOW) == DEFAULT_LOW_CAP


class TestValidationErrors:
    def test_rejects_non_mapping_top(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "[]\n")
        with pytest.raises(ValueError, match="top-level must be a mapping"):
            load_external_signal_trust(path)

    def test_rejects_missing_version(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "sources: {}\n")
        with pytest.raises(ValueError, match="'version' must be"):
            load_external_signal_trust(path)

    def test_rejects_unknown_trust_class(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "version: 1\nsources:\n  bad:\n    trust: WHATEVER\n",
        )
        with pytest.raises(ValueError, match="unknown trust class"):
            load_external_signal_trust(path)

    def test_rejects_cap_out_of_range(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "version: 1\nsources:\n  bad:\n    trust: EXTERNAL_LOW\n    cap: 1.5\n",
        )
        with pytest.raises(ValueError, match=r"'cap' must be in \[0.0, 1.0\]"):
            load_external_signal_trust(path)
