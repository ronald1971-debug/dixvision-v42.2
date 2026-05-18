"""Phase-6 P1-1 — governance-side defence-in-depth signal-trust cap audit.

The primary signal-trust cap is applied at the harness gate
(:func:`governance_engine.harness_approver.apply_signal_trust_cap`).
The audit nevertheless flagged that
:class:`governance_engine.engine.GovernanceEngine` was annotating its
``SIGNAL_AUDIT`` ledger row with no trust metadata — meaning a replay
could not reconstruct *which cap was active* at the time of the
signal. This test pins the new annotation:

* ``signal_trust`` / ``signal_source`` / ``confidence`` always
  present on the SIGNAL_AUDIT payload.
* ``cap_value`` reflects the registry's ``cap_for(source, trust)``
  result, or :func:`default_cap_for` when no registry is wired.
* ``cap_applied`` is ``True`` iff the cap would actually clamp the
  signal's confidence (i.e. the original confidence exceeds the
  cap).
* INTERNAL signals always have ``cap_value=None`` and
  ``cap_applied=False`` — they are not subject to the external
  trust cap.

The cap itself is idempotent at the harness gate; this engine path is
audit-only. The two tiers are independent (defence-in-depth).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from core.contracts.events import Side, SignalEvent
from core.contracts.external_signal_trust import load_external_signal_trust
from core.contracts.signal_trust import SignalTrust
from governance_engine.control_plane import LedgerAuthorityWriter
from governance_engine.engine import GovernanceEngine

REGISTRY_YAML = Path(__file__).resolve().parents[1] / "registry" / "external_signal_trust.yaml"


def _signal_audit_rows(eng: GovernanceEngine) -> list[dict[str, object]]:
    return [
        dataclasses.asdict(r) if dataclasses.is_dataclass(r) else dict(r.__dict__)
        for r in eng.ledger.read()
        if r.kind == "SIGNAL_AUDIT"
    ]


def _build_signal(
    *,
    confidence: float,
    trust: SignalTrust,
    source: str,
    ts_ns: int = 1_000,
) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="BTC-USD",
        side=Side.BUY,
        confidence=confidence,
        plugin_chain=("paper-s5",),
        signal_trust=trust,
        signal_source=source,
    )


def test_signal_audit_internal_signal_skips_cap() -> None:
    registry = load_external_signal_trust(REGISTRY_YAML)
    eng = GovernanceEngine(
        ledger=LedgerAuthorityWriter(),
        signal_trust_registry=registry,
    )
    eng.process(_build_signal(confidence=0.9, trust=SignalTrust.INTERNAL, source=""))
    rows = _signal_audit_rows(eng)
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["signal_trust"] == "INTERNAL"
    assert payload["cap_value"] is None
    assert payload["cap_applied"] is False


def test_signal_audit_external_low_with_registry_records_cap() -> None:
    registry = load_external_signal_trust(REGISTRY_YAML)
    eng = GovernanceEngine(
        ledger=LedgerAuthorityWriter(),
        signal_trust_registry=registry,
    )
    eng.process(
        _build_signal(
            confidence=0.9,
            trust=SignalTrust.EXTERNAL_LOW,
            source="SRC-SIGNAL-TRADINGVIEW-ALERT-001",
        )
    )
    rows = _signal_audit_rows(eng)
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["signal_trust"] == "EXTERNAL_LOW"
    assert payload["signal_source"] == "SRC-SIGNAL-TRADINGVIEW-ALERT-001"
    assert payload["cap_value"] == 0.4
    assert payload["cap_applied"] is True


def test_signal_audit_external_low_below_cap_records_not_applied() -> None:
    registry = load_external_signal_trust(REGISTRY_YAML)
    eng = GovernanceEngine(
        ledger=LedgerAuthorityWriter(),
        signal_trust_registry=registry,
    )
    eng.process(
        _build_signal(
            confidence=0.2,
            trust=SignalTrust.EXTERNAL_LOW,
            source="SRC-SIGNAL-TRADINGVIEW-ALERT-001",
        )
    )
    rows = _signal_audit_rows(eng)
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["cap_value"] == 0.4
    assert payload["cap_applied"] is False


def test_signal_audit_unknown_source_falls_back_to_default_cap() -> None:
    registry = load_external_signal_trust(REGISTRY_YAML)
    eng = GovernanceEngine(
        ledger=LedgerAuthorityWriter(),
        signal_trust_registry=registry,
    )
    eng.process(
        _build_signal(
            confidence=0.99,
            trust=SignalTrust.EXTERNAL_LOW,
            source="SRC-UNREGISTERED-NEW-FEED-999",
        )
    )
    rows = _signal_audit_rows(eng)
    payload = rows[0]["payload"]
    assert payload["signal_source"] == "SRC-UNREGISTERED-NEW-FEED-999"
    assert payload["cap_value"] is not None
    assert 0.0 < float(payload["cap_value"]) <= 1.0
    assert payload["cap_applied"] is True


def test_signal_audit_no_registry_uses_default_cap_function() -> None:
    eng = GovernanceEngine(ledger=LedgerAuthorityWriter())
    eng.process(
        _build_signal(
            confidence=0.99,
            trust=SignalTrust.EXTERNAL_LOW,
            source="SRC-DOES-NOT-MATTER",
        )
    )
    rows = _signal_audit_rows(eng)
    payload = rows[0]["payload"]
    assert payload["cap_value"] is not None
    assert payload["cap_applied"] is True
