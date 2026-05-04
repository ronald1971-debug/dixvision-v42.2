"""Paper-S7 -- DecisionTrace confidence-cap audit fields.

Paper-S5 / Paper-S6 enforce a per-source confidence cap at the harness
gate; Paper-S7 makes the cap-applied path observable on the audit
ledger by adding three optional fields to :class:`DecisionTrace`:

* ``original_confidence`` -- producer-emitted confidence BEFORE the
  cap was applied;
* ``confidence_cap_applied`` -- ``True`` iff the cap strictly clamped
  the value down;
* ``confidence_cap_value`` -- the cap that was used (``None`` for
  INTERNAL signals where no cap applies).

These tests pin:

* Default field values (``None`` / ``False``) on a pre-Paper-S7 trace.
* Builder accepts the new kwargs and threads them onto the trace.
* Round-trip via :func:`as_system_event` / :func:`trace_from_system_event`.
* :class:`DecisionTrace` contract validation:
    - ``final_confidence > original_confidence`` is rejected (cap is
      monotone -- the post-cap value never exceeds the pre-cap value).
    - ``confidence_cap_applied=True`` requires both
      ``original_confidence`` and ``confidence_cap_value``.
    - ``original_confidence`` outside [0.0, 1.0] is rejected.
    - ``confidence_cap_value`` outside [0.0, 1.0] is rejected.
* :func:`apply_signal_trust_cap_with_audit` semantics:
    - INTERNAL pass-through: ``cap_value=None``, ``applied=False``.
    - EXTERNAL_LOW with confidence at/under cap: ``applied=False``.
    - EXTERNAL_LOW with confidence over cap: ``applied=True``,
      ``original_confidence`` = pre-cap value.
    - Promotion store widens the cap; pre-cap value still recorded.
    - Idempotent under double-apply (clamp is monotone).
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from core.coherence.decision_trace import (
    as_system_event,
    build_decision_trace,
    trace_from_system_event,
)
from core.contracts.decision_trace import DECISION_TRACE_VERSION, DecisionTrace
from core.contracts.events import EventKind, Side, SignalEvent
from core.contracts.external_signal_trust import (
    ExternalSignalSource,
    ExternalSignalTrustRegistry,
)
from core.contracts.signal_trust import (
    DEFAULT_LOW_CAP,
    DEFAULT_MED_CAP,
    SignalTrust,
)
from core.contracts.source_trust_promotions import SourceTrustPromotionStore
from governance_engine.harness_approver import (
    ConfidenceCapAudit,
    apply_signal_trust_cap,
    apply_signal_trust_cap_with_audit,
)

_TS_NS = 1_700_000_000_000_000_000


def _signal(
    *,
    confidence: float,
    trust: SignalTrust,
    source: str = "",
) -> SignalEvent:
    return SignalEvent(
        ts_ns=_TS_NS,
        produced_by_engine="intelligence",
        symbol="BTC-USD",
        side=Side.BUY,
        confidence=confidence,
        plugin_chain=("test",),
        meta=(),
        signal_trust=trust,
        signal_source=source,
        kind=EventKind.SIGNAL,
    )


def _registry(
    *,
    sources: dict[str, ExternalSignalSource],
) -> ExternalSignalTrustRegistry:
    return ExternalSignalTrustRegistry(
        version=1,
        sources=MappingProxyType(dict(sources)),
    )


# -- DecisionTrace defaults (back-compat) ---------------------------


def test_decision_trace_defaults_for_pre_paper_s7_callers() -> None:
    """Builder without S7 kwargs emits a trace with the new fields cleared."""
    sig = _signal(confidence=0.4, trust=SignalTrust.INTERNAL)
    trace = build_decision_trace(signal=sig)
    assert trace.original_confidence is None
    assert trace.confidence_cap_applied is False
    assert trace.confidence_cap_value is None
    assert trace.final_confidence == 0.4


def test_decision_trace_carries_audit_fields_when_supplied() -> None:
    sig = _signal(
        confidence=DEFAULT_LOW_CAP,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )
    trace = build_decision_trace(
        signal=sig,
        original_confidence=0.95,
        confidence_cap_applied=True,
        confidence_cap_value=DEFAULT_LOW_CAP,
    )
    assert trace.original_confidence == 0.95
    assert trace.confidence_cap_applied is True
    assert trace.confidence_cap_value == DEFAULT_LOW_CAP


# -- contract validation --------------------------------------------


def test_decision_trace_rejects_final_above_original_confidence() -> None:
    sig = _signal(confidence=0.6, trust=SignalTrust.EXTERNAL_LOW)
    with pytest.raises(ValueError, match="final_confidence must not exceed"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="abc1234567890def",
            ts_ns=_TS_NS,
            symbol="BTC-USD",
            side=Side.BUY,
            final_confidence=0.6,
            plugin_chain=("test",),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
            why=None,
            signal_trust=sig.signal_trust,
            signal_source=None,
            validation_score=None,
            original_confidence=0.5,  # < final_confidence -- forbidden
            confidence_cap_applied=False,
            confidence_cap_value=None,
        )


def test_decision_trace_applied_requires_original() -> None:
    with pytest.raises(ValueError, match="original_confidence is None"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="abc1234567890def",
            ts_ns=_TS_NS,
            symbol="BTC-USD",
            side=Side.BUY,
            final_confidence=0.4,
            plugin_chain=("test",),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
            why=None,
            signal_trust=None,
            signal_source=None,
            validation_score=None,
            original_confidence=None,
            confidence_cap_applied=True,
            confidence_cap_value=DEFAULT_LOW_CAP,
        )


def test_decision_trace_applied_requires_cap_value() -> None:
    with pytest.raises(ValueError, match="confidence_cap_value is None"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="abc1234567890def",
            ts_ns=_TS_NS,
            symbol="BTC-USD",
            side=Side.BUY,
            final_confidence=0.4,
            plugin_chain=("test",),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
            why=None,
            signal_trust=None,
            signal_source=None,
            validation_score=None,
            original_confidence=0.9,
            confidence_cap_applied=True,
            confidence_cap_value=None,
        )


def test_decision_trace_rejects_out_of_range_original() -> None:
    sig = _signal(confidence=0.4, trust=SignalTrust.EXTERNAL_LOW)
    with pytest.raises(ValueError, match="original_confidence"):
        build_decision_trace(
            signal=sig,
            original_confidence=1.5,
            confidence_cap_applied=True,
            confidence_cap_value=DEFAULT_LOW_CAP,
        )


def test_decision_trace_rejects_out_of_range_cap_value() -> None:
    sig = _signal(confidence=0.4, trust=SignalTrust.EXTERNAL_LOW)
    with pytest.raises(ValueError, match="confidence_cap_value"):
        build_decision_trace(
            signal=sig,
            original_confidence=0.5,
            confidence_cap_applied=False,
            confidence_cap_value=2.0,
        )


# -- JSON round-trip -------------------------------------------------


def test_round_trip_preserves_audit_fields() -> None:
    sig = _signal(
        confidence=DEFAULT_LOW_CAP,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )
    trace = build_decision_trace(
        signal=sig,
        original_confidence=0.9,
        confidence_cap_applied=True,
        confidence_cap_value=DEFAULT_LOW_CAP,
    )
    event = as_system_event(trace)
    rt = trace_from_system_event(event)
    assert rt.original_confidence == 0.9
    assert rt.confidence_cap_applied is True
    assert rt.confidence_cap_value == DEFAULT_LOW_CAP


def test_round_trip_preserves_unset_audit_fields() -> None:
    sig = _signal(confidence=0.4, trust=SignalTrust.INTERNAL)
    trace = build_decision_trace(signal=sig)
    rt = trace_from_system_event(as_system_event(trace))
    assert rt.original_confidence is None
    assert rt.confidence_cap_applied is False
    assert rt.confidence_cap_value is None


# -- apply_signal_trust_cap_with_audit -------------------------------


def test_audit_helper_internal_pass_through() -> None:
    sig = _signal(confidence=0.92, trust=SignalTrust.INTERNAL)
    out, audit = apply_signal_trust_cap_with_audit(sig)
    assert out is sig
    assert audit == ConfidenceCapAudit(
        original_confidence=0.92,
        cap_value=None,
        applied=False,
    )


def test_audit_helper_external_low_under_cap_not_applied() -> None:
    sig = _signal(confidence=0.3, trust=SignalTrust.EXTERNAL_LOW)
    out, audit = apply_signal_trust_cap_with_audit(sig)
    assert out is sig  # unchanged when no clamp happens
    assert audit.original_confidence == 0.3
    assert audit.cap_value == DEFAULT_LOW_CAP
    assert audit.applied is False


def test_audit_helper_external_low_at_cap_not_applied() -> None:
    sig = _signal(confidence=DEFAULT_LOW_CAP, trust=SignalTrust.EXTERNAL_LOW)
    out, audit = apply_signal_trust_cap_with_audit(sig)
    # Confidence already exactly at the cap -> applied=False (no clamp needed).
    assert out is sig
    assert audit.applied is False
    assert audit.cap_value == DEFAULT_LOW_CAP


def test_audit_helper_external_low_over_cap_applied() -> None:
    sig = _signal(confidence=0.9, trust=SignalTrust.EXTERNAL_LOW)
    out, audit = apply_signal_trust_cap_with_audit(sig)
    assert out.confidence == DEFAULT_LOW_CAP
    assert audit == ConfidenceCapAudit(
        original_confidence=0.9,
        cap_value=DEFAULT_LOW_CAP,
        applied=True,
    )


def test_audit_helper_external_med_over_cap_applied() -> None:
    sig = _signal(confidence=0.9, trust=SignalTrust.EXTERNAL_MED)
    out, audit = apply_signal_trust_cap_with_audit(sig)
    assert out.confidence == DEFAULT_MED_CAP
    assert audit == ConfidenceCapAudit(
        original_confidence=0.9,
        cap_value=DEFAULT_MED_CAP,
        applied=True,
    )


def test_audit_helper_with_promotion_store_widens_cap() -> None:
    """Paper-S6 promotion: cap rises from LOW to MED; original recorded."""
    store = SourceTrustPromotionStore()
    store.promote(
        source_id="tradingview.public",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="trial widen",
        ts_ns=_TS_NS,
    )
    sig = _signal(
        confidence=0.95,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )
    out, audit = apply_signal_trust_cap_with_audit(sig, promotion_store=store)
    assert out.confidence == DEFAULT_MED_CAP
    assert audit.original_confidence == 0.95
    assert audit.cap_value == DEFAULT_MED_CAP
    assert audit.applied is True


def test_audit_helper_with_registry_more_restrictive_row() -> None:
    """Paper-S5 fail-closed -- registry row tighter than class default wins."""
    registry = _registry(
        sources={
            "tradingview.public": ExternalSignalSource(
                source_id="tradingview.public",
                trust=SignalTrust.EXTERNAL_LOW,
                cap=0.3,  # tighter than DEFAULT_LOW_CAP
            ),
        }
    )
    sig = _signal(
        confidence=0.95,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )
    out, audit = apply_signal_trust_cap_with_audit(sig, registry=registry)
    assert out.confidence == 0.3
    assert audit.cap_value == 0.3
    assert audit.applied is True
    assert audit.original_confidence == 0.95


def test_audit_helper_idempotent_under_double_apply() -> None:
    """Re-running the helper on already-clamped output yields applied=False."""
    sig = _signal(confidence=0.95, trust=SignalTrust.EXTERNAL_LOW)
    once, audit_once = apply_signal_trust_cap_with_audit(sig)
    twice, audit_twice = apply_signal_trust_cap_with_audit(once)
    assert audit_once.applied is True
    assert audit_twice.applied is False
    assert twice is once


def test_apply_signal_trust_cap_compat_unchanged() -> None:
    """The thin wrapper still returns just the clamped signal (Paper-S5 API)."""
    sig = _signal(confidence=0.9, trust=SignalTrust.EXTERNAL_LOW)
    out = apply_signal_trust_cap(sig)
    assert out.confidence == DEFAULT_LOW_CAP


# -- builder integration --------------------------------------------


def test_build_decision_trace_with_audit_for_capped_external_low() -> None:
    """End-to-end: harness clamps signal, trace records the audit triplet."""
    sig = _signal(
        confidence=0.95,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )
    clamped, audit = apply_signal_trust_cap_with_audit(sig)
    trace = build_decision_trace(
        signal=clamped,
        original_confidence=audit.original_confidence,
        confidence_cap_applied=audit.applied,
        confidence_cap_value=audit.cap_value,
    )
    assert trace.final_confidence == DEFAULT_LOW_CAP
    assert trace.original_confidence == 0.95
    assert trace.confidence_cap_applied is True
    assert trace.confidence_cap_value == DEFAULT_LOW_CAP


def test_build_decision_trace_with_audit_for_internal_no_cap() -> None:
    sig = _signal(confidence=0.92, trust=SignalTrust.INTERNAL)
    clamped, audit = apply_signal_trust_cap_with_audit(sig)
    trace = build_decision_trace(
        signal=clamped,
        original_confidence=audit.original_confidence,
        confidence_cap_applied=audit.applied,
        confidence_cap_value=audit.cap_value,
    )
    assert trace.final_confidence == 0.92
    assert trace.original_confidence == 0.92
    assert trace.confidence_cap_applied is False
    assert trace.confidence_cap_value is None
