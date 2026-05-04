"""Paper-S5 -- per-source confidence cap enforcement at the harness gate.

The infrastructure shipped in Paper-S1 (``SignalTrust`` enum,
``ExternalSignalTrustRegistry``, ``clamp_confidence``,
``default_cap_for``) was inert until Paper-S5 wired
:func:`apply_signal_trust_cap` into
:func:`approve_signal_for_execution`. These tests pin the resulting
behaviour:

* INTERNAL signals pass through unchanged (no cap).
* EXTERNAL_LOW signals with no per-source registry entry are clamped
  to :data:`DEFAULT_LOW_CAP` (0.5).
* EXTERNAL_MED signals with no per-source registry entry are clamped
  to :data:`DEFAULT_MED_CAP` (0.7).
* A registry row's per-source ``cap`` overrides the trust-class
  default *only if* it is more restrictive than the class default
  (fail-closed against an over-permissive YAML row).
* A registry row whose declared trust class disagrees with the
  producer's declared trust takes the more-restrictive of the two
  (fail-closed against producer/registry disagreement).
* A signal already below the cap passes through unchanged (clamp is
  monotone; never amplifies).
* Calling :func:`apply_signal_trust_cap` twice is a no-op on the
  second call (idempotent).
"""

from __future__ import annotations

from types import MappingProxyType

from core.contracts.events import EventKind, SignalEvent, Side
from core.contracts.external_signal_trust import (
    ExternalSignalSource,
    ExternalSignalTrustRegistry,
)
from core.contracts.signal_trust import (
    DEFAULT_LOW_CAP,
    DEFAULT_MED_CAP,
    SignalTrust,
)
from governance_engine.harness_approver import (
    apply_signal_trust_cap,
    approve_signal_for_execution,
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


# -- apply_signal_trust_cap (pure helper) ---------------------------


def test_internal_signal_passes_through_unchanged() -> None:
    sig = _signal(confidence=0.92, trust=SignalTrust.INTERNAL)
    out = apply_signal_trust_cap(sig)
    assert out is sig
    assert out.confidence == 0.92


def test_external_low_no_registry_clamps_to_default_low_cap() -> None:
    sig = _signal(confidence=0.95, trust=SignalTrust.EXTERNAL_LOW)
    out = apply_signal_trust_cap(sig)
    assert out.confidence == DEFAULT_LOW_CAP
    assert out.signal_trust is SignalTrust.EXTERNAL_LOW


def test_external_med_no_registry_clamps_to_default_med_cap() -> None:
    sig = _signal(confidence=0.95, trust=SignalTrust.EXTERNAL_MED)
    out = apply_signal_trust_cap(sig)
    assert out.confidence == DEFAULT_MED_CAP
    assert out.signal_trust is SignalTrust.EXTERNAL_MED


def test_external_low_below_cap_passes_through() -> None:
    sig = _signal(confidence=0.3, trust=SignalTrust.EXTERNAL_LOW)
    out = apply_signal_trust_cap(sig)
    assert out is sig
    assert out.confidence == 0.3


def test_per_source_cap_more_restrictive_wins() -> None:
    registry = _registry(
        sources={
            "SRC-X": ExternalSignalSource(
                source_id="SRC-X",
                trust=SignalTrust.EXTERNAL_LOW,
                cap=0.3,
                note="strict",
            ),
        }
    )
    sig = _signal(
        confidence=0.95, trust=SignalTrust.EXTERNAL_LOW, source="SRC-X"
    )
    out = apply_signal_trust_cap(sig, registry=registry)
    assert out.confidence == 0.3


def test_per_source_cap_more_permissive_is_clamped_to_class_default() -> None:
    # Registry says cap=0.9 for an EXTERNAL_LOW source, but
    # DEFAULT_LOW_CAP is 0.5 -- the gate must take the more
    # restrictive of the two so a permissive YAML row cannot
    # promote external confidence beyond the class default.
    registry = _registry(
        sources={
            "SRC-Y": ExternalSignalSource(
                source_id="SRC-Y",
                trust=SignalTrust.EXTERNAL_LOW,
                cap=0.9,
                note="too permissive",
            ),
        }
    )
    sig = _signal(
        confidence=0.95, trust=SignalTrust.EXTERNAL_LOW, source="SRC-Y"
    )
    out = apply_signal_trust_cap(sig, registry=registry)
    assert out.confidence == DEFAULT_LOW_CAP


def test_registry_trust_disagreement_takes_more_restrictive() -> None:
    # Producer declares EXTERNAL_LOW but the registry row declares
    # EXTERNAL_MED with cap 0.6 -- the gate takes the more
    # restrictive of (0.6, DEFAULT_LOW_CAP=0.5) = 0.5.
    registry = _registry(
        sources={
            "SRC-Z": ExternalSignalSource(
                source_id="SRC-Z",
                trust=SignalTrust.EXTERNAL_MED,
                cap=0.6,
                note="trust class mismatch",
            ),
        }
    )
    sig = _signal(
        confidence=0.95, trust=SignalTrust.EXTERNAL_LOW, source="SRC-Z"
    )
    out = apply_signal_trust_cap(sig, registry=registry)
    assert out.confidence == DEFAULT_LOW_CAP


def test_unregistered_source_falls_back_to_class_default() -> None:
    registry = _registry(sources={})
    sig = _signal(
        confidence=0.95, trust=SignalTrust.EXTERNAL_LOW, source="SRC-UNKNOWN"
    )
    out = apply_signal_trust_cap(sig, registry=registry)
    assert out.confidence == DEFAULT_LOW_CAP


def test_apply_signal_trust_cap_is_idempotent() -> None:
    sig = _signal(confidence=0.95, trust=SignalTrust.EXTERNAL_LOW)
    once = apply_signal_trust_cap(sig)
    twice = apply_signal_trust_cap(once)
    assert once.confidence == twice.confidence == DEFAULT_LOW_CAP
    # Second call is a no-op (already at cap).
    assert twice is once


# -- approve_signal_for_execution end-to-end ----------------------


def test_approve_signal_clamps_external_low_into_intent() -> None:
    sig = _signal(confidence=0.95, trust=SignalTrust.EXTERNAL_LOW)
    intent = approve_signal_for_execution(sig, ts_ns=_TS_NS)
    # The intent's embedded signal carries the clamped confidence so
    # the downstream DecisionTrace's ``final_confidence`` reflects
    # the cap (Paper-S7 will surface the original via dedicated
    # trace fields).
    assert intent.signal.confidence == DEFAULT_LOW_CAP
    assert intent.signal.signal_trust is SignalTrust.EXTERNAL_LOW
    # Identity fields preserved.
    assert intent.signal.symbol == "BTC-USD"
    assert intent.signal.ts_ns == _TS_NS


def test_approve_signal_internal_passes_through_full_confidence() -> None:
    sig = _signal(confidence=0.92, trust=SignalTrust.INTERNAL)
    intent = approve_signal_for_execution(sig, ts_ns=_TS_NS)
    assert intent.signal.confidence == 0.92
    assert intent.signal is sig


def test_approve_signal_with_per_source_override() -> None:
    registry = _registry(
        sources={
            "SRC-PIN": ExternalSignalSource(
                source_id="SRC-PIN",
                trust=SignalTrust.EXTERNAL_LOW,
                cap=0.2,
                note="paper-s5 pinned cap",
            ),
        }
    )
    sig = _signal(
        confidence=0.95,
        trust=SignalTrust.EXTERNAL_LOW,
        source="SRC-PIN",
    )
    intent = approve_signal_for_execution(
        sig, ts_ns=_TS_NS, registry=registry
    )
    assert intent.signal.confidence == 0.2


def test_approve_signal_external_med_default_cap_into_intent() -> None:
    sig = _signal(confidence=0.95, trust=SignalTrust.EXTERNAL_MED)
    intent = approve_signal_for_execution(sig, ts_ns=_TS_NS)
    assert intent.signal.confidence == DEFAULT_MED_CAP
