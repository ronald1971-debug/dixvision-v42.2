"""Paper-S6 -- operator overlay for source-trust promotions.

Paper-S5 wired :func:`apply_signal_trust_cap` into
:func:`approve_signal_for_execution` so every external SignalEvent's
``confidence`` is clamped to a per-source cap. Paper-S6 layers an
in-memory operator overlay on top -- the harness can promote a source
from ``EXTERNAL_LOW`` to ``EXTERNAL_MED`` without redeploying the YAML
registry. The promotion is recorded on the authority ledger so it
survives restarts via boot-time replay.

These tests pin:

* :class:`SourceTrustPromotionStore` semantics (effective_trust,
  promote/demote idempotence, fail-closed promotion target).
* The harness approver consults the store at cap-application time
  (an EXTERNAL_LOW source that was promoted is clamped at the
  ``EXTERNAL_MED`` default of 0.7 instead of the LOW default of 0.5).
* Per-source YAML caps still win when more restrictive than the
  promoted class default (fail-closed -- promotion only widens, it
  never bypasses an explicit per-source pin).
* The boot-time replay function rebuilds the overlay from the
  authority ledger so promotions survive restarts.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

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
from core.contracts.source_trust_promotions import (
    DEMOTION_LEDGER_KIND,
    PROMOTION_LEDGER_KIND,
    SourceTrustPromotion,
    SourceTrustPromotionStore,
    is_promotable_target,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
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
    source: str,
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


# -- SourceTrustPromotionStore semantics ----------------------------


def test_store_promote_records_overlay_row() -> None:
    store = SourceTrustPromotionStore()

    record = store.promote(
        source_id="tradingview.public",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="trial widen",
        ts_ns=_TS_NS,
    )

    assert isinstance(record, SourceTrustPromotion)
    assert record.source_id == "tradingview.public"
    assert record.target_trust is SignalTrust.EXTERNAL_MED
    assert store.is_promoted("tradingview.public") is True
    assert len(store) == 1


def test_store_promote_rejects_non_external_med_target() -> None:
    store = SourceTrustPromotionStore()

    for bad_target in (SignalTrust.INTERNAL, SignalTrust.EXTERNAL_LOW):
        with pytest.raises(ValueError):
            store.promote(
                source_id="trader.x",
                target_trust=bad_target,
                requestor="ops",
                reason="oops",
                ts_ns=_TS_NS,
            )


def test_store_promote_rejects_empty_source_id() -> None:
    store = SourceTrustPromotionStore()

    with pytest.raises(ValueError):
        store.promote(
            source_id="",
            target_trust=SignalTrust.EXTERNAL_MED,
            requestor="ops",
            reason="oops",
            ts_ns=_TS_NS,
        )


def test_store_demote_is_idempotent() -> None:
    store = SourceTrustPromotionStore()
    store.promote(
        source_id="tradingview.public",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="ok",
        ts_ns=_TS_NS,
    )

    first = store.demote("tradingview.public")
    assert first is not None
    second = store.demote("tradingview.public")
    assert second is None
    assert store.is_promoted("tradingview.public") is False


def test_store_effective_trust_passes_internal_through() -> None:
    store = SourceTrustPromotionStore()
    store.promote(
        source_id="intelligence.in_proc",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="should not promote",
        ts_ns=_TS_NS,
    )

    # INTERNAL signals are never promoted by overlay -- the in-process
    # intelligence path is canonical and operator decisions cannot
    # affect it.
    assert (
        store.effective_trust("intelligence.in_proc", SignalTrust.INTERNAL)
        is SignalTrust.INTERNAL
    )


def test_store_effective_trust_promotes_external_low_only() -> None:
    store = SourceTrustPromotionStore()
    store.promote(
        source_id="tradingview.public",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="ok",
        ts_ns=_TS_NS,
    )

    assert (
        store.effective_trust("tradingview.public", SignalTrust.EXTERNAL_LOW)
        is SignalTrust.EXTERNAL_MED
    )
    # A producer that already declared EXTERNAL_MED is untouched;
    # the overlay never demotes (fail-closed).
    assert (
        store.effective_trust("tradingview.public", SignalTrust.EXTERNAL_MED)
        is SignalTrust.EXTERNAL_MED
    )


def test_store_effective_trust_no_overlay_returns_declared() -> None:
    store = SourceTrustPromotionStore()

    assert (
        store.effective_trust("anyone", SignalTrust.EXTERNAL_LOW)
        is SignalTrust.EXTERNAL_LOW
    )
    assert (
        store.effective_trust("anyone", SignalTrust.INTERNAL)
        is SignalTrust.INTERNAL
    )


def test_is_promotable_target_only_accepts_external_med() -> None:
    assert is_promotable_target(SignalTrust.EXTERNAL_MED) is True
    assert is_promotable_target(SignalTrust.EXTERNAL_LOW) is False
    assert is_promotable_target(SignalTrust.INTERNAL) is False


# -- apply_signal_trust_cap with promotion overlay ------------------


def test_apply_cap_with_promotion_widens_external_low_default() -> None:
    """A promoted EXTERNAL_LOW source is clamped at the MED default."""

    store = SourceTrustPromotionStore()
    store.promote(
        source_id="tradingview.public",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="trial widen",
        ts_ns=_TS_NS,
    )
    sig = _signal(
        confidence=0.9,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )

    clamped = apply_signal_trust_cap(sig, promotion_store=store)

    # Without overlay this would be DEFAULT_LOW_CAP=0.5; with the
    # promotion the gate now uses DEFAULT_MED_CAP=0.7.
    assert clamped.confidence == DEFAULT_MED_CAP
    assert clamped.confidence > DEFAULT_LOW_CAP


def test_apply_cap_without_overlay_uses_declared_class() -> None:
    store = SourceTrustPromotionStore()
    sig = _signal(
        confidence=0.9,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )

    clamped = apply_signal_trust_cap(sig, promotion_store=store)

    assert clamped.confidence == DEFAULT_LOW_CAP


def test_apply_cap_promotion_respects_more_restrictive_yaml_pin() -> None:
    """Per-source YAML cap still wins when more restrictive than class default.

    The operator overlay can widen the *class* default but never
    bypass an explicit per-source pin. A YAML row pinning
    ``cap=0.4`` for an EXTERNAL_LOW source still clamps at 0.4
    even after the operator promotes the source to EXTERNAL_MED.
    """

    store = SourceTrustPromotionStore()
    store.promote(
        source_id="tradingview.public",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="trial widen",
        ts_ns=_TS_NS,
    )
    registry = _registry(
        sources={
            "tradingview.public": ExternalSignalSource(
                source_id="tradingview.public",
                trust=SignalTrust.EXTERNAL_LOW,
                cap=0.4,
            )
        }
    )
    sig = _signal(
        confidence=0.9,
        trust=SignalTrust.EXTERNAL_LOW,
        source="tradingview.public",
    )

    clamped = apply_signal_trust_cap(
        sig, registry=registry, promotion_store=store
    )

    assert clamped.confidence == 0.4


def test_apply_cap_internal_signal_ignores_overlay() -> None:
    store = SourceTrustPromotionStore()
    store.promote(
        source_id="intelligence.in_proc",
        target_trust=SignalTrust.EXTERNAL_MED,
        requestor="ops",
        reason="should not affect INTERNAL",
        ts_ns=_TS_NS,
    )
    sig = _signal(
        confidence=0.95,
        trust=SignalTrust.INTERNAL,
        source="intelligence.in_proc",
    )

    clamped = apply_signal_trust_cap(sig, promotion_store=store)

    # INTERNAL has no cap, so confidence passes through unchanged.
    assert clamped.confidence == 0.95


# -- approve_signal_for_execution wires the overlay -----------------


def test_approve_signal_threads_promotion_through_to_intent() -> None:
    """End-to-end: harness approval surfaces the widened cap on intent."""

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

    intent = approve_signal_for_execution(
        sig,
        ts_ns=_TS_NS,
        promotion_store=store,
    )

    # The intent's signal confidence is the clamped value; without
    # the overlay it would be DEFAULT_LOW_CAP=0.5, with it it is
    # the MED-class default 0.7.
    assert intent.signal.confidence == DEFAULT_MED_CAP


# -- boot-time replay rebuilds the overlay --------------------------


def test_replay_rebuilds_overlay_from_promotion_rows() -> None:
    from ui.server import _replay_source_trust_promotions

    ledger = LedgerAuthorityWriter()
    ledger.append(
        ts_ns=_TS_NS,
        kind=PROMOTION_LEDGER_KIND,
        payload={
            "source_id": "tradingview.public",
            "target_trust": SignalTrust.EXTERNAL_MED.value,
            "requestor": "ops",
            "reason": "trial",
            "ts_ns": str(_TS_NS),
        },
    )

    store = SourceTrustPromotionStore()
    _replay_source_trust_promotions(ledger_writer=ledger, store=store)

    assert store.is_promoted("tradingview.public") is True
    record = store.get("tradingview.public")
    assert record is not None
    assert record.target_trust is SignalTrust.EXTERNAL_MED
    assert record.requestor == "ops"
    assert record.reason == "trial"


def test_replay_applies_demotion_after_promotion() -> None:
    from ui.server import _replay_source_trust_promotions

    ledger = LedgerAuthorityWriter()
    ledger.append(
        ts_ns=_TS_NS,
        kind=PROMOTION_LEDGER_KIND,
        payload={
            "source_id": "tradingview.public",
            "target_trust": SignalTrust.EXTERNAL_MED.value,
            "requestor": "ops",
            "reason": "trial",
            "ts_ns": str(_TS_NS),
        },
    )
    ledger.append(
        ts_ns=_TS_NS + 1,
        kind=DEMOTION_LEDGER_KIND,
        payload={
            "source_id": "tradingview.public",
            "requestor": "ops",
            "reason": "trial ended",
            "ts_ns": str(_TS_NS + 1),
        },
    )

    store = SourceTrustPromotionStore()
    _replay_source_trust_promotions(ledger_writer=ledger, store=store)

    assert store.is_promoted("tradingview.public") is False
    assert len(store) == 0


def test_replay_skips_malformed_rows() -> None:
    """Replay is fail-soft per-row -- one bad payload cannot abort boot."""

    from ui.server import _replay_source_trust_promotions

    ledger = LedgerAuthorityWriter()
    ledger.append(
        ts_ns=_TS_NS,
        kind=PROMOTION_LEDGER_KIND,
        payload={
            "source_id": "good.source",
            "target_trust": SignalTrust.EXTERNAL_MED.value,
            "requestor": "ops",
            "reason": "ok",
            "ts_ns": str(_TS_NS),
        },
    )
    ledger.append(
        ts_ns=_TS_NS + 1,
        kind=PROMOTION_LEDGER_KIND,
        payload={
            "source_id": "bad.source",
            "target_trust": "NOT_A_REAL_TRUST_CLASS",
            "requestor": "ops",
            "reason": "bad",
            "ts_ns": str(_TS_NS + 1),
        },
    )
    ledger.append(
        ts_ns=_TS_NS + 2,
        kind=PROMOTION_LEDGER_KIND,
        payload={
            "source_id": "another.bad",
            "target_trust": SignalTrust.EXTERNAL_LOW.value,
            "requestor": "ops",
            "reason": "wrong target",
            "ts_ns": str(_TS_NS + 2),
        },
    )

    store = SourceTrustPromotionStore()
    _replay_source_trust_promotions(ledger_writer=ledger, store=store)

    assert store.is_promoted("good.source") is True
    assert store.is_promoted("bad.source") is False
    assert store.is_promoted("another.bad") is False


def test_replay_ignores_unrelated_ledger_kinds() -> None:
    """Only PROMOTION/DEMOTION rows mutate the overlay."""

    from ui.server import _replay_source_trust_promotions

    ledger = LedgerAuthorityWriter()
    ledger.append(
        ts_ns=_TS_NS,
        kind="MODE_TRANSITION",
        payload={"from": "SAFE", "to": "PAPER"},
    )
    ledger.append(
        ts_ns=_TS_NS + 1,
        kind="OPERATOR_SETTINGS_CHANGED",
        payload={"setting": "autonomy_mode"},
    )

    store = SourceTrustPromotionStore()
    _replay_source_trust_promotions(ledger_writer=ledger, store=store)

    assert len(store) == 0


# -- HTTP route integration tests -----------------------------------


import importlib  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    ui_server = importlib.import_module("ui.server")
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


def test_route_list_returns_registry_rows(client: TestClient) -> None:
    response = client.get("/api/operator/source-trust")
    assert response.status_code == 200
    body = response.json()
    assert "rows" in body
    assert isinstance(body["rows"], list)
    assert body["promotion_count"] == 0
    # Rows should be sorted by source_id deterministically.
    ids = [row["source_id"] for row in body["rows"]]
    assert ids == sorted(ids)


def test_route_promote_widens_effective_trust(client: TestClient) -> None:
    response = client.post(
        "/api/operator/source-trust/promote",
        json={
            "source_id": "operator.test.source",
            "target_trust": "EXTERNAL_MED",
            "requestor": "ops",
            "reason": "trial widen",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["promoted"] is True
    assert body["promoted_target_trust"] == "EXTERNAL_MED"
    assert body["effective_trust"] == "EXTERNAL_MED"
    assert body["ledger_kind"] == PROMOTION_LEDGER_KIND
    assert body["ledger_seq"] >= 0
    # The list endpoint now reflects the overlay.
    listed = client.get("/api/operator/source-trust").json()
    assert listed["promotion_count"] == 1


def test_route_promote_rejects_non_med_target(client: TestClient) -> None:
    response = client.post(
        "/api/operator/source-trust/promote",
        json={
            "source_id": "operator.test.source",
            "target_trust": "EXTERNAL_LOW",
            "requestor": "ops",
            "reason": "should fail",
        },
    )

    # Fail-closed BEFORE ledger row is written.
    assert response.status_code == 400


def test_route_promote_rejects_unknown_trust_class(client: TestClient) -> None:
    response = client.post(
        "/api/operator/source-trust/promote",
        json={
            "source_id": "operator.test.source",
            "target_trust": "NOT_A_REAL_CLASS",
            "requestor": "ops",
            "reason": "garbage",
        },
    )

    assert response.status_code == 400


def test_route_demote_reverts_overlay(client: TestClient) -> None:
    client.post(
        "/api/operator/source-trust/promote",
        json={
            "source_id": "operator.test.source",
            "target_trust": "EXTERNAL_MED",
            "requestor": "ops",
            "reason": "trial",
        },
    )

    response = client.post(
        "/api/operator/source-trust/demote",
        json={
            "source_id": "operator.test.source",
            "requestor": "ops",
            "reason": "rollback",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["promoted"] is False
    assert body["ledger_kind"] == DEMOTION_LEDGER_KIND


def test_route_demote_is_idempotent(client: TestClient) -> None:
    """Demoting a non-promoted source still writes the audit row."""

    response = client.post(
        "/api/operator/source-trust/demote",
        json={
            "source_id": "never.promoted",
            "requestor": "ops",
            "reason": "no-op rollback",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["promoted"] is False
    assert body["ledger_kind"] == DEMOTION_LEDGER_KIND
