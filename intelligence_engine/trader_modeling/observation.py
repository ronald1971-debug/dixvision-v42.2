"""TraderObservation ↔ SystemEvent projection (Wave-04 PR-2 / INV-15).

Pure helpers that serialise a :class:`TraderObservation` into a
canonical :class:`SystemEvent` row for the audit ledger and reverse
the projection on replay. Mirrors the pattern in
:mod:`evolution_engine.patch_pipeline.events` (PATCH_PROPOSED, …).

JSON encoding uses ``sort_keys=True`` and ``separators=(",", ":")`` so
two replays of the same input emit byte-identical ledger rows
(INV-15). All numeric mappings are sorted by key before encoding;
string mappings likewise. ``ConvictionStyle`` / ``RiskAttitude`` /
``TimeHorizon`` enums are encoded as their string values so the
projection survives a Python-version bump that re-orders enum
internals.

Discipline (B1 / Triad Lock):

* Imports only :mod:`core.contracts` types and this package's
  contract — no runtime engine.
* Pure functions; no clocks, no PRNG, no I/O.
* Round-trips losslessly: ``observation_from_system_event(
  observation_as_system_event(obs)) == obs`` for every legal input.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from core.contracts.events import SystemEvent, SystemEventKind
from core.contracts.trader_intelligence import (
    ConvictionStyle,
    PhilosophyProfile,
    RiskAttitude,
    TimeHorizon,
    TraderModel,
    TraderObservation,
)
from intelligence_engine.trader_modeling.aggregator import (
    TRADER_MODELING_SOURCE,
    make_trader_observation,
)

#: Wire-format version of the SystemEvent payload body. Bump on any
#: structural change (added field, renamed key, …); consumers can then
#: gate behaviour on the version they read.
OBSERVATION_EVENT_VERSION: int = 1

#: HARDEN-03 / INV-69 — engine label that stamped the wrapping
#: ``SystemEvent``. ``intelligence_engine`` is already in the
#: ``SystemEvent`` producer set so receivers can call
#: :func:`core.contracts.event_provenance.assert_event_provenance` in
#: strict mode without changing the producer registry.
_PRODUCED_BY_ENGINE: str = "intelligence_engine"


def _sorted_str_map(m: Mapping[str, str]) -> dict[str, str]:
    """Project a string→string mapping to a key-sorted ``dict``.

    Replays must produce byte-identical JSON regardless of the host
    dict's iteration order, so every mapping field is rebuilt with
    sorted keys before ``json.dumps`` (INV-15).
    """
    return {k: m[k] for k in sorted(m)}


def _sorted_float_map(m: Mapping[str, float]) -> dict[str, float]:
    """Same as :func:`_sorted_str_map` for ``str → float`` maps."""
    return {k: float(m[k]) for k in sorted(m)}


def _philosophy_to_body(p: PhilosophyProfile) -> dict[str, Any]:
    """Project a :class:`PhilosophyProfile` to a JSON-friendly dict."""
    return {
        "trader_id": p.trader_id,
        "belief_system": _sorted_float_map(p.belief_system),
        "risk_attitude": p.risk_attitude.value,
        "time_horizon": p.time_horizon.value,
        "conviction_style": p.conviction_style.value,
        "market_view": _sorted_str_map(p.market_view),
        "decision_biases": _sorted_float_map(p.decision_biases),
    }


def _model_to_body(m: TraderModel) -> dict[str, Any]:
    """Project a :class:`TraderModel` to a JSON-friendly dict."""
    body: dict[str, Any] = {
        "trader_id": m.trader_id,
        "source_feed": m.source_feed,
        "strategy_signatures": list(m.strategy_signatures),
        "performance_metrics": _sorted_float_map(m.performance_metrics),
        "risk_profile": _sorted_float_map(m.risk_profile),
        "regime_performance": _sorted_float_map(m.regime_performance),
        "behavioral_bias": _sorted_float_map(m.behavioral_bias),
        "meta": _sorted_str_map(m.meta),
    }
    body["philosophy"] = (
        _philosophy_to_body(m.philosophy) if m.philosophy is not None else None
    )
    return body


def _philosophy_from_body(body: Mapping[str, Any]) -> PhilosophyProfile:
    """Reverse of :func:`_philosophy_to_body`."""
    return PhilosophyProfile(
        trader_id=str(body["trader_id"]),
        belief_system=dict(body.get("belief_system", {})),
        risk_attitude=RiskAttitude(body.get("risk_attitude", "UNKNOWN")),
        time_horizon=TimeHorizon(body.get("time_horizon", "UNKNOWN")),
        conviction_style=ConvictionStyle(
            body.get("conviction_style", "UNKNOWN")
        ),
        market_view=dict(body.get("market_view", {})),
        decision_biases=dict(body.get("decision_biases", {})),
    )


def _model_from_body(body: Mapping[str, Any]) -> TraderModel:
    """Reverse of :func:`_model_to_body`."""
    raw_philosophy = body.get("philosophy")
    return TraderModel(
        trader_id=str(body["trader_id"]),
        source_feed=str(body["source_feed"]),
        strategy_signatures=tuple(body.get("strategy_signatures", ())),
        performance_metrics=dict(body.get("performance_metrics", {})),
        risk_profile=dict(body.get("risk_profile", {})),
        regime_performance=dict(body.get("regime_performance", {})),
        behavioral_bias=dict(body.get("behavioral_bias", {})),
        philosophy=(
            _philosophy_from_body(raw_philosophy)
            if isinstance(raw_philosophy, Mapping)
            else None
        ),
        meta=dict(body.get("meta", {})),
    )


def observation_as_system_event(
    observation: TraderObservation,
    *,
    source: str = TRADER_MODELING_SOURCE,
    ts_ns_override: int | None = None,
) -> SystemEvent:
    """Project a :class:`TraderObservation` into a ``TRADER_OBSERVED`` event.

    The original ``observation.ts_ns`` is always preserved inside the
    JSON body so :func:`observation_from_system_event` reverses the
    projection faithfully. ``ts_ns_override`` is the same escape hatch
    :func:`evolution_engine.patch_pipeline.events.proposal_as_system_event`
    exposes — it lets an orchestrator stamp the *outer* event timestamp
    without mutating the inner observation timestamp.
    """
    if not source:
        raise ValueError(
            "observation_as_system_event: source must be non-empty"
        )
    if not observation.trader_id:
        raise ValueError(
            "observation_as_system_event: observation.trader_id must be "
            "non-empty"
        )
    if ts_ns_override is not None and ts_ns_override < 0:
        raise ValueError(
            "observation_as_system_event: ts_ns_override must be non-negative"
        )
    body = {
        "version": OBSERVATION_EVENT_VERSION,
        "ts_ns": observation.ts_ns,
        "trader_id": observation.trader_id,
        "observation_kind": observation.observation_kind,
        "model": _model_to_body(observation.model),
        "meta": _sorted_str_map(observation.meta),
    }
    payload = {
        "observation": json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ),
    }
    return SystemEvent(
        ts_ns=(
            ts_ns_override
            if ts_ns_override is not None
            else observation.ts_ns
        ),
        sub_kind=SystemEventKind.TRADER_OBSERVED,
        source=source,
        payload=payload,
        produced_by_engine=_PRODUCED_BY_ENGINE,
    )


def observation_from_system_event(event: SystemEvent) -> TraderObservation:
    """Reverse of :func:`observation_as_system_event` (replay parity).

    Goes through :func:`make_trader_observation` so B29 authority +
    field validation are enforced even on the replay path — i.e. a
    rogue caller that hand-builds a ``TRADER_OBSERVED`` event with
    invalid payload still gets caught at deserialise time.
    """
    if event.sub_kind is not SystemEventKind.TRADER_OBSERVED:
        raise ValueError(
            "observation_from_system_event: event must be TRADER_OBSERVED; "
            f"got {event.sub_kind}"
        )
    raw = event.payload.get("observation")
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            "observation_from_system_event: payload missing 'observation' "
            "string"
        )
    body = json.loads(raw)
    model = _model_from_body(body["model"])
    return make_trader_observation(
        ts_ns=int(body["ts_ns"]),
        model=model,
        observation_kind=str(body["observation_kind"]),
        meta=dict(body.get("meta", {})),
    )


__all__ = [
    "OBSERVATION_EVENT_VERSION",
    "observation_as_system_event",
    "observation_from_system_event",
]
