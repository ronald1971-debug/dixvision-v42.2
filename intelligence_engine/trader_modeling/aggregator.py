"""Pure factory for :class:`TraderObservation` (Wave-04 PR-2 / B29).

This module is the **single runtime construction point** for
:class:`core.contracts.trader_intelligence.TraderObservation` records.
External trader-feed adapters (e.g.
:mod:`ui.feeds.tradingview_ideas`) parse raw payloads into
:class:`TraderModel` data, then call :func:`make_trader_observation` —
they do **not** construct the bus-transport record themselves. B29
lint enforces this; this factory enforces the *shape* (validation,
default propagation, kind discriminator).

INV-15: pure function of inputs. Caller-supplied ``ts_ns`` from the
TimeAuthority (T0-04) — never read from a system clock.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.contracts.trader_intelligence import (
    TRADER_OBSERVATION_PROFILE_UPDATE,
    TRADER_OBSERVATION_SIGNAL_OBSERVED,
    TraderModel,
    TraderObservation,
)

#: Canonical SystemEvent ``source`` tag for trader-modeling observations.
#: Mirrors :data:`evolution_engine.patch_pipeline.events.PATCH_EVENT_SOURCE_PROPOSAL`
#: in shape — a fully-qualified subsystem path so ledger consumers can
#: filter rows without unpacking the payload.
TRADER_MODELING_SOURCE: str = "intelligence_engine.trader_modeling"

#: Closed set of legal observation kinds. Adapters that pass anything
#: else fail-fast at the factory boundary so the typed bus never sees
#: an unknown discriminator value.
_LEGAL_OBSERVATION_KINDS: frozenset[str] = frozenset(
    {
        TRADER_OBSERVATION_PROFILE_UPDATE,
        TRADER_OBSERVATION_SIGNAL_OBSERVED,
    }
)


def make_trader_observation(
    *,
    ts_ns: int,
    model: TraderModel,
    observation_kind: str = TRADER_OBSERVATION_PROFILE_UPDATE,
    meta: Mapping[str, str] | None = None,
) -> TraderObservation:
    """Construct a :class:`TraderObservation` from a validated model.

    Args:
        ts_ns: Monotonic timestamp in nanoseconds. Caller (the adapter)
            supplies this from the TimeAuthority; the factory does not
            read any clock. Must be non-negative.
        model: The current :class:`TraderModel` snapshot. Its
            ``trader_id`` is mirrored onto the observation top-level
            so dispatch / filtering does not have to unpack the model.
        observation_kind: One of ``"PROFILE_UPDATE"`` /
            ``"SIGNAL_OBSERVED"`` (sentinels exported from
            :mod:`core.contracts.trader_intelligence`).
        meta: Optional free-form structural metadata. Stored
            verbatim; consumers must treat unknown keys as advisory.

    Returns:
        A fully-formed :class:`TraderObservation`.

    Raises:
        ValueError: ``ts_ns`` is negative, ``model.trader_id`` is
            empty, ``model.source_feed`` is empty, or
            ``observation_kind`` is not in the legal set.
    """

    if ts_ns < 0:
        raise ValueError(
            "make_trader_observation: ts_ns must be non-negative"
        )
    if not model.trader_id:
        raise ValueError(
            "make_trader_observation: model.trader_id must be non-empty"
        )
    if not model.source_feed:
        raise ValueError(
            "make_trader_observation: model.source_feed must be non-empty "
            "— mirror the SCVS source row id (e.g. SRC-TRADER-TRADINGVIEW-001)"
        )
    if observation_kind not in _LEGAL_OBSERVATION_KINDS:
        raise ValueError(
            "make_trader_observation: observation_kind must be one of "
            f"{sorted(_LEGAL_OBSERVATION_KINDS)}; got {observation_kind!r}"
        )
    return TraderObservation(
        ts_ns=ts_ns,
        trader_id=model.trader_id,
        observation_kind=observation_kind,
        model=model,
        meta=dict(meta) if meta is not None else {},
    )


__all__ = [
    "TRADER_MODELING_SOURCE",
    "make_trader_observation",
]
