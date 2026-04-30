"""Wave-04 PR-2 — Trader-Intelligence subsystem (B29-allowed producer).

This package is the **only** runtime location that may construct
:class:`core.contracts.trader_intelligence.TraderObservation` records
(per lint rule B29 in :mod:`tools.authority_lint`). The PR-1 contract
file defines the dataclass; this package is the producer authority.

Structure:

* :mod:`intelligence_engine.trader_modeling.aggregator` — pure factory
  ``make_trader_observation(...)`` that adapters call once they have a
  validated :class:`TraderModel` payload. The factory is the
  bottleneck through which every external trader-feed adapter (e.g.
  :mod:`ui.feeds.tradingview_ideas`) must pass.
* :mod:`intelligence_engine.trader_modeling.observation` — pure
  ``observation_as_system_event`` / ``observation_from_system_event``
  projection helpers (round-trip parity, INV-15 deterministic JSON).

Authority symmetry:

* B29 lint  — only ``intelligence_engine.trader_modeling.*`` and
  :mod:`core.contracts.trader_intelligence` may construct
  ``TraderObservation``.
* HARDEN-03 — emitted ``SystemEvent`` rows carry
  ``produced_by_engine="intelligence_engine"`` so receivers can call
  :func:`core.contracts.event_provenance.assert_event_provenance`.
"""

from __future__ import annotations

from intelligence_engine.trader_modeling.aggregator import (
    TRADER_MODELING_SOURCE,
    make_trader_observation,
)
from intelligence_engine.trader_modeling.observation import (
    OBSERVATION_EVENT_VERSION,
    observation_as_system_event,
    observation_from_system_event,
)

__all__ = [
    "OBSERVATION_EVENT_VERSION",
    "TRADER_MODELING_SOURCE",
    "make_trader_observation",
    "observation_as_system_event",
    "observation_from_system_event",
]
