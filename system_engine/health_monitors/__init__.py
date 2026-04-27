"""Dyon health monitors (Phase 4).

Three pure-Python deterministic monitors:

* :class:`HeartbeatMonitor` — engines emit ``record(engine, ts_ns)``;
  monitor exposes the last-seen timestamp per engine.
* :class:`LivenessChecker` — given a per-engine threshold, classifies
  each engine as ALIVE / SUSPECT / DEAD.
* :class:`Watchdog` — composes a monotonic step counter with one bound;
  yields a single STALL signal once when bumps stop arriving.
"""

from system_engine.health_monitors.heartbeat import HeartbeatMonitor
from system_engine.health_monitors.liveness import (
    EngineLiveness,
    LivenessChecker,
    LivenessState,
)
from system_engine.health_monitors.watchdog import Watchdog

__all__ = [
    "EngineLiveness",
    "HeartbeatMonitor",
    "LivenessChecker",
    "LivenessState",
    "Watchdog",
]
