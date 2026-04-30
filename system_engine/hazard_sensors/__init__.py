"""Hazard sensor pack (Phase 4 / Build Compiler Spec §2 — Dyon).

The 12 hazard sensors are the only cross-domain emitters of
:class:`HazardEvent` (EVT-04). All sensors are pure-Python, IO-free,
clock-free (caller passes ``ts_ns``), and deterministic (INV-15).

HAZ codes (frozen)::

  HAZ-01  WS_TIMEOUT             — websocket feed silent past tolerance
  HAZ-02  EXCHANGE_UNREACHABLE   — adapter cannot reach venue
  HAZ-03  CLOCK_DRIFT            — TimeAuthority drift exceeds tolerance
  HAZ-04  STALE_DATA             — quote feed gap exceeds bar_window_ns
  HAZ-05  MEMORY_OVERFLOW        — RSS / heap budget breached
  HAZ-06  LATENCY_SPIKE          — round-trip exceeds latency budget
  HAZ-07  HEARTBEAT_MISSED       — engine heartbeat absent
  HAZ-08  RISK_SNAPSHOT_STALE    — fast risk cache version unchanged too long
  HAZ-09  ORDER_FLOOD            — order rate breaches per-window cap
  HAZ-10  CIRCUIT_BREAKER_OPEN   — runtime monitor opened venue/global breaker
  HAZ-11  MARKET_ANOMALY         — price/spread anomaly (statistical)
  HAZ-12  SYSTEM_ANOMALY         — process / cpu / fd resource anomaly

Cross-engine boundary: sensors live inside ``system_engine`` (Dyon
domain) and emit :class:`HazardEvent` only. Governance is the sole
consumer that may act on a hazard (INV-08, INV-11). No sensor imports
across engine boundaries.
"""

from system_engine.hazard_sensors.clock_drift import ClockDriftSensor
from system_engine.hazard_sensors.exchange_unreachable import (
    ExchangeUnreachableSensor,
)
from system_engine.hazard_sensors.heartbeat_missed import (
    HeartbeatMissedSensor,
)
from system_engine.hazard_sensors.latency_spike import LatencySpikeSensor
from system_engine.hazard_sensors.market_anomaly import MarketAnomalySensor
from system_engine.hazard_sensors.memory_overflow import MemoryOverflowSensor
from system_engine.hazard_sensors.news_shock import (
    NEWS_SHOCK_VERSION,
    NewsShockSensor,
)
from system_engine.hazard_sensors.order_flood import OrderFloodSensor
from system_engine.hazard_sensors.risk_snapshot_stale import (
    RiskSnapshotStaleSensor,
)
from system_engine.hazard_sensors.runtime_breaker_open import (
    RuntimeBreakerOpenSensor,
)
from system_engine.hazard_sensors.sensor_array import HazardSensor, SensorArray
from system_engine.hazard_sensors.stale_data import StaleDataSensor
from system_engine.hazard_sensors.system_anomaly import SystemAnomalySensor
from system_engine.hazard_sensors.ws_timeout import WSTimeoutSensor

__all__ = [
    "NEWS_SHOCK_VERSION",
    "ClockDriftSensor",
    "ExchangeUnreachableSensor",
    "HazardSensor",
    "HeartbeatMissedSensor",
    "LatencySpikeSensor",
    "MarketAnomalySensor",
    "MemoryOverflowSensor",
    "NewsShockSensor",
    "OrderFloodSensor",
    "RiskSnapshotStaleSensor",
    "RuntimeBreakerOpenSensor",
    "SensorArray",
    "StaleDataSensor",
    "SystemAnomalySensor",
    "WSTimeoutSensor",
]
