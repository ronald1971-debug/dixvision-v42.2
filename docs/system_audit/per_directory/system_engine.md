# system_engine/ — Dyon hazard sensors + SCVS (45 files)

## Purpose

Dyon: hazard monitoring. SCVS: source & consumption validation.
Together they form the system's immune layer. Outputs only
`HazardEvent` (consumed by `governance_engine`).

Sub-packages:

* `engine.py` (37 lines) — composer.
* `dyon/` — hazard sensors HAZ-01..HAZ-13, anomaly detector,
  memory-overflow guard, sandbox sentry.
* `scvs/` — source registry, consumption registry, runtime liveness
  FSM (INV-58), per-packet schema/staleness guard + AI validator
  + fallback audit (INV-59).
* `coupling/hazard_throttle.py` — System → Governance hard coupling
  (INV-64, SAFE-67/68).
* `coupling/hazard_throttle_adapter.py` — closes the chain (PR #139).
* `time_source.py` — `TimeAuthority` canonical hot-path API
  (T0-4 / B-CLOCK lint, PR #135).
* `kill_switch.py` — centralized primitive (SAFE-01, PR #136).
* `metrics/`, `risk/fast_risk_cache.py`, `circuit_breakers/`,
  `stateful/`, `bootstrap/`, `runtime/` — supporting subsystems.
* `credentials/` — credential manifest + .env IO + status surface
  (PRs #70-72).
* `news_shock_sensor.py` — HAZ-NEWS-SHOCK (PR #119).

## Wiring

* `ui/server.py` instantiates the engine, registers SCVS validators
  on every external feed, and forwards `HazardEvent` to
  `governance_engine`.
* `B-CLOCK` lint forces every clock read through
  `system_engine.time_source.TimeAuthority`. Verified clean.

## Static-analysis result

* 45 files, 18 with findings — **all 18 are ruff-format drift only**.
* No orphan modules. No semantic findings.

## Deep-read observations

* `dyon/anomaly_detector.py` — INV-15 deterministic; uses ts_ns ^
  counter as the seed source.
* `scvs/runtime_liveness_fsm.py` — escalates critical-source
  flatlines to HAZ events.
* `coupling/hazard_throttle.py` — qty_multiplier + confidence_floor
  + block flag projected to a `ThrottleInfluence` consumed by the
  meta-controller.
* `news_shock_sensor.py` — fires HAZ-NEWS-SHOCK on a sentiment-z
  spike. Wired into `intelligence_engine/news/news_fanout.py`.

## Risks / gaps

* `system_engine/scvs/` does not yet enforce per-source confidence
  caps (a Paper-S5 deliverable). External signals currently flow
  through SCVS but the cap is a contract-field-only stub.
* No drift sentry on `external_signal_trust.yaml` itself; if an
  operator hand-edits the YAML to relax a cap, only the policy-hash
  anchor (PR #172) catches it. That is the intended chokepoint —
  no additional sentry needed.

## Verdict

**HEALTHY.** Dyon coverage is broad; SCVS is the right shape but
needs Paper-S5 wiring to be load-bearing for external signals.
