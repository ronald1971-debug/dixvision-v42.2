# Manifest delta — v3.6.1 (BEHAVIOR-P3 / hazard-governance hard coupling)

This delta extends the v3.6.0 manifest with the System → Governance
hard coupling layer. Hazard events from Dyon now *immediately* impact
the next risk snapshot the hot path consumes, without waiting for a
Governance Mode-FSM round trip. CRITICAL/HIGH events still escalate
through the existing emergency-LOCK path
(`EventClassifier._classify_hazard` → `StateTransitionManager`); the
new throttle layer **composes alongside** it rather than replacing it.

## New invariants

### INV-64 — Hazard throttle is pure / deterministic / no I/O

`compute_throttle(observations, now_ns, config)` and
`apply_throttle(snapshot, decision)` are pure functions. No clock
read, no PRNG, no I/O. Two replays return byte-identical output.

Stateful surface (`HazardObserver`) is bounded and deterministic
under any fixed sequence of `observe(...)` / `current_throttle(...)`
calls — same input sequence → same outputs.

## New safety rules

### SAFE-67 — Hazard throttle is monotonically restrictive

`apply_throttle` may only **tighten** the snapshot it projects:

* `halted` only goes `False → True`.
* `max_signal_confidence` only rises (`max(...)`).
* `max_position_qty` and every entry of `symbol_caps` are multiplied
  by `decision.qty_multiplier ∈ [0.0, 1.0]` — they only shrink (or
  stay unchanged when the multiplier is `1.0`).

`compute_throttle` aggregates active observations by taking the
strictest contribution per axis (`min` of qty multipliers, `max` of
confidence floors, `or` of block flags), so a noisier hazard set can
never silently relax a tighter constraint emitted earlier.

### SAFE-68 — CRITICAL/HIGH stays on the emergency-LOCK path

The throttle layer **does not replace** the existing CRITICAL/HIGH
escalation in `governance_engine.control_plane.event_classifier` /
`StateTransitionManager` — it composes alongside it. CRITICAL and
HIGH severities both:

1. Still drive the Governance Mode-FSM to `LOCKED` via the existing
   classifier (unchanged).
2. *Additionally* set `decision.block = True` in the throttle layer
   so the hot path halts on the very next snapshot, without waiting
   for the Mode FSM to round-trip.

LOW and MEDIUM hazards — which previously only audited a ledger row
— now apply a bounded qty multiplier and (for MEDIUM) raise the
signal confidence floor. INFO is a strict pass-through (multiplier
`1.0`, floor `0.0`, no block).

## New module surface

```
system_engine/
  coupling/
    __init__.py                     # public re-exports
    hazard_throttle.py              # policy + observer (pure)
    risk_snapshot_throttle.py       # apply_throttle (pure)

core/contracts/
  risk.py                           # RiskSnapshot relocated here so
                                    # both engines can read it without
                                    # crossing engine boundaries (B1)
```

`RiskSnapshot` was previously defined in
`execution_engine.hot_path.fast_execute`. Moving it under
`core.contracts` keeps the dependency arrows pointed at shared
contracts only — `execution_engine.hot_path` and
`system_engine.coupling` both import from `core.contracts.risk`.
The legacy export `from execution_engine.hot_path.fast_execute
import RiskSnapshot` continues to work (re-exported from the same
module's `__all__`).

## Default policy table

| Severity | qty multiplier | confidence floor | block | active window  |
|----------|---------------:|-----------------:|------:|---------------:|
| INFO     | 1.00           | 0.00             | no    |    60 s        |
| LOW      | 0.75           | 0.00             | no    |     2 min      |
| MEDIUM   | 0.50           | 0.60             | no    |     5 min      |
| HIGH     | 0.00           | 1.00             | yes   |    10 min      |
| CRITICAL | 0.00           | 1.00             | yes   |    10 min      |

`HazardCodeOverride` allows per-code overrides of any subset of
these fields (`qty_multiplier`, `confidence_floor`, `block`,
`active_window_ns`); each unset field falls back to the severity
default. There is no "unknown hazard" silent pass-through — every
observed hazard maps to its severity rule by construction.

## Wiring status

* Module + tests are landed (this PR). End-to-end test exercises:
  hazard observed → snapshot tightened → `FastExecutor.execute`
  rejects on the next tick.
* The runtime wire (bus subscription that feeds
  `HazardObserver.observe()` from the live `HazardEvent` stream
  and feeds the snapshot through `apply_throttle` before the hot
  path consumes it) is **not** yet wired in this PR — that's a
  separate small PR with its own audit surface, after this one is
  reviewed. The data path is fully in place; only the bus
  subscription seam is deferred.
* CRITICAL/HIGH emergency-LOCK path
  (`EventClassifier._classify_hazard` → `StateTransitionManager`)
  is unchanged.
