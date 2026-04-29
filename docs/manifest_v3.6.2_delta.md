# Manifest delta — v3.6.2 (BEHAVIOR-P4 / decision trace per trade)

This delta extends the v3.6.1 manifest with the per-decision audit
record. Every decision can now emit a structured
:class:`DecisionTrace` to the audit ledger that captures **why** the
decision happened — confidence breakdown, regime, pressure summary,
safety modifier, hazards active at decision time, throttle decision
applied, and (when known) the resulting execution outcome.

The trace is the data substrate the dashboard's Decision-Trace widget
(``DASH-04``) and the offline ``learning_engine`` calibrator both
read; nothing in this delta wires it into a runtime engine — that's
a separate small PR with its own audit surface, after this one is
reviewed.

## New invariants

### INV-65 — Decision trace is pure / deterministic / no I/O

Both the builder (:func:`core.coherence.decision_trace.build_decision_trace`)
and the serialiser (:func:`core.coherence.decision_trace.as_system_event`)
are pure functions of their inputs. ``trace_id`` is the first 16 hex
characters of ``sha256("{symbol}|{ts_ns}|{plugin_chain}")`` — same
inputs → same id, byte-identical across replays. JSON serialisation
uses ``sort_keys=True`` + ``separators=(",", ":")`` so the resulting
:class:`SystemEvent` payload is byte-identical too.

Reverse path (:func:`core.coherence.decision_trace.trace_from_system_event`)
is strict and lossless: round-tripping any built trace through
``as_system_event`` → ``trace_from_system_event`` returns an equal
:class:`DecisionTrace`.

## New module surface

```
core/contracts/
  decision_trace.py           # frozen dataclass contracts (data only)

core/coherence/
  decision_trace.py           # builder + serialiser + reverse path
                              # (pure, no clock / no PRNG / no I/O)
```

The new ``DECISION_TRACE`` :class:`SystemEventKind` value is the
**only** ledger surface added; existing event variants are unchanged.

### Contract types

| Type                       | Purpose                                       |
|----------------------------|-----------------------------------------------|
| ``ConfidenceContribution`` | One component (name, value, weight, weighted) of the confidence breakdown. |
| ``PressureSummary``        | 5-axis projection (perf / risk / drift / latency / uncertainty). |
| ``HazardInfluence``        | Single hazard active at decision time (code, severity, source, ts_ns). |
| ``ThrottleInfluence``      | The :class:`ThrottleDecision` applied to the snapshot, projected (block, qty\_multiplier, confidence\_floor, contributing\_codes). |
| ``ExecutionOutcome``       | The execution result observed for this signal (status, qty, price, venue, order\_id). |
| ``DecisionTrace``          | The per-decision audit record itself. |

All contract types are ``frozen=True`` + ``slots=True``, validate
ranges in ``__post_init__``, and live under ``core.contracts`` so any
engine can read them without violating B1.

### Builder discipline

The builder is dependency-free: it accepts only ``core.contracts``
types. Runtime engines that have richer state (e.g.
``MetaControllerOutput`` from ``intelligence_engine`` or
``ThrottleDecision`` from ``system_engine.coupling``) project that
state into the contract types **at the call site**, then hand the
projected values to :func:`build_decision_trace`. This keeps the
trace builder usable from any engine without crossing engine
boundaries (B1).

Validation:

* ``confidence_breakdown`` may be partial — its weighted-sum may be
  *less than* ``signal.confidence`` — but it may **never overstate**:
  the sum is rejected with ``ValueError`` if it exceeds
  ``signal.confidence + 1e-6``. This guarantees a downstream reader
  ("how much of this confidence did consensus alone explain?") is
  monotonically reading a non-overstated decomposition.
* All unit-interval fields are checked at construction.
* ``trace_id`` is recomputed on every build; it cannot be supplied by
  the caller.

## Wiring status

* Module + tests are landed (this PR). Round-trip + replay
  determinism are covered.
* The runtime emit path (orchestrator / executor calling
  :func:`build_decision_trace` and publishing the
  :class:`SystemEvent` to the ledger) is **not** wired in this PR —
  that's a separate small PR with its own audit surface, after this
  one is reviewed.
* The existing ``dashboard/control_plane/decision_trace.py``
  (``DASH-04``) presentation widget is unchanged in this PR; the
  Dashboard-2026 wave will replace its summary-string projection
  with the structured :class:`DecisionTrace` payload defined here.
