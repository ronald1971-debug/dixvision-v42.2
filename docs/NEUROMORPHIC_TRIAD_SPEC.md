# Neuromorphic Triad Spec (DIX VISION v42.2)

**Status**: specification locked; stub modules land in the Phase 0 PR; model implementation follows Phase 2 (market) / Phase 3 (system) / Phase 4 (governance).

## Locked rule

> Neuromorphic components may **observe, detect, and advise**. They may **never decide, execute, or modify system state**. Their outputs are **events**. Their models are **immutable at runtime**. Their **existence is audited**.

Encoded as axioms **N1..N8** in `immutable_core/neuromorphic_axioms.lean` and enforced by `authority_lint` rule **C2** (forbidden-call list).

## Authority split — nothing changes

| Layer | Decides | Neuromorphic role |
| --- | --- | --- |
| **Indira** | what trade to make | microstructure sensor (SPIKE_SIGNAL_EVENT) |
| **Dyon** | what to tell Governance about system health | anomaly sensor (SYSTEM_ANOMALY_EVENT → translates to SYSTEM_HAZARD_EVENT) |
| **Governance** | approve/reject/modify/halt | risk-acceleration sensor (RISK_SIGNAL_EVENT — advisory only) |
| **Operator** | override everything | — |

Neuromorphic is a **sensory layer** bolted onto the front of each. The deterministic decision path is untouched.

## Files

### 1. `mind/plugins/neuromorphic_signal.py` — Indira sensor

- **Inputs**: market microstructure stream (L2 book ticks, trade prints, order-flow imbalance, volume delta, realized vol estimator).
- **Detects**: volatility bursts, order-flow imbalance spikes, momentum ignition, liquidity shocks.
- **Output event** (`SPIKE_SIGNAL_EVENT`):
  ```json
  {
    "type": "VOLATILITY_SPIKE" | "OFI_SPIKE" | "MOMENTUM_IGNITION" | "LIQUIDITY_SHOCK",
    "intensity": 0.0..1.0,
    "direction": "UP" | "DOWN" | "NEUTRAL",
    "confidence": 0.0..1.0,
    "venue": "binance.btcusdt",
    "timestamp_utc": "...",
    "sequence": 12345
  }
  ```
- **Consumers**: Indira signal pipeline reads as one feature among many. SPIKE_SIGNAL_EVENT never directly triggers a trade.
- **Feature engineering**: returns, rolling vol, OFI (bid−ask weighted), volume delta, book imbalance — NOT raw price → spikes.
- **Temporal window**: ≥64-step sequence input to preserve SNN's temporal advantage (fixes the "stateless input window" flaw called out by the operator).

### 2. `execution/monitoring/neuromorphic_detector.py` — Dyon sensor

- **Inputs**: system telemetry (CPU%, RAM, event-tick rhythm, API latency histogram, heartbeat intervals).
- **Detects**: latency drift (accumulating, not spike), silent-data-failure (missing event rhythm), memory-leak gradient, event-rhythm-break (tick gap > threshold).
- **Output event** (`SYSTEM_ANOMALY_EVENT`):
  ```json
  {
    "type": "LATENCY_DRIFT" | "SILENT_DATA_FAILURE" | "MEMORY_PRESSURE_GRADIENT" | "EVENT_RHYTHM_BREAK",
    "severity": 0.0..1.0,
    "component": "binance_ws" | "governance.kernel" | "ledger" | ...,
    "window_seconds": 60,
    "timestamp_utc": "..."
  }
  ```
- **Dyon translates**: `SYSTEM_ANOMALY_EVENT(severity=0.87)` → `SYSTEM_HAZARD_EVENT(severity=HIGH|CRITICAL)` per `execution/hazard/severity_classifier.py`.
- **Dead-man (operator's critical addition)**:
  ```python
  def check_self(self) -> bool:
      return (time.monotonic() - self.last_emission) < (self.heartbeat_interval * 3)
  ```
  If `check_self()` returns False, the existing system dead-man trips → kill-switch → fail-closed. Tested in `tests/test_neuromorphic_deadman.py` (lands with Phase 3 implementation).

### 3. `governance/signals/neuromorphic_risk.py` — Governance sensor

- **Inputs**: rolling-window risk features (drawdown velocity, realised-vol expansion, per-strategy PnL dispersion, cross-asset correlation matrix).
- **Detects**: drawdown acceleration, vol regime shift, strategy instability, correlation breakdown.
- **Output event** (`RISK_SIGNAL_EVENT`):
  ```json
  {
    "type": "RISK_ACCELERATION" | "REGIME_SHIFT" | "STRATEGY_INSTABILITY" | "CORRELATION_BREAKDOWN",
    "severity": 0.0..1.0,
    "confidence": 0.0..1.0,
    "context": "drawdown_velocity" | "variance_expansion" | "sharpe_dispersion" | "cross_corr",
    "timestamp_utc": "..."
  }
  ```
- **Governance consumption** (advisory only): feeds ConstraintCompiler as an additional feature that can *tighten* limits but never loosen them. The final approve/reject decision is still a deterministic hard rule (per axiom N7):
  ```python
  if risk_signal.severity > policy.risk_tighten_threshold:
      constraints.max_order_size_usd *= 0.5
      constraints.max_position_pct *= 0.5
  # Decision is still deterministic given inputs — replayable.
  ```

## Flow summary

```
Market WS        →  NeuromorphicSignal    →  SPIKE_SIGNAL_EVENT     →  Indira strategy  →  Trade decision  →  Governance
System telemetry →  NeuromorphicDetector  →  SYSTEM_ANOMALY_EVENT   →  Dyon             →  SYSTEM_HAZARD   →  Governance
Risk features    →  NeuromorphicRisk      →  RISK_SIGNAL_EVENT      →  Governance (advisory)               →  Deterministic decision
```

## Non-negotiables (enforced)

- Every event emission writes a `LEDGER("NEUROMORPHIC", type, source, payload)` row (N4, tested).
- Every neuromorphic module is on the `authority_lint` forbidden-call list for: `governance.kernel.evaluate`, `mind.fast_execute.*`, `execution.engine.execute`, `security.operator.*`, any `fast_risk_cache` mutator, `core.registry.register` (N6, lints block the sandbox pipeline).
- Every detector carries a heartbeat + dead-man check (N5, tested).
- STDP and any online topology mutation are forbidden in prod (N8); retraining produces new weights → sandbox pipeline → two-person hardware-key gate loads them.
- Models are ONNX / pickle blobs hashed + signed, loaded read-only at boot (N3).

## Roadmap

- **Phase 0 (this PR)**: axioms + spec + three stub modules (no ML), authority_lint rule C2, unit tests proving each stub emits an event and nothing else.
- **Phase 2**: `neuromorphic_signal.py` — SNN backend (snntorch or spikingjelly), 64-step temporal window, offline-trained weights shipped as `models/neuromorphic/indira_signal.onnx`.
- **Phase 3**: `neuromorphic_detector.py` — lightweight LSM (Liquid State Machine) over telemetry; dead-man wired.
- **Phase 4**: `neuromorphic_risk.py` — rolling-window SNN over risk features; advisory output into ConstraintCompiler.

## Why this stays safe

1. No decision authority at any layer — enforced by authority_lint + unit tests.
2. All three layers emit events; governance/operator remains the decider.
3. Model weights frozen at boot; retraining is a sandbox-gated deploy.
4. Dead-man on every detector — silent detector failure cannot silently skip a hazard.
5. Full ledger audit — every spike is replayable.
6. Kill-switch, wallet policy, 4% drawdown floor, two-person gate — all untouched.
