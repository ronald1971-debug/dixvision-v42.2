// ADAPTED FROM: PyO3/pyo3
//   pyo3/guide/src/class.md       -- #[pyclass] + #[pymethods]
//   pyo3/guide/src/function.md    -- #[pyfunction], #[pymodule]
//   pyo3/guide/src/types.md       -- GIL handling + Python<->Rust mapping
//   pyo3/guide/src/parallelism.md -- Python::allow_threads (release GIL)
//
// B-20 -- PATTERN_ONLY Rust hot-path template for FastRiskCache /
// FastExecutor (see execution_engine/hot_path/fast_execute.py +
// core/contracts/risk.py for the canonical Python contracts this
// template MUST match bit-for-bit before any revival PR ships).
//
// This file is a TEMPLATE, not a live backend.
//
//   * It compiles under `maturin build --release` inside
//     tools/rust_bridge/ but is NOT a member of any workspace and
//     is NOT imported by any Python module.
//   * `execution_engine.hot_path.fast_execute.FastExecutor` runs
//     Python-only (single backend) per Reviewer #3 (audit v3, item 1)
//     and `docs/rust_revival_schedule.yaml`.
//   * The Python-vs-Rust shadow-equivalence harness required by the
//     revival checklist (`tests/test_fast_execute_parity.py` or
//     successor) must be written and proven bit-identical BEFORE
//     this template lands in production.
//
// What the template demonstrates (per B-20 / I-38 spec):
//
//   1. Rust struct -> Python class via #[pyclass] + #[pymethods].
//   2. Rust function -> Python function via #[pyfunction].
//   3. Releasing the GIL for the pure-Rust hot-path gate via
//      `Python::allow_threads`.
//   4. Per-call -> per-snapshot caching shape that matches the
//      Python `RiskSnapshot` (version / ts_ns / max_position_qty /
//      max_signal_confidence / symbol_caps / halted + `cap_for`).
//   5. Reject ladder ordering identical to the Python
//      `_execute_python` branch order (halted -> stale -> no-mark ->
//      low-confidence -> hold -> qty-over-cap -> approved).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// HotPathOutcome -- mirrors core/contracts/events.HotPathOutcome (StrEnum)
// ---------------------------------------------------------------------------
//
// The Python side is a StrEnum, so we expose this as a plain `&'static str`
// constant table. The parity harness MUST assert the Rust string values
// match the Python `HotPathOutcome.value` exactly.

pub const OUTCOME_APPROVED: &str = "APPROVED";
pub const OUTCOME_REJECTED_RISK_STALE: &str = "REJECTED_RISK_STALE";
pub const OUTCOME_REJECTED_NO_MARK: &str = "REJECTED_NO_MARK";
pub const OUTCOME_REJECTED_LIMIT: &str = "REJECTED_LIMIT";
pub const OUTCOME_REJECTED_HOLD: &str = "REJECTED_HOLD";
pub const OUTCOME_REJECTED_LOW_CONFIDENCE: &str = "REJECTED_LOW_CONFIDENCE";

// ---------------------------------------------------------------------------
// RiskSnapshot -- mirrors core/contracts/risk.RiskSnapshot
// ---------------------------------------------------------------------------

/// Frozen view of the FastRiskCache for a single tick.
///
/// Python interface (`core.contracts.risk.RiskSnapshot`):
///
/// ```python
/// @dataclass(frozen=True, slots=True)
/// class RiskSnapshot:
///     version: int
///     ts_ns: int
///     max_position_qty: float | None = None
///     max_signal_confidence: float = 0.0
///     symbol_caps: dict[str, float] = field(default_factory=dict)
///     halted: bool = False
///
///     def cap_for(self, symbol: str) -> float | None: ...
/// ```
#[pyclass(name = "RustRiskSnapshot", frozen, module = "dixvision_rust_bridge")]
#[derive(Clone, Debug)]
pub struct RustRiskSnapshot {
    #[pyo3(get)]
    pub version: i64,
    #[pyo3(get)]
    pub ts_ns: i64,
    #[pyo3(get)]
    pub max_position_qty: Option<f64>,
    #[pyo3(get)]
    pub max_signal_confidence: f64,
    #[pyo3(get)]
    pub halted: bool,
    pub symbol_caps: HashMap<String, f64>,
}

#[pymethods]
impl RustRiskSnapshot {
    #[new]
    #[pyo3(signature = (
        version,
        ts_ns,
        max_position_qty = None,
        max_signal_confidence = 0.0,
        symbol_caps = None,
        halted = false,
    ))]
    fn new(
        version: i64,
        ts_ns: i64,
        max_position_qty: Option<f64>,
        max_signal_confidence: f64,
        symbol_caps: Option<HashMap<String, f64>>,
        halted: bool,
    ) -> PyResult<Self> {
        if version < 0 {
            return Err(PyValueError::new_err("version must be >= 0"));
        }
        Ok(Self {
            version,
            ts_ns,
            max_position_qty,
            max_signal_confidence,
            symbol_caps: symbol_caps.unwrap_or_default(),
            halted,
        })
    }

    /// Mirrors `RiskSnapshot.cap_for` -- per-symbol cap override
    /// falls back to `max_position_qty` when absent.
    fn cap_for(&self, symbol: &str) -> Option<f64> {
        if let Some(cap) = self.symbol_caps.get(symbol) {
            return Some(*cap);
        }
        self.max_position_qty
    }
}

// ---------------------------------------------------------------------------
// HotPathDecision -- mirrors execution_engine.hot_path.HotPathDecision
// ---------------------------------------------------------------------------

/// Result of one hot-path gate evaluation.
///
/// The eventual revival must also produce a typed `ExecutionEvent`;
/// for the template we expose just the gate verdict + the price /
/// qty / reason so the parity harness can reconstruct the full
/// Python `ExecutionEvent` on the Python side without crossing the
/// GIL twice per signal.
#[pyclass(name = "RustHotPathDecision", frozen, module = "dixvision_rust_bridge")]
#[derive(Clone, Debug)]
pub struct RustHotPathDecision {
    #[pyo3(get)]
    pub outcome: String,
    #[pyo3(get)]
    pub risk_version: i64,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub qty: f64,
    #[pyo3(get)]
    pub reason: String,
}

// ---------------------------------------------------------------------------
// FastExecutor gate -- pure function over (snapshot, signal-tuple).
// ---------------------------------------------------------------------------
//
// We expose the gate as a free function rather than a stateful
// class so the GIL can be released around the actual decision. The
// Python wrapper (post-revival) holds the order counter; the Rust
// side is stateless and replay-deterministic.
//
// Branch order MUST match `_execute_python` in
// execution_engine/hot_path/fast_execute.py exactly:
//
//   1. halted             -> REJECTED_LIMIT  (reason="halted")
//   2. ts_ns staleness    -> REJECTED_RISK_STALE
//   3. mark_price <= 0.0  -> REJECTED_NO_MARK
//   4. confidence floor   -> REJECTED_LOW_CONFIDENCE
//   5. side == HOLD       -> REJECTED_HOLD
//   6. qty > cap          -> REJECTED_LIMIT  (reason="qty_above_cap")
//   7. otherwise          -> APPROVED

/// Pure Rust gate. Side is 0 = HOLD, 1 = BUY, -1 = SELL (matches
/// `core.contracts.events.Side` ordinal; the Python wrapper does
/// the enum -> int translation before crossing the GIL boundary).
#[allow(clippy::too_many_arguments)]
fn fast_execute_gate(
    snapshot: &RustRiskSnapshot,
    signal_ts_ns: i64,
    signal_symbol: &str,
    signal_side: i32,
    signal_confidence: f64,
    signal_qty: f64,
    mark_price: f64,
    max_staleness_ns: i64,
) -> RustHotPathDecision {
    if snapshot.halted {
        return RustHotPathDecision {
            outcome: OUTCOME_REJECTED_LIMIT.to_owned(),
            risk_version: snapshot.version,
            price: if mark_price > 0.0 { mark_price } else { 0.0 },
            qty: 0.0,
            reason: "halted".to_owned(),
        };
    }

    if signal_ts_ns - snapshot.ts_ns > max_staleness_ns {
        return RustHotPathDecision {
            outcome: OUTCOME_REJECTED_RISK_STALE.to_owned(),
            risk_version: snapshot.version,
            price: 0.0,
            qty: 0.0,
            reason: "risk_stale".to_owned(),
        };
    }

    if mark_price <= 0.0 {
        return RustHotPathDecision {
            outcome: OUTCOME_REJECTED_NO_MARK.to_owned(),
            risk_version: snapshot.version,
            price: 0.0,
            qty: 0.0,
            reason: "no_mark".to_owned(),
        };
    }

    if signal_confidence < snapshot.max_signal_confidence {
        return RustHotPathDecision {
            outcome: OUTCOME_REJECTED_LOW_CONFIDENCE.to_owned(),
            risk_version: snapshot.version,
            price: mark_price,
            qty: 0.0,
            reason: "confidence_floor".to_owned(),
        };
    }

    if signal_side == 0 {
        return RustHotPathDecision {
            outcome: OUTCOME_REJECTED_HOLD.to_owned(),
            risk_version: snapshot.version,
            price: mark_price,
            qty: 0.0,
            reason: "hold_signal".to_owned(),
        };
    }

    if let Some(cap) = snapshot.cap_for(signal_symbol) {
        if signal_qty > cap {
            return RustHotPathDecision {
                outcome: OUTCOME_REJECTED_LIMIT.to_owned(),
                risk_version: snapshot.version,
                price: mark_price,
                qty: 0.0,
                reason: "qty_above_cap".to_owned(),
            };
        }
    }

    RustHotPathDecision {
        outcome: OUTCOME_APPROVED.to_owned(),
        risk_version: snapshot.version,
        price: mark_price,
        qty: signal_qty,
        reason: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Python entry point.
// ---------------------------------------------------------------------------
//
// `Python::allow_threads` drops the GIL for the duration of the gate
// call so a Python caller dispatching N signals across a thread
// pool benefits from the Rust hot path scaling linearly in cores.
// Per PyO3 guide "parallelism.md", anything inside `allow_threads`
// MUST NOT touch Python objects -- which is why we accept primitive
// args + clone the snapshot before crossing.

#[pyfunction]
#[pyo3(signature = (
    snapshot,
    signal_ts_ns,
    signal_symbol,
    signal_side,
    signal_confidence,
    signal_qty,
    mark_price,
    max_staleness_ns = 2_000_000_000_i64,
))]
fn execute(
    py: Python<'_>,
    snapshot: RustRiskSnapshot,
    signal_ts_ns: i64,
    signal_symbol: String,
    signal_side: i32,
    signal_confidence: f64,
    signal_qty: f64,
    mark_price: f64,
    max_staleness_ns: i64,
) -> PyResult<RustHotPathDecision> {
    if max_staleness_ns <= 0 {
        return Err(PyValueError::new_err("max_staleness_ns must be > 0"));
    }
    Ok(py.allow_threads(|| {
        fast_execute_gate(
            &snapshot,
            signal_ts_ns,
            &signal_symbol,
            signal_side,
            signal_confidence,
            signal_qty,
            mark_price,
            max_staleness_ns,
        )
    }))
}

// ---------------------------------------------------------------------------
// #[pymodule] -- maturin entry point.
// ---------------------------------------------------------------------------

#[pymodule]
fn dixvision_rust_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustRiskSnapshot>()?;
    m.add_class::<RustHotPathDecision>()?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add("__doc__", "B-20 PATTERN_ONLY Rust hot-path template.")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// In-crate unit tests -- run via `cargo test --manifest-path
// tools/rust_bridge/Cargo.toml --no-default-features`. These cover
// the *Rust* gate ordering; the *Python-vs-Rust shadow equivalence*
// proof lives in `tests/test_fast_execute_parity.py` and is a
// prerequisite for any revival PR.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_snapshot() -> RustRiskSnapshot {
        RustRiskSnapshot {
            version: 1,
            ts_ns: 1_000,
            max_position_qty: Some(10.0),
            max_signal_confidence: 0.5,
            symbol_caps: HashMap::new(),
            halted: false,
        }
    }

    #[test]
    fn approves_in_band_signal() {
        let snap = base_snapshot();
        let dec = fast_execute_gate(&snap, 1_000, "BTCUSD", 1, 0.9, 1.0, 100.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_APPROVED);
        assert_eq!(dec.risk_version, 1);
        assert_eq!(dec.price, 100.0);
        assert_eq!(dec.qty, 1.0);
    }

    #[test]
    fn rejects_when_halted_first() {
        let mut snap = base_snapshot();
        snap.halted = true;
        let dec = fast_execute_gate(&snap, 1_000, "BTCUSD", 1, 0.9, 1.0, 100.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_REJECTED_LIMIT);
        assert_eq!(dec.reason, "halted");
    }

    #[test]
    fn rejects_stale_snapshot_before_mark() {
        let snap = base_snapshot();
        // ts delta 3s > 2s staleness budget; mark_price also bad but
        // staleness MUST win first (branch order).
        let dec =
            fast_execute_gate(&snap, 4_000_000_000_i64, "BTCUSD", 1, 0.9, 1.0, -1.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_REJECTED_RISK_STALE);
    }

    #[test]
    fn rejects_no_mark() {
        let snap = base_snapshot();
        let dec = fast_execute_gate(&snap, 1_000, "BTCUSD", 1, 0.9, 1.0, 0.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_REJECTED_NO_MARK);
    }

    #[test]
    fn rejects_low_confidence() {
        let snap = base_snapshot();
        let dec = fast_execute_gate(&snap, 1_000, "BTCUSD", 1, 0.1, 1.0, 100.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_REJECTED_LOW_CONFIDENCE);
    }

    #[test]
    fn rejects_hold() {
        let snap = base_snapshot();
        let dec = fast_execute_gate(&snap, 1_000, "BTCUSD", 0, 0.9, 1.0, 100.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_REJECTED_HOLD);
    }

    #[test]
    fn rejects_qty_over_cap_with_symbol_override() {
        let mut snap = base_snapshot();
        snap.symbol_caps.insert("BTCUSD".to_owned(), 0.5);
        let dec = fast_execute_gate(&snap, 1_000, "BTCUSD", 1, 0.9, 1.0, 100.0, 2_000_000_000);
        assert_eq!(dec.outcome, OUTCOME_REJECTED_LIMIT);
        assert_eq!(dec.reason, "qty_above_cap");
    }

    #[test]
    fn cap_for_falls_back_to_global() {
        let snap = base_snapshot();
        assert_eq!(snap.cap_for("BTCUSD"), Some(10.0));
        assert_eq!(snap.cap_for("UNKNOWN"), Some(10.0));
    }

    #[test]
    fn cap_for_symbol_override_wins() {
        let mut snap = base_snapshot();
        snap.symbol_caps.insert("BTCUSD".to_owned(), 0.25);
        assert_eq!(snap.cap_for("BTCUSD"), Some(0.25));
        assert_eq!(snap.cap_for("ETHUSD"), Some(10.0));
    }
}
