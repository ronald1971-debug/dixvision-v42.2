//! `PyO3` wrapper for `dixvision_execution::fast_execute`.
//!
//! This crate is intentionally a *thin shim*. It owns no logic — every
//! public function here unpacks Python primitives, calls the pure
//! Rust function, and packs the result back into a Python tuple.
//! The Python wrapper at
//! `execution_engine/hot_path/fast_execute.py` re-tags the tuple
//! onto the existing Python `HotPathOutcome` / `ExecutionEvent` /
//! `HotPathDecision` types.
//!
//! Keeping the seam this thin means:
//! * The audit surface is tiny — divergence between backends must
//!   come from the logic crate, not from this shim.
//! * The wheel is purely additive: callers that don't import it pay
//!   nothing.
//! * `extension-module` + `abi3-py310` produce a single wheel that
//!   works on any `CPython` 3.10..3.13.

#![forbid(unsafe_code)]
// PyO3 v0.22 macros expand into `pub(crate) struct …` inside the
// private file-level module. The `redundant_pub_crate` lint flags
// that because it can't see past the macro expansion. Suppress at
// module scope so future PyO3-generated functions don't have to
// repeat the allow.
#![allow(clippy::redundant_pub_crate)]

use dixvision_execution::fast_execute::{decide_gate, GateInputs};
use pyo3::prelude::*;

/// Evaluate the hot-path gate.
///
/// Mirrors the Python signature of the gate-decision portion of
/// `FastExecutor.execute`. Returns a tuple of
/// `(outcome_name, reason, price)` so the Python wrapper can
/// re-tag onto the existing `HotPathOutcome` `StrEnum` and build
/// the canonical `ExecutionEvent`.
///
/// All fundamentally-stateful work (the order-id counter, qty
/// fallback ladder, dataclass construction) stays Python-side.
#[pyfunction]
#[pyo3(signature = (
    *,
    signal_ts_ns,
    signal_confidence,
    signal_side,
    snapshot_version,
    snapshot_ts_ns,
    snapshot_halted,
    snapshot_max_signal_confidence,
    cap,
    mark_price,
    max_staleness_ns,
    qty,
))]
#[allow(clippy::too_many_arguments)]
fn decide_gate_py(
    signal_ts_ns: i64,
    signal_confidence: f64,
    signal_side: &str,
    snapshot_version: i64,
    snapshot_ts_ns: i64,
    snapshot_halted: bool,
    snapshot_max_signal_confidence: f64,
    cap: Option<f64>,
    mark_price: f64,
    max_staleness_ns: i64,
    qty: f64,
) -> (&'static str, &'static str, f64) {
    let inp = GateInputs {
        signal_ts_ns,
        signal_confidence,
        signal_side,
        snapshot_version,
        snapshot_ts_ns,
        snapshot_halted,
        snapshot_max_signal_confidence,
        cap,
        mark_price,
        max_staleness_ns,
        qty,
    };
    let d = decide_gate(&inp);
    (d.outcome.as_str(), d.reason, d.price)
}

/// Module entrypoint. Importable as `dixvision_py_execution`.
#[pymodule]
fn dixvision_py_execution(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decide_gate_py, m)?)?;
    // Expose the build-time crate version of the *logic* crate so
    // operator-side smoke tests can confirm that the wheel and the
    // Python wrapper are pinned to compatible audit-ledger reasons.
    m.add("LOGIC_CRATE_VERSION", dixvision_execution::CRATE_VERSION)?;
    Ok(())
}
