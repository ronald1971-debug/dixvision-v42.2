//! DIX VISION v42.2 — `PyO3` seam over `dixvision-system`.
//!
//! This crate is the **only** place in the polyglot tree allowed to
//! bridge Python ↔ Rust. Every symbol exported here is a thin
//! wrapper around a function in `dixvision-system`; no business
//! logic lives on this side of the seam.
//!
//! # Contract
//! * Python callers import this module as `dixvision_py_system`
//!   (see `pyproject.toml`).
//! * Return values are plain tuples (of ints / floats / bools /
//!   strings) or ints — never Python class objects constructed here.
//!   Mapping to Python dataclasses (e.g. `system.time_source.TimeStamp`,
//!   `system.fast_risk_cache.RiskConstraints`) is the Python wrapper's
//!   responsibility so the FFI surface stays ABI-stable across
//!   refactors of either side.
//! * No `unsafe` except what `#[pymodule]` / `#[pyfunction]` emit
//!   through macro expansion — audited by `cargo expand` in the
//!   port-PR review.

#![forbid(unsafe_code)]
// The `pyo3` proc-macros (`#[pyfunction]`, `#[pymodule]`) expand
// into `pub(crate)` wrapper structs inside a generated private
// module; `clippy::redundant_pub_crate` fires on every macro call
// and we cannot fix it without forking pyo3. Audited once per
// pyo3 bump.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::redundant_pub_crate
)]

use std::sync::OnceLock;

use dixvision_system::{fast_risk_cache, time_source};
use pyo3::prelude::*;

// ---------------------------------------------------------------- time_source

/// Current timestamp from the process-wide singleton.
///
/// Returns `(utc_nanos, monotonic_ns, sequence)` as plain ints so
/// the caller can build a Python `TimeStamp` dataclass without
/// paying for a Rust ↔ Python class round-trip on the hot path.
///
/// `utc_nanos` and `monotonic_ns` are 128-bit in Rust to tolerate
/// long-lived processes; we narrow to `i64` here because nanoseconds
/// since the Unix epoch fit comfortably in `i64` until year 2262,
/// and `i64` maps to a native Python `int` with no allocation.
#[pyfunction]
fn now() -> (i64, i64, u64) {
    let t = time_source::now();
    (
        narrow_i128_to_i64(t.utc_nanos),
        narrow_i128_to_i64(t.monotonic_ns),
        t.sequence,
    )
}

/// Monotonic nanoseconds since process anchor. Hot-path convenience;
/// equivalent to `now()[1]` but avoids the 3-tuple allocation.
#[pyfunction]
fn now_mono_ns() -> i64 {
    narrow_i128_to_i64(time_source::now_mono_ns())
}

/// Crate-version pin — surfaced for the Python wrapper's version
/// handshake (`system/time_source.py` asserts a minimum).
#[pyfunction]
#[allow(clippy::missing_const_for_fn)] // `#[pyfunction]` requires a plain fn
fn crate_version() -> &'static str {
    dixvision_system::CRATE_VERSION
}

/// Clamp an `i128` into an `i64`. `time_source` values are produced
/// from `Instant::elapsed()` so cannot be negative, and cannot
/// exceed `i64::MAX` ns (~292 years) within any realistic process
/// lifetime. Clamp rather than panic so a hot-path call never
/// raises across the FFI seam.
fn narrow_i128_to_i64(v: i128) -> i64 {
    i64::try_from(v).unwrap_or(if v < 0 { i64::MIN } else { i64::MAX })
}

// ------------------------------------------------------------ fast_risk_cache

/// Tuple shape for a `RiskConstraints` snapshot crossing the FFI
/// seam. The order here is **load-bearing** — the Python wrapper
/// unpacks by position. Any change must be mirrored in
/// `system/fast_risk_cache.py::_from_rust_tuple` in the same PR.
///
/// Fields, in order:
///   0. `max_position_pct: float`
///   1. `max_order_size_usd: float`
///   2. `volatility_band_high: float`
///   3. `volatility_band_low: float`
///   4. `circuit_breaker_drawdown: float`
///   5. `circuit_breaker_loss_pct: float`
///   6. `trading_allowed: bool`
///   7. `safe_mode: bool`
///   8. `last_updated_utc: str` (ISO-8601 UTC)
type RiskTuple = (f64, f64, f64, f64, f64, f64, bool, bool, String);

/// Patch-tuple accepted by `risk_update`. Same field order as
/// `RiskTuple`, each value wrapped in `Option` so the Python
/// wrapper can omit fields by passing `None`. Python keyword-only
/// marshalling happens on the Python side; the FFI seam takes
/// positional args for ABI stability.
#[allow(clippy::type_complexity)]
type RiskPatchTuple = (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<bool>,
    Option<bool>,
);

fn process_cache() -> &'static fast_risk_cache::FastRiskCache {
    static CACHE: OnceLock<fast_risk_cache::FastRiskCache> = OnceLock::new();
    CACHE.get_or_init(fast_risk_cache::FastRiskCache::new)
}

fn snapshot_to_tuple(c: &fast_risk_cache::RiskConstraints) -> RiskTuple {
    (
        c.max_position_pct,
        c.max_order_size_usd,
        c.volatility_band_high,
        c.volatility_band_low,
        c.circuit_breaker_drawdown,
        c.circuit_breaker_loss_pct,
        c.trading_allowed,
        c.safe_mode,
        c.last_updated_utc.clone(),
    )
}

/// Lock-free read of the process-wide risk cache.
#[pyfunction]
fn risk_get() -> RiskTuple {
    let snap = process_cache().get();
    snapshot_to_tuple(&snap)
}

/// Apply an 8-field patch (`None` = leave unchanged) and return the
/// new snapshot atomically.
#[pyfunction]
fn risk_update(patch: RiskPatchTuple) -> RiskTuple {
    let up = fast_risk_cache::RiskUpdate {
        max_position_pct: patch.0,
        max_order_size_usd: patch.1,
        volatility_band_high: patch.2,
        volatility_band_low: patch.3,
        circuit_breaker_drawdown: patch.4,
        circuit_breaker_loss_pct: patch.5,
        trading_allowed: patch.6,
        safe_mode: patch.7,
    };
    let snap = process_cache().update(&up);
    snapshot_to_tuple(&snap)
}

/// Enter safe mode (trading halted, safe_mode = true). Returns new
/// snapshot.
#[pyfunction]
fn risk_enter_safe_mode() -> RiskTuple {
    let snap = process_cache().enter_safe_mode();
    snapshot_to_tuple(&snap)
}

/// Exit safe mode (trading resumed, safe_mode = false). Returns new
/// snapshot.
#[pyfunction]
fn risk_exit_safe_mode() -> RiskTuple {
    let snap = process_cache().exit_safe_mode();
    snapshot_to_tuple(&snap)
}

/// Halt trading without entering safe mode. Returns new snapshot.
#[pyfunction]
fn risk_halt_trading() -> RiskTuple {
    let snap = process_cache().halt_trading();
    snapshot_to_tuple(&snap)
}

/// Resume trading (clears safe_mode as well). Returns new snapshot.
#[pyfunction]
fn risk_resume_trading() -> RiskTuple {
    let snap = process_cache().resume_trading();
    snapshot_to_tuple(&snap)
}

// ---------------------------------------------------------------- pymodule

/// `#[pymodule]` entry point. Module name MUST match
/// `[tool.maturin].module-name` in `pyproject.toml` and the `[lib].name`
/// in `Cargo.toml`, otherwise `import dixvision_py_system` fails at
/// load time with a cryptic "dynamic module does not define module
/// export function" — document this so the next diff that renames
/// one of the three keeps the other two in lock-step.
#[pymodule]
fn dixvision_py_system(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(now, m)?)?;
    m.add_function(wrap_pyfunction!(now_mono_ns, m)?)?;
    m.add_function(wrap_pyfunction!(crate_version, m)?)?;
    m.add_function(wrap_pyfunction!(risk_get, m)?)?;
    m.add_function(wrap_pyfunction!(risk_update, m)?)?;
    m.add_function(wrap_pyfunction!(risk_enter_safe_mode, m)?)?;
    m.add_function(wrap_pyfunction!(risk_exit_safe_mode, m)?)?;
    m.add_function(wrap_pyfunction!(risk_halt_trading, m)?)?;
    m.add_function(wrap_pyfunction!(risk_resume_trading, m)?)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn narrow_clamps_at_i64_bounds() {
        assert_eq!(narrow_i128_to_i64(0), 0);
        assert_eq!(narrow_i128_to_i64(i128::from(i64::MAX)), i64::MAX);
        assert_eq!(narrow_i128_to_i64(i128::from(i64::MIN)), i64::MIN);
        assert_eq!(narrow_i128_to_i64(i128::MAX), i64::MAX);
        assert_eq!(narrow_i128_to_i64(i128::MIN), i64::MIN);
    }

    // Pure-Rust callable smoke check — `#[pyfunction]` wrapper
    // compiles, `time_source::now()` reachable through the re-export.
    #[test]
    fn now_mono_ns_is_positive() {
        let v = time_source::now_mono_ns();
        assert!(v >= 0, "monotonic ns must be non-negative: {v}");
    }

    // Callable smoke for the fast_risk_cache seam: snapshot → tuple
    // round-trips, default shape is sane. We do NOT share the
    // process-global `CACHE` across tests (test order is
    // unpredictable), so assert on shape only.
    #[test]
    fn snapshot_tuple_has_nine_fields() {
        let c = fast_risk_cache::RiskConstraints::default();
        let t = snapshot_to_tuple(&c);
        // Verify trailing string and the two bool slots.
        assert!(t.6); // trading_allowed default true
        assert!(!t.7); // safe_mode default false
        assert_eq!(t.8, "");
    }
}
