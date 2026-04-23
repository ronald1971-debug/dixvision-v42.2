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
//! * Return values are plain integer tuples or ints — no Python
//!   class objects constructed here. Mapping to Python dataclasses
//!   (e.g. `system.time_source.TimeStamp`) is the Python wrapper's
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

use dixvision_system::time_source;
use pyo3::prelude::*;

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
}
