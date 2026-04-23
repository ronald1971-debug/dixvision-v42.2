//! DIX VISION v42.2 — execution domain (Dyon).
//!
//! This crate is the **Rust target** for the polyglot migration of the
//! `execution/` domain described in
//! [`docs/DIX_VISION_v42_2_COMPILED.md`]. It starts empty on purpose —
//! each port PR (`adapters/base` → `chaos/chaos_engine` →
//! `hazard/*`, …) lands one module and deletes the corresponding
//! `.py` file in the same commit.
//!
//! # Architectural invariants enforced here
//!
//! * Dyon owns the SYSTEM domain and the execution hot path; this
//!   crate NEVER calls into Indira (`mind/`) directly.
//! * `SYSTEM_HAZARD` is the only cross-domain signal emitted from
//!   anything in this crate.
//! * No panics on the hot path (see workspace `clippy::panic =
//!   "deny"`). Errors propagate via `Result<_, ExecutionError>`.

#![forbid(unsafe_code)]

pub mod circuit_breaker;

pub use circuit_breaker::{
    BreakerConfig, BreakerState, CircuitBreaker, MonotonicClock, SystemClock,
};

use thiserror::Error;

/// Canonical error type for the execution domain.
///
/// Port PRs add new variants as adapters, chaos, and hazard modules
/// are migrated. Consumers must be prepared for new variants; the
/// enum is `#[non_exhaustive]` for that reason.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExecutionError {
    /// Placeholder while the crate is empty. Removed in the first
    /// real port PR.
    #[error("execution scaffolding: not yet implemented ({0})")]
    NotImplemented(&'static str),
}

/// Crate version pin. Bumped by port PRs that change the public API.
/// Exposed for the bootstrap crate's reachability smoke-test and for
/// the (future) metrics tag that annotates every execution event.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_version_parses_as_semver_major() {
        let major = CRATE_VERSION
            .split('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok());
        assert!(
            major.is_some(),
            "CRATE_VERSION must start with a numeric major: {CRATE_VERSION}"
        );
    }

    #[test]
    fn error_formatting_is_stable() {
        let err = ExecutionError::NotImplemented("binance-adapter");
        assert_eq!(
            err.to_string(),
            "execution scaffolding: not yet implemented (binance-adapter)"
        );
    }
}
