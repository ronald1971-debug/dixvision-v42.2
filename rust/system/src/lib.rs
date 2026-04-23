//! DIX VISION v42.2 — system control plane.
//!
//! Target crate for the polyglot migration of the `system/` domain:
//! `fast_risk_cache`, `state_reconstructor`, `snapshots`, `time_source`,
//! `kill_switch`, `load_controller`, the `config/*` family, and
//! `metrics`. See [`docs/DIX_VISION_v42_2_COMPILED.md`] § SYSTEM.
//!
//! # Architectural invariants enforced here
//!
//! * Governance is the **sole writer** into any state this crate
//!   manages that crosses into MARKET. Reads are lock-free; writes
//!   never hold a lock across a projector apply.
//! * `FastRiskCache` is the only runtime interface from governance
//!   to Indira (`mind/`). That contract lives here once ported.
//! * No `unsafe` except documented FFI seams (`PyO3` glue).
//!   Workspace lint `unsafe_code = "deny"` pins this.

#![forbid(unsafe_code)]

use thiserror::Error;

pub mod fast_risk_cache;
pub mod metrics;
pub mod time_source;

pub use fast_risk_cache::{FastRiskCache, RiskConstraints, RiskUpdate, TradeVerdict};
pub use metrics::{MetricsSink, MetricsSnapshot};
pub use time_source::{TimeSource, TimeStamp};

/// Canonical error type for the system control plane.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SystemError {
    /// Retained for crates wired against the scaffold. Real errors
    /// are added per port PR (T0-4 `time_source` itself is pure —
    /// no fallible ops.).
    #[error("system scaffolding: not yet implemented ({0})")]
    NotImplemented(&'static str),
}

/// Crate version pin. Bumped by port PRs that change the public API.
/// Exposed for the bootstrap crate's reachability smoke-test and for
/// the (future) metrics tag stamped on every system event.
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
        let err = SystemError::NotImplemented("time_source");
        assert_eq!(
            err.to_string(),
            "system scaffolding: not yet implemented (time_source)"
        );
    }
}
