//! DIX VISION v42.2 — deterministic kernel boot.
//!
//! Target crate for `bootstrap/` (`kernel_boot`, `system_init`,
//! `dependency_resolver`). See [`docs/DIX_VISION_v42_2_COMPILED.md`]
//! § BOOTSTRAP.
//!
//! The deterministic boot sequence (empty until the real port PR):
//! load latest snapshot → replay ledger delta → lock the component
//! registry → emit BOOT heartbeat → hand control to the event loop.
//! Registry-lock failure in production must trip the global kill
//! switch; that wire-up lands with [`dixvision_system`]'s
//! `kill_switch` module.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Canonical error type for the boot sequence.
///
/// Any variant MUST be actionable in prod: a failure at this layer
/// either (a) trips the global kill switch or (b) is a programming
/// bug that should never happen after green CI.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BootError {
    /// Placeholder while the crate is empty.
    #[error("bootstrap scaffolding: not yet implemented ({0})")]
    NotImplemented(&'static str),
}

/// Crate version pin. Bumped by boot-sequence PRs that change the
/// public API.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Re-export of the two domains the boot sequence orchestrates.
pub mod deps {
    pub use dixvision_execution as execution;
    pub use dixvision_system as system;
}

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
    fn system_and_execution_are_reachable_via_deps() {
        // Transitive reachability smoke-test. If either crate failed
        // to link, this wouldn't compile.
        let sys_major = deps::system::CRATE_VERSION.split('.').next();
        let exec_major = deps::execution::CRATE_VERSION.split('.').next();
        assert!(sys_major.is_some());
        assert!(exec_major.is_some());
    }
}
