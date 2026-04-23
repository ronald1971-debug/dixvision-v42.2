//! DIX VISION v42.2 — hazard severity classifier (port of
//! `execution/hazard/severity_classifier.py`).
//!
//! Dyon emits `SYSTEM_HAZARD_EVENT`s via `execution/hazard/async_bus`.
//! Governance's hazard consumer uses the four pure functions in this
//! module to decide what to do with each event:
//!
//! * [`should_halt_trading`] — escalate to full trading halt.
//! * [`should_enter_safe_mode`] — degrade to safe-mode posture.
//! * [`classify_severity`] — promote type-implied severities
//!   (data-corruption / ledger-inconsistency always reach CRITICAL).
//! * [`classify_response`] — recommended governance action string.
//!
//! # Why this module is a port target
//!
//! The classifier runs on every hazard event handled by governance.
//! It is pure — no I/O, no allocation on the hot path — so porting
//! it to Rust eliminates a Python function-call cost from a code path
//! that executes for every hazard, which during a network or feed
//! anomaly can be thousands of events per second.
//!
//! # Why it takes `&str` parameters, not `enum`s
//!
//! The canonical `HazardType` / `HazardSeverity` values are defined
//! in `contracts/governance.proto` (polyglot source of truth). The
//! Python reference implementation uses `str`-backed enums for that
//! reason. Crossing the `PyO3` seam with a rust enum would force an
//! extra string-interning round-trip; accepting `&str` directly
//! (validated against the known set via exhaustive `match`) keeps the
//! FFI cost at a pointer copy.

/// Known hazard-type identifiers.
///
/// Keep this in lock-step with `execution.hazard.async_bus.HazardType`
/// and `contracts/governance.proto::HazardType`. Unknown types are
/// accepted by the classifier functions (fall through to the default
/// branch) so a new variant added on the producer side does not
/// crash the consumer — it is merely classified conservatively
/// (non-critical, OBSERVE response).
pub const HAZARD_TYPES: &[&str] = &[
    "EXCHANGE_TIMEOUT",
    "FEED_SILENCE",
    "EXECUTION_LATENCY_SPIKE",
    "DATA_CORRUPTION_SUSPECTED",
    "SYSTEM_DEGRADATION",
    "API_CONNECTIVITY_FAILURE",
    "LEDGER_INCONSISTENCY",
    "MEMORY_PRESSURE",
    "CPU_OVERLOAD",
];

/// Known severity identifiers.
///
/// Keep this in lock-step with
/// `execution.hazard.async_bus.HazardSeverity` and
/// `contracts/governance.proto::HazardSeverity`.
pub const HAZARD_SEVERITIES: &[&str] = &["LOW", "MEDIUM", "HIGH", "CRITICAL"];

/// Whether a hazard event should trigger an immediate trading halt.
///
/// Matches `execution.hazard.severity_classifier.should_halt_trading`
/// exactly:
///
/// * Any `CRITICAL` severity halts trading, regardless of type.
/// * The following types halt trading regardless of severity:
///   `DATA_CORRUPTION_SUSPECTED`, `LEDGER_INCONSISTENCY`,
///   `API_CONNECTIVITY_FAILURE`.
#[must_use]
pub fn should_halt_trading(hazard_type: &str, severity: &str) -> bool {
    if severity == "CRITICAL" {
        return true;
    }
    matches!(
        hazard_type,
        "DATA_CORRUPTION_SUSPECTED" | "LEDGER_INCONSISTENCY" | "API_CONNECTIVITY_FAILURE"
    )
}

/// Whether a hazard event should trigger entry into safe mode.
///
/// Matches `execution.hazard.severity_classifier.should_enter_safe_mode`
/// exactly:
///
/// * `HIGH` or `CRITICAL` severities enter safe mode, regardless of
///   type.
/// * The following types enter safe mode regardless of severity:
///   `FEED_SILENCE`, `EXCHANGE_TIMEOUT`.
#[must_use]
pub fn should_enter_safe_mode(hazard_type: &str, severity: &str) -> bool {
    if severity == "HIGH" || severity == "CRITICAL" {
        return true;
    }
    matches!(hazard_type, "FEED_SILENCE" | "EXCHANGE_TIMEOUT")
}

/// Effective severity after the type-implied promotion rule.
///
/// Matches `execution.hazard.severity_classifier.classify_severity`
/// exactly: `DATA_CORRUPTION_SUSPECTED` and `LEDGER_INCONSISTENCY`
/// are always promoted to `CRITICAL`. Everything else returns the
/// severity as-is.
///
/// Returns an owned `String` rather than `&'static str` so the FFI
/// seam can hand it straight back to Python without a second
/// interning round-trip.
#[must_use]
pub fn classify_severity(hazard_type: &str, severity: &str) -> String {
    if matches!(
        hazard_type,
        "DATA_CORRUPTION_SUSPECTED" | "LEDGER_INCONSISTENCY"
    ) {
        return "CRITICAL".to_string();
    }
    severity.to_string()
}

/// Recommended governance response for a hazard type.
///
/// Matches `execution.hazard.severity_classifier.classify_response`
/// exactly. Unknown types fall through to `OBSERVE` (conservative
/// default — never halt trading on a type we do not understand).
#[must_use]
pub fn classify_response(hazard_type: &str) -> &'static str {
    match hazard_type {
        "EXCHANGE_TIMEOUT" => "CANCEL_ALL_OPEN_ORDERS",
        "FEED_SILENCE" => "PAUSE_NEW_ORDERS",
        "EXECUTION_LATENCY_SPIKE" => "REDUCE_EXPOSURE",
        "DATA_CORRUPTION_SUSPECTED" | "API_CONNECTIVITY_FAILURE" => "HALT_TRADING",
        _ => "OBSERVE",
    }
}

/// Return `true` if `hazard_type` is one of the documented variants.
///
/// See [`HAZARD_TYPES`] for the full list. The classifier functions
/// themselves are tolerant of unknown variants — this helper exists
/// for callers that want to log or reject unknowns at ingest time.
#[must_use]
pub fn is_known_hazard_type(hazard_type: &str) -> bool {
    HAZARD_TYPES.contains(&hazard_type)
}

/// Return `true` if `severity` is one of the documented variants in
/// [`HAZARD_SEVERITIES`].
#[must_use]
pub fn is_known_severity(severity: &str) -> bool {
    HAZARD_SEVERITIES.contains(&severity)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn critical_severity_always_halts() {
        for ty in HAZARD_TYPES {
            assert!(
                should_halt_trading(ty, "CRITICAL"),
                "CRITICAL severity on {ty} should halt trading"
            );
        }
    }

    #[test]
    fn low_medium_non_critical_types_do_not_halt() {
        let non_halting = [
            "EXCHANGE_TIMEOUT",
            "FEED_SILENCE",
            "EXECUTION_LATENCY_SPIKE",
        ];
        for ty in non_halting {
            for sev in ["LOW", "MEDIUM"] {
                assert!(
                    !should_halt_trading(ty, sev),
                    "{sev} {ty} should NOT halt trading"
                );
            }
        }
    }

    #[test]
    fn critical_types_halt_regardless_of_severity() {
        for ty in [
            "DATA_CORRUPTION_SUSPECTED",
            "LEDGER_INCONSISTENCY",
            "API_CONNECTIVITY_FAILURE",
        ] {
            for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] {
                assert!(should_halt_trading(ty, sev), "{sev} {ty} must halt trading");
            }
        }
    }

    #[test]
    fn safe_mode_triggers_on_high_or_critical() {
        for ty in HAZARD_TYPES {
            for sev in ["HIGH", "CRITICAL"] {
                assert!(
                    should_enter_safe_mode(ty, sev),
                    "{sev} {ty} should enter safe mode"
                );
            }
        }
    }

    #[test]
    fn safe_mode_triggers_on_feed_silence_or_exchange_timeout_any_severity() {
        for ty in ["FEED_SILENCE", "EXCHANGE_TIMEOUT"] {
            for sev in HAZARD_SEVERITIES {
                assert!(
                    should_enter_safe_mode(ty, sev),
                    "{sev} {ty} should enter safe mode"
                );
            }
        }
    }

    #[test]
    fn safe_mode_does_not_trigger_on_low_medium_other_types() {
        assert!(!should_enter_safe_mode("MEMORY_PRESSURE", "LOW"));
        assert!(!should_enter_safe_mode("CPU_OVERLOAD", "MEDIUM"));
        assert!(!should_enter_safe_mode("EXECUTION_LATENCY_SPIKE", "LOW"));
    }

    #[test]
    fn classify_severity_promotes_critical_types() {
        assert_eq!(
            classify_severity("DATA_CORRUPTION_SUSPECTED", "LOW"),
            "CRITICAL"
        );
        assert_eq!(
            classify_severity("LEDGER_INCONSISTENCY", "MEDIUM"),
            "CRITICAL"
        );
        assert_eq!(
            classify_severity("DATA_CORRUPTION_SUSPECTED", "HIGH"),
            "CRITICAL"
        );
    }

    #[test]
    fn classify_severity_passes_through_non_critical_types() {
        assert_eq!(classify_severity("FEED_SILENCE", "HIGH"), "HIGH");
        assert_eq!(
            classify_severity("EXECUTION_LATENCY_SPIKE", "MEDIUM"),
            "MEDIUM"
        );
        assert_eq!(classify_severity("CPU_OVERLOAD", "LOW"), "LOW");
    }

    #[test]
    fn classify_response_covers_all_documented_types() {
        let expected = [
            ("EXCHANGE_TIMEOUT", "CANCEL_ALL_OPEN_ORDERS"),
            ("FEED_SILENCE", "PAUSE_NEW_ORDERS"),
            ("EXECUTION_LATENCY_SPIKE", "REDUCE_EXPOSURE"),
            ("DATA_CORRUPTION_SUSPECTED", "HALT_TRADING"),
            ("API_CONNECTIVITY_FAILURE", "HALT_TRADING"),
        ];
        for (ty, resp) in expected {
            assert_eq!(classify_response(ty), resp, "{ty}");
        }
    }

    #[test]
    fn classify_response_falls_through_to_observe() {
        // Types NOT explicitly mapped (or unknown types) default to
        // OBSERVE. That is the conservative choice — we never halt
        // trading on a type we do not recognise.
        assert_eq!(classify_response("SYSTEM_DEGRADATION"), "OBSERVE");
        assert_eq!(classify_response("MEMORY_PRESSURE"), "OBSERVE");
        assert_eq!(classify_response("CPU_OVERLOAD"), "OBSERVE");
        assert_eq!(classify_response("LEDGER_INCONSISTENCY"), "OBSERVE");
        assert_eq!(classify_response("TOTALLY_NOVEL_TYPE"), "OBSERVE");
    }

    #[test]
    fn known_type_predicate() {
        for ty in HAZARD_TYPES {
            assert!(is_known_hazard_type(ty));
        }
        assert!(!is_known_hazard_type("NOT_A_REAL_TYPE"));
        assert!(!is_known_hazard_type(""));
    }

    #[test]
    fn known_severity_predicate() {
        for sev in HAZARD_SEVERITIES {
            assert!(is_known_severity(sev));
        }
        assert!(!is_known_severity("VERY_BAD"));
        assert!(!is_known_severity(""));
    }

    #[test]
    fn unknown_types_are_classified_conservatively() {
        // should_halt_trading: CRITICAL on unknown still halts.
        assert!(should_halt_trading("NOVEL_TYPE", "CRITICAL"));
        // should_halt_trading: LOW on unknown does not halt.
        assert!(!should_halt_trading("NOVEL_TYPE", "LOW"));
        // classify_response on unknown returns OBSERVE.
        assert_eq!(classify_response("NOVEL_TYPE"), "OBSERVE");
        // classify_severity on unknown passes through as-is.
        assert_eq!(classify_severity("NOVEL_TYPE", "HIGH"), "HIGH");
    }

    #[test]
    fn known_types_and_severities_are_disjoint_sets() {
        // Guard against an accidental copy-paste that puts a type
        // into HAZARD_SEVERITIES or vice versa.
        let types: HashSet<&&str> = HAZARD_TYPES.iter().collect();
        let sevs: HashSet<&&str> = HAZARD_SEVERITIES.iter().collect();
        assert!(types.is_disjoint(&sevs));
    }
}
