//! Pre-trade hot-path gate (`EXEC-11`).
//!
//! This module is the **pure decision** half of the
//! `execution_engine.hot_path.fast_execute.FastExecutor` Python class.
//! It answers a single question per call: *given a signal, a frozen
//! risk snapshot, and a mark price — does this become a trade, or
//! does it reject, and why?*
//!
//! Everything stateful (the order-id counter, the `FastExecutor`
//! constructor's argument validation, the construction of
//! `ExecutionEvent` / `HotPathDecision`) lives on the Python side.
//! The Rust side is a pure function so it is trivially deterministic
//! (INV-15) and testable without `PyO3` in scope.
//!
//! # Why port this module
//!
//! It is the per-tick gate. Every signal that hits the engine flows
//! through this function before anything else; if it stays in
//! Python, the GIL + per-call attribute lookup cost dominates the
//! tick budget. The logic itself is small and branch-heavy, which is
//! exactly the shape Rust shines at.
//!
//! The Python reference is the source of truth: any divergence here
//! is a bug, and the parity tests in `tests/test_fast_execute_parity`
//! verify byte-for-byte agreement on every branch + the qty fallback
//! ladder.

/// Outcome of one gate evaluation.
///
/// Mirrors `execution_engine.hot_path.fast_execute.HotPathOutcome`
/// exactly. The variant order is irrelevant for FFI (we marshal
/// names, not discriminants), but kept stable for `Debug` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HotPathOutcome {
    /// Signal becomes an `ExecutionEvent` at `mark_price` and `qty`.
    Approved,
    /// `signal.ts_ns - snapshot.ts_ns` exceeds `max_staleness_ns`.
    RejectedRiskStale,
    /// `mark_price <= 0`.
    RejectedNoMark,
    /// Either the snapshot is `halted` or `qty > cap_for(symbol)`.
    /// Two distinct reasons share this outcome to mirror the Python
    /// reference.
    RejectedLimit,
    /// `signal.side` is `HOLD`.
    RejectedHold,
    /// `signal.confidence < snapshot.max_signal_confidence`.
    RejectedLowConfidence,
}

impl HotPathOutcome {
    /// Stable string name matching the Python `StrEnum` value.
    ///
    /// The `PyO3` seam returns this so the Python wrapper can re-tag
    /// the result onto the existing `HotPathOutcome` enum without a
    /// FFI enum round-trip.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Approved => "APPROVED",
            Self::RejectedRiskStale => "REJECTED_RISK_STALE",
            Self::RejectedNoMark => "REJECTED_NO_MARK",
            Self::RejectedLimit => "REJECTED_LIMIT",
            Self::RejectedHold => "REJECTED_HOLD",
            Self::RejectedLowConfidence => "REJECTED_LOW_CONFIDENCE",
        }
    }
}

/// Side of a signal. We accept any string here so the FFI seam is a
/// pure `&str`, but only `"BUY"`, `"SELL"`, `"HOLD"` are meaningful.
/// Anything else is treated as not-`HOLD` (which is also what the
/// Python reference would do — `side is Side.HOLD` is `False` for
/// any non-`HOLD` value).
fn is_hold(side: &str) -> bool {
    side == "HOLD"
}

/// Inputs to one gate decision. All values are owned primitives so
/// the logic is callable both from Rust tests and from the `PyO3`
/// seam.
#[derive(Debug, Clone, Copy)]
pub struct GateInputs<'a> {
    /// `signal.ts_ns`
    pub signal_ts_ns: i64,
    /// `signal.confidence`
    pub signal_confidence: f64,
    /// `signal.side` as a string. Only `"HOLD"` is treated specially.
    pub signal_side: &'a str,
    /// `snapshot.version`
    pub snapshot_version: i64,
    /// `snapshot.ts_ns`
    pub snapshot_ts_ns: i64,
    /// `snapshot.halted`
    pub snapshot_halted: bool,
    /// `snapshot.max_signal_confidence`
    pub snapshot_max_signal_confidence: f64,
    /// Caller resolves `snapshot.cap_for(symbol)` and passes the
    /// numeric cap (or `None` for unbounded). Done caller-side so the
    /// Rust function does not have to walk a Python dict over FFI.
    pub cap: Option<f64>,
    /// Mark price for the symbol.
    pub mark_price: f64,
    /// Per-`FastExecutor` staleness budget.
    pub max_staleness_ns: i64,
    /// Resolved order qty. Caller applies the
    /// `signal.meta["qty"] || default_qty || strictly-positive`
    /// fallback ladder before calling.
    pub qty: f64,
}

/// One gate decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateDecision {
    /// Which branch fired.
    pub outcome: HotPathOutcome,
    /// Stable reason tag. `""` when approved; otherwise one of
    /// `"halted"`, `"risk_stale"`, `"no_mark"`, `"confidence_floor"`,
    /// `"hold_signal"`, `"qty_above_cap"`. Tags match the Python
    /// reference verbatim and are part of the audit ledger contract.
    pub reason: &'static str,
    /// Price echoed onto the rejected/approved `ExecutionEvent`.
    /// Mirrors the Python reference's per-branch price assignment
    /// (zero for some branches, mark for others) — see the source
    /// code rather than re-explaining it here.
    pub price: f64,
}

/// Evaluate the gate. Pure function of its inputs; deterministic.
///
/// Branch order matches the Python reference exactly so the audit
/// reasons agree on the *first* failing condition when two would
/// fire (e.g. halted + stale would both reject; halted wins because
/// it is checked first).
#[must_use]
pub fn decide_gate(inp: &GateInputs<'_>) -> GateDecision {
    // Halted: fail fast, deterministically. Price is the mark when
    // it's positive, else zero — same as the Python reference.
    if inp.snapshot_halted {
        let price = if inp.mark_price > 0.0 {
            inp.mark_price
        } else {
            0.0
        };
        return GateDecision {
            outcome: HotPathOutcome::RejectedLimit,
            reason: "halted",
            price,
        };
    }

    if inp.signal_ts_ns - inp.snapshot_ts_ns > inp.max_staleness_ns {
        return GateDecision {
            outcome: HotPathOutcome::RejectedRiskStale,
            reason: "risk_stale",
            price: 0.0,
        };
    }

    if inp.mark_price <= 0.0 {
        return GateDecision {
            outcome: HotPathOutcome::RejectedNoMark,
            reason: "no_mark",
            price: 0.0,
        };
    }

    // `<` matches the Python reference. A signal whose confidence
    // exactly equals the floor passes.
    #[allow(clippy::float_cmp)]
    if inp.signal_confidence < inp.snapshot_max_signal_confidence {
        return GateDecision {
            outcome: HotPathOutcome::RejectedLowConfidence,
            reason: "confidence_floor",
            price: inp.mark_price,
        };
    }

    if is_hold(inp.signal_side) {
        return GateDecision {
            outcome: HotPathOutcome::RejectedHold,
            reason: "hold_signal",
            price: inp.mark_price,
        };
    }

    if let Some(cap) = inp.cap {
        if inp.qty > cap {
            return GateDecision {
                outcome: HotPathOutcome::RejectedLimit,
                reason: "qty_above_cap",
                price: inp.mark_price,
            };
        }
    }

    GateDecision {
        outcome: HotPathOutcome::Approved,
        reason: "",
        price: inp.mark_price,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn base_inputs(side: &str) -> GateInputs<'_> {
        GateInputs {
            signal_ts_ns: 1_000_000_000,
            signal_confidence: 0.9,
            signal_side: side,
            snapshot_version: 1,
            snapshot_ts_ns: 1_000_000_000,
            snapshot_halted: false,
            snapshot_max_signal_confidence: 0.5,
            cap: Some(5.0),
            mark_price: 50_000.0,
            max_staleness_ns: 2_000_000_000,
            qty: 1.0,
        }
    }

    #[test]
    fn approved_path_returns_mark_price() {
        let d = decide_gate(&base_inputs("BUY"));
        assert_eq!(d.outcome, HotPathOutcome::Approved);
        assert_eq!(d.reason, "");
        assert!((d.price - 50_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn halted_wins_over_everything() {
        let mut inp = base_inputs("BUY");
        inp.snapshot_halted = true;
        // also set up a stale risk; halted should still win.
        inp.signal_ts_ns = 1_000_000_000 + 10 * inp.max_staleness_ns;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedLimit);
        assert_eq!(d.reason, "halted");
        assert!((d.price - 50_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn halted_with_zero_mark_returns_zero_price() {
        let mut inp = base_inputs("BUY");
        inp.snapshot_halted = true;
        inp.mark_price = 0.0;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedLimit);
        assert_eq!(d.reason, "halted");
        assert!(d.price.abs() < f64::EPSILON);
    }

    #[test]
    fn stale_risk_rejects_with_zero_price() {
        let mut inp = base_inputs("BUY");
        inp.signal_ts_ns = 1_000_000_000 + inp.max_staleness_ns + 1;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedRiskStale);
        assert_eq!(d.reason, "risk_stale");
        assert!(d.price.abs() < f64::EPSILON);
    }

    #[test]
    fn stale_threshold_is_exclusive() {
        // delta == max_staleness_ns is allowed (not stale yet).
        let mut inp = base_inputs("BUY");
        inp.signal_ts_ns = 1_000_000_000 + inp.max_staleness_ns;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::Approved);
    }

    #[test]
    fn missing_mark_rejects() {
        let mut inp = base_inputs("BUY");
        inp.mark_price = 0.0;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedNoMark);
        assert_eq!(d.reason, "no_mark");
    }

    #[test]
    fn negative_mark_rejects() {
        let mut inp = base_inputs("BUY");
        inp.mark_price = -1.0;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedNoMark);
    }

    #[test]
    fn low_confidence_rejects() {
        let mut inp = base_inputs("BUY");
        inp.signal_confidence = 0.49;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedLowConfidence);
        assert_eq!(d.reason, "confidence_floor");
        // Echo the mark price on the rejection — matches Python.
        assert!((d.price - 50_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_at_floor_passes() {
        // `<` not `<=` — a signal exactly at the floor is approved.
        let mut inp = base_inputs("BUY");
        inp.signal_confidence = 0.5;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::Approved);
    }

    #[test]
    fn hold_signal_rejects() {
        let inp = base_inputs("HOLD");
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedHold);
        assert_eq!(d.reason, "hold_signal");
    }

    #[test]
    fn qty_above_cap_rejects() {
        let mut inp = base_inputs("BUY");
        inp.qty = 10.0;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::RejectedLimit);
        assert_eq!(d.reason, "qty_above_cap");
    }

    #[test]
    fn unbounded_cap_allows_any_qty() {
        let mut inp = base_inputs("BUY");
        inp.cap = None;
        inp.qty = 1_000_000.0;
        let d = decide_gate(&inp);
        assert_eq!(d.outcome, HotPathOutcome::Approved);
    }

    #[test]
    fn outcome_str_names_match_python_strenum() {
        // Bind the public string names so the FFI seam contract is
        // self-documenting: any rename here is a public-API break.
        assert_eq!(HotPathOutcome::Approved.as_str(), "APPROVED");
        assert_eq!(
            HotPathOutcome::RejectedRiskStale.as_str(),
            "REJECTED_RISK_STALE",
        );
        assert_eq!(HotPathOutcome::RejectedNoMark.as_str(), "REJECTED_NO_MARK",);
        assert_eq!(HotPathOutcome::RejectedLimit.as_str(), "REJECTED_LIMIT",);
        assert_eq!(HotPathOutcome::RejectedHold.as_str(), "REJECTED_HOLD");
        assert_eq!(
            HotPathOutcome::RejectedLowConfidence.as_str(),
            "REJECTED_LOW_CONFIDENCE",
        );
    }
}
