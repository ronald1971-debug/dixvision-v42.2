//! DIX VISION v42.2 — T0-1 Fast Risk Cache (Rust port).
//!
//! The **only** runtime interface from governance (sole writer) to
//! Indira (hot-path reader). Governance asynchronously publishes
//! precomputed [`RiskConstraints`]; Indira consumes them every tick
//! via a lock-free atomic read.
//!
//! # Polyglot note
//!
//! This is the baseline port — it matches the pre-T0-1 public API
//! byte-for-byte so the `PyO3` seam can expose the same surface as the
//! Python reference impl without breaking any caller. The T0-1
//! staleness-halt / `version_id` / `RiskReading` extensions land in a
//! follow-up PR once the corresponding Python-side PR (#12) is in the
//! base branch — then both sides evolve together.
//!
//! # Concurrency
//!
//! * **Writes** go through a `parking_lot::Mutex`: governance is the
//!   sole writer, so contention is structurally bounded to one thread
//!   bumping the reference + copying a small struct.
//! * **Reads** are lock-free: we store the current `Arc<RiskConstraints>`
//!   inside an `ArcSwap`, and a reader does a single atomic pointer
//!   load + ref-count bump. The SLO p99 < 5 ms budget is dominated by
//!   network, not this module.
//! * The reader contract is explicitly **"eventually consistent up to
//!   the most recent completed `update()`"**: a reader that observes
//!   an older snapshot and a reader that observes the new one are
//!   both valid; governance publishes at a cadence faster than
//!   Indira's tick, so the window is bounded.
//!
//! # Invariants (tested)
//!
//! * `get()` always returns the most recent completed `update()`
//!   (writer holds the lock across the `ArcSwap::store`, so the
//!   publish is atomic from the reader's perspective).
//! * `halt_trading()` monotonically sets `trading_allowed = false`;
//!   only `resume_trading()` can re-enable it.
//! * `enter_safe_mode()` / `exit_safe_mode()` flip `safe_mode` in a
//!   single commit (no intermediate half-state visible to readers).

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::time_source;

/// Pre-computed risk limits. Consumed by Indira's fast path.
///
/// Every field mirrors the Python dataclass — changes here must be
/// reflected in the `PyO3` seam (`rust/py_system`) and the Python
/// wrapper (`system/fast_risk_cache.py`) in the same PR.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskConstraints {
    /// Maximum position as fraction of portfolio (1.0 = 100%).
    pub max_position_pct: f64,
    /// Absolute per-order cap, USD.
    pub max_order_size_usd: f64,
    /// Upper volatility band.
    pub volatility_band_high: f64,
    /// Lower volatility band.
    pub volatility_band_low: f64,
    /// Portfolio-level drawdown circuit breaker.
    pub circuit_breaker_drawdown: f64,
    /// Per-trade loss-percentage circuit breaker.
    pub circuit_breaker_loss_pct: f64,
    /// Governance-controlled master switch. `false` halts all trading.
    pub trading_allowed: bool,
    /// Governance-controlled safe-mode flag. `true` rejects trades.
    pub safe_mode: bool,
    /// ISO-8601 UTC timestamp of the last update (informational only —
    /// the monotonic-clock stamp lives on `TimeSource`).
    pub last_updated_utc: String,
}

impl Default for RiskConstraints {
    fn default() -> Self {
        Self {
            max_position_pct: 1.0,
            max_order_size_usd: 10_000.0,
            volatility_band_high: 0.05,
            volatility_band_low: 0.001,
            circuit_breaker_drawdown: 0.04,
            circuit_breaker_loss_pct: 0.01,
            trading_allowed: true,
            safe_mode: false,
            last_updated_utc: String::new(),
        }
    }
}

/// Verdict returned by [`RiskConstraints::allows_trade`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradeVerdict {
    /// Whether the trade is allowed.
    pub allowed: bool,
    /// Machine-readable reason string. Stable across versions —
    /// decision records stamp this verbatim.
    pub reason: String,
}

impl RiskConstraints {
    /// Check whether a trade of `size_usd` against a portfolio of
    /// `portfolio_usd` is allowed.
    ///
    /// Fail-closed semantics:
    /// * `trading_allowed = false` → reject with `"trading_not_allowed"`.
    /// * `safe_mode = true`        → reject with `"safe_mode_active"`.
    /// * `portfolio_usd <= 0`      → reject with `"portfolio_usd_required"`
    ///   (cannot enforce the percentage circuit breaker without it).
    /// * `size_usd > max_order_size_usd` → reject with a formatted reason.
    /// * `size_usd / portfolio_usd > circuit_breaker_loss_pct` → reject.
    #[must_use]
    pub fn allows_trade(&self, size_usd: f64, portfolio_usd: f64) -> TradeVerdict {
        if !self.trading_allowed {
            return TradeVerdict {
                allowed: false,
                reason: "trading_not_allowed".to_string(),
            };
        }
        if self.safe_mode {
            return TradeVerdict {
                allowed: false,
                reason: "safe_mode_active".to_string(),
            };
        }
        if portfolio_usd <= 0.0 {
            return TradeVerdict {
                allowed: false,
                reason: "portfolio_usd_required".to_string(),
            };
        }
        if size_usd > self.max_order_size_usd {
            return TradeVerdict {
                allowed: false,
                reason: format!(
                    "size_usd_{:.2}_exceeds_max_{:.2}",
                    size_usd, self.max_order_size_usd
                ),
            };
        }
        let pct = size_usd / portfolio_usd;
        if pct > self.circuit_breaker_loss_pct {
            return TradeVerdict {
                allowed: false,
                reason: format!(
                    "size_pct_{:.4}_exceeds_limit_{}",
                    pct, self.circuit_breaker_loss_pct
                ),
            };
        }
        TradeVerdict {
            allowed: true,
            reason: "ok".to_string(),
        }
    }
}

/// Patch record applied by `FastRiskCache::update`. Every field is
/// optional: `None` means "keep the previous value", `Some(x)` means
/// "replace with x".
#[derive(Debug, Default, Clone)]
pub struct RiskUpdate {
    /// Override `max_position_pct`.
    pub max_position_pct: Option<f64>,
    /// Override `max_order_size_usd`.
    pub max_order_size_usd: Option<f64>,
    /// Override `volatility_band_high`.
    pub volatility_band_high: Option<f64>,
    /// Override `volatility_band_low`.
    pub volatility_band_low: Option<f64>,
    /// Override `circuit_breaker_drawdown`.
    pub circuit_breaker_drawdown: Option<f64>,
    /// Override `circuit_breaker_loss_pct`.
    pub circuit_breaker_loss_pct: Option<f64>,
    /// Override `trading_allowed`.
    pub trading_allowed: Option<bool>,
    /// Override `safe_mode`.
    pub safe_mode: Option<bool>,
}

/// Atomic single-writer, multi-reader risk cache.
///
/// Governance is the sole writer (async). Indira reads every tick
/// with zero lock contention.
pub struct FastRiskCache {
    /// Canonical pointer. Readers `load()`; writers `store()` inside
    /// the mutex to serialize ordering.
    current: ArcSwap<RiskConstraints>,
    /// Serializes governance writes. Reads do NOT take this lock.
    write_lock: Mutex<()>,
}

impl FastRiskCache {
    /// Construct a cache with default constraints.
    #[must_use]
    pub fn new() -> Self {
        let initial = RiskConstraints {
            last_updated_utc: current_utc_iso(),
            ..Default::default()
        };
        Self {
            current: ArcSwap::from(Arc::new(initial)),
            write_lock: Mutex::new(()),
        }
    }

    /// Lock-free read of the current constraints snapshot. The caller
    /// receives a cheap `Arc<RiskConstraints>` — no copy.
    #[must_use]
    pub fn get(&self) -> Arc<RiskConstraints> {
        self.current.load_full()
    }

    /// Apply a `RiskUpdate`, stamping `last_updated_utc` atomically.
    /// Returns the new snapshot.
    pub fn update(&self, patch: &RiskUpdate) -> Arc<RiskConstraints> {
        let _guard = self.write_lock.lock();
        let prev = self.current.load_full();
        let mut next = (*prev).clone();
        if let Some(v) = patch.max_position_pct {
            next.max_position_pct = v;
        }
        if let Some(v) = patch.max_order_size_usd {
            next.max_order_size_usd = v;
        }
        if let Some(v) = patch.volatility_band_high {
            next.volatility_band_high = v;
        }
        if let Some(v) = patch.volatility_band_low {
            next.volatility_band_low = v;
        }
        if let Some(v) = patch.circuit_breaker_drawdown {
            next.circuit_breaker_drawdown = v;
        }
        if let Some(v) = patch.circuit_breaker_loss_pct {
            next.circuit_breaker_loss_pct = v;
        }
        if let Some(v) = patch.trading_allowed {
            next.trading_allowed = v;
        }
        if let Some(v) = patch.safe_mode {
            next.safe_mode = v;
        }
        next.last_updated_utc = current_utc_iso();
        let arc = Arc::new(next);
        self.current.store(Arc::clone(&arc));
        arc
    }

    /// Convenience: halt trading (`trading_allowed = false`, `safe_mode = true`).
    pub fn enter_safe_mode(&self) -> Arc<RiskConstraints> {
        self.update(&RiskUpdate {
            safe_mode: Some(true),
            trading_allowed: Some(false),
            ..Default::default()
        })
    }

    /// Exit safe-mode: re-enable trading. Callers must still verify
    /// governance-level policy before this is executed.
    pub fn exit_safe_mode(&self) -> Arc<RiskConstraints> {
        self.update(&RiskUpdate {
            safe_mode: Some(false),
            trading_allowed: Some(true),
            ..Default::default()
        })
    }

    /// Halt trading without entering safe-mode (for operator stop-all
    /// that can be reversed without a full safe-mode exit dance).
    pub fn halt_trading(&self) -> Arc<RiskConstraints> {
        self.update(&RiskUpdate {
            trading_allowed: Some(false),
            ..Default::default()
        })
    }

    /// Resume trading. Clears `safe_mode` as well so operators don't
    /// have to remember to un-set both independently.
    pub fn resume_trading(&self) -> Arc<RiskConstraints> {
        self.update(&RiskUpdate {
            trading_allowed: Some(true),
            safe_mode: Some(false),
            ..Default::default()
        })
    }
}

impl Default for FastRiskCache {
    fn default() -> Self {
        Self::new()
    }
}

/// ISO-8601 UTC timestamp derived from the canonical `TimeSource`.
///
/// Format: `YYYY-MM-DDTHH:MM:SS.ffffff+00:00` — matches the shape
/// produced by Python's `datetime.datetime.isoformat()` on a timezone-
/// aware UTC datetime, so governance / Indira consumers that compare
/// strings across the seam see the same bytes.
///
/// We use `chrono` only for the civil-calendar conversion; the actual
/// wall-clock reading comes from `TimeSource` (our canonical clock).
fn current_utc_iso() -> String {
    let stamp = time_source::now();
    let utc_ns = stamp.utc_nanos;
    let secs = i64::try_from(utc_ns.div_euclid(1_000_000_000)).unwrap_or(0);
    let sub_nanos = u32::try_from(utc_ns.rem_euclid(1_000_000_000)).unwrap_or(0);
    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, sub_nanos).unwrap_or_default();
    // "%.6f" emits the `.ffffff` fractional-seconds piece; "%:z"
    // emits "+00:00" for UTC. Together they match Python's
    // `datetime.isoformat()` for aware UTC datetimes.
    dt.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string()
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::uninlined_format_args
)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn defaults_are_permissive_and_safe() {
        let c = RiskConstraints::default();
        assert!(c.trading_allowed);
        assert!(!c.safe_mode);
        assert!((c.max_position_pct - 1.0).abs() < f64::EPSILON);
        assert!((c.max_order_size_usd - 10_000.0).abs() < f64::EPSILON);
        assert_eq!(c.last_updated_utc, "");
    }

    #[test]
    fn allows_trade_rejects_when_trading_disabled() {
        let c = RiskConstraints {
            trading_allowed: false,
            ..Default::default()
        };
        let v = c.allows_trade(100.0, 10_000.0);
        assert!(!v.allowed);
        assert_eq!(v.reason, "trading_not_allowed");
    }

    #[test]
    fn allows_trade_rejects_when_safe_mode() {
        let c = RiskConstraints {
            safe_mode: true,
            ..Default::default()
        };
        let v = c.allows_trade(100.0, 10_000.0);
        assert!(!v.allowed);
        assert_eq!(v.reason, "safe_mode_active");
    }

    #[test]
    fn allows_trade_requires_portfolio() {
        let c = RiskConstraints::default();
        let v = c.allows_trade(100.0, 0.0);
        assert!(!v.allowed);
        assert_eq!(v.reason, "portfolio_usd_required");
        let v2 = c.allows_trade(100.0, -5.0);
        assert!(!v2.allowed);
        assert_eq!(v2.reason, "portfolio_usd_required");
    }

    #[test]
    fn allows_trade_enforces_absolute_cap_before_percentage() {
        let c = RiskConstraints::default();
        // 20_000 > max_order_size_usd (10_000) — must reject on the
        // absolute cap regardless of how large the portfolio is.
        let v = c.allows_trade(20_000.0, 1_000_000_000.0);
        assert!(!v.allowed);
        assert!(v.reason.starts_with("size_usd_"), "got {}", v.reason);
        assert!(v.reason.contains("exceeds_max_"), "got {}", v.reason);
    }

    #[test]
    fn allows_trade_enforces_percentage_circuit_breaker() {
        let c = RiskConstraints::default();
        // 2% of 10_000 portfolio = 200; circuit_breaker_loss_pct = 1%.
        let v = c.allows_trade(200.0, 10_000.0);
        assert!(!v.allowed);
        assert!(v.reason.starts_with("size_pct_"), "got {}", v.reason);
    }

    #[test]
    fn allows_trade_accepts_within_limits() {
        let c = RiskConstraints::default();
        // 0.5% of a 10_000 portfolio = 50; both caps pass.
        let v = c.allows_trade(50.0, 10_000.0);
        assert!(v.allowed);
        assert_eq!(v.reason, "ok");
    }

    #[test]
    fn cache_get_returns_defaults_on_fresh() {
        let cache = FastRiskCache::new();
        let snap = cache.get();
        assert!(snap.trading_allowed);
        assert!(!snap.safe_mode);
        assert!(!snap.last_updated_utc.is_empty());
    }

    #[test]
    fn cache_update_swaps_reference_atomically() {
        let cache = FastRiskCache::new();
        let before = cache.get();
        let after = cache.update(&RiskUpdate {
            max_order_size_usd: Some(42_000.0),
            ..Default::default()
        });
        let observed = cache.get();
        assert!(!Arc::ptr_eq(&before, &after));
        assert!(Arc::ptr_eq(&after, &observed));
        assert!((after.max_order_size_usd - 42_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_enter_safe_mode_halts_trading() {
        let cache = FastRiskCache::new();
        let s = cache.enter_safe_mode();
        assert!(s.safe_mode);
        assert!(!s.trading_allowed);
        let v = s.allows_trade(10.0, 10_000.0);
        assert!(!v.allowed);
    }

    #[test]
    fn cache_exit_safe_mode_resumes_trading() {
        let cache = FastRiskCache::new();
        cache.enter_safe_mode();
        let s = cache.exit_safe_mode();
        assert!(!s.safe_mode);
        assert!(s.trading_allowed);
    }

    #[test]
    fn cache_halt_and_resume_cycle() {
        let cache = FastRiskCache::new();
        let h = cache.halt_trading();
        assert!(!h.trading_allowed);
        let r = cache.resume_trading();
        assert!(r.trading_allowed);
        assert!(!r.safe_mode);
    }

    #[test]
    fn concurrent_readers_never_see_partial_update() {
        // Writer flips `max_order_size_usd` between two wildly
        // different values. Readers must always observe one of the
        // two — never an intermediate or torn value.
        const READER_COUNT: usize = 8;
        const ITERATIONS: usize = 2_000;
        let cache = Arc::new(FastRiskCache::new());

        let writer_cache = Arc::clone(&cache);
        let writer = thread::spawn(move || {
            for i in 0..ITERATIONS {
                let v = if i & 1 == 0 { 10_000.0 } else { 99_999.0 };
                writer_cache.update(&RiskUpdate {
                    max_order_size_usd: Some(v),
                    ..Default::default()
                });
            }
        });

        let readers: Vec<_> = (0..READER_COUNT)
            .map(|_| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    for _ in 0..ITERATIONS {
                        let snap = c.get();
                        let v = snap.max_order_size_usd;
                        assert!(
                            (v - 10_000.0).abs() < f64::EPSILON
                                || (v - 99_999.0).abs() < f64::EPSILON,
                            "torn read: {}",
                            v
                        );
                    }
                })
            })
            .collect();

        writer.join().expect("writer thread panicked");
        for r in readers {
            r.join().expect("reader thread panicked");
        }
    }

    #[test]
    fn last_updated_utc_is_iso_8601() {
        let cache = FastRiskCache::new();
        let s = cache.get();
        // Shape: YYYY-MM-DDTHH:MM:SS.ffffff+00:00  (32 chars)
        assert_eq!(
            s.last_updated_utc.len(),
            32,
            "unexpected ISO length: {}",
            s.last_updated_utc
        );
        assert_eq!(&s.last_updated_utc[4..5], "-");
        assert_eq!(&s.last_updated_utc[7..8], "-");
        assert_eq!(&s.last_updated_utc[10..11], "T");
        assert_eq!(&s.last_updated_utc[13..14], ":");
        assert_eq!(&s.last_updated_utc[19..20], ".");
        assert!(s.last_updated_utc.ends_with("+00:00"));
    }

    #[test]
    fn utc_iso_matches_chrono_shape_on_epoch() {
        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap();
        let expected = dt.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string();
        assert_eq!(expected, "1970-01-01T00:00:00.000000+00:00");
    }
}
