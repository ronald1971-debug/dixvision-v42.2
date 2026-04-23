//! # Circuit-breaker (T0-8 — adapter hot-path safety)
//!
//! Canonical Rust implementation of a three-state circuit-breaker
//! used by every exchange adapter under `execution/adapters/`. The
//! Python reference (`core/contracts/risk.py` + loose uses in
//! `governance/risk_engine.py`) only spoke about constraints; this
//! module lands the missing runtime state-machine so adapter calls
//! fail fast after a pathological burst and recover without
//! operator intervention.
//!
//! ## State machine
//!
//! * **Closed** — normal operation; every `allow()` returns `true`.
//!   Consecutive `record_failure()` calls increment a counter; when
//!   it reaches `failure_threshold` the breaker transitions to
//!   [`BreakerState::Open`].
//! * **Open** — `allow()` returns `false`. After
//!   `reset_timeout_ms` the breaker enters
//!   [`BreakerState::HalfOpen`] on the next `allow()` call.
//! * **`HalfOpen`** — exactly **one** probe call is allowed; every
//!   subsequent `allow()` returns `false` until the probe resolves.
//!   `record_success()` transitions back to [`BreakerState::Closed`];
//!   `record_failure()` re-opens for another `reset_timeout_ms`.
//!
//! ## Thread-safety
//!
//! State is serialised through a single `parking_lot::Mutex`. The
//! hot path (`allow()`) takes the lock, reads and possibly mutates
//! state, then releases — the critical section is a handful of
//! integer comparisons, well below the SLO budget even under heavy
//! contention.
//!
//! Time is sourced from the crate-local [`MonotonicClock`] trait so
//! tests can drive it deterministically. Production callers use
//! [`SystemClock`], which wraps `std::time::Instant::now()`.

use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Three-state breaker status. Observable through
/// [`CircuitBreaker::state`] for dashboards and debug output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal operation — every call is allowed.
    Closed,
    /// Tripped — every call is rejected until the reset timeout
    /// elapses.
    Open,
    /// Recovering — exactly one probe call is allowed; the probe's
    /// outcome decides the next state.
    HalfOpen,
}

impl BreakerState {
    /// Stable short name for logging and cross-language parity
    /// tests. Matches the Python wrapper's ``state()`` output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

/// Monotonic time source. Abstracted for test injection; production
/// code uses [`SystemClock`].
pub trait MonotonicClock: Send + Sync {
    /// Current instant. Must be monotonic across calls on a given
    /// clock instance.
    fn now(&self) -> Instant;
}

/// Real clock backed by [`Instant::now`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl MonotonicClock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Debug)]
struct Inner {
    state: BreakerState,
    consecutive_failures: u32,
    /// When the current open window expires. `None` unless state is
    /// `Open`.
    open_until: Option<Instant>,
    /// Whether a probe is currently in flight in `HalfOpen`. Exactly
    /// one probe is allowed per recovery window.
    probe_in_flight: bool,
}

impl Inner {
    const fn new() -> Self {
        Self {
            state: BreakerState::Closed,
            consecutive_failures: 0,
            open_until: None,
            probe_in_flight: false,
        }
    }
}

/// Circuit-breaker configuration. All thresholds are copied into
/// the breaker at construction time; mutating the config after
/// the fact requires constructing a new breaker.
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Consecutive failures required to trip from `Closed` → `Open`.
    /// Must be at least 1.
    pub failure_threshold: u32,
    /// How long the breaker stays open before allowing a probe call.
    pub reset_timeout: Duration,
}

impl BreakerConfig {
    /// Sane defaults for exchange-adapter use. Three consecutive
    /// failures trip the breaker; the reset window is 30 seconds.
    #[must_use]
    pub const fn default_for_adapter() -> Self {
        Self {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(30),
        }
    }
}

/// Three-state circuit-breaker.
///
/// Generic over the clock source so tests can advance time
/// deterministically. Most production callers construct with
/// [`CircuitBreaker::new`] which pins [`SystemClock`].
#[derive(Debug)]
pub struct CircuitBreaker<C: MonotonicClock = SystemClock> {
    config: BreakerConfig,
    clock: C,
    inner: Mutex<Inner>,
}

impl CircuitBreaker<SystemClock> {
    /// Construct with the real clock and the given config.
    ///
    /// # Panics
    ///
    /// Panics if `config.failure_threshold` is zero — a breaker that
    /// trips on zero failures is nonsensical and almost certainly a
    /// configuration bug the caller wants surfaced at construction
    /// time.
    #[must_use]
    pub fn new(config: BreakerConfig) -> Self {
        Self::with_clock(config, SystemClock)
    }
}

impl<C: MonotonicClock> CircuitBreaker<C> {
    /// Construct with an injected clock. Used by the parity suite.
    ///
    /// # Panics
    ///
    /// Panics if `config.failure_threshold` is zero — see [`new`].
    ///
    /// [`new`]: CircuitBreaker::new
    pub fn with_clock(config: BreakerConfig, clock: C) -> Self {
        assert!(
            config.failure_threshold > 0,
            "failure_threshold must be at least 1",
        );
        Self {
            config,
            clock,
            inner: Mutex::new(Inner::new()),
        }
    }

    /// Current observable state. Does NOT advance the state
    /// machine; callers that want the timer-driven transition
    /// from `Open` → `HalfOpen` should call [`Self::allow`].
    pub fn state(&self) -> BreakerState {
        self.inner.lock().state
    }

    /// Whether the next call should be allowed. Side-effectful:
    /// advances the state machine from `Open` → `HalfOpen` when the
    /// reset timeout has elapsed, and reserves the probe slot in
    /// `HalfOpen` when it returns `true`.
    pub fn allow(&self) -> bool {
        let now = self.clock.now();
        let mut guard = self.inner.lock();
        match guard.state {
            BreakerState::Closed => true,
            BreakerState::Open => {
                if guard.open_until.is_some_and(|deadline| now >= deadline) {
                    guard.state = BreakerState::HalfOpen;
                    guard.open_until = None;
                    guard.probe_in_flight = true;
                    true
                } else {
                    false
                }
            }
            BreakerState::HalfOpen => {
                if guard.probe_in_flight {
                    // Another thread already reserved the probe slot.
                    false
                } else {
                    guard.probe_in_flight = true;
                    true
                }
            }
        }
    }

    /// Mark the last allowed call as successful.
    ///
    /// * In `Closed`: reset the consecutive-failure counter.
    /// * In `HalfOpen`: close the breaker and clear all state.
    /// * In `Open`: no-op — callers shouldn't be reporting success
    ///   on a rejected call, but we tolerate the race quietly.
    pub fn record_success(&self) {
        let mut guard = self.inner.lock();
        match guard.state {
            BreakerState::Closed => {
                guard.consecutive_failures = 0;
            }
            BreakerState::HalfOpen => {
                guard.state = BreakerState::Closed;
                guard.consecutive_failures = 0;
                guard.probe_in_flight = false;
            }
            BreakerState::Open => {}
        }
    }

    /// Mark the last allowed call as failed.
    ///
    /// * In `Closed`: increment the failure counter; trip to `Open`
    ///   when it reaches `failure_threshold`.
    /// * In `HalfOpen`: re-open for another `reset_timeout`.
    /// * In `Open`: no-op.
    pub fn record_failure(&self) {
        let now = self.clock.now();
        let mut guard = self.inner.lock();
        match guard.state {
            BreakerState::Closed => {
                guard.consecutive_failures = guard.consecutive_failures.saturating_add(1);
                if guard.consecutive_failures >= self.config.failure_threshold {
                    guard.state = BreakerState::Open;
                    guard.open_until = Some(now + self.config.reset_timeout);
                }
            }
            BreakerState::HalfOpen => {
                guard.state = BreakerState::Open;
                guard.open_until = Some(now + self.config.reset_timeout);
                guard.probe_in_flight = false;
            }
            BreakerState::Open => {}
        }
    }

    /// Force-reset the breaker to `Closed`. Used by operators via
    /// the governance panel when manual recovery is warranted.
    pub fn reset(&self) {
        let mut guard = self.inner.lock();
        guard.state = BreakerState::Closed;
        guard.consecutive_failures = 0;
        guard.open_until = None;
        guard.probe_in_flight = false;
    }

    /// Current consecutive-failure count. Observable for dashboards.
    pub fn failure_count(&self) -> u32 {
        self.inner.lock().consecutive_failures
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    /// Clock the parity suite can drive forward explicitly. Backed
    /// by an `Arc<Mutex<Instant>>` so multiple clones share the
    /// same wall-clock cursor.
    #[derive(Debug, Clone)]
    struct FakeClock {
        cursor: Arc<Mutex<Instant>>,
    }

    impl FakeClock {
        fn new() -> Self {
            Self {
                cursor: Arc::new(Mutex::new(Instant::now())),
            }
        }
        fn advance(&self, d: Duration) {
            let mut g = self.cursor.lock();
            *g += d;
        }
    }

    impl MonotonicClock for FakeClock {
        fn now(&self) -> Instant {
            *self.cursor.lock()
        }
    }

    const fn cfg(threshold: u32, reset_ms: u64) -> BreakerConfig {
        BreakerConfig {
            failure_threshold: threshold,
            reset_timeout: Duration::from_millis(reset_ms),
        }
    }

    #[test]
    fn default_state_is_closed() {
        let b: CircuitBreaker = CircuitBreaker::new(BreakerConfig::default_for_adapter());
        assert_eq!(b.state(), BreakerState::Closed);
        assert!(b.allow());
    }

    #[test]
    fn trips_after_threshold_failures() {
        let clk = FakeClock::new();
        let b = CircuitBreaker::with_clock(cfg(3, 1_000), clk);
        for _ in 0..3 {
            b.record_failure();
        }
        assert_eq!(b.state(), BreakerState::Open);
        assert!(!b.allow());
    }

    #[test]
    fn success_resets_failure_counter_in_closed() {
        let clk = FakeClock::new();
        let b = CircuitBreaker::with_clock(cfg(3, 1_000), clk);
        b.record_failure();
        b.record_failure();
        assert_eq!(b.failure_count(), 2);
        b.record_success();
        assert_eq!(b.failure_count(), 0);
        assert_eq!(b.state(), BreakerState::Closed);
    }

    #[test]
    fn open_transitions_to_half_open_after_timeout() {
        let clk = FakeClock::new();
        let b = CircuitBreaker::with_clock(cfg(1, 500), clk.clone());
        b.record_failure();
        assert_eq!(b.state(), BreakerState::Open);
        // Before timeout: still open, allow returns false.
        clk.advance(Duration::from_millis(499));
        assert!(!b.allow());
        assert_eq!(b.state(), BreakerState::Open);
        // After timeout: allow reserves the probe, state becomes
        // half-open, subsequent allows are rejected.
        clk.advance(Duration::from_millis(2));
        assert!(b.allow());
        assert_eq!(b.state(), BreakerState::HalfOpen);
        assert!(!b.allow());
    }

    #[test]
    fn half_open_success_closes_breaker() {
        let clk = FakeClock::new();
        let b = CircuitBreaker::with_clock(cfg(1, 100), clk.clone());
        b.record_failure();
        clk.advance(Duration::from_millis(150));
        assert!(b.allow()); // reserves probe → half-open
        b.record_success();
        assert_eq!(b.state(), BreakerState::Closed);
        assert_eq!(b.failure_count(), 0);
        assert!(b.allow());
    }

    #[test]
    fn half_open_failure_reopens_for_new_window() {
        let clk = FakeClock::new();
        let b = CircuitBreaker::with_clock(cfg(1, 100), clk.clone());
        b.record_failure();
        clk.advance(Duration::from_millis(150));
        assert!(b.allow()); // probe
        b.record_failure();
        assert_eq!(b.state(), BreakerState::Open);
        // Another probe must wait a full new reset_timeout, not be
        // immediately available.
        assert!(!b.allow());
        clk.advance(Duration::from_millis(150));
        assert!(b.allow());
    }

    #[test]
    fn reset_clears_all_state() {
        let clk = FakeClock::new();
        let b = CircuitBreaker::with_clock(cfg(2, 100), clk);
        b.record_failure();
        b.record_failure();
        assert_eq!(b.state(), BreakerState::Open);
        b.reset();
        assert_eq!(b.state(), BreakerState::Closed);
        assert_eq!(b.failure_count(), 0);
        assert!(b.allow());
    }

    #[test]
    fn state_as_str_matches_python_contract() {
        assert_eq!(BreakerState::Closed.as_str(), "closed");
        assert_eq!(BreakerState::Open.as_str(), "open");
        assert_eq!(BreakerState::HalfOpen.as_str(), "half_open");
    }

    #[test]
    #[should_panic(expected = "failure_threshold must be at least 1")]
    fn zero_threshold_panics_at_construction() {
        let _ = CircuitBreaker::new(cfg(0, 100));
    }

    #[test]
    fn concurrent_failures_trip_exactly_once() {
        let clk = FakeClock::new();
        let b = Arc::new(CircuitBreaker::with_clock(cfg(10, 1_000), clk));
        let mut handles = vec![];
        for _ in 0..4 {
            let b2 = Arc::clone(&b);
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    b2.record_failure();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(b.state(), BreakerState::Open);
        // saturating_add is used internally so the counter must not
        // have wrapped regardless of interleaving.
        assert!(b.failure_count() >= 10);
    }

    #[test]
    fn half_open_allows_only_one_probe_under_contention() {
        let clk = FakeClock::new();
        let b = Arc::new(CircuitBreaker::with_clock(cfg(1, 50), clk.clone()));
        b.record_failure();
        clk.advance(Duration::from_millis(60));
        // Many threads race to call allow(); exactly one should
        // see true and reserve the probe slot. Collect the join
        // handles eagerly so every thread is actually spawned before
        // we start reading outcomes — the `collect` is load-bearing
        // (spawns must happen before joins).
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let b2 = Arc::clone(&b);
                thread::spawn(move || b2.allow())
            })
            .collect();
        let granted = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|x| *x)
            .count();
        assert_eq!(granted, 1, "exactly one probe slot");
    }
}
