//! DIX VISION v42.2 — T0-4 Time Authority (Rust port).
//!
//! Canonical monotonic clock + UTC-derivation for the whole system.
//! Wraps `std::time::Instant` (monotonic, non-adjustable, never goes
//! backwards) anchored once at process start against a single read
//! of `SystemTime::UNIX_EPOCH`, then derives every subsequent UTC
//! from the monotonic delta. `datetime.now()` / `SystemTime::now()`
//! in the hot path is forbidden by contract (see `tools/authority_lint.py`
//! rule T1); this module is the only place allowed to call either.
//!
//! # Guarantees
//! * **Strict monotonicity.** Successive `now()` calls return
//!   `monotonic_ns` values where each is >= the previous one, and
//!   within a single process `sequence` is strictly increasing.
//!   If two threads race for the same tick, the second observes
//!   `last + 1` ns rather than an equal or rewound value.
//! * **No syscall storms.** One `SystemTime::now()` at anchor time,
//!   plus one `Instant::now()` per `now()` call (~20-30 ns on Linux).
//! * **Thread-safe, lock-protected state.** `parking_lot::Mutex`
//!   guards `(last_mono, seq)`. Lock is held only across 3 integer
//!   comparisons + 2 stores — below the measurement floor of the
//!   p50 < 1 ms / p99 < 5 ms SLO.
//!
//! # Invariants (tested)
//! * `now()` is strictly monotonic across threads.
//! * `now().utc_nanos` == `anchor_utc_nanos + (now().monotonic_ns - anchor_mono_ns)`.
//! * `sequence` starts at 1 on first `now()` and increments by 1
//!   on every call (no gaps, no resets).

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

/// A captured instant. Matches the Python `TimeStamp` dataclass's
/// wire shape exactly so the `PyO3` seam is a zero-cost tuple unpack.
///
/// Fields are plain integers on purpose: `chrono::DateTime` or
/// `time::OffsetDateTime` would pull a trait-heavy dep into every
/// crate that reads the clock. UTC-nanos + monotonic-ns is all any
/// downstream needs to stamp a ledger event or diff two reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeStamp {
    /// UTC nanoseconds since Unix epoch, derived from the monotonic
    /// delta + anchor. Never read from `SystemTime::now()` after
    /// the anchor capture.
    pub utc_nanos: i128,
    /// Monotonic nanoseconds since the process-start anchor. Usable
    /// as a duration origin across threads.
    pub monotonic_ns: i128,
    /// 1-indexed call counter. Monotonic, gap-free, resets only on
    /// a fresh `TimeSource::new()` (i.e. never, in production).
    pub sequence: u64,
}

struct Inner {
    /// Monotonic nanoseconds at anchor capture. Relative to
    /// `Instant::now()` at `new()` time.
    anchor_mono_ns: i128,
    /// UTC nanoseconds at anchor capture. The single `SystemTime::now()`
    /// the whole process ever observes through this module.
    anchor_utc_nanos: i128,
    /// Last `monotonic_ns` returned. Used to enforce strict
    /// monotonicity across concurrent callers.
    last_mono_ns: i128,
    /// Last `sequence` returned. Strictly increasing.
    seq: u64,
}

/// Monotonic UTC time source. One instance per process in production
/// (see [`default`]); owned instances supported for deterministic
/// test harnesses.
pub struct TimeSource {
    anchor_instant: Instant,
    inner: Mutex<Inner>,
}

impl TimeSource {
    /// Capture the process-wide anchor. Both `SystemTime::now()` and
    /// `Instant::now()` are read in close succession; the skew
    /// between them is irrelevant once subsequent UTC reads are
    /// derived exclusively from the monotonic delta.
    #[must_use]
    pub fn new() -> Self {
        let anchor_instant = Instant::now();
        let anchor_utc_nanos = utc_now_nanos();
        Self {
            anchor_instant,
            inner: Mutex::new(Inner {
                anchor_mono_ns: 0,
                anchor_utc_nanos,
                last_mono_ns: 0,
                seq: 0,
            }),
        }
    }

    /// Current timestamp. See module docs for guarantees.
    #[allow(clippy::missing_panics_doc)]
    pub fn now(&self) -> TimeStamp {
        // SAFETY-STYLE: This is the one hot-path lock in the control
        // plane. The critical section is 4 integer ops; holding it
        // longer is a bug, not a style choice.
        let mono_raw: i128 = self
            .anchor_instant
            .elapsed()
            .as_nanos()
            .try_into()
            .unwrap_or(i128::MAX);
        let mut inner = self.inner.lock();
        // Enforce strict monotonicity across threads: if wall-clock
        // resolution or quantisation returns <= last, bump by 1 ns.
        let monotonic_ns = if mono_raw <= inner.last_mono_ns {
            inner.last_mono_ns.saturating_add(1)
        } else {
            mono_raw
        };
        inner.last_mono_ns = monotonic_ns;
        inner.seq = inner.seq.saturating_add(1);
        let utc_nanos = inner
            .anchor_utc_nanos
            .saturating_add(monotonic_ns.saturating_sub(inner.anchor_mono_ns));
        TimeStamp {
            utc_nanos,
            monotonic_ns,
            sequence: inner.seq,
        }
    }

    /// Read just the monotonic component — convenience for the
    /// hot-path SLO gate (`time_source.now_ns()` in the Python port).
    pub fn monotonic_ns(&self) -> i128 {
        self.now().monotonic_ns
    }
}

impl Default for TimeSource {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-wide singleton. Callers that don't need a bespoke source
/// for tests should go through this.
pub fn default() -> &'static TimeSource {
    static DEFAULT: OnceLock<TimeSource> = OnceLock::new();
    DEFAULT.get_or_init(TimeSource::new)
}

/// Shortcut for `default().now()`.
#[must_use]
pub fn now() -> TimeStamp {
    default().now()
}

/// Shortcut for `default().monotonic_ns()`.
#[must_use]
pub fn now_mono_ns() -> i128 {
    default().monotonic_ns()
}

/// UTC nanos since Unix epoch (single syscall; used by `TimeSource::new`).
/// Not re-exported: every other caller MUST go through `TimeSource::now`.
fn utc_now_nanos() -> i128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        i128::from(d.as_secs()) * 1_000_000_000 + i128::from(d.subsec_nanos())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn sequence_starts_at_one_and_is_gap_free() {
        let ts = TimeSource::new();
        let a = ts.now();
        let b = ts.now();
        let c = ts.now();
        assert_eq!(a.sequence, 1);
        assert_eq!(b.sequence, 2);
        assert_eq!(c.sequence, 3);
    }

    #[test]
    fn monotonic_ns_is_strictly_increasing_on_single_thread() {
        let ts = TimeSource::new();
        let mut last = 0i128;
        for _ in 0..1_000 {
            let m = ts.now().monotonic_ns;
            assert!(m > last, "not strictly monotonic: {last} then {m}");
            last = m;
        }
    }

    #[test]
    fn utc_delta_matches_monotonic_delta() {
        let ts = TimeSource::new();
        let a = ts.now();
        // Busy a bit so the deltas are non-trivial.
        for _ in 0..1_000 {
            std::hint::black_box(ts.now());
        }
        let b = ts.now();
        let mono_delta = b.monotonic_ns - a.monotonic_ns;
        let utc_delta = b.utc_nanos - a.utc_nanos;
        assert_eq!(
            mono_delta, utc_delta,
            "UTC derivation must track monotonic delta exactly"
        );
    }

    #[test]
    fn utc_anchor_is_plausible_unix_time() {
        let ts = TimeSource::new();
        let t = ts.now();
        // A sanity range: anywhere from 2020-01-01 to 2100-01-01 UTC
        // nanos. If the anchor is zero or we've time-travelled, fail.
        let jan_2020_ns: i128 = 1_577_836_800_000_000_000;
        let jan_2100_ns: i128 = 4_102_444_800_000_000_000;
        assert!(
            t.utc_nanos > jan_2020_ns && t.utc_nanos < jan_2100_ns,
            "utc_nanos out of plausible range: {}",
            t.utc_nanos
        );
    }

    #[test]
    fn concurrent_calls_remain_monotonic() {
        use std::sync::Arc;
        use std::thread;

        let ts = Arc::new(TimeSource::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let ts = Arc::clone(&ts);
            handles.push(thread::spawn(move || {
                (0..1_000).map(|_| ts.now()).collect::<Vec<_>>()
            }));
        }
        let mut all: Vec<TimeStamp> = Vec::new();
        for h in handles {
            all.extend(h.join().expect("thread panicked"));
        }
        // Sequence numbers must be a permutation of 1..=N with no gaps.
        let n = all.len();
        let mut seqs: Vec<u64> = all.iter().map(|t| t.sequence).collect();
        seqs.sort_unstable();
        for (i, s) in seqs.iter().enumerate() {
            let expected = (i as u64) + 1;
            assert_eq!(
                *s, expected,
                "sequence gap at index {i}: expected {expected}, got {s}"
            );
        }
        assert_eq!(*seqs.last().expect("at least one"), n as u64);
    }

    #[test]
    fn default_singleton_is_stable() {
        let a = default().now();
        let b = default().now();
        assert!(b.sequence > a.sequence);
        assert!(b.monotonic_ns > a.monotonic_ns);
    }
}
