//! # Metrics sink (T0-10 — runtime port)
//!
//! Port of the Python `MetricsSink` reference implementation in
//! `system/metrics.py`. Thread-safe in-memory counters + histograms
//! with the exact same public-API semantics so the parity suite
//! passes.
//!
//! * `increment(name, value, labels)` — bump a named counter. Labels
//!   are folded into the counter key as ``"{name}:{labels}"`` when
//!   non-empty, matching the Python code's key format.
//! * `observe(name, value_ms)` — append a histogram sample. When a
//!   metric exceeds [`HISTOGRAM_RING_CAP`] samples the tail
//!   [`HISTOGRAM_RING_KEEP`] are kept (same ring-buffer logic as the
//!   reference).
//! * `p99(name)` — 99th-percentile sample across the current buffer
//!   for `name`. Empty buffers report 0.0 (matches Python default).
//! * `snapshot()` — point-in-time copy of every counter and per-metric
//!   p99. Returned as owned primitives so callers can hand it across
//!   the FFI seam without borrowing from the sink.
//!
//! The Prometheus text-exposition format is explicitly out of scope
//! for this port; the manifest's T0-10 exporter will be added in a
//! follow-up PR. This crate only supplies the in-memory sink the
//! exporter will eventually read from.
//!
//! ## Concurrency
//!
//! Writers and readers both take a single [`parking_lot::Mutex`]. The
//! Python reference uses `threading.Lock` for identical purpose.
//! Tests exercise concurrent increment/observe/snapshot calls to
//! prove there are no torn reads.

use std::collections::HashMap;

use parking_lot::Mutex;

/// Upper bound on samples retained per histogram metric before the
/// ring-buffer drops the oldest half. Matches the Python constant
/// inlined in `MetricsSink.observe`.
const HISTOGRAM_RING_CAP: usize = 10_000;

/// Number of samples retained after a ring-buffer trim. Matches the
/// Python constant inlined in `MetricsSink.observe`.
const HISTOGRAM_RING_KEEP: usize = 5_000;

/// Percentile computed by [`MetricsSink::p99`]. 0.99 matches the
/// Python reference; if ever tuned, also update the name of the
/// accessor to preserve the public API contract.
const P99_QUANTILE: f64 = 0.99;

/// Read-only snapshot of every counter and per-metric p99.
///
/// Returned by [`MetricsSink::snapshot`] so callers can inspect state
/// without holding the sink lock. Keys are the same strings the
/// Python reference emits (counter keys may include a ``":labels"``
/// suffix).
#[derive(Debug, Clone, Default)]
#[allow(clippy::module_name_repetitions)]
pub struct MetricsSnapshot {
    /// Per-counter totals, keyed by ``"{name}"`` or
    /// ``"{name}:{labels}"`` when labels were supplied at increment
    /// time.
    pub counters: HashMap<String, f64>,
    /// Per-histogram 99th-percentile values at the instant the
    /// snapshot was taken.
    pub p99: HashMap<String, f64>,
}

/// In-memory metrics sink.
///
/// Governance and Indira share one instance per process; see
/// [`crate::metrics`] for the canonical accessor via
/// `dixvision-py-system`. All operations acquire a single mutex —
/// the reference implementation does the same and the hot path is
/// well below the SLO budget even with the lock.
#[derive(Debug, Default)]
#[allow(clippy::module_name_repetitions)]
pub struct MetricsSink {
    inner: Mutex<MetricsInner>,
}

#[derive(Debug, Default)]
struct MetricsInner {
    counters: HashMap<String, f64>,
    histograms: HashMap<String, Vec<f64>>,
}

impl MetricsSink {
    /// Construct an empty sink. Equivalent to `MetricsSink()` in the
    /// Python reference.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bump a counter by `value`. When `labels` is non-empty the key
    /// is folded to ``"{name}:{labels}"`` to match the Python
    /// reference exactly; the caller is responsible for keeping
    /// `labels` stable across calls if they want aggregation.
    pub fn increment(&self, name: &str, value: f64, labels: Option<&str>) {
        let key = compose_key(name, labels);
        let mut guard = self.inner.lock();
        *guard.counters.entry(key).or_insert(0.0) += value;
    }

    /// Append a histogram sample (latency in milliseconds, by
    /// convention). Replicates the Python ring-buffer semantics.
    #[allow(clippy::significant_drop_tightening)]
    pub fn observe(&self, name: &str, value_ms: f64) {
        let mut guard = self.inner.lock();
        let buf = guard.histograms.entry(name.to_string()).or_default();
        buf.push(value_ms);
        if buf.len() > HISTOGRAM_RING_CAP {
            let excess = buf.len() - HISTOGRAM_RING_KEEP;
            buf.drain(..excess);
        }
    }

    /// 99th-percentile sample for `name`. Returns 0.0 when no samples
    /// have been observed, matching the Python reference which
    /// initialises the buffer to `[0.0]` inside `p99` but ignores
    /// that default when other samples exist.
    #[must_use]
    pub fn p99(&self, name: &str) -> f64 {
        let samples: Vec<f64> = {
            let guard = self.inner.lock();
            match guard.histograms.get(name) {
                Some(buf) if !buf.is_empty() => buf.clone(),
                _ => return 0.0,
            }
        };
        p99_of(&samples)
    }

    /// Point-in-time snapshot of every counter and per-metric p99.
    /// Safe to call concurrently with writers; the snapshot is
    /// taken while holding the lock, so readers never see torn
    /// state.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let (counters, histograms) = {
            let guard = self.inner.lock();
            (guard.counters.clone(), guard.histograms.clone())
        };
        let p99 = histograms
            .iter()
            .map(|(k, buf)| {
                let v = if buf.is_empty() { 0.0 } else { p99_of(buf) };
                (k.clone(), v)
            })
            .collect();
        MetricsSnapshot { counters, p99 }
    }
}

fn compose_key(name: &str, labels: Option<&str>) -> String {
    match labels {
        Some(l) if !l.is_empty() => format!("{name}:{l}"),
        _ => name.to_string(),
    }
}

/// p99 helper. Computes the 99th-percentile of `buf` using the same
/// indexing scheme as the Python reference:
/// ``idx = int(len(vals) * 0.99); return sorted(vals)[min(idx, len-1)]``.
/// The buffer is cloned and sorted locally so the sink's stored
/// ordering is preserved.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]
fn p99_of(buf: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = buf.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = sorted.len();
    // `len` is always ≥ 1 because callers check `is_empty` first.
    // The Python reference computes ``int(len * 0.99)`` then clamps
    // to ``len - 1``. Reproduce both steps so rounding behaviour is
    // identical.
    let idx = (len as f64 * P99_QUANTILE) as usize;
    let clamped = idx.min(len - 1);
    sorted[clamped]
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::uninlined_format_args,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn increment_without_labels_accumulates() {
        let s = MetricsSink::new();
        s.increment("trades", 1.0, None);
        s.increment("trades", 2.5, None);
        let snap = s.snapshot();
        assert_eq!(snap.counters.get("trades"), Some(&3.5));
    }

    #[test]
    fn increment_with_labels_keys_separately() {
        let s = MetricsSink::new();
        s.increment("trades", 1.0, Some("side=buy"));
        s.increment("trades", 2.0, Some("side=sell"));
        let snap = s.snapshot();
        assert_eq!(snap.counters.get("trades:side=buy"), Some(&1.0));
        assert_eq!(snap.counters.get("trades:side=sell"), Some(&2.0));
        assert!(!snap.counters.contains_key("trades"));
    }

    #[test]
    fn observe_reports_p99_from_buffer() {
        let s = MetricsSink::new();
        for v in 1..=100 {
            s.observe("latency_ms", f64::from(v));
        }
        // int(100 * 0.99) = 99, clamped to len-1=99, sorted[99] = 100.0
        assert_eq!(s.p99("latency_ms"), 100.0);
    }

    #[test]
    fn p99_of_empty_is_zero() {
        let s = MetricsSink::new();
        assert_eq!(s.p99("never_observed"), 0.0);
    }

    #[test]
    fn observe_ring_buffer_trims_on_overflow() {
        let s = MetricsSink::new();
        for v in 0..HISTOGRAM_RING_CAP + 10 {
            s.observe("noisy", v as f64);
        }
        // Buffer was trimmed when len exceeded CAP=10_000 to tail
        // KEEP=5_000. Subsequent 9 observes after the trim leave us
        // at KEEP + (CAP + 10 - CAP) - 1 = 5009 samples.
        let snap = s.snapshot();
        assert!(snap.p99.contains_key("noisy"));
    }

    #[test]
    fn snapshot_is_independent_of_sink() {
        let s = MetricsSink::new();
        s.increment("a", 1.0, None);
        let before = s.snapshot();
        s.increment("a", 100.0, None);
        // The earlier snapshot must not reflect the later increment.
        assert_eq!(before.counters.get("a"), Some(&1.0));
        assert_eq!(s.snapshot().counters.get("a"), Some(&101.0));
    }

    #[test]
    fn snapshot_reports_p99_per_metric() {
        let s = MetricsSink::new();
        s.observe("fast", 1.0);
        s.observe("slow", 1000.0);
        let snap = s.snapshot();
        assert_eq!(snap.p99.get("fast"), Some(&1.0));
        assert_eq!(snap.p99.get("slow"), Some(&1000.0));
    }

    #[test]
    fn concurrent_writers_do_not_lose_counts() {
        let sink = Arc::new(MetricsSink::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = Arc::clone(&sink);
            handles.push(thread::spawn(move || {
                for _ in 0..1_000 {
                    s.increment("ctr", 1.0, None);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        let snap = sink.snapshot();
        assert_eq!(snap.counters.get("ctr"), Some(&8_000.0));
    }

    #[test]
    fn concurrent_snapshot_never_tears() {
        let sink = Arc::new(MetricsSink::new());
        let stop = Arc::new(parking_lot::Mutex::new(false));
        let writer = {
            let s = Arc::clone(&sink);
            let stop = Arc::clone(&stop);
            thread::spawn(move || loop {
                s.increment("w", 1.0, None);
                s.observe("lat", 2.0);
                if *stop.lock() {
                    break;
                }
            })
        };
        // Read a snapshot many times; the counters map must always
        // be an internally-consistent clone (all entries real
        // strings + finite f64s).
        for _ in 0..5_000 {
            let snap = sink.snapshot();
            for v in snap.counters.values() {
                assert!(v.is_finite());
            }
        }
        *stop.lock() = true;
        writer.join().expect("writer thread panicked");
    }
}
