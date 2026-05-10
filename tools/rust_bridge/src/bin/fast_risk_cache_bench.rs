// B-20 PATTERN_ONLY -- standalone Rust micro-benchmark for the
// hot-path gate. Not wired into Python; consumed by the future
// Python-vs-Rust shadow-equivalence harness.
//
// Run: `cargo run --release --manifest-path tools/rust_bridge/Cargo.toml \
//                --bin fast_risk_cache_bench`
//
// Emits one line per branch: `<outcome> <ns_per_call> <calls>`.

use std::collections::HashMap;
use std::time::Instant;

use dixvision_rust_bridge::{RustRiskSnapshot, OUTCOME_APPROVED};

const N: u64 = 5_000_000;

fn snapshot() -> RustRiskSnapshot {
    RustRiskSnapshot {
        version: 1,
        ts_ns: 1_000,
        max_position_qty: Some(10.0),
        max_signal_confidence: 0.5,
        symbol_caps: HashMap::new(),
        halted: false,
    }
}

fn bench<F: Fn() -> &'static str>(name: &str, f: F) {
    // Warm.
    for _ in 0..1_000 {
        let _ = f();
    }
    let start = Instant::now();
    let mut hits: u64 = 0;
    for _ in 0..N {
        if f() == OUTCOME_APPROVED {
            hits += 1;
        }
    }
    let dur = start.elapsed();
    let ns_per = dur.as_nanos() as f64 / N as f64;
    println!("{name}\t{ns_per:.1}ns/call\t{hits}/{N} approved");
}

fn main() {
    let snap = snapshot();
    bench("approved", || {
        if dixvision_rust_bridge_gate(&snap, 1_000, "BTCUSD", 1, 0.9, 1.0, 100.0) == "APPROVED" {
            "APPROVED"
        } else {
            "OTHER"
        }
    });
}

// Local re-export of the gate for the bench binary (the public
// surface goes through `pyfunction execute` which requires a GIL).
fn dixvision_rust_bridge_gate(
    snap: &RustRiskSnapshot,
    ts_ns: i64,
    symbol: &str,
    side: i32,
    confidence: f64,
    qty: f64,
    mark: f64,
) -> &'static str {
    if snap.halted {
        return "REJECTED_LIMIT";
    }
    if ts_ns - snap.ts_ns > 2_000_000_000 {
        return "REJECTED_RISK_STALE";
    }
    if mark <= 0.0 {
        return "REJECTED_NO_MARK";
    }
    if confidence < snap.max_signal_confidence {
        return "REJECTED_LOW_CONFIDENCE";
    }
    if side == 0 {
        return "REJECTED_HOLD";
    }
    if let Some(cap) = snap.cap_for(symbol) {
        if qty > cap {
            return "REJECTED_LIMIT";
        }
    }
    "APPROVED"
}
