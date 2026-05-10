# `tools/rust_bridge/` — B-20 / I-38 PATTERN_ONLY Rust hot-path template

> **Classification:** PATTERN_ONLY · **Status:** template, **not** a live backend.
>
> Adapted from [PyO3 guide](https://pyo3.rs/) (`class.md`, `function.md`,
> `types.md`, `parallelism.md`) and the
> [maturin docs](https://www.maturin.rs/).
>
> ```
> Repo:    https://github.com/PyO3/pyo3 + https://github.com/PyO3/maturin
> License: MIT
> pip:     pip install maturin
> ```

## Why this lives here and is **not** wired into Python

Reviewer #3 (audit v3, item 1) flagged the previous dual-backend
Python+Rust state of `execution_engine.hot_path.fast_execute` as a
determinism hazard against `INV-15` (byte-identical replay) and
`TEST-01`. The conservative resolution per the operator's directive
was to delete the Rust crates and run a 30-day shadow window on the
Python-only path before revisiting — see
[`docs/rust_revival_schedule.yaml`](../../docs/rust_revival_schedule.yaml)
and [`tools/rust_revival_reminder.py`](../rust_revival_reminder.py).

This crate is the canonical **template** the eventual revival PR must
match. It is intentionally:

- **Not** a member of any Python build (`pyproject.toml` does not
  reference it, no `maturin develop` is wired into the launcher).
- **Not** imported by `execution_engine.hot_path` or any other engine.
- **Not** exempted from the Rust deletion (no shadow toggle, no
  `prefer_rust` parameter resurrection).

`FastExecutor.execute()` remains Python-only until the revival
checklist in
[`docs/rust_revival_schedule.yaml`](../../docs/rust_revival_schedule.yaml)
holds.

## What the template demonstrates

Per [B-20 spec lines 2283–2318](../../DIX_MASTER_CANONICAL.md) the
template must show:

| # | PyO3 / maturin technique                       | Where in this crate                            |
|---|-----------------------------------------------|------------------------------------------------|
| 1 | `#[pyclass]` exposing a Rust struct           | `RustRiskSnapshot` in `src/lib.rs`             |
| 2 | `#[pyfunction]` + `#[pymodule]`               | `execute` + `dixvision_rust_bridge` module     |
| 3 | Releasing the GIL for pure-Rust compute       | `Python::allow_threads` around the gate body   |
| 4 | Exact Python-interface shape                  | `cap_for` + 6 named fields matching `RiskSnapshot` |
| 5 | Branch order matching `_execute_python`       | `fast_execute_gate` in `src/lib.rs`            |
| 6 | Release-profile build for hot-path latency    | `[profile.release]` in `Cargo.toml`            |

### Build profile (per `Cargo.toml`)

```
[profile.release]
lto         = "fat"        # cross-crate inlining (reject ladder)
codegen-units = 1          # single-unit codegen for best inlining
panic       = "abort"      # drop unwind tables from the .so
opt-level   = 3
strip       = "symbols"
```

`lto = "fat"` + `codegen-units = 1` collapse the seven-branch reject
ladder into a single dispatch table; `panic = "abort"` removes the
unwind machinery you would otherwise pay for on every Python ↔ Rust
boundary crossing.

## Compile

```bash
# Plain Rust compile (no Python wheel, runs cargo unit tests).
cargo test --release --manifest-path tools/rust_bridge/Cargo.toml

# Build a Python wheel locally (consumer side, post-revival only).
pip install maturin
maturin build --release --manifest-path tools/rust_bridge/Cargo.toml
```

The wheel is **not** installed into the venv automatically; the
revival PR is the first time `maturin develop` will be wired into a
launcher script.

## Bench

```bash
cargo run --release --manifest-path tools/rust_bridge/Cargo.toml \
          --bin fast_risk_cache_bench
```

Emits one line: `<branch> <ns_per_call> <approved>/<N>`. The revival
checklist requires the Python p50 to exceed the Rust p50 and the
Python p99 to exceed 5 ms on the gate before any merge.

## Python-vs-Rust shadow-equivalence harness (revival prerequisite)

**Before any revival PR is reviewed**, the following must hold (see
[`docs/rust_revival_schedule.yaml`](../../docs/rust_revival_schedule.yaml)):

- [ ] At least one full week of LIVE-or-CANARY trading on the
      Python-only hot path with no replay or invariant breaches.
- [ ] A documented bit-identical shadow-equivalence harness
      (`tests/test_fast_execute_parity.py` or successor) covering
      every branch under both backends.
- [ ] A latency benchmark proving the Rust port crosses a measurable
      hot-path bottleneck (`p50 < Python p50`, `p99 < 5 ms`).
- [ ] A deletion plan: the Python path is removed in the **same** PR
      that adds the Rust port (no dual-backend state ever again).
- [ ] An update to the revival schedule (or its successor) so the
      next window is enforced.

The proof harness is the *first* code that should land — bit-identical
parity for every reject branch, three-run replay digest equality
through both backends, deterministic order on the seed/timestamp
axes. Only after the harness is green for one week of replay does
the Rust port replace the Python path.

## Branch order is canonical

Both backends MUST evaluate the reject ladder in this order (mirrors
`_execute_python` in
[`execution_engine/hot_path/fast_execute.py`](../../execution_engine/hot_path/fast_execute.py)):

1. `halted`              → `REJECTED_LIMIT` (`reason="halted"`)
2. `ts_ns` staleness     → `REJECTED_RISK_STALE`
3. `mark_price <= 0.0`   → `REJECTED_NO_MARK`
4. `confidence` floor    → `REJECTED_LOW_CONFIDENCE`
5. `side == HOLD`        → `REJECTED_HOLD`
6. `qty > cap`           → `REJECTED_LIMIT` (`reason="qty_above_cap"`)
7. otherwise             → `APPROVED`

Any reordering breaks bit-identical replay and is a `INV-15` violation.

## DIX invariants this template is built to satisfy

| Invariant | How it is honoured in the template                                                  |
|-----------|--------------------------------------------------------------------------------------|
| `INV-15`  | Pure function over primitive args; no clock / random / IO.                           |
| `INV-17`  | T1 hot-path purity: no governance / intelligence / system imports.                   |
| `INV-69`  | Output carries `risk_version`; revival wrapper produces `produced_by_engine=…`.      |
| B27/B28   | Template does **not** construct typed events; the Python wrapper does that slow-path.|

## Filesystem layout

```
tools/rust_bridge/
├── Cargo.toml                              # build template + release profile
├── README.md                               # this file
└── src/
    ├── lib.rs                              # #[pyclass] + #[pyfunction] + gate
    └── bin/
        └── fast_risk_cache_bench.rs        # standalone Rust micro-bench
```

`tools/rust_bridge/` is **not** a workspace member and **not** part of
`tools/authority_lint.py` scanning (the lint walks `*.py` only). It is
a documentation+template surface.
