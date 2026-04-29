# `rust/` — DIX VISION v42.2 polyglot staging area

This directory hosts the Rust workspace for the polyglot migration
described in [`docs/DIX_VISION_v42_2_COMPILED.md`](../docs/DIX_VISION_v42_2_COMPILED.md).

It is a **staging area**: crates live at `rust/<domain>/` while the
equivalent Python modules still sit at `<domain>/`. Each port PR
lands one module in Rust and deletes the corresponding `.py` file in
the same commit. When the migration is complete, crates may be
relocated to the manifest's target tree (`<domain>/Cargo.toml`
directly); that move is cosmetic and changes no imports.

## Workspace members

| Crate                                       | Domain       | Manifest section                       |
| ------------------------------------------- | ------------ | -------------------------------------- |
| [`dixvision-execution`](./execution)        | `execution/` | § EXECUTION (RUST — Dyon)              |
| [`dixvision-system`](./system)              | `system/`    | § SYSTEM (RUST — CONTROL PLANE)        |
| [`dixvision-bootstrap`](./bootstrap)        | `bootstrap/` | § BOOTSTRAP (RUST)                     |

## Conventions

- **Edition 2021**, MSRV pinned in `[workspace.package].rust-version`.
- **No `unsafe`** outside documented FFI seams (`PyO3` glue). Enforced
  in three independent ways:
  1. `unsafe_code = "deny"` workspace lint.
  2. `#![forbid(unsafe_code)]` at every crate root.
  3. A `no_unsafe` grep gate in `.github/workflows/rust.yml` that
     rejects any `unsafe` block in `rust/**/*.rs`. When a PyO3 seam
     eventually needs `unsafe`, the exemption is added inline with a
     `// SAFETY:` comment and the grep gate is relaxed to ignore
     lines containing `// SAFETY:`.
- **No panics on the hot path.** `clippy::panic`, `unwrap_used`,
  `expect_used`, `indexing_slicing`, `integer_division`, `float_cmp`,
  and `unreachable` are all denied or warned at workspace level.
  Test code may opt out with `#[allow(clippy::...)]`.
- **`Result<_, CrateError>` for every public surface.** Each crate
  owns its canonical error enum (`ExecutionError`, `SystemError`,
  `BootError`). Errors are `#[non_exhaustive]` so port PRs can add
  variants without breaking callers at compile time.
- **Version pins.** All shared deps (`thiserror`, `prost`,
  `parking_lot`, …) live in `[workspace.dependencies]`. Member crates
  reference them via `{ workspace = true }`.

## Local commands

```sh
cd rust

cargo fmt --all --check           # style
cargo clippy --all-targets --all-features -- -D warnings
cargo test  --all --all-features --locked
```

CI runs all three plus a grep-based `unsafe` guard
(`.github/workflows/rust.yml`).

## Planned port sequence

Same as the Python Phase 1 PRs, in strict dependency order:

1. `system/time_source` → `dixvision_system::time_source` + PyO3 glue
   (canary for the FFI build pipeline; smallest blast radius).
2. `system/fast_risk_cache` → `dixvision_system::fast_risk_cache`.
3. `system/state_reconstructor` + `snapshots`.
4. `execution/adapters/base` (T0-8 circuit breakers).
5. `execution/hazard/*`.
6. `execution/chaos/chaos_engine` (T0-13).
7. `system/kill_switch` (T0-9).
8. `system/load_controller` (T0-2).
9. `system/config_*` + `feature_flags` + `fallback_manager`
   (T0-12 / T0-16).
10. `system/metrics` + Prometheus exporter (T0-10).
11. `bootstrap/kernel_boot.rs` — deterministic startup sequence.

`mind/` and `governance/` stay in Python per the manifest. `state/ledger/`
is ported to Go in a separate module (`state/ledger/go.mod`).
