# Manifest v3.5.5 — Wave 5: Strategic Execution

This delta closes Wave 5 from the locked sequence
`scvs-2 → scvs-3 → authority → constraint → wave-5 → behavior`
(CRL deferred per operator direction at the post-PR-#60 pivot point).

## 0. Why v3.5.5 exists

Until now, the executor handled child-order lifecycles but did **not**
plan the slicing of large parent intents. A naive engine would push
the entire parent quantity at the venue in one shot, paying maximum
market impact. Real desks slice the parent across time and trade off
two costs:

* **temporary impact** — the spread the executor pays per unit of
  *trading speed*;
* **timing risk** — the variance the executor accumulates as the
  unfilled position is exposed to volatility.

Wave 5 introduces the canonical **Almgren-Chriss** scheduler as the
first member of the new strategic-execution layer. Future revisions
may add VWAP, POV, and Implementation-Shortfall variants behind the
same `ExecutionSchedule` shape.

## 1. Specification deltas

### 1.1 INV-62 — strategic execution is pure

> Every strategic-execution scheduler MUST be a pure function of its
> inputs. No clock reads, no PRNG, no I/O. INV-15 applies.

### 1.2 SAFE-61 — schedule conserves the parent quantity

> The sum of the child slice quantities MUST equal the parent
> ``quantity`` to floating-point precision, and the holdings after the
> final slice MUST be zero. The executor MUST NOT consume a schedule
> whose total drifts.

### 1.3 SAFE-62 — sign-preserving

> Every child slice MUST share the sign of the parent quantity. A
> liquidation parent never produces an acquisition slice.

### 1.4 PERF-03 — schedule complexity

> Solver complexity MUST be O(N) in the number of slices. The
> Almgren-Chriss closed-form satisfies this trivially; future variants
> MUST not regress past O(N log N).

## 2. New artefacts

* `execution_engine/strategic/almgren_chriss.py` — closed-form solver
  with numerically stable `sinh` ratio for large kappa·T (no overflow
  for any realistic risk-aversion).
* `execution_engine/strategic/__init__.py` — public surface.
* `tests/test_strategic_execution.py` — schedule shape, limiting cases
  (TWAP, front-load), determinism, validation, mathematical
  invariants.

## 3. Solver model

For a parent of size ``X`` to be worked over horizon ``T`` in ``N``
equal-length slices of duration ``τ = T/N``:

* permanent linear impact ``γ·v``
* temporary linear impact ``η·v``
* return volatility ``σ`` (per unit time)
* risk-aversion ``λ``

Define ``η̃ = η - γτ/2`` and ``κ² = λσ² / η̃``. The optimal holdings
are

```
x_k = sinh(κ(T - kτ)) / sinh(κT) · X
```

and the slice trade is ``n_k = x_{k-1} - x_k``.

* As ``λ → 0`` (risk-neutral) the schedule degenerates to TWAP.
* As ``λ → ∞`` the schedule front-loads.
* When ``η̃ ≤ 0`` (permanent impact too large for the chosen slicing)
  the solver rejects with `ValueError`.

The solver is INV-15 deterministic and returns a frozen
`ExecutionSchedule` whose slice tuple is replay-stable.

## 4. Scope

### In

* The strategic-execution scheduler + tests + docs.
* INV-62, SAFE-61/62, PERF-03 spec rows.

### Out (deferred, in the new committed order)

* **Wiring `solve_almgren_chriss` into the executor's parent-order
  planner.** Wave 5 ships the math. The executor adopts it in the
  next phase.
* **CRL** — deferred per operator direction at the post-PR-#60 pivot.
* **Behavior priorities P2-P5** — closed learning loop, System →
  Governance hard coupling, decision trace, and evolution wiring.
  These are the next focus per the operator's "no more architecture"
  direction.

### Unchanged

* Authority matrix (v3.5.3).
* Constraint engine (v3.5.4).
* SCVS surfaces (v3.5.0–v3.5.2).
* Triad Lock (v3.4 + INV-56 + B20/B21/B22).
