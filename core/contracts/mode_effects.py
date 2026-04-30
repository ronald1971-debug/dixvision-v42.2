"""GOV-CP-08 — Mode Effect Table (Wave-04.6 PR-A).

Single canonical mapping from :class:`SystemMode` to a frozen
:class:`ModeEffect` record describing **what every engine does
differently in this mode**. Every engine that conditions on mode must
import its decision from here rather than hard-coding mode comparisons
(enforced by lint rule **B31** in ``tools/authority_lint.py``).

The 7 modes already exist in ``core/contracts/governance.py`` and the
forward-chain FSM is enforced by
``governance_engine/control_plane/state_transition_manager.py``. Wave-
04.6 PR-A's job is *not* to extend the FSM (it is already the right
shape) but to make each state behaviourally distinct in a single
auditable place.

Determinism contract: the table is content-hashed via
:func:`mode_effect_table_hash` so replayers and the bootstrap
orchestrator can install one ``MODE_EFFECTS_INSTALLED`` ledger row per
table version (SAFE-47-style anchor; INV-15).

Reference values (one row per mode, mirroring ``docs/wave_04_6_plan.md``
§2 PR-A):

================  =======  =====  =========  ===========  ============  =======  ================
mode              signals  exec   size_cap   learn_emit   learn_apply   op_auth  oversight
================  =======  =====  =========  ===========  ============  =======  ================
LOCKED            False    False  0%         False        False         n/a      none
SAFE              False    False  0%         False        False         False    per_trade
PAPER             True     paper  0%         True         False         False    per_trade
SHADOW            True     False  0%         True         False         False    per_trade
CANARY            True     True   1%         True         True          True     per_trade
LIVE              True     True   None       True         True          True     per_trade
AUTO              True     True   None       True         True          True     exception_only
================  =======  =====  =========  ===========  ============  =======  ================

The table is *declarative*. PR-B / PR-C / PR-E in the same wave wire
each row into the engines that act on it; this PR only ships the table
+ the lint rule that bans hard-coded mode comparisons in the affected
modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from core.contracts.governance import SystemMode

# ---------------------------------------------------------------------------
# ModeEffect record
# ---------------------------------------------------------------------------

OversightKind = Literal["per_trade", "exception_only", "none"]


@dataclass(frozen=True, slots=True)
class ModeEffect:
    """Per-mode behaviour cell.

    Every field is read by exactly one engine layer:

    * ``signals_emit`` — IntelligenceEngine emits :class:`SignalEvent`
      iff ``True``. Read by ``intelligence_engine.engine``.
    * ``executions_dispatch`` — ExecutionEngine dispatches an
      :class:`ExecutionIntent` to a broker iff ``True``. Read by
      ``execution_engine.engine``.
    * ``size_cap_pct`` — PolicyEngine clamps the notional fraction of
      the account on every ``ExecutionIntent`` to this value (in
      percent of equity); ``None`` means uncapped. Read by
      ``governance_engine.control_plane.policy_engine``.
    * ``learning_emit`` — :class:`UpdateEmitter` is unfrozen iff
      ``True`` (HARDEN-04 / INV-70). Read by
      ``learning_engine.update_emitter`` via
      ``LearningEvolutionFreezePolicy``.
    * ``learning_apply`` — UpdateValidator may ratify
      ``UPDATE_PROPOSED`` iff ``True``. Wired by Wave-04.6 PR-E.
    * ``operator_auth_required`` — Forward ratchet *into* this mode
      requires ``operator_authorized=True`` on the
      :class:`ModeTransitionRequest`. Read by ``policy_engine``.
    * ``oversight_kind`` — How operator approval gates ordinary
      ``OperatorAction``: per-trade prompts, exception-only (alerts
      on hazard), or none (LOCKED). Wired by Wave-04.6 PR-F.
    """

    signals_emit: bool
    executions_dispatch: bool
    size_cap_pct: float | None
    learning_emit: bool
    learning_apply: bool
    operator_auth_required: bool
    oversight_kind: OversightKind


# ---------------------------------------------------------------------------
# Canonical table
# ---------------------------------------------------------------------------

_MODE_EFFECTS_RAW: dict[SystemMode, ModeEffect] = {
    SystemMode.LOCKED: ModeEffect(
        signals_emit=False,
        executions_dispatch=False,
        size_cap_pct=0.0,
        learning_emit=False,
        learning_apply=False,
        operator_auth_required=False,
        oversight_kind="none",
    ),
    SystemMode.SAFE: ModeEffect(
        signals_emit=False,
        executions_dispatch=False,
        size_cap_pct=0.0,
        learning_emit=False,
        learning_apply=False,
        operator_auth_required=False,
        oversight_kind="per_trade",
    ),
    SystemMode.PAPER: ModeEffect(
        signals_emit=True,
        executions_dispatch=True,  # paper broker dispatch, no live venue
        size_cap_pct=0.0,
        learning_emit=True,
        learning_apply=False,
        operator_auth_required=False,
        oversight_kind="per_trade",
    ),
    SystemMode.SHADOW: ModeEffect(
        signals_emit=True,
        executions_dispatch=False,  # signals fire, no broker dispatch
        size_cap_pct=0.0,
        learning_emit=True,
        learning_apply=False,
        operator_auth_required=False,
        oversight_kind="per_trade",
    ),
    SystemMode.CANARY: ModeEffect(
        signals_emit=True,
        executions_dispatch=True,
        size_cap_pct=1.0,  # 1% of equity per intent
        learning_emit=True,
        learning_apply=True,
        operator_auth_required=True,
        oversight_kind="per_trade",
    ),
    SystemMode.LIVE: ModeEffect(
        signals_emit=True,
        executions_dispatch=True,
        size_cap_pct=None,
        learning_emit=True,
        learning_apply=True,
        operator_auth_required=True,
        oversight_kind="per_trade",
    ),
    SystemMode.AUTO: ModeEffect(
        signals_emit=True,
        executions_dispatch=True,
        size_cap_pct=None,
        learning_emit=True,
        learning_apply=True,
        operator_auth_required=True,
        oversight_kind="exception_only",
    ),
}


MODE_EFFECTS: Final[Mapping[SystemMode, ModeEffect]] = MappingProxyType(
    _MODE_EFFECTS_RAW
)
"""Read-only canonical view; mutate via release-time edit, never at runtime."""


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

MODE_EFFECTS_INSTALLED_KIND: Final[str] = "MODE_EFFECTS_INSTALLED"
"""Ledger kind produced when the table version is anchored at startup."""

MODE_EFFECTS_HASH_KEY: Final[str] = "table_hash"
"""Payload key for :data:`MODE_EFFECTS_INSTALLED_KIND` rows."""


def mode_effect_table_hash(
    table: Mapping[SystemMode, ModeEffect] = MODE_EFFECTS,
) -> str:
    """Stable SHA-256 over the canonical-sorted mode-effect table.

    Mirrors :func:`policy_engine._hash_decision_table` so the same
    auditing tooling can fingerprint both tables. Determinism contract:
    same inputs → same digest, byte-identical across replays.
    """

    h = hashlib.sha256()
    for mode in sorted(table, key=lambda m: int(m)):
        eff = table[mode]
        size_cap = "" if eff.size_cap_pct is None else f"{eff.size_cap_pct:.6f}"
        h.update(
            "|".join(
                (
                    f"mode={int(mode)}",
                    f"signals_emit={int(eff.signals_emit)}",
                    f"executions_dispatch={int(eff.executions_dispatch)}",
                    f"size_cap_pct={size_cap}",
                    f"learning_emit={int(eff.learning_emit)}",
                    f"learning_apply={int(eff.learning_apply)}",
                    f"operator_auth_required={int(eff.operator_auth_required)}",
                    f"oversight_kind={eff.oversight_kind}",
                )
            ).encode()
        )
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------


def effect_for(mode: SystemMode) -> ModeEffect:
    """Return the :class:`ModeEffect` for ``mode``.

    Equivalent to ``MODE_EFFECTS[mode]`` but offers a single import
    path for the lint rule (B31) to point engines at.
    """

    try:
        return MODE_EFFECTS[mode]
    except KeyError as exc:  # pragma: no cover — every SystemMode is in the table
        raise ValueError(f"no ModeEffect for mode {mode!r}") from exc


# ---------------------------------------------------------------------------
# Wave-04.6 PR-C — CANARY size cap
# ---------------------------------------------------------------------------


def clamp_qty_for_mode(
    qty: float, mode: SystemMode
) -> tuple[float, bool]:
    """Clamp a candidate fill quantity by the mode-effect ``size_cap_pct``.

    Pure function — no IO, no state, deterministic across replays
    (INV-15 / TEST-01).

    Semantics of :attr:`ModeEffect.size_cap_pct`:

    * ``None`` — uncapped (LIVE, AUTO).
    * ``0.0`` — non-applicable / passthrough. Modes whose dispatch is
      blocked elsewhere (SAFE, SHADOW, LOCKED) or whose venue is not
      a real venue (PAPER) do not need a positive cap; treating
      ``0.0`` as "no cap" preserves PAPER fills exactly.
    * ``> 0.0`` — interpreted as a *percent* of the candidate ``qty``.
      CANARY's ``1.0`` therefore clamps to ``qty * 0.01``.

    Args:
        qty: Candidate fill quantity (must be ``>= 0``).
        mode: Active :class:`SystemMode`.

    Returns:
        A ``(clamped_qty, was_clamped)`` tuple. ``was_clamped`` is
        ``True`` only when the cap actually reduced the quantity.
    """

    if qty < 0.0:
        raise ValueError("qty must be >= 0")
    cap_pct = effect_for(mode).size_cap_pct
    if cap_pct is None or cap_pct <= 0.0:
        return qty, False
    max_qty = qty * (cap_pct / 100.0)
    if qty <= max_qty:
        return qty, False
    return max_qty, True


__all__ = [
    "MODE_EFFECTS",
    "MODE_EFFECTS_HASH_KEY",
    "MODE_EFFECTS_INSTALLED_KIND",
    "ModeEffect",
    "OversightKind",
    "clamp_qty_for_mode",
    "effect_for",
    "mode_effect_table_hash",
]
