"""Wave-04.6 PR-F — OperatorAttention contract tests.

Pins the truth table that drives the AUTO mode oversight relaxation:

    | mode    | oversight_kind   | hazard | per_trade_required |
    |---------|------------------|--------|--------------------|
    | LOCKED  | none             | -      | True (fail-closed) |
    | SAFE    | per_trade        | -      | True               |
    | PAPER   | per_trade        | -      | True               |
    | SHADOW  | per_trade        | -      | True               |
    | CANARY  | per_trade        | -      | True               |
    | LIVE    | per_trade        | -      | True               |
    | AUTO    | exception_only   | False  | False              |
    | AUTO    | exception_only   | True   | True               |

These rows are the *only* observable contract of the module. Any
divergence breaks the cognitive chat runtime's auto-approval path
(Wave-04.6 PR-F) and is therefore an INV-15 replay-equivalence
violation.
"""

from __future__ import annotations

import pytest

from core.contracts.governance import SystemMode
from governance_engine.control_plane.operator_attention import (
    AUTO_DECIDED_BY_TAG,
    OperatorAttention,
)


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.SHADOW,
        SystemMode.CANARY,
        SystemMode.LIVE,
    ],
)
def test_per_trade_modes_always_require_attention(mode: SystemMode) -> None:
    """Modes with ``oversight_kind=per_trade`` always need an operator click.

    Hazard state is irrelevant because per-trade approval is the
    invariant, not a fallback for unsafe conditions."""

    attention = OperatorAttention(
        mode_provider=lambda m=mode: m,
        hazard_active_provider=lambda: False,
    )
    assert attention.per_trade_required() is True

    attention_with_hazard = OperatorAttention(
        mode_provider=lambda m=mode: m,
        hazard_active_provider=lambda: True,
    )
    assert attention_with_hazard.per_trade_required() is True


def test_locked_mode_fails_closed() -> None:
    """LOCKED has ``oversight_kind=none`` but PR-F still demands per-trade.

    LOCKED suppresses signal emission and execution dispatch upstream,
    so no proposal should reach the attention seam. If one does
    (misrouted / racing transition), require an operator click rather
    than auto-approving — the conservative interpretation of a state
    whose declarative intent is "do nothing"."""

    attention = OperatorAttention(
        mode_provider=lambda: SystemMode.LOCKED,
        hazard_active_provider=lambda: False,
    )
    assert attention.per_trade_required() is True


def test_auto_with_no_hazard_relaxes_to_auto_approve() -> None:
    """The headline AUTO behaviour: routine traffic auto-emits.

    When the system is in AUTO and no hazard is active the runtime is
    allowed to drive the approval edge with
    ``decided_by=AUTO_DECIDED_BY_TAG`` instead of waiting for an
    operator click. This is the operational distinction between AUTO
    and LIVE."""

    attention = OperatorAttention(
        mode_provider=lambda: SystemMode.AUTO,
        hazard_active_provider=lambda: False,
    )
    assert attention.per_trade_required() is False


def test_auto_with_active_hazard_re_arms_per_trade() -> None:
    """AUTO is exception-based — a hazard *is* the exception.

    During an active hazard window AUTO must require operator
    attention again. This is the safety property reviewer #3 called
    out: AUTO is "governance exception-based only", not "no
    governance"."""

    attention = OperatorAttention(
        mode_provider=lambda: SystemMode.AUTO,
        hazard_active_provider=lambda: True,
    )
    assert attention.per_trade_required() is True


def test_seams_are_called_each_invocation() -> None:
    """Mode + hazard are sampled per call — supports live transitions.

    The runtime keeps a single :class:`OperatorAttention` for the
    lifetime of the process; the seams must be re-evaluated on every
    proposal so a SHADOW→AUTO transition takes effect immediately
    without restarting the runtime."""

    mode_calls: list[None] = []
    haz_calls: list[None] = []

    def _mode() -> SystemMode:
        mode_calls.append(None)
        return SystemMode.AUTO

    def _haz() -> bool:
        haz_calls.append(None)
        return False

    attention = OperatorAttention(
        mode_provider=_mode,
        hazard_active_provider=_haz,
    )
    for _ in range(5):
        attention.per_trade_required()
    assert len(mode_calls) == 5
    assert len(haz_calls) == 5


def test_hazard_provider_short_circuited_for_per_trade() -> None:
    """Per-trade modes never read the hazard provider — minor perf
    invariant *and* an isolation guarantee.

    If the hazard provider raises (e.g. transient SCVS read failure),
    a non-AUTO system must still be able to gate proposals. Skipping
    the hazard read is the only way to keep SHADOW / CANARY / LIVE
    fail-soft against a noisy hazard subsystem."""

    def _exploding_haz() -> bool:
        raise RuntimeError("hazard provider should not be consulted")

    attention = OperatorAttention(
        mode_provider=lambda: SystemMode.LIVE,
        hazard_active_provider=_exploding_haz,
    )
    # Must not raise — hazard provider not consulted under per_trade.
    assert attention.per_trade_required() is True


def test_decided_by_tag_is_canonical() -> None:
    """The ledger tag is the cross-session contract — pin its value.

    Replay equivalence requires this string be byte-stable across
    versions. A rename without a migration would corrupt the
    ``OPERATOR_APPROVED_SIGNAL`` projection."""

    assert AUTO_DECIDED_BY_TAG == "auto:AUTO_MODE_EXCEPTION_ONLY"
