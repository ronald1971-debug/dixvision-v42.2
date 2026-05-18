"""Phase-6 P1-3 — LearningEngine / EvolutionEngine dormancy health.

The Phase-6 audit flagged that both engine shells returned
:data:`HealthState.OK` while their wired runtime hot-paths
(:class:`ClosedLearningLoop` / :class:`StructuralEvolutionLoop`)
were under HARDEN-04 freeze. The dormancy was visible at
``/api/operator/runtime/dormant`` but ``/api/health`` was still
reporting OK, producing two contradictory sources of truth.

These tests pin the corrected behaviour:

* with no ``is_active_fn`` injected, both shells default to
  :data:`HealthState.DEGRADED` with a detail string that points at
  ``/api/operator/runtime/dormant``;
* with an injected ``is_active_fn`` that returns ``True``, the
  shell reports :data:`HealthState.OK`;
* with an injected ``is_active_fn`` that returns ``False``, the
  shell reports :data:`HealthState.DEGRADED` again.

A future PR may not silently restore the old
:data:`HealthState.OK` default — that drift is what produced the
audit-finding in the first place.
"""

from __future__ import annotations

from core.contracts.engine import HealthState
from evolution_engine.engine import EvolutionEngine
from learning_engine.engine import LearningEngine


def test_learning_engine_defaults_to_degraded_when_unwired() -> None:
    engine = LearningEngine()
    status = engine.check_self()
    assert status.state is HealthState.DEGRADED
    assert "/api/operator/runtime/dormant" in status.detail


def test_evolution_engine_defaults_to_degraded_when_unwired() -> None:
    engine = EvolutionEngine()
    status = engine.check_self()
    assert status.state is HealthState.DEGRADED
    assert "/api/operator/runtime/dormant" in status.detail


def test_learning_engine_reports_ok_when_active_fn_true() -> None:
    engine = LearningEngine(is_active_fn=lambda: True)
    status = engine.check_self()
    assert status.state is HealthState.OK
    assert "unfrozen" in status.detail.lower()


def test_learning_engine_reports_degraded_when_active_fn_false() -> None:
    engine = LearningEngine(is_active_fn=lambda: False)
    status = engine.check_self()
    assert status.state is HealthState.DEGRADED
    assert "/api/operator/runtime/dormant" in status.detail


def test_evolution_engine_reports_ok_when_active_fn_true() -> None:
    engine = EvolutionEngine(is_active_fn=lambda: True)
    status = engine.check_self()
    assert status.state is HealthState.OK
    assert "unfrozen" in status.detail.lower()


def test_evolution_engine_reports_degraded_when_active_fn_false() -> None:
    engine = EvolutionEngine(is_active_fn=lambda: False)
    status = engine.check_self()
    assert status.state is HealthState.DEGRADED
    assert "/api/operator/runtime/dormant" in status.detail


def test_active_fn_invoked_per_check_self_call() -> None:
    """``check_self`` must not cache; freeze state is mutable."""

    state = {"active": True}
    engine = LearningEngine(is_active_fn=lambda: state["active"])
    assert engine.check_self().state is HealthState.OK
    state["active"] = False
    assert engine.check_self().state is HealthState.DEGRADED
    state["active"] = True
    assert engine.check_self().state is HealthState.OK
