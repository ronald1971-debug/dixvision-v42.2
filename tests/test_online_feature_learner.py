"""S-07 — Tests for the online feature learner lane.

Coverage:

* Config validation (dim, adwin_delta, pa_C, pa_variant,
  adwin_min_window, version)
* Observation validation (ts_ns, learner_id, x type/dim, y label,
  finite-float guard)
* State validation + checkpoint round-trip (to_payload / load_state)
* Initial state factory
* Pure ``predict`` (zero-state, after-update)
* Passive-Aggressive update math (PA, PA-I, PA-II analytical cases)
* No-update on already-correct margin
* Cap on PA-I τ
* ADWIN no-drift on stationary stream
* ADWIN drift on shift (negative-margin run → positive-margin run)
* ADWIN delta-sensitivity (smaller δ → fewer cuts)
* DriftReport contents + proposed LearningUpdate
* INV-12: lane never constructs SignalEvent / ExecutionIntent — AST
  pin
* INV-15: replay determinism — three identical runs produce
  byte-identical state and identical drift report / proposed update
* AST: no clock / os / asyncio imports at module scope; no
  governance / execution / system / hot-path imports; no ``river``
  import (zero-pip-dep adaptation)
* ADAPTED FROM header present
"""

from __future__ import annotations

import ast
import math
import pathlib

import pytest

from core.contracts.learning import LearningUpdate
from learning_engine.lanes import online_feature_learner as ofl
from learning_engine.lanes.online_feature_learner import (
    ONLINE_LEARNER_VERSION,
    PA_VARIANT_PA,
    PA_VARIANT_PA_I,
    PA_VARIANT_PA_II,
    DriftReport,
    OnlineLearnerConfig,
    OnlineLearnerError,
    OnlineLearnerObservation,
    OnlineLearnerState,
    OnlineLearnerStepOutcome,
    Prediction,
    build_drift_update,
    load_state,
    make_initial_state,
    predict,
    step,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides: object) -> OnlineLearnerConfig:
    base: dict[str, object] = {
        "dim": 3,
        "adwin_delta": 0.002,
        "pa_C": 1.0,
        "pa_variant": PA_VARIANT_PA_I,
        "adwin_min_window": 4,
    }
    base.update(overrides)
    return OnlineLearnerConfig(**base)  # type: ignore[arg-type]


def _obs(
    *,
    ts_ns: int,
    x: tuple[float, ...] = (1.0, 0.0, 0.0),
    y: int = 1,
    learner_id: str = "alpha",
) -> OnlineLearnerObservation:
    return OnlineLearnerObservation(ts_ns=ts_ns, learner_id=learner_id, x=x, y=y)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_dim_must_be_int() -> None:
    with pytest.raises(TypeError, match="dim must be int"):
        OnlineLearnerConfig(dim=3.0)  # type: ignore[arg-type]


def test_config_dim_must_be_positive() -> None:
    with pytest.raises(OnlineLearnerError, match="dim must be > 0"):
        OnlineLearnerConfig(dim=0)


def test_config_dim_rejects_bool() -> None:
    with pytest.raises(TypeError, match="dim must be int"):
        OnlineLearnerConfig(dim=True)  # type: ignore[arg-type]


def test_config_adwin_delta_must_be_float() -> None:
    with pytest.raises(TypeError, match="adwin_delta must be float"):
        OnlineLearnerConfig(dim=3, adwin_delta=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
def test_config_adwin_delta_must_be_in_open_unit(delta: float) -> None:
    with pytest.raises(OnlineLearnerError, match=r"adwin_delta must be in"):
        OnlineLearnerConfig(dim=3, adwin_delta=delta)


def test_config_pa_C_must_be_float() -> None:
    with pytest.raises(TypeError, match="pa_C must be float"):
        OnlineLearnerConfig(dim=3, pa_C=1)  # type: ignore[arg-type]


def test_config_pa_C_must_be_positive() -> None:
    with pytest.raises(OnlineLearnerError, match=r"pa_C must be > 0"):
        OnlineLearnerConfig(dim=3, pa_C=0.0)


def test_config_rejects_unknown_pa_variant() -> None:
    with pytest.raises(OnlineLearnerError, match="pa_variant must be one of"):
        OnlineLearnerConfig(dim=3, pa_variant="nope")


def test_config_min_window_must_be_int() -> None:
    with pytest.raises(TypeError, match="adwin_min_window must be int"):
        OnlineLearnerConfig(dim=3, adwin_min_window=4.0)  # type: ignore[arg-type]


def test_config_min_window_must_be_at_least_two() -> None:
    with pytest.raises(OnlineLearnerError, match="adwin_min_window must be >= 2"):
        OnlineLearnerConfig(dim=3, adwin_min_window=1)


def test_config_rejects_empty_version() -> None:
    with pytest.raises(OnlineLearnerError, match="version must be non-empty"):
        OnlineLearnerConfig(dim=3, version="")


def test_config_frozen() -> None:
    cfg = _cfg()
    with pytest.raises(AttributeError):
        cfg.dim = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------------


def test_observation_rejects_non_int_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        OnlineLearnerObservation(
            ts_ns=1.0,  # type: ignore[arg-type]
            learner_id="x",
            x=(1.0,),
            y=1,
        )


def test_observation_rejects_non_positive_ts_ns() -> None:
    with pytest.raises(OnlineLearnerError, match="ts_ns must be positive"):
        OnlineLearnerObservation(ts_ns=0, learner_id="x", x=(1.0,), y=1)


def test_observation_rejects_empty_learner_id() -> None:
    with pytest.raises(OnlineLearnerError, match="learner_id must be non-empty"):
        OnlineLearnerObservation(ts_ns=1, learner_id="", x=(1.0,), y=1)


def test_observation_rejects_non_tuple_x() -> None:
    with pytest.raises(TypeError, match=r"x must be tuple"):
        OnlineLearnerObservation(
            ts_ns=1,
            learner_id="x",
            x=[1.0, 2.0],  # type: ignore[arg-type]
            y=1,
        )


def test_observation_rejects_non_float_x_element() -> None:
    with pytest.raises(TypeError, match=r"x\[0\] must be float"):
        OnlineLearnerObservation(
            ts_ns=1,
            learner_id="x",
            x=(1,),  # type: ignore[arg-type]
            y=1,
        )


def test_observation_rejects_nan_in_x() -> None:
    with pytest.raises(OnlineLearnerError, match=r"x\[0\] must be finite"):
        OnlineLearnerObservation(ts_ns=1, learner_id="x", x=(float("nan"),), y=1)


def test_observation_rejects_inf_in_x() -> None:
    with pytest.raises(OnlineLearnerError, match=r"x\[0\] must be finite"):
        OnlineLearnerObservation(ts_ns=1, learner_id="x", x=(float("inf"),), y=1)


def test_observation_rejects_invalid_label() -> None:
    with pytest.raises(OnlineLearnerError, match=r"y must be in"):
        OnlineLearnerObservation(ts_ns=1, learner_id="x", x=(1.0,), y=0)


def test_observation_rejects_bool_label() -> None:
    with pytest.raises(TypeError, match="y must be int"):
        OnlineLearnerObservation(
            ts_ns=1,
            learner_id="x",
            x=(1.0,),
            y=True,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Initial state + state validation
# ---------------------------------------------------------------------------


def test_initial_state_zero_weights_and_no_window() -> None:
    s = make_initial_state(learner_id="alpha", dim=3)
    assert s.weights == (0.0, 0.0, 0.0)
    assert s.bias == 0.0
    assert s.n_observations == 0
    assert s.adwin_window == ()
    assert s.last_drift_ts_ns == 0
    assert s.version == ONLINE_LEARNER_VERSION


def test_initial_state_rejects_bad_dim() -> None:
    with pytest.raises(OnlineLearnerError, match="dim must be > 0"):
        make_initial_state(learner_id="alpha", dim=0)


def test_state_rejects_nan_weight() -> None:
    with pytest.raises(OnlineLearnerError, match=r"weights\[0\] must be finite"):
        OnlineLearnerState(
            learner_id="x",
            version=ONLINE_LEARNER_VERSION,
            weights=(float("nan"),),
            bias=0.0,
            n_observations=0,
            adwin_window=(),
            last_drift_ts_ns=0,
        )


def test_state_rejects_inf_bias() -> None:
    with pytest.raises(OnlineLearnerError, match="bias must be finite"):
        OnlineLearnerState(
            learner_id="x",
            version=ONLINE_LEARNER_VERSION,
            weights=(0.0,),
            bias=float("inf"),
            n_observations=0,
            adwin_window=(),
            last_drift_ts_ns=0,
        )


def test_state_payload_round_trip() -> None:
    s = OnlineLearnerState(
        learner_id="alpha",
        version=ONLINE_LEARNER_VERSION,
        weights=(0.25, -0.5, 1.0),
        bias=-0.125,
        n_observations=7,
        adwin_window=(0.1, -0.2, 0.3),
        last_drift_ts_ns=42,
    )
    payload = s.to_payload()
    restored = load_state(payload)
    assert restored == s


def test_load_state_rejects_missing_key() -> None:
    s = make_initial_state(learner_id="x", dim=2)
    payload = dict(s.to_payload())
    del payload["bias"]
    with pytest.raises(OnlineLearnerError, match="missing keys"):
        load_state(payload)


def test_load_state_rejects_unknown_key() -> None:
    s = make_initial_state(learner_id="x", dim=2)
    payload = dict(s.to_payload())
    payload["surprise"] = "1"
    with pytest.raises(OnlineLearnerError, match="unknown keys"):
        load_state(payload)


def test_load_state_rejects_version_mismatch() -> None:
    s = make_initial_state(learner_id="x", dim=2)
    payload = dict(s.to_payload())
    payload["version"] = "different"
    with pytest.raises(OnlineLearnerError, match="checkpoint version"):
        load_state(payload)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_zero_state_returns_positive_label() -> None:
    s = make_initial_state(learner_id="x", dim=3)
    p = predict(s, (1.0, 0.0, 0.0))
    assert p.score == 0.0
    assert p.label == 1


def test_predict_uses_weights_and_bias() -> None:
    s = OnlineLearnerState(
        learner_id="x",
        version=ONLINE_LEARNER_VERSION,
        weights=(0.5, -0.25, 1.0),
        bias=0.1,
        n_observations=0,
        adwin_window=(),
        last_drift_ts_ns=0,
    )
    p = predict(s, (2.0, 4.0, 0.5))
    expected = 0.5 * 2.0 + (-0.25) * 4.0 + 1.0 * 0.5 + 0.1
    assert p.score == pytest.approx(expected)
    assert p.label == 1


def test_predict_rejects_dim_mismatch() -> None:
    s = make_initial_state(learner_id="x", dim=3)
    with pytest.raises(OnlineLearnerError, match="x dim"):
        predict(s, (1.0, 2.0))


# ---------------------------------------------------------------------------
# Passive-Aggressive update math
# ---------------------------------------------------------------------------


def test_pa_no_update_when_margin_already_safe() -> None:
    """``y · (w·x + b) >= 1`` → hinge loss is 0 → no update."""
    s = OnlineLearnerState(
        learner_id="x",
        version=ONLINE_LEARNER_VERSION,
        weights=(2.0,),
        bias=0.0,
        n_observations=0,
        adwin_window=(),
        last_drift_ts_ns=0,
    )
    cfg = _cfg(dim=1)
    out = step(s, _obs(ts_ns=1, x=(1.0,), y=1, learner_id="x"), cfg)
    assert out.new_state.weights == (2.0,)
    assert out.new_state.bias == 0.0
    assert out.drift_report is None


def test_pa_full_step_when_margin_is_zero() -> None:
    """PA / PA-I: zero-margin step has loss=1, x_norm_sq=1 →
    τ=1 → w grows by y·x = +1."""
    s = make_initial_state(learner_id="x", dim=1)
    cfg = _cfg(dim=1, pa_variant=PA_VARIANT_PA_I, pa_C=10.0)
    out = step(s, _obs(ts_ns=1, x=(1.0,), y=1, learner_id="x"), cfg)
    assert out.new_state.weights == (1.0,)
    assert out.new_state.bias == 1.0


def test_pa_i_caps_step_at_C() -> None:
    """PA-I clamps τ to C; PA does not — verify they diverge."""
    s = make_initial_state(learner_id="x", dim=1)
    cfg_pa_i = _cfg(dim=1, pa_variant=PA_VARIANT_PA_I, pa_C=0.1)
    cfg_pa = _cfg(dim=1, pa_variant=PA_VARIANT_PA, pa_C=0.1)
    o = _obs(ts_ns=1, x=(1.0,), y=1, learner_id="x")
    out_pa_i = step(s, o, cfg_pa_i)
    out_pa = step(s, o, cfg_pa)
    assert out_pa_i.new_state.weights[0] == pytest.approx(0.1)
    assert out_pa.new_state.weights[0] == pytest.approx(1.0)


def test_pa_ii_uses_smoothed_denominator() -> None:
    """PA-II: τ = loss / (||x||² + 1/(2C))."""
    s = make_initial_state(learner_id="x", dim=1)
    cfg = _cfg(dim=1, pa_variant=PA_VARIANT_PA_II, pa_C=0.5)
    out = step(s, _obs(ts_ns=1, x=(1.0,), y=1, learner_id="x"), cfg)
    expected_tau = 1.0 / (1.0 + 1.0 / (2.0 * 0.5))
    assert out.new_state.weights[0] == pytest.approx(expected_tau)
    assert out.new_state.bias == pytest.approx(expected_tau)


def test_pa_negative_label_pushes_weights_negative() -> None:
    s = make_initial_state(learner_id="x", dim=2)
    cfg = _cfg(dim=2, pa_variant=PA_VARIANT_PA_I, pa_C=10.0)
    out = step(s, _obs(ts_ns=1, x=(1.0, 2.0), y=-1, learner_id="x"), cfg)
    assert out.new_state.weights[0] < 0.0
    assert out.new_state.weights[1] < 0.0


def test_pa_zero_x_norm_does_not_divide() -> None:
    """All-zero x → no update for PA / PA-I (x_norm_sq == 0)."""
    s = make_initial_state(learner_id="x", dim=2)
    cfg = _cfg(dim=2, pa_variant=PA_VARIANT_PA_I)
    out = step(s, _obs(ts_ns=1, x=(0.0, 0.0), y=1, learner_id="x"), cfg)
    assert out.new_state.weights == (0.0, 0.0)
    assert out.new_state.bias == 0.0


# ---------------------------------------------------------------------------
# step argument validation
# ---------------------------------------------------------------------------


def test_step_rejects_state_type_mismatch() -> None:
    cfg = _cfg(dim=1)
    with pytest.raises(TypeError, match="state must be"):
        step("nope", _obs(ts_ns=1, x=(1.0,), learner_id="x"), cfg)  # type: ignore[arg-type]


def test_step_rejects_observation_type_mismatch() -> None:
    cfg = _cfg(dim=1)
    s = make_initial_state(learner_id="x", dim=1)
    with pytest.raises(TypeError, match="observation must be"):
        step(s, "nope", cfg)  # type: ignore[arg-type]


def test_step_rejects_config_type_mismatch() -> None:
    s = make_initial_state(learner_id="x", dim=1)
    with pytest.raises(TypeError, match="config must be"):
        step(s, _obs(ts_ns=1, x=(1.0,), learner_id="x"), "nope")  # type: ignore[arg-type]


def test_step_rejects_learner_id_mismatch() -> None:
    cfg = _cfg(dim=1)
    s = make_initial_state(learner_id="alpha", dim=1)
    with pytest.raises(OnlineLearnerError, match="learner_id"):
        step(s, _obs(ts_ns=1, x=(1.0,), learner_id="beta"), cfg)


def test_step_rejects_x_dim_mismatch() -> None:
    cfg = _cfg(dim=2)
    s = make_initial_state(learner_id="x", dim=2)
    with pytest.raises(OnlineLearnerError, match="x dim"):
        step(s, _obs(ts_ns=1, x=(1.0,), learner_id="x"), cfg)


def test_step_rejects_state_config_dim_mismatch() -> None:
    cfg = _cfg(dim=3)
    s = make_initial_state(learner_id="x", dim=2)
    with pytest.raises(OnlineLearnerError, match="config.dim"):
        step(s, _obs(ts_ns=1, x=(1.0, 2.0), learner_id="x"), cfg)


# ---------------------------------------------------------------------------
# ADWIN drift detection
# ---------------------------------------------------------------------------


def _drive(
    cfg: OnlineLearnerConfig,
    observations: list[OnlineLearnerObservation],
) -> tuple[OnlineLearnerState, list[OnlineLearnerStepOutcome]]:
    s = make_initial_state(learner_id=observations[0].learner_id, dim=cfg.dim)
    outs: list[OnlineLearnerStepOutcome] = []
    for o in observations:
        out = step(s, o, cfg)
        s = out.new_state
        outs.append(out)
    return s, outs


def test_adwin_no_drift_on_stationary_safe_stream() -> None:
    """Once the model classifies correctly the margin saturates and
    ADWIN sees a stationary post-update margin → no cut."""
    cfg = _cfg(dim=1, adwin_min_window=4, adwin_delta=0.05)
    obs = [_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x") for i in range(50)]
    _, outs = _drive(cfg, obs)
    assert all(o.drift_report is None for o in outs)


def test_adwin_drift_on_label_flip() -> None:
    """20 +1 samples then 20 -1 samples should fire at least once."""
    cfg = _cfg(dim=1, adwin_min_window=4, adwin_delta=0.05)
    obs: list[OnlineLearnerObservation] = []
    for i in range(20):
        obs.append(_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x"))
    for i in range(20):
        obs.append(_obs(ts_ns=20 + i + 1, x=(1.0,), y=-1, learner_id="x"))
    _, outs = _drive(cfg, obs)
    assert any(o.drift_report is not None for o in outs)


def test_adwin_smaller_delta_means_fewer_or_equal_cuts() -> None:
    """Smaller δ → stricter Hoeffding bound → ≤ cuts than larger δ."""
    obs: list[OnlineLearnerObservation] = []
    for i in range(20):
        obs.append(_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x"))
    for i in range(20):
        obs.append(_obs(ts_ns=20 + i + 1, x=(1.0,), y=-1, learner_id="x"))
    _, outs_loose = _drive(_cfg(dim=1, adwin_min_window=4, adwin_delta=0.5), obs)
    _, outs_strict = _drive(_cfg(dim=1, adwin_min_window=4, adwin_delta=1e-6), obs)
    n_loose = sum(1 for o in outs_loose if o.drift_report is not None)
    n_strict = sum(1 for o in outs_strict if o.drift_report is not None)
    assert n_strict <= n_loose


def test_adwin_min_window_blocks_early_cuts() -> None:
    cfg = _cfg(dim=1, adwin_min_window=100, adwin_delta=0.5)
    obs: list[OnlineLearnerObservation] = []
    for i in range(10):
        obs.append(_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x"))
    for i in range(10):
        obs.append(_obs(ts_ns=10 + i + 1, x=(1.0,), y=-1, learner_id="x"))
    _, outs = _drive(cfg, obs)
    assert all(o.drift_report is None for o in outs)


def test_drift_report_change_point_drops_older_half() -> None:
    cfg = _cfg(dim=1, adwin_min_window=4, adwin_delta=0.05)
    obs: list[OnlineLearnerObservation] = []
    for i in range(15):
        obs.append(_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x"))
    for i in range(15):
        obs.append(_obs(ts_ns=15 + i + 1, x=(1.0,), y=-1, learner_id="x"))
    final_state, outs = _drive(cfg, obs)
    drifts = [o for o in outs if o.drift_report is not None]
    assert drifts, "expected at least one drift"
    first = drifts[0].drift_report
    assert first is not None
    assert first.change_point >= 1
    assert first.change_point < first.n_observations_before
    assert first.magnitude > first.epsilon_cut
    assert final_state.last_drift_ts_ns == drifts[-1].drift_report.ts_ns  # type: ignore[union-attr]


def test_drift_report_propagates_observation_ts_ns() -> None:
    cfg = _cfg(dim=1, adwin_min_window=4, adwin_delta=0.05)
    obs: list[OnlineLearnerObservation] = []
    for i in range(15):
        obs.append(_obs(ts_ns=1000 + i, x=(1.0,), y=1, learner_id="x"))
    for i in range(15):
        obs.append(_obs(ts_ns=2000 + i, x=(1.0,), y=-1, learner_id="x"))
    _, outs = _drive(cfg, obs)
    for o in outs:
        if o.drift_report is not None:
            assert o.drift_report.ts_ns >= 1000


# ---------------------------------------------------------------------------
# build_drift_update / governance hand-off
# ---------------------------------------------------------------------------


def test_build_drift_update_returns_learning_update() -> None:
    drift = DriftReport(
        ts_ns=42,
        learner_id="alpha",
        change_point=5,
        magnitude=0.8,
        n_observations_before=20,
        epsilon_cut=0.3,
    )
    upd = build_drift_update(drift=drift, version="s-07.v1")
    assert isinstance(upd, LearningUpdate)
    assert upd.ts_ns == 42
    assert upd.strategy_id == "alpha"
    assert upd.parameter == "weights"
    assert upd.reason == "adwin_drift_detected"
    assert upd.meta["change_point"] == "5"
    assert upd.meta["version"] == "s-07.v1"


def test_build_drift_update_rejects_non_drift_input() -> None:
    with pytest.raises(TypeError, match="drift must be DriftReport"):
        build_drift_update(drift="nope", version="v")  # type: ignore[arg-type]


def test_step_emits_learning_update_alongside_drift() -> None:
    cfg = _cfg(dim=1, adwin_min_window=4, adwin_delta=0.05)
    obs: list[OnlineLearnerObservation] = []
    for i in range(15):
        obs.append(_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x"))
    for i in range(15):
        obs.append(_obs(ts_ns=15 + i + 1, x=(1.0,), y=-1, learner_id="x"))
    _, outs = _drive(cfg, obs)
    drifts = [o for o in outs if o.drift_report is not None]
    for o in drifts:
        assert isinstance(o.proposed_update, LearningUpdate)
        assert o.proposed_update.reason == "adwin_drift_detected"
        assert o.proposed_update.strategy_id == "x"


def test_step_no_drift_emits_no_update() -> None:
    cfg = _cfg(dim=1, adwin_min_window=4, adwin_delta=0.05)
    obs = [_obs(ts_ns=i + 1, x=(1.0,), y=1, learner_id="x") for i in range(20)]
    _, outs = _drive(cfg, obs)
    assert all(o.proposed_update is None for o in outs)
    assert all(o.drift_report is None for o in outs)


# ---------------------------------------------------------------------------
# DriftReport validation
# ---------------------------------------------------------------------------


def test_drift_report_rejects_nan_magnitude() -> None:
    with pytest.raises(OnlineLearnerError, match="magnitude"):
        DriftReport(
            ts_ns=1,
            learner_id="x",
            change_point=1,
            magnitude=float("nan"),
            n_observations_before=2,
            epsilon_cut=0.0,
        )


def test_drift_report_rejects_negative_change_point() -> None:
    with pytest.raises(OnlineLearnerError, match="change_point"):
        DriftReport(
            ts_ns=1,
            learner_id="x",
            change_point=-1,
            magnitude=0.0,
            n_observations_before=2,
            epsilon_cut=0.0,
        )


def test_drift_report_rejects_zero_n_observations_before() -> None:
    with pytest.raises(OnlineLearnerError, match="n_observations_before"):
        DriftReport(
            ts_ns=1,
            learner_id="x",
            change_point=0,
            magnitude=0.0,
            n_observations_before=0,
            epsilon_cut=0.0,
        )


# ---------------------------------------------------------------------------
# Replay determinism (INV-15)
# ---------------------------------------------------------------------------


def test_replay_determinism_byte_identical_states_and_drifts() -> None:
    cfg = _cfg(dim=2, adwin_min_window=4, adwin_delta=0.05)

    def run_once() -> tuple[OnlineLearnerState, list[float], list[DriftReport]]:
        s = make_initial_state(learner_id="x", dim=2)
        scores: list[float] = []
        drifts: list[DriftReport] = []
        for i in range(15):
            out = step(
                s,
                _obs(ts_ns=i + 1, x=(1.0, -0.5), y=1, learner_id="x"),
                cfg,
            )
            s = out.new_state
            scores.append(out.prediction.score)
            if out.drift_report is not None:
                drifts.append(out.drift_report)
        for i in range(15):
            out = step(
                s,
                _obs(ts_ns=100 + i, x=(1.0, -0.5), y=-1, learner_id="x"),
                cfg,
            )
            s = out.new_state
            scores.append(out.prediction.score)
            if out.drift_report is not None:
                drifts.append(out.drift_report)
        return s, scores, drifts

    s1, sc1, d1 = run_once()
    s2, sc2, d2 = run_once()
    s3, sc3, d3 = run_once()
    assert s1 == s2 == s3
    assert sc1 == sc2 == sc3
    assert d1 == d2 == d3


def test_step_outcome_is_frozen() -> None:
    s = make_initial_state(learner_id="x", dim=1)
    cfg = _cfg(dim=1)
    out = step(s, _obs(ts_ns=1, x=(1.0,), learner_id="x"), cfg)
    assert isinstance(out, OnlineLearnerStepOutcome)
    with pytest.raises(AttributeError):
        out.new_state = s  # type: ignore[misc]


def test_prediction_is_frozen() -> None:
    s = make_initial_state(learner_id="x", dim=1)
    p = predict(s, (1.0,))
    assert isinstance(p, Prediction)
    with pytest.raises(AttributeError):
        p.score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AST / static checks
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return pathlib.Path(ofl.__file__).read_text(encoding="utf-8")


def _module_ast() -> ast.Module:
    return ast.parse(_module_source())


def _imported_roots(tree: ast.Module) -> set[str]:
    """Top-level imports only (function bodies excluded)."""
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def test_module_does_not_import_clock_or_environ() -> None:
    """B-CLOCK / T1 / INV-15."""
    forbidden = {"time", "datetime", "os"}
    found = _imported_roots(_module_ast()) & forbidden
    assert found == set(), f"online_feature_learner imports forbidden roots: {sorted(found)}"


def test_module_does_not_import_governance_or_execution_or_system() -> None:
    """L2 / L3 / B1 — OFFLINE-tier leaf must stay leaf."""
    forbidden = {
        "governance_engine",
        "execution_engine",
        "system_engine",
    }
    found = _imported_roots(_module_ast()) & forbidden
    assert found == set(), f"online_feature_learner imports forbidden engines: {sorted(found)}"


def test_module_does_not_import_river_at_top_level() -> None:
    """river is *not* a runtime dep — adaptation is verbatim Python."""
    assert "river" not in _imported_roots(_module_ast())


def test_module_does_not_import_numpy_or_pandas() -> None:
    """OFFLINE tier uses plain Python floats for byte determinism."""
    forbidden = {"numpy", "pandas", "scipy"}
    found = _imported_roots(_module_ast()) & forbidden
    assert found == set()


def test_module_does_not_emit_signal_or_execution_intent() -> None:
    """INV-12: lane never constructs execution-side records.

    Walks the AST for ``Name`` nodes only — string-literal mentions
    in docstrings are explicitly tolerated since the lane *cannot*
    construct an event without referencing the class.
    """
    forbidden = {"SignalEvent", "ExecutionIntent"}
    tree = _module_ast()
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            seen.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            seen.add(node.attr)
    assert seen == set(), f"online_feature_learner references execution-side names: {sorted(seen)}"


def test_module_has_adapted_from_header() -> None:
    """PART 1 rule 7."""
    src = _module_source()
    first_line = src.splitlines()[0]
    assert first_line.startswith("# ADAPTED FROM:")
    assert "river/drift/adwin.py" in first_line
    assert "river/linear_model/pa.py" in first_line


def test_new_pip_dependencies_is_empty_tuple() -> None:
    """river is *not* added — algorithms reproduced verbatim."""
    assert ofl.NEW_PIP_DEPENDENCIES == ()


def test_public_api_matches_all() -> None:
    expected = {
        "NEW_PIP_DEPENDENCIES",
        "ONLINE_LEARNER_VERSION",
        "PA_VARIANT_PA",
        "PA_VARIANT_PA_I",
        "PA_VARIANT_PA_II",
        "VALID_LABELS",
        "DriftReport",
        "OnlineLearnerConfig",
        "OnlineLearnerError",
        "OnlineLearnerObservation",
        "OnlineLearnerState",
        "OnlineLearnerStepOutcome",
        "Prediction",
        "build_drift_update",
        "load_state",
        "make_initial_state",
        "predict",
        "step",
    }
    assert set(ofl.__all__) == expected


# ---------------------------------------------------------------------------
# Hoeffding bound sanity
# ---------------------------------------------------------------------------


def test_epsilon_cut_decreases_with_window_size() -> None:
    """Larger m → smaller bound → easier to cut on the same magnitude."""
    eps_small = ofl._epsilon_cut(n0=5, n1=5, delta_prime=0.01)
    eps_large = ofl._epsilon_cut(n0=500, n1=500, delta_prime=0.01)
    assert math.isfinite(eps_small)
    assert math.isfinite(eps_large)
    assert eps_large < eps_small


def test_epsilon_cut_decreases_with_loose_delta() -> None:
    eps_strict = ofl._epsilon_cut(n0=50, n1=50, delta_prime=1e-8)
    eps_loose = ofl._epsilon_cut(n0=50, n1=50, delta_prime=0.1)
    assert eps_loose < eps_strict


def test_epsilon_cut_returns_inf_on_empty_half() -> None:
    assert math.isinf(ofl._epsilon_cut(n0=0, n1=10, delta_prime=0.01))
    assert math.isinf(ofl._epsilon_cut(n0=10, n1=0, delta_prime=0.01))
