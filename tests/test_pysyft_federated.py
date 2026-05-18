"""Tests for C-12 federated_pysyft — differential-privacy federated lane.

OFFLINE-only deterministic DP aggregation. Mirrors the discipline of
``tests/test_federated.py`` (C-09), ``tests/test_fedml.py`` (C-10), and
``tests/test_openfl.py`` (C-11):

* module surface + lazy seam,
* value-object validation (happy path + frozen + edge cases),
* deterministic noise (BLAKE2b Box-Muller / inverse-CDF),
* privacy accountant composition + budget exhaustion hard stop,
* aggregate_private_round math equivalence to C-09 FedAvg,
* INV-15 3-run byte-identical replay (digest + report + accountant),
* privacy guards inherited from C-09 (8 forbidden meta keys),
* AST guardrails (forbidden imports, transport-layer typed-event
  constructors, runtime-tier imports, no ``random`` / ``time`` /
  ``datetime`` / ``asyncio`` / ``os`` calls).
"""

from __future__ import annotations

import ast
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.contracts.learning import LearningUpdate
from learning_engine.lanes.federated import GradientUpdate, fed_avg_aggregate
from learning_engine.lanes.federated_pysyft import (
    NEW_PIP_DEPENDENCIES,
    PYSYFT_VERSION,
    NoiseConfig,
    PrivacyAccountant,
    PrivacyBudget,
    PrivateContribution,
    PrivateRoundReport,
    aggregate_private_round,
    apply_dp_noise,
)

PYSYFT_PATH = Path("learning_engine/lanes/federated_pysyft.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grad(client: str, delta: float, n: int, ts: int = 100) -> GradientUpdate:
    return GradientUpdate(
        client_id=client,
        parameter="lr",
        delta=delta,
        num_samples=n,
        ts_ns=ts,
    )


def _noise_g(sens: float = 1.0, mult: float = 1.0) -> NoiseConfig:
    return NoiseConfig(
        mechanism="gaussian",
        sensitivity=sens,
        noise_multiplier=mult,
    )


def _noise_l(sens: float = 1.0, mult: float = 1.0) -> NoiseConfig:
    return NoiseConfig(
        mechanism="laplace",
        sensitivity=sens,
        noise_multiplier=mult,
    )


def _budget(eps: float = 10.0, delta: float = 1e-5) -> PrivacyBudget:
    return PrivacyBudget(epsilon=eps, delta=delta)


def _accountant(eps: float = 10.0, delta: float = 1e-5) -> PrivacyAccountant:
    return PrivacyAccountant(budget=_budget(eps=eps, delta=delta))


def _private_contribution(
    client: str,
    delta: float,
    n: int,
    eps: float = 1.0,
    d: float = 0.0,
    parameter: str = "lr",
    ts: int = 100,
) -> PrivateContribution:
    return PrivateContribution(
        client_id=client,
        parameter=parameter,
        delta=delta,
        num_samples=n,
        ts_ns=ts,
        epsilon_consumed=eps,
        delta_consumed=d,
    )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_version_present(self) -> None:
        assert PYSYFT_VERSION == "v3.7-C12"

    def test_pip_dependencies(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ("syft",)

    def test_exports(self) -> None:
        from learning_engine.lanes import federated_pysyft as mod

        for name in (
            "PYSYFT_VERSION",
            "NEW_PIP_DEPENDENCIES",
            "PrivacyBudget",
            "NoiseConfig",
            "PrivateContribution",
            "PrivateRoundReport",
            "PrivacyAccountant",
            "apply_dp_noise",
            "aggregate_private_round",
        ):
            assert hasattr(mod, name), name


# ---------------------------------------------------------------------------
# PrivacyBudget
# ---------------------------------------------------------------------------


class TestPrivacyBudget:
    def test_happy(self) -> None:
        b = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert b.epsilon == 1.0
        assert b.delta == 1e-5

    def test_default_delta(self) -> None:
        b = PrivacyBudget(epsilon=1.0)
        assert b.delta == 0.0

    def test_frozen(self) -> None:
        b = PrivacyBudget(epsilon=1.0)
        with pytest.raises(FrozenInstanceError):
            b.epsilon = 2.0  # type: ignore[misc]

    def test_eps_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=0.0)

    def test_eps_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=-0.1)

    def test_eps_nan_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=float("nan"))

    def test_eps_inf_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=float("inf"))

    def test_delta_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=1.0, delta=-1e-9)

    def test_delta_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=1.0, delta=1.0)

    def test_delta_above_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon=1.0, delta=1.5)


# ---------------------------------------------------------------------------
# NoiseConfig
# ---------------------------------------------------------------------------


class TestNoiseConfig:
    def test_gaussian(self) -> None:
        n = NoiseConfig(
            mechanism="gaussian",
            sensitivity=1.0,
            noise_multiplier=1.5,
        )
        assert n.mechanism == "gaussian"

    def test_laplace(self) -> None:
        n = NoiseConfig(
            mechanism="laplace",
            sensitivity=1.0,
            noise_multiplier=1.0,
        )
        assert n.mechanism == "laplace"

    def test_frozen(self) -> None:
        n = _noise_g()
        with pytest.raises(FrozenInstanceError):
            n.mechanism = "laplace"  # type: ignore[misc]

    def test_unsupported_mechanism(self) -> None:
        with pytest.raises(ValueError):
            NoiseConfig(
                mechanism="exponential",
                sensitivity=1.0,
                noise_multiplier=1.0,
            )

    def test_sensitivity_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoiseConfig(
                mechanism="gaussian",
                sensitivity=0.0,
                noise_multiplier=1.0,
            )

    def test_sensitivity_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoiseConfig(
                mechanism="gaussian",
                sensitivity=-1.0,
                noise_multiplier=1.0,
            )

    def test_noise_mult_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoiseConfig(
                mechanism="gaussian",
                sensitivity=1.0,
                noise_multiplier=0.0,
            )

    def test_noise_mult_nan_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoiseConfig(
                mechanism="gaussian",
                sensitivity=1.0,
                noise_multiplier=float("nan"),
            )


# ---------------------------------------------------------------------------
# PrivateContribution
# ---------------------------------------------------------------------------


class TestPrivateContribution:
    def test_happy(self) -> None:
        c = _private_contribution("c-a", 0.5, 10)
        assert c.client_id == "c-a"
        assert c.delta == 0.5
        assert c.num_samples == 10

    def test_frozen(self) -> None:
        c = _private_contribution("c-a", 0.5, 10)
        with pytest.raises(FrozenInstanceError):
            c.delta = 1.0  # type: ignore[misc]

    def test_empty_client_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("", 0.5, 10)

    def test_empty_parameter_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", 0.5, 10, parameter="")

    def test_delta_nan_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", float("nan"), 10)

    def test_num_samples_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", 0.5, -1)

    def test_ts_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", 0.5, 10, ts=-1)

    def test_epsilon_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", 0.5, 10, eps=-0.5)

    def test_delta_consumed_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", 0.5, 10, d=-1e-9)

    def test_delta_consumed_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            _private_contribution("c-a", 0.5, 10, d=1.0)

    def test_as_gradient_update_projection(self) -> None:
        c = _private_contribution("c-a", 0.5, 10, ts=200)
        g = c.as_gradient_update()
        assert isinstance(g, GradientUpdate)
        assert g.client_id == "c-a"
        assert g.parameter == "lr"
        assert g.delta == 0.5
        assert g.num_samples == 10
        assert g.ts_ns == 200


# ---------------------------------------------------------------------------
# PrivateRoundReport
# ---------------------------------------------------------------------------


class TestPrivateRoundReport:
    def _good(self, **kw: object) -> PrivateRoundReport:
        defaults: dict[str, object] = {
            "round_id": "r-1",
            "parameter": "lr",
            "n_clients": 2,
            "aggregated_delta": 0.5,
            "total_samples": 20,
            "epsilon_spent": 1.0,
            "delta_spent": 1e-6,
            "ts_ns": 100,
            "digest": "a" * 32,
        }
        defaults.update(kw)
        return PrivateRoundReport(**defaults)  # type: ignore[arg-type]

    def test_happy(self) -> None:
        r = self._good()
        assert r.n_clients == 2

    def test_frozen(self) -> None:
        r = self._good()
        with pytest.raises(FrozenInstanceError):
            r.n_clients = 5  # type: ignore[misc]

    def test_empty_round_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._good(round_id="")

    def test_bad_digest_length(self) -> None:
        with pytest.raises(ValueError):
            self._good(digest="ab")

    def test_epsilon_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._good(epsilon_spent=-1.0)

    def test_negative_clients_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._good(n_clients=-1)


# ---------------------------------------------------------------------------
# PrivacyAccountant
# ---------------------------------------------------------------------------


class TestPrivacyAccountant:
    def test_initial_state(self) -> None:
        a = _accountant(eps=10.0, delta=1e-5)
        assert a.epsilon_spent == 0.0
        assert a.delta_spent == 0.0
        assert a.n_rounds == 0
        assert a.epsilon_remaining == 10.0
        assert a.delta_remaining == 1e-5

    def test_account_round_returns_new_instance(self) -> None:
        a = _accountant()
        b = a.account_round(epsilon=1.0)
        assert b is not a
        assert a.n_rounds == 0
        assert b.n_rounds == 1
        assert b.epsilon_spent == 1.0

    def test_basic_composition(self) -> None:
        a = _accountant(eps=10.0)
        b = a.account_round(epsilon=1.0)
        c = b.account_round(epsilon=2.5)
        assert c.epsilon_spent == pytest.approx(3.5)
        assert c.delta_spent == 0.0
        assert c.n_rounds == 2

    def test_delta_composition(self) -> None:
        a = _accountant(eps=10.0, delta=1e-5)
        b = a.account_round(epsilon=1.0, delta=2e-6)
        c = b.account_round(epsilon=1.0, delta=3e-6)
        assert c.delta_spent == pytest.approx(5e-6)

    def test_budget_exhaustion_rejected(self) -> None:
        a = _accountant(eps=2.0)
        b = a.account_round(epsilon=1.5)
        with pytest.raises(ValueError, match="privacy budget exhausted"):
            b.account_round(epsilon=1.0)

    def test_budget_at_limit_accepted(self) -> None:
        a = _accountant(eps=2.0)
        b = a.account_round(epsilon=2.0)
        assert b.epsilon_remaining == 0.0

    def test_can_afford_positive(self) -> None:
        a = _accountant(eps=5.0)
        assert a.can_afford(2.0)
        assert a.can_afford(5.0)
        assert not a.can_afford(5.1)

    def test_can_afford_rejects_negative(self) -> None:
        a = _accountant()
        assert not a.can_afford(-1.0)
        assert not a.can_afford(1.0, -1.0)

    def test_frozen(self) -> None:
        a = _accountant()
        with pytest.raises(FrozenInstanceError):
            a.epsilon_spent = 100.0  # type: ignore[misc]

    def test_neg_epsilon_charge_rejected(self) -> None:
        a = _accountant()
        with pytest.raises(ValueError):
            a.account_round(epsilon=-1.0)

    def test_nan_epsilon_charge_rejected(self) -> None:
        a = _accountant()
        with pytest.raises(ValueError):
            a.account_round(epsilon=float("nan"))


# ---------------------------------------------------------------------------
# Deterministic noise — apply_dp_noise
# ---------------------------------------------------------------------------


class TestApplyDpNoise:
    def test_gaussian_returns_contribution(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert isinstance(c, PrivateContribution)
        assert c.client_id == "c-a"
        assert c.num_samples == 10
        assert c.epsilon_consumed == 1.0

    def test_gaussian_finite(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert math.isfinite(c.delta)

    def test_laplace_returns_contribution(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c = apply_dp_noise(
            update=u,
            noise=_noise_l(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert isinstance(c, PrivateContribution)
        assert math.isfinite(c.delta)

    def test_noise_deterministic_for_same_seed(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c1 = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        c2 = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert c1.delta == c2.delta

    def test_noise_differs_across_rounds(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c1 = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        c2 = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-2",
            epsilon=1.0,
        )
        assert c1.delta != c2.delta

    def test_noise_differs_across_clients(self) -> None:
        u1 = _grad("c-a", 1.0, 10)
        u2 = _grad("c-b", 1.0, 10)
        c1 = apply_dp_noise(
            update=u1,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        c2 = apply_dp_noise(
            update=u2,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert c1.delta != c2.delta

    def test_noise_differs_across_mechanism(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c1 = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        c2 = apply_dp_noise(
            update=u,
            noise=_noise_l(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert c1.delta != c2.delta

    def test_noise_preserves_meta(self) -> None:
        u = GradientUpdate(
            client_id="c-a",
            parameter="lr",
            delta=1.0,
            num_samples=10,
            ts_ns=100,
            meta={"region": "eu"},
        )
        c = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
        )
        assert c.meta["region"] == "eu"

    def test_noise_meta_overlay(self) -> None:
        u = _grad("c-a", 1.0, 10)
        c = apply_dp_noise(
            update=u,
            noise=_noise_g(),
            round_id="r-1",
            epsilon=1.0,
            meta={"tag": "x"},
        )
        assert c.meta["tag"] == "x"

    def test_empty_round_id_rejected(self) -> None:
        u = _grad("c-a", 1.0, 10)
        with pytest.raises(ValueError):
            apply_dp_noise(
                update=u,
                noise=_noise_g(),
                round_id="",
                epsilon=1.0,
            )

    def test_epsilon_zero_rejected(self) -> None:
        u = _grad("c-a", 1.0, 10)
        with pytest.raises(ValueError):
            apply_dp_noise(
                update=u,
                noise=_noise_g(),
                round_id="r-1",
                epsilon=0.0,
            )

    def test_epsilon_negative_rejected(self) -> None:
        u = _grad("c-a", 1.0, 10)
        with pytest.raises(ValueError):
            apply_dp_noise(
                update=u,
                noise=_noise_g(),
                round_id="r-1",
                epsilon=-1.0,
            )

    def test_meta_non_str_value_rejected(self) -> None:
        u = _grad("c-a", 1.0, 10)
        with pytest.raises(TypeError):
            apply_dp_noise(
                update=u,
                noise=_noise_g(),
                round_id="r-1",
                epsilon=1.0,
                meta={"bad": 1},  # type: ignore[dict-item]
            )

    @pytest.mark.parametrize(
        "key",
        [
            "raw_data",
            "training_data",
            "dataset",
            "samples",
            "features",
            "labels",
            "X",
            "y",
        ],
    )
    def test_privacy_keys_rejected_on_input_gradient(self, key: str) -> None:
        u = GradientUpdate(
            client_id="c-a",
            parameter="lr",
            delta=1.0,
            num_samples=10,
            ts_ns=100,
            meta={key: "leak"},
        )
        with pytest.raises(ValueError):
            apply_dp_noise(
                update=u,
                noise=_noise_g(),
                round_id="r-1",
                epsilon=1.0,
            )

    def test_gaussian_noise_scales_with_multiplier(self) -> None:
        """Larger multiplier produces larger absolute noise on average."""
        small_noises = []
        big_noises = []
        for i in range(20):
            u_i = _grad(f"c-{i}", 0.0, 10)
            cs = apply_dp_noise(
                update=u_i,
                noise=_noise_g(sens=1.0, mult=0.1),
                round_id="r-1",
                epsilon=1.0,
            )
            cb = apply_dp_noise(
                update=u_i,
                noise=_noise_g(sens=1.0, mult=10.0),
                round_id="r-1",
                epsilon=1.0,
            )
            small_noises.append(abs(cs.delta))
            big_noises.append(abs(cb.delta))
        # 100x noise multiplier should be visibly larger on average
        assert sum(big_noises) > 10 * sum(small_noises)


# ---------------------------------------------------------------------------
# aggregate_private_round — happy path
# ---------------------------------------------------------------------------


class TestAggregatePrivateRoundHappy:
    def test_single_round(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        report, update, next_a = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=contribs,
            accountant=a,
            ts_ns=100,
        )
        assert isinstance(report, PrivateRoundReport)
        assert isinstance(update, LearningUpdate)
        assert isinstance(next_a, PrivacyAccountant)
        assert report.n_clients == 2
        assert report.total_samples == 20
        assert next_a.n_rounds == 1
        assert next_a.epsilon_spent == 2.0

    def test_math_matches_fedavg(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        report, _, _ = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=contribs,
            accountant=a,
            ts_ns=100,
        )
        # Manually compute FedAvg over the projected gradients
        expected_delta, expected_total = fed_avg_aggregate(
            [c.as_gradient_update() for c in contribs],
        )
        assert report.aggregated_delta == expected_delta
        assert report.total_samples == expected_total

    def test_learning_update_carries_post_aggregation_value(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        report, update, _ = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=1.0,
            contributions=contribs,
            accountant=a,
            ts_ns=100,
        )
        # old_value/new_value are repr() of floats
        assert update.old_value == repr(1.0)
        assert update.new_value == repr(1.0 + report.aggregated_delta)

    def test_learning_update_meta(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        _, update, _ = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=contribs,
            accountant=a,
            ts_ns=100,
        )
        assert update.meta["lane"] == "federated_pysyft"
        assert update.meta["version"] == PYSYFT_VERSION
        assert update.meta["round_id"] == "r-1"
        assert update.meta["n_rounds"] == "1"

    def test_permutation_invariant(self) -> None:
        a = _accountant(eps=10.0)
        c1 = _private_contribution("c-a", 0.5, 10, eps=1.0)
        c2 = _private_contribution("c-b", 0.25, 10, eps=1.0)
        c3 = _private_contribution("c-c", 0.1, 5, eps=1.0)
        r1, _, n1 = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=[c1, c2, c3],
            accountant=a,
            ts_ns=100,
        )
        r2, _, n2 = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=[c3, c1, c2],
            accountant=a,
            ts_ns=100,
        )
        assert r1.digest == r2.digest
        assert r1.aggregated_delta == r2.aggregated_delta
        assert n1.epsilon_spent == n2.epsilon_spent

    def test_three_run_byte_identical_replay(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        digests = []
        eps_spents = []
        for _ in range(3):
            r, _, n = aggregate_private_round(
                round_id="r-1",
                parameter="lr",
                strategy_id="strategy-A",
                current_value=0.0,
                contributions=contribs,
                accountant=a,
                ts_ns=100,
            )
            digests.append(r.digest)
            eps_spents.append(n.epsilon_spent)
        assert digests[0] == digests[1] == digests[2]
        assert eps_spents[0] == eps_spents[1] == eps_spents[2]

    def test_multi_round_chain_accountant(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        running_value = 0.0
        for i in range(3):
            r, update, a = aggregate_private_round(
                round_id=f"r-{i}",
                parameter="lr",
                strategy_id="strategy-A",
                current_value=running_value,
                contributions=contribs,
                accountant=a,
                ts_ns=100 + i,
            )
            running_value = float(update.new_value)
        assert a.n_rounds == 3
        assert a.epsilon_spent == pytest.approx(6.0)
        assert math.isfinite(running_value)

    def test_round_digest_changes_with_round_id(self) -> None:
        a = _accountant(eps=10.0)
        contribs = [
            _private_contribution("c-a", 0.5, 10, eps=1.0),
            _private_contribution("c-b", 0.25, 10, eps=1.0),
        ]
        r1, _, _ = aggregate_private_round(
            round_id="r-1",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=contribs,
            accountant=a,
            ts_ns=100,
        )
        r2, _, _ = aggregate_private_round(
            round_id="r-2",
            parameter="lr",
            strategy_id="strategy-A",
            current_value=0.0,
            contributions=contribs,
            accountant=a,
            ts_ns=100,
        )
        assert r1.digest != r2.digest


# ---------------------------------------------------------------------------
# aggregate_private_round — rejection paths
# ---------------------------------------------------------------------------


class TestAggregatePrivateRoundRejection:
    def _go(self, **kw: object) -> None:
        defaults: dict[str, object] = {
            "round_id": "r-1",
            "parameter": "lr",
            "strategy_id": "strategy-A",
            "current_value": 0.0,
            "contributions": [
                _private_contribution("c-a", 0.5, 10, eps=1.0),
                _private_contribution("c-b", 0.25, 10, eps=1.0),
            ],
            "accountant": _accountant(eps=10.0),
            "ts_ns": 100,
        }
        defaults.update(kw)
        aggregate_private_round(**defaults)  # type: ignore[arg-type]

    def test_empty_round_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._go(round_id="")

    def test_empty_parameter_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._go(parameter="")

    def test_empty_strategy_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._go(strategy_id="")

    def test_current_value_nan_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._go(current_value=float("nan"))

    def test_ts_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._go(ts_ns=-1)

    def test_empty_contributions_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._go(contributions=[])

    def test_parameter_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="parameter"):
            self._go(
                contributions=[
                    _private_contribution(
                        "c-a",
                        0.5,
                        10,
                        eps=1.0,
                        parameter="other",
                    ),
                    _private_contribution("c-b", 0.25, 10, eps=1.0),
                ],
            )

    def test_duplicate_client_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            self._go(
                contributions=[
                    _private_contribution("c-a", 0.5, 10, eps=1.0),
                    _private_contribution("c-a", 0.25, 10, eps=1.0),
                ],
            )

    def test_budget_exhaustion_rejected(self) -> None:
        with pytest.raises(ValueError, match="privacy budget exhausted"):
            self._go(
                accountant=_accountant(eps=1.5),  # < total 2.0
            )


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


class TestASTGuardrails:
    @pytest.fixture(scope="class")
    def tree(self) -> ast.Module:
        return ast.parse(PYSYFT_PATH.read_text())

    def test_no_forbidden_top_level_imports(self, tree: ast.Module) -> None:
        forbidden = {
            "time",
            "datetime",
            "random",
            "asyncio",
            "os",
            "subprocess",
            "socket",
            "ssl",
            "numpy",
            "torch",
            "polars",
            "pandas",
            "requests",
            "httpx",
            "aiohttp",
            "tornado",
            "sqlite3",
            "syft",
            "openfl",
            "flwr",
            "fedml",
        }
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in forbidden, alias.name
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                assert top not in forbidden, node.module

    def test_no_runtime_tier_imports(self, tree: ast.Module) -> None:
        forbidden_tiers = (
            "intelligence_engine",
            "execution_engine",
            "governance_engine",
            "evolution_engine",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for tier in forbidden_tiers:
                        assert not alias.name.startswith(tier), alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for tier in forbidden_tiers:
                    assert not module.startswith(tier), module

    def test_syft_package_never_imported(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("syft"), alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("syft"), module

    def test_no_typed_event_constructors(self, tree: ast.Module) -> None:
        forbidden = {
            "SystemEvent",
            "HazardEvent",
            "SignalEvent",
            "ExecutionEvent",
            "PatchProposal",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in forbidden:
                    raise AssertionError(f"forbidden typed-event constructor: {fn.id}")
                if isinstance(fn, ast.Attribute) and fn.attr in forbidden:
                    raise AssertionError(f"forbidden typed-event constructor: {fn.attr}")

    def test_only_learning_update_emitted(self, tree: ast.Module) -> None:
        learning_update_seen = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "LearningUpdate":
                    learning_update_seen = True
        assert learning_update_seen

    def test_no_time_or_datetime_calls(self, tree: ast.Module) -> None:
        forbidden_attrs = {
            "time_ns",
            "monotonic_ns",
            "monotonic",
            "perf_counter_ns",
            "now",
            "utcnow",
            "today",
        }
        forbidden_callables = {"time", "datetime"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr in forbidden_attrs:
                    raise AssertionError(f"forbidden time/date call: {fn.attr}")
                if isinstance(fn, ast.Name) and fn.id in forbidden_callables:
                    raise AssertionError(f"forbidden call: {fn.id}")

    def test_no_random_module_calls(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "random":
                    raise AssertionError(f"forbidden random call: {node.attr}")
