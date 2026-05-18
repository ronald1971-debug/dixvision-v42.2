# ADAPTED FROM: AI4Finance-Foundation/FinRL
# (finrl/meta/env_stock_trading/env_stocktrading.py — StockTradingEnv
#  gym.Env; state-vector layout (cash | holdings | close prices | tech
#  indicators); continuous per-asset action in [-1, 1] scaled by hmax;
#  per-side transaction-cost penalty; reward = delta(portfolio_value) *
#  reward_scaling. License: MIT.)
"""B-12 — finrl_env: deterministic multi-asset RL environment.

FinRL's ``StockTradingEnv`` exposes a multi-stock portfolio to RL
trainers (SB3, CleanRL, ElegantRL) as a single ``gym.Env``. Its core
contract is:

* **State** ``= [cash, n_held[0..N-1], close[0..N-1], tech[0..N-1, ...]]``
* **Action** ``= a in [-1, 1]^N``; scaled by ``hmax`` -> integer
  share-delta per asset.
* **Step** advances one bar; applies sells first (frees cash), then buys
  (gated by available cash), each with a per-side proportional
  transaction cost.
* **Reward** ``= (portfolio_value(t+1) - portfolio_value(t)) * reward_scaling``.

This module adopts the **contract shape** of that env so any RL
trainer that already speaks gym/gymnasium can step DIX simulation data
unchanged. We do **not** import ``finrl``, ``gym``, ``gymnasium``, or
``pandas``.  The bar series is a deterministic, frozen tuple of
:class:`Bar` records supplied by the caller (typically a projection
from the audit ledger or :class:`~core.contracts.backtest_result.BacktestResult`).

Tier
----
**OFFLINE_ONLY.**  ``learning_engine/lanes/`` is the slow-cadence RL
lane tier. Never imported from ``execution_engine/`` /
``governance_engine/`` / ``system_engine/`` / ``core/`` /
``intelligence_engine/meta_controller/hot_path.py``.

What survives from upstream FinRL
---------------------------------
* The 5-tuple ``step(action) -> (observation, reward, terminated,
  truncated, info)`` and ``reset(*, seed, options=None) -> (observation,
  info)`` shapes (Gymnasium ≥ 0.26 split-done convention).
* The portfolio state-vector layout (cash | holdings | prices).
* The per-side transaction-cost model
  (``buy_cost_pct`` / ``sell_cost_pct``).
* The reward shaper ``delta(portfolio_value) * reward_scaling``.
* The ``hmax``-scaled continuous action space in ``[-1, 1]^N``.

What we replaced
----------------
* FinRL's ``pandas.DataFrame`` of historical OHLCV ->
  a frozen tuple of :class:`Bar` records sorted by ``(ts_ns, symbol)``.
* FinRL's gym ``Box`` action / observation spaces ->
  frozen :class:`PortfolioAction` / :class:`PortfolioObservation` value
  objects + helper :func:`observation_to_tuple` for flat-vector consumers.
* FinRL's float-share fractional holdings ->
  integer share counts (FinRL's default; we keep it explicit so the
  ``hmax`` clamp matches upstream).
* FinRL's ``np.random.RandomState`` reset PRNG ->
  caller-supplied seed (INV-15); the env itself is fully deterministic
  given ``(bars, episode_config, seed, action_sequence)``.

Authority constraints
---------------------
* OFFLINE_ONLY, RUNTIME_SAFE-shape — wall-clock free, IO free,
  randomness free, no engine cross-imports.
* B27 / B28 / INV-71 — module never constructs ``SignalEvent`` /
  ``ExecutionIntent`` / ``HazardEvent`` / ``GovernanceDecision`` /
  ``PatchProposal``; outputs are advisory value objects only.
* INV-15 byte-identical replay; pinned by 3-run equality test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""B-12 introduces no new pip dependencies.

FinRL itself ships as a heavy ML stack (``pandas``, ``torch``, several
exchange data adapters). We adapt the env contract clean-room in pure
stdlib so the offline RL lanes import cleanly on every host.
"""

# Hard upper bounds for offline RL training safety.
MAX_ASSETS: Final[int] = 1024
MAX_BARS_PER_EPISODE: Final[int] = 1_000_000

# IEEE-754 round-off threshold used when comparing notional / cost
# computations (e.g. asserting cost-conservation in tests).
_FLOAT_EPSILON: Final[float] = 1e-9


class FinRLEnvError(ValueError):
    """Raised on any malformed input to the env constructor / step / reset."""


class EpisodeNotStartedError(RuntimeError):
    """Raised when :meth:`FinRLPortfolioEnv.step` is called before reset."""


@dataclass(frozen=True, slots=True)
class Bar:
    """One bar of multi-asset OHLCV.

    Only ``close`` is required for the FinRL state vector; ``open`` /
    ``high`` / ``low`` / ``volume`` are reserved for downstream
    tech-indicator computations and default to the close price.
    """

    ts_ns: int
    symbol: str
    close: float
    open_: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise FinRLEnvError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not self.symbol:
            raise FinRLEnvError("symbol must be non-empty")
        if not (self.close > 0.0 and math.isfinite(self.close)):
            raise FinRLEnvError(f"close must be > 0 and finite; got {self.close}")
        for name in ("open_", "high", "low", "volume"):
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be float; got {type(value)!r}")
            if not math.isfinite(value):
                raise FinRLEnvError(f"{name} must be finite; got {value}")
            if value < 0.0:
                raise FinRLEnvError(f"{name} must be >= 0; got {value}")


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Episode-level FinRL configuration.

    Mirrors FinRL's :class:`StockTradingEnv` constructor:

    * ``symbols`` — universe (alphabetically sorted on validation).
    * ``initial_cash`` — starting wallet in quote currency.
    * ``hmax`` — per-asset share-delta scaling for the ``[-1, 1]``
      continuous action.
    * ``buy_cost_pct`` / ``sell_cost_pct`` — proportional transaction
      cost per side, in decimal (``0.001`` == 10 bps).
    * ``reward_scaling`` — caller-supplied multiplier on
      ``delta(portfolio_value)`` so the env stays unitful regardless
      of cash scale.
    * ``max_steps`` — episode horizon; ``None`` means run through every
      timestamp in the bar series.
    """

    symbols: tuple[str, ...]
    initial_cash: float = 1_000_000.0
    hmax: int = 100
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 1.0
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.symbols:
            raise FinRLEnvError("symbols must be non-empty")
        if len(self.symbols) > MAX_ASSETS:
            raise FinRLEnvError(f"too many assets: {len(self.symbols)} > {MAX_ASSETS}")
        if len(self.symbols) != len(set(self.symbols)):
            raise FinRLEnvError("symbols must be unique")
        if tuple(sorted(self.symbols)) != tuple(self.symbols):
            raise FinRLEnvError(f"symbols must be sorted ascending; got {self.symbols!r}")
        if not (self.initial_cash > 0.0 and math.isfinite(self.initial_cash)):
            raise FinRLEnvError(f"initial_cash must be > 0 and finite; got {self.initial_cash}")
        if not isinstance(self.hmax, int) or isinstance(self.hmax, bool):
            raise TypeError(f"hmax must be int; got {type(self.hmax)!r}")
        if self.hmax <= 0:
            raise FinRLEnvError(f"hmax must be > 0; got {self.hmax}")
        for name in ("buy_cost_pct", "sell_cost_pct"):
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be float; got {type(value)!r}")
            if not (0.0 <= value < 1.0):
                raise FinRLEnvError(f"{name} must be in [0, 1); got {value}")
        if not isinstance(self.reward_scaling, float):
            raise TypeError(f"reward_scaling must be float; got {type(self.reward_scaling)!r}")
        if not math.isfinite(self.reward_scaling):
            raise FinRLEnvError(f"reward_scaling must be finite; got {self.reward_scaling}")
        if self.max_steps is not None:
            if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
                raise TypeError(f"max_steps must be int|None; got {type(self.max_steps)!r}")
            if self.max_steps <= 0:
                raise FinRLEnvError(f"max_steps must be > 0; got {self.max_steps}")


@dataclass(frozen=True, slots=True)
class PortfolioAction:
    """Continuous per-asset action in ``[-1, 1]^N``.

    Same order as ``EpisodeConfig.symbols``. ``+1`` requests buying
    ``hmax`` shares; ``-1`` requests selling ``hmax`` shares; ``0``
    holds. Out-of-range values raise immediately rather than silently
    clipping so the caller is informed.
    """

    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.targets:
            raise FinRLEnvError("targets must be non-empty")
        for i, value in enumerate(self.targets):
            if not isinstance(value, float):
                raise TypeError(f"targets[{i}] must be float; got {type(value)!r}")
            if not math.isfinite(value):
                raise FinRLEnvError(f"targets[{i}] must be finite; got {value}")
            if not (-1.0 - _FLOAT_EPSILON <= value <= 1.0 + _FLOAT_EPSILON):
                raise FinRLEnvError(f"targets[{i}] must be in [-1, 1]; got {value}")


@dataclass(frozen=True, slots=True)
class PortfolioObservation:
    """One observation of the FinRL portfolio state.

    Layout matches FinRL's flat state vector when projected via
    :func:`observation_to_tuple`::

        [cash, holdings[0..N-1], close[0..N-1]]
    """

    step_idx: int
    ts_ns: int
    cash: float
    holdings: tuple[int, ...]
    prices: tuple[float, ...]
    portfolio_value: float

    def __post_init__(self) -> None:
        if self.step_idx < 0:
            raise FinRLEnvError(f"step_idx must be >= 0; got {self.step_idx}")
        if self.ts_ns < 0:
            raise FinRLEnvError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not (math.isfinite(self.cash) and math.isfinite(self.portfolio_value)):
            raise FinRLEnvError("cash / portfolio_value must be finite")
        if len(self.holdings) != len(self.prices):
            raise FinRLEnvError(
                f"holdings ({len(self.holdings)}) / prices ({len(self.prices)}) length mismatch"
            )
        for i, h in enumerate(self.holdings):
            if not isinstance(h, int) or isinstance(h, bool):
                raise TypeError(f"holdings[{i}] must be int; got {type(h)!r}")
            if h < 0:
                raise FinRLEnvError(f"holdings[{i}] must be >= 0; got {h}")
        for i, p in enumerate(self.prices):
            if not isinstance(p, float):
                raise TypeError(f"prices[{i}] must be float; got {type(p)!r}")
            if not (p > 0.0 and math.isfinite(p)):
                raise FinRLEnvError(f"prices[{i}] must be > 0 and finite; got {p}")


@dataclass(frozen=True, slots=True)
class StepResult:
    """The 5-tuple any Gymnasium ≥ 0.26 caller expects from ``step``.

    Wrapped as a frozen value object to keep the public API typed; a
    helper :meth:`as_tuple` returns the canonical Gymnasium 5-tuple.
    """

    observation: PortfolioObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reward, float):
            raise TypeError(f"reward must be float; got {type(self.reward)!r}")
        if not math.isfinite(self.reward):
            raise FinRLEnvError(f"reward must be finite; got {self.reward}")
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be bool")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be bool")
        for key, value in self.info.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("info keys and values must be str")

    def as_tuple(
        self,
    ) -> tuple[PortfolioObservation, float, bool, bool, Mapping[str, str]]:
        """Return the Gymnasium-canonical 5-tuple."""
        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
        )


def _validate_bars(
    bars: Sequence[Bar],
    symbols: tuple[str, ...],
) -> tuple[tuple[int, dict[str, float]], ...]:
    """Project a flat bar tuple into a sorted, per-timestamp price map.

    Bars are grouped by ``ts_ns``; every timestamp must carry exactly
    one close price per symbol in :attr:`EpisodeConfig.symbols`.
    Returns a tuple of ``(ts_ns, {symbol: close})`` pairs sorted by
    ``ts_ns`` ascending — the deterministic step stream the env walks.
    """

    if not bars:
        raise FinRLEnvError("bars must be non-empty")
    if len(bars) > MAX_BARS_PER_EPISODE:
        raise FinRLEnvError(f"too many bars: {len(bars)} > {MAX_BARS_PER_EPISODE}")
    symbol_set = set(symbols)
    grouped: dict[int, dict[str, float]] = {}
    for bar in bars:
        if not isinstance(bar, Bar):
            raise TypeError(f"bars entry must be Bar; got {type(bar)!r}")
        if bar.symbol not in symbol_set:
            raise FinRLEnvError(f"bar.symbol {bar.symbol!r} not in episode symbols {symbols!r}")
        per_ts = grouped.setdefault(bar.ts_ns, {})
        if bar.symbol in per_ts:
            raise FinRLEnvError(f"duplicate bar for ({bar.ts_ns}, {bar.symbol!r})")
        per_ts[bar.symbol] = bar.close
    for ts, per_ts in grouped.items():
        if set(per_ts.keys()) != symbol_set:
            missing = symbol_set - per_ts.keys()
            raise FinRLEnvError(f"timestamp {ts} missing close for {sorted(missing)!r}")
    return tuple((ts, grouped[ts]) for ts in sorted(grouped.keys()))


class FinRLPortfolioEnv:
    """Deterministic FinRL-shaped portfolio environment.

    Caller usage::

        env = FinRLPortfolioEnv(bars=..., config=...)
        obs, info = env.reset(seed=42)
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action).as_tuple()
            if terminated or truncated:
                break

    The env is **single-episode-per-instance** in the sense that
    :meth:`reset` rewinds it cleanly; concurrent stepping requires
    separate instances (mirrors FinRL's upstream contract).
    """

    __slots__ = (
        "_bars",
        "_config",
        "_n_assets",
        "_started",
        "_step_idx",
        "_seed",
        "_cash",
        "_holdings",
        "_prev_portfolio_value",
    )

    def __init__(
        self,
        bars: Sequence[Bar],
        config: EpisodeConfig,
    ) -> None:
        if not isinstance(config, EpisodeConfig):
            raise TypeError(f"config must be EpisodeConfig; got {type(config)!r}")
        self._bars = _validate_bars(bars, config.symbols)
        self._config = config
        self._n_assets = len(config.symbols)
        self._started = False
        self._step_idx = 0
        self._seed = 0
        self._cash = 0.0
        self._holdings: list[int] = []
        self._prev_portfolio_value = 0.0

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def n_bars(self) -> int:
        return len(self._bars)

    @property
    def config(self) -> EpisodeConfig:
        return self._config

    def _current_prices(self) -> tuple[float, ...]:
        _, prices = self._bars[self._step_idx]
        return tuple(prices[s] for s in self._config.symbols)

    def _portfolio_value(self, prices: tuple[float, ...]) -> float:
        notional = math.fsum(h * p for h, p in zip(self._holdings, prices, strict=True))
        return self._cash + notional

    def _build_observation(self) -> PortfolioObservation:
        ts, _ = self._bars[self._step_idx]
        prices = self._current_prices()
        return PortfolioObservation(
            step_idx=self._step_idx,
            ts_ns=ts,
            cash=self._cash,
            holdings=tuple(self._holdings),
            prices=prices,
            portfolio_value=self._portfolio_value(prices),
        )

    def reset(
        self,
        *,
        seed: int = 0,
        options: Mapping[str, str] | None = None,  # noqa: ARG002 — Gym contract
    ) -> tuple[PortfolioObservation, Mapping[str, str]]:
        """Rewind to the first bar and return the initial observation.

        Mirrors Gymnasium's
        ``Env.reset(*, seed, options) -> (observation, info)`` contract.
        ``options`` is accepted for Gym-shape compatibility but is not
        used — DIX seeds bar-series determinism through the env
        constructor, not through reset options.
        """

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be int; got {type(seed)!r}")
        if seed < 0:
            raise FinRLEnvError(f"seed must be >= 0; got {seed}")
        self._seed = seed
        self._step_idx = 0
        self._cash = self._config.initial_cash
        self._holdings = [0] * self._n_assets
        obs = self._build_observation()
        self._prev_portfolio_value = obs.portfolio_value
        self._started = True
        info: dict[str, str] = {
            "seed": str(seed),
            "n_assets": str(self._n_assets),
            "n_bars": str(self.n_bars),
        }
        return obs, info

    def _apply_sells(
        self,
        action: PortfolioAction,
        prices: tuple[float, ...],
    ) -> float:
        """Apply all sell legs first (FinRL ordering). Returns proceeds."""

        proceeds = 0.0
        for i, target in enumerate(action.targets):
            if target >= 0.0:
                continue
            desired = int(target * self._config.hmax)
            # target is negative -> desired is negative; cap at holdings.
            qty_to_sell = min(-desired, self._holdings[i])
            if qty_to_sell <= 0:
                continue
            gross = qty_to_sell * prices[i]
            cost = gross * self._config.sell_cost_pct
            net = gross - cost
            self._holdings[i] -= qty_to_sell
            proceeds += net
        return proceeds

    def _apply_buys(
        self,
        action: PortfolioAction,
        prices: tuple[float, ...],
    ) -> None:
        """Apply all buy legs in symbol order (FinRL ordering)."""

        for i, target in enumerate(action.targets):
            if target <= 0.0:
                continue
            desired = int(target * self._config.hmax)
            if desired <= 0:
                continue
            unit_cost = prices[i] * (1.0 + self._config.buy_cost_pct)
            if unit_cost <= 0.0:
                continue
            affordable = int(self._cash // unit_cost)
            qty_to_buy = min(desired, affordable)
            if qty_to_buy <= 0:
                continue
            outlay = qty_to_buy * unit_cost
            self._holdings[i] += qty_to_buy
            self._cash -= outlay

    def step(self, action: PortfolioAction) -> StepResult:
        """Apply *action*, advance one bar, return the 5-tuple."""

        if not self._started:
            raise EpisodeNotStartedError("reset() must be called before step()")
        if not isinstance(action, PortfolioAction):
            raise TypeError(f"action must be PortfolioAction; got {type(action)!r}")
        if len(action.targets) != self._n_assets:
            raise FinRLEnvError(
                f"action.targets length ({len(action.targets)}) != n_assets ({self._n_assets})"
            )
        prices_at_action = self._current_prices()
        # FinRL canonical ordering: sells first to free cash, then buys.
        self._cash += self._apply_sells(action, prices_at_action)
        self._apply_buys(action, prices_at_action)

        # Advance one bar; the reward uses the new bar's close.
        terminated = False
        truncated = False
        if self._step_idx + 1 >= self.n_bars:
            terminated = True
        else:
            self._step_idx += 1
            if self._config.max_steps is not None and self._step_idx >= self._config.max_steps:
                truncated = True

        new_prices = self._current_prices()
        new_value = self._portfolio_value(new_prices)
        reward = (new_value - self._prev_portfolio_value) * self._config.reward_scaling
        self._prev_portfolio_value = new_value

        obs = PortfolioObservation(
            step_idx=self._step_idx,
            ts_ns=self._bars[self._step_idx][0],
            cash=self._cash,
            holdings=tuple(self._holdings),
            prices=new_prices,
            portfolio_value=new_value,
        )
        info: dict[str, str] = {
            "step_idx": str(self._step_idx),
            "terminated": str(terminated).lower(),
            "truncated": str(truncated).lower(),
        }
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )


def observation_to_tuple(obs: PortfolioObservation) -> tuple[float, ...]:
    """Flatten ``obs`` into FinRL's canonical ``[cash, holdings, prices]`` vector.

    Useful for callers that consume gym ``Box``-style observation
    spaces (SB3, CleanRL). The order is the same as
    :attr:`EpisodeConfig.symbols` for both ``holdings`` and ``prices``.
    """

    out: list[float] = [obs.cash]
    out.extend(float(h) for h in obs.holdings)
    out.extend(obs.prices)
    return tuple(out)


def run_episode(
    env: FinRLPortfolioEnv,
    actions: Iterable[PortfolioAction],
    *,
    seed: int = 0,
) -> tuple[PortfolioObservation, ...]:
    """Run *env* through *actions* and return the observation trajectory.

    Pure convenience for the test suite + deterministic replay
    harnesses. Stops at the first terminated / truncated step.
    """

    trajectory: list[PortfolioObservation] = []
    obs, _ = env.reset(seed=seed)
    trajectory.append(obs)
    for action in actions:
        result = env.step(action)
        trajectory.append(result.observation)
        if result.terminated or result.truncated:
            break
    return tuple(trajectory)


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MAX_ASSETS",
    "MAX_BARS_PER_EPISODE",
    "FinRLEnvError",
    "EpisodeNotStartedError",
    "Bar",
    "EpisodeConfig",
    "PortfolioAction",
    "PortfolioObservation",
    "StepResult",
    "FinRLPortfolioEnv",
    "observation_to_tuple",
    "run_episode",
)
