"""Wave-04 PR-2 — ``make_trader_observation`` factory tests.

Pins the validation gates of the **only** B29-allowed runtime
construction point for :class:`TraderObservation`. Adapters
(:mod:`ui.feeds.tradingview_ideas`) hand off to this factory; if the
factory's contract weakens, every adapter weakens with it.

Coverage:

* Happy path — defaults and explicit kinds emit a well-formed record.
* Negative gates — non-negative ``ts_ns``, non-empty ``trader_id`` /
  ``source_feed``, observation-kind allowlist.
* ``meta`` is copied (not aliased) so callers can mutate their input.
* The factory does not read a clock — same inputs → same output.
"""

from __future__ import annotations

import pytest

from core.contracts.trader_intelligence import (
    TRADER_OBSERVATION_PROFILE_UPDATE,
    TRADER_OBSERVATION_SIGNAL_OBSERVED,
    TraderModel,
)
from intelligence_engine.trader_modeling.aggregator import (
    TRADER_MODELING_SOURCE,
    make_trader_observation,
)


def _model(**overrides: object) -> TraderModel:
    base: dict[str, object] = {
        "trader_id": "tv:alpha",
        "source_feed": "SRC-TRADER-TRADINGVIEW-001",
    }
    base.update(overrides)
    return TraderModel(**base)  # type: ignore[arg-type]


def test_factory_default_kind_is_profile_update():
    obs = make_trader_observation(ts_ns=1_700_000_000_000_000_000, model=_model())
    assert obs.observation_kind == TRADER_OBSERVATION_PROFILE_UPDATE
    assert obs.trader_id == "tv:alpha"
    assert obs.model.source_feed == "SRC-TRADER-TRADINGVIEW-001"
    assert obs.meta == {}


def test_factory_accepts_signal_observed():
    obs = make_trader_observation(
        ts_ns=1,
        model=_model(),
        observation_kind=TRADER_OBSERVATION_SIGNAL_OBSERVED,
    )
    assert obs.observation_kind == TRADER_OBSERVATION_SIGNAL_OBSERVED


def test_factory_rejects_unknown_observation_kind():
    with pytest.raises(ValueError, match="observation_kind"):
        make_trader_observation(
            ts_ns=1,
            model=_model(),
            observation_kind="WHIMSY",
        )


def test_factory_rejects_negative_ts_ns():
    with pytest.raises(ValueError, match="ts_ns"):
        make_trader_observation(ts_ns=-1, model=_model())


def test_factory_rejects_empty_trader_id():
    with pytest.raises(ValueError, match="trader_id"):
        make_trader_observation(ts_ns=0, model=_model(trader_id=""))


def test_factory_rejects_empty_source_feed():
    with pytest.raises(ValueError, match="source_feed"):
        make_trader_observation(ts_ns=0, model=_model(source_feed=""))


def test_factory_meta_is_copied_not_aliased():
    src: dict[str, str] = {"k": "v"}
    obs = make_trader_observation(ts_ns=0, model=_model(), meta=src)
    src["k"] = "MUTATED"
    assert obs.meta == {"k": "v"}


def test_factory_is_pure_same_input_same_output():
    a = make_trader_observation(ts_ns=42, model=_model())
    b = make_trader_observation(ts_ns=42, model=_model())
    assert a == b


def test_factory_constant_module_path_matches_b29_allowlist():
    # Belt-and-suspenders: the factory's source tag must be the same
    # string the B29 allowlist references, so ledger consumers can
    # filter trader-modeling rows without unpacking the payload.
    assert TRADER_MODELING_SOURCE == "intelligence_engine.trader_modeling"
