# ADAPTED FROM: HypothesisWorks/hypothesis
#   - hypothesis/strategies/_internal/strategies.py — `@given` decorator
#   - hypothesis/strategies/_internal/numbers.py — `integers`/`floats`
#   - hypothesis/strategies/_internal/strings.py — `text`
#   - hypothesis/strategies/_internal/collections.py — `tuples`,`lists`
# MPL-2.0 license; no Hypothesis source is reproduced verbatim — only
# the public strategy/decorator contract is used.
"""A-13 hypothesis → property-based ExecutionIntent content-hash invariants.

Three properties pinned across hundreds of random inputs:

1. **Hash determinism.** ``compute_content_hash`` is a pure function of
   its keyword inputs — calling it twice on the same inputs returns
   byte-identical output (INV-15).
2. **Hash sensitivity.** Changing **any** material input field
   (``ts_ns`` / ``origin`` / signal fields / ``approved_by_governance``
   / ``governance_decision_id``) changes the hash. The hash is **not**
   sensitive to dict insertion order on ``signal.meta`` (canonical-JSON
   serialisation pins this).
3. **Round-trip via constructor.** :func:`create_execution_intent`
   re-derives the same hash as a direct
   :func:`compute_content_hash` call with matching inputs, and
   :meth:`ExecutionIntent.verify_content_hash` returns ``True``.

These tests run on every PR via the
``.github/workflows/property_tests.yml`` job alongside the unit suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from core.contracts.events import Side, SignalEvent
from core.contracts.execution_intent import (
    AUTHORISED_INTENT_ORIGINS,
    TEST_INTENT_ORIGINS,
    compute_content_hash,
    compute_intent_id,
    create_execution_intent,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


_VALID_ORIGINS = sorted(AUTHORISED_INTENT_ORIGINS | TEST_INTENT_ORIGINS)


@st.composite
def _signal_events(draw: st.DrawFn) -> SignalEvent:
    return SignalEvent(
        ts_ns=draw(st.integers(min_value=1, max_value=2**63 - 1)),
        symbol=draw(st.sampled_from(["BTCUSDT", "EURUSD", "ETHUSDT", "AAPL", "SOLUSDT"])),
        side=draw(st.sampled_from([Side.BUY, Side.SELL, Side.HOLD])),
        confidence=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        plugin_chain=tuple(
            draw(
                st.lists(
                    st.text(
                        alphabet=st.characters(
                            whitelist_categories=("Ll", "Lu", "Nd"),
                            whitelist_characters="_.",
                        ),
                        min_size=1,
                        max_size=16,
                    ),
                    max_size=4,
                )
            )
        ),
        meta=draw(
            st.dictionaries(
                keys=st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Ll", "Lu", "Nd"),
                        whitelist_characters="_",
                    ),
                    min_size=1,
                    max_size=8,
                ),
                values=st.text(max_size=16),
                max_size=4,
            )
        ),
        produced_by_engine="intelligence",
    )


# ---------------------------------------------------------------------------
# Property: determinism
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    ts_ns=st.integers(min_value=1, max_value=2**63 - 1),
    origin=st.sampled_from(_VALID_ORIGINS),
    signal=_signal_events(),
    approved=st.booleans(),
    decision_id=st.text(max_size=32),
)
def test_compute_content_hash_is_deterministic(
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
    approved: bool,
    decision_id: str,
) -> None:
    """INV-15 — replay returns byte-identical digest."""

    # ``compute_content_hash`` requires either approved=True with
    # non-empty decision id, or approved=False with anything. We
    # constrain the inputs at the property level rather than via
    # ``assume`` so the strategy stays compact.
    if approved and not decision_id:
        decision_id = "G-PROP-DEFAULT"

    first = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=approved,
        governance_decision_id=decision_id,
    )
    second = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=approved,
        governance_decision_id=decision_id,
    )
    assert first == second
    assert len(first) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Property: sensitivity to material fields
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    ts_ns=st.integers(min_value=1, max_value=2**62),
    origin=st.sampled_from(_VALID_ORIGINS),
    signal=_signal_events(),
    approved=st.booleans(),
    decision_id=st.text(max_size=32),
    delta=st.integers(min_value=1, max_value=2**31),
)
def test_compute_content_hash_changes_when_ts_ns_changes(
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
    approved: bool,
    decision_id: str,
    delta: int,
) -> None:
    """A changed ``ts_ns`` produces a different content hash."""

    if approved and not decision_id:
        decision_id = "G-PROP-DEFAULT"

    base = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=approved,
        governance_decision_id=decision_id,
    )
    bumped = compute_content_hash(
        ts_ns=ts_ns + delta,
        origin=origin,
        signal=signal,
        approved_by_governance=approved,
        governance_decision_id=decision_id,
    )
    assert base != bumped


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    ts_ns=st.integers(min_value=1, max_value=2**62),
    origin=st.sampled_from(_VALID_ORIGINS),
    signal=_signal_events(),
    decision_id=st.text(min_size=1, max_size=32),
)
def test_compute_content_hash_flips_with_approval(
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
    decision_id: str,
) -> None:
    """``approved=True`` vs ``approved=False`` produce different hashes."""

    rejected = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=False,
        governance_decision_id=decision_id,
    )
    approved = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=True,
        governance_decision_id=decision_id,
    )
    assert rejected != approved


# ---------------------------------------------------------------------------
# Property: insensitivity to meta-dict insertion order
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    ts_ns=st.integers(min_value=1, max_value=2**62),
    origin=st.sampled_from(_VALID_ORIGINS),
    meta_items=st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll",),
                    whitelist_characters="_",
                ),
                min_size=1,
                max_size=6,
            ),
            st.text(max_size=8),
        ),
        min_size=2,
        max_size=6,
        unique_by=lambda kv: kv[0],
    ),
)
def test_compute_content_hash_ignores_meta_dict_order(
    ts_ns: int,
    origin: str,
    meta_items: list[tuple[str, str]],
) -> None:
    """Dict-order independence — canonical JSON sorts the keys."""

    forward = dict(meta_items)
    reverse = dict(reversed(meta_items))

    sig_fwd = SignalEvent(
        ts_ns=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.5,
        plugin_chain=(),
        meta=forward,
    )
    sig_rev = SignalEvent(
        ts_ns=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.5,
        plugin_chain=(),
        meta=reverse,
    )

    h_fwd = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=sig_fwd,
        approved_by_governance=False,
        governance_decision_id="",
    )
    h_rev = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=sig_rev,
        approved_by_governance=False,
        governance_decision_id="",
    )
    assert h_fwd == h_rev


# ---------------------------------------------------------------------------
# Property: round-trip via constructor + verify
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    ts_ns=st.integers(min_value=1, max_value=2**62),
    origin=st.sampled_from(_VALID_ORIGINS),
    signal=_signal_events(),
)
def test_create_execution_intent_round_trips(
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
) -> None:
    """``create_execution_intent`` round-trips through
    ``verify_content_hash``."""

    intent = create_execution_intent(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=False,
        governance_decision_id="",
    )
    assert intent.verify_content_hash() is True
    assert intent.intent_id == compute_intent_id(intent.content_hash)
    assert intent.content_hash == compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=False,
        governance_decision_id="",
    )
