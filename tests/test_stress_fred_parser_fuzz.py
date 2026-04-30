"""Wave-Stress-Tests — adversarial fuzzing of the FRED HTTP parser.

The :func:`parse_observations_payload` function (PR #108, Wave-04.5
PR-2) is the projection from raw HTTP response bodies into
:class:`MacroObservation` tuples. The build plan's INV-15 invariant
requires that:

* The parser is **pure** — same bytes in, same tuples out, no clock,
  no I/O, no PRNG.
* The parser **never raises** on hostile input (the run-loop must not
  blow up on a single bad payload).
* The parser **never silently emits** observations whose contract
  fields would be invalid (empty ``date`` is dropped).

This file points adversarial inputs at the parser and asserts those
invariants hold. INV-15: every random draw uses a seeded
:class:`random.Random` so failures are deterministic.

Backend-only — no UI surface.
"""

from __future__ import annotations

import json
import random

from core.contracts.macro import MacroObservation
from ui.feeds.fred_http import (
    SOURCE_TAG,
    parse_observations_payload,
)

# ---------------------------------------------------------------------------
# 1. Hostile-payload fuzz — must never raise
# ---------------------------------------------------------------------------


_HOSTILE_PAYLOADS: tuple[bytes | str, ...] = (
    b"",
    b" ",
    b"\x00",
    b"\xff\xfe\xfd",  # invalid UTF-8
    b"null",
    b"true",
    b"123",
    b"[]",
    b"\"hello\"",
    b"{}",
    b'{"observations": null}',
    b'{"observations": "not-a-list"}',
    b'{"observations": 42}',
    b'{"observations": [null, 1, "x", []]}',
    b'{"observations": [{}]}',
    b'{"observations": [{"date": null, "value": "1.0"}]}',
    b'{"observations": [{"date": "", "value": "1.0"}]}',
    b'{"observations": [{"date": 20240101, "value": "1.0"}]}',
    b'{"observations": [{"date": "not-a-date", "value": "1.0"}]}',
    b'{"observations": [{"date": "2024-01-01", "value": "."}]}',
    b'{"observations": [{"date": "2024-01-01", "value": null}]}',
    b'{"observations": [{"date": "2024-13-99", "value": "1.0"}]}',
    b'{"observations": [{"date": "2024-01-01"}]}',  # missing value
    b'{"junk_key": [1,2,3]}',
    b"{",  # malformed JSON
    b"]]]",  # malformed JSON
    "{\"observations\": []}",
    "",
)


def test_parse_never_raises_on_hostile_input() -> None:
    """Every hostile input must surface as an empty tuple, never an exception."""
    for payload in _HOSTILE_PAYLOADS:
        out = parse_observations_payload(
            payload,
            ts_ns=1_000,
            series_id="X",
        )
        assert isinstance(out, tuple)
        for o in out:
            assert isinstance(o, MacroObservation)


def test_parse_dropping_invalid_dates_does_not_yield_empty_date_observations() -> None:
    """Observations missing a ``date`` field must not become MacroObservation.

    INV-15 contract: ``MacroObservation.observation_date`` must be
    non-empty.
    """
    payload = json.dumps(
        {
            "observations": [
                {"date": "", "value": "1.0"},
                {"date": None, "value": "2.0"},
                {"value": "3.0"},  # missing date entirely
                {"date": "2024-01-15", "value": "4.5"},
            ]
        }
    ).encode("utf-8")

    out = parse_observations_payload(
        payload, ts_ns=2_000, series_id="DGS10"
    )

    # Only the well-formed row survives.
    assert len(out) == 1
    assert out[0].observation_date == "2024-01-15"


# ---------------------------------------------------------------------------
# 2. Determinism — same bytes in, same tuples out
# ---------------------------------------------------------------------------


def _random_well_formed_payload(rng: random.Random) -> bytes:
    n = rng.randint(0, 30)
    obs = []
    for _ in range(n):
        year = rng.randint(1900, 2099)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        # value can be number-string, "." (FRED's missing marker),
        # or junk that the parser must drop
        choice = rng.randint(0, 4)
        if choice == 0:
            value = f"{rng.uniform(-1e6, 1e6):.6f}"
        elif choice == 1:
            value = "."  # FRED's missing-data marker
        elif choice == 2:
            value = ""
        elif choice == 3:
            value = "abc"
        else:
            value = str(rng.randint(-1000, 1000))
        obs.append(
            {
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "value": value,
            }
        )
    return json.dumps({"observations": obs}).encode("utf-8")


def test_parse_is_pure_byte_identical_replay() -> None:
    """Same payload + ts_ns + series_id => byte-identical tuple twice."""
    rng = random.Random(20260420)
    for _ in range(100):
        payload = _random_well_formed_payload(rng)
        ts_ns = rng.randint(1, 1_000_000_000)
        series = rng.choice(("X", "DGS10", "UNRATE", "CPIAUCNS"))
        a = parse_observations_payload(
            payload, ts_ns=ts_ns, series_id=series
        )
        b = parse_observations_payload(
            payload, ts_ns=ts_ns, series_id=series
        )
        assert a == b, "parser is not pure under repeated invocation"


def test_parse_ts_ns_propagates_unchanged_to_every_observation() -> None:
    """Caller-supplied ``ts_ns`` must appear on every observation.

    INV-15 (pure projection): ``ts_ns`` is supplied by the caller and
    never derived from the payload or a system clock.
    """
    rng = random.Random(123)
    payload = _random_well_formed_payload(rng)
    ts_ns = 42_424_242
    out = parse_observations_payload(
        payload, ts_ns=ts_ns, series_id="DGS10"
    )
    for o in out:
        assert o.ts_ns == ts_ns


def test_parse_series_id_propagates_unchanged_to_every_observation() -> None:
    """Caller-supplied ``series_id`` must appear verbatim on every row."""
    rng = random.Random(7)
    payload = _random_well_formed_payload(rng)
    out = parse_observations_payload(
        payload, ts_ns=1, series_id="UNRATE"
    )
    for o in out:
        assert o.series_id == "UNRATE"


def test_parse_default_source_tag_is_FRED() -> None:
    rng = random.Random(11)
    payload = _random_well_formed_payload(rng)
    out = parse_observations_payload(
        payload, ts_ns=1, series_id="X"
    )
    for o in out:
        assert o.source == SOURCE_TAG == "FRED"


def test_parse_custom_source_tag_propagates() -> None:
    rng = random.Random(13)
    payload = _random_well_formed_payload(rng)
    out = parse_observations_payload(
        payload, ts_ns=1, series_id="X", source="ALT"
    )
    for o in out:
        assert o.source == "ALT"


# ---------------------------------------------------------------------------
# 3. Required-field guards
# ---------------------------------------------------------------------------


def test_parse_rejects_empty_series_id() -> None:
    try:
        parse_observations_payload(
            b'{"observations": []}', ts_ns=1, series_id=""
        )
    except ValueError as exc:
        assert "series_id" in str(exc)
        return
    raise AssertionError("empty series_id was not rejected")


def test_parse_invalid_utf8_returns_empty_tuple() -> None:
    """Bytes that aren't valid UTF-8 must surface as ``()``, not raise."""
    out = parse_observations_payload(
        b"\xff\xfe\xfd\xfc",
        ts_ns=1,
        series_id="X",
    )
    assert out == ()


# ---------------------------------------------------------------------------
# 4. Many-call fuzz — invariants hold across long runs
# ---------------------------------------------------------------------------


def test_parse_many_iterations_no_invariant_violation() -> None:
    """1000 random payloads. Every observation that is emitted has:
    * non-empty ``observation_date``
    * caller's ``ts_ns``
    * caller's ``series_id``
    * caller's ``source``
    """
    rng = random.Random(20262026)
    for _ in range(1000):
        payload = _random_well_formed_payload(rng)
        ts_ns = rng.randint(1, 1_000_000_000)
        series = rng.choice(("DGS10", "CPIAUCNS", "UNRATE"))
        source = rng.choice(("FRED", "FRED-MIRROR-A"))

        out = parse_observations_payload(
            payload, ts_ns=ts_ns, series_id=series, source=source
        )

        for o in out:
            assert o.observation_date, "empty observation_date emitted"
            assert o.ts_ns == ts_ns
            assert o.series_id == series
            assert o.source == source


def test_parse_handles_payload_bytes_or_str_identically() -> None:
    """``bytes`` and ``str`` of the same JSON must yield equal tuples."""
    rng = random.Random(99)
    for _ in range(20):
        payload_bytes = _random_well_formed_payload(rng)
        payload_str = payload_bytes.decode("utf-8")

        a = parse_observations_payload(
            payload_bytes, ts_ns=42, series_id="X"
        )
        b = parse_observations_payload(
            payload_str, ts_ns=42, series_id="X"
        )
        assert a == b, "bytes vs str inputs diverged"
