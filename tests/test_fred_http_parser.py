"""Tests for the pure FRED HTTP parser layer.

These tests exercise :func:`ui.feeds.fred_http.parse_observations_payload`
and :class:`core.contracts.macro.MacroObservation` validation. No I/O,
no clock, no network — the parser is pure and the caller supplies
``ts_ns``.
"""

from __future__ import annotations

import pytest

from core.contracts.macro import MacroObservation
from ui.feeds.fred_http import (
    FRED_API_BASE,
    SOURCE_TAG,
    make_fred_observations_url,
    parse_observations_payload,
)

# ---------------------------------------------------------------------------
# Fixture documents
# ---------------------------------------------------------------------------

_FRED_TWO_OBS = b"""{
  "realtime_start": "2026-04-21",
  "realtime_end": "2026-04-21",
  "observations": [
    {"date": "2026-04-18", "value": "4.06"},
    {"date": "2026-04-19", "value": "4.12"}
  ]
}"""

_FRED_MISSING_VALUE = b"""{
  "observations": [
    {"date": "2026-04-19", "value": "."},
    {"date": "2026-04-20", "value": "4.21"}
  ]
}"""

_FRED_EMPTY_LIST = b'{"observations": []}'
_FRED_NO_OBS_KEY = b'{"realtime_start": "2026-04-21"}'
_FRED_MALFORMED_JSON = b"{not-json,"
_FRED_OBS_NOT_LIST = b'{"observations": "oops"}'
_FRED_DOC_IS_LIST = b"[1, 2, 3]"
_FRED_BAD_DATE = b"""{
  "observations": [
    {"date": "", "value": "1.0"},
    {"date": "2026-04-19", "value": "5.0"}
  ]
}"""
_FRED_NUMERIC_VALUE = b"""{
  "observations": [
    {"date": "2026-04-19", "value": 4.5}
  ]
}"""
_FRED_INFINITE_VALUE = b"""{
  "observations": [
    {"date": "2026-04-19", "value": "inf"},
    {"date": "2026-04-20", "value": "1.0"}
  ]
}"""


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def test_make_fred_observations_url_default_shape() -> None:
    url = make_fred_observations_url("DGS10", "key-abc")
    assert url.startswith(f"{FRED_API_BASE}/series/observations?")
    assert "series_id=DGS10" in url
    assert "api_key=key-abc" in url
    assert "file_type=json" in url
    assert "sort_order=asc" in url


def test_make_fred_observations_url_url_encodes_reserved_chars() -> None:
    url = make_fred_observations_url("CPIAUCSL", "key+with/slashes&amp")
    # urlencode escapes +, /, &; the original token must not appear raw.
    assert "key+with/slashes&amp" not in url
    assert "key%2Bwith%2Fslashes%26amp" in url


def test_make_fred_observations_url_includes_limit_when_set() -> None:
    url = make_fred_observations_url("DGS10", "k", limit=12)
    assert "limit=12" in url


def test_make_fred_observations_url_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        make_fred_observations_url("", "k")
    with pytest.raises(ValueError):
        make_fred_observations_url("DGS10", "")
    with pytest.raises(ValueError):
        make_fred_observations_url("DGS10", "k", limit=0)
    with pytest.raises(ValueError):
        make_fred_observations_url("DGS10", "k", limit=-3)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parse_payload_emits_observations_in_order() -> None:
    obs = parse_observations_payload(
        _FRED_TWO_OBS,
        ts_ns=42,
        series_id="DGS10",
        units="Percent",
        title="10-Year Treasury",
    )
    assert len(obs) == 2
    first, second = obs
    assert first.observation_date == "2026-04-18"
    assert first.value == pytest.approx(4.06)
    assert first.series_id == "DGS10"
    assert first.units == "Percent"
    assert first.title == "10-Year Treasury"
    assert first.source == SOURCE_TAG
    assert first.ts_ns == 42
    assert first.observed_ts_ns is not None
    assert first.observed_ts_ns > 0
    assert second.observation_date == "2026-04-19"
    assert second.value == pytest.approx(4.12)
    assert second.observed_ts_ns is not None
    assert second.observed_ts_ns > first.observed_ts_ns


def test_parse_payload_string_input_decodes_utf8() -> None:
    obs = parse_observations_payload(
        _FRED_TWO_OBS.decode(),
        ts_ns=1,
        series_id="DGS10",
    )
    assert len(obs) == 2


def test_parse_payload_missing_value_becomes_none() -> None:
    obs = parse_observations_payload(
        _FRED_MISSING_VALUE,
        ts_ns=10,
        series_id="DGS10",
    )
    assert len(obs) == 2
    assert obs[0].value is None
    assert obs[0].observation_date == "2026-04-19"
    assert obs[1].value == pytest.approx(4.21)


def test_parse_payload_numeric_value_passes_through() -> None:
    obs = parse_observations_payload(
        _FRED_NUMERIC_VALUE,
        ts_ns=2,
        series_id="DGS10",
    )
    assert len(obs) == 1
    assert obs[0].value == pytest.approx(4.5)


def test_parse_payload_filters_inf_and_nan() -> None:
    obs = parse_observations_payload(
        _FRED_INFINITE_VALUE,
        ts_ns=2,
        series_id="DGS10",
    )
    assert len(obs) == 2
    assert obs[0].value is None  # inf → None
    assert obs[1].value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


def test_parse_payload_empty_returns_empty() -> None:
    assert parse_observations_payload(b"", ts_ns=1, series_id="DGS10") == ()
    assert parse_observations_payload("", ts_ns=1, series_id="DGS10") == ()


def test_parse_payload_malformed_json_returns_empty() -> None:
    assert (
        parse_observations_payload(
            _FRED_MALFORMED_JSON, ts_ns=1, series_id="DGS10"
        )
        == ()
    )


def test_parse_payload_no_obs_key_returns_empty() -> None:
    assert (
        parse_observations_payload(
            _FRED_NO_OBS_KEY, ts_ns=1, series_id="DGS10"
        )
        == ()
    )


def test_parse_payload_obs_key_not_list_returns_empty() -> None:
    assert (
        parse_observations_payload(
            _FRED_OBS_NOT_LIST, ts_ns=1, series_id="DGS10"
        )
        == ()
    )


def test_parse_payload_top_level_list_returns_empty() -> None:
    assert (
        parse_observations_payload(
            _FRED_DOC_IS_LIST, ts_ns=1, series_id="DGS10"
        )
        == ()
    )


def test_parse_payload_skips_blank_dates() -> None:
    obs = parse_observations_payload(
        _FRED_BAD_DATE, ts_ns=1, series_id="DGS10"
    )
    assert len(obs) == 1
    assert obs[0].observation_date == "2026-04-19"


def test_parse_payload_empty_obs_list_returns_empty() -> None:
    assert (
        parse_observations_payload(
            _FRED_EMPTY_LIST, ts_ns=1, series_id="DGS10"
        )
        == ()
    )


def test_parse_payload_invalid_utf8_returns_empty() -> None:
    bad = b"\xff\xfe\xfd"
    assert (
        parse_observations_payload(bad, ts_ns=1, series_id="DGS10") == ()
    )


def test_parse_payload_rejects_empty_series_id() -> None:
    with pytest.raises(ValueError):
        parse_observations_payload(_FRED_TWO_OBS, ts_ns=1, series_id="")


# ---------------------------------------------------------------------------
# INV-15 — pure projection: identical (payload, ts_ns) ⇒ identical output
# ---------------------------------------------------------------------------


def test_parse_payload_is_pure() -> None:
    a = parse_observations_payload(
        _FRED_TWO_OBS, ts_ns=99, series_id="DGS10", units="Percent"
    )
    b = parse_observations_payload(
        _FRED_TWO_OBS, ts_ns=99, series_id="DGS10", units="Percent"
    )
    assert a == b


# ---------------------------------------------------------------------------
# MacroObservation contract sanity
# ---------------------------------------------------------------------------


def test_macro_observation_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError):
        MacroObservation(
            ts_ns=1, source="", series_id="X", observation_date="2026-04-19"
        )
    with pytest.raises(ValueError):
        MacroObservation(
            ts_ns=1, source="FRED", series_id="", observation_date="2026-04-19"
        )
    with pytest.raises(ValueError):
        MacroObservation(
            ts_ns=1, source="FRED", series_id="X", observation_date=""
        )


def test_macro_observation_rejects_non_positive_observed_ts_ns() -> None:
    with pytest.raises(ValueError):
        MacroObservation(
            ts_ns=1,
            source="FRED",
            series_id="X",
            observation_date="2026-04-19",
            observed_ts_ns=0,
        )
    with pytest.raises(ValueError):
        MacroObservation(
            ts_ns=1,
            source="FRED",
            series_id="X",
            observation_date="2026-04-19",
            observed_ts_ns=-1,
        )
