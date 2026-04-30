"""Parser-level tests for ui.feeds.bls_http (Wave-04.5 PR-3).

The parser MUST be pure (caller-supplied ``ts_ns``), MUST never raise
on malformed input, and MUST tokenise BLS period codes into a stable
ISO observation_date so the rest of the macro pipeline can treat
multi-series payloads uniformly.
"""

from __future__ import annotations

import json

import pytest

from core.contracts.macro import MacroObservation
from ui.feeds.bls_http import (
    SOURCE_TAG,
    _period_to_observation_date,
    _redact_request_body,
    make_bls_request_body,
    parse_bls_payload,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _payload(
    *,
    status: str = "REQUEST_SUCCEEDED",
    series: list[dict[str, object]] | None = None,
) -> bytes:
    doc: dict[str, object] = {
        "status": status,
        "responseTime": 200,
        "message": [],
        "Results": {"series": series if series is not None else []},
    }
    return json.dumps(doc).encode("utf-8")


def _series_block(
    series_id: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    return {"seriesID": series_id, "data": rows}


# ---------------------------------------------------------------------------
# make_bls_request_body
# ---------------------------------------------------------------------------


def test_make_body_default_shape() -> None:
    body = make_bls_request_body(
        ("CPIAUCSL", "UNRATE"),
        registration_key="abc-123",
    )
    parsed = json.loads(body)
    assert parsed["seriesid"] == ["CPIAUCSL", "UNRATE"]
    assert parsed["registrationkey"] == "abc-123"
    assert parsed["catalog"] is False
    assert parsed["calculations"] is False
    assert parsed["annualaverage"] is False


def test_make_body_dedups_preserving_first_seen_order() -> None:
    body = make_bls_request_body(
        ("CPIAUCSL", "UNRATE", "CPIAUCSL"),
        registration_key="k",
    )
    parsed = json.loads(body)
    assert parsed["seriesid"] == ["CPIAUCSL", "UNRATE"]


def test_make_body_drops_empty_series_ids() -> None:
    body = make_bls_request_body(
        ("", "CPIAUCSL", ""),
        registration_key="k",
    )
    parsed = json.loads(body)
    assert parsed["seriesid"] == ["CPIAUCSL"]


def test_make_body_year_range_round_trip() -> None:
    body = make_bls_request_body(
        ("CPIAUCSL",),
        registration_key="k",
        start_year=2020,
        end_year=2024,
    )
    parsed = json.loads(body)
    assert parsed["startyear"] == "2020"
    assert parsed["endyear"] == "2024"


def test_make_body_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="registration_key"):
        make_bls_request_body(("CPIAUCSL",), registration_key="")


def test_make_body_rejects_no_series_after_filter() -> None:
    with pytest.raises(ValueError, match="series id"):
        make_bls_request_body(("", ""), registration_key="k")


def test_make_body_rejects_partial_year_range() -> None:
    with pytest.raises(ValueError, match="start_year and end_year"):
        make_bls_request_body(
            ("CPIAUCSL",),
            registration_key="k",
            start_year=2020,
        )


def test_make_body_rejects_inverted_year_range() -> None:
    with pytest.raises(ValueError, match="end_year"):
        make_bls_request_body(
            ("CPIAUCSL",),
            registration_key="k",
            start_year=2024,
            end_year=2020,
        )


def test_make_body_rejects_pre_1900_year() -> None:
    with pytest.raises(ValueError, match=">= 1900"):
        make_bls_request_body(
            ("CPIAUCSL",),
            registration_key="k",
            start_year=1800,
            end_year=2024,
        )


def test_make_body_is_deterministic() -> None:
    a = make_bls_request_body(("CPIAUCSL",), registration_key="k")
    b = make_bls_request_body(("CPIAUCSL",), registration_key="k")
    assert a == b


# ---------------------------------------------------------------------------
# _redact_request_body
# ---------------------------------------------------------------------------


def test_redact_strips_registration_key() -> None:
    body = make_bls_request_body(("CPIAUCSL",), registration_key="secret-key")
    redacted = _redact_request_body(body)
    assert "secret-key" not in redacted
    assert "REDACTED" in redacted


def test_redact_unparseable_returns_placeholder() -> None:
    assert _redact_request_body("not json") == "<redacted-body>"


def test_redact_non_object_returns_placeholder() -> None:
    assert _redact_request_body(json.dumps([1, 2])) == "<redacted-body>"


def test_redact_passthrough_when_no_key_field() -> None:
    redacted = _redact_request_body(json.dumps({"seriesid": ["X"]}))
    assert "REDACTED" not in redacted
    assert json.loads(redacted) == {"seriesid": ["X"]}


# ---------------------------------------------------------------------------
# _period_to_observation_date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "period", "expected"),
    [
        ("2024", "M01", "2024-01-01"),
        ("2024", "M12", "2024-12-01"),
        ("2024", "M07", "2024-07-01"),
        ("2024", "Q01", "2024-03-01"),
        ("2024", "Q02", "2024-06-01"),
        ("2024", "Q03", "2024-09-01"),
        ("2024", "Q04", "2024-12-01"),
        ("2024", "Q05", "2024-12-01"),
        ("2024", "S01", "2024-06-01"),
        ("2024", "S02", "2024-12-01"),
        ("2024", "A01", "2024-12-01"),
    ],
)
def test_period_to_date_known_codes(
    year: str, period: str, expected: str
) -> None:
    assert _period_to_observation_date(year, period) == expected


@pytest.mark.parametrize(
    ("year", "period"),
    [
        ("", "M01"),
        ("2024", ""),
        ("24", "M01"),  # not 4 digits
        ("abcd", "M01"),
        ("1899", "M01"),  # pre-1900
        ("2024", "M00"),  # month out of range
        ("2024", "M13"),  # month out of range (annual avg, intentionally dropped)
        ("2024", "MXX"),  # non-numeric suffix
        ("2024", "X01"),  # unknown prefix
    ],
)
def test_period_to_date_rejects_malformed(year: str, period: str) -> None:
    assert _period_to_observation_date(year, period) is None


# ---------------------------------------------------------------------------
# parse_bls_payload — happy paths
# ---------------------------------------------------------------------------


def test_parse_emits_observations_in_order() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [
                    {"year": "2024", "period": "M01", "value": "319.086"},
                    {"year": "2024", "period": "M02", "value": "320.481"},
                ],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=42)
    assert len(obs) == 2
    assert obs[0].series_id == "CPIAUCSL"
    assert obs[0].observation_date == "2024-01-01"
    assert obs[0].value == pytest.approx(319.086)
    assert obs[1].observation_date == "2024-02-01"
    assert obs[1].value == pytest.approx(320.481)
    for o in obs:
        assert o.ts_ns == 42
        assert o.source == SOURCE_TAG


def test_parse_handles_multi_series_batch() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "319.086"}],
            ),
            _series_block(
                "UNRATE",
                [{"year": "2024", "period": "M01", "value": "3.7"}],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 2
    assert {o.series_id for o in obs} == {"CPIAUCSL", "UNRATE"}


def test_parse_string_input_decodes_utf8() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "319.086"}],
            ),
        ],
    ).decode("utf-8")
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 1


def test_parse_applies_units_and_title_overrides() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "319.086"}],
            ),
        ],
    )
    obs = parse_bls_payload(
        payload,
        ts_ns=1,
        units_overrides={"CPIAUCSL": "Index 1982-1984=100"},
        title_overrides={"CPIAUCSL": "Headline CPI"},
    )
    assert obs[0].units == "Index 1982-1984=100"
    assert obs[0].title == "Headline CPI"


def test_parse_overrides_only_apply_to_matching_series() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "319.086"}],
            ),
            _series_block(
                "UNRATE",
                [{"year": "2024", "period": "M01", "value": "3.7"}],
            ),
        ],
    )
    obs = parse_bls_payload(
        payload,
        ts_ns=1,
        units_overrides={"CPIAUCSL": "Index"},
    )
    by_id = {o.series_id: o for o in obs}
    assert by_id["CPIAUCSL"].units == "Index"
    assert by_id["UNRATE"].units == ""


# ---------------------------------------------------------------------------
# parse_bls_payload — value handling
# ---------------------------------------------------------------------------


def test_parse_missing_sentinel_becomes_none() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [
                    {"year": "2024", "period": "M01", "value": ""},
                    {"year": "2024", "period": "M02", "value": "-"},
                    {"year": "2024", "period": "M03", "value": "(NA)"},
                    {"year": "2024", "period": "M04", "value": "(R)"},
                ],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 4
    assert all(o.value is None for o in obs)


def test_parse_numeric_value_passes_through() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": 319.086}],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert obs[0].value == pytest.approx(319.086)


def test_parse_filters_inf_and_nan() -> None:
    # Build directly because json.dumps(NaN) emits 'NaN' which would be
    # rejected at the JSON layer; we exercise the float path.
    payload = json.dumps(
        {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "CPIAUCSL",
                        "data": [
                            {"year": "2024", "period": "M01", "value": "inf"},
                            {"year": "2024", "period": "M02", "value": "-inf"},
                            {"year": "2024", "period": "M03", "value": "nan"},
                        ],
                    }
                ]
            },
        }
    ).encode("utf-8")
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 3
    assert all(o.value is None for o in obs)


# ---------------------------------------------------------------------------
# parse_bls_payload — defensive paths
# ---------------------------------------------------------------------------


def test_parse_empty_returns_empty() -> None:
    assert parse_bls_payload(b"", ts_ns=1) == ()
    assert parse_bls_payload("", ts_ns=1) == ()


def test_parse_malformed_json_returns_empty() -> None:
    assert parse_bls_payload(b"{not json", ts_ns=1) == ()


def test_parse_top_level_list_returns_empty() -> None:
    assert parse_bls_payload(b"[]", ts_ns=1) == ()


def test_parse_status_failure_returns_empty() -> None:
    payload = _payload(status="REQUEST_NOT_PROCESSED")
    assert parse_bls_payload(payload, ts_ns=1) == ()


def test_parse_no_results_key_returns_empty() -> None:
    payload = json.dumps({"status": "REQUEST_SUCCEEDED"}).encode("utf-8")
    assert parse_bls_payload(payload, ts_ns=1) == ()


def test_parse_series_not_list_returns_empty() -> None:
    payload = json.dumps(
        {"status": "REQUEST_SUCCEEDED", "Results": {"series": "x"}}
    ).encode("utf-8")
    assert parse_bls_payload(payload, ts_ns=1) == ()


def test_parse_skips_non_dict_series_entry() -> None:
    payload = json.dumps(
        {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    "not-a-dict",
                    {
                        "seriesID": "CPIAUCSL",
                        "data": [
                            {"year": "2024", "period": "M01", "value": "1.0"}
                        ],
                    },
                ]
            },
        }
    ).encode("utf-8")
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 1
    assert obs[0].series_id == "CPIAUCSL"


def test_parse_skips_series_with_empty_id() -> None:
    payload = _payload(
        series=[
            _series_block(
                "",
                [{"year": "2024", "period": "M01", "value": "1.0"}],
            ),
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "1.0"}],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 1
    assert obs[0].series_id == "CPIAUCSL"


def test_parse_skips_rows_with_unparseable_period() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [
                    {"year": "2024", "period": "M99", "value": "1.0"},
                    {"year": "2024", "period": "M01", "value": "1.0"},
                ],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 1
    assert obs[0].observation_date == "2024-01-01"


def test_parse_skips_rows_missing_year_or_period() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [
                    {"period": "M01", "value": "1.0"},
                    {"year": "2024", "value": "1.0"},
                    {"year": "2024", "period": "M01", "value": "1.0"},
                ],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert len(obs) == 1


def test_parse_invalid_utf8_returns_empty() -> None:
    assert parse_bls_payload(b"\xff\xfeinvalid", ts_ns=1) == ()


def test_parse_is_pure() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "1.0"}],
            ),
        ],
    )
    a = parse_bls_payload(payload, ts_ns=1)
    b = parse_bls_payload(payload, ts_ns=1)
    assert a == b


def test_parse_observed_ts_ns_set_for_valid_dates() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "1.0"}],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert obs[0].observed_ts_ns is not None
    assert obs[0].observed_ts_ns > 0


def test_parse_returns_macro_observation_instances() -> None:
    payload = _payload(
        series=[
            _series_block(
                "CPIAUCSL",
                [{"year": "2024", "period": "M01", "value": "1.0"}],
            ),
        ],
    )
    obs = parse_bls_payload(payload, ts_ns=1)
    assert isinstance(obs[0], MacroObservation)
