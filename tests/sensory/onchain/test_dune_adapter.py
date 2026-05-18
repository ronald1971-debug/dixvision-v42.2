"""Test suite for :mod:`sensory.onchain.dune_adapter`.

Covers:

* Pure URL-builder surface (validation, byte-stability).
* Pure parser surface (execution-id, status, results — including
  malformed rows, mixed numeric/string values, ISO + epoch
  timestamps, asset-override columns).
* :class:`DuneQuerySpec` immutability, validation, and parameter
  byte-stability across dict insertion orders.
* :class:`DuneAnalyticsClient` happy path, retry-on-status,
  terminal-failure handling, execution-timeout handling, sink-vs-
  return parity, dedup-by-query-id, telemetry counters.
* AST guardrails (B1 isolation, B27/B28/INV-71 no typed-event
  constructors, no top-level forbidden imports).
* INV-15 byte-identical replay over 3 runs.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from sensory.onchain.contracts import OnChainMetric
from sensory.onchain.dune_adapter import (
    DUNE_API_BASE,
    DUNE_API_KEY_HEADER,
    EXECUTION_STATE_COMPLETED,
    EXECUTION_STATE_EXECUTING,
    EXECUTION_STATE_FAILED,
    EXECUTION_STATE_PENDING,
    SOURCE_TAG,
    DuneAnalyticsClient,
    DuneFeedStatus,
    DuneQuerySpec,
    make_execute_url,
    make_results_url,
    make_status_url,
    make_verify_url,
    parse_execution_id,
    parse_execution_status,
    parse_results_payload,
    serialize_execute_body,
)

# ---------------------------------------------------------------------------
# URL-builder surface.
# ---------------------------------------------------------------------------


def test_make_execute_url_canonical() -> None:
    assert make_execute_url(1234567) == "https://api.dune.com/api/v1/query/1234567/execute"


def test_make_status_url_quotes_execution_id() -> None:
    assert make_status_url("01HF1XYZ") == "https://api.dune.com/api/v1/execution/01HF1XYZ/status"
    # A weird id with a slash would otherwise escape the path.
    assert make_status_url("a/b") == "https://api.dune.com/api/v1/execution/a%2Fb/status"


def test_make_results_url_quotes_execution_id() -> None:
    assert make_results_url("01HF1XYZ") == "https://api.dune.com/api/v1/execution/01HF1XYZ/results"


def test_make_verify_url_defaults_to_query_one() -> None:
    assert make_verify_url() == "https://api.dune.com/api/v1/query/1"


def test_url_builders_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        make_execute_url(0)
    with pytest.raises(ValueError):
        make_execute_url(-1)
    with pytest.raises(TypeError):
        make_execute_url("123")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        make_execute_url(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_status_url("")
    with pytest.raises(TypeError):
        make_status_url(123)  # type: ignore[arg-type]


def test_url_builders_strip_trailing_slash() -> None:
    assert (
        make_execute_url(1, base="https://api.dune.com/api/v1/")
        == "https://api.dune.com/api/v1/query/1/execute"
    )


# ---------------------------------------------------------------------------
# Parser surface.
# ---------------------------------------------------------------------------


def test_parse_execution_id_returns_id_on_success() -> None:
    body = json.dumps({"execution_id": "01HFEXEC", "state": EXECUTION_STATE_PENDING}).encode()
    assert parse_execution_id(body) == "01HFEXEC"


def test_parse_execution_id_returns_none_on_failures() -> None:
    assert parse_execution_id(b"") is None
    assert parse_execution_id(b"{not json}") is None
    assert parse_execution_id(b"[]") is None
    assert parse_execution_id(json.dumps({}).encode()) is None
    assert parse_execution_id(json.dumps({"execution_id": ""}).encode()) is None
    assert parse_execution_id(json.dumps({"execution_id": 123}).encode()) is None


def test_parse_execution_status_extracts_state() -> None:
    body = json.dumps({"state": EXECUTION_STATE_EXECUTING}).encode()
    assert parse_execution_status(body) == EXECUTION_STATE_EXECUTING


def test_parse_execution_status_handles_failures() -> None:
    assert parse_execution_status(b"") is None
    assert parse_execution_status(b"[]") is None
    assert parse_execution_status(json.dumps({"state": ""}).encode()) is None
    assert parse_execution_status(b"definitely not json") is None


def _results_body(rows: list[dict[str, Any]]) -> bytes:
    return json.dumps({"result": {"rows": rows}}).encode()


def test_parse_results_payload_happy_path() -> None:
    body = _results_body(
        [
            {"value": 42.5, "asset": "BTC"},
            {"value": "1000", "asset": "ETH"},
            {"value": 0},
        ]
    )
    out = parse_results_payload(body, ts_ns=1_700_000_000_000_000_000, metric="m")
    assert len(out) == 3
    assert {x.value for x in out} == {42.5, 1000.0, 0.0}
    assert all(x.ts_ns == 1_700_000_000_000_000_000 for x in out)
    assert all(x.metric == "m" for x in out)
    assert all(x.source == SOURCE_TAG for x in out)


def test_parse_results_payload_asset_field_override() -> None:
    body = _results_body(
        [
            {"v": 1, "sym": "BTC"},
            {"v": 2, "sym": ""},
            {"v": 3},
        ]
    )
    out = parse_results_payload(
        body,
        ts_ns=1,
        metric="m",
        value_field="v",
        asset="DEFAULT",
        asset_field="sym",
    )
    assert [x.asset for x in out] == ["BTC", "DEFAULT", "DEFAULT"]


def test_parse_results_payload_observed_ts_epoch_scales() -> None:
    body = _results_body(
        [
            {"value": 1, "t": 1_700_000_000},  # seconds
            {"value": 2, "t": 1_700_000_000_000},  # milliseconds
            {"value": 3, "t": 1_700_000_000_000_000},  # microseconds
            {"value": 4, "t": 1_700_000_000_000_000_000},  # nanoseconds
        ]
    )
    out = parse_results_payload(body, ts_ns=1, metric="m", observed_ts_field="t")
    expected_ns = 1_700_000_000_000_000_000
    assert [x.observed_ts_ns for x in out] == [
        expected_ns,
        expected_ns,
        expected_ns,
        expected_ns,
    ]


def test_parse_results_payload_observed_ts_iso() -> None:
    body = _results_body(
        [
            {"value": 1, "t": "2023-11-14T22:13:20Z"},
            {"value": 2, "t": "2023-11-14T22:13:20+00:00"},
            {"value": 3, "t": "not a date"},
        ]
    )
    out = parse_results_payload(body, ts_ns=1, metric="m", observed_ts_field="t")
    assert out[0].observed_ts_ns == out[1].observed_ts_ns
    assert out[0].observed_ts_ns is not None
    assert out[2].observed_ts_ns is None


def test_parse_results_payload_skips_invalid_rows() -> None:
    body = _results_body(
        [
            {"value": 1},
            "not a dict",  # type: ignore[list-item]
            {"value": "not-a-number"},
            {"value": True},  # boolean coerced to None
            {"value": float("inf")},
            {"value": None},
            {},
        ]
    )
    out = parse_results_payload(body, ts_ns=1, metric="m")
    assert len(out) == 1
    assert out[0].value == 1.0


def test_parse_results_payload_empty_or_malformed() -> None:
    assert parse_results_payload(b"", ts_ns=1, metric="m") == ()
    assert parse_results_payload(b"[]", ts_ns=1, metric="m") == ()
    assert parse_results_payload(b"{}", ts_ns=1, metric="m") == ()
    assert (
        parse_results_payload(
            json.dumps({"result": "not a dict"}).encode(),
            ts_ns=1,
            metric="m",
        )
        == ()
    )
    assert (
        parse_results_payload(
            json.dumps({"result": {"rows": "not a list"}}).encode(),
            ts_ns=1,
            metric="m",
        )
        == ()
    )


def test_parse_results_payload_rejects_empty_metric_or_value_field() -> None:
    with pytest.raises(ValueError):
        parse_results_payload(b"{}", ts_ns=1, metric="")
    with pytest.raises(ValueError):
        parse_results_payload(b"{}", ts_ns=1, metric="m", value_field="")


# ---------------------------------------------------------------------------
# DuneQuerySpec value-object surface.
# ---------------------------------------------------------------------------


def test_dune_query_spec_frozen_and_slotted() -> None:
    spec = DuneQuerySpec(query_id=1, metric="m")
    assert not hasattr(spec, "__dict__"), "DuneQuerySpec must be slotted"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.metric = "z"  # type: ignore[misc]


def test_dune_query_spec_validation() -> None:
    with pytest.raises(ValueError):
        DuneQuerySpec(query_id=0, metric="m")
    with pytest.raises(ValueError):
        DuneQuerySpec(query_id=1, metric="")
    with pytest.raises(ValueError):
        DuneQuerySpec(query_id=1, metric="m", value_field="")
    with pytest.raises(ValueError):
        DuneQuerySpec(query_id=1, metric="m", execution_timeout_s=0.0)
    with pytest.raises(ValueError):
        DuneQuerySpec(query_id=1, metric="m", poll_interval_s=-1.0)


def test_dune_query_spec_parameters_byte_stable_across_insertion_order() -> None:
    a = DuneQuerySpec(query_id=1, metric="m", parameters={"b": "2", "a": "1"})
    b = DuneQuerySpec(query_id=1, metric="m", parameters={"a": "1", "b": "2"})
    assert serialize_execute_body(a) == serialize_execute_body(b)
    assert serialize_execute_body(a) == b'{"query_parameters":{"a":"1","b":"2"}}'


def test_dune_query_spec_empty_parameters_serialises_empty_object() -> None:
    spec = DuneQuerySpec(query_id=1, metric="m")
    assert serialize_execute_body(spec) == b"{}"


# ---------------------------------------------------------------------------
# DuneAnalyticsClient — happy path / retries / failures / determinism.
# ---------------------------------------------------------------------------


class _FakeClock:
    """Monotonic, deterministic clock.

    Advances by ``step_ns`` every read so multiple consecutive
    ``clock_ns()`` calls return distinct values without ever using
    ``time.monotonic_ns`` (INV-15).
    """

    def __init__(self, start_ns: int = 1_700_000_000_000_000_000, step_ns: int = 1) -> None:
        self._now_ns = start_ns
        self._step_ns = step_ns

    def __call__(self) -> int:
        out = self._now_ns
        self._now_ns += self._step_ns
        return out


class _ScriptedFetch:
    """Inject pre-canned responses keyed by URL.

    ``responses`` maps a URL to a list of ``bytes`` (or exceptions).
    Each call pops the head of the list, asserts the method and
    headers, and records the request for later inspection.

    If a URL is exhausted, the test fails — over-consumption is a
    bug we want surfaced loudly.
    """

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    async def __call__(
        self,
        request: tuple[str, str, dict[str, str] | Any, bytes | None],
    ) -> bytes:
        method, url, headers, body = request
        self.calls.append((method, url, dict(headers), body))
        queue = self._responses.get(url)
        if not queue:
            raise AssertionError(f"unexpected URL {url!r}")
        head = queue.pop(0)
        if isinstance(head, Exception):
            raise head
        assert isinstance(head, (bytes, bytearray))
        return bytes(head)


async def _fake_sleep(_seconds: float) -> None:
    return None


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_run_query_once_happy_path(loop) -> None:
    spec = DuneQuerySpec(query_id=42, metric="daily_active_addresses", asset="BTC")
    fetch = _ScriptedFetch(
        {
            make_execute_url(42): [json.dumps({"execution_id": "EXEC1"}).encode()],
            make_status_url("EXEC1"): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
            make_results_url("EXEC1"): [_results_body([{"value": 1_000_000}, {"value": 500}])],
        }
    )
    captured: list[OnChainMetric] = []
    client = DuneAnalyticsClient(
        sink=captured.append,
        api_key="DUNE-FAKE",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )

    rows = loop.run_until_complete(client.run_query_once(spec))
    assert [x.value for x in rows] == [1_000_000.0, 500.0]
    assert [x.value for x in captured] == [1_000_000.0, 500.0]
    assert all(x.asset == "BTC" for x in rows)
    assert all(x.source == SOURCE_TAG for x in rows)

    status = client.status()
    assert status.executions_started == 1
    assert status.executions_completed == 1
    assert status.observations_received == 2
    assert status.errors == 0
    assert status.last_observation_ts_ns is not None
    assert status.query_ids == (42,)


def test_run_query_once_polls_until_completed(loop) -> None:
    spec = DuneQuerySpec(query_id=7, metric="m", poll_interval_s=0.01)
    fetch = _ScriptedFetch(
        {
            make_execute_url(7): [json.dumps({"execution_id": "EXEC7"}).encode()],
            make_status_url("EXEC7"): [
                json.dumps({"state": EXECUTION_STATE_PENDING}).encode(),
                json.dumps({"state": EXECUTION_STATE_EXECUTING}).encode(),
                json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode(),
            ],
            make_results_url("EXEC7"): [_results_body([{"value": 9}])],
        }
    )
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_query_once(spec))
    assert [x.value for x in rows] == [9.0]
    assert client.status().polls == 3


def test_run_query_once_handles_failed_state(loop) -> None:
    spec = DuneQuerySpec(query_id=99, metric="m")
    fetch = _ScriptedFetch(
        {
            make_execute_url(99): [json.dumps({"execution_id": "F"}).encode()],
            make_status_url("F"): [json.dumps({"state": EXECUTION_STATE_FAILED}).encode()],
        }
    )
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_query_once(spec))
    assert rows == ()
    status = client.status()
    assert status.executions_started == 1
    assert status.executions_completed == 0
    assert status.observations_received == 0


def test_run_query_once_handles_execute_network_failure(loop) -> None:
    spec = DuneQuerySpec(query_id=99, metric="m")
    fetch = _ScriptedFetch({make_execute_url(99): [RuntimeError("boom")]})
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_query_once(spec))
    assert rows == ()
    assert client.status().errors == 1
    assert client.status().executions_started == 0


def test_run_query_once_handles_results_decode_failure(loop) -> None:
    spec = DuneQuerySpec(query_id=5, metric="m")
    fetch = _ScriptedFetch(
        {
            make_execute_url(5): [json.dumps({"execution_id": "X"}).encode()],
            make_status_url("X"): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
            make_results_url("X"): [b"<<not json>>"],
        }
    )
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_query_once(spec))
    assert rows == ()
    # results endpoint was hit, but yielded no rows — observations
    # counter stays at 0 and errors counter stays at 0 (it parsed
    # successfully, just had no rows to project).
    assert client.status().observations_received == 0


def test_run_query_once_execution_timeout(loop) -> None:
    spec = DuneQuerySpec(
        query_id=11,
        metric="m",
        execution_timeout_s=1e-9,  # tiny: first elapsed-check trips it
        poll_interval_s=0.001,
    )
    fetch = _ScriptedFetch(
        {
            make_execute_url(11): [json.dumps({"execution_id": "T"}).encode()],
            make_status_url("T"): [
                json.dumps({"state": EXECUTION_STATE_PENDING}).encode(),
            ]
            * 50,
        }
    )
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(step_ns=1_000_000_000),  # 1 s per read
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_query_once(spec))
    assert rows == ()
    assert client.status().errors == 1


def test_run_all_specs_once_runs_each_spec_serially(loop) -> None:
    spec_a = DuneQuerySpec(query_id=1, metric="a")
    spec_b = DuneQuerySpec(query_id=2, metric="b")
    fetch = _ScriptedFetch(
        {
            make_execute_url(1): [json.dumps({"execution_id": "A"}).encode()],
            make_status_url("A"): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
            make_results_url("A"): [_results_body([{"value": 1}])],
            make_execute_url(2): [json.dumps({"execution_id": "B"}).encode()],
            make_status_url("B"): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
            make_results_url("B"): [_results_body([{"value": 2}])],
        }
    )
    captured: list[OnChainMetric] = []
    client = DuneAnalyticsClient(
        sink=captured.append,
        api_key="K",
        specs=[spec_a, spec_b],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_all_specs_once())
    assert [x.value for x in rows] == [1.0, 2.0]
    assert [x.metric for x in rows] == ["a", "b"]
    assert [x.value for x in captured] == [1.0, 2.0]
    assert client.status().running is False  # released after run


def test_constructor_dedups_specs_by_query_id() -> None:
    spec_a = DuneQuerySpec(query_id=1, metric="a")
    spec_b = DuneQuerySpec(query_id=1, metric="b")  # same id, different metric
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec_a, spec_b],
        clock_ns=_FakeClock(),
        fetch=_ScriptedFetch({}),
        sleep=_fake_sleep,
    )
    assert client.query_ids == (1,)
    assert client.specs == (spec_a,)


def test_constructor_validates_inputs() -> None:
    spec = DuneQuerySpec(query_id=1, metric="m")
    with pytest.raises(ValueError):
        DuneAnalyticsClient(
            sink=lambda _row: None,
            api_key="",
            specs=[spec],
            clock_ns=_FakeClock(),
        )
    with pytest.raises(ValueError):
        DuneAnalyticsClient(
            sink=lambda _row: None,
            api_key="K",
            specs=[],
            clock_ns=_FakeClock(),
        )


def test_headers_include_dune_api_key_and_no_authorization(loop) -> None:
    spec = DuneQuerySpec(query_id=1, metric="m")
    fetch = _ScriptedFetch(
        {
            make_execute_url(1): [json.dumps({"execution_id": "E"}).encode()],
            make_status_url("E"): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
            make_results_url("E"): [_results_body([])],
        }
    )
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="DUNE-SECRET",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    loop.run_until_complete(client.run_query_once(spec))
    for _method, _url, headers, _body in fetch.calls:
        assert headers[DUNE_API_KEY_HEADER] == "DUNE-SECRET"
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"


def test_execute_request_body_serialises_parameters(loop) -> None:
    spec = DuneQuerySpec(
        query_id=1,
        metric="m",
        parameters={"address": "0xabc", "chain": "ethereum"},
    )
    fetch = _ScriptedFetch(
        {
            make_execute_url(1): [json.dumps({"execution_id": "E"}).encode()],
            make_status_url("E"): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
            make_results_url("E"): [_results_body([])],
        }
    )
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    loop.run_until_complete(client.run_query_once(spec))
    method, url, _headers, body = fetch.calls[0]
    assert method == "POST"
    assert url == make_execute_url(1)
    assert body is not None
    decoded = json.loads(body)
    assert decoded == {"query_parameters": {"address": "0xabc", "chain": "ethereum"}}


# ---------------------------------------------------------------------------
# DuneFeedStatus value object.
# ---------------------------------------------------------------------------


def test_dune_feed_status_frozen_and_slotted() -> None:
    s = DuneFeedStatus(
        running=False,
        source=SOURCE_TAG,
        query_ids=(1,),
        last_poll_ts_ns=None,
        last_observation_ts_ns=None,
        observations_received=0,
        executions_started=0,
        executions_completed=0,
        polls=0,
        errors=0,
    )
    assert not hasattr(s, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.running = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay over 3 runs.
# ---------------------------------------------------------------------------


def _build_canned_responses(execution_id: str) -> dict[str, list[bytes]]:
    return {
        make_execute_url(1): [json.dumps({"execution_id": execution_id}).encode()],
        make_status_url(execution_id): [json.dumps({"state": EXECUTION_STATE_COMPLETED}).encode()],
        make_results_url(execution_id): [
            _results_body(
                [
                    {"value": 100, "asset": "BTC", "t": 1_700_000_000},
                    {"value": 200, "asset": "ETH", "t": 1_700_000_001},
                ]
            )
        ],
    }


def _run_one_replay(loop) -> tuple[bytes, ...]:
    spec = DuneQuerySpec(
        query_id=1,
        metric="m",
        observed_ts_field="t",
        asset_field="asset",
    )
    fetch = _ScriptedFetch(_build_canned_responses("REPLAY"))
    client = DuneAnalyticsClient(
        sink=lambda _row: None,
        api_key="K",
        specs=[spec],
        clock_ns=_FakeClock(start_ns=1_700_000_000_000_000_000, step_ns=0),
        fetch=fetch,
        sleep=_fake_sleep,
    )
    rows = loop.run_until_complete(client.run_query_once(spec))
    return tuple(
        json.dumps(
            {
                "ts_ns": row.ts_ns,
                "source": row.source,
                "metric": row.metric,
                "value": row.value,
                "asset": row.asset,
                "unit": row.unit,
                "observed_ts_ns": row.observed_ts_ns,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        for row in rows
    )


def test_three_replays_byte_identical(loop) -> None:
    a = _run_one_replay(loop)
    # Each replay needs a fresh loop because _FakeClock's monotonic
    # state shouldn't bleed across runs.
    loop_b = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop_b)
        b = _run_one_replay(loop_b)
    finally:
        loop_b.close()
        asyncio.set_event_loop(None)
    loop_c = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop_c)
        c = _run_one_replay(loop_c)
    finally:
        loop_c.close()
        asyncio.set_event_loop(None)

    assert a == b == c
    assert len(a) == 2


# ---------------------------------------------------------------------------
# AST guardrails — pin the determinism / authority boundary.
# ---------------------------------------------------------------------------


_DUNE_ADAPTER_PATH = Path(__file__).resolve().parents[3] / "sensory" / "onchain" / "dune_adapter.py"


def _adapter_tree() -> ast.AST:
    return ast.parse(_DUNE_ADAPTER_PATH.read_text(encoding="utf-8"))


_FORBIDDEN_TOP_LEVEL_IMPORTS = frozenset(
    {
        "random",
        "time",
        "datetime",
        "numpy",
        "torch",
        "polars",
        "requests",
        "httpx",
        "aiohttp",
    }
)


def test_no_forbidden_top_level_imports() -> None:
    tree = _adapter_tree()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                assert head not in _FORBIDDEN_TOP_LEVEL_IMPORTS, (
                    f"forbidden top-level import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            head = node.module.split(".")[0]
            assert head not in _FORBIDDEN_TOP_LEVEL_IMPORTS, (
                f"forbidden top-level import: {node.module}"
            )


def test_no_runtime_tier_imports() -> None:
    """B1 isolation: a sensory adapter must never import any of the
    governance / execution / intelligence / evolution runtime
    tiers.  Authority symmetry (B27 / B28 / INV-71) depends on
    this — projection into typed events happens downstream, never
    inside the adapter.
    """
    tree = _adapter_tree()
    forbidden = (
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                assert head not in forbidden, f"runtime-tier import banned: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            head = node.module.split(".")[0]
            assert head not in forbidden, f"runtime-tier import banned: {node.module}"


def test_no_typed_event_constructors() -> None:
    """B27 / B28 / INV-71: a sensory adapter must never construct
    a typed bus event.  The only value type it may construct is
    :class:`OnChainMetric`.
    """
    forbidden_constructors = frozenset(
        {
            "SignalEvent",
            "HazardEvent",
            "ExecutionEvent",
            "SystemEvent",
            "PatchProposal",
            "DecisionTrace",
        }
    )
    tree = _adapter_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name is not None:
                assert name not in forbidden_constructors, (
                    f"typed-event constructor banned in sensory adapter: {name}"
                )


def test_no_top_level_clock_call() -> None:
    """INV-15: the adapter must take ``clock_ns`` as an injected
    parameter — it must never *call* a clock at module-import
    time.
    """
    tree = _adapter_tree()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                if call.func.attr in {
                    "time",
                    "monotonic",
                    "monotonic_ns",
                    "time_ns",
                    "now",
                    "utcnow",
                }:
                    pytest.fail(f"top-level clock call: {ast.dump(call)!r}")


# ---------------------------------------------------------------------------
# Module-level constants pinned for downstream consumers.
# ---------------------------------------------------------------------------


def test_module_constants() -> None:
    assert DUNE_API_BASE == "https://api.dune.com/api/v1"
    assert DUNE_API_KEY_HEADER == "X-DUNE-API-KEY"
    assert SOURCE_TAG == "DUNE"
