"""Tests for :mod:`system_engine.streaming.nats_bus` (C-05).

Validates subject grammar, wildcard matching, in-memory pub/sub,
JetStream persistence + durable consumers (ack / nak / term),
deterministic ``run_app`` simulator with INV-15 3-run replay
equality, cross-process worker bridge, lazy seams, AST
guardrails (no forbidden imports, no runtime-tier imports, no
typed-event constructors).
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
from collections.abc import Iterable

import pytest

from system_engine.streaming.nats_bus import (
    NATS_BUS_VERSION,
    NEW_PIP_DEPENDENCIES,
    App,
    AppResult,
    DeliveredRecord,
    DurableConsumerConfig,
    InMemoryJetStream,
    InMemoryNATSClient,
    JetStreamConfig,
    NATSBusSentinel,
    NATSConfig,
    PublishRecord,
    Subject,
    SubscriptionPattern,
    bus_digest,
    deserialize_record,
    jetstream_factory,
    nats_client_factory,
    run_app,
    serialize_record,
    spawn_nats_worker,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_nats_bus_version_pinned() -> None:
    assert NATS_BUS_VERSION == 1


def test_new_pip_dependencies_pinned() -> None:
    assert NEW_PIP_DEPENDENCIES == ("nats-py",)


# ---------------------------------------------------------------------------
# Subject grammar
# ---------------------------------------------------------------------------


def test_subject_accepts_dotted_token_subjects() -> None:
    s = Subject("intel.signals.btc")
    assert s.name == "intel.signals.btc"


def test_subject_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Subject("")


def test_subject_rejects_wildcard_in_publish() -> None:
    with pytest.raises(ValueError):
        Subject("intel.*")
    with pytest.raises(ValueError):
        Subject("intel.>")


def test_subject_rejects_whitespace_in_token() -> None:
    with pytest.raises(ValueError):
        Subject("intel.sig nals")


def test_subject_rejects_empty_token() -> None:
    with pytest.raises(ValueError):
        Subject("intel..signals")


def test_subject_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        Subject(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Subscription patterns
# ---------------------------------------------------------------------------


def test_pattern_exact_match() -> None:
    p = SubscriptionPattern("intel.signals.btc")
    assert p.matches("intel.signals.btc")
    assert not p.matches("intel.signals.eth")
    assert not p.matches("intel.signals")
    assert not p.matches("intel.signals.btc.lvl1")


def test_pattern_single_token_wildcard() -> None:
    p = SubscriptionPattern("intel.*.btc")
    assert p.matches("intel.signals.btc")
    assert p.matches("intel.fills.btc")
    assert not p.matches("intel.signals.eth")
    assert not p.matches("intel.btc")
    assert not p.matches("intel.signals.btc.lvl1")


def test_pattern_rest_wildcard() -> None:
    p = SubscriptionPattern("intel.>")
    assert p.matches("intel.signals")
    assert p.matches("intel.signals.btc")
    assert p.matches("intel.signals.btc.lvl1")
    assert not p.matches("intel")
    assert not p.matches("execution.fills")


def test_pattern_rest_wildcard_only_terminal() -> None:
    with pytest.raises(ValueError):
        SubscriptionPattern("intel.>.btc")


def test_pattern_rejects_empty_token() -> None:
    with pytest.raises(ValueError):
        SubscriptionPattern("intel..btc")


def test_pattern_rejects_empty() -> None:
    with pytest.raises(ValueError):
        SubscriptionPattern("")


def test_pattern_rejects_whitespace() -> None:
    with pytest.raises(ValueError):
        SubscriptionPattern("intel.sig nals")


def test_pattern_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        SubscriptionPattern(7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_nats_config_defaults() -> None:
    c = NATSConfig()
    assert c.servers == ("nats://localhost:4222",)
    assert c.name == "dix-nats-client"


def test_nats_config_rejects_empty_servers() -> None:
    with pytest.raises(ValueError):
        NATSConfig(servers=())


def test_nats_config_rejects_non_tuple_servers() -> None:
    with pytest.raises(TypeError):
        NATSConfig(servers=["x"])  # type: ignore[arg-type]


def test_nats_config_rejects_empty_server() -> None:
    with pytest.raises(ValueError):
        NATSConfig(servers=("",))


def test_nats_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        NATSConfig(name="")


def test_nats_config_rejects_negative_reconnect() -> None:
    with pytest.raises(ValueError):
        NATSConfig(max_reconnect_attempts=-1)


def test_nats_config_rejects_nonpositive_timeouts() -> None:
    with pytest.raises(ValueError):
        NATSConfig(reconnect_time_wait_ns=0)
    with pytest.raises(ValueError):
        NATSConfig(connect_timeout_ns=0)


def test_jetstream_config_rejects_empty_subjects() -> None:
    with pytest.raises(ValueError):
        JetStreamConfig(name="X", subjects=())


def test_jetstream_config_rejects_negative_retention() -> None:
    with pytest.raises(ValueError):
        JetStreamConfig(name="X", subjects=("a",), max_messages=-1)
    with pytest.raises(ValueError):
        JetStreamConfig(name="X", subjects=("a",), max_age_ns=-1)


def test_jetstream_config_rejects_non_tuple_subjects() -> None:
    with pytest.raises(TypeError):
        JetStreamConfig(
            name="X",
            subjects=["a"],  # type: ignore[arg-type]
        )


def test_jetstream_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        JetStreamConfig(name="", subjects=("a",))


def test_durable_consumer_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        DurableConsumerConfig(durable_name="", filter_pattern="a")


def test_durable_consumer_rejects_negative_ack_wait() -> None:
    with pytest.raises(ValueError):
        DurableConsumerConfig(
            durable_name="d",
            filter_pattern="a",
            ack_wait_ns=-1,
        )


# ---------------------------------------------------------------------------
# PublishRecord validation
# ---------------------------------------------------------------------------


def test_publish_record_round_trip() -> None:
    r = PublishRecord(
        subject_name="intel.signals.btc",
        value=b"x",
        ts_ns=42,
    )
    assert r.subject_name == "intel.signals.btc"
    assert r.value == b"x"
    assert r.ts_ns == 42


def test_publish_record_rejects_wildcard_subject() -> None:
    with pytest.raises(ValueError):
        PublishRecord(
            subject_name="intel.*",
            value=b"x",
            ts_ns=0,
        )


def test_publish_record_rejects_non_bytes_value() -> None:
    with pytest.raises(TypeError):
        PublishRecord(
            subject_name="a",
            value="x",  # type: ignore[arg-type]
            ts_ns=0,
        )


def test_publish_record_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError):
        PublishRecord(subject_name="a", value=b"x", ts_ns=-1)


def test_publish_record_rejects_bad_headers() -> None:
    with pytest.raises(TypeError):
        PublishRecord(
            subject_name="a",
            value=b"x",
            ts_ns=0,
            headers=[("k", "v")],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        PublishRecord(
            subject_name="a",
            value=b"x",
            ts_ns=0,
            headers=(("k",),),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# DeliveredRecord validation
# ---------------------------------------------------------------------------


def test_delivered_record_round_trip() -> None:
    d = DeliveredRecord(
        stream_name="s",
        consumer_name="c",
        subject_name="a",
        seq=1,
        value=b"v",
        ts_ns=10,
        deliveries=1,
    )
    assert d.seq == 1
    assert d.deliveries == 1


def test_delivered_record_rejects_zero_seq() -> None:
    with pytest.raises(ValueError):
        DeliveredRecord(
            stream_name="s",
            consumer_name="c",
            subject_name="a",
            seq=0,
            value=b"v",
            ts_ns=10,
            deliveries=1,
        )


def test_delivered_record_rejects_negative_deliveries() -> None:
    with pytest.raises(ValueError):
        DeliveredRecord(
            stream_name="s",
            consumer_name="c",
            subject_name="a",
            seq=1,
            value=b"v",
            ts_ns=10,
            deliveries=-1,
        )


def test_delivered_record_rejects_empty_stream() -> None:
    with pytest.raises(ValueError):
        DeliveredRecord(
            stream_name="",
            consumer_name="c",
            subject_name="a",
            seq=1,
            value=b"v",
            ts_ns=10,
            deliveries=0,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialize_record_byte_stable() -> None:
    a = serialize_record({"b": 1, "a": 2})
    b = serialize_record({"a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1}'


def test_serialize_record_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        serialize_record([1, 2])  # type: ignore[arg-type]


def test_deserialize_record_round_trip() -> None:
    blob = serialize_record({"k": "v", "n": 7})
    assert deserialize_record(blob) == {"k": "v", "n": 7}


def test_deserialize_record_rejects_bytes_array_payload() -> None:
    with pytest.raises(TypeError):
        deserialize_record(b"[1, 2]")


def test_deserialize_record_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        deserialize_record(
            "{}"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# In-memory core pub/sub
# ---------------------------------------------------------------------------


def _record(subject: str, value: bytes, ts_ns: int = 0) -> PublishRecord:
    return PublishRecord(subject_name=subject, value=value, ts_ns=ts_ns)


def test_publish_no_subscribers_returns_zero() -> None:
    client = InMemoryNATSClient()
    n = client.publish(_record("intel.signals.btc", b"x"))
    assert n == 0


def test_publish_dispatches_to_exact_subscriber() -> None:
    client = InMemoryNATSClient()
    seen: list[PublishRecord] = []
    client.subscribe("intel.signals.btc", lambda r: seen.append(r))
    n = client.publish(_record("intel.signals.btc", b"x"))
    assert n == 1
    assert len(seen) == 1
    assert seen[0].value == b"x"


def test_publish_skips_non_matching_subscribers() -> None:
    client = InMemoryNATSClient()
    seen: list[PublishRecord] = []
    client.subscribe("intel.signals.eth", lambda r: seen.append(r))
    n = client.publish(_record("intel.signals.btc", b"x"))
    assert n == 0
    assert seen == []


def test_publish_wildcard_subscriber() -> None:
    client = InMemoryNATSClient()
    seen: list[PublishRecord] = []
    client.subscribe("intel.*.btc", lambda r: seen.append(r))
    client.publish(_record("intel.signals.btc", b"a"))
    client.publish(_record("intel.fills.btc", b"b"))
    client.publish(_record("intel.signals.eth", b"c"))
    assert [r.value for r in seen] == [b"a", b"b"]


def test_publish_rest_wildcard_subscriber() -> None:
    client = InMemoryNATSClient()
    seen: list[PublishRecord] = []
    client.subscribe("intel.>", lambda r: seen.append(r))
    client.publish(_record("intel.signals.btc", b"a"))
    client.publish(_record("intel.fills", b"b"))
    client.publish(_record("execution.fills", b"c"))
    assert [r.value for r in seen] == [b"a", b"b"]


def test_unsubscribe_removes_callback() -> None:
    client = InMemoryNATSClient()
    seen: list[PublishRecord] = []

    def cb(r: PublishRecord) -> None:
        seen.append(r)

    pat = client.subscribe("intel.signals.btc", cb)
    client.publish(_record("intel.signals.btc", b"x"))
    assert len(seen) == 1
    assert client.unsubscribe(pat, cb) is True
    client.publish(_record("intel.signals.btc", b"y"))
    assert len(seen) == 1
    assert client.unsubscribe(pat, cb) is False


def test_subscribe_rejects_non_callable() -> None:
    client = InMemoryNATSClient()
    with pytest.raises(TypeError):
        client.subscribe("intel.signals.btc", "not-callable")


def test_subscribe_rejects_bad_pattern() -> None:
    client = InMemoryNATSClient()
    with pytest.raises(TypeError):
        client.subscribe(
            7,
            lambda r: None,  # type: ignore[arg-type]
        )


def test_publish_rejects_non_publish_record() -> None:
    client = InMemoryNATSClient()
    with pytest.raises(TypeError):
        client.publish("nope")  # type: ignore[arg-type]


def test_publish_count_increments() -> None:
    client = InMemoryNATSClient()
    assert client.publish_count() == 0
    client.publish(_record("a", b""))
    client.publish(_record("a", b""))
    assert client.publish_count() == 2


# ---------------------------------------------------------------------------
# JetStream — streams, consumers, publish, retention
# ---------------------------------------------------------------------------


def _make_app() -> App:
    return (
        App()
        .with_stream(
            JetStreamConfig(
                name="INTEL",
                subjects=("intel.signals.>",),
                max_messages=1_000,
            )
        )
        .with_consumer(
            "INTEL",
            DurableConsumerConfig(
                durable_name="alpha",
                filter_pattern="intel.signals.btc",
            ),
        )
    )


def test_add_stream_registers_log() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    assert "A" in js.stream_names()
    assert js.stream_length("A") == 0


def test_add_stream_rejects_duplicate_name() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    with pytest.raises(ValueError):
        js.add_stream(JetStreamConfig(name="A", subjects=("b.>",)))


def test_add_stream_rejects_non_config() -> None:
    js = InMemoryJetStream()
    with pytest.raises(TypeError):
        js.add_stream("nope")  # type: ignore[arg-type]


def test_publish_appends_to_matching_streams() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_stream(JetStreamConfig(name="B", subjects=("b.>",)))
    accepted = js.publish(_record("a.x", b"1"))
    assert accepted == 1
    assert js.stream_length("A") == 1
    assert js.stream_length("B") == 0


def test_publish_appends_to_multiple_matching_streams() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("intel.>",)))
    js.add_stream(JetStreamConfig(name="ALL", subjects=("intel.signals.btc",)))
    accepted = js.publish(_record("intel.signals.btc", b"1"))
    assert accepted == 2
    assert js.stream_length("A") == 1
    assert js.stream_length("ALL") == 1


def test_publish_no_matching_streams_is_zero() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    accepted = js.publish(_record("b.x", b"1"))
    assert accepted == 0


def test_retention_max_messages_drops_oldest() -> None:
    js = InMemoryJetStream()
    js.add_stream(
        JetStreamConfig(
            name="A",
            subjects=("a.>",),
            max_messages=2,
        )
    )
    js.publish(_record("a.x", b"1", ts_ns=1))
    js.publish(_record("a.x", b"2", ts_ns=2))
    js.publish(_record("a.x", b"3", ts_ns=3))
    assert js.stream_length("A") == 2
    msgs = js.stream_messages("A")
    assert [m.value for m in msgs] == [b"2", b"3"]


def test_retention_max_age_drops_old() -> None:
    js = InMemoryJetStream()
    js.add_stream(
        JetStreamConfig(
            name="A",
            subjects=("a.>",),
            max_age_ns=1_000,
        )
    )
    js.publish(_record("a.x", b"1", ts_ns=0))
    js.publish(_record("a.x", b"2", ts_ns=500))
    js.publish(_record("a.x", b"3", ts_ns=2_000))
    # at ts=2000 with age 1000 cutoff, 0 and 500 should be dropped
    msgs = js.stream_messages("A")
    assert [m.value for m in msgs] == [b"3"]


# ---------------------------------------------------------------------------
# Durable consumers — fetch + ack
# ---------------------------------------------------------------------------


def test_add_consumer_requires_known_stream() -> None:
    js = InMemoryJetStream()
    with pytest.raises(KeyError):
        js.add_consumer(
            "MISSING",
            DurableConsumerConfig(durable_name="d", filter_pattern="a"),
        )


def test_add_consumer_rejects_duplicate() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    cfg = DurableConsumerConfig(durable_name="d", filter_pattern="a.>")
    js.add_consumer("A", cfg)
    with pytest.raises(ValueError):
        js.add_consumer("A", cfg)


def test_add_consumer_rejects_non_config() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    with pytest.raises(TypeError):
        js.add_consumer("A", "nope")  # type: ignore[arg-type]


def test_fetch_returns_filtered_records() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("intel.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(
            durable_name="d",
            filter_pattern="intel.signals.btc",
        ),
    )
    js.publish(_record("intel.signals.btc", b"1", ts_ns=1))
    js.publish(_record("intel.signals.eth", b"2", ts_ns=2))
    js.publish(_record("intel.signals.btc", b"3", ts_ns=3))
    out = js.fetch("A", "d", batch=10)
    assert [r.value for r in out] == [b"1", b"3"]
    assert [r.seq for r in out] == [1, 3]
    assert all(r.deliveries == 1 for r in out)


def test_fetch_advances_delivered_seq() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    js.publish(_record("a.x", b"1", ts_ns=1))
    js.publish(_record("a.x", b"2", ts_ns=2))
    first = js.fetch("A", "d", batch=1)
    second = js.fetch("A", "d", batch=1)
    third = js.fetch("A", "d", batch=1)
    assert len(first) == 1 and first[0].seq == 1
    assert len(second) == 1 and second[0].seq == 2
    assert third == ()


def test_fetch_batch_caps_results() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    for i in range(5):
        js.publish(_record("a.x", str(i).encode(), ts_ns=i))
    out = js.fetch("A", "d", batch=2)
    assert len(out) == 2


def test_fetch_rejects_nonpositive_batch() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    with pytest.raises(ValueError):
        js.fetch("A", "d", batch=0)
    with pytest.raises(ValueError):
        js.fetch("A", "d", batch=-1)


def test_fetch_rejects_unknown_consumer() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    with pytest.raises(KeyError):
        js.fetch("A", "missing", batch=1)


def test_ack_clears_pending() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    js.publish(_record("a.x", b"1", ts_ns=1))
    out = js.fetch("A", "d", batch=1)
    assert js.consumer_state("A", "d") == (1, 0, (1,))
    assert js.ack("A", "d", out[0].seq) is True
    assert js.consumer_state("A", "d") == (1, 1, ())
    assert js.ack("A", "d", out[0].seq) is False


def test_ack_advances_floor_only_when_contiguous() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    js.publish(_record("a.x", b"1", ts_ns=1))
    js.publish(_record("a.x", b"2", ts_ns=2))
    js.publish(_record("a.x", b"3", ts_ns=3))
    js.fetch("A", "d", batch=3)
    # ack out-of-order: 2 then 3 then 1
    js.ack("A", "d", 2)
    assert js.consumer_state("A", "d") == (3, 0, (1, 3))
    js.ack("A", "d", 3)
    assert js.consumer_state("A", "d") == (3, 0, (1,))
    js.ack("A", "d", 1)
    assert js.consumer_state("A", "d") == (3, 3, ())


def test_nak_redelivers_on_next_fetch() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    js.publish(_record("a.x", b"1", ts_ns=1))
    js.publish(_record("a.x", b"2", ts_ns=2))
    first = js.fetch("A", "d", batch=2)
    assert [r.seq for r in first] == [1, 2]
    assert [r.deliveries for r in first] == [1, 1]
    assert js.nak("A", "d", 1) is True
    redelivered = js.fetch("A", "d", batch=2)
    assert [r.seq for r in redelivered] == [1, 2]
    assert redelivered[0].deliveries == 2
    assert redelivered[1].deliveries == 2


def test_nak_unknown_seq_returns_false() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    assert js.nak("A", "d", 99) is False


def test_term_drops_pending_permanently() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    js.publish(_record("a.x", b"1", ts_ns=1))
    js.publish(_record("a.x", b"2", ts_ns=2))
    js.fetch("A", "d", batch=2)
    assert js.term("A", "d", 1) is True
    assert js.consumer_state("A", "d") == (2, 1, (2,))
    js.ack("A", "d", 2)
    assert js.consumer_state("A", "d") == (2, 2, ())


def test_term_unknown_seq_returns_false() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
    )
    assert js.term("A", "d", 7) is False


def test_consumer_names_sorted() -> None:
    js = InMemoryJetStream()
    js.add_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="zeta", filter_pattern="a.>"),
    )
    js.add_consumer(
        "A",
        DurableConsumerConfig(durable_name="alpha", filter_pattern="a.>"),
    )
    assert js.consumer_names("A") == ("alpha", "zeta")


# ---------------------------------------------------------------------------
# App + run_app — INV-15 replay equality
# ---------------------------------------------------------------------------


def test_app_with_stream_requires_jetstream_config() -> None:
    with pytest.raises(TypeError):
        App().with_stream("nope")  # type: ignore[arg-type]


def test_app_with_stream_rejects_duplicate() -> None:
    app = App().with_stream(JetStreamConfig(name="A", subjects=("a.>",)))
    with pytest.raises(ValueError):
        app.with_stream(JetStreamConfig(name="A", subjects=("b.>",)))


def test_app_with_consumer_requires_known_stream() -> None:
    with pytest.raises(ValueError):
        App().with_consumer(
            "MISSING",
            DurableConsumerConfig(durable_name="d", filter_pattern="a"),
        )


def test_app_with_consumer_rejects_duplicate() -> None:
    app = (
        App()
        .with_stream(JetStreamConfig(name="A", subjects=("a.>",)))
        .with_consumer(
            "A",
            DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
        )
    )
    with pytest.raises(ValueError):
        app.with_consumer(
            "A",
            DurableConsumerConfig(durable_name="d", filter_pattern="a.>"),
        )


def _inbound() -> Iterable[PublishRecord]:
    return [
        _record("intel.signals.btc", b"a", ts_ns=1),
        _record("intel.signals.eth", b"b", ts_ns=2),
        _record("intel.signals.btc", b"c", ts_ns=3),
    ]


def test_run_app_round_trip_filters_by_consumer() -> None:
    app = _make_app()
    result = run_app(app, _inbound())
    assert isinstance(result, AppResult)
    assert [r.value for r in result.records] == [b"a", b"c"]
    assert all(r.consumer_name == "alpha" for r in result.records)
    assert all(r.subject_name == "intel.signals.btc" for r in result.records)


def test_run_app_digest_byte_stable_3_run() -> None:
    app = _make_app()
    a = run_app(app, _inbound())
    b = run_app(app, _inbound())
    c = run_app(app, _inbound())
    assert a.app_digest == b.app_digest == c.app_digest
    assert len(a.app_digest) == 32


def test_run_app_digest_changes_with_payload() -> None:
    app = _make_app()
    a = run_app(app, _inbound())
    b = run_app(
        app,
        [_record("intel.signals.btc", b"different", ts_ns=1)],
    )
    assert a.app_digest != b.app_digest


def test_run_app_rejects_bad_app() -> None:
    with pytest.raises(TypeError):
        run_app("nope", [])  # type: ignore[arg-type]


def test_run_app_rejects_bad_batch() -> None:
    app = _make_app()
    with pytest.raises(ValueError):
        run_app(app, _inbound(), batch=0)
    with pytest.raises(ValueError):
        run_app(app, _inbound(), batch=-1)


def test_run_app_two_consumers_filter_independently() -> None:
    app = (
        App()
        .with_stream(JetStreamConfig(name="INTEL", subjects=("intel.signals.>",)))
        .with_consumer(
            "INTEL",
            DurableConsumerConfig(
                durable_name="btc-only",
                filter_pattern="intel.signals.btc",
            ),
        )
        .with_consumer(
            "INTEL",
            DurableConsumerConfig(
                durable_name="all",
                filter_pattern="intel.signals.>",
            ),
        )
    )
    result = run_app(app, _inbound())
    btc = [r for r in result.records if r.consumer_name == "btc-only"]
    allc = [r for r in result.records if r.consumer_name == "all"]
    assert [r.value for r in btc] == [b"a", b"c"]
    assert [r.value for r in allc] == [b"a", b"b", b"c"]


def test_bus_digest_canonical_sort() -> None:
    records = [
        DeliveredRecord(
            stream_name="S",
            consumer_name="C",
            subject_name="a",
            seq=2,
            value=b"2",
            ts_ns=20,
            deliveries=1,
        ),
        DeliveredRecord(
            stream_name="S",
            consumer_name="C",
            subject_name="a",
            seq=1,
            value=b"1",
            ts_ns=10,
            deliveries=1,
        ),
    ]
    rev = list(reversed(records))
    assert bus_digest(records) == bus_digest(rev)
    assert len(bus_digest(records)) == 32


# ---------------------------------------------------------------------------
# Cross-process worker bridge
# ---------------------------------------------------------------------------


def test_spawn_worker_processes_publish_and_terminates() -> None:
    app = _make_app()
    proc, inbound_q, outbound_q = spawn_nats_worker(app)
    try:
        for r in _inbound():
            inbound_q.put(r)
        inbound_q.put(NATSBusSentinel())
        received: list[object] = []
        while True:
            item = outbound_q.get(timeout=15)
            if isinstance(item, NATSBusSentinel):
                break
            received.append(item)
        proc.join(timeout=15)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    assert all(isinstance(r, DeliveredRecord) for r in received)
    values = sorted(r.value for r in received)  # type: ignore[union-attr]
    assert values == [b"a", b"c"]
    assert proc.exitcode == 0


def test_worker_rejects_bad_inbound_item() -> None:
    app = _make_app()
    proc, inbound_q, outbound_q = spawn_nats_worker(app)
    try:
        inbound_q.put("not-a-publish-record")
        item = outbound_q.get(timeout=15)
        assert isinstance(item, TypeError)
        inbound_q.put(NATSBusSentinel())
        tail = outbound_q.get(timeout=15)
        assert isinstance(tail, NATSBusSentinel)
        proc.join(timeout=15)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


# ---------------------------------------------------------------------------
# Lazy seams
# ---------------------------------------------------------------------------


def test_nats_client_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        nats_client_factory(NATSConfig())


def test_nats_client_factory_rejects_bad_config() -> None:
    with pytest.raises(TypeError):
        nats_client_factory(
            "nope"  # type: ignore[arg-type]
        )


def test_jetstream_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        jetstream_factory(NATSConfig())


def test_jetstream_factory_rejects_bad_config() -> None:
    with pytest.raises(TypeError):
        jetstream_factory(
            "nope"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "system_engine" / "streaming" / "nats_bus.py"
)

_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "nats",
        "nats.aio",
        "nats.js",
        "numpy",
        "torch",
        "polars",
        "requests",
        "aiokafka",
        "confluent_kafka",
        "hiredis",
        "redis",
    }
)

_FORBIDDEN_RUNTIME_ROOTS = frozenset(
    {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    }
)

_FORBIDDEN_TYPED_EVENTS = frozenset(
    {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
    }
)


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text())


def test_no_forbidden_top_level_imports() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in _FORBIDDEN_TOP_LEVEL, f"forbidden top-level import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            assert root not in _FORBIDDEN_TOP_LEVEL, (
                f"forbidden top-level import from {node.module}"
            )


def test_no_runtime_tier_imports() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in _FORBIDDEN_RUNTIME_ROOTS, (
                f"B1 violation: forbidden runtime-tier import from {node.module}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in _FORBIDDEN_RUNTIME_ROOTS, (
                    f"B1 violation: forbidden runtime-tier import {alias.name}"
                )


def test_no_typed_event_constructors() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _FORBIDDEN_TYPED_EVENTS:
                raise AssertionError(
                    f"B27/B28/INV-71 violation: typed-event constructor call {name}"
                )


def test_module_reimports_clean() -> None:
    name = "system_engine.streaming.nats_bus"
    sys.modules.pop(name, None)
    importlib.import_module(name)
