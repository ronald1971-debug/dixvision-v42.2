"""Comprehensive test suite for C-06 pulsar-client → pulsar_bus.

Pins every surface and every invariant the module declares:

* Module constants (version, declared pip deps).
* Qualified-topic grammar (``persistent://tenant/namespace/topic``).
* Value-objects (Topic / Subscription / MessageId / ProducerRecord /
  Message / PulsarConfig) — validation + frozen + slotted + ordering.
* Partition assignment (BLAKE2b-8 mod num_partitions).
* Reference serialize / deserialize byte-stability.
* InMemoryPulsarBroker append + per-subscription fetch + ack flavours.
* SubscriptionType semantics — EXCLUSIVE / SHARED / FAILOVER /
  KEY_SHARED routing oracle + cumulative-ack policy.
* InMemoryProducer round-robin + key-shared partitioning.
* InMemoryConsumer attach + ack / acknowledge_cumulative /
  negative_acknowledge + redelivery counter.
* App builder + duplicate guards + subscription-references-known-topic
  validation.
* :func:`run_app` deterministic simulator + INV-15 3-run BLAKE2b-16
  byte-identical replay equality.
* Cross-process :func:`spawn_pulsar_worker` bridge.
* Lazy seam :func:`pulsar_client_factory`.
* AST guardrails — no forbidden top-level imports, no typed-event
  constructors, no runtime-tier imports (B1).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
from dataclasses import FrozenInstanceError, fields

import pytest

import system_engine.streaming.pulsar_bus as pb
from system_engine.streaming.pulsar_bus import (
    NEW_PIP_DEPENDENCIES,
    PULSAR_BUS_VERSION,
    App,
    AppResult,
    DeliveredRecord,
    InMemoryConsumer,
    InMemoryProducer,
    InMemoryPulsarBroker,
    Message,
    MessageId,
    ProducerRecord,
    PulsarBusSentinel,
    PulsarConfig,
    Subscription,
    SubscriptionType,
    Topic,
    bus_digest,
    deserialize_record,
    parse_qualified_topic,
    partition_for_key,
    pulsar_client_factory,
    qualified_topic,
    run_app,
    serialize_record,
    spawn_pulsar_worker,
)

T_ALPHA = "persistent://t1/ns1/alpha"
T_BETA = "persistent://t1/ns1/beta"
T_OTHER_TENANT = "persistent://t2/ns1/alpha"


# ---------------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------------


def test_module_version_is_int() -> None:
    assert isinstance(PULSAR_BUS_VERSION, int)
    assert PULSAR_BUS_VERSION >= 1


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("pulsar-client",)


# ---------------------------------------------------------------------------
# parse_qualified_topic / qualified_topic.
# ---------------------------------------------------------------------------


def test_parse_qualified_topic_ok() -> None:
    assert parse_qualified_topic(T_ALPHA) == ("persistent", "t1", "ns1", "alpha")


def test_parse_qualified_topic_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        parse_qualified_topic(123)  # type: ignore[arg-type]


def test_parse_qualified_topic_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("")


def test_parse_qualified_topic_rejects_missing_scheme() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("t1/ns1/alpha")


def test_parse_qualified_topic_rejects_non_persistent_scheme() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("non-persistent://t1/ns1/alpha")


def test_parse_qualified_topic_rejects_too_few_parts() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("persistent://t1/alpha")


def test_parse_qualified_topic_rejects_too_many_parts() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("persistent://t1/ns1/sub/alpha")


def test_parse_qualified_topic_rejects_empty_tenant() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("persistent:///ns1/alpha")


def test_parse_qualified_topic_rejects_whitespace_component() -> None:
    with pytest.raises(ValueError):
        parse_qualified_topic("persistent://t 1/ns1/alpha")


def test_qualified_topic_builder_ok() -> None:
    assert qualified_topic("t1", "ns1", "alpha") == T_ALPHA


def test_qualified_topic_builder_rejects_invalid_component() -> None:
    with pytest.raises(ValueError):
        qualified_topic("t1", "ns1", "")


# ---------------------------------------------------------------------------
# SubscriptionType enum.
# ---------------------------------------------------------------------------


def test_subscription_type_values() -> None:
    assert {t.value for t in SubscriptionType} == {
        "exclusive",
        "shared",
        "failover",
        "key_shared",
    }


# ---------------------------------------------------------------------------
# Topic.
# ---------------------------------------------------------------------------


def test_topic_ok() -> None:
    t = Topic(T_ALPHA, num_partitions=3)
    assert t.tenant == "t1"
    assert t.namespace == "ns1"
    assert t.short_name == "alpha"
    assert t.num_partitions == 3


def test_topic_rejects_unqualified() -> None:
    with pytest.raises(ValueError):
        Topic("alpha")


def test_topic_rejects_zero_partitions() -> None:
    with pytest.raises(ValueError):
        Topic(T_ALPHA, num_partitions=0)


def test_topic_rejects_bool_partitions() -> None:
    with pytest.raises(TypeError):
        Topic(T_ALPHA, num_partitions=True)  # type: ignore[arg-type]


def test_topic_rejects_zero_retention() -> None:
    with pytest.raises(ValueError):
        Topic(T_ALPHA, retention_ns=0)


def test_topic_is_frozen_and_slotted() -> None:
    t = Topic(T_ALPHA)
    with pytest.raises(FrozenInstanceError):
        t.num_partitions = 9  # type: ignore[misc]
    assert not hasattr(t, "__dict__")


# ---------------------------------------------------------------------------
# Subscription.
# ---------------------------------------------------------------------------


def test_subscription_default_exclusive() -> None:
    s = Subscription("sub1", T_ALPHA)
    assert s.type == SubscriptionType.EXCLUSIVE
    assert s.initial_position_earliest is True


def test_subscription_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Subscription("", T_ALPHA)


def test_subscription_rejects_bad_topic() -> None:
    with pytest.raises(ValueError):
        Subscription("sub1", "not-qualified")


def test_subscription_rejects_non_enum_type() -> None:
    with pytest.raises(TypeError):
        Subscription("sub1", T_ALPHA, type="shared")  # type: ignore[arg-type]


def test_subscription_is_frozen() -> None:
    s = Subscription("sub1", T_ALPHA)
    with pytest.raises(FrozenInstanceError):
        s.name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MessageId.
# ---------------------------------------------------------------------------


def test_message_id_ok() -> None:
    m = MessageId(0, 5, 1, 0)
    assert m.entry_id == 5
    assert m.partition == 1


def test_message_id_rejects_negative() -> None:
    with pytest.raises(ValueError):
        MessageId(0, -1, 0)


def test_message_id_rejects_bool() -> None:
    with pytest.raises(TypeError):
        MessageId(0, True, 0)  # type: ignore[arg-type]


def test_message_id_orderable() -> None:
    a = MessageId(0, 0, 0)
    b = MessageId(0, 1, 0)
    assert a < b
    c = MessageId(0, 0, 1)
    assert a < c


# ---------------------------------------------------------------------------
# ProducerRecord.
# ---------------------------------------------------------------------------


def test_producer_record_ok() -> None:
    r = ProducerRecord(T_ALPHA, b"payload", key=b"k", ts_ns=10)
    assert r.value == b"payload"
    assert r.event_time_ns == 0


def test_producer_record_rejects_non_bytes_value() -> None:
    with pytest.raises(TypeError):
        ProducerRecord(T_ALPHA, "str")  # type: ignore[arg-type]


def test_producer_record_rejects_non_bytes_key() -> None:
    with pytest.raises(TypeError):
        ProducerRecord(T_ALPHA, b"v", key="str")  # type: ignore[arg-type]


def test_producer_record_rejects_negative_ts() -> None:
    with pytest.raises(ValueError):
        ProducerRecord(T_ALPHA, b"v", ts_ns=-1)


def test_producer_record_rejects_bad_properties_entry() -> None:
    with pytest.raises(TypeError):
        ProducerRecord(
            T_ALPHA,
            b"v",
            properties=(("k", b"bad"),),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# PulsarConfig.
# ---------------------------------------------------------------------------


def test_pulsar_config_defaults() -> None:
    c = PulsarConfig()
    assert c.service_url == "pulsar://localhost:6650"
    assert c.operation_timeout_ns > 0


def test_pulsar_config_rejects_empty_service_url() -> None:
    with pytest.raises(TypeError):
        PulsarConfig(service_url="")


def test_pulsar_config_rejects_zero_timeout() -> None:
    with pytest.raises(ValueError):
        PulsarConfig(operation_timeout_ns=0)


# ---------------------------------------------------------------------------
# partition_for_key.
# ---------------------------------------------------------------------------


def test_partition_for_key_none_is_zero() -> None:
    assert partition_for_key(None, 8) == 0


def test_partition_for_key_deterministic() -> None:
    a = partition_for_key(b"hello", 16)
    b = partition_for_key(b"hello", 16)
    assert a == b
    assert 0 <= a < 16


def test_partition_for_key_rejects_zero_partitions() -> None:
    with pytest.raises(ValueError):
        partition_for_key(b"x", 0)


def test_partition_for_key_rejects_str_key() -> None:
    with pytest.raises(TypeError):
        partition_for_key("x", 4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Serialize / deserialize.
# ---------------------------------------------------------------------------


def test_serialize_roundtrip() -> None:
    payload = {"b": 2, "a": 1}
    blob = serialize_record(payload)
    assert deserialize_record(blob) == payload


def test_serialize_byte_stable_under_reorder() -> None:
    a = serialize_record({"a": 1, "b": 2})
    b = serialize_record({"b": 2, "a": 1})
    assert a == b


def test_serialize_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        serialize_record([("a", 1)])  # type: ignore[arg-type]


def test_deserialize_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        deserialize_record("{}")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InMemoryPulsarBroker basic append / fetch.
# ---------------------------------------------------------------------------


def _broker(topics: tuple[Topic, ...] = ()) -> InMemoryPulsarBroker:
    return InMemoryPulsarBroker(topics=topics or (Topic(T_ALPHA, num_partitions=4),))


def test_broker_rejects_empty_topics() -> None:
    with pytest.raises(ValueError):
        InMemoryPulsarBroker(topics=())


def test_broker_rejects_duplicate_topic_names() -> None:
    with pytest.raises(ValueError):
        InMemoryPulsarBroker(topics=(Topic(T_ALPHA), Topic(T_ALPHA)))


def test_broker_append_assigns_entry_id_zero_first() -> None:
    b = _broker()
    msg = b.append(ProducerRecord(T_ALPHA, b"v", key=b"k", ts_ns=1))
    assert msg.message_id.entry_id == 0


def test_broker_append_increments_per_partition() -> None:
    b = _broker()
    m0 = b.append(ProducerRecord(T_ALPHA, b"v0", key=b"k", ts_ns=1))
    m1 = b.append(ProducerRecord(T_ALPHA, b"v1", key=b"k", ts_ns=2))
    assert m0.message_id.partition == m1.message_id.partition
    assert m1.message_id.entry_id == m0.message_id.entry_id + 1


def test_broker_append_rejects_unknown_topic() -> None:
    b = _broker()
    with pytest.raises(KeyError):
        b.append(ProducerRecord(T_BETA, b"v"))


def test_broker_append_at_chosen_partition() -> None:
    b = _broker()
    msg = b.append_at(T_ALPHA, 2, b"x", key=None, ts_ns=0, event_time_ns=0, properties=())
    assert msg.message_id.partition == 2


def test_broker_append_at_rejects_out_of_range() -> None:
    b = _broker()
    with pytest.raises(ValueError):
        b.append_at(T_ALPHA, 99, b"x", key=None, ts_ns=0, event_time_ns=0, properties=())


def test_broker_event_time_defaults_to_ts_ns() -> None:
    b = _broker()
    msg = b.append(ProducerRecord(T_ALPHA, b"v", ts_ns=42))
    assert msg.event_time_ns == 42


# ---------------------------------------------------------------------------
# attach_subscription + exclusive semantics.
# ---------------------------------------------------------------------------


def test_attach_subscription_exclusive_rejects_second_consumer() -> None:
    b = _broker()
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    b.attach_subscription(sub, "c1")
    with pytest.raises(ValueError):
        b.attach_subscription(sub, "c2")


def test_attach_subscription_shared_accumulates_consumers() -> None:
    b = _broker()
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    s1 = b.attach_subscription(sub, "c1")
    s2 = b.attach_subscription(sub, "c2")
    assert s1 is s2
    assert s2.consumers == ("c1", "c2")


def test_attach_subscription_is_idempotent_per_consumer() -> None:
    b = _broker()
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    state = b.attach_subscription(sub, "c1")
    assert state.consumers == ("c1",)


def test_attach_subscription_type_mismatch_raises() -> None:
    b = _broker()
    s_excl = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    s_shared = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(s_excl, "c1")
    with pytest.raises(ValueError):
        b.attach_subscription(s_shared, "c2")


def test_attach_subscription_latest_position_skips_existing() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    # publish before subscribing
    b.append(ProducerRecord(T_ALPHA, b"old", ts_ns=1))
    sub = Subscription(
        "s1",
        T_ALPHA,
        type=SubscriptionType.EXCLUSIVE,
        initial_position_earliest=False,
    )
    b.attach_subscription(sub, "c1")
    batch = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    assert batch == ()


def test_detach_subscription_removes_consumer() -> None:
    b = _broker()
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.attach_subscription(sub, "c2")
    b.detach_subscription(T_ALPHA, "s1", "c1")
    assert b.subscription_state(T_ALPHA, "s1").consumers == ("c2",)


def test_detach_last_consumer_keeps_cursor() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    b.attach_subscription(sub, "c1")
    b.append(ProducerRecord(T_ALPHA, b"x"))
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    b.acknowledge(T_ALPHA, "s1", msgs[0].message_id, cumulative=True)
    b.detach_subscription(T_ALPHA, "s1", "c1")
    # re-attach picks up where left off
    b.append(ProducerRecord(T_ALPHA, b"y"))
    b.attach_subscription(sub, "c2")
    msgs2 = b.fetch_for(T_ALPHA, "s1", "c2", max_records=10)
    assert len(msgs2) == 1
    assert msgs2[0].value == b"y"


# ---------------------------------------------------------------------------
# Subscription-type routing oracle.
# ---------------------------------------------------------------------------


def test_routing_exclusive_only_owner() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    b.attach_subscription(sub, "alice")
    assert b.consumer_owns(T_ALPHA, "s1", "alice", 0, 0, None) is True
    assert b.consumer_owns(T_ALPHA, "s1", "bob", 0, 0, None) is False


def test_routing_failover_picks_smallest_name() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.FAILOVER)
    b.attach_subscription(sub, "bob")
    b.attach_subscription(sub, "alice")
    # roster sorted -> "alice" wins
    assert b.consumer_owns(T_ALPHA, "s1", "alice", 0, 0, None) is True
    assert b.consumer_owns(T_ALPHA, "s1", "bob", 0, 0, None) is False


def test_routing_shared_round_robins() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.attach_subscription(sub, "c2")
    # entry_id 0 -> consumers[0]="c1"; entry_id 1 -> consumers[1]="c2"
    assert b.consumer_owns(T_ALPHA, "s1", "c1", 0, 0, None) is True
    assert b.consumer_owns(T_ALPHA, "s1", "c2", 0, 1, None) is True


def test_routing_key_shared_same_key_same_consumer() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.KEY_SHARED)
    b.attach_subscription(sub, "c1")
    b.attach_subscription(sub, "c2")
    owners1 = [c for c in ("c1", "c2") if b.consumer_owns(T_ALPHA, "s1", c, 0, 0, b"key-a")]
    owners2 = [c for c in ("c1", "c2") if b.consumer_owns(T_ALPHA, "s1", c, 0, 1, b"key-a")]
    assert len(owners1) == 1
    assert owners1 == owners2


def test_routing_key_shared_none_key_goes_to_smallest() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.KEY_SHARED)
    b.attach_subscription(sub, "alice")
    b.attach_subscription(sub, "bob")
    assert b.consumer_owns(T_ALPHA, "s1", "alice", 0, 0, None) is True
    assert b.consumer_owns(T_ALPHA, "s1", "bob", 0, 0, None) is False


def test_routing_returns_false_with_empty_roster() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.detach_subscription(T_ALPHA, "s1", "c1")
    assert b.consumer_owns(T_ALPHA, "s1", "c1", 0, 0, None) is False


# ---------------------------------------------------------------------------
# fetch_for — partition-ordered delivery.
# ---------------------------------------------------------------------------


def test_fetch_for_exclusive_drains_in_partition_order() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=2),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    b.attach_subscription(sub, "c1")
    b.append_at(T_ALPHA, 1, b"p1-0", key=None, ts_ns=0, event_time_ns=0, properties=())
    b.append_at(T_ALPHA, 0, b"p0-0", key=None, ts_ns=0, event_time_ns=0, properties=())
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    # partition 0 drained first then partition 1
    assert [m.message_id.partition for m in msgs] == [0, 1]


def test_fetch_for_shared_returns_only_owned_messages() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.attach_subscription(sub, "c2")
    for i in range(4):
        b.append(ProducerRecord(T_ALPHA, f"v{i}".encode()))
    got_c1 = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    got_c2 = b.fetch_for(T_ALPHA, "s1", "c2", max_records=10)
    # c1 owns even entry_ids, c2 owns odd (consumers sorted [c1,c2])
    assert {m.value for m in got_c1} == {b"v0", b"v2"}
    assert {m.value for m in got_c2} == {b"v1", b"v3"}


def test_fetch_for_rejects_unknown_subscription() -> None:
    b = _broker()
    with pytest.raises(KeyError):
        b.fetch_for(T_ALPHA, "missing", "c1", max_records=1)


def test_fetch_for_rejects_zero_max_records() -> None:
    b = _broker()
    sub = Subscription("s1", T_ALPHA)
    b.attach_subscription(sub, "c1")
    with pytest.raises(ValueError):
        b.fetch_for(T_ALPHA, "s1", "c1", max_records=0)


# ---------------------------------------------------------------------------
# Acknowledge — individual + cumulative.
# ---------------------------------------------------------------------------


def test_cumulative_ack_advances_floor() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    b.attach_subscription(sub, "c1")
    for i in range(3):
        b.append(ProducerRecord(T_ALPHA, f"v{i}".encode()))
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    b.acknowledge(T_ALPHA, "s1", msgs[2].message_id, cumulative=True)
    state = b.subscription_state(T_ALPHA, "s1")
    assert state.cursor[0] == 3
    assert state.pending_acks[0] == set()


def test_cumulative_ack_rejected_for_shared() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    with pytest.raises(ValueError):
        b.acknowledge(T_ALPHA, "s1", msgs[0].message_id, cumulative=True)


def test_cumulative_ack_rejected_for_key_shared() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.KEY_SHARED)
    b.attach_subscription(sub, "c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    with pytest.raises(ValueError):
        b.acknowledge(T_ALPHA, "s1", msgs[0].message_id, cumulative=True)


def test_individual_ack_tracks_pending_below_cursor() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    b.acknowledge(T_ALPHA, "s1", msgs[0].message_id, cumulative=False)
    state = b.subscription_state(T_ALPHA, "s1")
    # entry_id 0 was acked and cursor advanced to 1 by fetch
    assert state.pending_acks[0] == set()


def test_negative_ack_rewinds_cursor_and_redelivers() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    b.attach_subscription(sub, "c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    first = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    assert first[0].redelivery_count == 0
    b.negative_acknowledge(T_ALPHA, "s1", first[0].message_id)
    redelivered = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    assert len(redelivered) == 1
    assert redelivered[0].value == b"v"
    assert redelivered[0].redelivery_count == 1


def test_negative_ack_unknown_subscription_raises() -> None:
    b = _broker()
    with pytest.raises(KeyError):
        b.negative_acknowledge(T_ALPHA, "missing", MessageId(0, 0, 0))


def test_acknowledge_unknown_subscription_raises() -> None:
    b = _broker()
    with pytest.raises(KeyError):
        b.acknowledge(T_ALPHA, "missing", MessageId(0, 0, 0), cumulative=True)


def test_subscription_state_unknown_raises() -> None:
    b = _broker()
    with pytest.raises(KeyError):
        b.subscription_state(T_ALPHA, "missing")


# ---------------------------------------------------------------------------
# InMemoryProducer.
# ---------------------------------------------------------------------------


def test_producer_send_with_key_hashes_partition() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=4),))
    p = InMemoryProducer(broker=b)
    expected = partition_for_key(b"alpha", 4)
    msg = p.send(T_ALPHA, b"v", key=b"alpha")
    assert msg.message_id.partition == expected


def test_producer_round_robin_no_key() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=3),))
    p = InMemoryProducer(broker=b)
    parts = [p.send(T_ALPHA, f"v{i}".encode()).message_id.partition for i in range(6)]
    assert parts == [0, 1, 2, 0, 1, 2]


def test_producer_send_after_close_raises() -> None:
    b = _broker()
    p = InMemoryProducer(broker=b)
    p.close()
    with pytest.raises(RuntimeError):
        p.send(T_ALPHA, b"v")


def test_producer_flush_after_close_raises() -> None:
    b = _broker()
    p = InMemoryProducer(broker=b)
    p.close()
    with pytest.raises(RuntimeError):
        p.flush()


# ---------------------------------------------------------------------------
# InMemoryConsumer.
# ---------------------------------------------------------------------------


def test_consumer_attach_on_init() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA)
    c = InMemoryConsumer(broker=b, subscription=sub, consumer_name="c1")
    assert b.subscription_state(T_ALPHA, "s1").consumers == ("c1",)
    c.close()
    assert b.subscription_state(T_ALPHA, "s1").consumers == ()


def test_consumer_receive_after_close_raises() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA)
    c = InMemoryConsumer(broker=b, subscription=sub, consumer_name="c1")
    c.close()
    with pytest.raises(RuntimeError):
        c.receive()


def test_consumer_acknowledges_cumulative_via_message() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE)
    c = InMemoryConsumer(broker=b, subscription=sub, consumer_name="c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    msgs = c.receive(max_records=10)
    c.acknowledge_cumulative(msgs[0])
    assert b.subscription_state(T_ALPHA, "s1").cursor[0] == 1
    c.close()


def test_consumer_nack_round_trip_increments_counter() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    c = InMemoryConsumer(broker=b, subscription=sub, consumer_name="c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    first = c.receive(max_records=10)
    c.negative_acknowledge(first[0])
    again = c.receive(max_records=10)
    assert again[0].redelivery_count == 1
    c.close()


def test_consumer_rejects_empty_name() -> None:
    b = _broker()
    with pytest.raises(ValueError):
        InMemoryConsumer(broker=b, subscription=Subscription("s1", T_ALPHA), consumer_name="")


def test_consumer_acknowledge_via_message_id() -> None:
    b = _broker(topics=(Topic(T_ALPHA, num_partitions=1),))
    sub = Subscription("s1", T_ALPHA, type=SubscriptionType.SHARED)
    c = InMemoryConsumer(broker=b, subscription=sub, consumer_name="c1")
    b.append(ProducerRecord(T_ALPHA, b"v"))
    msgs = c.receive(max_records=10)
    c.acknowledge(msgs[0].message_id)
    c.close()


# ---------------------------------------------------------------------------
# App builder.
# ---------------------------------------------------------------------------


def test_app_with_topic_appends_immutably() -> None:
    app = App()
    a1 = app.with_topic(Topic(T_ALPHA))
    a2 = a1.with_topic(Topic(T_BETA))
    assert app.topics == ()
    assert a1.topics == (Topic(T_ALPHA),)
    assert a2.topics == (Topic(T_ALPHA), Topic(T_BETA))


def test_app_with_topic_rejects_duplicate() -> None:
    app = App().with_topic(Topic(T_ALPHA))
    with pytest.raises(ValueError):
        app.with_topic(Topic(T_ALPHA))


def test_app_with_topic_rejects_non_topic() -> None:
    with pytest.raises(TypeError):
        App().with_topic("not-topic")  # type: ignore[arg-type]


def test_app_with_subscription_requires_known_topic() -> None:
    app = App().with_topic(Topic(T_ALPHA))
    with pytest.raises(ValueError):
        app.with_subscription(Subscription("s1", T_BETA))


def test_app_with_subscription_rejects_duplicate() -> None:
    app = App().with_topic(Topic(T_ALPHA)).with_subscription(Subscription("s1", T_ALPHA))
    with pytest.raises(ValueError):
        app.with_subscription(Subscription("s1", T_ALPHA))


def test_app_with_subscription_rejects_non_subscription() -> None:
    app = App().with_topic(Topic(T_ALPHA))
    with pytest.raises(TypeError):
        app.with_subscription("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_app — deterministic simulator + INV-15 replay equality.
# ---------------------------------------------------------------------------


def _replay_app() -> App:
    return (
        App()
        .with_topic(Topic(T_ALPHA, num_partitions=2))
        .with_topic(Topic(T_BETA, num_partitions=1))
        .with_subscription(Subscription("alpha-excl", T_ALPHA, type=SubscriptionType.EXCLUSIVE))
        .with_subscription(Subscription("beta-shared", T_BETA, type=SubscriptionType.SHARED))
    )


def _replay_inbound() -> list[ProducerRecord]:
    out: list[ProducerRecord] = []
    for i in range(6):
        out.append(
            ProducerRecord(
                T_ALPHA,
                f"alpha-{i}".encode(),
                key=str(i % 3).encode(),
                ts_ns=10 + i,
            )
        )
        out.append(
            ProducerRecord(
                T_BETA,
                f"beta-{i}".encode(),
                ts_ns=20 + i,
            )
        )
    return out


def test_run_app_returns_app_result() -> None:
    res = run_app(_replay_app(), _replay_inbound())
    assert isinstance(res, AppResult)
    assert isinstance(res.app_digest, str)
    assert len(res.app_digest) == 32


def test_run_app_delivers_every_record() -> None:
    inbound = _replay_inbound()
    res = run_app(_replay_app(), inbound)
    # alpha has 2 subscriptions worth of deliveries? Only one alpha sub.
    alpha_count = sum(1 for r in res.records if r.subscription_name == "alpha-excl")
    beta_count = sum(1 for r in res.records if r.subscription_name == "beta-shared")
    assert alpha_count == 6
    assert beta_count == 6


def test_run_app_records_canonical_sorted() -> None:
    res = run_app(_replay_app(), _replay_inbound())
    keys = [(r.subscription_name, r.message.topic_name, r.message.message_id) for r in res.records]
    assert keys == sorted(keys)


def test_run_app_three_run_byte_equality() -> None:
    a = run_app(_replay_app(), _replay_inbound())
    b = run_app(_replay_app(), _replay_inbound())
    c = run_app(_replay_app(), _replay_inbound())
    assert a.app_digest == b.app_digest == c.app_digest
    assert a.records == b.records == c.records


def test_run_app_rejects_non_producer_record() -> None:
    with pytest.raises(TypeError):
        run_app(_replay_app(), ["not-a-record"])  # type: ignore[list-item]


def test_run_app_rejects_non_app() -> None:
    with pytest.raises(TypeError):
        run_app("not-app", [])  # type: ignore[arg-type]


def test_run_app_rejects_empty_topics() -> None:
    with pytest.raises(ValueError):
        run_app(App(), [])


def test_bus_digest_pure_function() -> None:
    res = run_app(_replay_app(), _replay_inbound())
    assert bus_digest(res.records) == res.app_digest


# ---------------------------------------------------------------------------
# Multi-tenancy — different tenants live in the same broker.
# ---------------------------------------------------------------------------


def test_multi_tenant_topics_have_independent_logs() -> None:
    b = InMemoryPulsarBroker(
        topics=(Topic(T_ALPHA, num_partitions=1), Topic(T_OTHER_TENANT, num_partitions=1))
    )
    b.append(ProducerRecord(T_ALPHA, b"a"))
    b.append(ProducerRecord(T_OTHER_TENANT, b"b"))
    assert b.stream_length(T_ALPHA, 0) == 1
    assert b.stream_length(T_OTHER_TENANT, 0) == 1


def test_subscription_on_one_tenant_does_not_see_other() -> None:
    b = InMemoryPulsarBroker(
        topics=(Topic(T_ALPHA, num_partitions=1), Topic(T_OTHER_TENANT, num_partitions=1))
    )
    b.attach_subscription(Subscription("s1", T_ALPHA), "c1")
    b.append(ProducerRecord(T_OTHER_TENANT, b"other"))
    b.append(ProducerRecord(T_ALPHA, b"mine"))
    msgs = b.fetch_for(T_ALPHA, "s1", "c1", max_records=10)
    assert {m.value for m in msgs} == {b"mine"}


# ---------------------------------------------------------------------------
# Lazy seam.
# ---------------------------------------------------------------------------


def test_pulsar_client_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        pulsar_client_factory(PulsarConfig())


def test_pulsar_client_factory_rejects_bad_config() -> None:
    with pytest.raises(TypeError):
        pulsar_client_factory("not-config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cross-process worker bridge.
# ---------------------------------------------------------------------------


def _worker_app() -> App:
    return (
        App()
        .with_topic(Topic(T_ALPHA, num_partitions=1))
        .with_subscription(Subscription("s1", T_ALPHA, type=SubscriptionType.EXCLUSIVE))
    )


def test_spawn_pulsar_worker_round_trip() -> None:
    app = _worker_app()
    process, in_q, out_q = spawn_pulsar_worker(app, poll_timeout_s=0.05)
    try:
        in_q.put(ProducerRecord(T_ALPHA, b"hello", ts_ns=1))
        # expect: Message + DeliveredRecord
        msg = out_q.get(timeout=5)
        assert isinstance(msg, Message)
        delivered = out_q.get(timeout=5)
        assert isinstance(delivered, DeliveredRecord)
        assert delivered.message.value == b"hello"
        in_q.put(PulsarBusSentinel())
        sentinel = out_q.get(timeout=5)
        assert isinstance(sentinel, PulsarBusSentinel)
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_spawn_pulsar_worker_rejects_bad_item() -> None:
    app = _worker_app()
    process, in_q, out_q = spawn_pulsar_worker(app, poll_timeout_s=0.05)
    try:
        in_q.put("garbage")  # type: ignore[arg-type]
        err = out_q.get(timeout=5)
        assert isinstance(err, ValueError)
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_spawn_pulsar_worker_rejects_bad_app() -> None:
    with pytest.raises(TypeError):
        spawn_pulsar_worker("not-app")  # type: ignore[arg-type]


def test_spawn_pulsar_worker_rejects_zero_timeout() -> None:
    with pytest.raises(ValueError):
        spawn_pulsar_worker(_worker_app(), poll_timeout_s=0)


# ---------------------------------------------------------------------------
# AST guardrails — pin the canonical authority discipline.
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return pathlib.Path(inspect.getfile(pb)).read_text()


def _module_tree() -> ast.Module:
    return ast.parse(_module_source())


FORBIDDEN_TOP_LEVEL_IMPORTS = frozenset(
    {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "pulsar",
        "numpy",
        "torch",
        "polars",
        "aiokafka",
        "confluent_kafka",
        "redis",
        "hiredis",
        "nats",
        "requests",
    }
)

FORBIDDEN_TYPED_EVENT_CTORS = frozenset(
    {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
    }
)

FORBIDDEN_RUNTIME_TIER_PREFIXES = (
    "intelligence_engine",
    "execution_engine",
    "governance_engine",
    "evolution_engine",
    "learning_engine",
)


def test_ast_no_forbidden_top_level_imports() -> None:
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in FORBIDDEN_TOP_LEVEL_IMPORTS, (
                    f"forbidden top-level import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_root = (node.module or "").split(".")[0]
            assert mod_root not in FORBIDDEN_TOP_LEVEL_IMPORTS, (
                f"forbidden top-level from-import: {node.module}"
            )


def test_ast_no_runtime_tier_imports() -> None:
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in FORBIDDEN_RUNTIME_TIER_PREFIXES:
                    assert not alias.name.startswith(prefix), (
                        f"runtime-tier import forbidden: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in FORBIDDEN_RUNTIME_TIER_PREFIXES:
                assert not module.startswith(prefix), (
                    f"runtime-tier from-import forbidden: {node.module}"
                )


def test_ast_no_typed_event_constructors() -> None:
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_TYPED_EVENT_CTORS, (
                f"forbidden typed-event ctor call: {node.func.id}"
            )


def test_ast_module_has_canonical_adapted_from_header() -> None:
    src = _module_source()
    assert "# ADAPTED FROM:" in src
    assert "apache/pulsar" in src


def test_ast_module_declares_offline_only_tier() -> None:
    src = _module_source()
    assert "Tier: OFFLINE_ONLY" in src


# ---------------------------------------------------------------------------
# Module can be re-imported without side effects (no global state).
# ---------------------------------------------------------------------------


def test_module_reimport_idempotent() -> None:
    reimported = importlib.reload(pb)
    assert reimported.PULSAR_BUS_VERSION == PULSAR_BUS_VERSION
    assert reimported.NEW_PIP_DEPENDENCIES == NEW_PIP_DEPENDENCIES


# ---------------------------------------------------------------------------
# Value-object hygiene — every dataclass is frozen + slotted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        Topic,
        Subscription,
        MessageId,
        ProducerRecord,
        Message,
        PulsarConfig,
        App,
        AppResult,
        DeliveredRecord,
        PulsarBusSentinel,
    ],
)
def test_value_objects_are_frozen_and_slotted(cls: type) -> None:
    params = getattr(cls, "__dataclass_params__", None)
    assert params is not None, f"{cls.__name__} must be a dataclass"
    assert params.frozen is True, f"{cls.__name__} must be frozen"
    # slotted dataclasses have a __slots__ tuple.
    assert hasattr(cls, "__slots__"), f"{cls.__name__} must declare __slots__"


@pytest.mark.parametrize(
    "cls,expected",
    [
        (Topic, {"name", "num_partitions", "retention_ns"}),
        (
            Subscription,
            {"name", "topic_name", "type", "initial_position_earliest"},
        ),
        (MessageId, {"ledger_id", "entry_id", "partition", "batch_idx"}),
    ],
)
def test_value_object_field_set(cls: type, expected: set[str]) -> None:
    assert {f.name for f in fields(cls)} == expected
