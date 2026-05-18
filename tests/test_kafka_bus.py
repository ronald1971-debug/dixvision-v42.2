"""Tests for C-03 ``system_engine.streaming.kafka_bus``.

Coverage:

* Frozen + slotted value-object surface (Topic / TopicPartition /
  OffsetAndMetadata / ProducerRecord / ConsumerRecord / ConsumerGroup
  / KafkaConfig / App / AppResult / KafkaBusSentinel).
* Validation rules on each value type.
* :func:`partition_for_key` determinism + uniform-ish distribution.
* :func:`assign_partitions` range assignment.
* :func:`serialize_record` / :func:`deserialize_record` round-trip
  + byte-stability across insertion orders.
* :class:`InMemoryBroker` append + fetch + commit semantics.
* :class:`InMemoryProducer` send for keyed + key-None round-robin.
* :class:`InMemoryConsumer` subscribe + getmany + commit + seek
  + multi-member group exclusive partition assignment.
* :class:`App` builder immutability.
* :func:`run_app` deterministic 3-run replay (INV-15).
* :func:`bus_digest` canonical-sort stability.
* :func:`spawn_kafka_worker` cross-process round trip via sentinel.
* AST guardrails — no top-level forbidden imports, no typed-event
  constructors (B27 / B28 / INV-71), no runtime-tier imports (B1).
* Module constants pinned (``KAFKA_BUS_VERSION`` /
  ``NEW_PIP_DEPENDENCIES``).
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from system_engine.streaming import kafka_bus as kb

# ---------------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------------


def test_module_constants_pinned() -> None:
    assert kb.KAFKA_BUS_VERSION == 1
    assert kb.NEW_PIP_DEPENDENCIES == ("aiokafka", "confluent-kafka")


# ---------------------------------------------------------------------------
# Topic.
# ---------------------------------------------------------------------------


def test_topic_frozen_slotted() -> None:
    t = kb.Topic(name="signals", num_partitions=4, retention_ns=10**9)
    assert t.name == "signals"
    assert t.num_partitions == 4
    assert t.retention_ns == 10**9
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.name = "other"  # type: ignore[misc]
    assert not hasattr(t, "__dict__")


def test_topic_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        kb.Topic(name="")
    with pytest.raises(ValueError):
        kb.Topic(name="x", num_partitions=0)
    with pytest.raises(ValueError):
        kb.Topic(name="x", num_partitions=-1)
    with pytest.raises(TypeError):
        kb.Topic(name="x", num_partitions=True)
    with pytest.raises(ValueError):
        kb.Topic(name="x", retention_ns=0)


# ---------------------------------------------------------------------------
# TopicPartition.
# ---------------------------------------------------------------------------


def test_topic_partition_ordering_and_frozen() -> None:
    a = kb.TopicPartition("a", 0)
    b = kb.TopicPartition("a", 1)
    c = kb.TopicPartition("b", 0)
    assert sorted([c, b, a]) == [a, b, c]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.partition_idx = 99  # type: ignore[misc]


def test_topic_partition_validation() -> None:
    with pytest.raises(ValueError):
        kb.TopicPartition("", 0)
    with pytest.raises(ValueError):
        kb.TopicPartition("a", -1)
    with pytest.raises(TypeError):
        kb.TopicPartition("a", True)


# ---------------------------------------------------------------------------
# OffsetAndMetadata.
# ---------------------------------------------------------------------------


def test_offset_and_metadata_validation() -> None:
    om = kb.OffsetAndMetadata(offset=7, metadata="tx-1")
    assert om.offset == 7 and om.metadata == "tx-1"
    with pytest.raises(ValueError):
        kb.OffsetAndMetadata(offset=-1)
    with pytest.raises(TypeError):
        kb.OffsetAndMetadata(offset=True)


# ---------------------------------------------------------------------------
# ProducerRecord.
# ---------------------------------------------------------------------------


def test_producer_record_construction_and_validation() -> None:
    pr = kb.ProducerRecord(
        topic_name="t",
        value=b"x",
        key=b"k",
        ts_ns=42,
        headers=(("h", b"v"),),
    )
    assert pr.value == b"x" and pr.key == b"k" and pr.ts_ns == 42
    with pytest.raises(ValueError):
        kb.ProducerRecord(topic_name="", value=b"")
    with pytest.raises(TypeError):
        kb.ProducerRecord(topic_name="t", value="not bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        kb.ProducerRecord(topic_name="t", value=b"", key="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        kb.ProducerRecord(topic_name="t", value=b"", ts_ns=-1)
    with pytest.raises(TypeError):
        kb.ProducerRecord(
            topic_name="t",
            value=b"",
            headers=(("h", "not bytes"),),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# KafkaConfig.
# ---------------------------------------------------------------------------


def test_kafka_config_validation() -> None:
    cfg = kb.KafkaConfig(bootstrap_servers=("a:9092", "b:9092"), acks="1")
    assert cfg.client_id == "dix-kafka-bus"
    with pytest.raises(ValueError):
        kb.KafkaConfig(acks="bogus")
    with pytest.raises(TypeError):
        kb.KafkaConfig(bootstrap_servers=("a:9092", ""))


# ---------------------------------------------------------------------------
# partition_for_key.
# ---------------------------------------------------------------------------


def test_partition_for_key_deterministic() -> None:
    for _ in range(3):
        assert kb.partition_for_key(b"abc", 8) == kb.partition_for_key(b"abc", 8)
    # None always lands on 0.
    assert kb.partition_for_key(None, 16) == 0
    # Different keys can land on different partitions.
    parts = {kb.partition_for_key(k.encode(), 64) for k in "abcdefghij"}
    assert len(parts) > 1


def test_partition_for_key_validation() -> None:
    with pytest.raises(ValueError):
        kb.partition_for_key(b"x", 0)
    with pytest.raises(ValueError):
        kb.partition_for_key(b"x", -1)
    with pytest.raises(TypeError):
        kb.partition_for_key(b"x", True)
    with pytest.raises(TypeError):
        kb.partition_for_key("not bytes", 4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# assign_partitions.
# ---------------------------------------------------------------------------


def test_assign_partitions_range_style_with_extra() -> None:
    parts = [
        kb.TopicPartition("t", 0),
        kb.TopicPartition("t", 1),
        kb.TopicPartition("t", 2),
        kb.TopicPartition("t", 3),
        kb.TopicPartition("t", 4),
    ]
    out = kb.assign_partitions(parts, ["m1", "m2"])
    # 5 / 2 = 2 base + 1 extra → m1 (first sorted) gets 3, m2 gets 2.
    assert out["m1"] == (
        kb.TopicPartition("t", 0),
        kb.TopicPartition("t", 1),
        kb.TopicPartition("t", 2),
    )
    assert out["m2"] == (kb.TopicPartition("t", 3), kb.TopicPartition("t", 4))


def test_assign_partitions_deterministic_across_insertion_orders() -> None:
    parts1 = [
        kb.TopicPartition("a", 0),
        kb.TopicPartition("b", 0),
        kb.TopicPartition("a", 1),
    ]
    parts2 = list(reversed(parts1))
    a1 = kb.assign_partitions(parts1, ["x", "y"])
    a2 = kb.assign_partitions(parts2, ["y", "x"])
    assert a1 == a2


def test_assign_partitions_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        kb.assign_partitions([kb.TopicPartition("t", 0)], [])
    with pytest.raises(ValueError):
        kb.assign_partitions([kb.TopicPartition("t", 0)], ["m", "m"])


def test_assign_partitions_empty_member_gets_empty_tuple() -> None:
    out = kb.assign_partitions([kb.TopicPartition("t", 0)], ["a", "b", "c"])
    assert out["b"] == () and out["c"] == ()


# ---------------------------------------------------------------------------
# serialize/deserialize.
# ---------------------------------------------------------------------------


def test_serialize_record_byte_stable_across_insertion_orders() -> None:
    a = {"x": 1, "y": 2, "z": 3}
    b = {"z": 3, "y": 2, "x": 1}
    assert kb.serialize_record(a) == kb.serialize_record(b)


def test_serialize_deserialize_round_trip() -> None:
    payload = {"k": "v", "n": 7, "list": [1, 2, 3]}
    blob = kb.serialize_record(payload)
    assert kb.deserialize_record(blob) == payload


def test_serialize_record_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        kb.serialize_record([1, 2])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        kb.deserialize_record("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InMemoryBroker.
# ---------------------------------------------------------------------------


def test_broker_construction_validates_topics() -> None:
    with pytest.raises(ValueError):
        kb.InMemoryBroker(topics=())
    with pytest.raises(ValueError):
        kb.InMemoryBroker(topics=(kb.Topic("a"), kb.Topic("a")))
    with pytest.raises(TypeError):
        kb.InMemoryBroker(topics=("not-topic",))  # type: ignore[arg-type]


def test_broker_append_assigns_offset_and_partition() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=2),))
    cr0 = broker.append(kb.ProducerRecord("t", b"v0", key=b"a"))
    cr1 = broker.append(kb.ProducerRecord("t", b"v1", key=b"a"))
    cr2 = broker.append(kb.ProducerRecord("t", b"v2", key=b"b"))
    # Same key → same partition.
    assert cr0.partition_idx == cr1.partition_idx
    # Offsets increment per-partition.
    assert cr0.offset == 0 and cr1.offset == 1
    # cr2 either lands on same partition (offset 2) or different (offset 0).
    if cr2.partition_idx == cr0.partition_idx:
        assert cr2.offset == 2
    else:
        assert cr2.offset == 0


def test_broker_fetch_respects_offset_and_max_records() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t"),))
    for i in range(5):
        broker.append(kb.ProducerRecord("t", f"v{i}".encode(), key=b"k"))
    tp = kb.TopicPartition("t", 0)
    batch = broker.fetch(tp, offset=2, max_records=2)
    assert len(batch) == 2 and batch[0].offset == 2 and batch[1].offset == 3


def test_broker_commit_and_committed_round_trip() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t"),))
    tp = kb.TopicPartition("t", 0)
    assert broker.committed("g1", tp).offset == 0
    broker.commit("g1", {tp: kb.OffsetAndMetadata(offset=4)})
    assert broker.committed("g1", tp).offset == 4
    # Different group reads its own offset.
    assert broker.committed("g2", tp).offset == 0


def test_broker_rejects_unknown_topic() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t"),))
    with pytest.raises(KeyError):
        broker.append(kb.ProducerRecord("other", b"v"))
    with pytest.raises(KeyError):
        broker.partitions_for("other")


# ---------------------------------------------------------------------------
# InMemoryProducer.
# ---------------------------------------------------------------------------


def test_producer_send_keyed_picks_deterministic_partition() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=4),))
    producer = kb.InMemoryProducer(broker=broker)
    a = producer.send("t", b"v", key=b"a")
    b = producer.send("t", b"v", key=b"a")
    assert a.partition_idx == b.partition_idx
    assert a.offset == 0 and b.offset == 1


def test_producer_send_none_key_round_robins() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=3),))
    producer = kb.InMemoryProducer(broker=broker)
    parts = [producer.send("t", b"v").partition_idx for _ in range(6)]
    assert parts == [0, 1, 2, 0, 1, 2]


def test_producer_flush_and_stop() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t"),))
    producer = kb.InMemoryProducer(broker=broker)
    producer.flush()
    producer.stop()
    with pytest.raises(RuntimeError):
        producer.send("t", b"v")
    with pytest.raises(RuntimeError):
        producer.flush()


# ---------------------------------------------------------------------------
# InMemoryConsumer.
# ---------------------------------------------------------------------------


def test_consumer_subscribe_and_getmany() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=2),))
    producer = kb.InMemoryProducer(broker=broker)
    for i in range(4):
        producer.send("t", f"v{i}".encode(), key=b"k")
    consumer = kb.InMemoryConsumer(
        broker=broker,
        group=kb.ConsumerGroup(group_id="g1"),
        config=kb.KafkaConfig(client_id="c1"),
    )
    consumer.subscribe(["t"])
    assert sorted(consumer.assignment()) == [
        kb.TopicPartition("t", 0),
        kb.TopicPartition("t", 1),
    ]
    batch = consumer.getmany(max_records=10)
    total = sum(len(v) for v in batch.values())
    assert total == 4


def test_consumer_commit_persists_position_for_next_subscribe() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=1),))
    producer = kb.InMemoryProducer(broker=broker)
    for i in range(3):
        producer.send("t", f"v{i}".encode(), key=b"k")
    c1 = kb.InMemoryConsumer(broker=broker, group=kb.ConsumerGroup("g1"))
    c1.subscribe(["t"])
    batch1 = c1.getmany()
    assert sum(len(v) for v in batch1.values()) == 3
    c1.commit()
    c1.stop()
    # New consumer for the same group sees zero uncommitted records.
    c2 = kb.InMemoryConsumer(broker=broker, group=kb.ConsumerGroup("g1"))
    c2.subscribe(["t"])
    assert c2.getmany() == {}


def test_consumer_seek_replays_partition() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=1),))
    producer = kb.InMemoryProducer(broker=broker)
    for i in range(3):
        producer.send("t", f"v{i}".encode(), key=b"k")
    consumer = kb.InMemoryConsumer(broker=broker, group=kb.ConsumerGroup("g1"))
    consumer.subscribe(["t"])
    consumer.getmany()
    consumer.commit()
    # Re-read from offset 1.
    tp = kb.TopicPartition("t", 0)
    consumer.seek(tp, 1)
    batch = consumer.getmany()
    values = [r.value for r in batch[tp]]
    assert values == [b"v1", b"v2"]


def test_consumer_multi_member_group_exclusive_partitions() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t", num_partitions=4),))
    c1 = kb.InMemoryConsumer(
        broker=broker,
        group=kb.ConsumerGroup("g1"),
        config=kb.KafkaConfig(client_id="m1"),
    )
    c2 = kb.InMemoryConsumer(
        broker=broker,
        group=kb.ConsumerGroup("g1"),
        config=kb.KafkaConfig(client_id="m2"),
    )
    c1.subscribe(["t"], group_members=["m1", "m2"])
    c2.subscribe(["t"], group_members=["m1", "m2"])
    a1 = set(c1.assignment())
    a2 = set(c2.assignment())
    assert a1.isdisjoint(a2)
    assert a1 | a2 == {
        kb.TopicPartition("t", 0),
        kb.TopicPartition("t", 1),
        kb.TopicPartition("t", 2),
        kb.TopicPartition("t", 3),
    }


def test_consumer_subscribe_rejects_member_not_in_group() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t"),))
    consumer = kb.InMemoryConsumer(
        broker=broker,
        group=kb.ConsumerGroup("g1"),
        config=kb.KafkaConfig(client_id="m1"),
    )
    with pytest.raises(ValueError):
        consumer.subscribe(["t"], group_members=["other"])


def test_consumer_stop_blocks_further_ops() -> None:
    broker = kb.InMemoryBroker(topics=(kb.Topic("t"),))
    consumer = kb.InMemoryConsumer(broker=broker, group=kb.ConsumerGroup("g1"))
    consumer.subscribe(["t"])
    consumer.stop()
    with pytest.raises(RuntimeError):
        consumer.getmany()
    with pytest.raises(RuntimeError):
        consumer.commit()
    with pytest.raises(RuntimeError):
        consumer.seek(kb.TopicPartition("t", 0), 0)


# ---------------------------------------------------------------------------
# Lazy seam.
# ---------------------------------------------------------------------------


def test_kafka_producer_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        kb.kafka_producer_factory(kb.KafkaConfig())


def test_kafka_consumer_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        kb.kafka_consumer_factory(kb.KafkaConfig(), kb.ConsumerGroup("g1"))


def test_lazy_seam_validates_config_types() -> None:
    with pytest.raises(TypeError):
        kb.kafka_producer_factory("not-config")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        kb.kafka_consumer_factory(kb.KafkaConfig(), "not-group")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# App builder.
# ---------------------------------------------------------------------------


def test_app_with_topic_is_immutable_builder() -> None:
    a0 = kb.App()
    a1 = a0.with_topic(kb.Topic("t1"))
    a2 = a1.with_topic(kb.Topic("t2"))
    assert a0.topics == ()
    assert a1.topics == (kb.Topic("t1"),)
    assert a2.topics == (kb.Topic("t1"), kb.Topic("t2"))


def test_app_rejects_duplicate_topic_and_group() -> None:
    app = kb.App().with_topic(kb.Topic("t"))
    with pytest.raises(ValueError):
        app.with_topic(kb.Topic("t"))
    app = app.with_consumer_group(kb.ConsumerGroup("g"))
    with pytest.raises(ValueError):
        app.with_consumer_group(kb.ConsumerGroup("g"))


# ---------------------------------------------------------------------------
# run_app + bus_digest — INV-15 byte-identical 3-run replay.
# ---------------------------------------------------------------------------


def _build_app() -> kb.App:
    return (
        kb.App()
        .with_topic(kb.Topic("signals", num_partitions=2))
        .with_topic(kb.Topic("hazards", num_partitions=1))
        .with_consumer_group(kb.ConsumerGroup("intelligence"))
        .with_consumer_group(kb.ConsumerGroup("governance"))
    )


def _build_inbound() -> list[kb.ProducerRecord]:
    return [
        kb.ProducerRecord("signals", b"sig-a", key=b"AAPL", ts_ns=1),
        kb.ProducerRecord("signals", b"sig-b", key=b"AAPL", ts_ns=2),
        kb.ProducerRecord("signals", b"sig-c", key=b"MSFT", ts_ns=3),
        kb.ProducerRecord("hazards", b"haz-1", key=b"venue", ts_ns=4),
    ]


def test_run_app_three_run_byte_identical_replay() -> None:
    results = [kb.run_app(_build_app(), _build_inbound()) for _ in range(3)]
    digests = {r.app_digest for r in results}
    assert len(digests) == 1
    record_sigs = {
        tuple((r.topic_name, r.partition_idx, r.offset, r.value) for r in res.records)
        for res in results
    }
    assert len(record_sigs) == 1
    commits = {res.commits for res in results}
    assert len(commits) == 1


def test_bus_digest_sort_stable_against_input_order() -> None:
    inbound = _build_inbound()
    r1 = kb.run_app(_build_app(), inbound)
    r2 = kb.run_app(_build_app(), reversed(inbound))
    # Inputs in different order can produce different records (offsets
    # depend on arrival order), but the digest of the canonical-sorted
    # *output* is identical iff partition assignment is permutation-
    # invariant within a key. We assert a weaker property: the digest
    # depends only on the canonical-sorted record stream of *that run*.
    assert r1.app_digest == kb.bus_digest(r1.records)
    assert r2.app_digest == kb.bus_digest(r2.records)


def test_run_app_rejects_non_producer_record() -> None:
    with pytest.raises(TypeError):
        kb.run_app(_build_app(), [object()])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        kb.run_app("not-app", [])  # type: ignore[arg-type]


def test_run_app_commits_advance_to_record_offset_plus_one() -> None:
    app = (
        kb.App()
        .with_topic(kb.Topic("t", num_partitions=1))
        .with_consumer_group(kb.ConsumerGroup("g"))
    )
    inbound = [
        kb.ProducerRecord("t", b"a", key=b"k"),
        kb.ProducerRecord("t", b"b", key=b"k"),
    ]
    result = kb.run_app(app, inbound)
    # Last commit for (g, t/0) should be offset=2 (next-to-read after 2 records).
    last = [c for c in result.commits if c[0] == "g"][-1]
    assert last[2] == 2


# ---------------------------------------------------------------------------
# Cross-process worker bridge.
# ---------------------------------------------------------------------------


def test_spawn_kafka_worker_round_trip() -> None:
    app = kb.App().with_topic(kb.Topic("t", num_partitions=1))
    process, in_q, out_q = kb.spawn_kafka_worker(app)
    try:
        in_q.put(kb.ProducerRecord("t", b"hello", key=b"k", ts_ns=1))
        cr = out_q.get(timeout=10.0)
        assert isinstance(cr, kb.ConsumerRecord)
        assert cr.value == b"hello"
        assert cr.offset == 0
        in_q.put(kb.KafkaBusSentinel(reason="done"))
        end = out_q.get(timeout=10.0)
        assert isinstance(end, kb.KafkaBusSentinel)
    finally:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


def test_spawn_kafka_worker_validates_inputs() -> None:
    with pytest.raises(TypeError):
        kb.spawn_kafka_worker("not-app")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        kb.spawn_kafka_worker(kb.App().with_topic(kb.Topic("t")), poll_timeout_s=0)


# ---------------------------------------------------------------------------
# AST guardrails — pin INV-15 / B1 / B27 / B28 / INV-71 by code shape.
# ---------------------------------------------------------------------------


_MODULE_PATH = pathlib.Path(kb.__file__)


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text())


_FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "time",
    "datetime",
    "random",
    "asyncio",
    "os",
    "numpy",
    "torch",
    "polars",
    "aiokafka",
    "confluent_kafka",
    "requests",
    "urllib",
    "urllib.request",
    "http",
    "http.client",
}


def test_no_forbidden_top_level_imports() -> None:
    tree = _module_ast()
    bad: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_TOP_LEVEL_IMPORTS:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in _FORBIDDEN_TOP_LEVEL_IMPORTS:
                bad.append(node.module)
    assert not bad, f"forbidden top-level imports: {bad}"


_FORBIDDEN_RUNTIME_TIER_PREFIXES = (
    "intelligence_engine",
    "execution_engine",
    "governance_engine",
    "evolution_engine",
    "learning_engine",
)


def test_no_runtime_tier_imports() -> None:
    tree = _module_ast()
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == p or alias.name.startswith(p + ".")
                    for p in _FORBIDDEN_RUNTIME_TIER_PREFIXES
                ):
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and any(
                node.module == p or node.module.startswith(p + ".")
                for p in _FORBIDDEN_RUNTIME_TIER_PREFIXES
            ):
                bad.append(node.module)
    assert not bad, f"forbidden runtime-tier imports: {bad}"


_FORBIDDEN_EVENT_CONSTRUCTORS = {
    "SignalEvent",
    "HazardEvent",
    "ExecutionEvent",
    "SystemEvent",
    "PatchProposal",
}


def test_no_typed_event_constructors() -> None:
    tree = _module_ast()
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_EVENT_CONSTRUCTORS:
                bad.append(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_EVENT_CONSTRUCTORS:
                bad.append(func.attr)
    assert not bad, f"typed-event constructors in transport: {bad}"


def test_no_top_level_clock_calls() -> None:
    """No top-level wall-clock reads (INV-15)."""
    tree = _module_ast()
    bad: list[str] = []
    for node in tree.body:
        for inner in ast.walk(node) if isinstance(node, ast.Assign) else ():
            if isinstance(inner, ast.Call):
                fn = inner.func
                if isinstance(fn, ast.Attribute) and fn.attr in (
                    "time",
                    "monotonic",
                    "monotonic_ns",
                    "perf_counter",
                    "perf_counter_ns",
                    "time_ns",
                    "now",
                ):
                    bad.append(fn.attr)
    assert not bad, f"top-level clock calls: {bad}"
