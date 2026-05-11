"""C-03 aiokafka + confluent-kafka — Distributed Event Bus.

# ADAPTED FROM: aio-libs/aiokafka — ``aiokafka/producer/producer.py``
# (``AIOKafkaProducer`` async send/flush surface),
# ``aiokafka/consumer/consumer.py`` (``AIOKafkaConsumer`` async
# ``getmany`` + explicit ``commit`` offset management),
# ``aiokafka/structs.py`` (``ConsumerRecord`` / ``TopicPartition`` /
# ``OffsetAndMetadata`` envelope shape), and
# ``confluentinc/confluent-kafka-python`` — ``cimpl.Producer`` (sync
# fall-back surface) + topic/partition/consumer-group semantics.
#
# Tier: OFFLINE_ONLY — the Kafka bus runs as a SEPARATE process feeding
# DIX via cross-process IPC. The bus never imports any RUNTIME tier
# (no ``intelligence_engine``, ``execution_engine``,
# ``governance_engine``, ``evolution_engine``, ``learning_engine``
# imports). The bus emits :class:`ConsumerRecord` advisory records that
# *carry* already-constructed DIX events (``SignalEvent`` /
# ``HazardEvent`` / ``ExecutionEvent`` / ``SystemEvent`` /
# ``PatchProposal``); the bus NEVER constructs new typed events itself.
# This preserves B27 / B28 / INV-71 authority symmetry — only the
# engine that produced an event may construct it, never the transport.

This module is the canonical alternative to
:mod:`system_engine.streaming.event_fabric` (C-01 bytewax) and
:mod:`system_engine.streaming.faust_bus` (C-02 faust-streaming) for
*distributed multi-process* deployments where a single host is
insufficient. The three transports expose interchangeable contract
surfaces so production callers can swap between them based on the
deployment topology:

* :mod:`event_fabric` (bytewax) — dataflow-style operator chain,
  single-process or fixed worker count.
* :mod:`faust_bus` (faust-streaming) — agent / topic CEP-style routing
  with first-class event-time tumbling-window tables, single-process.
* :mod:`kafka_bus` (aiokafka) — distributed, partitioned, consumer-
  group-coordinated event bus suitable for multi-host deployments.

Kafka's selling point over Bytewax and Faust is *distributed*
partitioned consumption with explicit offset management and consumer
groups — multiple processes (potentially on different hosts) consume
non-overlapping partition subsets of the same topic, and consumer
position is durably persisted via explicit ``commit()`` calls. The
in-process :class:`InMemoryBroker` emulates that surface deterministically
in pure Python; the lazy seam :func:`kafka_producer_factory` /
:func:`kafka_consumer_factory` gates activation of the real
:mod:`aiokafka` / :mod:`confluent_kafka` PyPI packages behind a future
research-acceptance PR.

Topic-per-event-type design (mandated by canonical block C-03):

* Every DIX event class maps to exactly one Kafka topic.
* Topic name == event class name (no inline prefixes / suffixes — the
  bus is a leaf transport; routing policy belongs upstream).
* The bus stores raw bytes per record; the caller chooses serialization.
  The reference helpers :func:`serialize_record` / :func:`deserialize_record`
  use ``json.dumps(..., sort_keys=True, separators=(",", ":")).encode()``
  to give an orjson-compatible UTF-8 wire shape without taking a
  dependency on :mod:`orjson` (which is a separate canonical block).

Determinism (INV-15):

* No top-level imports of :mod:`time` / :mod:`datetime` / :mod:`random`
  / :mod:`asyncio` / :mod:`aiokafka` / :mod:`confluent_kafka` /
  :mod:`os` / :mod:`numpy` / :mod:`torch` / :mod:`polars`.
* Partition assignment is a pure hash of the record key via
  :func:`partition_for_key` (BLAKE2b-8 mod ``num_partitions``). Same
  key, same partition, always.
* Consumer-group partition assignment is *range* — partitions are
  sorted by ``(topic_name, partition_idx)`` and assigned to sorted
  ``member_id`` strings in contiguous slices. Deterministic and
  Kafka-compatible.
* In-memory broker log: append-only ``(topic, partition) → list[ConsumerRecord]``
  keyed by offset; offset starts at ``0`` and increments by ``1`` per
  produce.
* :func:`run_app` drains a deterministic in-process simulator and
  returns records in canonical
  ``(topic_name asc, partition_idx asc, offset asc)`` order. Two
  byte-identical input streams produce two byte-identical output
  streams.
* Frozen, slotted dataclasses everywhere. The :class:`KafkaConfig` /
  :class:`Topic` / :class:`ProducerRecord` / :class:`ConsumerRecord` /
  :class:`TopicPartition` / :class:`OffsetAndMetadata` envelopes are
  all value-objects; mutation = returning a new instance.
* BLAKE2b-16 :func:`bus_digest` over the topic / consumer-group spec
  gives byte-identical replay equality.

Worker bridge:

* :func:`spawn_kafka_worker` uses
  ``multiprocessing.get_context("spawn")`` so the child process has an
  independent interpreter (no inherited module state). Callbacks
  passed to consumer agents must be top-level module-importable
  callables for cross-process use; lambdas and closures are fine for
  in-process replay tests.
* The worker terminates cleanly on a :class:`KafkaBusSentinel` on the
  inbound queue.

Authority discipline:

* B27 / B28 / INV-71: this module does **not** call
  ``PatchProposal(...)``, ``HazardEvent(...)``, ``SignalEvent(...)``,
  ``ExecutionEvent(...)`` or ``SystemEvent(...)`` directly. AST tests
  pin the constraint.
* B1 isolation: no imports from ``intelligence_engine``,
  ``execution_engine``, ``governance_engine``, ``evolution_engine``,
  ``learning_engine``. The bus is a leaf transport.

Outputs declared by canonical block C-03:

1. ``system_engine/streaming/kafka_bus.py`` (this file)
2. ``tests/test_kafka_bus.py``
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from queue import Empty
from typing import Any

KAFKA_BUS_VERSION: int = 1

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("aiokafka", "confluent-kafka")
"""Declared so the canonical pin-set is complete.

The packages themselves are NEVER imported in this module — see the
module docstring for the rationale and :func:`kafka_producer_factory`
/ :func:`kafka_consumer_factory` for the lazy seams where a future PR
can wire them up after the research-acceptance gate is documented.
"""


# ---------------------------------------------------------------------------
# Topic — named partitioned channel.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Topic:
    """Named partitioned channel.

    ``name`` is the topic identifier (one per DIX event type per the
    topic-per-event-type rule). ``num_partitions`` is the parallel-
    consumption fan-out factor; consumer-group members are assigned
    contiguous partition slices. ``retention_ns`` is advisory — the
    in-memory broker keeps every record for the lifetime of the
    process and ignores retention.

    # ADAPTED FROM: aiokafka admin client + confluent_kafka topic
    # creation (``NewTopic(name, num_partitions, replication_factor)``).
    """

    name: str
    num_partitions: int = 1
    retention_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Topic.name must be non-empty")
        if not isinstance(self.num_partitions, int) or isinstance(
            self.num_partitions, bool
        ):
            raise TypeError("Topic.num_partitions must be int")
        if self.num_partitions <= 0:
            raise ValueError("Topic.num_partitions must be > 0")
        if self.retention_ns is not None and self.retention_ns <= 0:
            raise ValueError("Topic.retention_ns must be > 0 when set")


# ---------------------------------------------------------------------------
# TopicPartition — addressable shard.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class TopicPartition:
    """Addressable shard ``(topic_name, partition_idx)``.

    Ordered for deterministic iteration during consumer-group rebalancing.

    # ADAPTED FROM: aiokafka.structs.TopicPartition.
    """

    topic_name: str
    partition_idx: int

    def __post_init__(self) -> None:
        if not self.topic_name:
            raise ValueError("TopicPartition.topic_name must be non-empty")
        if not isinstance(self.partition_idx, int) or isinstance(
            self.partition_idx, bool
        ):
            raise TypeError("TopicPartition.partition_idx must be int")
        if self.partition_idx < 0:
            raise ValueError("TopicPartition.partition_idx must be >= 0")


# ---------------------------------------------------------------------------
# OffsetAndMetadata — durable consumer position.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OffsetAndMetadata:
    """Durable consumer position for a single partition.

    ``offset`` is the *next* offset the consumer will read — Kafka
    convention. ``metadata`` is an opaque caller string commonly used
    to carry a transaction id or a snapshot tag.

    # ADAPTED FROM: aiokafka.structs.OffsetAndMetadata.
    """

    offset: int
    metadata: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.offset, int) or isinstance(self.offset, bool):
            raise TypeError("OffsetAndMetadata.offset must be int")
        if self.offset < 0:
            raise ValueError("OffsetAndMetadata.offset must be >= 0")


# ---------------------------------------------------------------------------
# ProducerRecord / ConsumerRecord — envelopes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProducerRecord:
    """Envelope handed to :meth:`InMemoryProducer.send`.

    ``value`` is opaque ``bytes`` — the producer never inspects payload
    semantics. ``key`` selects the partition deterministically via
    :func:`partition_for_key`; ``key=None`` round-robins on
    ``partition_idx % num_partitions`` using a caller-supplied counter
    on :meth:`InMemoryProducer.send`. ``ts_ns`` is the event-time
    timestamp from the caller (NEVER wall-clock from inside the bus,
    INV-15). ``headers`` is a sorted tuple of ``(key, value)`` byte
    pairs to keep the wire shape byte-stable.

    # ADAPTED FROM: aiokafka.structs.RecordMetadata producer side.
    """

    topic_name: str
    value: bytes
    key: bytes | None = None
    ts_ns: int = 0
    headers: tuple[tuple[str, bytes], ...] = ()

    def __post_init__(self) -> None:
        if not self.topic_name:
            raise ValueError("ProducerRecord.topic_name must be non-empty")
        if not isinstance(self.value, bytes):
            raise TypeError("ProducerRecord.value must be bytes")
        if self.key is not None and not isinstance(self.key, bytes):
            raise TypeError("ProducerRecord.key must be bytes or None")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError("ProducerRecord.ts_ns must be int")
        if self.ts_ns < 0:
            raise ValueError("ProducerRecord.ts_ns must be >= 0")
        if not isinstance(self.headers, tuple):
            raise TypeError("ProducerRecord.headers must be tuple")
        for entry in self.headers:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], bytes)
            ):
                raise TypeError(
                    "ProducerRecord.headers entries must be (str, bytes) tuples"
                )


@dataclass(frozen=True, slots=True)
class ConsumerRecord:
    """Envelope handed back from :meth:`InMemoryConsumer.getmany`.

    All fields are populated by the broker on append. ``offset`` is the
    *assigned* offset of this record in its partition log (not the
    next-to-read position).

    # ADAPTED FROM: aiokafka.structs.ConsumerRecord.
    """

    topic_name: str
    partition_idx: int
    offset: int
    value: bytes
    key: bytes | None
    ts_ns: int
    headers: tuple[tuple[str, bytes], ...]


# ---------------------------------------------------------------------------
# ConsumerGroup — multi-consumer coordination handle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsumerGroup:
    """Logical consumer group identifier.

    A consumer group is a set of consumer processes that share
    partition consumption. The bus assigns partitions to members in
    *range* style: partitions are sorted by ``(topic_name,
    partition_idx)``, members are sorted by ``member_id``, and each
    member gets a contiguous slice.

    # ADAPTED FROM: aiokafka group-coordinator partition-assignment.
    """

    group_id: str

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("ConsumerGroup.group_id must be non-empty")


# ---------------------------------------------------------------------------
# KafkaConfig — bus-wide config value-object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KafkaConfig:
    """Bus-wide config envelope.

    ``bootstrap_servers`` is advisory for the in-memory broker (it has
    no real network). The real ``aiokafka`` clients consume this field
    via the lazy factory. ``client_id`` is the producer/consumer
    process tag for telemetry. ``acks`` mirrors Kafka's durability
    knob — ``"all"`` waits for in-sync replica acknowledgement.

    # ADAPTED FROM: aiokafka.AIOKafkaProducer/Consumer ctor kwargs.
    """

    bootstrap_servers: tuple[str, ...] = ()
    client_id: str = "dix-kafka-bus"
    acks: str = "all"

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap_servers, tuple):
            raise TypeError("KafkaConfig.bootstrap_servers must be tuple")
        for srv in self.bootstrap_servers:
            if not isinstance(srv, str) or not srv:
                raise TypeError(
                    "KafkaConfig.bootstrap_servers entries must be non-empty str"
                )
        if self.acks not in ("0", "1", "all"):
            raise ValueError("KafkaConfig.acks must be one of '0' / '1' / 'all'")


# ---------------------------------------------------------------------------
# Partitioning — deterministic key → partition_idx.
# ---------------------------------------------------------------------------


def partition_for_key(key: bytes | None, num_partitions: int) -> int:
    """Deterministic partition assignment for a key.

    ``key=None`` deterministically returns ``0``; callers that want
    round-robin must drive a counter externally (the in-memory producer
    does this) so the bus stays pure.

    Hash is BLAKE2b-8 of the key, taken mod ``num_partitions``. Stable
    across runs / processes / platforms — same key always maps to the
    same partition.

    # ADAPTED FROM: aiokafka.partitioner.DefaultPartitioner (murmur2),
    # adjusted to BLAKE2b-8 to keep the module's import set minimal
    # (BLAKE2b is stdlib :mod:`hashlib`; murmur2 would need an
    # additional pin).
    """
    if not isinstance(num_partitions, int) or isinstance(num_partitions, bool):
        raise TypeError("num_partitions must be int")
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    if key is None:
        return 0
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes or None")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_partitions


# ---------------------------------------------------------------------------
# Range-style consumer-group partition assignment.
# ---------------------------------------------------------------------------


def assign_partitions(
    partitions: Sequence[TopicPartition],
    member_ids: Sequence[str],
) -> dict[str, tuple[TopicPartition, ...]]:
    """Range-assign ``partitions`` across ``member_ids``.

    Deterministic: ``partitions`` is sorted by ``(topic_name,
    partition_idx)`` (via :class:`TopicPartition`'s ``order=True``),
    ``member_ids`` are sorted lexicographically, and each member gets
    a contiguous slice. When ``len(partitions) % len(member_ids) != 0``,
    earlier members get one extra partition each — exactly the
    aiokafka ``RangePartitionAssignor`` rule.

    Returns ``{member_id: tuple_of_topic_partitions}`` with every
    member_id present (empty tuple if no partitions).

    # ADAPTED FROM: aiokafka/coordinator/assignors/range.py.
    """
    if not member_ids:
        raise ValueError("member_ids must be non-empty")
    sorted_parts = sorted(partitions)
    sorted_members = sorted(member_ids)
    if len(set(sorted_members)) != len(sorted_members):
        raise ValueError("member_ids must be unique")
    result: dict[str, list[TopicPartition]] = {m: [] for m in sorted_members}
    n_parts = len(sorted_parts)
    n_members = len(sorted_members)
    base = n_parts // n_members
    extra = n_parts % n_members
    cursor = 0
    for i, member in enumerate(sorted_members):
        share = base + (1 if i < extra else 0)
        for part in sorted_parts[cursor : cursor + share]:
            result[member].append(part)
        cursor += share
    return {m: tuple(parts) for m, parts in result.items()}


# ---------------------------------------------------------------------------
# Reference serializer / deserializer.
# ---------------------------------------------------------------------------


def serialize_record(payload: Mapping[str, Any]) -> bytes:
    """Reference helper — canonical JSON encoder.

    ``json.dumps(payload, sort_keys=True, separators=(",", ":"))`` then
    UTF-8 encode. Byte-stable across insertion orders, matches the wire
    shape :mod:`orjson` produces for plain dict-of-scalars payloads.

    The bus itself never calls this — it stores opaque bytes — but
    tests and downstream callers use it to get a reproducible payload.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def deserialize_record(blob: bytes) -> dict[str, Any]:
    """Reference helper — inverse of :func:`serialize_record`."""
    if not isinstance(blob, bytes):
        raise TypeError("blob must be bytes")
    return json.loads(blob.decode("utf-8"))


# ---------------------------------------------------------------------------
# InMemoryBroker — append-only partition log.
# ---------------------------------------------------------------------------


@dataclass
class InMemoryBroker:
    """In-process Kafka-shaped broker.

    Holds an append-only log per ``(topic_name, partition_idx)`` and
    durable consumer-group offsets per ``(group_id, topic_name,
    partition_idx)``. Used by the lazy seams' default factories and by
    the deterministic :func:`run_app` simulator.

    The broker itself is mutable (it accumulates log entries) but is
    only ever touched through the immutable producer / consumer
    handles — callers never see the live ``_logs`` / ``_offsets``
    dicts. INV-15 byte-identical replay holds because every mutation
    is driven by a caller-supplied value-object ordering.
    """

    topics: tuple[Topic, ...]
    _logs: dict[TopicPartition, list[ConsumerRecord]] = field(
        default_factory=dict
    )
    _offsets: dict[tuple[str, TopicPartition], int] = field(
        default_factory=dict
    )
    _topic_index: dict[str, Topic] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.topics, tuple):
            raise TypeError("InMemoryBroker.topics must be tuple")
        if not self.topics:
            raise ValueError("InMemoryBroker.topics must be non-empty")
        names: set[str] = set()
        for topic in self.topics:
            if not isinstance(topic, Topic):
                raise TypeError("InMemoryBroker.topics entries must be Topic")
            if topic.name in names:
                raise ValueError(
                    f"InMemoryBroker.topics duplicate name {topic.name!r}"
                )
            names.add(topic.name)
            self._topic_index[topic.name] = topic
            for idx in range(topic.num_partitions):
                self._logs[TopicPartition(topic.name, idx)] = []

    # ----- producer-side -----

    def append(self, record: ProducerRecord) -> ConsumerRecord:
        topic = self._topic_index.get(record.topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {record.topic_name!r}")
        partition_idx = partition_for_key(record.key, topic.num_partitions)
        tp = TopicPartition(record.topic_name, partition_idx)
        log = self._logs[tp]
        consumer_record = ConsumerRecord(
            topic_name=record.topic_name,
            partition_idx=partition_idx,
            offset=len(log),
            value=record.value,
            key=record.key,
            ts_ns=record.ts_ns,
            headers=record.headers,
        )
        log.append(consumer_record)
        return consumer_record

    # ----- consumer-side -----

    def partitions_for(self, topic_name: str) -> tuple[TopicPartition, ...]:
        topic = self._topic_index.get(topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {topic_name!r}")
        return tuple(
            TopicPartition(topic_name, idx)
            for idx in range(topic.num_partitions)
        )

    def fetch(
        self,
        tp: TopicPartition,
        offset: int,
        max_records: int,
    ) -> tuple[ConsumerRecord, ...]:
        if max_records <= 0:
            raise ValueError("max_records must be > 0")
        log = self._logs.get(tp)
        if log is None:
            raise KeyError(f"unknown partition {tp!r}")
        return tuple(log[offset : offset + max_records])

    def committed(
        self, group_id: str, tp: TopicPartition
    ) -> OffsetAndMetadata:
        return OffsetAndMetadata(
            offset=self._offsets.get((group_id, tp), 0),
            metadata="",
        )

    def commit(
        self,
        group_id: str,
        positions: Mapping[TopicPartition, OffsetAndMetadata],
    ) -> None:
        for tp, om in positions.items():
            if tp not in self._logs:
                raise KeyError(f"unknown partition {tp!r}")
            self._offsets[(group_id, tp)] = om.offset


# ---------------------------------------------------------------------------
# InMemoryProducer — value-shape producer handle.
# ---------------------------------------------------------------------------


@dataclass
class InMemoryProducer:
    """Pure in-process producer.

    Mirrors :class:`AIOKafkaProducer`'s ``send`` / ``flush`` /
    ``stop`` surface synchronously. ``send`` returns the
    :class:`ConsumerRecord` that was appended (the same shape ``await
    producer.send_and_wait()`` would return ``RecordMetadata`` for).

    ``round_robin_counter`` drives ``key=None`` partition selection so
    callers don't have to maintain state externally. The counter is
    advanced per-topic, not globally — same byte-identical replay.
    """

    broker: InMemoryBroker
    config: KafkaConfig = field(default_factory=KafkaConfig)
    _rr_counters: dict[str, int] = field(default_factory=dict)
    _running: bool = True

    def send(
        self,
        topic_name: str,
        value: bytes,
        *,
        key: bytes | None = None,
        ts_ns: int = 0,
        headers: tuple[tuple[str, bytes], ...] = (),
    ) -> ConsumerRecord:
        if not self._running:
            raise RuntimeError("InMemoryProducer is stopped")
        if key is None:
            rr = self._rr_counters.get(topic_name, 0)
            topic = self.broker._topic_index.get(topic_name)
            if topic is None:
                raise KeyError(f"unknown topic {topic_name!r}")
            partition_idx = rr % topic.num_partitions
            self._rr_counters[topic_name] = rr + 1
            # encode partition_idx as a synthetic key so the broker's
            # partitioner picks the same partition deterministically
            # *and* mirrors a real round-robin partitioner that bypasses
            # the hash. We bypass the hash by constructing the record
            # directly on the broker's chosen partition.
            tp = TopicPartition(topic_name, partition_idx)
            log = self.broker._logs[tp]
            consumer_record = ConsumerRecord(
                topic_name=topic_name,
                partition_idx=partition_idx,
                offset=len(log),
                value=value,
                key=None,
                ts_ns=ts_ns,
                headers=headers,
            )
            log.append(consumer_record)
            return consumer_record
        record = ProducerRecord(
            topic_name=topic_name,
            value=value,
            key=key,
            ts_ns=ts_ns,
            headers=headers,
        )
        return self.broker.append(record)

    def flush(self) -> None:
        # In-memory append is synchronous; flush is a no-op.
        if not self._running:
            raise RuntimeError("InMemoryProducer is stopped")

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# InMemoryConsumer — value-shape consumer handle.
# ---------------------------------------------------------------------------


@dataclass
class InMemoryConsumer:
    """Pure in-process consumer.

    Mirrors :class:`AIOKafkaConsumer`'s ``subscribe`` /
    ``assignment`` / ``getmany`` / ``commit`` / ``seek`` /
    ``position`` / ``stop`` surface synchronously.

    ``group`` is the consumer group; multiple consumers sharing the
    same ``group.group_id`` and ``member_id`` set will get *disjoint*
    partition assignments. ``member_id`` defaults to ``client_id``
    from :class:`KafkaConfig` so single-process callers don't need to
    invent one.
    """

    broker: InMemoryBroker
    group: ConsumerGroup
    config: KafkaConfig = field(default_factory=KafkaConfig)
    member_id: str = ""
    _subscribed_topics: tuple[str, ...] = ()
    _group_members: tuple[str, ...] = ()
    _assigned: tuple[TopicPartition, ...] = ()
    _positions: dict[TopicPartition, int] = field(default_factory=dict)
    _running: bool = True

    def __post_init__(self) -> None:
        if not self.member_id:
            object.__setattr__(self, "member_id", self.config.client_id)
        if not self._group_members:
            object.__setattr__(self, "_group_members", (self.member_id,))

    def subscribe(
        self,
        topics: Sequence[str],
        *,
        group_members: Sequence[str] | None = None,
    ) -> None:
        if not self._running:
            raise RuntimeError("InMemoryConsumer is stopped")
        if not topics:
            raise ValueError("topics must be non-empty")
        members = (
            tuple(group_members)
            if group_members is not None
            else (self.member_id,)
        )
        if self.member_id not in members:
            raise ValueError("member_id must be present in group_members")
        # Build the full partition set the group is consuming.
        all_partitions: list[TopicPartition] = []
        for topic_name in topics:
            all_partitions.extend(self.broker.partitions_for(topic_name))
        assignments = assign_partitions(all_partitions, members)
        self._subscribed_topics = tuple(topics)
        self._group_members = members
        self._assigned = assignments[self.member_id]
        # Seed positions from durable offsets.
        new_positions: dict[TopicPartition, int] = {}
        for tp in self._assigned:
            new_positions[tp] = self.broker.committed(
                self.group.group_id, tp
            ).offset
        self._positions = new_positions

    def assignment(self) -> tuple[TopicPartition, ...]:
        return self._assigned

    def position(self, tp: TopicPartition) -> int:
        if tp not in self._positions:
            raise KeyError(f"partition {tp!r} not assigned")
        return self._positions[tp]

    def getmany(
        self,
        *,
        max_records: int = 100,
    ) -> dict[TopicPartition, tuple[ConsumerRecord, ...]]:
        """Fetch up to ``max_records`` from each assigned partition.

        Returned dict iteration order is the canonical
        ``(topic_name asc, partition_idx asc)`` so callers iterating
        over it see a byte-stable order. Empty partitions are omitted.
        """
        if not self._running:
            raise RuntimeError("InMemoryConsumer is stopped")
        if max_records <= 0:
            raise ValueError("max_records must be > 0")
        out: dict[TopicPartition, tuple[ConsumerRecord, ...]] = {}
        for tp in sorted(self._assigned):
            position = self._positions[tp]
            batch = self.broker.fetch(tp, position, max_records)
            if batch:
                out[tp] = batch
                self._positions[tp] = position + len(batch)
        return out

    def commit(
        self,
        offsets: Mapping[TopicPartition, OffsetAndMetadata] | None = None,
    ) -> None:
        """Commit consumer-group offsets.

        If ``offsets is None``, commits the consumer's current
        in-memory position for every assigned partition. This mirrors
        ``AIOKafkaConsumer.commit()``'s default.
        """
        if not self._running:
            raise RuntimeError("InMemoryConsumer is stopped")
        if offsets is None:
            offsets = {
                tp: OffsetAndMetadata(offset=self._positions[tp], metadata="")
                for tp in self._assigned
            }
        self.broker.commit(self.group.group_id, offsets)

    def seek(self, tp: TopicPartition, offset: int) -> None:
        if not self._running:
            raise RuntimeError("InMemoryConsumer is stopped")
        if tp not in self._positions:
            raise KeyError(f"partition {tp!r} not assigned")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        self._positions[tp] = offset

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Lazy seam — gated activation of the real aiokafka / confluent_kafka.
# ---------------------------------------------------------------------------


def kafka_producer_factory(config: KafkaConfig) -> Any:
    """Lazy seam — returns a real :class:`AIOKafkaProducer` once gated.

    This function is intentionally NOT implemented in this PR. It is
    the canonical hook a future research-acceptance PR will fill in,
    after that PR documents the shadow-equivalence comparison between
    :class:`InMemoryProducer` and :class:`aiokafka.AIOKafkaProducer`
    against the same byte-identical replay stream.

    Until then, calling it raises :class:`NotImplementedError`. The
    module-level :data:`NEW_PIP_DEPENDENCIES` declares the gated
    packages so the canonical pin-set is complete.
    """
    if not isinstance(config, KafkaConfig):
        raise TypeError("config must be KafkaConfig")
    raise NotImplementedError(
        "kafka_producer_factory is a lazy seam; activation gated on "
        "research-acceptance PR"
    )


def kafka_consumer_factory(
    config: KafkaConfig, group: ConsumerGroup
) -> Any:
    """Lazy seam — returns a real :class:`AIOKafkaConsumer` once gated.

    See :func:`kafka_producer_factory` for the activation rules.
    """
    if not isinstance(config, KafkaConfig):
        raise TypeError("config must be KafkaConfig")
    if not isinstance(group, ConsumerGroup):
        raise TypeError("group must be ConsumerGroup")
    raise NotImplementedError(
        "kafka_consumer_factory is a lazy seam; activation gated on "
        "research-acceptance PR"
    )


# ---------------------------------------------------------------------------
# App / run_app — deterministic in-process simulator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class App:
    """Pure value-object spec of a Kafka bus.

    Mirrors the C-01 / C-02 ``Dataflow`` / ``App`` shape — registering
    a topic / consumer group returns a *new* :class:`App` so the spec
    stays a builder. :func:`run_app` materialises one over an
    :class:`InMemoryBroker`.
    """

    topics: tuple[Topic, ...] = ()
    consumer_groups: tuple[ConsumerGroup, ...] = ()
    config: KafkaConfig = field(default_factory=KafkaConfig)

    def with_topic(self, topic: Topic) -> App:
        if not isinstance(topic, Topic):
            raise TypeError("topic must be Topic")
        for existing in self.topics:
            if existing.name == topic.name:
                raise ValueError(f"duplicate topic {topic.name!r}")
        return replace(self, topics=(*self.topics, topic))

    def with_consumer_group(self, group: ConsumerGroup) -> App:
        if not isinstance(group, ConsumerGroup):
            raise TypeError("group must be ConsumerGroup")
        for existing in self.consumer_groups:
            if existing.group_id == group.group_id:
                raise ValueError(
                    f"duplicate consumer group {group.group_id!r}"
                )
        return replace(
            self,
            consumer_groups=(*self.consumer_groups, group),
        )


@dataclass(frozen=True, slots=True)
class AppResult:
    """Outcome of :func:`run_app`.

    ``records`` is the full ordered log produced during the run,
    canonical-sorted by ``(topic_name asc, partition_idx asc, offset
    asc)``. ``commits`` mirrors the durable consumer-group offsets at
    end-of-run. ``app_digest`` is a BLAKE2b-16 hex digest over the
    sorted record bytes + commits, suitable for INV-15 3-run replay
    equality assertions.
    """

    records: tuple[ConsumerRecord, ...]
    commits: tuple[tuple[str, TopicPartition, int], ...]
    app_digest: str


def bus_digest(records: Iterable[ConsumerRecord]) -> str:
    """BLAKE2b-16 digest over the canonical-sorted record stream.

    Useful as a stand-alone replay-equality fingerprint when callers
    want to compare two runs without invoking :func:`run_app`.
    """
    h = hashlib.blake2b(digest_size=16)
    for rec in sorted(
        records, key=lambda r: (r.topic_name, r.partition_idx, r.offset)
    ):
        h.update(rec.topic_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(rec.partition_idx.to_bytes(4, "big"))
        h.update(rec.offset.to_bytes(8, "big"))
        h.update(len(rec.value).to_bytes(8, "big"))
        h.update(rec.value)
        h.update(b"\x00" if rec.key is None else b"\x01")
        if rec.key is not None:
            h.update(len(rec.key).to_bytes(8, "big"))
            h.update(rec.key)
        h.update(rec.ts_ns.to_bytes(8, "big"))
        for hk, hv in rec.headers:
            h.update(hk.encode("utf-8"))
            h.update(b"\x00")
            h.update(len(hv).to_bytes(8, "big"))
            h.update(hv)
    return h.hexdigest()


def run_app(
    app: App,
    inbound: Iterable[ProducerRecord],
) -> AppResult:
    """Drain ``inbound`` through ``app`` deterministically.

    For each record:

    1. Append to the broker via :meth:`InMemoryBroker.append` (key
       partitions via :func:`partition_for_key`, ``key=None`` lands on
       partition ``0`` — round-robin is a producer-side concern).
    2. For each consumer group, advance the *implicit* consumer to the
       record's offset and commit immediately. This makes
       :func:`run_app` a "fire-and-commit-each-message" simulator,
       useful for replay tests; it is NOT how a real Kafka consumer
       behaves (real consumers batch via ``getmany`` then commit).
       For batch-shape testing call the :class:`InMemoryConsumer` API
       directly.

    Returns :class:`AppResult` with records canonical-sorted and
    commits in deterministic order.
    """
    if not isinstance(app, App):
        raise TypeError("app must be App")
    broker = InMemoryBroker(topics=app.topics)
    appended: list[ConsumerRecord] = []
    for record in inbound:
        if not isinstance(record, ProducerRecord):
            raise TypeError("inbound records must be ProducerRecord")
        cr = broker.append(record)
        appended.append(cr)
        for group in app.consumer_groups:
            broker.commit(
                group.group_id,
                {
                    TopicPartition(cr.topic_name, cr.partition_idx): (
                        OffsetAndMetadata(offset=cr.offset + 1, metadata="")
                    )
                },
            )
    sorted_records = tuple(
        sorted(
            appended,
            key=lambda r: (r.topic_name, r.partition_idx, r.offset),
        )
    )
    commits = tuple(
        sorted(
            (g, tp, offset)
            for (g, tp), offset in broker._offsets.items()
        )
    )
    return AppResult(
        records=sorted_records,
        commits=commits,
        app_digest=bus_digest(sorted_records),
    )


# ---------------------------------------------------------------------------
# Cross-process worker bridge.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KafkaBusSentinel:
    """Sentinel posted on the worker's inbound queue to terminate it."""

    reason: str = "shutdown"


def _kafka_worker_loop(
    app: App,
    in_q: Any,
    out_q: Any,
    poll_timeout_s: float,
) -> None:
    """Cross-process worker body. Drains ``in_q`` until a sentinel."""
    broker = InMemoryBroker(topics=app.topics)
    while True:
        try:
            item = in_q.get(timeout=poll_timeout_s)
        except Empty:
            continue
        if isinstance(item, KafkaBusSentinel):
            out_q.put(item)
            return
        if not isinstance(item, ProducerRecord):
            out_q.put(
                ValueError(
                    f"worker received non-ProducerRecord: {type(item).__name__}"
                )
            )
            return
        cr = broker.append(item)
        out_q.put(cr)


def spawn_kafka_worker(
    app: App,
    *,
    poll_timeout_s: float = 0.05,
) -> tuple[Any, Any, Any]:
    """Spawn a ``multiprocessing`` worker draining produce records.

    Returns ``(process, inbound_queue, outbound_queue)``. Producers
    feed :class:`ProducerRecord` instances into the inbound queue;
    the worker appends each to its own :class:`InMemoryBroker` and
    posts the resulting :class:`ConsumerRecord` back on the outbound
    queue. Terminate by posting a :class:`KafkaBusSentinel`.

    Uses ``multiprocessing.get_context("spawn")`` so the worker has
    no inherited module state — critical for INV-15 byte-identical
    replay across hosts.
    """
    if not isinstance(app, App):
        raise TypeError("app must be App")
    if poll_timeout_s <= 0:
        raise ValueError("poll_timeout_s must be > 0")
    ctx = multiprocessing.get_context("spawn")
    in_q: Any = ctx.Queue()
    out_q: Any = ctx.Queue()
    process = ctx.Process(
        target=_kafka_worker_loop,
        args=(app, in_q, out_q, poll_timeout_s),
        daemon=False,
    )
    process.start()
    return process, in_q, out_q


__all__ = [
    "App",
    "AppResult",
    "ConsumerGroup",
    "ConsumerRecord",
    "InMemoryBroker",
    "InMemoryConsumer",
    "InMemoryProducer",
    "KAFKA_BUS_VERSION",
    "KafkaBusSentinel",
    "KafkaConfig",
    "NEW_PIP_DEPENDENCIES",
    "OffsetAndMetadata",
    "ProducerRecord",
    "Topic",
    "TopicPartition",
    "assign_partitions",
    "bus_digest",
    "deserialize_record",
    "kafka_consumer_factory",
    "kafka_producer_factory",
    "partition_for_key",
    "run_app",
    "serialize_record",
    "spawn_kafka_worker",
]
