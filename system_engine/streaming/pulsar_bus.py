"""C-06 pulsar-client — Multi-Tenant Streaming Event Bus.

# ADAPTED FROM: apache/pulsar — ``python/pulsar/__init__.py``
# (``Client.create_producer`` / ``Client.subscribe``, ``Producer.send``
# / ``Producer.send_async``, ``Consumer.receive`` / ``Consumer.acknowledge``
# / ``Consumer.acknowledge_cumulative`` / ``Consumer.negative_acknowledge``,
# ``MessageId`` (ledger_id / entry_id / partition / batch_index)
# envelope shape, ``SubscriptionType`` enum
# (Exclusive / Shared / Failover / Key_Shared)) and the canonical
# ``persistent://tenant/namespace/topic`` qualified topic-name grammar.
#
# Tier: OFFLINE_ONLY — the Pulsar bus runs as a SEPARATE process feeding
# DIX via cross-process IPC. The bus never imports any RUNTIME tier
# (no ``intelligence_engine``, ``execution_engine``, ``governance_engine``,
# ``evolution_engine``, ``learning_engine`` imports). The bus emits
# :class:`Message` advisory records that *carry* already-constructed DIX
# events; the bus NEVER constructs new typed events itself. This preserves
# B27 / B28 / INV-71 authority symmetry — only the engine that produced
# an event may construct it, never the transport.

This module is the canonical alternative to :mod:`event_fabric` (C-01
bytewax), :mod:`faust_bus` (C-02 faust-streaming), :mod:`kafka_bus` (C-03
aiokafka) and :mod:`nats_bus` (C-05 nats-py) for *multi-tenant* multi-
process deployments where one host serves several logical operators on
the same bus. The five transports expose interchangeable value-object
contract surfaces so production callers can swap between them based on
the deployment topology:

* :mod:`event_fabric` (bytewax) — dataflow-style operator chain,
  single-process or fixed worker count.
* :mod:`faust_bus` (faust-streaming) — agent / topic CEP-style routing
  with first-class event-time tumbling-window tables, single-process.
* :mod:`kafka_bus` (aiokafka) — distributed, partitioned, consumer-
  group-coordinated event bus suitable for multi-host deployments.
* :mod:`nats_bus` (nats-py) — subject wildcard pub/sub with JetStream
  durable consumers, lightweight pre-Pulsar transport.
* :mod:`pulsar_bus` (pulsar-client) — *multi-tenant* partitioned event
  bus with first-class subscription types (Exclusive / Shared / Failover
  / Key_Shared) and a fully-qualified ``persistent://tenant/namespace/
  topic`` name space.

Pulsar's selling point over Kafka / NATS is *multi-tenancy* via the
``tenant / namespace`` hierarchy plus first-class subscription
semantics. The in-process :class:`InMemoryPulsarBroker` emulates that
surface deterministically in pure Python; the lazy seam
:func:`pulsar_client_factory` gates activation of the real
:mod:`pulsar` (``pulsar-client``) PyPI package behind a future
research-acceptance PR.

Topic-per-event-type design with tenancy (mandated by canonical block
C-06):

* Every DIX event class maps to exactly one Pulsar topic.
* Topic name == ``persistent://<tenant>/<namespace>/<event-class-name>``
  (fully qualified — the bus rejects unqualified names).
* Multi-tenant routing: different operators (tenants) may publish to
  topics in their own namespaces; one broker instance carries them all.
* The bus stores raw bytes per record; the caller chooses serialization.
  The reference helpers :func:`serialize_record` / :func:`deserialize_record`
  use ``json.dumps(..., sort_keys=True, separators=(",", ":")).encode()``
  to give an orjson-compatible UTF-8 wire shape without taking a
  dependency on :mod:`orjson` (which is a separate canonical block).

Subscription types (mandated by Pulsar semantics):

* :class:`SubscriptionType.EXCLUSIVE` — exactly one consumer per
  subscription. Additional connect attempts raise.
* :class:`SubscriptionType.SHARED` — N consumers split messages
  round-robin; same-key messages may land on different consumers.
* :class:`SubscriptionType.FAILOVER` — N consumers, the
  lexicographically-smallest active ``consumer_name`` is primary;
  others are standby and only receive if the primary disconnects.
* :class:`SubscriptionType.KEY_SHARED` — N consumers, same partition
  key always lands on the same consumer (BLAKE2b-8 ``key % N``).

Determinism (INV-15):

* No top-level imports of :mod:`time` / :mod:`datetime` / :mod:`random`
  / :mod:`asyncio` / :mod:`pulsar` / :mod:`os` / :mod:`numpy` /
  :mod:`torch` / :mod:`polars` / :mod:`aiokafka` / :mod:`confluent_kafka`
  / :mod:`redis` / :mod:`hiredis` / :mod:`nats` / :mod:`requests`.
* Partition assignment is a pure hash of the record key via
  :func:`partition_for_key` (BLAKE2b-8 mod ``num_partitions``). Same
  key, same partition, always.
* Message IDs are assigned monotonically per partition:
  ``MessageId(ledger_id=0, entry_id=offset_in_partition,
  partition=partition_idx, batch_idx=0)`` — stable across runs.
* In-memory broker log: append-only ``(qualified_name, partition_idx) →
  list[Message]`` keyed by entry id; entry id starts at ``0`` and
  increments by ``1`` per produce.
* Each subscription maintains its own cursor (independent of others on
  the same topic). Cumulative ack advances the cursor to ``message_id
  + 1``; individual ack records the id in the cumulative-pending set
  and only advances the cursor through contiguous acks.
* :func:`run_app` drains a deterministic in-process simulator and
  returns records canonical-sorted by
  ``(qualified_name asc, partition_idx asc, entry_id asc)``. Two
  byte-identical input streams produce two byte-identical output
  streams.
* Frozen, slotted dataclasses everywhere. The :class:`PulsarConfig` /
  :class:`Topic` / :class:`Subscription` / :class:`MessageId` /
  :class:`ProducerRecord` / :class:`Message` envelopes are value-objects;
  mutation = returning a new instance.
* BLAKE2b-16 :func:`bus_digest` over the topic / subscription spec
  gives byte-identical replay equality.

Worker bridge:

* :func:`spawn_pulsar_worker` uses
  ``multiprocessing.get_context("spawn")`` so the child process has an
  independent interpreter (no inherited module state). The worker
  terminates cleanly on a :class:`PulsarBusSentinel` on the inbound
  queue.

Authority discipline:

* B27 / B28 / INV-71: this module does **not** call
  ``PatchProposal(...)``, ``HazardEvent(...)``, ``SignalEvent(...)``,
  ``ExecutionEvent(...)`` or ``SystemEvent(...)`` directly. AST tests
  pin the constraint.
* B1 isolation: no imports from ``intelligence_engine``,
  ``execution_engine``, ``governance_engine``, ``evolution_engine``,
  ``learning_engine``. The bus is a leaf transport.

Outputs declared by canonical block C-06:

1. ``system_engine/streaming/pulsar_bus.py`` (this file)
2. ``tests/test_pulsar_bus.py``
"""

from __future__ import annotations

import enum
import hashlib
import json
import multiprocessing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from queue import Empty
from typing import Any

PULSAR_BUS_VERSION: int = 1

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("pulsar-client",)
"""Declared so the canonical pin-set is complete.

The package itself is NEVER imported in this module — see the module
docstring for the rationale and :func:`pulsar_client_factory` for the
lazy seam where a future PR can wire it up after the research-acceptance
gate is documented.
"""


# ---------------------------------------------------------------------------
# SubscriptionType enum — first-class Pulsar subscription semantics.
# ---------------------------------------------------------------------------


class SubscriptionType(enum.Enum):
    """First-class Pulsar subscription semantics.

    # ADAPTED FROM: pulsar.SubscriptionType (Exclusive / Shared /
    # Failover / Key_Shared).
    """

    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    FAILOVER = "failover"
    KEY_SHARED = "key_shared"


# ---------------------------------------------------------------------------
# Qualified topic name — persistent://tenant/namespace/topic.
# ---------------------------------------------------------------------------


def parse_qualified_topic(name: str) -> tuple[str, str, str, str]:
    """Parse ``persistent://<tenant>/<namespace>/<topic>`` into parts.

    Returns ``(scheme, tenant, namespace, topic)``. ``scheme`` is
    always ``"persistent"`` in this canonical block; transient
    (``non-persistent://``) topics are not supported by the in-memory
    broker (durability is implicit).

    Raises :class:`ValueError` on any malformed form.
    """
    if not isinstance(name, str):
        raise TypeError("topic name must be str")
    if not name:
        raise ValueError("topic name must be non-empty")
    scheme_marker = "://"
    if scheme_marker not in name:
        raise ValueError(
            f"topic {name!r} missing scheme marker {scheme_marker!r}"
        )
    scheme, body = name.split(scheme_marker, 1)
    if scheme != "persistent":
        raise ValueError(
            f"topic {name!r} scheme must be 'persistent', got {scheme!r}"
        )
    parts = body.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"topic {name!r} must be persistent://tenant/namespace/topic"
        )
    tenant, namespace, topic = parts
    for component, label in (
        (tenant, "tenant"),
        (namespace, "namespace"),
        (topic, "topic"),
    ):
        if not component:
            raise ValueError(f"topic {name!r} {label} must be non-empty")
        if any(ch.isspace() for ch in component):
            raise ValueError(
                f"topic {name!r} {label} must not contain whitespace"
            )
        if "/" in component:
            raise ValueError(
                f"topic {name!r} {label} must not contain '/'"
            )
    return ("persistent", tenant, namespace, topic)


def qualified_topic(tenant: str, namespace: str, topic: str) -> str:
    """Build a ``persistent://<tenant>/<namespace>/<topic>`` string.

    Validates each component the same way :func:`parse_qualified_topic`
    does. Useful for tests + downstream callers that want to construct
    qualified names without manual string interpolation.
    """
    candidate = f"persistent://{tenant}/{namespace}/{topic}"
    parse_qualified_topic(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Topic — qualified multi-tenant partitioned channel.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Topic:
    """Qualified, partitioned, multi-tenant channel.

    ``name`` is the *fully qualified* ``persistent://tenant/namespace/
    topic`` identifier (one per DIX event type per the topic-per-event-
    type rule). ``num_partitions`` is the parallel-consumption fan-out
    factor; key-shared subscriptions assign partitions to consumers
    deterministically. ``retention_ns`` is advisory — the in-memory
    broker keeps every record for the lifetime of the process and
    ignores retention.

    # ADAPTED FROM: pulsar.Client.create_producer(topic=...) qualified
    # name grammar + ``num_partitions`` partition spec.
    """

    name: str
    num_partitions: int = 1
    retention_ns: int | None = None

    def __post_init__(self) -> None:
        # parse_qualified_topic raises with a precise message on any
        # malformed form.
        parse_qualified_topic(self.name)
        if not isinstance(self.num_partitions, int) or isinstance(
            self.num_partitions, bool
        ):
            raise TypeError("Topic.num_partitions must be int")
        if self.num_partitions <= 0:
            raise ValueError("Topic.num_partitions must be > 0")
        if self.retention_ns is not None and self.retention_ns <= 0:
            raise ValueError("Topic.retention_ns must be > 0 when set")

    @property
    def tenant(self) -> str:
        return parse_qualified_topic(self.name)[1]

    @property
    def namespace(self) -> str:
        return parse_qualified_topic(self.name)[2]

    @property
    def short_name(self) -> str:
        return parse_qualified_topic(self.name)[3]


# ---------------------------------------------------------------------------
# Subscription — durable cursor with first-class type semantics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Subscription:
    """Durable cursor over a topic with first-class subscription type.

    ``name`` is the subscription identifier (operator-stable; the
    cursor persists by it). ``topic_name`` is the qualified topic
    name the subscription consumes. ``type`` selects one of the four
    Pulsar subscription semantics. ``initial_position_earliest``
    starts the cursor at entry id ``0`` (Pulsar's
    ``InitialPosition.Earliest``); ``False`` skips existing records
    and reads only new ones (``InitialPosition.Latest``).

    # ADAPTED FROM: pulsar.Client.subscribe(subscription_name,
    # subscription_type, initial_position).
    """

    name: str
    topic_name: str
    type: SubscriptionType = SubscriptionType.EXCLUSIVE
    initial_position_earliest: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Subscription.name must be non-empty")
        # Topic name validation delegated to parse_qualified_topic.
        parse_qualified_topic(self.topic_name)
        if not isinstance(self.type, SubscriptionType):
            raise TypeError("Subscription.type must be SubscriptionType")


# ---------------------------------------------------------------------------
# MessageId — Pulsar's richer Kafka-offset analogue.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class MessageId:
    """Addressable identifier of a message in a partition log.

    Pulsar's ``MessageId`` is richer than a Kafka offset — it carries
    ``ledger_id`` (BookKeeper write-ahead group), ``entry_id``
    (position in ledger), ``partition`` (partition index), and
    ``batch_idx`` (position inside a producer-side batch).

    The in-memory broker uses ``ledger_id=0`` everywhere (a single
    synthetic ledger per partition), ``entry_id`` as the canonical
    offset, ``partition`` as the partition index, and ``batch_idx=0``
    (no producer-side batching in the in-memory broker).

    Ordered for deterministic iteration.

    # ADAPTED FROM: pulsar.MessageId(ledger_id, entry_id, partition,
    # batch_index).
    """

    ledger_id: int
    entry_id: int
    partition: int
    batch_idx: int = 0

    def __post_init__(self) -> None:
        for field_name in ("ledger_id", "entry_id", "partition", "batch_idx"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"MessageId.{field_name} must be int")
            if value < 0:
                raise ValueError(f"MessageId.{field_name} must be >= 0")


# ---------------------------------------------------------------------------
# ProducerRecord / Message — envelopes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProducerRecord:
    """Envelope handed to :meth:`InMemoryProducer.send`.

    ``value`` is opaque ``bytes`` — the producer never inspects payload
    semantics. ``key`` selects the partition deterministically via
    :func:`partition_for_key`. ``ts_ns`` is the event-time timestamp
    from the caller (NEVER wall-clock from inside the bus, INV-15).
    ``properties`` is a sorted tuple of ``(key, value)`` string pairs
    mirroring Pulsar's ``Message.properties`` map; we coerce to a
    sorted tuple to keep the wire shape byte-stable. ``event_time_ns``
    mirrors Pulsar's distinct "event time" field used for time-based
    indexing; defaults to ``ts_ns`` when omitted.

    # ADAPTED FROM: pulsar.Producer.send(content, partition_key,
    # event_timestamp, properties, sequence_id).
    """

    topic_name: str
    value: bytes
    key: bytes | None = None
    ts_ns: int = 0
    event_time_ns: int = 0
    properties: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        parse_qualified_topic(self.topic_name)
        if not isinstance(self.value, bytes):
            raise TypeError("ProducerRecord.value must be bytes")
        if self.key is not None and not isinstance(self.key, bytes):
            raise TypeError("ProducerRecord.key must be bytes or None")
        for field_name in ("ts_ns", "event_time_ns"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"ProducerRecord.{field_name} must be int")
            if value < 0:
                raise ValueError(f"ProducerRecord.{field_name} must be >= 0")
        if not isinstance(self.properties, tuple):
            raise TypeError("ProducerRecord.properties must be tuple")
        for entry in self.properties:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
            ):
                raise TypeError(
                    "ProducerRecord.properties entries must be (str, str) tuples"
                )


@dataclass(frozen=True, slots=True)
class Message:
    """Envelope handed back from :meth:`InMemoryConsumer.receive`.

    ``message_id`` is the broker-assigned identifier; ``value`` is the
    payload bytes. All fields are populated by the broker on append.

    # ADAPTED FROM: pulsar.Message (data, message_id, partition_key,
    # event_timestamp, properties).
    """

    topic_name: str
    message_id: MessageId
    value: bytes
    key: bytes | None
    ts_ns: int
    event_time_ns: int
    properties: tuple[tuple[str, str], ...]
    redelivery_count: int = 0


# ---------------------------------------------------------------------------
# PulsarConfig — bus-wide config value-object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PulsarConfig:
    """Bus-wide config envelope.

    ``service_url`` is advisory for the in-memory broker (no network).
    The real ``pulsar.Client`` consumes this field via the lazy seam.
    ``operation_timeout_ns`` mirrors Pulsar's per-call timeout.
    ``connection_timeout_ns`` is the initial-connect timeout.
    ``stats_interval_ns`` is the cadence at which the real client
    posts stats (advisory in-memory).

    # ADAPTED FROM: pulsar.Client(service_url, operation_timeout_seconds,
    # connection_timeout_ms, stats_interval_seconds).
    """

    service_url: str = "pulsar://localhost:6650"
    operation_timeout_ns: int = 30_000_000_000
    connection_timeout_ns: int = 10_000_000_000
    stats_interval_ns: int = 60_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.service_url, str) or not self.service_url:
            raise TypeError("PulsarConfig.service_url must be non-empty str")
        for field_name in (
            "operation_timeout_ns",
            "connection_timeout_ns",
            "stats_interval_ns",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"PulsarConfig.{field_name} must be int")
            if value <= 0:
                raise ValueError(f"PulsarConfig.{field_name} must be > 0")


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
    same partition. Matches the partitioning used by C-03 kafka_bus so
    callers can shadow-compare the two transports byte-for-byte.

    # ADAPTED FROM: pulsar.PartitionsRoutingMode.UseSinglePartition with
    # a hashing partition key router; BLAKE2b-8 chosen to keep the
    # import set minimal (stdlib :mod:`hashlib`).
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
# Reference serializer / deserializer.
# ---------------------------------------------------------------------------


def serialize_record(payload: Mapping[str, Any]) -> bytes:
    """Reference helper — canonical JSON encoder.

    ``json.dumps(payload, sort_keys=True, separators=(",", ":"))`` then
    UTF-8 encode. Byte-stable across insertion orders, matches the wire
    shape :mod:`orjson` produces for plain dict-of-scalars payloads.

    The bus itself never calls this — it stores opaque bytes — but
    tests and downstream callers use it to get a reproducible payload
    that can be shadow-compared against C-02 / C-03 / C-04 / C-05.
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
# InMemoryPulsarBroker — append-only multi-tenant partition log.
# ---------------------------------------------------------------------------


@dataclass
class _SubscriptionState:
    """Live subscription cursor + member roster.

    Owned by :class:`InMemoryPulsarBroker`; mutated only through
    consumer-side methods.

    * ``cursor[partition]`` — the contiguous-ack *floor*: smallest
      entry_id that has not yet been acked. fetch_for scans from this
      floor up, skipping ids that are already in ``pending_acks``
      (in-flight, awaiting ack) or ``acked`` (acked but not yet
      contiguous with the floor).
    * ``pending_acks[partition]`` — in-flight delivered-but-unacked
      entry_ids. Shared / Key_Shared rely on this so a second consumer
      can scan past entry_ids already handed to a first consumer.
    * ``acked[partition]`` — individually-acked entry_ids above the
      floor, awaiting contiguous collapse.
    * ``redelivery_counts[(partition, entry_id)]`` — per-message nack
      counter; surfaced on :class:`Message.redelivery_count`.

    ``consumers`` is the sorted tuple of consumer names currently
    attached to the subscription (drives Failover primary selection +
    Shared / Key_Shared partition fan-out).
    """

    subscription: Subscription
    cursor: dict[int, int] = field(default_factory=dict)
    pending_acks: dict[int, set[int]] = field(default_factory=dict)
    acked: dict[int, set[int]] = field(default_factory=dict)
    redelivery_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    consumers: tuple[str, ...] = ()


@dataclass
class InMemoryPulsarBroker:
    """In-process Pulsar-shaped multi-tenant broker.

    Holds an append-only log per ``(qualified_topic_name,
    partition_idx)`` and per-subscription cursors. Used by the lazy
    seam's default factory and by the deterministic :func:`run_app`
    simulator.

    The broker itself is mutable (it accumulates log entries) but is
    only ever touched through the immutable producer / consumer
    handles — callers never see the live ``_logs`` / ``_subscriptions``
    dicts. INV-15 byte-identical replay holds because every mutation
    is driven by a caller-supplied value-object ordering.
    """

    topics: tuple[Topic, ...]
    _logs: dict[tuple[str, int], list[Message]] = field(default_factory=dict)
    _topic_index: dict[str, Topic] = field(default_factory=dict)
    _subscriptions: dict[tuple[str, str], _SubscriptionState] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.topics, tuple):
            raise TypeError("InMemoryPulsarBroker.topics must be tuple")
        if not self.topics:
            raise ValueError("InMemoryPulsarBroker.topics must be non-empty")
        names: set[str] = set()
        for topic in self.topics:
            if not isinstance(topic, Topic):
                raise TypeError(
                    "InMemoryPulsarBroker.topics entries must be Topic"
                )
            if topic.name in names:
                raise ValueError(
                    f"InMemoryPulsarBroker.topics duplicate name {topic.name!r}"
                )
            names.add(topic.name)
            self._topic_index[topic.name] = topic
            for idx in range(topic.num_partitions):
                self._logs[(topic.name, idx)] = []

    # ----- producer-side -----

    def append(self, record: ProducerRecord) -> Message:
        topic = self._topic_index.get(record.topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {record.topic_name!r}")
        partition_idx = partition_for_key(record.key, topic.num_partitions)
        log = self._logs[(record.topic_name, partition_idx)]
        entry_id = len(log)
        message = Message(
            topic_name=record.topic_name,
            message_id=MessageId(
                ledger_id=0,
                entry_id=entry_id,
                partition=partition_idx,
                batch_idx=0,
            ),
            value=record.value,
            key=record.key,
            ts_ns=record.ts_ns,
            event_time_ns=record.event_time_ns or record.ts_ns,
            properties=record.properties,
            redelivery_count=0,
        )
        log.append(message)
        return message

    def append_at(
        self,
        topic_name: str,
        partition_idx: int,
        value: bytes,
        *,
        key: bytes | None,
        ts_ns: int,
        event_time_ns: int,
        properties: tuple[tuple[str, str], ...],
    ) -> Message:
        """Round-robin path — append onto a caller-chosen partition.

        Bypasses :func:`partition_for_key` for the ``key=None`` case so
        :class:`InMemoryProducer` can drive a per-topic round-robin
        counter without smuggling a synthetic key.
        """
        topic = self._topic_index.get(topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {topic_name!r}")
        if not 0 <= partition_idx < topic.num_partitions:
            raise ValueError(f"partition_idx out of range for {topic_name!r}")
        log = self._logs[(topic_name, partition_idx)]
        entry_id = len(log)
        message = Message(
            topic_name=topic_name,
            message_id=MessageId(
                ledger_id=0,
                entry_id=entry_id,
                partition=partition_idx,
                batch_idx=0,
            ),
            value=value,
            key=key,
            ts_ns=ts_ns,
            event_time_ns=event_time_ns or ts_ns,
            properties=properties,
            redelivery_count=0,
        )
        log.append(message)
        return message

    # ----- subscription / consumer-side -----

    def partitions_for(self, topic_name: str) -> int:
        topic = self._topic_index.get(topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {topic_name!r}")
        return topic.num_partitions

    def stream_length(self, topic_name: str, partition_idx: int) -> int:
        log = self._logs.get((topic_name, partition_idx))
        if log is None:
            raise KeyError(
                f"unknown partition ({topic_name!r}, {partition_idx})"
            )
        return len(log)

    def attach_subscription(
        self, subscription: Subscription, consumer_name: str
    ) -> _SubscriptionState:
        """Idempotent-by-name subscription attach.

        Creates the cursor on first attach. Subsequent attaches of the
        same ``(topic_name, subscription_name)`` add the consumer to
        the roster but do NOT reset the cursor. Validates the
        subscription contract (exclusive type rejects a second
        consumer).
        """
        if not isinstance(subscription, Subscription):
            raise TypeError("subscription must be Subscription")
        if not isinstance(consumer_name, str) or not consumer_name:
            raise ValueError("consumer_name must be non-empty str")
        topic = self._topic_index.get(subscription.topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {subscription.topic_name!r}")
        key = (subscription.topic_name, subscription.name)
        state = self._subscriptions.get(key)
        if state is None:
            initial_cursor = (
                {idx: 0 for idx in range(topic.num_partitions)}
                if subscription.initial_position_earliest
                else {
                    idx: len(self._logs[(subscription.topic_name, idx)])
                    for idx in range(topic.num_partitions)
                }
            )
            state = _SubscriptionState(
                subscription=subscription,
                cursor=initial_cursor,
                pending_acks={
                    idx: set() for idx in range(topic.num_partitions)
                },
                acked={
                    idx: set() for idx in range(topic.num_partitions)
                },
                redelivery_counts={},
                consumers=(consumer_name,),
            )
            self._subscriptions[key] = state
            return state
        # Validate subscription-type compatibility on re-attach.
        if state.subscription.type != subscription.type:
            raise ValueError(
                f"subscription {subscription.name!r} on "
                f"{subscription.topic_name!r} already exists with type "
                f"{state.subscription.type.value!r}; cannot re-attach "
                f"with {subscription.type.value!r}"
            )
        if (
            state.subscription.type == SubscriptionType.EXCLUSIVE
            and state.consumers
        ):
            raise ValueError(
                f"exclusive subscription {subscription.name!r} on "
                f"{subscription.topic_name!r} already has a consumer "
                f"({state.consumers[0]!r}); cannot attach {consumer_name!r}"
            )
        if consumer_name in state.consumers:
            return state
        state.consumers = tuple(sorted((*state.consumers, consumer_name)))
        return state

    def detach_subscription(
        self, topic_name: str, subscription_name: str, consumer_name: str
    ) -> None:
        """Remove ``consumer_name`` from the subscription roster.

        Cursor is preserved (durable subscription). Removes the
        subscription entirely if the last consumer detaches.
        """
        key = (topic_name, subscription_name)
        state = self._subscriptions.get(key)
        if state is None:
            return
        remaining = tuple(c for c in state.consumers if c != consumer_name)
        if not remaining:
            # keep cursor durable; just clear the roster.
            state.consumers = ()
            return
        state.consumers = remaining

    def consumer_owns(
        self,
        topic_name: str,
        subscription_name: str,
        consumer_name: str,
        partition_idx: int,
        entry_id: int,
        key: bytes | None,
    ) -> bool:
        """Routing oracle — does ``consumer_name`` own this delivery?

        Implements the four subscription-type semantics deterministically:

        * EXCLUSIVE — only the single consumer is owner.
        * SHARED — round-robin by ``entry_id mod len(consumers)``
          against the sorted consumer roster.
        * FAILOVER — only the lexicographically-smallest consumer
          owns; others are standby.
        * KEY_SHARED — BLAKE2b-8 of ``key`` mod ``len(consumers)`` —
          ``key=None`` lands on the smallest consumer
          (Pulsar's "no key" handling).
        """
        key_ = (topic_name, subscription_name)
        state = self._subscriptions.get(key_)
        if state is None or not state.consumers:
            return False
        sub_type = state.subscription.type
        if sub_type == SubscriptionType.EXCLUSIVE:
            return state.consumers == (consumer_name,)
        if sub_type == SubscriptionType.FAILOVER:
            return consumer_name == state.consumers[0]
        if sub_type == SubscriptionType.SHARED:
            idx = entry_id % len(state.consumers)
            return state.consumers[idx] == consumer_name
        if sub_type == SubscriptionType.KEY_SHARED:
            if key is None:
                return consumer_name == state.consumers[0]
            digest = hashlib.blake2b(key, digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % len(state.consumers)
            return state.consumers[idx] == consumer_name
        raise AssertionError(f"unreachable: {sub_type!r}")

    def fetch_for(
        self,
        topic_name: str,
        subscription_name: str,
        consumer_name: str,
        max_records: int,
    ) -> tuple[Message, ...]:
        """Pop up to ``max_records`` deliverable messages for a consumer.

        Iterates the subscription's per-partition cursors in
        partition-id order, advances the cursor across messages this
        consumer does NOT own (Shared / Failover / Key_Shared can
        skip), and yields the messages the consumer DOES own up to
        ``max_records``.

        Cursor advancement implies "delivered" — ack semantics still
        govern the *durable* cursor floor (see :meth:`acknowledge`).
        Re-delivery is driven by :meth:`negative_acknowledge` rewind.
        """
        if max_records <= 0:
            raise ValueError("max_records must be > 0")
        key = (topic_name, subscription_name)
        state = self._subscriptions.get(key)
        if state is None:
            raise KeyError(
                f"no subscription {subscription_name!r} on {topic_name!r}"
            )
        topic = self._topic_index.get(topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {topic_name!r}")
        out: list[Message] = []
        for partition_idx in range(topic.num_partitions):
            log = self._logs[(topic_name, partition_idx)]
            pending = state.pending_acks[partition_idx]
            acked = state.acked[partition_idx]
            position = state.cursor[partition_idx]
            while position < len(log) and len(out) < max_records:
                if position in pending or position in acked:
                    position += 1
                    continue
                msg = log[position]
                if self.consumer_owns(
                    topic_name,
                    subscription_name,
                    consumer_name,
                    partition_idx,
                    position,
                    msg.key,
                ):
                    out.append(
                        replace(
                            msg,
                            redelivery_count=state.redelivery_counts.get(
                                (partition_idx, position), 0
                            ),
                        )
                    )
                    pending.add(position)
                position += 1
            if len(out) >= max_records:
                break
        return tuple(out)

    def acknowledge(
        self,
        topic_name: str,
        subscription_name: str,
        message_id: MessageId,
        *,
        cumulative: bool = False,
    ) -> None:
        """Ack a delivery — individual or cumulative.

        Individual ack records the id; the durable cursor floor is
        the contiguous run of acked ids starting at ``0`` (Pulsar's
        ``acknowledge`` semantic for shared subscriptions).
        Cumulative ack (Pulsar's ``acknowledge_cumulative``) advances
        the floor to ``message_id.entry_id + 1`` directly and is only
        valid on Exclusive / Failover subscriptions.
        """
        key = (topic_name, subscription_name)
        state = self._subscriptions.get(key)
        if state is None:
            raise KeyError(
                f"no subscription {subscription_name!r} on {topic_name!r}"
            )
        if cumulative and state.subscription.type in (
            SubscriptionType.SHARED,
            SubscriptionType.KEY_SHARED,
        ):
            raise ValueError(
                f"cumulative ack rejected for {state.subscription.type.value!r}"
                " subscription"
            )
        partition_idx = message_id.partition
        if partition_idx not in state.cursor:
            raise KeyError(f"unknown partition {partition_idx}")
        if cumulative:
            # Floor advances to next entry; drop everything below the
            # new floor (in-flight, acked-but-uncollapsed, redelivery).
            new_floor = message_id.entry_id + 1
            state.cursor[partition_idx] = max(
                state.cursor[partition_idx], new_floor
            )
            state.pending_acks[partition_idx] = {
                eid
                for eid in state.pending_acks[partition_idx]
                if eid >= new_floor
            }
            state.acked[partition_idx] = {
                eid
                for eid in state.acked[partition_idx]
                if eid >= new_floor
            }
            state.redelivery_counts = {
                (p, e): c
                for (p, e), c in state.redelivery_counts.items()
                if not (p == partition_idx and e < new_floor)
            }
            return
        # Individual ack — remove from in-flight, mark acked, advance
        # the contiguous floor through any run of acked ids.
        eid = message_id.entry_id
        state.pending_acks[partition_idx].discard(eid)
        if eid >= state.cursor[partition_idx]:
            state.acked[partition_idx].add(eid)
        acked = state.acked[partition_idx]
        while state.cursor[partition_idx] in acked:
            acked.discard(state.cursor[partition_idx])
            state.cursor[partition_idx] += 1
        # Drop redelivery counts the floor has now passed.
        floor = state.cursor[partition_idx]
        state.redelivery_counts = {
            (p, e): c
            for (p, e), c in state.redelivery_counts.items()
            if not (p == partition_idx and e < floor)
        }

    def negative_acknowledge(
        self,
        topic_name: str,
        subscription_name: str,
        message_id: MessageId,
    ) -> None:
        """Negative-ack — rewinds the cursor so the message is re-delivered.

        Increments the ``(partition, entry_id)`` redelivery counter so
        the next :meth:`fetch_for` yields the message with the counter
        bumped. Removes the id from pending acks (it's no longer
        delivered).
        """
        key = (topic_name, subscription_name)
        state = self._subscriptions.get(key)
        if state is None:
            raise KeyError(
                f"no subscription {subscription_name!r} on {topic_name!r}"
            )
        partition_idx = message_id.partition
        if partition_idx not in state.cursor:
            raise KeyError(f"unknown partition {partition_idx}")
        eid = message_id.entry_id
        # Removing from pending makes the id eligible for re-scan;
        # fetch_for never advanced the cursor past this id (only acks
        # do that), so no rewind is needed.
        state.pending_acks[partition_idx].discard(eid)
        state.acked[partition_idx].discard(eid)
        rc_key = (partition_idx, eid)
        state.redelivery_counts[rc_key] = (
            state.redelivery_counts.get(rc_key, 0) + 1
        )

    def subscription_state(
        self, topic_name: str, subscription_name: str
    ) -> _SubscriptionState:
        key = (topic_name, subscription_name)
        state = self._subscriptions.get(key)
        if state is None:
            raise KeyError(
                f"no subscription {subscription_name!r} on {topic_name!r}"
            )
        return state


# ---------------------------------------------------------------------------
# InMemoryProducer — value-shape producer handle.
# ---------------------------------------------------------------------------


@dataclass
class InMemoryProducer:
    """Pure in-process producer.

    Mirrors :class:`pulsar.Producer`'s ``send`` / ``flush`` / ``close``
    surface synchronously. ``send`` returns the :class:`Message` that
    was appended (a shape ``pulsar.Producer.send_async``'s callback
    receives as its second arg).

    ``round_robin_counter`` drives ``key=None`` partition selection so
    callers don't have to maintain state externally. The counter is
    advanced per-topic, not globally — preserves byte-identical replay.
    """

    broker: InMemoryPulsarBroker
    config: PulsarConfig = field(default_factory=PulsarConfig)
    _rr_counters: dict[str, int] = field(default_factory=dict)
    _running: bool = True

    def send(
        self,
        topic_name: str,
        value: bytes,
        *,
        key: bytes | None = None,
        ts_ns: int = 0,
        event_time_ns: int = 0,
        properties: tuple[tuple[str, str], ...] = (),
    ) -> Message:
        if not self._running:
            raise RuntimeError("InMemoryProducer is closed")
        if key is None:
            rr = self._rr_counters.get(topic_name, 0)
            num_parts = self.broker.partitions_for(topic_name)
            partition_idx = rr % num_parts
            self._rr_counters[topic_name] = rr + 1
            return self.broker.append_at(
                topic_name,
                partition_idx,
                value,
                key=None,
                ts_ns=ts_ns,
                event_time_ns=event_time_ns,
                properties=properties,
            )
        record = ProducerRecord(
            topic_name=topic_name,
            value=value,
            key=key,
            ts_ns=ts_ns,
            event_time_ns=event_time_ns,
            properties=properties,
        )
        return self.broker.append(record)

    def flush(self) -> None:
        if not self._running:
            raise RuntimeError("InMemoryProducer is closed")

    def close(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# InMemoryConsumer — value-shape consumer handle.
# ---------------------------------------------------------------------------


@dataclass
class InMemoryConsumer:
    """Pure in-process consumer.

    Mirrors :class:`pulsar.Consumer`'s ``receive`` / ``acknowledge`` /
    ``acknowledge_cumulative`` / ``negative_acknowledge`` / ``close``
    surface synchronously. ``subscription`` is the durable cursor;
    ``consumer_name`` identifies this consumer within the subscription
    (drives Failover primary selection + Shared / Key_Shared
    fan-out).
    """

    broker: InMemoryPulsarBroker
    subscription: Subscription
    consumer_name: str = "default"
    config: PulsarConfig = field(default_factory=PulsarConfig)
    _running: bool = True

    def __post_init__(self) -> None:
        if not self.consumer_name:
            raise ValueError("InMemoryConsumer.consumer_name must be non-empty")
        self.broker.attach_subscription(self.subscription, self.consumer_name)

    def receive(self, max_records: int = 1) -> tuple[Message, ...]:
        if not self._running:
            raise RuntimeError("InMemoryConsumer is closed")
        return self.broker.fetch_for(
            self.subscription.topic_name,
            self.subscription.name,
            self.consumer_name,
            max_records,
        )

    def acknowledge(self, message: Message | MessageId) -> None:
        if not self._running:
            raise RuntimeError("InMemoryConsumer is closed")
        mid = message.message_id if isinstance(message, Message) else message
        self.broker.acknowledge(
            self.subscription.topic_name,
            self.subscription.name,
            mid,
            cumulative=False,
        )

    def acknowledge_cumulative(self, message: Message | MessageId) -> None:
        if not self._running:
            raise RuntimeError("InMemoryConsumer is closed")
        mid = message.message_id if isinstance(message, Message) else message
        self.broker.acknowledge(
            self.subscription.topic_name,
            self.subscription.name,
            mid,
            cumulative=True,
        )

    def negative_acknowledge(self, message: Message | MessageId) -> None:
        if not self._running:
            raise RuntimeError("InMemoryConsumer is closed")
        mid = message.message_id if isinstance(message, Message) else message
        self.broker.negative_acknowledge(
            self.subscription.topic_name,
            self.subscription.name,
            mid,
        )

    def close(self) -> None:
        self.broker.detach_subscription(
            self.subscription.topic_name,
            self.subscription.name,
            self.consumer_name,
        )
        self._running = False


# ---------------------------------------------------------------------------
# Lazy seam — gated activation of the real pulsar-client.
# ---------------------------------------------------------------------------


def pulsar_client_factory(config: PulsarConfig) -> Any:
    """Lazy seam — returns a real :class:`pulsar.Client` once gated.

    This function is intentionally NOT implemented in this PR. It is
    the canonical hook a future research-acceptance PR will fill in,
    after that PR documents the shadow-equivalence comparison between
    :class:`InMemoryPulsarBroker` / :class:`InMemoryProducer` /
    :class:`InMemoryConsumer` and ``pulsar.Client.create_producer`` /
    ``pulsar.Client.subscribe`` against the same byte-identical replay
    stream — covering all four subscription types, multi-tenant
    routing, and the cumulative-ack / individual-ack / nack semantics.

    Until then, calling it raises :class:`NotImplementedError`. The
    module-level :data:`NEW_PIP_DEPENDENCIES` declares the gated
    packages so the canonical pin-set is complete.
    """
    if not isinstance(config, PulsarConfig):
        raise TypeError("config must be PulsarConfig")
    raise NotImplementedError(
        "pulsar_client_factory is a lazy seam; activation gated on "
        "research-acceptance PR documenting shadow-equivalence vs. "
        "InMemoryPulsarBroker"
    )


# ---------------------------------------------------------------------------
# App / run_app — deterministic in-process simulator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class App:
    """Pure value-object spec of a Pulsar bus.

    Mirrors the C-01 / C-02 / C-03 / C-05 ``App`` shape — registering
    a topic / subscription returns a *new* :class:`App` so the spec
    stays a builder. :func:`run_app` materialises one over an
    :class:`InMemoryPulsarBroker`.
    """

    topics: tuple[Topic, ...] = ()
    subscriptions: tuple[Subscription, ...] = ()
    config: PulsarConfig = field(default_factory=PulsarConfig)

    def with_topic(self, topic: Topic) -> App:
        if not isinstance(topic, Topic):
            raise TypeError("topic must be Topic")
        for existing in self.topics:
            if existing.name == topic.name:
                raise ValueError(f"duplicate topic {topic.name!r}")
        return replace(self, topics=(*self.topics, topic))

    def with_subscription(self, subscription: Subscription) -> App:
        if not isinstance(subscription, Subscription):
            raise TypeError("subscription must be Subscription")
        # subscription.topic_name must reference a declared topic.
        if subscription.topic_name not in {t.name for t in self.topics}:
            raise ValueError(
                f"subscription {subscription.name!r} references unknown topic "
                f"{subscription.topic_name!r}"
            )
        for existing in self.subscriptions:
            if (
                existing.name == subscription.name
                and existing.topic_name == subscription.topic_name
            ):
                raise ValueError(
                    f"duplicate subscription {subscription.name!r} on "
                    f"{subscription.topic_name!r}"
                )
        return replace(
            self,
            subscriptions=(*self.subscriptions, subscription),
        )


@dataclass(frozen=True, slots=True)
class DeliveredRecord:
    """A subscription delivery emitted by :func:`run_app`.

    ``subscription_name`` identifies the durable cursor that received
    the message; ``consumer_name`` identifies which Shared / Key_Shared
    / Failover consumer was the owner. ``message`` is the canonical
    :class:`Message` envelope.
    """

    subscription_name: str
    consumer_name: str
    message: Message


@dataclass(frozen=True, slots=True)
class AppResult:
    """Outcome of :func:`run_app`.

    ``records`` is the full ordered list of subscription deliveries,
    canonical-sorted by ``(subscription_name asc, message.topic_name
    asc, message.message_id asc)``. ``app_digest`` is a BLAKE2b-16 hex
    digest over the sorted record bytes, suitable for INV-15 3-run
    replay equality assertions.
    """

    records: tuple[DeliveredRecord, ...]
    app_digest: str


def bus_digest(records: Iterable[DeliveredRecord]) -> str:
    """BLAKE2b-16 digest over the canonical-sorted delivery stream.

    Useful as a stand-alone replay-equality fingerprint when callers
    want to compare two runs without invoking :func:`run_app`.
    """
    h = hashlib.blake2b(digest_size=16)
    for rec in sorted(
        records,
        key=lambda r: (
            r.subscription_name,
            r.message.topic_name,
            r.message.message_id,
        ),
    ):
        h.update(rec.subscription_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(rec.consumer_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(rec.message.topic_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(rec.message.message_id.partition.to_bytes(4, "big"))
        h.update(rec.message.message_id.entry_id.to_bytes(8, "big"))
        h.update(len(rec.message.value).to_bytes(8, "big"))
        h.update(rec.message.value)
        h.update(b"\x00" if rec.message.key is None else b"\x01")
        if rec.message.key is not None:
            h.update(len(rec.message.key).to_bytes(8, "big"))
            h.update(rec.message.key)
        h.update(rec.message.ts_ns.to_bytes(8, "big"))
        h.update(rec.message.event_time_ns.to_bytes(8, "big"))
        for pk, pv in rec.message.properties:
            h.update(pk.encode("utf-8"))
            h.update(b"\x00")
            h.update(pv.encode("utf-8"))
            h.update(b"\x00")
        h.update(rec.message.redelivery_count.to_bytes(4, "big"))
    return h.hexdigest()


def run_app(
    app: App,
    inbound: Iterable[ProducerRecord],
) -> AppResult:
    """Drain ``inbound`` through ``app`` deterministically.

    For each subscription declared on ``app``, attach a single
    canonical consumer named ``"run_app"``. Publish every inbound
    record. After every record, drain every subscription
    (``receive(max_records=large)``) and auto-acknowledge each delivery
    (cumulative if subscription type is Exclusive / Failover, else
    individual). Each delivery becomes a :class:`DeliveredRecord` in
    the result.

    Returns :class:`AppResult` with records canonical-sorted and
    digest pinned for replay-equality assertions.
    """
    if not isinstance(app, App):
        raise TypeError("app must be App")
    if not app.topics:
        raise ValueError("App.topics must be non-empty for run_app")
    broker = InMemoryPulsarBroker(topics=app.topics)
    for sub in app.subscriptions:
        broker.attach_subscription(sub, "run_app")
    delivered: list[DeliveredRecord] = []
    for record in inbound:
        if not isinstance(record, ProducerRecord):
            raise TypeError("inbound records must be ProducerRecord")
        broker.append(record)
        for sub in app.subscriptions:
            batch = broker.fetch_for(
                sub.topic_name, sub.name, "run_app", max_records=1024
            )
            for msg in batch:
                delivered.append(
                    DeliveredRecord(
                        subscription_name=sub.name,
                        consumer_name="run_app",
                        message=msg,
                    )
                )
                if sub.type in (
                    SubscriptionType.EXCLUSIVE,
                    SubscriptionType.FAILOVER,
                ):
                    broker.acknowledge(
                        sub.topic_name,
                        sub.name,
                        msg.message_id,
                        cumulative=True,
                    )
                else:
                    broker.acknowledge(
                        sub.topic_name,
                        sub.name,
                        msg.message_id,
                        cumulative=False,
                    )
    sorted_records = tuple(
        sorted(
            delivered,
            key=lambda r: (
                r.subscription_name,
                r.message.topic_name,
                r.message.message_id,
            ),
        )
    )
    return AppResult(records=sorted_records, app_digest=bus_digest(sorted_records))


# ---------------------------------------------------------------------------
# Cross-process worker bridge.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PulsarBusSentinel:
    """Sentinel posted on the worker's inbound queue to terminate it."""

    reason: str = "shutdown"


def _pulsar_worker_loop(
    app: App,
    in_q: Any,
    out_q: Any,
    poll_timeout_s: float,
) -> None:
    """Cross-process worker body. Drains ``in_q`` until a sentinel.

    Worker maintains its own :class:`InMemoryPulsarBroker` over the
    canonical app topics + subscriptions. Each :class:`ProducerRecord`
    is appended and the resulting :class:`Message` is posted to
    ``out_q``. Any subscriptions are drained after each publish and
    every :class:`DeliveredRecord` is also posted to ``out_q`` so the
    parent can pin the full delivery stream.
    """
    broker = InMemoryPulsarBroker(topics=app.topics)
    for sub in app.subscriptions:
        broker.attach_subscription(sub, "worker")
    while True:
        try:
            item = in_q.get(timeout=poll_timeout_s)
        except Empty:
            continue
        if isinstance(item, PulsarBusSentinel):
            out_q.put(item)
            return
        if not isinstance(item, ProducerRecord):
            out_q.put(
                ValueError(
                    f"worker received non-ProducerRecord: {type(item).__name__}"
                )
            )
            return
        msg = broker.append(item)
        out_q.put(msg)
        for sub in app.subscriptions:
            batch = broker.fetch_for(
                sub.topic_name, sub.name, "worker", max_records=1024
            )
            for delivery in batch:
                out_q.put(
                    DeliveredRecord(
                        subscription_name=sub.name,
                        consumer_name="worker",
                        message=delivery,
                    )
                )
                if sub.type in (
                    SubscriptionType.EXCLUSIVE,
                    SubscriptionType.FAILOVER,
                ):
                    broker.acknowledge(
                        sub.topic_name,
                        sub.name,
                        delivery.message_id,
                        cumulative=True,
                    )
                else:
                    broker.acknowledge(
                        sub.topic_name,
                        sub.name,
                        delivery.message_id,
                        cumulative=False,
                    )


def spawn_pulsar_worker(
    app: App,
    *,
    poll_timeout_s: float = 0.05,
) -> tuple[Any, Any, Any]:
    """Spawn a ``multiprocessing`` worker draining produce records.

    Returns ``(process, inbound_queue, outbound_queue)``. Producers
    feed :class:`ProducerRecord` instances into the inbound queue;
    the worker appends each to its own :class:`InMemoryPulsarBroker`
    and posts the resulting :class:`Message` back on the outbound
    queue, followed by any :class:`DeliveredRecord` produced by the
    subscriptions. Terminate by posting a :class:`PulsarBusSentinel`.

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
        target=_pulsar_worker_loop,
        args=(app, in_q, out_q, poll_timeout_s),
        daemon=False,
    )
    process.start()
    return process, in_q, out_q
