"""C-05 nats-py — Lightweight Subject-Based Pub/Sub.

# ADAPTED FROM: nats-io/nats.py — ``nats/aio/client.py``
# (``Client.connect`` / ``publish`` / ``subscribe``) and
# ``nats/js/client.py`` (JetStream ``add_stream``, durable consumer
# ``pull_subscribe`` + ack/nak/term).
#
# Tier: OFFLINE_ONLY — this module provides a deterministic
# in-process mirror of NATS core pub/sub + JetStream semantics
# suitable for cross-engine messaging *within one host*. The real
# ``nats`` / ``nats-py`` PyPI packages are NEVER imported in this
# module; the lazy seams :func:`nats_client_factory` /
# :func:`jetstream_factory` raise :class:`NotImplementedError`
# until a future research-acceptance PR documents the
# shadow-equivalence comparison vs. the real client.
#
# Surface mirrors :mod:`system_engine.streaming.kafka_bus` so
# production callers can swap transports — the same opaque
# ``bytes`` payload travels through ``send`` / ``publish`` on
# either bus. The CEP differentiator over Kafka is **subject
# wildcards**: NATS subjects can be matched by ``*`` (single
# token) or ``>`` (rest-of-subject), and the consumer-side
# routing here implements the same tree-walk semantics.
#
# Authority discipline:
#
# * **B27 / B28 / INV-71** — this module never calls
#   ``PatchProposal(...)``, ``HazardEvent(...)``,
#   ``SignalEvent(...)``, ``ExecutionEvent(...)`` or
#   ``SystemEvent(...)``. Transport carries opaque ``bytes`` only.
# * **B1 isolation** — no imports from ``intelligence_engine`` /
#   ``execution_engine`` / ``governance_engine`` /
#   ``evolution_engine`` / ``learning_engine``.
#
# Determinism (INV-15):
#
# * No top-level imports of :mod:`time` / :mod:`datetime` /
#   :mod:`random` / :mod:`asyncio` / :mod:`os` / :mod:`nats` /
#   :mod:`numpy` / :mod:`torch` / :mod:`polars`.
# * All ordering uses caller-supplied event-time ``ts_ns`` or
#   in-process publish counters (monotone integers).
# * Frozen, slotted dataclasses everywhere.
# * BLAKE2b-16 ``bus_digest`` over canonical-sorted record stream
#   gives byte-identical replay equality.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing as _mp
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

NATS_BUS_VERSION: Final[int] = 1
"""Bumped on any wire-shape change to subject grammar / record
shape / digest."""

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("nats-py",)
"""PyPI packages activated by the lazy seams below. Declared so the
canonical pin-set is complete, but the package itself is NEVER
imported in this module.
"""


# ---------------------------------------------------------------------------
# Value objects — subjects, streams, configs, records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Subject:
    """A NATS subject — dot-separated tokens, no wildcards.

    Wildcards (``*`` and ``>``) are valid in *subscription patterns*
    (see :class:`SubscriptionPattern`) but NEVER in publish
    subjects. Tokens are ASCII and forbid ``.`` / ``*`` / ``>`` /
    whitespace.
    """

    name: str

    def __post_init__(self) -> None:
        _validate_publish_subject(self.name)


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class SubscriptionPattern:
    """A NATS subscription pattern — dot-separated tokens, may
    include ``*`` (single token) or ``>`` (rest-of-subject, must be
    the final token).
    """

    pattern: str

    def __post_init__(self) -> None:
        _validate_subscription_pattern(self.pattern)

    def matches(self, subject_name: str) -> bool:
        return _pattern_matches(self.pattern, subject_name)


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class JetStreamConfig:
    """JetStream stream configuration — persistence + retention.

    ``max_messages`` is the maximum number of messages retained per
    stream; 0 means unlimited (mirrors NATS server semantics).
    ``max_age_ns`` is age-based retention in nanoseconds; 0 means
    no age limit.
    """

    name: str
    subjects: tuple[str, ...]
    max_messages: int = 0
    max_age_ns: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "JetStreamConfig.name must be non-empty"
            )
        if not isinstance(self.subjects, tuple):
            raise TypeError(
                "JetStreamConfig.subjects must be tuple; "
                f"got {type(self.subjects).__name__}"
            )
        if not self.subjects:
            raise ValueError(
                "JetStreamConfig.subjects must be non-empty"
            )
        for s in self.subjects:
            _validate_subscription_pattern(s)
        if self.max_messages < 0:
            raise ValueError(
                "JetStreamConfig.max_messages must be >= 0; "
                f"got {self.max_messages}"
            )
        if self.max_age_ns < 0:
            raise ValueError(
                "JetStreamConfig.max_age_ns must be >= 0; "
                f"got {self.max_age_ns}"
            )


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class DurableConsumerConfig:
    """JetStream durable consumer — durable name + filter subject.

    ``durable_name`` identifies the consumer across restarts.
    ``filter_pattern`` is a subscription pattern (wildcards
    allowed) restricting which stream subjects this consumer
    sees. ``ack_wait_ns`` is the redelivery timeout (0 means
    no redelivery — single-shot ack).
    """

    durable_name: str
    filter_pattern: str
    ack_wait_ns: int = 0

    def __post_init__(self) -> None:
        if not self.durable_name:
            raise ValueError(
                "DurableConsumerConfig.durable_name must be "
                "non-empty"
            )
        _validate_subscription_pattern(self.filter_pattern)
        if self.ack_wait_ns < 0:
            raise ValueError(
                "DurableConsumerConfig.ack_wait_ns must be >= 0; "
                f"got {self.ack_wait_ns}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class NATSConfig:
    """Bus-wide configuration for the (future) real NATS client."""

    servers: tuple[str, ...] = ("nats://localhost:4222",)
    name: str = "dix-nats-client"
    max_reconnect_attempts: int = 60
    reconnect_time_wait_ns: int = 2_000_000_000
    connect_timeout_ns: int = 5_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.servers, tuple):
            raise TypeError(
                "NATSConfig.servers must be tuple; "
                f"got {type(self.servers).__name__}"
            )
        if not self.servers:
            raise ValueError(
                "NATSConfig.servers must be non-empty"
            )
        for s in self.servers:
            if not isinstance(s, str) or not s:
                raise ValueError(
                    "NATSConfig.servers entries must be "
                    "non-empty strings"
                )
        if not self.name:
            raise ValueError(
                "NATSConfig.name must be non-empty"
            )
        if self.max_reconnect_attempts < 0:
            raise ValueError(
                "NATSConfig.max_reconnect_attempts must be >= 0"
            )
        if self.reconnect_time_wait_ns <= 0:
            raise ValueError(
                "NATSConfig.reconnect_time_wait_ns must be "
                "positive"
            )
        if self.connect_timeout_ns <= 0:
            raise ValueError(
                "NATSConfig.connect_timeout_ns must be positive"
            )


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class PublishRecord:
    """Envelope handed to :meth:`InMemoryNATSClient.publish`."""

    subject_name: str
    value: bytes
    ts_ns: int
    reply_subject: str = ""
    headers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_publish_subject(self.subject_name)
        if not isinstance(self.value, bytes):
            raise TypeError(
                "PublishRecord.value must be bytes; "
                f"got {type(self.value).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                "PublishRecord.ts_ns must be >= 0; "
                f"got {self.ts_ns}"
            )
        if self.reply_subject:
            _validate_publish_subject(self.reply_subject)
        if not isinstance(self.headers, tuple):
            raise TypeError(
                "PublishRecord.headers must be tuple; "
                f"got {type(self.headers).__name__}"
            )
        for h in self.headers:
            if not isinstance(h, tuple) or len(h) != 2:
                raise TypeError(
                    "PublishRecord.headers entries must be "
                    "(str, str) tuples"
                )


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class DeliveredRecord:
    """Envelope yielded by consumer-side :meth:`fetch`/iteration.

    ``seq`` is the stream-wide sequence number assigned by the
    in-process JetStream mirror at publish time. ``deliveries`` is
    the redelivery count (0 on first delivery, incremented on
    each NAK + redeliver cycle).
    """

    stream_name: str
    consumer_name: str
    subject_name: str
    seq: int
    value: bytes
    ts_ns: int
    deliveries: int

    def __post_init__(self) -> None:
        if not self.stream_name:
            raise ValueError(
                "DeliveredRecord.stream_name must be non-empty"
            )
        if not self.consumer_name:
            raise ValueError(
                "DeliveredRecord.consumer_name must be non-empty"
            )
        _validate_publish_subject(self.subject_name)
        if self.seq < 1:
            raise ValueError(
                "DeliveredRecord.seq must be >= 1; "
                f"got {self.seq}"
            )
        if not isinstance(self.value, bytes):
            raise TypeError(
                "DeliveredRecord.value must be bytes; "
                f"got {type(self.value).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                "DeliveredRecord.ts_ns must be >= 0"
            )
        if self.deliveries < 0:
            raise ValueError(
                "DeliveredRecord.deliveries must be >= 0; "
                f"got {self.deliveries}"
            )


# ---------------------------------------------------------------------------
# Pure utility functions — subject grammar, serialization, digest
# ---------------------------------------------------------------------------


def serialize_record(payload: Mapping[str, object]) -> bytes:
    """Byte-stable JSON codec — same shape as kafka_bus / faust_bus."""
    if not isinstance(payload, Mapping):
        raise TypeError(
            "serialize_record requires a Mapping; "
            f"got {type(payload).__name__}"
        )
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def deserialize_record(blob: bytes) -> dict[str, object]:
    if not isinstance(blob, bytes):
        raise TypeError(
            "deserialize_record requires bytes; "
            f"got {type(blob).__name__}"
        )
    out = json.loads(blob.decode("utf-8"))
    if not isinstance(out, dict):
        raise TypeError(
            "deserialize_record only round-trips dict payloads"
        )
    return out


def bus_digest(records: Iterable[DeliveredRecord]) -> str:
    """Stable BLAKE2b-16 hex over canonical-sorted delivered records.

    Sort order: ``(stream_name asc, consumer_name asc, seq asc)``.
    Each record contributes
    ``stream | b"\\x1f" | consumer | b"\\x1f" | subject | b"\\x1f" |
    seq | b"\\x1f" | ts_ns | b"\\x1f" | deliveries | b"\\x1f" |
    value | b"\\x1e"`` to the hash.
    """
    h = hashlib.blake2b(digest_size=16)
    ordered = sorted(
        list(records),
        key=lambda r: (r.stream_name, r.consumer_name, r.seq),
    )
    for r in ordered:
        h.update(r.stream_name.encode("utf-8"))
        h.update(b"\x1f")
        h.update(r.consumer_name.encode("utf-8"))
        h.update(b"\x1f")
        h.update(r.subject_name.encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(r.seq).encode("ascii"))
        h.update(b"\x1f")
        h.update(str(r.ts_ns).encode("ascii"))
        h.update(b"\x1f")
        h.update(str(r.deliveries).encode("ascii"))
        h.update(b"\x1f")
        h.update(r.value)
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Subject grammar — validation + wildcard matching
# ---------------------------------------------------------------------------


_FORBIDDEN_TOKEN_CHARS = frozenset(" \t\r\n.")


def _validate_publish_subject(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(
            "Subject must be str; "
            f"got {type(name).__name__}"
        )
    if not name:
        raise ValueError("Subject must be non-empty")
    tokens = name.split(".")
    for tok in tokens:
        if not tok:
            raise ValueError(
                f"Subject {name!r} has empty token"
            )
        if tok in ("*", ">"):
            raise ValueError(
                "Publish subjects may not contain wildcards; "
                f"got {name!r}"
            )
        for ch in tok:
            if ch in _FORBIDDEN_TOKEN_CHARS:
                raise ValueError(
                    f"Subject token {tok!r} contains "
                    f"forbidden char {ch!r}"
                )


def _validate_subscription_pattern(pattern: str) -> None:
    if not isinstance(pattern, str):
        raise TypeError(
            "Subscription pattern must be str; "
            f"got {type(pattern).__name__}"
        )
    if not pattern:
        raise ValueError(
            "Subscription pattern must be non-empty"
        )
    tokens = pattern.split(".")
    for i, tok in enumerate(tokens):
        if not tok:
            raise ValueError(
                f"Pattern {pattern!r} has empty token"
            )
        if tok == ">":
            if i != len(tokens) - 1:
                raise ValueError(
                    f"'>' must be the final token in pattern; "
                    f"got {pattern!r}"
                )
            continue
        if tok == "*":
            continue
        for ch in tok:
            if ch in _FORBIDDEN_TOKEN_CHARS:
                raise ValueError(
                    f"Pattern token {tok!r} contains "
                    f"forbidden char {ch!r}"
                )


def _pattern_matches(pattern: str, subject_name: str) -> bool:
    p_tokens = pattern.split(".")
    s_tokens = subject_name.split(".")
    for i, p_tok in enumerate(p_tokens):
        if p_tok == ">":
            return i < len(s_tokens)
        if i >= len(s_tokens):
            return False
        if p_tok == "*":
            continue
        if p_tok != s_tokens[i]:
            return False
    return len(p_tokens) == len(s_tokens)


# ---------------------------------------------------------------------------
# InMemoryNATSClient — core pub/sub
# ---------------------------------------------------------------------------


class InMemoryNATSClient:
    """In-process deterministic mirror of :class:`nats.aio.Client`.

    Subscribers register *callbacks* by subscription pattern;
    :meth:`publish` synchronously dispatches each matching
    subscriber's callback in registration order.

    No threads, no event loop — callbacks run inline. This matches
    NATS core (non-JetStream) semantics where a publish without
    subscribers is a no-op and message delivery is at-most-once.
    """

    __slots__ = ("_subscriptions", "_publish_counter")

    def __init__(self) -> None:
        self._subscriptions: list[
            tuple[SubscriptionPattern, object]
        ] = []
        self._publish_counter: int = 0

    def subscribe(
        self,
        pattern: str | SubscriptionPattern,
        callback: object,
    ) -> SubscriptionPattern:
        if isinstance(pattern, str):
            pattern = SubscriptionPattern(pattern)
        elif not isinstance(pattern, SubscriptionPattern):
            raise TypeError(
                "subscribe pattern must be str or "
                "SubscriptionPattern"
            )
        if not callable(callback):
            raise TypeError(
                "subscribe callback must be callable"
            )
        self._subscriptions.append((pattern, callback))
        return pattern

    def unsubscribe(
        self, pattern: SubscriptionPattern, callback: object
    ) -> bool:
        for i, (p, c) in enumerate(self._subscriptions):
            if p == pattern and c is callback:
                del self._subscriptions[i]
                return True
        return False

    def publish(self, record: PublishRecord) -> int:
        """Dispatch ``record`` to every matching subscriber.

        Returns the number of subscribers that received the message.
        """
        if not isinstance(record, PublishRecord):
            raise TypeError(
                "publish requires PublishRecord; "
                f"got {type(record).__name__}"
            )
        self._publish_counter += 1
        delivered = 0
        for pattern, callback in list(self._subscriptions):
            if pattern.matches(record.subject_name):
                callback(record)
                delivered += 1
        return delivered

    def publish_count(self) -> int:
        return self._publish_counter


# ---------------------------------------------------------------------------
# InMemoryJetStream — persistence + durable consumers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class _StreamMessage:
    seq: int
    subject_name: str
    value: bytes
    ts_ns: int


@dataclasses.dataclass(slots=True)
class _ConsumerState:
    config: DurableConsumerConfig
    delivered_seq: int = 0  # highest seq returned to a fetcher
    ack_floor_seq: int = 0  # highest contiguous acked seq
    pending: dict[int, int] = dataclasses.field(default_factory=dict)
    # pending: seq -> deliveries count


class InMemoryJetStream:
    """Persistent in-process mirror of NATS JetStream.

    Holds an append-only log per stream + durable consumer
    state (delivered_seq, ack_floor_seq, redelivery counters).
    Filters incoming messages against stream subjects on
    :meth:`publish` and against consumer filter patterns on
    :meth:`fetch`.
    """

    __slots__ = ("_streams", "_messages", "_consumers")

    def __init__(self) -> None:
        self._streams: dict[str, JetStreamConfig] = {}
        self._messages: dict[str, list[_StreamMessage]] = {}
        # consumers keyed by (stream_name, durable_name)
        self._consumers: dict[
            tuple[str, str], _ConsumerState
        ] = {}

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def add_stream(self, config: JetStreamConfig) -> None:
        if not isinstance(config, JetStreamConfig):
            raise TypeError(
                "add_stream requires JetStreamConfig; "
                f"got {type(config).__name__}"
            )
        if config.name in self._streams:
            raise ValueError(
                f"JetStream {config.name!r} already exists"
            )
        self._streams[config.name] = config
        self._messages[config.name] = []

    def stream_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._streams))

    def stream_length(self, stream_name: str) -> int:
        if stream_name not in self._messages:
            raise KeyError(stream_name)
        return len(self._messages[stream_name])

    def stream_messages(
        self, stream_name: str
    ) -> tuple[_StreamMessage, ...]:
        """Test-only inspector — returns the stream log."""
        if stream_name not in self._messages:
            raise KeyError(stream_name)
        return tuple(self._messages[stream_name])

    # ------------------------------------------------------------------
    # Consumer management
    # ------------------------------------------------------------------

    def add_consumer(
        self, stream_name: str, config: DurableConsumerConfig
    ) -> None:
        if stream_name not in self._streams:
            raise KeyError(
                f"Unknown JetStream {stream_name!r}"
            )
        if not isinstance(config, DurableConsumerConfig):
            raise TypeError(
                "add_consumer requires DurableConsumerConfig; "
                f"got {type(config).__name__}"
            )
        key = (stream_name, config.durable_name)
        if key in self._consumers:
            raise ValueError(
                f"Consumer {config.durable_name!r} already "
                f"exists on stream {stream_name!r}"
            )
        self._consumers[key] = _ConsumerState(config=config)

    def consumer_names(
        self, stream_name: str
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for (s, name) in self._consumers
                if s == stream_name
            )
        )

    def consumer_state(
        self, stream_name: str, durable_name: str
    ) -> tuple[int, int, tuple[int, ...]]:
        """Return ``(delivered_seq, ack_floor_seq, pending_seqs)``."""
        state = self._consumers[(stream_name, durable_name)]
        return (
            state.delivered_seq,
            state.ack_floor_seq,
            tuple(sorted(state.pending)),
        )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, record: PublishRecord) -> int:
        """Append ``record`` to every stream that matches its subject.

        Returns the count of streams that accepted the message.
        Applies per-stream retention (max_messages and max_age_ns).
        """
        if not isinstance(record, PublishRecord):
            raise TypeError(
                "publish requires PublishRecord; "
                f"got {type(record).__name__}"
            )
        accepted = 0
        for name, cfg in self._streams.items():
            matched = any(
                _pattern_matches(p, record.subject_name)
                for p in cfg.subjects
            )
            if not matched:
                continue
            log = self._messages[name]
            seq = len(log) + 1
            log.append(
                _StreamMessage(
                    seq=seq,
                    subject_name=record.subject_name,
                    value=record.value,
                    ts_ns=record.ts_ns,
                )
            )
            self._apply_retention(name, cfg, record.ts_ns)
            accepted += 1
        return accepted

    def _apply_retention(
        self,
        stream_name: str,
        cfg: JetStreamConfig,
        now_ns: int,
    ) -> None:
        log = self._messages[stream_name]
        if cfg.max_messages > 0 and len(log) > cfg.max_messages:
            drop = len(log) - cfg.max_messages
            del log[:drop]
        if cfg.max_age_ns > 0:
            cutoff = now_ns - cfg.max_age_ns
            keep_from = 0
            for i, m in enumerate(log):
                if m.ts_ns >= cutoff:
                    keep_from = i
                    break
                keep_from = i + 1
            if keep_from > 0:
                del log[:keep_from]

    # ------------------------------------------------------------------
    # Fetch / ack / nak / term
    # ------------------------------------------------------------------

    def fetch(
        self,
        stream_name: str,
        durable_name: str,
        *,
        batch: int = 1,
    ) -> tuple[DeliveredRecord, ...]:
        """Pull-subscribe — return up to ``batch`` unacked records.

        Each fetch advances the consumer's ``delivered_seq`` and
        increments per-seq delivery counters. Records remain in
        ``pending`` until :meth:`ack` / :meth:`term`.
        """
        if batch <= 0:
            raise ValueError(
                f"fetch batch must be positive; got {batch}"
            )
        key = (stream_name, durable_name)
        if key not in self._consumers:
            raise KeyError(key)
        state = self._consumers[key]
        log = self._messages[stream_name]
        out: list[DeliveredRecord] = []
        for msg in log:
            if msg.seq <= state.delivered_seq:
                continue
            if not _pattern_matches(
                state.config.filter_pattern, msg.subject_name
            ):
                continue
            deliveries = state.pending.get(msg.seq, 0) + 1
            state.pending[msg.seq] = deliveries
            state.delivered_seq = msg.seq
            out.append(
                DeliveredRecord(
                    stream_name=stream_name,
                    consumer_name=durable_name,
                    subject_name=msg.subject_name,
                    seq=msg.seq,
                    value=msg.value,
                    ts_ns=msg.ts_ns,
                    deliveries=deliveries,
                )
            )
            if len(out) >= batch:
                break
        return tuple(out)

    def ack(
        self,
        stream_name: str,
        durable_name: str,
        seq: int,
    ) -> bool:
        """Ack a single sequence. Advances ack_floor_seq when
        contiguous from the current floor.
        """
        state = self._consumers[(stream_name, durable_name)]
        if seq not in state.pending:
            return False
        del state.pending[seq]
        floor = state.ack_floor_seq
        # advance ack_floor_seq to the largest contiguous acked seq
        while (floor + 1) not in state.pending and (
            floor + 1
        ) <= state.delivered_seq:
            floor += 1
        state.ack_floor_seq = floor
        return True

    def nak(
        self,
        stream_name: str,
        durable_name: str,
        seq: int,
    ) -> bool:
        """Re-queue ``seq`` for redelivery on next fetch."""
        state = self._consumers[(stream_name, durable_name)]
        if seq not in state.pending:
            return False
        # rewind delivered_seq so the message is redelivered
        state.delivered_seq = min(
            state.delivered_seq, seq - 1
        )
        return True

    def term(
        self,
        stream_name: str,
        durable_name: str,
        seq: int,
    ) -> bool:
        """Permanently discard ``seq`` without ack-counting."""
        state = self._consumers[(stream_name, durable_name)]
        if seq not in state.pending:
            return False
        del state.pending[seq]
        floor = state.ack_floor_seq
        while (floor + 1) not in state.pending and (
            floor + 1
        ) <= state.delivered_seq:
            floor += 1
        state.ack_floor_seq = floor
        return True


# ---------------------------------------------------------------------------
# App — immutable builder for cross-process replay
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class App:
    """Immutable builder spec composed from streams + consumers.

    Mirrors the role of ``kafka_bus.App`` / ``faust_bus.App``:
    a value-object that the deterministic simulator
    :func:`run_app` consumes to produce a byte-stable
    :class:`AppResult`.
    """

    streams: tuple[JetStreamConfig, ...] = ()
    consumers: tuple[tuple[str, DurableConsumerConfig], ...] = ()

    def with_stream(self, config: JetStreamConfig) -> App:
        if not isinstance(config, JetStreamConfig):
            raise TypeError(
                "App.with_stream requires JetStreamConfig"
            )
        for s in self.streams:
            if s.name == config.name:
                raise ValueError(
                    f"App stream {config.name!r} already added"
                )
        return App(
            streams=self.streams + (config,),
            consumers=self.consumers,
        )

    def with_consumer(
        self, stream_name: str, config: DurableConsumerConfig
    ) -> App:
        if not any(s.name == stream_name for s in self.streams):
            raise ValueError(
                f"App.with_consumer references unknown stream "
                f"{stream_name!r}"
            )
        for sn, c in self.consumers:
            if sn == stream_name and c.durable_name == (
                config.durable_name
            ):
                raise ValueError(
                    f"Consumer {config.durable_name!r} already "
                    f"on stream {stream_name!r}"
                )
        return App(
            streams=self.streams,
            consumers=self.consumers
            + ((stream_name, config),),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AppResult:
    """Outcome of :func:`run_app`."""

    records: tuple[DeliveredRecord, ...]
    app_digest: str

    def __post_init__(self) -> None:
        if len(self.app_digest) != 32:
            raise ValueError(
                "AppResult.app_digest must be 32 hex chars"
            )


def run_app(
    app: App,
    inbound: Iterable[PublishRecord],
    *,
    batch: int = 16,
) -> AppResult:
    """Deterministic in-process simulator over ``app`` + ``inbound``.

    Publishes every inbound record into the JetStream, then drains
    each consumer with pull-fetch + auto-ack until no further
    messages are available. Returns the canonical-sorted delivered
    stream + a BLAKE2b-16 digest for INV-15 3-run replay.
    """
    if not isinstance(app, App):
        raise TypeError(
            "run_app requires App; "
            f"got {type(app).__name__}"
        )
    if batch <= 0:
        raise ValueError(
            f"run_app batch must be positive; got {batch}"
        )
    js = InMemoryJetStream()
    for cfg in app.streams:
        js.add_stream(cfg)
    for stream_name, consumer in app.consumers:
        js.add_consumer(stream_name, consumer)
    for record in inbound:
        js.publish(record)
    delivered: list[DeliveredRecord] = []
    for stream_name, consumer in app.consumers:
        while True:
            chunk = js.fetch(
                stream_name,
                consumer.durable_name,
                batch=batch,
            )
            if not chunk:
                break
            for d in chunk:
                js.ack(
                    stream_name, consumer.durable_name, d.seq
                )
            delivered.extend(chunk)
    return AppResult(
        records=tuple(delivered),
        app_digest=bus_digest(delivered),
    )


# ---------------------------------------------------------------------------
# Cross-process worker bridge
# ---------------------------------------------------------------------------


class NATSBusSentinel:
    """Sentinel terminating the cross-process worker loop."""

    __slots__ = ()


def _worker_main(
    app: App,
    inbound_q: _mp.Queue[object],
    outbound_q: _mp.Queue[object],
) -> None:
    js = InMemoryJetStream()
    for cfg in app.streams:
        js.add_stream(cfg)
    for stream_name, consumer in app.consumers:
        js.add_consumer(stream_name, consumer)
    while True:
        item = inbound_q.get()
        if isinstance(item, NATSBusSentinel):
            break
        if not isinstance(item, PublishRecord):
            outbound_q.put(
                TypeError(
                    "worker expected PublishRecord or "
                    f"sentinel; got {type(item).__name__}"
                )
            )
            continue
        js.publish(item)
        for stream_name, consumer in app.consumers:
            chunk = js.fetch(
                stream_name,
                consumer.durable_name,
                batch=64,
            )
            for d in chunk:
                js.ack(
                    stream_name, consumer.durable_name, d.seq
                )
                outbound_q.put(d)
    outbound_q.put(NATSBusSentinel())


def spawn_nats_worker(
    app: App,
) -> tuple[_mp.Process, _mp.Queue[object], _mp.Queue[object]]:
    """Spawn a worker process running the JetStream simulator.

    Returns ``(process, inbound_queue, outbound_queue)``. Send
    :class:`PublishRecord` instances to ``inbound_queue`` and read
    :class:`DeliveredRecord` instances off ``outbound_queue``. Send
    a :class:`NATSBusSentinel` to terminate; the worker echoes a
    final sentinel before exiting.
    """
    ctx = _mp.get_context("spawn")
    inbound_q: _mp.Queue[object] = ctx.Queue()
    outbound_q: _mp.Queue[object] = ctx.Queue()
    process = ctx.Process(
        target=_worker_main,
        args=(app, inbound_q, outbound_q),
    )
    process.start()
    return process, inbound_q, outbound_q


# ---------------------------------------------------------------------------
# Lazy seam factories
# ---------------------------------------------------------------------------


def nats_client_factory(config: NATSConfig) -> object:
    """Lazy seam to :class:`nats.aio.Client`.

    Implementation deferred to a future research-acceptance PR
    that documents shadow-equivalence vs. :class:`InMemoryNATSClient`
    (subject-wildcard semantics, queue groups, reconnect behaviour)
    and wire-compatibility of :func:`serialize_record` with at
    least one other DIX bus (kafka_bus / faust_bus / redis_store).
    """
    if not isinstance(config, NATSConfig):
        raise TypeError(
            "nats_client_factory config must be NATSConfig"
        )
    raise NotImplementedError(
        "Real nats.aio.Client activation is gated on a "
        "research-acceptance PR documenting shadow-equivalence "
        "vs. InMemoryNATSClient."
    )


def jetstream_factory(config: NATSConfig) -> object:
    """Lazy seam to :class:`nats.js.JetStreamContext`. Same gate."""
    if not isinstance(config, NATSConfig):
        raise TypeError(
            "jetstream_factory config must be NATSConfig"
        )
    raise NotImplementedError(
        "Real nats.js.JetStreamContext activation is gated on a "
        "research-acceptance PR documenting shadow-equivalence "
        "vs. InMemoryJetStream."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_records(records: Sequence[object]) -> None:
    for r in records:
        if not isinstance(r, DeliveredRecord):
            raise TypeError(
                "expected DeliveredRecord; "
                f"got {type(r).__name__}"
            )
