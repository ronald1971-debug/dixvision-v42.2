"""AuditLedgerCheckpointSaver — LangGraph checkpoints persisted to the audit ledger.

Wave-03 PR-2 (Dashboard-2026 cognitive plan §4.4).

LangGraph's default checkpoint backend is SQLite. For DIX VISION the
**only** durable record is the hash-chained authority ledger
(``GOV-CP-05``). Cognitive-graph state must therefore land in the
ledger, not in a side-channel database — otherwise a tamper of the
graph state could not be detected by the same chain that protects
governance decisions.

Design constraints:

* **B1 cross-engine isolation** — ``intelligence_engine.cognitive.*``
  cannot import ``governance_engine.*``. The saver receives a
  :data:`LedgerAppend` callable at construction; production wiring
  binds it to ``LedgerAuthorityWriter.append`` against the live
  ledger; tests bind a list-appender.
* **B24 cognitive scope** — only this package and
  ``evolution_engine.dyon`` may import ``langgraph``; the saver is
  the runtime use of that allowance.
* **Tamper evidence** — every checkpoint's serialized bytes are
  hashed with sha256, and that hash plus the checkpoint identity
  flow into the ledger payload. The ledger's own chain hash then
  binds the checkpoint into the same tamper-evident structure that
  protects governance rows.

The saver subclasses :class:`langgraph.checkpoint.memory.InMemorySaver`
so retrieval (``get_tuple`` / ``list``) keeps the fast in-memory
semantics LangGraph expects. The audit row is the *additional*
side-effect — never the primary store — because LangGraph polls
checkpoints synchronously inside the graph hot path.

Ledger row kinds emitted by this saver:

* ``COGNITIVE_CHECKPOINT`` — one per :meth:`put` call.
* ``COGNITIVE_CHECKPOINT_WRITES`` — one per :meth:`put_writes` call.

Both kinds carry stable string-only payload keys so the ledger's
``Mapping[str, str]`` contract is honoured without lossy coercion
of caller-provided types.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import InMemorySaver

__all__ = [
    "AuditLedgerCheckpointSaver",
    "CHECKPOINT_KIND",
    "CHECKPOINT_WRITES_KIND",
    "LedgerAppend",
]


CHECKPOINT_KIND = "COGNITIVE_CHECKPOINT"
"""Ledger ``kind`` for one :meth:`AuditLedgerCheckpointSaver.put` row."""

CHECKPOINT_WRITES_KIND = "COGNITIVE_CHECKPOINT_WRITES"
"""Ledger ``kind`` for one :meth:`AuditLedgerCheckpointSaver.put_writes` row."""


LedgerAppend = Callable[[str, Mapping[str, str]], None]
"""Zero-overhead callable that records one audit row.

Signature: ``(kind, payload) -> None``. Production wiring binds this
to ``lambda kind, payload: writer.append(ts_ns=time_source.now_ns(),
kind=kind, payload=payload)`` against
:class:`governance_engine.control_plane.ledger_authority_writer.LedgerAuthorityWriter`;
tests bind a list-appender.

Inverting the dependency keeps the cognitive package free of any
direct ``governance_engine`` import (B1)."""


def _payload_hash(serialized: bytes) -> str:
    return hashlib.sha256(serialized).hexdigest()


class AuditLedgerCheckpointSaver(InMemorySaver):
    """LangGraph checkpoint saver that mirrors every write to the audit ledger.

    Construction:

    * ``ledger_append`` — :data:`LedgerAppend`. The single seam
      through which audit rows reach the ledger.

    All other behaviour (storage, serde, ``get_tuple``, ``list``,
    ``delete_thread``) is inherited from :class:`InMemorySaver` and
    behaves identically.

    The saver is **not** a backwards-compat replacement for
    :class:`InMemorySaver` in legacy tests: the ``ledger_append``
    parameter is required and has no default, by design — silently
    swallowing audit rows would defeat the entire point of the
    saver.
    """

    def __init__(self, *, ledger_append: LedgerAppend) -> None:
        super().__init__()
        self._ledger_append = ledger_append

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint and emit one ``COGNITIVE_CHECKPOINT`` row.

        The audit row carries the checkpoint identity (``thread_id``,
        ``checkpoint_ns``, ``checkpoint_id``, optional
        ``parent_checkpoint_id``) plus the serializer name, the
        sha256 of the serialized checkpoint bytes, and the byte
        length. Storing the *hash* — not the bytes — keeps the
        ledger payload small while preserving tamper evidence.
        """

        result = super().put(config, checkpoint, metadata, new_versions)

        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        parent_id = configurable.get("checkpoint_id")

        ser_type, ser_bytes = self.serde.dumps_typed(checkpoint)
        source = metadata.get("source", "")
        step = metadata.get("step", "")

        payload: dict[str, str] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": str(checkpoint["id"]),
            "parent_checkpoint_id": str(parent_id) if parent_id else "",
            "serializer": ser_type,
            "payload_hash": _payload_hash(ser_bytes),
            "bytes_len": str(len(ser_bytes)),
            "source": str(source),
            "step": str(step),
        }
        self._ledger_append(CHECKPOINT_KIND, payload)
        return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist task writes and emit one ``COGNITIVE_CHECKPOINT_WRITES`` row.

        ``writes`` is materialised once before delegating to the
        in-memory store so the parent class and the audit hash see
        the same sequence — passing an iterator would let one of the
        two consumers see an empty sequence.
        """

        materialised: tuple[tuple[str, Any], ...] = tuple(writes)
        super().put_writes(config, materialised, task_id, task_path)

        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable.get("checkpoint_id", ""))

        ser_type, ser_bytes = self.serde.dumps_typed(list(materialised))
        payload: dict[str, str] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "task_path": task_path,
            "writes_count": str(len(materialised)),
            "serializer": ser_type,
            "payload_hash": _payload_hash(ser_bytes),
            "bytes_len": str(len(ser_bytes)),
        }
        self._ledger_append(CHECKPOINT_WRITES_KIND, payload)
