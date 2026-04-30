"""Unit tests for AuditLedgerCheckpointSaver (wave-03 PR-2).

These tests exercise the saver against a list-backed fake ledger so
they can run without dragging in ``governance_engine`` (B1: the
cognitive package must not import the governance engine — the seam
is the :data:`LedgerAppend` callable).

The MemorySaver parent gives us free regression coverage of
``get_tuple`` / ``list`` / ``delete_thread`` semantics; the tests
here pin the audit-row contract: every ``put`` emits exactly one
``COGNITIVE_CHECKPOINT`` row with the expected payload, every
``put_writes`` emits exactly one ``COGNITIVE_CHECKPOINT_WRITES``
row, payload_hash matches the sha256 of the serialized bytes, and
no governance-engine import sneaks in via the back door.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)

from intelligence_engine.cognitive.checkpointing import (
    AuditLedgerCheckpointSaver,
    LedgerAppend,
)
from intelligence_engine.cognitive.checkpointing.audit_ledger_checkpoint_saver import (
    CHECKPOINT_KIND,
    CHECKPOINT_WRITES_KIND,
)


def _empty_metadata() -> CheckpointMetadata:
    return CheckpointMetadata(source="input", step=0, parents={})


def _config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


class _RecordingLedger:
    """List-backed :data:`LedgerAppend` for assertion-friendly tests."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, str]]] = []

    def __call__(self, kind: str, payload: Mapping[str, str]) -> None:
        self.rows.append((kind, dict(payload)))


def _make_saver() -> tuple[AuditLedgerCheckpointSaver, _RecordingLedger]:
    ledger = _RecordingLedger()
    saver = AuditLedgerCheckpointSaver(ledger_append=ledger)
    return saver, ledger


def test_put_appends_exactly_one_checkpoint_row() -> None:
    saver, ledger = _make_saver()
    cfg = _config("t1")
    ck = empty_checkpoint()

    saver.put(cfg, ck, _empty_metadata(), {})

    assert len(ledger.rows) == 1
    kind, payload = ledger.rows[0]
    assert kind == CHECKPOINT_KIND
    assert payload["thread_id"] == "t1"
    assert payload["checkpoint_id"] == ck["id"]
    assert payload["parent_checkpoint_id"] == ""
    assert payload["checkpoint_ns"] == ""


def test_put_payload_hash_matches_serialized_bytes() -> None:
    saver, ledger = _make_saver()
    cfg = _config("t1")
    ck = empty_checkpoint()

    saver.put(cfg, ck, _empty_metadata(), {})

    _kind, payload = ledger.rows[0]
    ser_type, ser_bytes = saver.serde.dumps_typed(ck)
    expected_hash = hashlib.sha256(ser_bytes).hexdigest()
    assert payload["payload_hash"] == expected_hash
    assert payload["serializer"] == ser_type
    assert payload["bytes_len"] == str(len(ser_bytes))


def test_put_records_parent_checkpoint_id_when_provided() -> None:
    saver, ledger = _make_saver()
    parent_ck = empty_checkpoint()
    saver.put(_config("t1"), parent_ck, _empty_metadata(), {})

    child_ck = empty_checkpoint()
    saver.put(
        _config("t1", checkpoint_id=parent_ck["id"]),
        child_ck,
        _empty_metadata(),
        {},
    )

    assert len(ledger.rows) == 2
    _kind, second = ledger.rows[1]
    assert second["checkpoint_id"] == child_ck["id"]
    assert second["parent_checkpoint_id"] == parent_ck["id"]


def test_put_records_metadata_source_and_step() -> None:
    saver, ledger = _make_saver()
    cfg = _config("t1")
    ck = empty_checkpoint()
    metadata: CheckpointMetadata = {
        "source": "loop",
        "step": 7,
        "parents": {},
    }

    saver.put(cfg, ck, metadata, {})

    _kind, payload = ledger.rows[0]
    assert payload["source"] == "loop"
    assert payload["step"] == "7"


def test_put_writes_appends_exactly_one_writes_row() -> None:
    saver, ledger = _make_saver()
    ck = empty_checkpoint()
    saver.put(_config("t1"), ck, _empty_metadata(), {})
    ledger.rows.clear()

    cfg = _config("t1", checkpoint_id=ck["id"])
    saver.put_writes(cfg, [("ch", "v1"), ("ch", "v2")], task_id="task-A")

    assert len(ledger.rows) == 1
    kind, payload = ledger.rows[0]
    assert kind == CHECKPOINT_WRITES_KIND
    assert payload["thread_id"] == "t1"
    assert payload["checkpoint_id"] == ck["id"]
    assert payload["task_id"] == "task-A"
    assert payload["task_path"] == ""
    assert payload["writes_count"] == "2"


def test_put_writes_payload_hash_matches_serialized_bytes() -> None:
    saver, ledger = _make_saver()
    ck = empty_checkpoint()
    saver.put(_config("t1"), ck, _empty_metadata(), {})
    ledger.rows.clear()

    writes = [("ch", "v1"), ("ch", "v2")]
    saver.put_writes(
        _config("t1", checkpoint_id=ck["id"]),
        writes,
        task_id="task-A",
    )

    _kind, payload = ledger.rows[0]
    ser_type, ser_bytes = saver.serde.dumps_typed(list(writes))
    expected_hash = hashlib.sha256(ser_bytes).hexdigest()
    assert payload["payload_hash"] == expected_hash
    assert payload["serializer"] == ser_type
    assert payload["bytes_len"] == str(len(ser_bytes))


def test_put_writes_materialises_iterator_once() -> None:
    """A generator passed for ``writes`` must reach both the parent
    store and the ledger row — never just one of the two."""

    saver, ledger = _make_saver()
    ck = empty_checkpoint()
    saver.put(_config("t1"), ck, _empty_metadata(), {})
    ledger.rows.clear()

    def _gen() -> Any:
        yield ("ch", "v1")
        yield ("ch", "v2")

    saver.put_writes(
        _config("t1", checkpoint_id=ck["id"]),
        _gen(),  # type: ignore[arg-type]
        task_id="task-A",
    )

    _kind, payload = ledger.rows[0]
    assert payload["writes_count"] == "2"

    cfg = _config("t1", checkpoint_id=ck["id"])
    tup = saver.get_tuple(cfg)
    assert tup is not None
    assert len(tup.pending_writes) == 2


def test_get_tuple_returns_inserted_checkpoint() -> None:
    saver, _ledger = _make_saver()
    ck = empty_checkpoint()
    saver.put(_config("t1"), ck, _empty_metadata(), {})

    tup = saver.get_tuple(_config("t1"))
    assert tup is not None
    assert tup.checkpoint["id"] == ck["id"]


def test_list_returns_checkpoints_for_thread() -> None:
    saver, _ledger = _make_saver()
    ck1 = empty_checkpoint()
    saver.put(_config("t1"), ck1, _empty_metadata(), {})
    ck2 = empty_checkpoint()
    saver.put(
        _config("t1", checkpoint_id=ck1["id"]),
        ck2,
        _empty_metadata(),
        {},
    )

    listed = list(saver.list(_config("t1")))
    listed_ids = {tup.checkpoint["id"] for tup in listed}
    assert {ck1["id"], ck2["id"]} <= listed_ids


def test_threads_are_isolated_in_ledger_rows() -> None:
    saver, ledger = _make_saver()
    a = empty_checkpoint()
    b = empty_checkpoint()
    saver.put(_config("thread-A"), a, _empty_metadata(), {})
    saver.put(_config("thread-B"), b, _empty_metadata(), {})

    threads = [payload["thread_id"] for _kind, payload in ledger.rows]
    assert threads == ["thread-A", "thread-B"]


def test_payload_keys_are_all_strings() -> None:
    """The ledger contract is ``Mapping[str, str]``; numeric or
    None values must be coerced to ``str`` before they reach the
    ledger seam."""

    saver, ledger = _make_saver()
    ck = empty_checkpoint()
    metadata: CheckpointMetadata = {"source": "loop", "step": 9, "parents": {}}
    saver.put(_config("t1"), ck, metadata, {})

    for _kind, payload in ledger.rows:
        for k, v in payload.items():
            assert isinstance(k, str), k
            assert isinstance(v, str), (k, v)


def test_constructor_requires_keyword_only_ledger_append() -> None:
    with pytest.raises(TypeError):
        AuditLedgerCheckpointSaver()  # type: ignore[call-arg]


def test_module_does_not_import_governance_engine() -> None:
    """B1 isolation: the cognitive checkpointing module must not
    import ``governance_engine.*``. The seam is the ``LedgerAppend``
    callable — production wires it to the live ledger writer at
    construction time."""

    here = Path(__file__).resolve().parent.parent
    target = (
        here
        / "intelligence_engine"
        / "cognitive"
        / "checkpointing"
        / "audit_ledger_checkpoint_saver.py"
    )
    tree = ast.parse(target.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith(
                "governance_engine"
            ), f"forbidden cross-engine import: {node.module}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("governance_engine"), (
                    f"forbidden cross-engine import: {alias.name}"
                )


def test_ledger_append_type_is_callable_alias() -> None:
    """Smoke check that the public type alias is a callable
    annotation other modules can reuse — keeps test/wiring code
    aligned with the module's public surface."""

    spec = importlib.util.find_spec(
        "intelligence_engine.cognitive.checkpointing.audit_ledger_checkpoint_saver"
    )
    assert spec is not None
    assert LedgerAppend is not None


def test_serializer_name_round_trips_through_payload() -> None:
    saver, ledger = _make_saver()
    ck = empty_checkpoint()
    saver.put(_config("t1"), ck, _empty_metadata(), {})

    _kind, payload = ledger.rows[0]
    ser_type, _ser_bytes = saver.serde.dumps_typed(ck)
    # The serializer name appears verbatim so a forensic auditor can
    # pick the matching deserialiser when verifying the chain.
    assert payload["serializer"] == ser_type
