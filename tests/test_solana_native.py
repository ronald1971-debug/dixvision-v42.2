"""Tests for I-23 — Solana native stack adapter.

Coverage:

* Base58 codec — round-trip, leading-zero handling, invalid chars
* Value-object contracts (Pubkey / KeypairHandle / Instruction /
  Transaction / SignedTransaction / TxResult) — frozen, slotted,
  validation
* Adapter lifecycle (DISCONNECTED → READY → submit returns FILLED /
  REJECTED) — same operational-honesty contract as the other
  ``LiveAdapterBase`` adapters
* INV-15 byte-identical 3-run replay
* AST guardrails: no top-level vendor SDK imports, no clock / random /
  os.environ / asyncio / requests / urllib, no typed-event ctors
  (B27 / B28 / INV-71)
* Lazy seam: ``enable_solana_native_factory`` exists and only imports
  vendor SDKs inside its body
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib

import pytest

from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from execution_engine.adapters import solana_native as M
from execution_engine.adapters._live_base import AdapterState

# Two canonical 32-byte values (base58-encoded) used across tests.
_PUB_A = M.b58encode(b"A" * 32)
_PUB_B = M.b58encode(b"B" * 32)
_BLOCKHASH = M.b58encode(b"\x11" * 32)


def _make_signal(symbol: str = "SOL-USDC") -> SignalEvent:
    return SignalEvent(
        ts_ns=1_000_000_000,
        symbol=symbol,
        side=Side.BUY,
        confidence=0.75,
        produced_by_engine="intelligence",
    )


def _make_instructions(
    sig: SignalEvent, mark: float, fee_payer: M.Pubkey
) -> tuple[M.Instruction, ...]:
    return (
        M.Instruction(
            program_id=M.Pubkey(_PUB_A),
            accounts=(fee_payer, M.Pubkey(_PUB_B)),
            data=b"\x01" + sig.symbol.encode("utf-8"),
        ),
    )


def _make_transport(
    blockhash: str = _BLOCKHASH,
    status: str = "CONFIRMED",
) -> M.InProcessSolanaTransport:
    return M.InProcessSolanaTransport(
        recent_blockhash=blockhash,
        scripted_results=(
            M.TxResult(
                signature=M.b58encode(b"\x77" * 32),
                status=status,
                slot=12345,
                detail="ok",
            ),
        ),
    )


def _make_adapter(
    *,
    transport: M.SolanaTransport | None = None,
    default_qty: float = 0.5,
) -> M.SolanaNativeAdapter:
    tport = transport or _make_transport()
    return M.SolanaNativeAdapter(
        transport=tport,
        signer=M.deterministic_test_signer,
        keypair=M.KeypairHandle(
            credential_id="ops.solana.main",
            pubkey=M.Pubkey(_PUB_A),
        ),
        instruction_builder=_make_instructions,
        default_qty=default_qty,
    )


# ---------------------------------------------------------------------------
# 1. Module invariants
# ---------------------------------------------------------------------------


def test_module_declares_new_pip_dependencies() -> None:
    assert M.NEW_PIP_DEPENDENCIES == (
        "solana",
        "solders",
        "anchorpy",
        "base58",
    )


def test_module_exposes_public_surface() -> None:
    for name in (
        "Pubkey",
        "KeypairHandle",
        "Instruction",
        "Transaction",
        "SignedTransaction",
        "TxResult",
        "SolanaTransport",
        "InProcessSolanaTransport",
        "SolanaNativeAdapter",
        "b58encode",
        "b58decode",
        "deterministic_test_signer",
        "enable_solana_native_factory",
    ):
        assert hasattr(M, name), name


# ---------------------------------------------------------------------------
# 2. Base58 codec
# ---------------------------------------------------------------------------


def test_b58_roundtrip_random_bytes() -> None:
    for raw in (b"hello", b"\x00\x01\x02", b"\xff" * 16, b"abc123"):
        assert M.b58decode(M.b58encode(raw)) == raw


def test_b58_leading_zero_bytes_preserved() -> None:
    raw = b"\x00\x00\x00" + b"abc"
    encoded = M.b58encode(raw)
    assert encoded.startswith("111")
    assert M.b58decode(encoded) == raw


def test_b58_empty_input() -> None:
    assert M.b58encode(b"") == ""
    assert M.b58decode("") == b""


def test_b58encode_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        M.b58encode("not bytes")  # type: ignore[arg-type]


def test_b58decode_rejects_invalid_char() -> None:
    with pytest.raises(ValueError):
        M.b58decode("0OIl")  # all four are NOT in the alphabet


def test_b58decode_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        M.b58decode(b"binary")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Pubkey
# ---------------------------------------------------------------------------


def test_pubkey_happy_path() -> None:
    pk = M.Pubkey(_PUB_A)
    assert pk.as_bytes() == b"A" * 32


def test_pubkey_rejects_empty() -> None:
    with pytest.raises(ValueError):
        M.Pubkey("")


def test_pubkey_rejects_non_str() -> None:
    with pytest.raises(ValueError):
        M.Pubkey(123)  # type: ignore[arg-type]


def test_pubkey_rejects_wrong_length() -> None:
    short = M.b58encode(b"A" * 16)  # 16-byte payload — not 32
    with pytest.raises(ValueError):
        M.Pubkey(short)


def test_pubkey_is_frozen_and_slotted() -> None:
    pk = M.Pubkey(_PUB_A)
    assert dataclasses.is_dataclass(pk)
    assert M.Pubkey.__dataclass_params__.frozen is True
    assert "__slots__" in M.Pubkey.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        pk.value = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. KeypairHandle
# ---------------------------------------------------------------------------


def test_keypair_handle_happy_path() -> None:
    handle = M.KeypairHandle(
        credential_id="ops.solana.main",
        pubkey=M.Pubkey(_PUB_A),
    )
    assert handle.credential_id == "ops.solana.main"
    assert handle.pubkey.as_bytes() == b"A" * 32


def test_keypair_handle_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        M.KeypairHandle(credential_id="", pubkey=M.Pubkey(_PUB_A))


def test_keypair_handle_rejects_non_pubkey() -> None:
    with pytest.raises(TypeError):
        M.KeypairHandle(credential_id="ops.x", pubkey=_PUB_A)  # type: ignore[arg-type]


def test_keypair_handle_is_frozen_and_slotted() -> None:
    assert M.KeypairHandle.__dataclass_params__.frozen is True
    assert "__slots__" in M.KeypairHandle.__dict__


# ---------------------------------------------------------------------------
# 5. Instruction / Transaction / SignedTransaction
# ---------------------------------------------------------------------------


def test_instruction_happy_path() -> None:
    ins = M.Instruction(
        program_id=M.Pubkey(_PUB_A),
        accounts=(M.Pubkey(_PUB_B),),
        data=b"\x00\x01",
    )
    assert ins.data == b"\x00\x01"


def test_instruction_validates_program_id() -> None:
    with pytest.raises(TypeError):
        M.Instruction(program_id=_PUB_A, accounts=(), data=b"")  # type: ignore[arg-type]


def test_instruction_validates_accounts_tuple() -> None:
    with pytest.raises(TypeError):
        M.Instruction(
            program_id=M.Pubkey(_PUB_A),
            accounts=[M.Pubkey(_PUB_B)],  # type: ignore[arg-type]
            data=b"",
        )


def test_instruction_validates_account_type() -> None:
    with pytest.raises(TypeError):
        M.Instruction(
            program_id=M.Pubkey(_PUB_A),
            accounts=(_PUB_B,),  # type: ignore[arg-type]
            data=b"",
        )


def test_instruction_validates_data_type() -> None:
    with pytest.raises(TypeError):
        M.Instruction(
            program_id=M.Pubkey(_PUB_A),
            accounts=(),
            data="not bytes",  # type: ignore[arg-type]
        )


def test_transaction_happy_path() -> None:
    fee_payer = M.Pubkey(_PUB_A)
    tx = M.Transaction(
        fee_payer=fee_payer,
        recent_blockhash=_BLOCKHASH,
        instructions=(
            M.Instruction(
                program_id=M.Pubkey(_PUB_A),
                accounts=(fee_payer,),
                data=b"\x00",
            ),
        ),
    )
    msg = tx.to_message_bytes()
    assert isinstance(msg, bytes)
    assert len(msg) >= 64  # fee_payer 32 + blockhash 32 + rest


def test_transaction_rejects_empty_instructions() -> None:
    with pytest.raises(ValueError):
        M.Transaction(
            fee_payer=M.Pubkey(_PUB_A),
            recent_blockhash=_BLOCKHASH,
            instructions=(),
        )


def test_transaction_rejects_bad_blockhash() -> None:
    bad = M.b58encode(b"A" * 16)
    with pytest.raises(ValueError):
        M.Transaction(
            fee_payer=M.Pubkey(_PUB_A),
            recent_blockhash=bad,
            instructions=(
                M.Instruction(
                    program_id=M.Pubkey(_PUB_A),
                    accounts=(),
                    data=b"",
                ),
            ),
        )


def test_signed_transaction_rejects_wrong_sig_length() -> None:
    tx = M.Transaction(
        fee_payer=M.Pubkey(_PUB_A),
        recent_blockhash=_BLOCKHASH,
        instructions=(
            M.Instruction(
                program_id=M.Pubkey(_PUB_A),
                accounts=(),
                data=b"",
            ),
        ),
    )
    with pytest.raises(ValueError):
        M.SignedTransaction(
            transaction=tx,
            signature=b"\x00" * 32,
            signer_pubkey=M.Pubkey(_PUB_A),
        )


def test_signed_transaction_wire_bytes_includes_sig_and_message() -> None:
    tx = M.Transaction(
        fee_payer=M.Pubkey(_PUB_A),
        recent_blockhash=_BLOCKHASH,
        instructions=(
            M.Instruction(
                program_id=M.Pubkey(_PUB_A),
                accounts=(),
                data=b"",
            ),
        ),
    )
    sig = b"\x42" * 64
    s = M.SignedTransaction(
        transaction=tx,
        signature=sig,
        signer_pubkey=M.Pubkey(_PUB_A),
    )
    wire = s.to_wire_bytes()
    assert wire.startswith(sig)
    assert wire[64:] == tx.to_message_bytes()


# ---------------------------------------------------------------------------
# 6. TxResult
# ---------------------------------------------------------------------------


def test_tx_result_happy_path() -> None:
    r = M.TxResult(
        signature=M.b58encode(b"\x77" * 32),
        status="CONFIRMED",
        slot=10,
        detail="ok",
    )
    assert r.status == "CONFIRMED"


@pytest.mark.parametrize("status", ["CONFIRMED", "FAILED", "DROPPED"])
def test_tx_result_accepts_allowed_status(status: str) -> None:
    r = M.TxResult(
        signature=M.b58encode(b"\x77" * 32),
        status=status,
        slot=0,
        detail="",
    )
    assert r.status == status


def test_tx_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        M.TxResult(
            signature=M.b58encode(b"\x77" * 32),
            status="OOPS",
            slot=0,
            detail="",
        )


def test_tx_result_rejects_negative_slot() -> None:
    with pytest.raises(ValueError):
        M.TxResult(
            signature=M.b58encode(b"\x77" * 32),
            status="CONFIRMED",
            slot=-1,
            detail="",
        )


# ---------------------------------------------------------------------------
# 7. Deterministic test signer
# ---------------------------------------------------------------------------


def test_deterministic_signer_byte_identical() -> None:
    handle = M.KeypairHandle(
        credential_id="ops.x",
        pubkey=M.Pubkey(_PUB_A),
    )
    msg = b"some message bytes"
    a = M.deterministic_test_signer(msg, handle)
    b = M.deterministic_test_signer(msg, handle)
    assert a == b
    assert len(a) == 64


def test_deterministic_signer_changes_with_credential_id() -> None:
    msg = b"x"
    a = M.deterministic_test_signer(
        msg,
        M.KeypairHandle(credential_id="ops.a", pubkey=M.Pubkey(_PUB_A)),
    )
    b = M.deterministic_test_signer(
        msg,
        M.KeypairHandle(credential_id="ops.b", pubkey=M.Pubkey(_PUB_A)),
    )
    assert a != b


def test_deterministic_signer_rejects_non_bytes() -> None:
    handle = M.KeypairHandle(
        credential_id="ops.x",
        pubkey=M.Pubkey(_PUB_A),
    )
    with pytest.raises(TypeError):
        M.deterministic_test_signer("not bytes", handle)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. InProcessSolanaTransport
# ---------------------------------------------------------------------------


def test_in_process_transport_returns_blockhash() -> None:
    t = _make_transport()
    assert t.get_recent_blockhash() == _BLOCKHASH


def test_in_process_transport_returns_scripted_result() -> None:
    t = _make_transport(status="FAILED")
    tx = M.Transaction(
        fee_payer=M.Pubkey(_PUB_A),
        recent_blockhash=_BLOCKHASH,
        instructions=(
            M.Instruction(
                program_id=M.Pubkey(_PUB_A),
                accounts=(),
                data=b"",
            ),
        ),
    )
    signed = M.SignedTransaction(
        transaction=tx,
        signature=b"\x00" * 64,
        signer_pubkey=M.Pubkey(_PUB_A),
    )
    result = t.send_transaction(signed)
    assert result.status == "FAILED"


def test_in_process_transport_exhausted_raises() -> None:
    t = M.InProcessSolanaTransport(
        recent_blockhash=_BLOCKHASH,
        scripted_results=(),
    )
    tx = M.Transaction(
        fee_payer=M.Pubkey(_PUB_A),
        recent_blockhash=_BLOCKHASH,
        instructions=(
            M.Instruction(
                program_id=M.Pubkey(_PUB_A),
                accounts=(),
                data=b"",
            ),
        ),
    )
    signed = M.SignedTransaction(
        transaction=tx,
        signature=b"\x00" * 64,
        signer_pubkey=M.Pubkey(_PUB_A),
    )
    with pytest.raises(IndexError):
        t.send_transaction(signed)


# ---------------------------------------------------------------------------
# 9. SolanaNativeAdapter — lifecycle + submit
# ---------------------------------------------------------------------------


def test_adapter_starts_disconnected() -> None:
    adapter = _make_adapter()
    assert adapter.status().state is AdapterState.DISCONNECTED


def test_adapter_submit_before_connect_returns_rejected() -> None:
    adapter = _make_adapter()
    signal = _make_signal()
    ev = adapter.submit(signal, mark_price=150.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.qty == 0.0
    assert ev.meta["reason"] == "adapter_not_ready"


def test_adapter_connect_flips_ready_on_good_blockhash() -> None:
    adapter = _make_adapter()
    adapter.connect()
    assert adapter.status().state is AdapterState.READY


def test_adapter_connect_stays_disconnected_on_bad_blockhash() -> None:
    bad_transport = M.InProcessSolanaTransport(
        recent_blockhash=M.b58encode(b"A" * 16),
        scripted_results=(),
    )
    adapter = _make_adapter(transport=bad_transport)
    adapter.connect()
    assert adapter.status().state is AdapterState.DISCONNECTED


def test_adapter_submit_after_connect_returns_filled() -> None:
    adapter = _make_adapter(default_qty=1.25)
    adapter.connect()
    signal = _make_signal()
    ev = adapter.submit(signal, mark_price=150.0)
    assert ev.status is ExecutionStatus.FILLED
    assert ev.qty == 1.25
    assert ev.price == 150.0
    assert ev.venue == M._DEFAULT_VENUE
    assert ev.meta["tx_status"] == "CONFIRMED"
    assert ev.meta["tx_slot"] == "12345"
    assert ev.order_id == ev.meta["tx_signature"]


def test_adapter_submit_rejects_on_failed_tx_result() -> None:
    transport = _make_transport(status="FAILED")
    adapter = _make_adapter(transport=transport)
    adapter.connect()
    signal = _make_signal()
    ev = adapter.submit(signal, mark_price=150.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.qty == 0.0
    assert ev.meta["tx_status"] == "FAILED"


def test_adapter_constructor_rejects_bad_transport() -> None:
    with pytest.raises(TypeError):
        M.SolanaNativeAdapter(
            transport="not transport",  # type: ignore[arg-type]
            signer=M.deterministic_test_signer,
            keypair=M.KeypairHandle(
                credential_id="x",
                pubkey=M.Pubkey(_PUB_A),
            ),
            instruction_builder=_make_instructions,
        )


def test_adapter_constructor_rejects_non_callable_signer() -> None:
    with pytest.raises(TypeError):
        M.SolanaNativeAdapter(
            transport=_make_transport(),
            signer="not callable",  # type: ignore[arg-type]
            keypair=M.KeypairHandle(
                credential_id="x",
                pubkey=M.Pubkey(_PUB_A),
            ),
            instruction_builder=_make_instructions,
        )


def test_adapter_constructor_rejects_negative_default_qty() -> None:
    with pytest.raises(ValueError):
        M.SolanaNativeAdapter(
            transport=_make_transport(),
            signer=M.deterministic_test_signer,
            keypair=M.KeypairHandle(
                credential_id="x",
                pubkey=M.Pubkey(_PUB_A),
            ),
            instruction_builder=_make_instructions,
            default_qty=-1.0,
        )


# ---------------------------------------------------------------------------
# 10. INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def test_transaction_to_message_bytes_byte_identical() -> None:
    """Three identical Transactions produce identical byte projections."""
    bytes_runs: list[bytes] = []
    for _ in range(3):
        tx = M.Transaction(
            fee_payer=M.Pubkey(_PUB_A),
            recent_blockhash=_BLOCKHASH,
            instructions=(
                M.Instruction(
                    program_id=M.Pubkey(_PUB_A),
                    accounts=(M.Pubkey(_PUB_B),),
                    data=b"abc",
                ),
            ),
        )
        bytes_runs.append(tx.to_message_bytes())
    assert bytes_runs[0] == bytes_runs[1] == bytes_runs[2]


def test_adapter_submit_byte_identical_three_runs() -> None:
    """Three independent adapter constructions + submits produce
    byte-identical ExecutionEvent payloads (INV-15)."""

    payloads: list[tuple[object, ...]] = []
    for _ in range(3):
        adapter = _make_adapter(default_qty=0.5)
        adapter.connect()
        ev = adapter.submit(_make_signal(), mark_price=42.0)
        payloads.append(
            (
                ev.ts_ns,
                ev.symbol,
                ev.side.value,
                ev.qty,
                ev.price,
                ev.status.value,
                ev.venue,
                ev.order_id,
                tuple(sorted(ev.meta.items())),
            )
        )
    assert payloads[0] == payloads[1] == payloads[2]


# ---------------------------------------------------------------------------
# 11. AST guardrails on production module
# ---------------------------------------------------------------------------


_MODULE_PATH = pathlib.Path(M.__file__)
_MODULE_TREE = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.append(mod)
    return out


def test_no_vendor_sdks_imported_at_module_top() -> None:
    forbidden_roots = {"solana", "solders", "anchorpy", "base58"}
    for name in _top_level_imports(_MODULE_TREE):
        root = name.split(".")[0]
        assert root not in forbidden_roots, (
            f"vendor SDK {name!r} must not be imported at module top (lazy seam pattern)"
        )


def test_no_forbidden_runtime_imports_at_module_top() -> None:
    forbidden_roots = {
        "time",
        "datetime",
        "random",
        "asyncio",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
    }
    for name in _top_level_imports(_MODULE_TREE):
        root = name.split(".")[0]
        assert root not in forbidden_roots, (
            f"forbidden runtime import {name!r} at module top (RUNTIME_SAFE / INV-15)"
        )


def test_no_typed_event_constructors_outside_helpers() -> None:
    """B27 / B28 / INV-71 — adapters MAY construct ExecutionEvent (they are
    the Executor), but must NEVER construct SignalEvent / HazardEvent /
    SystemEvent. This pins the asymmetry."""

    forbidden = {"SignalEvent", "HazardEvent", "SystemEvent"}
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden, (
                f"adapter constructed forbidden typed event {node.func.id}"
            )


def test_no_os_environ_lookup() -> None:
    """Credentials must never come from ``os.environ`` here — they must
    come through the credentials seam (``KeypairHandle`` / ``SolanaSigner``)."""

    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr == "environ"
            ):
                raise AssertionError("os.environ accessed in module")
            if node.attr == "getenv":
                raise AssertionError("os.getenv accessed in module")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "getenv"


def test_lazy_seam_imports_inside_factory_body() -> None:
    """``enable_solana_native_factory`` is the seam — vendor SDKs must
    be imported only inside the factory / inner function body."""

    # Find the function definition
    fn: ast.FunctionDef | None = None
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "enable_solana_native_factory":
            fn = node
            break
    assert fn is not None
    found_vendor_imports: set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                found_vendor_imports.add(alias.name.split(".")[0])
        elif isinstance(sub, ast.ImportFrom):
            mod = (sub.module or "").split(".")[0]
            if mod:
                found_vendor_imports.add(mod)
    # The lazy seam MUST reference at least one vendor SDK inside its body
    assert "solana" in found_vendor_imports
    assert "base58" in found_vendor_imports
    assert "solders" in found_vendor_imports


# ---------------------------------------------------------------------------
# 12. ExecutionEvent projection sanity
# ---------------------------------------------------------------------------


def test_execution_event_inherits_signal_ts_and_symbol() -> None:
    adapter = _make_adapter(default_qty=1.0)
    adapter.connect()
    signal = _make_signal(symbol="BTCUSDT")
    ev = adapter.submit(signal, mark_price=30_000.0)
    assert isinstance(ev, ExecutionEvent)
    assert ev.ts_ns == signal.ts_ns
    assert ev.symbol == "BTCUSDT"
    assert ev.side is signal.side
    assert ev.kind is EventKind.EXECUTION


# ---------------------------------------------------------------------------
# 13. Module reload deterministic (re-import yields equivalent constants)
# ---------------------------------------------------------------------------


def test_module_reload_idempotent() -> None:
    m2 = importlib.reload(M)
    assert m2.NEW_PIP_DEPENDENCIES == M.NEW_PIP_DEPENDENCIES
    assert m2._ALLOWED_TX_STATUS == M._ALLOWED_TX_STATUS
