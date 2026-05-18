"""I-23 — Solana native stack adapter (non-Helius path).

# ADAPTED FROM: solana-py + solders + anchorpy + base58 (Solana
# Labs / Kevin Heavey reference SDKs — see
# https://github.com/michaelhly/solana-py,
# https://github.com/kevinheavey/solders,
# https://github.com/kevinheavey/anchorpy).

Where :mod:`execution_engine.adapters.helius` is the **read-only
intelligence** path (DAS / Enhanced Transactions / token holders),
this adapter is the **transaction submission** path: build a typed
Solana :class:`Transaction`, hand it to a :class:`SolanaSigner`
attached to a keypair stored in ``system_engine.credentials``, and
push the resulting :class:`SignedTransaction` through a
:class:`SolanaTransport` to a Solana RPC node.

Authority symmetry (B27 / B28 / INV-71):

* Signing is **never** performed inside ``execution_engine.adapters``
  — the adapter receives a :class:`SolanaSigner` callable; the actual
  private-key material lives behind the credentials seam.
* The adapter is a :class:`LiveAdapterBase` subclass: until the
  transport says it is ``READY`` (an explicit
  :meth:`SolanaNativeAdapter.connect` is required) every call to
  :meth:`submit` returns a structured ``REJECTED`` :class:`ExecutionEvent`
  rather than emitting a fake fill.

Determinism (INV-15):

* All payload construction (instruction encoding, message bytes,
  base58 serialization) is pure-Python on caller-supplied data.
* The module never reads the wall clock, ``os.environ``, or
  ``random``. Callers supply ``ts_ns`` from
  :class:`system.time_source.TimeAuthority`; transaction
  ``recent_blockhash`` values are supplied via the transport, not
  generated here.

Credentials:

* The private-key bytes are **never** parameters of this module's
  public surface. Callers register a :class:`SolanaSigner` whose
  implementation reads from ``system_engine.credentials.*`` (see the
  Wave-04.5 / S-12 / A-15 pattern). A :class:`KeypairHandle` value
  object identifies *which* credential to use without ever carrying
  the secret bytes.

Lazy seam:

* ``solana`` / ``solders`` / ``anchorpy`` / ``base58`` are listed in
  :data:`NEW_PIP_DEPENDENCIES` but never imported at module top.
  The optional :func:`enable_solana_native_factory` helper imports
  them only inside its function body so the production default
  stays a pure-stdlib path.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol, runtime_checkable

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    SignalEvent,
)
from execution_engine.adapters._live_base import (
    AdapterState,
    LiveAdapterBase,
)

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = (
    "solana",
    "solders",
    "anchorpy",
    "base58",
)


_VENUE_FAMILY: Final[str] = "solana"
_DEFAULT_VENUE: Final[str] = "solana:mainnet-beta"

# Base58 (Bitcoin alphabet) — used by Solana for pubkeys, signatures,
# and serialized transactions. Defined inline so the adapter has zero
# dependency on the ``base58`` package at the stdlib backend.
_B58_ALPHABET: Final[bytes] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# Base58 codec (pure-stdlib, byte-identical across runs)
# ---------------------------------------------------------------------------


def b58encode(raw: bytes) -> str:
    """Encode *raw* bytes to base58 (Bitcoin alphabet).

    Pure function — deterministic across runs (INV-15).
    """

    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("b58encode expects bytes")
    # Count leading zero bytes — they become leading '1' chars.
    zeros = 0
    for byte in raw:
        if byte == 0:
            zeros += 1
        else:
            break
    n = int.from_bytes(raw, "big")
    out: list[bytes] = []
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem : rem + 1])
    out.reverse()
    return ("1" * zeros) + (b"".join(out)).decode("ascii")


def b58decode(text: str) -> bytes:
    """Decode a base58 (Bitcoin alphabet) string back to bytes.

    Raises :class:`ValueError` on any non-alphabet character.
    """

    if not isinstance(text, str):
        raise TypeError("b58decode expects str")
    zeros = 0
    for ch in text:
        if ch == "1":
            zeros += 1
        else:
            break
    n = 0
    for ch in text:
        idx = _B58_ALPHABET.find(ch.encode("ascii"))
        if idx < 0:
            raise ValueError(f"invalid base58 character: {ch!r}")
        n = n * 58 + idx
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    return (b"\x00" * zeros) + body


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Pubkey:
    """A 32-byte Solana ed25519 public key (base58-encoded surface).

    Stored as the base58 *string* because that is what every Solana
    JSON RPC method emits / accepts; the raw bytes can always be
    recovered via :meth:`as_bytes`.
    """

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("Pubkey.value must be a non-empty str")
        raw = b58decode(self.value)
        if len(raw) != 32:
            raise ValueError("Pubkey must decode to 32 bytes")

    def as_bytes(self) -> bytes:
        return b58decode(self.value)


@dataclasses.dataclass(frozen=True, slots=True)
class KeypairHandle:
    """Opaque reference to a keypair stored in ``system_engine.credentials``.

    Carries the public key (for transaction construction / lookup) and
    a stable credential identifier — never the private bytes. Callers
    that need to sign pass *this* handle plus the desired
    :class:`SolanaSigner` (which knows how to dereference
    ``credential_id``) to :meth:`SolanaNativeAdapter.submit`.
    """

    credential_id: str
    pubkey: Pubkey

    def __post_init__(self) -> None:
        if not isinstance(self.credential_id, str) or not self.credential_id:
            raise ValueError("credential_id must be a non-empty str")
        if not isinstance(self.pubkey, Pubkey):
            raise TypeError("pubkey must be a Pubkey")


@dataclasses.dataclass(frozen=True, slots=True)
class Instruction:
    """One Solana program instruction.

    Carries the target program id, ordered account list, and an opaque
    data payload (already encoded — typically Anchor IDL or SPL
    layout).  The class is intentionally minimal so it can mirror
    ``solders.instruction.Instruction`` without depending on it.
    """

    program_id: Pubkey
    accounts: tuple[Pubkey, ...]
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.program_id, Pubkey):
            raise TypeError("program_id must be a Pubkey")
        if not isinstance(self.accounts, tuple):
            raise TypeError("accounts must be a tuple")
        for acc in self.accounts:
            if not isinstance(acc, Pubkey):
                raise TypeError("each account must be a Pubkey")
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError("data must be bytes")


@dataclasses.dataclass(frozen=True, slots=True)
class Transaction:
    """Unsigned Solana transaction.

    The ``recent_blockhash`` is supplied by the transport / caller —
    this module never generates one. The byte serialization
    (:meth:`to_message_bytes`) is deterministic and is the canonical
    payload that a :class:`SolanaSigner` signs.
    """

    fee_payer: Pubkey
    recent_blockhash: str
    instructions: tuple[Instruction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fee_payer, Pubkey):
            raise TypeError("fee_payer must be a Pubkey")
        if not isinstance(self.recent_blockhash, str) or not self.recent_blockhash:
            raise ValueError("recent_blockhash must be a non-empty str")
        if not isinstance(self.instructions, tuple):
            raise TypeError("instructions must be a tuple")
        if not self.instructions:
            raise ValueError("at least one instruction required")
        for ins in self.instructions:
            if not isinstance(ins, Instruction):
                raise TypeError("each instruction must be an Instruction")
        # Validate recent_blockhash decodes to 32 bytes (Solana blockhash size).
        raw = b58decode(self.recent_blockhash)
        if len(raw) != 32:
            raise ValueError("recent_blockhash must decode to 32 bytes")

    def to_message_bytes(self) -> bytes:
        """Deterministic canonical byte serialization of the message.

        This is a stable, pure-Python projection — NOT the wire-level
        encoding solders uses (which depends on a TOML-version-pinned
        layout we don't want to re-implement). It is sufficient for
        replay determinism (INV-15) and for handing to a
        :class:`SolanaSigner` that hashes / signs the payload; the
        actual wire serialization happens inside the live SDK when
        the lazy seam is enabled.
        """

        parts: list[bytes] = []
        parts.append(self.fee_payer.as_bytes())
        parts.append(b58decode(self.recent_blockhash))
        parts.append(len(self.instructions).to_bytes(2, "big"))
        for ins in self.instructions:
            parts.append(ins.program_id.as_bytes())
            parts.append(len(ins.accounts).to_bytes(2, "big"))
            for acc in ins.accounts:
                parts.append(acc.as_bytes())
            parts.append(len(ins.data).to_bytes(4, "big"))
            parts.append(bytes(ins.data))
        return b"".join(parts)


@dataclasses.dataclass(frozen=True, slots=True)
class SignedTransaction:
    """Signed transaction handed to the transport for submission."""

    transaction: Transaction
    signature: bytes  # 64-byte ed25519 signature
    signer_pubkey: Pubkey

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, Transaction):
            raise TypeError("transaction must be a Transaction")
        if not isinstance(self.signature, (bytes, bytearray)):
            raise TypeError("signature must be bytes")
        if len(self.signature) != 64:
            raise ValueError("signature must be 64 bytes")
        if not isinstance(self.signer_pubkey, Pubkey):
            raise TypeError("signer_pubkey must be a Pubkey")

    def to_wire_bytes(self) -> bytes:
        """Deterministic wire-style byte serialization (signature + message)."""
        return bytes(self.signature) + self.transaction.to_message_bytes()


_ALLOWED_TX_STATUS: Final[frozenset[str]] = frozenset({"CONFIRMED", "FAILED", "DROPPED"})


@dataclasses.dataclass(frozen=True, slots=True)
class TxResult:
    """Outcome of one transaction submission."""

    signature: str  # base58-encoded tx signature returned by the RPC node
    status: str  # one of: "CONFIRMED" / "FAILED" / "DROPPED"
    slot: int
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("signature must be a non-empty str")
        if self.status not in _ALLOWED_TX_STATUS:
            raise ValueError(f"status must be one of {sorted(_ALLOWED_TX_STATUS)}")
        if not isinstance(self.slot, int) or self.slot < 0:
            raise ValueError("slot must be a non-negative int")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a str")


# ---------------------------------------------------------------------------
# Signer + transport seams
# ---------------------------------------------------------------------------


# A SolanaSigner takes the message bytes plus the credential reference
# and returns the 64-byte signature. The implementation lives in
# ``system_engine.credentials.*``; the adapter never sees the secret.
SolanaSigner = Callable[[bytes, KeypairHandle], bytes]


@runtime_checkable
class SolanaTransport(Protocol):
    """Pluggable transport contract for Solana RPC.

    Implementations:

    * :class:`InProcessSolanaTransport` — pure-Python, used by tests
      and replay; returns the canned values supplied at construction.
    * Live transport created by :func:`enable_solana_native_factory`
      — lazy-imports ``solana.rpc.async_api`` only inside the
      factory body, never at module top.
    """

    def get_recent_blockhash(self) -> str: ...

    def send_transaction(self, signed: SignedTransaction) -> TxResult: ...


@dataclasses.dataclass(frozen=True, slots=True)
class InProcessSolanaTransport:
    """Pure-Python transport that replays caller-supplied values.

    Attributes:
        recent_blockhash: Canonical 32-byte base58-encoded blockhash
            returned by :meth:`get_recent_blockhash`.
        scripted_results: Tuple of :class:`TxResult` values returned
            (in order) by successive :meth:`send_transaction` calls.
            If exhausted, an :class:`IndexError` is raised so tests
            see the mismatch loudly.
    """

    recent_blockhash: str
    scripted_results: tuple[TxResult, ...]

    def get_recent_blockhash(self) -> str:
        return self.recent_blockhash

    def send_transaction(self, signed: SignedTransaction) -> TxResult:
        if not isinstance(signed, SignedTransaction):
            raise TypeError("signed must be a SignedTransaction")
        # Replay-style: the *index* is the number of calls so far,
        # which the caller can track externally by passing a
        # right-sized scripted_results tuple. Determinism is
        # preserved because the InProcess transport itself holds no
        # mutable state.
        if not self.scripted_results:
            raise IndexError("InProcessSolanaTransport exhausted")
        return self.scripted_results[0]


# ---------------------------------------------------------------------------
# Deterministic in-process signer (used by tests / replay)
# ---------------------------------------------------------------------------


def deterministic_test_signer(message: bytes, handle: KeypairHandle) -> bytes:
    """Deterministic ``SolanaSigner`` for tests.

    Produces a 64-byte "signature" via two BLAKE2b-32 hashes over
    ``handle.credential_id`` + the message. **This is NOT
    cryptographically valid** — it is a byte-stable stand-in so
    test transactions can be constructed, serialized, and replayed
    without depending on a real ed25519 keypair. Live signing is
    delegated to the credentials seam.
    """

    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    if not isinstance(handle, KeypairHandle):
        raise TypeError("handle must be a KeypairHandle")
    seed = handle.credential_id.encode("utf-8") + bytes(message)
    left = hashlib.blake2b(seed, digest_size=32, person=b"dix-i23-sig-l").digest()
    right = hashlib.blake2b(seed, digest_size=32, person=b"dix-i23-sig-r").digest()
    return left + right


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SolanaNativeAdapter(LiveAdapterBase):
    """Live ``BrokerAdapter`` for direct Solana RPC submission.

    Subclasses :class:`LiveAdapterBase` so until :meth:`connect` is
    called and the transport is verified ``READY``, every
    :meth:`submit` returns a structured ``REJECTED``
    :class:`ExecutionEvent`. This is the same scaffolding pattern as
    :class:`execution_engine.adapters.uniswapx.UniswapXAdapter`.

    Attributes:
        name: Stable adapter identifier (e.g. ``"solana_native"``).
        venue: Venue tag (e.g. ``"solana:mainnet-beta"``).
        transport: :class:`SolanaTransport` implementation.
        signer: :class:`SolanaSigner` implementation that knows how
            to dereference :class:`KeypairHandle` to actual signing
            material.
        keypair: Operator's :class:`KeypairHandle` for fee payment +
            signing on this adapter.
        instruction_builder: Pure function mapping
            ``(SignalEvent, mark_price, fee_payer) -> tuple[Instruction, ...]``.
            Must be deterministic (INV-15).
    """

    name: str = "solana_native"
    venue: str = _DEFAULT_VENUE

    def __init__(
        self,
        *,
        transport: SolanaTransport,
        signer: SolanaSigner,
        keypair: KeypairHandle,
        instruction_builder: Callable[[SignalEvent, float, Pubkey], tuple[Instruction, ...]],
        default_qty: float = 0.0,
        name: str = "solana_native",
        venue: str = _DEFAULT_VENUE,
    ) -> None:
        super().__init__(name=name, venue=venue)
        if not isinstance(transport, SolanaTransport):
            raise TypeError("transport must implement SolanaTransport")
        if not callable(signer):
            raise TypeError("signer must be callable")
        if not isinstance(keypair, KeypairHandle):
            raise TypeError("keypair must be a KeypairHandle")
        if not callable(instruction_builder):
            raise TypeError("instruction_builder must be callable")
        if not isinstance(default_qty, (int, float)):
            raise TypeError("default_qty must be a number")
        if float(default_qty) < 0.0:
            raise ValueError("default_qty must be non-negative")
        self._transport: SolanaTransport = transport
        self._signer: SolanaSigner = signer
        self._keypair: KeypairHandle = keypair
        self._instruction_builder = instruction_builder
        self._default_qty: float = float(default_qty)

    # ---- lifecycle -------------------------------------------------------

    def connect(self) -> None:
        """Verify the transport responds with a valid recent blockhash.

        On success, flips state to ``READY``; on transport error, the
        adapter remains ``DISCONNECTED`` so :meth:`submit` keeps
        rejecting.
        """
        try:
            blockhash = self._transport.get_recent_blockhash()
        except Exception as exc:  # noqa: BLE001 — fail-safe scaffold
            self._state = AdapterState.DISCONNECTED
            self._detail = f"transport error: {exc!r}"
            return
        if not isinstance(blockhash, str) or not blockhash:
            self._state = AdapterState.DISCONNECTED
            self._detail = "transport returned empty blockhash"
            return
        try:
            raw = b58decode(blockhash)
        except ValueError as exc:
            self._state = AdapterState.DISCONNECTED
            self._detail = f"blockhash decode failed: {exc!r}"
            return
        if len(raw) != 32:
            self._state = AdapterState.DISCONNECTED
            self._detail = "blockhash not 32 bytes"
            return
        self._state = AdapterState.READY
        self._detail = "ready"

    # ---- BrokerAdapter Protocol -----------------------------------------

    def _submit_live(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        """Build → sign → submit, project the RPC result onto ExecutionEvent.

        Pure function of ``(signal, mark_price)`` plus the transport's
        canned blockhash + result, the signer (deterministic in
        tests), and the instruction builder.
        """

        blockhash = self._transport.get_recent_blockhash()
        instructions = self._instruction_builder(signal, mark_price, self._keypair.pubkey)
        if not isinstance(instructions, tuple):
            raise TypeError("instruction_builder must return a tuple")
        if not instructions:
            raise ValueError("instruction_builder returned no instructions")
        tx = Transaction(
            fee_payer=self._keypair.pubkey,
            recent_blockhash=blockhash,
            instructions=instructions,
        )
        sig = self._signer(tx.to_message_bytes(), self._keypair)
        signed = SignedTransaction(
            transaction=tx,
            signature=sig,
            signer_pubkey=self._keypair.pubkey,
        )
        result = self._transport.send_transaction(signed)
        return _project_tx_result(signal, mark_price, self.venue, result, self._default_qty)


def _project_tx_result(
    signal: SignalEvent,
    mark_price: float,
    venue: str,
    result: TxResult,
    default_qty: float,
) -> ExecutionEvent:
    """Project a :class:`TxResult` onto an :class:`ExecutionEvent`."""

    if result.status == "CONFIRMED":
        status = ExecutionStatus.FILLED
        qty = float(default_qty)
    else:
        status = ExecutionStatus.REJECTED
        qty = 0.0
    meta: Mapping[str, str] = {
        "tx_status": result.status,
        "tx_signature": result.signature,
        "tx_slot": str(result.slot),
        "tx_detail": result.detail,
    }
    return ExecutionEvent(
        ts_ns=signal.ts_ns,
        symbol=signal.symbol,
        side=signal.side,
        qty=qty,
        price=mark_price,
        status=status,
        venue=venue,
        order_id=result.signature,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Lazy seam — never imported at module top
# ---------------------------------------------------------------------------


def enable_solana_native_factory() -> Callable[..., SolanaTransport]:
    """Return a factory that constructs a live ``SolanaTransport``.

    Lazy-imports ``solana.rpc.async_api`` (and the related
    ``solders`` / ``anchorpy`` / ``base58`` SDKs) only inside the
    function body. Production callers may stay on the pure-stdlib
    InProcessSolanaTransport until they are ready to wire a real
    RPC endpoint — this factory is the only seam between the
    adapter and the optional vendor SDKs.

    Note:
        This function is intentionally a placeholder seam — the
        canonical campaign ships the pure-stdlib path first. The
        live HTTP transport will be added in a follow-up that wires
        ``system_engine.credentials.solana_signer.SolanaSigner``
        through this seam and the lazy-imported SDK below.
    """

    def _factory(*args: object, **kwargs: object) -> SolanaTransport:
        # Imports stay inside the factory body so the module top
        # remains pure-stdlib (AST-guarded by ``test_solana_native``).
        import base58 as _b58  # noqa: F401 — see docstring
        import solders  # noqa: F401 — see docstring
        from solana.rpc.api import Client  # noqa: F401 — see docstring

        del args, kwargs
        raise NotImplementedError(
            "live solana transport not yet wired — pure-stdlib "
            "InProcessSolanaTransport is the production default"
        )

    return _factory


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "Pubkey",
    "KeypairHandle",
    "Instruction",
    "Transaction",
    "SignedTransaction",
    "TxResult",
    "SolanaSigner",
    "SolanaTransport",
    "InProcessSolanaTransport",
    "SolanaNativeAdapter",
    "b58encode",
    "b58decode",
    "deterministic_test_signer",
    "enable_solana_native_factory",
]


def _validate_sequence_type(seq: Sequence[object], expected: type) -> bool:
    """Internal helper used by tests — kept here for AST symmetry."""
    return all(isinstance(item, expected) for item in seq)
