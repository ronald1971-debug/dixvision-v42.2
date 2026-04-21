"""
security.wallet_connect \u2014 safe crypto wallet onboarding.

Design goals:
    - private-key material NEVER hits ledger, log, chat, screenshot, or stdout
    - default backend is watch-only (public address, zero signing)
    - signing backends are opt-in, gated by governance + dead-man + kill-switch
    - chain-agnostic: EVM, Solana, Bitcoin (watch-only), WalletConnect v2
    - phase clock (security.wallet_policy):
        days  0–30  WARMUP      live signing disabled for everyone
        days 30–60  SUPERVISED  live ok after governance approval,
                                HARD $100 USD / 24h / wallet AND system cap
        days 60+    OPERATOR    cap configurable (still governance-gated)

This module is *infrastructure* \u2014 it stores addresses + backend kind +
policy flags. Actual signing is a pluggable adapter that lives in a
backend-specific module (e.g. ``security.backends.ledger``). The key
exchange is always short-lived and never persisted here.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.secrets import store_secret
from state.ledger.writer import get_writer
from system.time_source import utc_now


def utc_now_iso() -> str:
    return utc_now().isoformat()

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "wallets.sqlite"
_lock = threading.RLock()


class Backend(str, Enum):
    WATCH_ONLY = "watch_only"
    LOCAL_SIGNER = "local_signer"                                                # key in OS keyring
    HARDWARE = "hardware"                                                        # ledger / trezor
    WALLETCONNECT = "walletconnect"


class Chain(str, Enum):
    ETHEREUM = "ethereum"
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    BSC = "bsc"
    SOLANA = "solana"
    BITCOIN = "bitcoin"


@dataclass
class WalletRecord:
    id: int
    label: str
    chain: Chain
    backend: Backend
    address: str
    added_utc: str
    live_signing_allowed: bool = False
    last_approved_by: str = ""                                                   # governance event id
    approval_expires_utc: str = ""
    last_used_utc: str = ""
    notes: str = ""

    def mask(self) -> str:
        if len(self.address) < 12:
            return self.address
        return self.address[:6] + "\u2026" + self.address[-4:]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    chain TEXT NOT NULL,
    backend TEXT NOT NULL,
    address TEXT NOT NULL,
    added_utc TEXT NOT NULL,
    live_signing_allowed INTEGER NOT NULL DEFAULT 0,
    last_approved_by TEXT NOT NULL DEFAULT '',
    approval_expires_utc TEXT NOT NULL DEFAULT '',
    last_used_utc TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(chain, address)
);
CREATE INDEX IF NOT EXISTS wallets_chain_idx ON wallets(chain);
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH))
    c.executescript(_SCHEMA)
    c.row_factory = sqlite3.Row
    return c


def _row(r: sqlite3.Row) -> WalletRecord:
    return WalletRecord(
        id=int(r["id"]), label=r["label"],
        chain=Chain(r["chain"]), backend=Backend(r["backend"]),
        address=r["address"], added_utc=r["added_utc"],
        live_signing_allowed=bool(r["live_signing_allowed"]),
        last_approved_by=r["last_approved_by"],
        approval_expires_utc=r["approval_expires_utc"],
        last_used_utc=r["last_used_utc"],
        notes=r["notes"],
    )


def list_wallets(chain: Chain | None = None) -> list[WalletRecord]:
    with _lock, _connect() as c:
        if chain is None:
            rs = c.execute("SELECT * FROM wallets ORDER BY id").fetchall()
        else:
            rs = c.execute("SELECT * FROM wallets WHERE chain = ? ORDER BY id",
                           (chain.value,)).fetchall()
        return [_row(r) for r in rs]


def get_wallet(chain: Chain, address: str) -> WalletRecord | None:
    with _lock, _connect() as c:
        r = c.execute("SELECT * FROM wallets WHERE chain=? AND address=?",
                      (chain.value, address)).fetchone()
        return _row(r) if r else None


def connect_wallet(*, label: str, chain: Chain, backend: Backend,
                   address: str, notes: str = "") -> WalletRecord:
    """Register a wallet. Default is watch-only. Signing remains disabled.

    If a wallet with ``(chain, address)`` already exists, only mutable
    descriptive fields (label, backend, notes) are updated and the
    existing signing-authorization columns (live_signing_allowed,
    last_approved_by, approval_expires_utc) are preserved. Authorization
    state must never change without an explicit governance event.
    """
    if backend is Backend.LOCAL_SIGNER and len(address) < 10:
        raise ValueError("address too short")
    with _lock, _connect() as c:
        now = utc_now_iso()
        existing = c.execute(
            "SELECT id FROM wallets WHERE chain=? AND address=?",
            (chain.value, address),
        ).fetchone()
        if existing is None:
            c.execute(
                "INSERT INTO wallets "
                "(label, chain, backend, address, added_utc, "
                "live_signing_allowed, last_approved_by, "
                "approval_expires_utc, last_used_utc, notes) "
                "VALUES (?, ?, ?, ?, ?, 0, '', '', '', ?)",
                (label, chain.value, backend.value, address, now, notes),
            )
        else:
            c.execute(
                "UPDATE wallets SET label=?, backend=?, notes=? "
                "WHERE chain=? AND address=?",
                (label, backend.value, notes, chain.value, address),
            )
        c.commit()
        row = c.execute("SELECT * FROM wallets WHERE chain=? AND address=?",
                        (chain.value, address)).fetchone()
    get_writer().write("SECURITY", "WALLET_CONNECTED", "DYON", {
        "chain": chain.value, "backend": backend.value,
        "address_masked": address[:6] + "\u2026" + address[-4:]
        if len(address) > 12 else address,
        "live_signing_allowed": bool(row["live_signing_allowed"]),
        "updated_existing": existing is not None,
    })
    return _row(row)


def disconnect_wallet(chain: Chain, address: str) -> bool:
    with _lock, _connect() as c:
        cur = c.execute("DELETE FROM wallets WHERE chain=? AND address=?",
                        (chain.value, address))
        c.commit()
        ok = cur.rowcount > 0
    if ok:
        get_writer().write("SECURITY", "WALLET_DISCONNECTED", "DYON", {
            "chain": chain.value,
            "address_masked": address[:6] + "\u2026" + address[-4:]
            if len(address) > 12 else address,
        })
    return ok


def approve_live_signing(chain: Chain, address: str, *,
                         approved_by: str, expires_utc: str) -> WalletRecord | None:
    """Governance-gated: enable live signing for this wallet until expiry.

    Refused during the 30-day WARMUP phase (security.wallet_policy)."""
    from security import wallet_policy as _wp
    if _wp.snapshot().phase is _wp.Phase.WARMUP:
        raise PermissionError(
            "live signing disabled during 30-day WARMUP phase "
            f"({_wp.snapshot().warmup_days_remaining} days remaining)")
    w = get_wallet(chain, address)
    if w is None:
        return None
    if w.backend is Backend.WATCH_ONLY:
        raise PermissionError("watch_only wallets cannot be promoted to live signing")
    with _lock, _connect() as c:
        c.execute(
            "UPDATE wallets SET live_signing_allowed=1, "
            "last_approved_by=?, approval_expires_utc=? "
            "WHERE chain=? AND address=?",
            (approved_by, expires_utc, chain.value, address),
        )
        c.commit()
    get_writer().write("GOVERNANCE", "WALLET_SIGN_APPROVED", "GOVERNANCE", {
        "chain": chain.value,
        "address_masked": w.mask(),
        "approved_by": approved_by,
        "expires_utc": expires_utc,
    })
    return get_wallet(chain, address)


def revoke_live_signing(chain: Chain, address: str) -> WalletRecord | None:
    with _lock, _connect() as c:
        c.execute(
            "UPDATE wallets SET live_signing_allowed=0, "
            "last_approved_by='', approval_expires_utc='' "
            "WHERE chain=? AND address=?",
            (chain.value, address),
        )
        c.commit()
    get_writer().write("GOVERNANCE", "WALLET_SIGN_REVOKED", "GOVERNANCE",
                       {"chain": chain.value, "address": address[:6] + "..."})
    return get_wallet(chain, address)


def can_sign(chain: Chain, address: str, *,
             usd_notional: float = 0.0) -> bool:
    """Runtime gate. INDIRA must call this before any sign attempt.

    Combines wallet-registry flags with the phase/budget policy in
    security.wallet_policy. Rejects if any layer says no."""
    w = get_wallet(chain, address)
    if w is None:
        return False
    if w.backend is Backend.WATCH_ONLY:
        return False
    if not w.live_signing_allowed:
        return False
    if w.approval_expires_utc and w.approval_expires_utc < utc_now_iso():
        return False
    from security import wallet_policy as _wp
    ok, _ = _wp.can_sign(chain.value, address, usd_notional=float(usd_notional))
    return bool(ok)


def store_encrypted_key(key_env_name: str, key_value: str) -> None:
    """Convenience wrapper around core.secrets for storing a signing key.
    The raw key leaves this function via OS keyring only; never returned."""
    if not key_value:
        raise ValueError("empty key")
    store_secret(key_env_name, key_value)
    get_writer().write("SECURITY", "SECRET_STORED", "DYON",
                       {"key_env": key_env_name})


__all__ = [
    "Backend", "Chain", "WalletRecord",
    "connect_wallet", "disconnect_wallet",
    "list_wallets", "get_wallet",
    "approve_live_signing", "revoke_live_signing", "can_sign",
    "store_encrypted_key",
]
