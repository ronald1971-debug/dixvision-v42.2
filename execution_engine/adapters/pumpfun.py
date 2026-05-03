"""Pump.fun / Raydium memecoin adapter (D-track scaffold).

The MEMECOIN trading domain (per ``execution_engine/adapters/router.py``)
is hard-isolated from NORMAL and COPY_TRADING. This adapter is the
only sanctioned executor for that domain — a Pump.fun bonding-curve
buy or a Raydium pool swap, depending on whether the token has graduated.

Until the SOL keypair + Solana RPC URL are wired (``DIX_SOLANA_RPC_URL``
+ ``DIX_PUMPFUN_KEYPAIR_PATH``), the adapter stays in
``DISCONNECTED`` and rejects ``submit()`` with a structured meta.
This is intentional: a memecoin "fill" with a non-existent on-chain
tx would silently violate INV-65 (per-decision audit truthfulness).

Routing rule (matches Hummingbot's pump.fun connector spec, 2025-06):

* token still on bonding curve → ``pump.fun`` API
* token graduated (>= ~85 % progress) → Raydium AMM via Jupiter
"""

from __future__ import annotations

from execution_engine.adapters._live_base import (
    AdapterState,
    LiveAdapterBase,
)


class PumpFunAdapter(LiveAdapterBase):
    """Pump.fun + Raydium memecoin executor.

    Args:
        rpc_url: Solana RPC endpoint (None keeps the adapter in
            scaffold mode).
        keypair_path: Filesystem path to the SOL signer keypair.
            None keeps the adapter in scaffold mode.
        priority_fee_micro_lamports: Compute-budget priority fee. The
            memecoin venue is highly contested so this matters; default
            is conservative.
    """

    def __init__(
        self,
        *,
        rpc_url: str | None = None,
        keypair_path: str | None = None,
        priority_fee_micro_lamports: int = 10_000,
    ) -> None:
        super().__init__(name="pumpfun", venue="pumpfun:solana")
        if priority_fee_micro_lamports < 0:
            raise ValueError("priority fee must be >= 0")
        self._rpc_url = rpc_url
        self._keypair_path = keypair_path
        self._priority_fee = priority_fee_micro_lamports

    @property
    def priority_fee_micro_lamports(self) -> int:
        return self._priority_fee

    def connect(self) -> None:
        if self._rpc_url is None or self._keypair_path is None:
            self._state = AdapterState.DISCONNECTED
            missing = []
            if self._rpc_url is None:
                missing.append("DIX_SOLANA_RPC_URL")
            if self._keypair_path is None:
                missing.append("DIX_PUMPFUN_KEYPAIR_PATH")
            self._detail = (
                "missing credentials: "
                + ", ".join(missing)
                + " — scaffold mode active"
            )
            return
        self._state = AdapterState.CONNECTING
        self._detail = "awaiting Solana RPC handshake"


__all__ = ["PumpFunAdapter"]
