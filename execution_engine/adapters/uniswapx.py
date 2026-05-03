"""UniswapX intent-based DEX adapter (D-track scaffold).

UniswapX (and CowSwap / Across) replace the classic AMM "swap a fixed
token for at-least N output" with a *signed intent* the operator
broadcasts to a fillers network. The fillers compete to fill the
intent within the operator's slippage band; the winning filler pays
gas and takes a small spread.

This adapter therefore differs structurally from a CEX adapter: there
is no per-order ``buy`` / ``sell`` HTTP call — instead we sign an EIP-712
``OrderQuoter`` payload and POST it to the UniswapX backend. The
operator never broadcasts an on-chain tx themselves; that is the
filler's job.

Until the EVM signer (``DIX_EVM_RPC_URL`` + ``DIX_EVM_PRIVATE_KEY_PATH``)
is wired, the adapter stays in ``DISCONNECTED`` and rejects
``submit()`` with a structured meta. The order-book quote endpoint
(``DIX_UNISWAPX_API_URL``) is also required so we can pre-fetch the
fillers-network quote before signing.
"""

from __future__ import annotations

from execution_engine.adapters._live_base import (
    AdapterState,
    LiveAdapterBase,
)


class UniswapXAdapter(LiveAdapterBase):
    """Signs UniswapX intents and broadcasts them to the fillers network.

    Args:
        chain_id: EVM chain id (1 = mainnet, 8453 = base, 42161 = arb).
            Default is mainnet.
        rpc_url: EVM RPC endpoint. None keeps the adapter in scaffold mode.
        private_key_path: Filesystem path to the EVM signer key. None keeps
            scaffold mode.
        api_url: UniswapX backend base URL (orders + quotes).
        max_slippage_bps: Slippage tolerance encoded into the intent
            ``minAmountOut``. Operator-overridable per-order via
            ``signal.meta``.
    """

    def __init__(
        self,
        *,
        chain_id: int = 1,
        rpc_url: str | None = None,
        private_key_path: str | None = None,
        api_url: str | None = None,
        max_slippage_bps: int = 50,
    ) -> None:
        super().__init__(
            name=f"uniswapx:chain_{chain_id}",
            venue=f"uniswapx:chain_{chain_id}",
        )
        if chain_id <= 0:
            raise ValueError("chain_id must be > 0")
        if max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be >= 0")
        self._chain_id = chain_id
        self._rpc_url = rpc_url
        self._private_key_path = private_key_path
        self._api_url = api_url
        self._max_slippage_bps = max_slippage_bps

    @property
    def chain_id(self) -> int:
        return self._chain_id

    @property
    def max_slippage_bps(self) -> int:
        return self._max_slippage_bps

    def connect(self) -> None:
        missing = []
        if self._rpc_url is None:
            missing.append("DIX_EVM_RPC_URL")
        if self._private_key_path is None:
            missing.append("DIX_EVM_PRIVATE_KEY_PATH")
        if self._api_url is None:
            missing.append("DIX_UNISWAPX_API_URL")
        if missing:
            self._state = AdapterState.DISCONNECTED
            self._detail = (
                "missing credentials: "
                + ", ".join(missing)
                + " — scaffold mode active"
            )
            return
        self._state = AdapterState.CONNECTING
        self._detail = (
            "awaiting EIP-712 signer + UniswapX quote endpoint handshake"
        )


__all__ = ["UniswapXAdapter"]
