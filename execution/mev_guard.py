"""execution.mev_guard \u2014 MEV-aware DEX transaction wrapper.

Guard-rails, not signers:
    - rejects raw public-mempool sends unless `allow_public=True`
    - recommends private-mempool relay per-chain (Flashbots on EVM, Jito on
      Solana)
    - enforces min-out (slippage floor) + deadline seconds
    - can "simulate" via caller-provided callback before sign

Actual signing happens in a backend adapter. This module only assembles a
`GuardedSwap` spec + emits audit events. It never holds keys.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from execution.slippage import estimate as estimate_slip
from state.ledger.writer import get_writer

# canonical public private-relay URLs (defaults; overridable via env/config)
FLASHBOTS_DEFAULT = "https://relay.flashbots.net"
JITO_DEFAULT = "https://mainnet.block-engine.jito.wtf"

_PRIVATE_RELAYS: dict[str, str] = {
    "ethereum": FLASHBOTS_DEFAULT,
    "base": "https://relay.flashbots.net",
    "arbitrum": "https://relay.flashbots.net",
    "optimism": "https://relay.flashbots.net",
    "polygon": "https://relay.flashbots.net",
    "bsc": "",                                                    # no canonical private relay
    "solana": JITO_DEFAULT,
}


@dataclass
class GuardedSwap:
    chain: str
    dex: str
    token_in: str
    token_out: str
    amount_in: float
    min_amount_out: float
    deadline_sec: int
    private_relay_url: str = ""
    allow_public: bool = False
    simulated_ok: bool = False
    expected_slippage_bps: float = 0.0
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "chain": self.chain, "dex": self.dex,
            "token_in": self.token_in, "token_out": self.token_out,
            "amount_in": self.amount_in,
            "min_amount_out": self.min_amount_out,
            "deadline_sec": self.deadline_sec,
            "private_relay_url": self.private_relay_url,
            "allow_public": self.allow_public,
            "simulated_ok": self.simulated_ok,
            "expected_slippage_bps": self.expected_slippage_bps,
            "notes": self.notes,
        }


def private_relay_for(chain: str) -> str:
    return _PRIVATE_RELAYS.get(chain.lower(), "")


def prepare_swap(*, chain: str, dex: str, token_in: str, token_out: str,
                 amount_in: float, mid_price: float, adv_qty: float,
                 spread_bps: float = 20.0,
                 max_slippage_bps: float = 50.0,
                 deadline_sec: int = 60,
                 allow_public: bool = False) -> GuardedSwap:
    """Build a GuardedSwap with MEV protection and sane min_out bounds."""
    if amount_in <= 0 or mid_price <= 0:
        raise ValueError("amount_in/mid_price must be > 0")
    slip = estimate_slip(qty=amount_in, adv_qty=adv_qty or amount_in * 100,
                         spread_bps=spread_bps)
    effective_bps = slip.exp_slippage_bps + max_slippage_bps
    out_expected = amount_in * mid_price
    min_out = out_expected * (1.0 - effective_bps / 10000.0)
    relay = private_relay_for(chain)
    note_bits = []
    if not relay and not allow_public:
        note_bits.append(
            f"no canonical private relay for {chain!r} \u2014 guard requires "
            "allow_public=True to send via public mempool")
    return GuardedSwap(
        chain=chain.lower(), dex=dex, token_in=token_in, token_out=token_out,
        amount_in=float(amount_in),
        min_amount_out=float(max(0.0, min_out)),
        deadline_sec=int(max(1, deadline_sec)),
        private_relay_url=relay,
        allow_public=bool(allow_public),
        expected_slippage_bps=float(slip.exp_slippage_bps),
        notes=" | ".join(note_bits),
    )


SimulateFn = Callable[[GuardedSwap], bool]


def validate_and_emit(swap: GuardedSwap, *,
                      simulate: SimulateFn | None = None) -> bool:
    """Run pre-flight checks, optionally simulate, emit ledger audit event.

    Returns True if the swap is ready to sign, False otherwise. Never
    signs or broadcasts itself.
    """
    reasons = []
    if swap.amount_in <= 0 or swap.min_amount_out <= 0:
        reasons.append("invalid_amounts")
    if not swap.private_relay_url and not swap.allow_public:
        reasons.append("no_private_relay")
    if swap.deadline_sec <= 0 or swap.deadline_sec > 3600:
        reasons.append("bad_deadline")
    sim_ok = True
    if simulate is not None:
        try:
            sim_ok = bool(simulate(swap))
        except Exception as exc:
            sim_ok = False
            reasons.append(f"sim_error:{exc}")
        swap.simulated_ok = sim_ok
        if not sim_ok and "sim_failed" not in reasons:
            reasons.append("sim_failed")
    ok = not reasons
    get_writer().write("EXECUTION", "MEV_GUARD_CHECK", "INDIRA", {
        **swap.as_dict(),
        "ok": ok, "reasons": reasons,
    })
    return ok


__all__ = ["GuardedSwap", "private_relay_for",
           "prepare_swap", "validate_and_emit"]
