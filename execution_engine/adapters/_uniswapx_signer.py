"""EIP-712 signer for UniswapX intents (D3).

UniswapX V2 ``ExclusiveDutchOrder`` is an *intent*: the operator signs
an EIP-712 typed-data payload and POSTs it to the UniswapX backend.
The fillers network competes to settle the intent within the operator's
slippage band; the winning filler pays gas and broadcasts the on-chain
transaction. The operator never broadcasts a tx themselves.

EIP-712 reference: https://eips.ethereum.org/EIPS/eip-712
UniswapX docs:     https://docs.uniswap.org/contracts/uniswapx/overview
Permit2 docs:      https://github.com/Uniswap/permit2

Two helpers live here:

1. :func:`build_exclusive_dutch_order_typed_data` — pure data builder.
   Produces the canonical EIP-712 typed-data dict for an
   ``ExclusiveDutchOrder``. No private key, no network — safe to call
   from any thread, any test.

2. :func:`sign_typed_data` — wraps ``eth_account.Account.sign_typed_data``.
   Takes the dict from (1) and a hex-encoded private key and returns
   the 65-byte signature as a ``0x``-prefixed hex string + the signer's
   address (so the adapter can sanity-check the signer matches the
   wallet declared at construction time).

Both helpers are pure with respect to system clock — caller passes
``deadline_unix_s`` explicitly so replay tests stay byte-identical
(INV-15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_account.signers.local import LocalAccount

# Permit2 contract address (same on every chain UniswapX deploys to).
# https://github.com/Uniswap/permit2/blob/main/script/DeployPermit2.s.sol
PERMIT2_ADDRESS = "0x000000000022D473030F116dDEE9F6B43aC78BA3"


@dataclass(frozen=True)
class DutchInput:
    """Single ``ExclusiveDutchOrder`` input leg.

    Attributes:
        token: ERC-20 token contract address (``0x…``). Lowercase
            preferred; the signer normalises before encoding.
        start_amount: Initial decay-start amount, in token base units
            (e.g. 1_000_000 for 1.0 USDC at 6 decimals).
        end_amount: Decay-end amount, in token base units.
            ``end_amount >= start_amount`` for inputs (the swapper
            offers more input as the auction decays).
    """

    token: str
    start_amount: int
    end_amount: int


@dataclass(frozen=True)
class DutchOutput:
    """Single ``ExclusiveDutchOrder`` output leg.

    Attributes:
        token: ERC-20 token contract address.
        start_amount: Initial decay-start amount (highest price for
            the swapper).
        end_amount: Decay-end amount (lowest acceptable price).
            ``end_amount <= start_amount`` for outputs.
        recipient: Destination wallet (typically the swapper's own
            wallet; can be a router for hooks).
    """

    token: str
    start_amount: int
    end_amount: int
    recipient: str


@dataclass(frozen=True)
class ExclusiveDutchOrderIntent:
    """Frozen, fully-specified ExclusiveDutchOrder.

    This is the operator-side representation; before POSTing to the
    UniswapX backend the adapter wraps it in the canonical Permit2
    typed-data shape via
    :func:`build_exclusive_dutch_order_typed_data`.
    """

    chain_id: int
    reactor: str
    swapper: str
    nonce: int
    deadline_unix_s: int
    decay_start_time_unix_s: int
    decay_end_time_unix_s: int
    exclusive_filler: str
    exclusivity_override_bps: int
    input: DutchInput
    outputs: tuple[DutchOutput, ...]

    def __post_init__(self) -> None:
        if self.chain_id <= 0:
            raise ValueError("chain_id must be > 0")
        if self.nonce < 0:
            raise ValueError("nonce must be >= 0")
        if not self.outputs:
            raise ValueError("outputs must be non-empty")
        if self.exclusivity_override_bps < 0:
            raise ValueError("exclusivity_override_bps must be >= 0")
        if self.decay_end_time_unix_s < self.decay_start_time_unix_s:
            raise ValueError(
                "decay_end_time must be >= decay_start_time"
            )
        if self.deadline_unix_s < self.decay_end_time_unix_s:
            raise ValueError(
                "deadline must be >= decay_end_time"
            )


def build_exclusive_dutch_order_typed_data(
    intent: ExclusiveDutchOrderIntent,
) -> dict[str, Any]:
    """Build the canonical EIP-712 typed-data dict for ``intent``.

    The dict shape matches what
    ``eth_account.messages.encode_typed_data`` and the UniswapX
    ``/v2/order`` endpoint expect. Domain is pinned to ``"Permit2"`` /
    ``intent.chain_id`` / ``PERMIT2_ADDRESS`` because UniswapX's
    Reactor settles via Permit2 ``permitWitnessTransferFrom``.

    Returns:
        Plain Python dict suitable for both
        ``Account.sign_typed_data(full_message=...)`` and JSON
        serialisation onto the UniswapX REST surface.
    """
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "PermitWitnessTransferFrom": [
                {"name": "permitted", "type": "TokenPermissions"},
                {"name": "spender", "type": "address"},
                {"name": "nonce", "type": "uint256"},
                {"name": "deadline", "type": "uint256"},
                {"name": "witness", "type": "ExclusiveDutchOrder"},
            ],
            "TokenPermissions": [
                {"name": "token", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ],
            "ExclusiveDutchOrder": [
                {"name": "reactor", "type": "address"},
                {"name": "swapper", "type": "address"},
                {"name": "nonce", "type": "uint256"},
                {"name": "deadline", "type": "uint256"},
                {"name": "decayStartTime", "type": "uint256"},
                {"name": "decayEndTime", "type": "uint256"},
                {"name": "exclusiveFiller", "type": "address"},
                {"name": "exclusivityOverrideBps", "type": "uint256"},
                {"name": "inputToken", "type": "address"},
                {"name": "inputStartAmount", "type": "uint256"},
                {"name": "inputEndAmount", "type": "uint256"},
                {"name": "outputs", "type": "DutchOutput[]"},
            ],
            "DutchOutput": [
                {"name": "token", "type": "address"},
                {"name": "startAmount", "type": "uint256"},
                {"name": "endAmount", "type": "uint256"},
                {"name": "recipient", "type": "address"},
            ],
        },
        "primaryType": "PermitWitnessTransferFrom",
        "domain": {
            "name": "Permit2",
            "chainId": intent.chain_id,
            "verifyingContract": PERMIT2_ADDRESS,
        },
        "message": {
            "permitted": {
                "token": intent.input.token,
                "amount": intent.input.start_amount,
            },
            "spender": intent.reactor,
            "nonce": intent.nonce,
            "deadline": intent.deadline_unix_s,
            "witness": {
                "reactor": intent.reactor,
                "swapper": intent.swapper,
                "nonce": intent.nonce,
                "deadline": intent.deadline_unix_s,
                "decayStartTime": intent.decay_start_time_unix_s,
                "decayEndTime": intent.decay_end_time_unix_s,
                "exclusiveFiller": intent.exclusive_filler,
                "exclusivityOverrideBps": (
                    intent.exclusivity_override_bps
                ),
                "inputToken": intent.input.token,
                "inputStartAmount": intent.input.start_amount,
                "inputEndAmount": intent.input.end_amount,
                "outputs": [
                    {
                        "token": out.token,
                        "startAmount": out.start_amount,
                        "endAmount": out.end_amount,
                        "recipient": out.recipient,
                    }
                    for out in intent.outputs
                ],
            },
        },
    }


@dataclass(frozen=True)
class SignedIntent:
    """Output of :func:`sign_typed_data`.

    Attributes:
        signature: 65-byte ECDSA signature, ``0x``-prefixed hex.
        signer_address: Address recovered from the signature; the
            adapter compares this against the configured wallet to
            catch a key-mismatch *before* it touches the network.
    """

    signature: str
    signer_address: str


def sign_typed_data(
    *, private_key: str, typed_data: dict[str, Any]
) -> SignedIntent:
    """Sign ``typed_data`` with ``private_key`` (EIP-712).

    Args:
        private_key: ``0x``-prefixed hex private key (32 bytes).
        typed_data: Result of
            :func:`build_exclusive_dutch_order_typed_data`.

    Returns:
        :class:`SignedIntent` carrying the hex signature and the
        recovered signer address.

    Raises:
        ValueError: If ``private_key`` is not a valid hex private key.
    """
    if not private_key:
        raise ValueError("private_key required")
    account: LocalAccount = Account.from_key(private_key)
    encoded = encode_typed_data(full_message=typed_data)
    signed = account.sign_message(encoded)
    sig: str = signed.signature.hex()
    if not sig.startswith("0x"):
        sig = "0x" + sig
    return SignedIntent(
        signature=sig, signer_address=account.address
    )


__all__ = [
    "PERMIT2_ADDRESS",
    "DutchInput",
    "DutchOutput",
    "ExclusiveDutchOrderIntent",
    "SignedIntent",
    "build_exclusive_dutch_order_typed_data",
    "sign_typed_data",
]
