"""Unit tests for the UniswapX EIP-712 signer (D3)."""

from __future__ import annotations

import pytest
from eth_account import Account
from eth_account.messages import encode_typed_data

from execution_engine.adapters._uniswapx_signer import (
    PERMIT2_ADDRESS,
    DutchInput,
    DutchOutput,
    ExclusiveDutchOrderIntent,
    build_exclusive_dutch_order_typed_data,
    sign_typed_data,
)

_PRIVATE_KEY = "0x" + "11" * 32


def _make_intent(**overrides: object) -> ExclusiveDutchOrderIntent:
    base = {
        "chain_id": 1,
        "reactor": "0x" + "a" * 40,
        "swapper": "0x" + "b" * 40,
        "nonce": 42,
        "deadline_unix_s": 2000,
        "decay_start_time_unix_s": 1000,
        "decay_end_time_unix_s": 1900,
        "exclusive_filler": "0x" + "0" * 40,
        "exclusivity_override_bps": 0,
        "input": DutchInput(
            token="0x" + "c" * 40,
            start_amount=1_000_000,
            end_amount=1_000_000,
        ),
        "outputs": (
            DutchOutput(
                token="0x" + "d" * 40,
                start_amount=900_000,
                end_amount=890_000,
                recipient="0x" + "b" * 40,
            ),
        ),
    }
    base.update(overrides)
    return ExclusiveDutchOrderIntent(**base)  # type: ignore[arg-type]


def test_intent_validates_decay_window() -> None:
    with pytest.raises(ValueError):
        _make_intent(
            decay_start_time_unix_s=2000, decay_end_time_unix_s=1000
        )


def test_intent_validates_deadline_after_decay_end() -> None:
    with pytest.raises(ValueError):
        _make_intent(
            decay_start_time_unix_s=1000,
            decay_end_time_unix_s=1900,
            deadline_unix_s=1500,
        )


def test_intent_validates_chain_id() -> None:
    with pytest.raises(ValueError):
        _make_intent(chain_id=0)


def test_intent_requires_outputs() -> None:
    with pytest.raises(ValueError):
        _make_intent(outputs=())


def test_typed_data_shape_is_canonical() -> None:
    td = build_exclusive_dutch_order_typed_data(_make_intent())
    assert td["primaryType"] == "PermitWitnessTransferFrom"
    assert td["domain"] == {
        "name": "Permit2",
        "chainId": 1,
        "verifyingContract": PERMIT2_ADDRESS,
    }
    types = td["types"]
    assert "PermitWitnessTransferFrom" in types
    assert "ExclusiveDutchOrder" in types
    assert "DutchOutput" in types
    assert "TokenPermissions" in types
    msg = td["message"]
    assert msg["spender"] == "0x" + "a" * 40
    assert msg["nonce"] == 42
    assert msg["deadline"] == 2000
    assert msg["permitted"]["amount"] == 1_000_000
    assert msg["witness"]["decayStartTime"] == 1000
    assert msg["witness"]["outputs"][0]["startAmount"] == 900_000


def test_typed_data_serialises_outputs_array() -> None:
    intent = _make_intent(
        outputs=(
            DutchOutput(
                token="0x" + "1" * 40,
                start_amount=100,
                end_amount=90,
                recipient="0x" + "b" * 40,
            ),
            DutchOutput(
                token="0x" + "2" * 40,
                start_amount=200,
                end_amount=180,
                recipient="0x" + "b" * 40,
            ),
        )
    )
    td = build_exclusive_dutch_order_typed_data(intent)
    outs = td["message"]["witness"]["outputs"]
    assert len(outs) == 2
    assert outs[0]["startAmount"] == 100
    assert outs[1]["startAmount"] == 200


def test_sign_typed_data_returns_recoverable_signature() -> None:
    """Sign with a known key, then recover the address from the
    signature and assert it matches the signer the helper reported.
    """
    td = build_exclusive_dutch_order_typed_data(_make_intent())
    signed = sign_typed_data(private_key=_PRIVATE_KEY, typed_data=td)
    expected_address = Account.from_key(_PRIVATE_KEY).address
    assert signed.signer_address == expected_address
    assert signed.signature.startswith("0x")
    # 65-byte signature == 130 hex chars + ``0x`` prefix == 132.
    assert len(signed.signature) == 132
    # Round-trip: recover the address from the EIP-712 signed message.
    encoded = encode_typed_data(full_message=td)
    recovered = Account.recover_message(
        encoded, signature=signed.signature
    )
    assert recovered == expected_address


def test_sign_is_deterministic_for_same_input() -> None:
    """ECDSA signatures from ``eth_account`` are deterministic
    (RFC 6979). Two calls with the same key + payload must produce
    byte-identical signatures, which we rely on for replay tests.
    """
    td = build_exclusive_dutch_order_typed_data(_make_intent())
    a = sign_typed_data(private_key=_PRIVATE_KEY, typed_data=td)
    b = sign_typed_data(private_key=_PRIVATE_KEY, typed_data=td)
    assert a.signature == b.signature


def test_sign_different_keys_produce_different_signatures() -> None:
    td = build_exclusive_dutch_order_typed_data(_make_intent())
    a = sign_typed_data(private_key=_PRIVATE_KEY, typed_data=td)
    b = sign_typed_data(
        private_key="0x" + "22" * 32, typed_data=td
    )
    assert a.signature != b.signature
    assert a.signer_address != b.signer_address


def test_sign_rejects_empty_private_key() -> None:
    td = build_exclusive_dutch_order_typed_data(_make_intent())
    with pytest.raises(ValueError):
        sign_typed_data(private_key="", typed_data=td)
