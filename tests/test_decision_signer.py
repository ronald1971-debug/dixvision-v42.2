"""Hardening-S1 item 2 — DecisionSigner unit tests.

The signer is a pure HMAC-SHA256 primitive: round-trip success,
constant-time mismatch on tamper, and per-instance key isolation.
"""

from __future__ import annotations

import pytest

from governance_engine.control_plane.decision_signer import (
    DECISION_SIGNATURE_BYTE_LENGTH,
    DecisionSigner,
    SignatureMismatchError,
    canonical_signing_input,
)


def test_signer_round_trip_with_explicit_secret():
    secret = b"\x00" * DECISION_SIGNATURE_BYTE_LENGTH
    signer = DecisionSigner(secret=secret)
    sig = signer.sign(content_hash="abc123", governance_decision_id="GOV-1")
    assert signer.verify(
        content_hash="abc123",
        governance_decision_id="GOV-1",
        signature=sig,
    )


def test_signer_round_trip_with_random_secret():
    signer = DecisionSigner()
    sig = signer.sign(content_hash="abc123", governance_decision_id="GOV-1")
    assert signer.verify(
        content_hash="abc123",
        governance_decision_id="GOV-1",
        signature=sig,
    )


def test_two_signers_have_different_keys():
    """Each instance mints its own secret -- a signature from one
    instance must never verify against another."""

    a = DecisionSigner()
    b = DecisionSigner()
    sig_a = a.sign(content_hash="abc123", governance_decision_id="GOV-1")
    assert not b.verify(
        content_hash="abc123",
        governance_decision_id="GOV-1",
        signature=sig_a,
    )


def test_signer_rejects_short_secret():
    with pytest.raises(ValueError):
        DecisionSigner(secret=b"\x00" * (DECISION_SIGNATURE_BYTE_LENGTH - 1))


def test_signer_rejects_non_bytes_secret():
    with pytest.raises(TypeError):
        DecisionSigner(secret="not-bytes")  # type: ignore[arg-type]


def test_signer_rejects_empty_signature_in_verify():
    signer = DecisionSigner()
    assert not signer.verify(
        content_hash="abc123",
        governance_decision_id="GOV-1",
        signature="",
    )


def test_signer_rejects_tampered_content_hash():
    signer = DecisionSigner()
    sig = signer.sign(content_hash="abc123", governance_decision_id="GOV-1")
    assert not signer.verify(
        content_hash="abc124",
        governance_decision_id="GOV-1",
        signature=sig,
    )


def test_signer_rejects_tampered_decision_id():
    signer = DecisionSigner()
    sig = signer.sign(content_hash="abc123", governance_decision_id="GOV-1")
    assert not signer.verify(
        content_hash="abc123",
        governance_decision_id="GOV-2",
        signature=sig,
    )


def test_assert_verified_raises_on_mismatch():
    signer = DecisionSigner()
    with pytest.raises(SignatureMismatchError):
        signer.assert_verified(
            content_hash="abc123",
            governance_decision_id="GOV-1",
            signature="deadbeef",
        )


def test_canonical_input_rejects_empty_fields():
    with pytest.raises(ValueError):
        canonical_signing_input(content_hash="", governance_decision_id="GOV-1")
    with pytest.raises(ValueError):
        canonical_signing_input(content_hash="abc", governance_decision_id="")


def test_canonical_input_format_is_versioned_and_delimited():
    """The canonical input includes a version tag so a future format
    change can break verification deterministically rather than
    silently colliding with old signatures."""

    payload = canonical_signing_input(
        content_hash="abc",
        governance_decision_id="GOV-1",
    )
    assert payload.startswith(b"DIX/HMAC-SHA256/v1\x1f")
    assert payload.endswith(b"GOV-1")
    # Unit Separator (0x1F) splits the three sections.
    assert payload.count(b"\x1f") == 2


def test_signing_is_deterministic_for_same_secret():
    secret = b"\x42" * DECISION_SIGNATURE_BYTE_LENGTH
    a = DecisionSigner(secret=secret)
    b = DecisionSigner(secret=secret)
    sig_a = a.sign(content_hash="abc", governance_decision_id="GOV-1")
    sig_b = b.sign(content_hash="abc", governance_decision_id="GOV-1")
    assert sig_a == sig_b
