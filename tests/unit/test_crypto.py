from __future__ import annotations

import logging

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from projetv0_voice.crypto import (
    CryptoDecryptionError,
    CryptoKeyring,
    EncryptedValue,
    InvalidKeyringError,
    UnknownKeyVersionError,
)

KEY_V1 = bytes(range(32))
KEY_V2 = bytes(range(32, 64))


def test_aesgcm_v1_round_trip_uses_active_version_and_unique_nonce() -> None:
    keyring = CryptoKeyring(((1, KEY_V1), (2, KEY_V2)), active_version=2)

    first = keyring.encrypt(b"bonjour", aad=b"turn:1")
    second = keyring.encrypt(b"bonjour", aad=b"turn:1")

    assert first.key_version == 2
    assert len(first.nonce) == 12
    assert first.nonce != second.nonce
    assert first.ciphertext != b"bonjour"
    assert keyring.decrypt(first, aad=b"turn:1") == b"bonjour"
    assert first.nonce.hex() not in repr(first)
    assert first.ciphertext.hex() not in repr(first)


@pytest.mark.parametrize("aad", [b"", bytearray()])
def test_aad_is_mandatory(aad: bytes | bytearray) -> None:
    keyring = CryptoKeyring({1: KEY_V1}, active_version=1)

    with pytest.raises(ValueError, match="AAD must be non-empty"):
        keyring.encrypt(b"content", aad=bytes(aad))


def test_wrong_aad_and_tamper_fail_closed_without_sensitive_values() -> None:
    keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    value = keyring.encrypt(b"TOP-SECRET-TRANSCRIPT", aad=b"correct-aad")
    tampered = EncryptedValue(
        key_version=value.key_version,
        nonce=value.nonce,
        ciphertext=value.ciphertext[:-1] + bytes([value.ciphertext[-1] ^ 1]),
    )

    for candidate, aad in ((value, b"wrong-aad"), (tampered, b"correct-aad")):
        with pytest.raises(CryptoDecryptionError) as caught:
            keyring.decrypt(candidate, aad=aad)
        rendered = repr(caught.value)
        assert "TOP-SECRET-TRANSCRIPT" not in rendered
        assert value.ciphertext.hex() not in rendered
        assert KEY_V1.hex() not in rendered


def test_retained_old_key_decrypts_and_unknown_version_is_safe() -> None:
    nonce = bytes(range(12))
    old_value = EncryptedValue(
        key_version=1,
        nonce=nonce,
        ciphertext=AESGCM(KEY_V1).encrypt(nonce, b"restored turn", b"restore:v1"),
    )
    keyring = CryptoKeyring({1: KEY_V1, 2: KEY_V2}, active_version=2)

    assert keyring.decrypt(old_value, aad=b"restore:v1") == b"restored turn"

    unknown = EncryptedValue(key_version=9, nonce=nonce, ciphertext=old_value.ciphertext)
    with pytest.raises(UnknownKeyVersionError) as caught:
        keyring.decrypt(unknown, aad=b"restore:v1")
    assert KEY_V1.hex() not in repr(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        EncryptedValue(key_version=1, nonce=b"short", ciphertext=b"x" * 16),
        EncryptedValue(key_version=1, nonce=b"n" * 12, ciphertext=b"x" * 15),
    ],
)
def test_v1_decrypt_rejects_structurally_invalid_envelopes(value: EncryptedValue) -> None:
    keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    with pytest.raises(CryptoDecryptionError):
        keyring.decrypt(value, aad=b"required")

    rendered = repr(value)
    assert "nonce" not in rendered
    assert "ciphertext" not in rendered


def test_decrypt_rejects_empty_aad() -> None:
    keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    value = keyring.encrypt(b"content", aad=b"valid")
    with pytest.raises(ValueError, match="AAD must be non-empty"):
        keyring.decrypt(value, aad=b"")


@pytest.mark.parametrize(
    ("keys", "active_version"),
    [
        ([], 1),
        ([(1, KEY_V1), (1, KEY_V2)], 1),
        ({True: KEY_V1}, 1),
        ({0: KEY_V1}, 0),
        ({1: b"short"}, 1),
        ({1: KEY_V1}, 2),
        ({1: KEY_V1}, True),
        ({1: KEY_V1}, 1.0),
    ],
)
def test_invalid_keyrings_are_rejected(
    keys: object, active_version: int
) -> None:
    with pytest.raises(InvalidKeyringError):
        CryptoKeyring(keys, active_version=active_version)  # type: ignore[arg-type]


def test_keyring_repr_and_logs_never_disclose_key_material(
    caplog: pytest.LogCaptureFixture,
) -> None:
    keyring = CryptoKeyring({1: KEY_V1, 2: KEY_V2}, active_version=2)

    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test.crypto").debug("keyring=%r", keyring)

    rendered = repr(keyring) + caplog.text
    assert "active_version=2" in rendered
    assert KEY_V1.hex() not in rendered
    assert KEY_V2.hex() not in rendered
    assert str(list(KEY_V1)) not in rendered
