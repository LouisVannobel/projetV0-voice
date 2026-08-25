"""Versioned transcript encryption using the upstream AES-GCM primitive."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CRYPTO_VERSION = 1
AES_256_KEY_BYTES = 32
AES_GCM_NONCE_BYTES = 12
AES_GCM_TAG_BYTES = 16


class CryptoError(RuntimeError):
    """A safe crypto boundary error containing only a constant code."""


class InvalidKeyringError(CryptoError):
    """The injected keyring does not satisfy the v1 contract."""


class UnknownKeyVersionError(CryptoError):
    """A ciphertext references a key version that is not retained."""


class CryptoDecryptionError(CryptoError):
    """An encrypted value failed structural or authenticity checks."""


@dataclass(frozen=True, slots=True)
class EncryptedValue:
    """A redacted AES-GCM envelope."""

    key_version: int
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)


class CryptoKeyring:
    """Immutable versioned AES-256-GCM keys with one active write version."""

    __slots__ = ("_keys", "_nonce_factory", "active_version")

    def __init__(
        self,
        keys: Mapping[int, bytes] | Iterable[tuple[int, bytes]],
        *,
        active_version: int,
        nonce_factory: Callable[[int], bytes] = os.urandom,
    ) -> None:
        if type(active_version) is not int or active_version <= 0:
            raise InvalidKeyringError("invalid_keyring")
        items = list(keys.items()) if isinstance(keys, Mapping) else list(keys)
        if not items:
            raise InvalidKeyringError("invalid_keyring")

        copied: dict[int, bytes] = {}
        for version, raw_key in items:
            if (
                type(version) is not int
                or version <= 0
                or version in copied
                or type(raw_key) is not bytes
                or len(raw_key) != AES_256_KEY_BYTES
            ):
                raise InvalidKeyringError("invalid_keyring")
            copied[version] = bytes(bytearray(raw_key))
        if active_version not in copied:
            raise InvalidKeyringError("invalid_keyring")

        self._keys = MappingProxyType(copied)
        self._nonce_factory = nonce_factory
        self.active_version = active_version

    def __repr__(self) -> str:
        versions = tuple(sorted(self._keys))
        return f"CryptoKeyring(active_version={self.active_version}, key_versions={versions!r})"

    @staticmethod
    def _require_aad(aad: bytes) -> None:
        if type(aad) is not bytes or not aad:
            raise ValueError("AAD must be non-empty bytes")

    def encrypt(self, plaintext: bytes, *, aad: bytes) -> EncryptedValue:
        self._require_aad(aad)
        if type(plaintext) is not bytes:
            raise TypeError("plaintext must be bytes")
        nonce = self._nonce_factory(AES_GCM_NONCE_BYTES)
        if type(nonce) is not bytes or len(nonce) != AES_GCM_NONCE_BYTES:
            raise CryptoError("nonce_generation_failed")
        key = self._keys[self.active_version]
        return EncryptedValue(
            key_version=self.active_version,
            nonce=nonce,
            ciphertext=AESGCM(key).encrypt(nonce, plaintext, aad),
        )

    def decrypt(self, value: EncryptedValue, *, aad: bytes) -> bytes:
        self._require_aad(aad)
        if type(value.key_version) is not int or value.key_version <= 0:
            raise UnknownKeyVersionError("unknown_key_version")
        key = self._keys.get(value.key_version)
        if key is None:
            raise UnknownKeyVersionError("unknown_key_version")
        if (
            len(value.nonce) != AES_GCM_NONCE_BYTES
            or len(value.ciphertext) < AES_GCM_TAG_BYTES
        ):
            raise CryptoDecryptionError("crypto_decryption_failed")
        try:
            return AESGCM(key).decrypt(value.nonce, value.ciphertext, aad)
        except InvalidTag as error:
            raise CryptoDecryptionError("crypto_decryption_failed") from error
