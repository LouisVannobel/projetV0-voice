"""Typed persistence commands and canonical encrypted-operation envelopes."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, overload

from pydantic import ValidationError

from projetv0_voice.audio_contract import (
    MAX_AUDIO_AAD_BYTES,
    MAX_AUDIO_ENVELOPE_BYTES,
    MAX_AUDIO_METADATA_BYTES,
    AudioChunkPayloadV2,
    VoiceOperationV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.crypto import AES_GCM_NONCE_BYTES, CryptoKeyring, EncryptedValue
from projetv0_voice.models import VoiceOperationV1

MAX_ENCRYPTED_COMMAND_BYTES = 65_536
AAD_PREFIX = b"projetv0-voice/outbox/aes-256-gcm/v1\x00"


class PersistenceError(RuntimeError):
    """A persistence error whose message is a constant safe code."""


class FatalPersistenceError(PersistenceError):
    """The local persistence boundary can no longer accept work."""


class CommandConflictError(FatalPersistenceError):
    """A durable identity was reused with different canonical content."""


class CommandSerializationError(FatalPersistenceError):
    """A command could not be canonically serialized or decoded."""


class EncryptedCommandTooLarge(FatalPersistenceError):
    """A stored encrypted envelope exceeds the fixed v1 limit."""


@dataclass(frozen=True, slots=True)
class PersistenceCommand:
    kind: Literal[
        "webhook_effect",
        "webhook_receipt_status",
        "webhook_enrichment",
        "lease",
        "outbox",
        "relay_batch",
        "qualification_run_status",
        "call_lifecycle_read",
        "sparra_activation",
        "transfer_intent",
        "transfer_observation",
        "sparra_content",
        "recording_archive",
        "shutdown",
    ]
    payload: Mapping[str, object] = field(repr=False)
    committed: asyncio.Future[None] | None = field(repr=False)
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class PreparedOperation:
    operation: VoiceOperationV1 = field(repr=False)
    plaintext: bytes = field(repr=False)
    aad: bytes = field(repr=False)
    encrypted: EncryptedValue = field(repr=False)
    envelope_size: int


@dataclass(frozen=True, slots=True)
class PreparedAudioOperation:
    operation: VoiceOperationV2 = field(repr=False)
    plaintext: bytes = field(repr=False)
    aad: bytes = field(repr=False)
    encrypted: EncryptedValue = field(repr=False)
    envelope_size: int


@overload
def canonical_operation_bytes(operation: VoiceOperationV1) -> bytes: ...


@overload
def canonical_operation_bytes(operation: VoiceOperationV2) -> bytes: ...


def canonical_operation_bytes(operation: VoiceOperationV1 | VoiceOperationV2) -> bytes:
    try:
        return json.dumps(
            operation.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CommandSerializationError("command_serialization_failed") from error


def _aad_metadata(operation: VoiceOperationV1 | VoiceOperationV2) -> dict[str, object]:
    return {
        "call_id": str(operation.call_id),
        "deployment_id": operation.deployment_id,
        "kind": operation.kind,
        "operation_id": str(operation.operation_id),
        "schema_version": operation.schema_version,
    }


def operation_aad_from_metadata(metadata: Mapping[str, object]) -> bytes:
    required = ("call_id", "deployment_id", "kind", "operation_id", "schema_version")
    try:
        normalized = {name: metadata[name] for name in required}
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError) as error:
        raise CommandSerializationError("command_aad_failed") from error
    return AAD_PREFIX + encoded


@overload
def operation_aad(operation: VoiceOperationV1) -> bytes: ...


@overload
def operation_aad(operation: VoiceOperationV2) -> bytes: ...


def operation_aad(operation: VoiceOperationV1 | VoiceOperationV2) -> bytes:
    return operation_aad_from_metadata(_aad_metadata(operation))


def encrypt_operation(operation: VoiceOperationV1, keyring: CryptoKeyring) -> PreparedOperation:
    plaintext = canonical_operation_bytes(operation)
    if len(plaintext) > MAX_ENCRYPTED_COMMAND_BYTES:
        raise EncryptedCommandTooLarge("encrypted_command_too_large")
    aad = operation_aad(operation)
    encrypted = keyring.encrypt(plaintext, aad=aad)
    envelope_size = len(aad) + AES_GCM_NONCE_BYTES + len(encrypted.ciphertext)
    if envelope_size > MAX_ENCRYPTED_COMMAND_BYTES:
        raise EncryptedCommandTooLarge("encrypted_command_too_large")
    return PreparedOperation(
        operation=operation,
        plaintext=plaintext,
        aad=aad,
        encrypted=encrypted,
        envelope_size=envelope_size,
    )


def encrypt_audio_operation(
    operation: VoiceOperationV2, keyring: CryptoKeyring
) -> PreparedAudioOperation:
    if not isinstance(operation, VoiceOperationV2) or operation.kind not in {
        "audio.chunk", "audio.finish", "audio.revoke"
    }:
        raise CommandSerializationError("invalid_audio_command")
    plaintext = canonical_operation_bytes(operation)
    media_bytes = 0
    if isinstance(operation.payload, AudioChunkPayloadV2):
        canonical_audio_chunk_aad(operation)
        media_bytes = len(operation.payload.ciphertext_b64)
    aad = operation_aad(operation)
    if (
        len(plaintext) - media_bytes > MAX_AUDIO_METADATA_BYTES
        or len(aad) > MAX_AUDIO_AAD_BYTES
    ):
        raise EncryptedCommandTooLarge("encrypted_command_too_large")
    encrypted = keyring.encrypt(plaintext, aad=aad)
    envelope_size = len(aad) + AES_GCM_NONCE_BYTES + len(encrypted.ciphertext)
    if envelope_size > min(MAX_AUDIO_ENVELOPE_BYTES, MAX_ENCRYPTED_COMMAND_BYTES):
        raise EncryptedCommandTooLarge("encrypted_command_too_large")
    return PreparedAudioOperation(
        operation=operation,
        plaintext=plaintext,
        aad=aad,
        encrypted=encrypted,
        envelope_size=envelope_size,
    )


def decode_operation(plaintext: bytes) -> VoiceOperationV1:
    try:
        return VoiceOperationV1.model_validate_json(plaintext)
    except ValidationError as error:
        raise CommandSerializationError("command_deserialization_failed") from error


def decode_operation_v2(plaintext: bytes) -> VoiceOperationV2:
    try:
        return VoiceOperationV2.model_validate_json(plaintext)
    except ValidationError:
        pass
    raise CommandSerializationError("command_deserialization_failed")


def require_operation(payload: Mapping[str, object]) -> VoiceOperationV1:
    operation = payload.get("operation")
    if not isinstance(operation, VoiceOperationV1):
        raise CommandSerializationError("invalid_outbox_command")
    return operation
