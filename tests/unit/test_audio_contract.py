"""Offline RED contracts for the native V2 local-audio wire and AEAD consumers."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from projetv0_voice.audio_contract import (
    AudioChunkPayloadV2,
    AudioFinishPayloadV2,
    AudioRevokePayloadV2,
    VoiceOperationV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.crypto import CryptoDecryptionError, CryptoKeyring, EncryptedValue
from projetv0_voice.models import (
    CallUpsertPayloadV1,
    RecordingUpsertPayloadV1,
    TurnUpsertPayloadV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import (
    CommandSerializationError,
    canonical_operation_bytes,
    decode_operation,
    decode_operation_v2,
    encrypt_audio_operation,
    encrypt_operation,
    operation_aad,
)

WORKSPACE_ID = "11111111-1111-4111-8111-111111111111"
CALL_ID = "22222222-2222-4222-8222-222222222222"
OPERATION_ID = "33333333-3333-4333-8333-333333333333"
RECORDING_ID = "55555555-5555-4555-8555-555555555555"
RETENTION = "2026-10-31T10:00:00.123Z"
KEY = bytes(range(32))
AAD_DOMAIN = b"sparra.audio.chunk.v1\x00"


def chunk_payload(**updates: object) -> dict[str, object]:
    return {
        "schema_version": 2,
        "workspace_id": WORKSPACE_ID,
        "recording_id": RECORDING_ID,
        "sequence": 0,
        "sample_count": 8000,
        "sample_rate": 8000,
        "channels": 2,
        "sample_format": "s16le",
        "configuration_revision": 7,
        "retention_until": RETENTION,
        "crypto_version": 1,
        "key_version": 1,
        "nonce_b64": base64.b64encode(bytes(range(12))).decode("ascii"),
        "ciphertext_b64": base64.b64encode(b"x" * 32016).decode("ascii"),
        **updates,
    }


def finish_payload(**updates: object) -> dict[str, object]:
    return {
        "schema_version": 2,
        "workspace_id": WORKSPACE_ID,
        "recording_id": RECORDING_ID,
        "configuration_revision": 7,
        "retention_until": RETENTION,
        "last_sequence": 599,
        "total_samples": 4_800_000,
        "reason": "complete",
        **updates,
    }


def revoke_payload(**updates: object) -> dict[str, object]:
    return {
        "schema_version": 2,
        "workspace_id": WORKSPACE_ID,
        "recording_id": RECORDING_ID,
        "configuration_revision": 7,
        "retention_until": RETENTION,
        "reason": "caller_declined",
        **updates,
    }


def operation(kind: str, payload: dict[str, object], **updates: object) -> dict[str, object]:
    return {
        "schema_version": 2,
        "operation_id": OPERATION_ID,
        "deployment_id": "voice-agent-a",
        "call_id": CALL_ID,
        "occurred_at": "2026-10-01T10:00:01.123000Z",
        "kind": kind,
        "payload": payload,
        **updates,
    }


LEGACY_PAYLOADS = {
    "call.upsert": {
        "telnyx_call_control_id": "fixture-control",
        "telnyx_call_leg_id": None,
        "telnyx_call_session_id": None,
        "status": "pending",
        "disclosure_state": "pending",
        "started_at": None,
        "ended_at": None,
        "end_reason": None,
        "retention_until": "2026-10-31T10:00:00.123000Z",
    },
    "turn.upsert": {
        "turn_id": "44444444-4444-4444-8444-444444444444",
        "turn_no": 1,
        "role": "user",
        "source": "stt_final",
        "crypto_version": 1,
        "key_version": 1,
        "nonce_b64": base64.b64encode(bytes(range(12))).decode("ascii"),
        "ciphertext_b64": base64.b64encode(b"x" * 20).decode("ascii"),
        "started_at": "2026-10-01T10:00:00.123000Z",
        "ended_at": "2026-10-01T10:00:01.123000Z",
        "interrupted": False,
    },
    "recording.upsert": {
        "recording_id": RECORDING_ID,
        "status": "purged",
        "telnyx_recording_id": "fixture-provider-recording",
        "channels": "dual",
        "format": "wav",
        "started_at": "2026-10-01T10:00:00.123000Z",
        "ended_at": "2026-10-01T10:00:01.123000Z",
        "retention_until": "2026-10-31T10:00:00.123000Z",
    },
}


@pytest.mark.parametrize(
    ("kind", "payload", "payload_type"),
    [
        ("audio.chunk", chunk_payload(), AudioChunkPayloadV2),
        ("audio.finish", finish_payload(), AudioFinishPayloadV2),
        ("audio.revoke", revoke_payload(), AudioRevokePayloadV2),
        ("call.upsert", LEGACY_PAYLOADS["call.upsert"], CallUpsertPayloadV1),
        ("turn.upsert", LEGACY_PAYLOADS["turn.upsert"], TurnUpsertPayloadV1),
        ("recording.upsert", LEGACY_PAYLOADS["recording.upsert"], RecordingUpsertPayloadV1),
    ],
)
def test_v2_fixed_decoder_round_trips_the_matching_typed_payload(kind, payload, payload_type):
    wire = operation(kind, deepcopy(payload))
    decoded = decode_operation_v2(json.dumps(wire).encode("utf-8"))
    assert type(decoded) is VoiceOperationV2
    assert type(decoded.payload) is payload_type
    assert decoded.model_dump(mode="json") == wire
    assert decode_operation_v2(decoded.model_dump_json().encode("utf-8")) == decoded
    with pytest.raises(ValidationError, match="frozen"):
        decoded.call_id = UUID(int=9)


@pytest.mark.parametrize("value", [True, False, 1, 2.0, "2", None, 3])
@pytest.mark.parametrize("location", ["operation", "payload"])
def test_audio_schema_versions_are_exact_integer_two(value, location):
    payload = chunk_payload()
    wire = operation("audio.chunk", payload)
    wire["schema_version"] = value if location == "operation" else 2
    payload["schema_version"] = value if location == "payload" else 2
    with pytest.raises(ValidationError):
        VoiceOperationV2.model_validate(wire)


@pytest.mark.parametrize("version", [1, True, False, 2.0, "2", 3, None])
def test_v2_decoder_never_falls_back_to_the_legacy_contract(version):
    wire = operation(
        "call.upsert", deepcopy(LEGACY_PAYLOADS["call.upsert"]), schema_version=version
    )
    with pytest.raises(CommandSerializationError, match="command_deserialization_failed"):
        decode_operation_v2(json.dumps(wire).encode("utf-8"))


@pytest.mark.parametrize("version", [2, True, False, 3])
def test_legacy_decoder_stays_v1_only(version):
    wire = operation(
        "call.upsert", deepcopy(LEGACY_PAYLOADS["call.upsert"]), schema_version=version
    )
    with pytest.raises(CommandSerializationError, match="command_deserialization_failed"):
        decode_operation(json.dumps(wire).encode("utf-8"))


@pytest.mark.parametrize("sequence", [0, 599])
@pytest.mark.parametrize(("samples", "ciphertext_bytes"), [(1, 20), (8000, 32016)])
def test_audio_chunk_accepts_exact_sequence_and_pcm_frame_boundaries(
    sequence, samples, ciphertext_bytes
):
    parsed = AudioChunkPayloadV2.model_validate(
        chunk_payload(
            sequence=sequence,
            sample_count=samples,
            ciphertext_b64=base64.b64encode(b"x" * ciphertext_bytes).decode("ascii"),
        )
    )
    assert parsed.sequence == sequence
    assert parsed.sample_count == samples
    assert (parsed.sample_rate, parsed.channels, parsed.sample_format) == (8000, 2, "s16le")


@pytest.mark.parametrize(
    "updates",
    [
        {"sequence": -1}, {"sequence": 600}, {"sequence": True},
        {"sequence": 1.0}, {"sequence": "1"}, {"sequence": None},
        {"sample_count": 0}, {"sample_count": 8001}, {"sample_count": -1},
        {"sample_count": True, "ciphertext_b64": base64.b64encode(b"x" * 20).decode()},
        {"sample_count": 1.0, "ciphertext_b64": base64.b64encode(b"x" * 20).decode()},
        {"sample_count": "1", "ciphertext_b64": base64.b64encode(b"x" * 20).decode()},
        {"sample_rate": 16000}, {"sample_rate": 8000.0}, {"sample_rate": "8000"},
        {"channels": 1}, {"channels": 2.0}, {"channels": "2"},
        {"sample_format": "f32le"}, {"sample_format": "s16be"},
        {"configuration_revision": 0}, {"configuration_revision": 2_147_483_648},
        {"configuration_revision": True}, {"configuration_revision": 7.0},
        {"crypto_version": True}, {"crypto_version": 1.0}, {"crypto_version": 2},
        {"key_version": True}, {"key_version": 0}, {"key_version": 1.0},
        {"key_version": 9_007_199_254_740_992},
    ],
)
def test_audio_chunk_rejects_coercion_wrong_format_and_numeric_overflow(updates):
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(chunk_payload(**updates))


@pytest.mark.parametrize("nonce_bytes", [0, 11, 13])
def test_audio_chunk_requires_a_twelve_byte_nonce(nonce_bytes):
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(
            chunk_payload(nonce_b64=base64.b64encode(b"n" * nonce_bytes).decode("ascii"))
        )


@pytest.mark.parametrize("ciphertext_bytes", [0, 15, 16, 19, 21, 32016, 32017])
def test_audio_ciphertext_must_match_four_pcm_bytes_per_sample_plus_tag(ciphertext_bytes):
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(
            chunk_payload(
                sample_count=1,
                ciphertext_b64=base64.b64encode(b"x" * ciphertext_bytes).decode("ascii"),
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nonce_b64", "AAECAwQFBgcICQoL="),
        ("nonce_b64", "AAECAwQFBgcI\nCQoL"),
        ("nonce_b64", "________________"),
        ("nonce_b64", b"AAECAwQFBgcICQoL"),
        ("ciphertext_b64", "A" * 26 + "B="),
        ("ciphertext_b64", "A" * 27 + "=\n"),
        ("ciphertext_b64", "A" * 27),
        ("ciphertext_b64", None),
    ],
)
def test_audio_base64_is_canonical_text_with_no_alphabet_padding_or_pad_bit_aliases(field, value):
    payload = chunk_payload(
        sample_count=1, ciphertext_b64=base64.b64encode(bytes(20)).decode("ascii")
    )
    payload[field] = value
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("nonce_b64", "A" * 17), ("ciphertext_b64", "A" * 42689)],
    ids=["nonce-max-plus-one", "cipher-max-plus-one"],
)
def test_oversized_audio_media_is_rejected_before_base64_allocation(monkeypatch, field, value):
    def forbidden_decode(*args, **kwargs):
        raise RuntimeError("oversized media reached the native decoder")

    monkeypatch.setattr("projetv0_voice.audio_contract.base64.b64decode", forbidden_decode)
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(chunk_payload(**{field: value}))


@pytest.mark.parametrize("field", ["workspace_id", "recording_id"])
@pytest.mark.parametrize(
    "value",
    ["not-a-uuid", "{11111111-1111-4111-8111-111111111111}",
     "EEEEEEEE-EEEE-4EEE-8EEE-EEEEEEEEEEEE", 1, True, None],
)
def test_audio_pin_identifiers_require_canonical_uuids(field, value):
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(chunk_payload(**{field: value}))


@pytest.mark.parametrize("field", ["operation_id", "call_id"])
def test_v2_outer_identities_reject_noncanonical_uuid_aliases(field):
    with pytest.raises(ValidationError):
        VoiceOperationV2.model_validate(
            operation("audio.chunk", chunk_payload(), **{field: "E" * 32})
        )


@pytest.mark.parametrize(
    "deadline",
    [True, 0, None, "infinity", "2026-02-30T10:00:00.123Z",
     "2026-10-31T10:00:00.123", "2026-10-31T12:00:00.123+02:00",
     "2026-10-31T10:00:00Z", "2026-10-31T10:00:00.123456Z",
     datetime(2026, 10, 31, 10, 0, 0, 123456, tzinfo=UTC)],
)
def test_audio_original_expiry_rejects_noncanonical_or_nonfinite_deadlines(deadline):
    with pytest.raises(ValidationError):
        AudioChunkPayloadV2.model_validate(chunk_payload(retention_until=deadline))


def test_audio_original_expiry_and_configuration_pin_survive_native_utc_input():
    original = datetime(2026, 10, 31, 10, 0, 0, 123000, tzinfo=UTC)
    payload = AudioChunkPayloadV2.model_validate(chunk_payload(retention_until=original))
    assert payload.retention_until == original
    assert payload.configuration_revision == 7
    assert payload.model_dump(mode="json")["retention_until"] == RETENTION


@pytest.mark.parametrize("reason", ["complete", "transfer", "interrupted", "limit", "failure"])
def test_audio_finish_is_bounded_metadata_only_with_explicit_terminal_reason(reason):
    parsed = AudioFinishPayloadV2.model_validate(finish_payload(reason=reason))
    assert parsed.last_sequence == 599
    assert parsed.total_samples == 4_800_000
    empty = AudioFinishPayloadV2.model_validate(
        finish_payload(last_sequence=None, total_samples=0, reason=reason)
    )
    assert empty.last_sequence is None and empty.total_samples == 0


@pytest.mark.parametrize(
    "updates",
    [{"last_sequence": 600}, {"last_sequence": -1}, {"last_sequence": True},
     {"last_sequence": 1.0}, {"last_sequence": None, "total_samples": 1},
     {"last_sequence": 0, "total_samples": 0}, {"total_samples": -1},
     {"total_samples": 4_800_001}, {"total_samples": True},
     {"total_samples": 8000.0}, {"reason": "resumed"}, {"reason": "x" * 257},
     {"last_sequence": 0, "total_samples": 8001},
     {"last_sequence": 599, "total_samples": 599}],
)
def test_audio_finish_rejects_impossible_empty_or_overflowed_completion(updates):
    with pytest.raises(ValidationError):
        AudioFinishPayloadV2.model_validate(finish_payload(**updates))


@pytest.mark.parametrize(
    ("payload_type", "payload"),
    [(AudioFinishPayloadV2, finish_payload()), (AudioRevokePayloadV2, revoke_payload())],
)
@pytest.mark.parametrize("media", ["nonce_b64", "ciphertext_b64", "pcm", "sample_count"])
def test_audio_finish_and_revoke_cannot_carry_media(payload_type, payload, media):
    with pytest.raises(ValidationError):
        payload_type.model_validate({**payload, media: "fixture-only-media"})


def test_audio_revoke_remains_valid_after_original_expiry_for_cleanup():
    wire = operation("audio.revoke", revoke_payload(), occurred_at="2026-11-01T10:00:00.123000Z")
    parsed = decode_operation_v2(json.dumps(wire).encode())
    assert type(parsed.payload) is AudioRevokePayloadV2
    assert parsed.payload.reason == "caller_declined"
    assert parsed.model_dump(mode="json") == wire
    with pytest.raises(ValidationError):
        AudioRevokePayloadV2.model_validate(revoke_payload(reason="resume"))


@pytest.mark.parametrize(
    ("kind", "payload"),
    [("audio.chunk", finish_payload()), ("audio.finish", revoke_payload()),
     ("audio.revoke", chunk_payload()), ("call.upsert", chunk_payload()),
     ("audio.chunk", LEGACY_PAYLOADS["turn.upsert"])],
)
def test_v2_kind_selects_only_its_own_strict_payload(kind, payload):
    with pytest.raises(ValidationError):
        VoiceOperationV2.model_validate(operation(kind, deepcopy(payload)))


@pytest.mark.parametrize("kind", ["audio.chunk", "audio.finish", "audio.revoke"])
@pytest.mark.parametrize("location", ["operation", "payload"])
def test_v2_unknown_fields_and_nested_kind_cannot_add_an_authority(kind, location):
    payloads = {"audio.chunk": chunk_payload(), "audio.finish": finish_payload(),
                "audio.revoke": revoke_payload()}
    wire = operation(kind, payloads[kind])
    if location == "operation":
        wire["workspace_id"] = WORKSPACE_ID
    else:
        payloads[kind]["kind"] = kind
    with pytest.raises(ValidationError):
        VoiceOperationV2.model_validate(wire)


def test_audio_aad_has_an_independent_canonical_domain_and_immutable_pin_golden_value():
    parsed = VoiceOperationV2.model_validate(operation("audio.chunk", chunk_payload()))
    expected = (
        b"sparra.audio.chunk.v1\x00"
        b'{"call_id":"22222222-2222-4222-8222-222222222222","channels":2,'
        b'"configuration_revision":7,"crypto_version":1,"deployment_id":"voice-agent-a",'
        b'"key_version":1,"recording_id":"55555555-5555-4555-8555-555555555555",'
        b'"retention_until":"2026-10-31T10:00:00.123Z","sample_count":8000,'
        b'"sample_format":"s16le","sample_rate":8000,"schema_version":2,"sequence":0,'
        b'"workspace_id":"11111111-1111-4111-8111-111111111111"}'
    )
    assert canonical_audio_chunk_aad(parsed) == expected
    replay = VoiceOperationV2.model_validate(
        operation("audio.chunk", chunk_payload(), operation_id=str(UUID(int=9)),
                  occurred_at="2026-10-01T10:00:02.123000Z")
    )
    assert canonical_audio_chunk_aad(replay) == expected


@pytest.mark.parametrize(
    ("location", "updates"),
    [("operation", {"call_id": str(UUID(int=9))}),
     ("operation", {"deployment_id": "voice-agent-b"}),
     ("payload", {"workspace_id": str(UUID(int=9))}),
     ("payload", {"recording_id": str(UUID(int=9))}),
     ("payload", {"sequence": 1}),
     ("payload", {"sample_count": 7999,
                  "ciphertext_b64": base64.b64encode(b"x" * 32012).decode("ascii")}),
     ("payload", {"configuration_revision": 8}),
     ("payload", {"retention_until": "2026-10-31T10:00:00.124Z"}),
     ("payload", {"key_version": 2})],
)
def test_real_audio_aead_rejects_cross_identity_sequence_sample_pin_and_expiry_swaps(
    location, updates
):
    keyring = CryptoKeyring({1: KEY, 2: bytes(reversed(range(32)))}, active_version=1)
    original = VoiceOperationV2.model_validate(operation("audio.chunk", chunk_payload()))
    encrypted = keyring.encrypt(b"\x01\x00\x02\x00" * 8000,
                               aad=canonical_audio_chunk_aad(original))
    changed = operation("audio.chunk", chunk_payload())
    (changed if location == "operation" else changed["payload"]).update(updates)
    swapped = VoiceOperationV2.model_validate(changed)
    with pytest.raises(CryptoDecryptionError, match="^crypto_decryption_failed$"):
        keyring.decrypt(encrypted, aad=canonical_audio_chunk_aad(swapped))


@pytest.mark.parametrize("tamper", ["nonce", "ciphertext", "domain"])
def test_real_audio_aead_round_trip_authenticates_before_exposing_pcm(tamper):
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    parsed = VoiceOperationV2.model_validate(operation("audio.chunk", chunk_payload()))
    aad = canonical_audio_chunk_aad(parsed)
    pcm = b"\x01\x00\x02\x00" * 8000
    encrypted = keyring.encrypt(pcm, aad=aad)
    assert len(encrypted.nonce) == 12 and len(encrypted.ciphertext) == 32016
    assert keyring.decrypt(encrypted, aad=aad) == pcm
    nonce, ciphertext = encrypted.nonce, encrypted.ciphertext
    if tamper == "nonce":
        nonce = bytes([nonce[0] ^ 1]) + nonce[1:]
    elif tamper == "ciphertext":
        ciphertext = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
    else:
        aad = aad.replace(AAD_DOMAIN, b"sparra.audio.chunk.v2\x00", 1)
    with pytest.raises(CryptoDecryptionError) as failure:
        keyring.decrypt(EncryptedValue(encrypted.key_version, nonce, ciphertext), aad=aad)
    assert str(failure.value) == "crypto_decryption_failed"
    assert pcm.hex() not in str(failure.value)
    assert base64.b64encode(ciphertext).decode("ascii") not in str(failure.value)


def test_actual_maximum_unicode_audio_command_keeps_both_metadata_and_envelope_budgets():
    maximum_key_version = 9_007_199_254_740_991
    keyring = CryptoKeyring({maximum_key_version: KEY}, active_version=maximum_key_version)
    wire = operation(
        "audio.chunk",
        chunk_payload(sequence=599, configuration_revision=2_147_483_647,
                      key_version=maximum_key_version, retention_until="9999-12-31T23:59:59.999Z"),
        deployment_id="😀" * 256,
        occurred_at="9999-12-01T23:59:59.999Z",
    )
    parsed = VoiceOperationV2.model_validate(wire)
    pcm = b"\x01\x00\x02\x00" * 8000
    inner = keyring.encrypt(pcm, aad=canonical_audio_chunk_aad(parsed))
    wire["payload"].update(
        nonce_b64=base64.b64encode(inner.nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(inner.ciphertext).decode("ascii"),
    )
    parsed = VoiceOperationV2.model_validate(wire)
    prepared = encrypt_audio_operation(parsed, keyring)
    decoded_plaintext = keyring.decrypt(prepared.encrypted, aad=prepared.aad)
    assert decoded_plaintext == canonical_operation_bytes(parsed) == prepared.plaintext
    assert decode_operation_v2(decoded_plaintext) == parsed
    assert len(parsed.deployment_id.encode("utf-8")) == 1024
    assert len(wire["payload"]["ciphertext_b64"].encode("ascii")) == 42688
    assert len(canonical_audio_chunk_aad(parsed)) <= 2048
    assert len(prepared.plaintext) - 42688 <= 2048
    assert len(prepared.aad) <= 2048
    assert prepared.envelope_size == len(prepared.aad) + len(prepared.encrypted.nonce) + len(
        prepared.encrypted.ciphertext
    )
    assert prepared.envelope_size <= 46812 < 65536
    assert keyring.decrypt(inner, aad=canonical_audio_chunk_aad(parsed)) == pcm
    rendered = repr(parsed) + repr(prepared)
    assert wire["payload"]["nonce_b64"] not in rendered
    assert wire["payload"]["ciphertext_b64"] not in rendered
    assert pcm.hex() not in rendered


@pytest.mark.parametrize("deployment", ["", "x" * 257, "bad\nidentity", "\ud800"])
def test_v2_deployments_keep_the_existing_native_identity_validation(deployment):
    with pytest.raises(ValidationError):
        VoiceOperationV2.model_validate(operation("audio.chunk", chunk_payload(),
                                                 deployment_id=deployment))


def test_legacy_provider_purge_bytes_aad_and_encrypted_consumer_remain_unchanged():
    wire = operation("recording.upsert", deepcopy(LEGACY_PAYLOADS["recording.upsert"]),
                     schema_version=1)
    golden = (
        b'{"call_id":"22222222-2222-4222-8222-222222222222","deployment_id":"voice-agent-a",'
        b'"kind":"recording.upsert","occurred_at":"2026-10-01T10:00:01.123000Z",'
        b'"operation_id":"33333333-3333-4333-8333-333333333333","payload":{"channels":"dual",'
        b'"ended_at":"2026-10-01T10:00:01.123000Z","format":"wav",'
        b'"recording_id":"55555555-5555-4555-8555-555555555555",'
        b'"retention_until":"2026-10-31T10:00:00.123000Z",'
        b'"started_at":"2026-10-01T10:00:00.123000Z",'
        b'"status":"purged","telnyx_recording_id":"fixture-provider-recording"},'
        b'"schema_version":1}'
    )
    golden_aad = (
        b"projetv0-voice/outbox/aes-256-gcm/v1\x00"
        b'{"call_id":"22222222-2222-4222-8222-222222222222","deployment_id":"voice-agent-a",'
        b'"kind":"recording.upsert","operation_id":"33333333-3333-4333-8333-333333333333",'
        b'"schema_version":1}'
    )
    legacy = VoiceOperationV1.model_validate(wire)
    assert canonical_operation_bytes(legacy) == golden
    assert operation_aad(legacy) == golden_aad
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    prepared = encrypt_operation(legacy, keyring)
    assert type(prepared.operation) is VoiceOperationV1
    assert prepared.plaintext == golden and prepared.aad == golden_aad
    assert keyring.decrypt(prepared.encrypted, aad=golden_aad) == golden
    assert type(decode_operation(golden)) is VoiceOperationV1
    assert decode_operation(golden).model_dump(mode="json") == wire


def test_v2_decode_failure_retains_no_raw_media_validation_cause_or_context():
    sentinel = "fixture-only-malformed-ciphertext"
    wire = operation("audio.chunk", chunk_payload(ciphertext_b64=sentinel))
    with pytest.raises(CommandSerializationError) as failure:
        decode_operation_v2(json.dumps(wire).encode("utf-8"))
    assert str(failure.value) == "command_deserialization_failed"
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert sentinel not in repr(failure.value)
