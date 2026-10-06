"""Strict local-audio V2 wire models and the domain-separated chunk AAD."""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from projetv0_voice.models import (
    BusinessInstant,
    CallUpsertPayloadV1,
    CanonicalUUID,
    PositiveInt,
    RecordingUpsertPayloadV1,
    TurnUpsertPayloadV1,
    _business_utc,
    _require_exact_int,
    _StrictFrozenModel,
    validate_deployment_id,
)

MAX_AUDIO_METADATA_BYTES = 2048
MAX_AUDIO_AAD_BYTES = 2048
MAX_AUDIO_ENVELOPE_BYTES = 46_812
AUDIO_CHUNK_AAD_DOMAIN = b"sparra.audio.chunk.v1\x00"

SchemaVersionV2 = Annotated[Literal[2], BeforeValidator(_require_exact_int)]
AudioSequence = Annotated[int, Field(ge=0, le=599), BeforeValidator(_require_exact_int)]
AudioSampleCount = Annotated[int, Field(ge=1, le=8000), BeforeValidator(_require_exact_int)]
AudioTotalSamples = Annotated[
    int, Field(ge=0, le=4_800_000), BeforeValidator(_require_exact_int)
]
_UTC_MILLISECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


class _AudioModel(_StrictFrozenModel):
    model_config = ConfigDict(hide_input_in_errors=True)


class _AudioIdentityV2(_AudioModel):
    schema_version: SchemaVersionV2
    workspace_id: CanonicalUUID
    recording_id: CanonicalUUID
    configuration_revision: Annotated[PositiveInt, Field(le=2_147_483_647)]
    retention_until: BusinessInstant

    @field_validator("retention_until", mode="before")
    @classmethod
    def canonical_original_expiry(cls, value: object) -> object:
        if isinstance(value, str) and _UTC_MILLISECONDS.fullmatch(value) is None:
            raise ValueError("audio_expiry_not_canonical")
        if isinstance(value, datetime) and (
            value.tzinfo is None
            or value.utcoffset() != UTC.utcoffset(None)
            or value.microsecond % 1000
        ):
            raise ValueError("audio_expiry_not_canonical")
        return value

    _normalize_retention = field_validator("retention_until")(_business_utc)

    @field_serializer("retention_until")
    def serialize_retention(self, value: datetime) -> str:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AudioChunkPayloadV2(_AudioIdentityV2):
    sequence: AudioSequence
    sample_count: AudioSampleCount
    sample_rate: Annotated[Literal[8000], BeforeValidator(_require_exact_int)]
    channels: Annotated[Literal[2], BeforeValidator(_require_exact_int)]
    sample_format: Literal["s16le"]
    crypto_version: Annotated[Literal[1], BeforeValidator(_require_exact_int)]
    key_version: Annotated[PositiveInt, Field(le=9_007_199_254_740_991)]
    nonce_b64: StrictStr = Field(min_length=16, max_length=16, repr=False)
    ciphertext_b64: StrictStr = Field(min_length=28, max_length=42688, repr=False)

    @model_validator(mode="after")
    def canonical_media_and_sample_length(self) -> Self:
        try:
            nonce = base64.b64decode(self.nonce_b64, validate=True)
            ciphertext = base64.b64decode(self.ciphertext_b64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("audio_media_not_canonical") from error
        if (
            base64.b64encode(nonce).decode("ascii") != self.nonce_b64
            or base64.b64encode(ciphertext).decode("ascii") != self.ciphertext_b64
            or len(nonce) != 12
            or len(ciphertext) != self.sample_count * 4 + 16
        ):
            raise ValueError("audio_media_invalid")
        return self


class AudioFinishPayloadV2(_AudioIdentityV2):
    last_sequence: AudioSequence | None
    total_samples: AudioTotalSamples
    reason: Literal["complete", "transfer", "interrupted", "limit", "failure"]

    @model_validator(mode="after")
    def empty_capture_has_no_sequence(self) -> Self:
        if (self.last_sequence is None) != (self.total_samples == 0):
            raise ValueError("audio_finish_empty_mismatch")
        if self.last_sequence is not None:
            chunk_count = self.last_sequence + 1
            if not chunk_count <= self.total_samples <= chunk_count * 8000:
                raise ValueError("audio_finish_samples_mismatch")
        return self


class AudioRevokePayloadV2(_AudioIdentityV2):
    reason: Literal["caller_declined"]


AudioPayloadV2 = AudioChunkPayloadV2 | AudioFinishPayloadV2 | AudioRevokePayloadV2
OperationPayloadV2 = (
    CallUpsertPayloadV1 | TurnUpsertPayloadV1 | RecordingUpsertPayloadV1 | AudioPayloadV2
)


class VoiceOperationV2(_AudioModel):
    schema_version: SchemaVersionV2
    operation_id: CanonicalUUID
    deployment_id: StrictStr
    call_id: CanonicalUUID
    occurred_at: BusinessInstant
    kind: Literal[
        "call.upsert", "turn.upsert", "recording.upsert",
        "audio.chunk", "audio.finish", "audio.revoke",
    ]
    payload: OperationPayloadV2 = Field(repr=False)

    _deployment = field_validator("deployment_id")(validate_deployment_id)
    _normalize_occurred_at = field_validator("occurred_at")(_business_utc)

    @model_validator(mode="after")
    def matching_kind_and_legacy_timeline(self) -> Self:
        expected: dict[str, type[_StrictFrozenModel]] = {
            "call.upsert": CallUpsertPayloadV1,
            "turn.upsert": TurnUpsertPayloadV1,
            "recording.upsert": RecordingUpsertPayloadV1,
            "audio.chunk": AudioChunkPayloadV2,
            "audio.finish": AudioFinishPayloadV2,
            "audio.revoke": AudioRevokePayloadV2,
        }
        if not isinstance(self.payload, expected[self.kind]):
            raise ValueError("payload_does_not_match_kind")
        if isinstance(self.payload, (CallUpsertPayloadV1, TurnUpsertPayloadV1,
                                     RecordingUpsertPayloadV1)):
            boundary = self.payload.ended_at or self.payload.started_at
            if boundary is not None and self.occurred_at < boundary:
                raise ValueError("occurred_at_precedes_payload_timeline")
        return self


def canonical_audio_chunk_aad(operation: VoiceOperationV2) -> bytes:
    """Bind a chunk to its exact original identities, pin, format and sample count."""
    if not isinstance(operation, VoiceOperationV2) or operation.kind != "audio.chunk":
        raise ValueError("invalid_audio_chunk_operation")
    payload = operation.payload
    if not isinstance(payload, AudioChunkPayloadV2):
        raise ValueError("invalid_audio_chunk_operation")
    metadata = {
        "schema_version": operation.schema_version,
        "workspace_id": str(payload.workspace_id),
        "deployment_id": operation.deployment_id,
        "call_id": str(operation.call_id),
        "recording_id": str(payload.recording_id),
        "sequence": payload.sequence,
        "sample_count": payload.sample_count,
        "sample_rate": payload.sample_rate,
        "channels": payload.channels,
        "sample_format": payload.sample_format,
        "configuration_revision": payload.configuration_revision,
        "retention_until": payload.retention_until.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "crypto_version": payload.crypto_version,
        "key_version": payload.key_version,
    }
    aad = AUDIO_CHUNK_AAD_DOMAIN + json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(aad) > MAX_AUDIO_AAD_BYTES:
        raise ValueError("audio_aad_too_large")
    return aad
