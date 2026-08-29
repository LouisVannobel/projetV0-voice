"""Typed and versioned durable-operation contracts."""

from __future__ import annotations

import base64
import binascii
import string
from datetime import UTC, datetime
from typing import Annotated, Literal, Self, TypeGuard
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)


def _require_exact_int(value: object) -> object:
    if type(value) is not int:
        raise ValueError("value must be an exact integer")
    return value


SchemaVersionV1 = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
PositiveInt = Annotated[int, Field(gt=0), BeforeValidator(_require_exact_int)]
MAX_PROVIDER_RECORDING_ID_CHARS = 256
_PROVIDER_RECORDING_ID_CHARS = frozenset(
    string.ascii_letters + string.digits + "-._~"
)


def is_valid_provider_recording_id(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_PROVIDER_RECORDING_ID_CHARS
        and all(character in _PROVIDER_RECORDING_ID_CHARS for character in value)
    )


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _validate_datetime_input(value: object) -> object:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("datetime input must be an ISO-8601 string") from error
        return value
    if value is None:
        return value
    raise ValueError("datetime input must be a datetime object or ISO-8601 string")


def _optional_utc_datetime(value: datetime | None) -> datetime | None:
    return None if value is None else _utc_datetime(value)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CallUpsertPayloadV1(_StrictFrozenModel):
    telnyx_call_control_id: str = Field(min_length=1)
    telnyx_call_leg_id: str | None
    telnyx_call_session_id: str | None
    status: Literal["pending", "active", "closing", "closed", "failed"]
    disclosure_state: Literal["pending", "completed", "failed"]
    started_at: datetime | None
    ended_at: datetime | None
    end_reason: str | None
    retention_until: datetime

    _validate_datetime_inputs = field_validator(
        "started_at", "ended_at", "retention_until", mode="before"
    )(_validate_datetime_input)
    _normalize_datetimes = field_validator(
        "started_at", "ended_at", "retention_until", mode="after"
    )(_optional_utc_datetime)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.status == "pending":
            if (
                self.started_at is not None
                or self.ended_at is not None
                or self.end_reason is not None
            ):
                raise ValueError("pending call cannot contain start/end fields")
        elif self.status in {"active", "closing"}:
            if self.started_at is None:
                raise ValueError("active call requires started_at")
            if self.ended_at is not None:
                raise ValueError("active or closing call cannot contain ended_at")
            if self.status == "active" and self.end_reason is not None:
                raise ValueError("active call cannot contain end_reason")
        elif self.status == "closed" and (
            self.started_at is None or self.ended_at is None or not self.end_reason
        ):
            raise ValueError("closed call requires start, end and reason")
        elif self.status == "failed" and (self.ended_at is None or not self.end_reason):
            raise ValueError("failed call requires end and reason")

        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("ended_at must not precede started_at")
        boundary = self.ended_at or self.started_at
        if boundary is not None and self.retention_until <= boundary:
            raise ValueError("retention_until must follow the call timeline")
        return self


class TurnUpsertPayloadV1(_StrictFrozenModel):
    turn_id: UUID
    turn_no: PositiveInt
    role: Literal["user", "assistant"]
    source: Literal["stt_final", "pipecat_assistant"]
    crypto_version: SchemaVersionV1
    key_version: PositiveInt
    nonce_b64: str
    ciphertext_b64: str
    started_at: datetime
    ended_at: datetime
    interrupted: StrictBool

    _validate_datetime_inputs = field_validator("started_at", "ended_at", mode="before")(
        _validate_datetime_input
    )
    _normalize_datetimes = field_validator("started_at", "ended_at", mode="after")(
        _utc_datetime
    )

    @field_validator("nonce_b64", "ciphertext_b64")
    @classmethod
    def validate_encrypted_value(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "encrypted value")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"{field_name} must be canonical base64") from error
        if base64.b64encode(decoded).decode("ascii") != value:
            raise ValueError(f"{field_name} must be canonical base64")
        maximum_size = 128 if field_name == "nonce_b64" else 65_536
        if not decoded or len(decoded) > maximum_size:
            raise ValueError(f"{field_name} decoded size is outside the structural bound")
        return value

    @model_validator(mode="after")
    def validate_turn(self) -> Self:
        expected_source = "stt_final" if self.role == "user" else "pipecat_assistant"
        if self.source != expected_source:
            raise ValueError("turn role and source do not match")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        return self


class RecordingUpsertPayloadV1(_StrictFrozenModel):
    recording_id: UUID
    status: Literal["off", "pending", "active", "saved", "failed", "purged"]
    telnyx_recording_id: str | None = None
    channels: Literal["dual"] | None
    format: Literal["wav"] | None
    started_at: datetime | None
    ended_at: datetime | None
    retention_until: datetime | None

    _validate_datetime_inputs = field_validator(
        "started_at", "ended_at", "retention_until", mode="before"
    )(_validate_datetime_input)
    _normalize_datetimes = field_validator(
        "started_at", "ended_at", "retention_until", mode="after"
    )(_optional_utc_datetime)

    @field_validator("telnyx_recording_id")
    @classmethod
    def validate_provider_recording_id(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_provider_recording_id(value):
            raise ValueError("provider recording ID is invalid")
        return value

    @model_validator(mode="after")
    def validate_recording(self) -> Self:
        metadata = (
            self.telnyx_recording_id,
            self.channels,
            self.format,
            self.started_at,
            self.ended_at,
            self.retention_until,
        )
        if self.status == "off" and any(item is not None for item in metadata):
            raise ValueError("off recording cannot contain recording metadata")
        if self.status != "off" and (self.channels != "dual" or self.format != "wav"):
            raise ValueError("enabled recording requires dual-channel WAV metadata")
        timeline_present = self.started_at is not None or self.ended_at is not None
        if timeline_present and (self.started_at is None or self.ended_at is None):
            raise ValueError("recording timeline must be wholly present or absent")
        if self.status in {"pending", "active"} and any(
            item is not None
            for item in (
                self.telnyx_recording_id,
                self.started_at,
                self.ended_at,
                self.retention_until,
            )
        ):
            raise ValueError("pending or reserved active recording has no provider truth")
        if self.status == "failed" and self.retention_until is not None:
            raise ValueError("failed recording has no retention deadline")
        if self.status in {"saved", "purged"} and (
            self.started_at is None
            or self.ended_at is None
            or self.retention_until is None
        ):
            raise ValueError("saved or purged recording requires timeline and retention")
        if self.status == "purged" and self.telnyx_recording_id is None:
            raise ValueError("purged recording requires provider identity")
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("ended_at must not precede started_at")
        if self.retention_until is not None:
            boundary = self.ended_at or self.started_at
            if boundary is None or self.retention_until <= boundary:
                raise ValueError("recording retention must follow its timeline")
        return self


OperationPayloadV1 = CallUpsertPayloadV1 | TurnUpsertPayloadV1 | RecordingUpsertPayloadV1


class VoiceOperationV1(_StrictFrozenModel):
    schema_version: SchemaVersionV1
    operation_id: UUID
    deployment_id: str = Field(min_length=1)
    call_id: UUID
    occurred_at: datetime
    kind: Literal["call.upsert", "turn.upsert", "recording.upsert"]
    payload: OperationPayloadV1

    _validate_occurred_at_input = field_validator("occurred_at", mode="before")(
        _validate_datetime_input
    )
    _normalize_occurred_at = field_validator("occurred_at", mode="after")(_utc_datetime)

    @model_validator(mode="after")
    def validate_kind_and_timeline(self) -> Self:
        expected_type: dict[str, type[_StrictFrozenModel]] = {
            "call.upsert": CallUpsertPayloadV1,
            "turn.upsert": TurnUpsertPayloadV1,
            "recording.upsert": RecordingUpsertPayloadV1,
        }
        if not isinstance(self.payload, expected_type[self.kind]):
            raise ValueError("payload does not match kind")

        payload_started = self.payload.started_at
        payload_ended = self.payload.ended_at
        timeline_boundary = payload_ended or payload_started
        if timeline_boundary is not None and self.occurred_at < timeline_boundary:
            raise ValueError("occurred_at must not precede the payload timeline")
        return self
