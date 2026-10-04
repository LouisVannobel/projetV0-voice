"""Typed and versioned durable-operation contracts."""

from __future__ import annotations

import base64
import binascii
import re
import string
from datetime import UTC, datetime
from typing import Annotated, Literal, Self, TypeGuard
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictStr,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
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
        and value not in {".", ".."}
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


_INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")
_E164 = re.compile(r"^\+[1-9][0-9]{1,14}$")


def _business_datetime_input(value: object) -> object:
    if not isinstance(value, datetime) and (
        not isinstance(value, str) or _INSTANT.fullmatch(value) is None
    ):
        raise ValueError("datetime must be a finite ISO instant")
    return _validate_datetime_input(value)


def _canonical_milliseconds(value: datetime) -> datetime:
    normalized = _business_utc(value)
    return normalized.replace(microsecond=normalized.microsecond // 1000 * 1000)


def _business_utc(value: datetime) -> datetime:
    try:
        return _utc_datetime(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("datetime must normalize to a finite UTC instant") from error


def _optional_business_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _business_utc(value)


def _canonical_uuid(value: object) -> object:
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str):
        raise ValueError("UUID must be canonical")
    parsed = UUID(value)
    if str(parsed) != value:
        raise ValueError("UUID must be canonical")
    return parsed


CanonicalUUID = Annotated[UUID, BeforeValidator(_canonical_uuid)]
BusinessInstant = Annotated[
    datetime,
    BeforeValidator(_business_datetime_input),
]
LossCount = Annotated[int, Field(ge=0, le=2_147_483_647), BeforeValidator(_require_exact_int)]


def _text(value: str, maximum: int) -> str:
    if any(
        0xD800 <= ord(character) <= 0xDFFF
        or ord(character) < 32
        and character not in "\n\r\t"
        or 0x7F <= ord(character) <= 0x9F
        for character in value
    ):
        raise ValueError("text contains invalid Unicode or controls")
    if len(value.encode("utf-16-le")) // 2 > maximum:
        raise ValueError("text exceeds UTF16 bound")
    return value


def _provider_id(value: str, maximum: int) -> str:
    if (
        not value
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 32 or 0x7F <= ord(character) <= 0x9F for character in value)
    ):
        raise ValueError("provider identity exceeds bound or contains controls")
    return value


def validate_deployment_id(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError("deployment_id is outside the supported range")
    _text(value, 512)
    if any(ord(character) < 32 for character in value):
        raise ValueError("deployment_id contains controls")
    return value


def _e164(value: str) -> str:
    if _E164.fullmatch(value) is None:
        raise ValueError("number must be E164")
    return value


class RoutingV1(_StrictFrozenModel):
    schema_version: SchemaVersionV1
    direction: Literal["incoming"]
    connection_id: StrictStr
    to_e164: StrictStr
    from_e164: StrictStr | None
    telnyx_call_control_id: StrictStr
    telnyx_call_leg_id: StrictStr | None
    telnyx_call_session_id: StrictStr | None
    admitted_at: BusinessInstant

    _admission = field_validator("admitted_at")(_canonical_milliseconds)

    @field_serializer("admitted_at")
    def serialize_admission(self, value: datetime) -> str:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @field_validator(
        "connection_id", "telnyx_call_control_id", "telnyx_call_leg_id", "telnyx_call_session_id"
    )
    @classmethod
    def provider_identity(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _provider_id(value, 256 if info.field_name == "connection_id" else 1024)

    @field_validator("to_e164", "from_e164")
    @classmethod
    def number(cls, value: str | None) -> str | None:
        return None if value is None else _e164(value)

    @model_validator(mode="after")
    def routing_size(self) -> Self:
        if len(self.model_dump_json().encode("utf-8")) > 8192:
            raise ValueError("routing exceeds UTF8 bound")
        return self


class BusinessKnowledgeV1(_StrictFrozenModel):
    business_name: StrictStr
    sector: Literal["garage", "controle-technique"]
    opening_hours: StrictStr
    services: StrictStr
    prices: StrictStr
    faq: StrictStr
    instructions: StrictStr

    @field_validator("business_name", "opening_hours", "services", "prices", "faq", "instructions")
    @classmethod
    def knowledge_text(cls, value: str, info: ValidationInfo) -> str:
        name = info.field_name or ""
        normalized = value.replace("\r\n", "\n")
        bounds = {
            "business_name": 80,
            "opening_hours": 1000,
            "services": 2000,
            "prices": 1500,
            "faq": 3000,
            "instructions": 2000,
        }
        if name == "business_name":
            normalized = normalized.strip()
            if not normalized or any(ord(char) < 32 for char in normalized):
                raise ValueError("business name must be nonempty and single line")
        return _text(normalized, bounds[name])

    @model_validator(mode="after")
    def knowledge_size(self) -> Self:
        fields = (self.opening_hours, self.services, self.prices, self.faq, self.instructions)
        if sum(len(value.encode("utf-16-le")) // 2 for value in fields) > 9500:
            raise ValueError("knowledge exceeds total UTF16 bound")
        return self


class BeginCallSnapshotV1(_StrictFrozenModel):
    schema_version: SchemaVersionV1
    call_id: CanonicalUUID
    configuration_revision: Annotated[PositiveInt, Field(le=2_147_483_647)]
    knowledge: BusinessKnowledgeV1
    transfer_destination: StrictStr | None
    retention_until: BusinessInstant
    recording_enabled: StrictBool = False

    _retention = field_validator("retention_until")(_canonical_milliseconds)

    @field_serializer("retention_until")
    def serialize_retention(self, value: datetime) -> str:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @field_validator("transfer_destination")
    @classmethod
    def destination(cls, value: str | None) -> str | None:
        return None if value is None else _e164(value)


class DisclosureEvidenceV1(_StrictFrozenModel):
    schema_version: SchemaVersionV1
    started_at: BusinessInstant | None
    completed_at: BusinessInstant | None
    failed_at: BusinessInstant | None
    input_gate_opened_at: BusinessInstant | None

    _dates = field_validator("started_at", "completed_at", "failed_at", "input_gate_opened_at")(
        _optional_business_utc
    )

    @model_validator(mode="after")
    def chronology(self) -> Self:
        if self.input_gate_opened_at is not None and self.completed_at is None:
            raise ValueError("input gate requires disclosure completion evidence")
        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("disclosure completion precedes start")
        if (
            self.input_gate_opened_at is not None
            and self.completed_at is not None
            and self.input_gate_opened_at < self.completed_at
        ):
            raise ValueError("input gate precedes disclosure completion")
        return self


def _exact_false(value: object) -> object:
    if type(value) is not bool or value:
        raise ValueError("pilot confirmation must be false")
    return value


PilotFalse = Annotated[Literal[False], BeforeValidator(_exact_false)]


class MessageContactV1(_StrictFrozenModel):
    name: StrictStr | None
    callback_e164: StrictStr | None
    preference: StrictStr | None
    callback_source: Literal["caller", "provider", "missing"]
    callback_confirmed: PilotFalse

    @field_validator("name", "preference")
    @classmethod
    def contact_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _text(value, 120 if info.field_name == "name" else 300)

    @field_validator("callback_e164")
    @classmethod
    def callback(cls, value: str | None) -> str | None:
        return None if value is None else _e164(value)

    @model_validator(mode="after")
    def callback_provenance(self) -> Self:
        if (self.callback_e164 is None) != (self.callback_source == "missing"):
            raise ValueError("callback number and observation source disagree")
        return self


class MessageEvidenceV1(_StrictFrozenModel):
    turn_id: CanonicalUUID
    role: Literal["user", "assistant"]


class MessageResultV1(_StrictFrozenModel):
    schema_version: SchemaVersionV1
    quality: Literal["partial"]
    category: Literal["callback", "information", "appointment_to_confirm", "declared_urgent"]
    summary: StrictStr
    contact: MessageContactV1
    next_action: StrictStr
    evidence: tuple[MessageEvidenceV1, ...] = Field(max_length=64)
    request_confirmed: PilotFalse

    @field_validator("summary", "next_action")
    @classmethod
    def result_text(cls, value: str, info: ValidationInfo) -> str:
        return _text(value, 3000 if info.field_name == "summary" else 500)

    @model_validator(mode="after")
    def result_size_and_evidence(self) -> Self:
        if len({evidence.turn_id for evidence in self.evidence}) != len(self.evidence):
            raise ValueError("result evidence references must be unique")
        if len(self.model_dump_json().encode("utf-8")) > 8192:
            raise ValueError("result exceeds UTF8 bound")
        return self


class MessageResultEnvelopeV1(_StrictFrozenModel):
    schema_version: SchemaVersionV1
    crypto_version: SchemaVersionV1
    key_version: Annotated[PositiveInt, Field(le=9_007_199_254_740_991)]
    nonce_b64: StrictStr
    ciphertext_b64: StrictStr

    @field_validator("nonce_b64", "ciphertext_b64")
    @classmethod
    def encrypted_value(cls, value: str, info: ValidationInfo) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("result ciphertext must be canonical base64") from error
        if base64.b64encode(decoded).decode("ascii") != value:
            raise ValueError("result ciphertext must be canonical base64")
        if info.field_name == "nonce_b64":
            if len(decoded) != 12:
                raise ValueError("result nonce must be twelve bytes")
        elif not 16 <= len(decoded) <= 8208:
            raise ValueError("result ciphertext outside structural bound")
        return value


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
    message_result: MessageResultEnvelopeV1 | None = None
    disclosure_evidence: DisclosureEvidenceV1 | None = None
    transcript_loss_count: LossCount | None = None

    @field_validator(
        "message_result", "disclosure_evidence", "transcript_loss_count", mode="before"
    )
    @classmethod
    def reject_explicit_null_extension(cls, value: object) -> object:
        if value is None:
            raise ValueError("present extension must not be null")
        return value

    @model_serializer(mode="wrap")
    def serialize_extensions(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized: dict[str, object] = handler(self)
        for name in ("message_result", "disclosure_evidence", "transcript_loss_count"):
            if name not in self.model_fields_set:
                serialized.pop(name, None)
        return serialized

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


class RecordingArchiveReceiptV1(_StrictFrozenModel):
    """Bounded metadata for durable ciphertext; contains no access scope or bytes."""

    recording_id: CanonicalUUID
    ciphertext_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    encrypted_bytes: Annotated[
        int, Field(ge=17, le=33_554_448), BeforeValidator(_require_exact_int)
    ]
    key_version: Annotated[PositiveInt, Field(le=9_007_199_254_740_991)]
    retention_until: BusinessInstant

    @field_validator("retention_until", mode="before")
    @classmethod
    def strict_deadline(cls, value: object) -> object:
        if isinstance(value, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value
        ) is None:
            raise ValueError("archive deadline must be canonical UTC milliseconds")
        if isinstance(value, datetime) and (
            value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None)
            or value.microsecond % 1000
        ):
            raise ValueError("archive deadline must be canonical UTC milliseconds")
        return value

    _retention = field_validator("retention_until")(_business_utc)

    @field_serializer("retention_until")
    def serialize_deadline(self, value: datetime) -> str:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RecordingUpsertPayloadV1(_StrictFrozenModel):
    recording_id: UUID
    status: Literal["off", "pending", "active", "saved", "failed", "purged"]
    telnyx_recording_id: str | None = None
    channels: Literal["dual"] | None
    format: Literal["wav"] | None
    started_at: datetime | None
    ended_at: datetime | None
    retention_until: datetime | None
    archive_receipt: RecordingArchiveReceiptV1 | None = None

    @field_validator("archive_receipt", mode="before")
    @classmethod
    def reject_null_receipt(cls, value: object) -> object:
        if value is None:
            raise ValueError("present archive receipt must not be null")
        return value

    @model_serializer(mode="wrap")
    def serialize_archive_extension(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        serialized: dict[str, object] = handler(self)
        if "archive_receipt" not in self.model_fields_set:
            serialized.pop("archive_receipt", None)
        return serialized

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
        if self.archive_receipt is not None and (
            self.status not in {"saved", "purged"}
            or self.telnyx_recording_id is None
            or self.archive_receipt.recording_id != self.recording_id
            or self.archive_receipt.retention_until != self.retention_until
        ):
            raise ValueError("archive receipt must match saved recording identity and deadline")
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
