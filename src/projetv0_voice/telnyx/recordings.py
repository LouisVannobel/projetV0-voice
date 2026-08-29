"""Optional Telnyx recording identity, correlation, and lifecycle policy."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, NoReturn, Protocol, cast
from uuid import UUID, uuid4, uuid5

from pydantic import SecretStr

from projetv0_voice.models import (
    RecordingUpsertPayloadV1,
    VoiceOperationV1,
    is_valid_provider_recording_id,
)
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    PersistenceCommand,
    PersistenceError,
    canonical_operation_bytes,
)
from projetv0_voice.persistence.postgres_sink import (
    POOL_ACQUIRE_TIMEOUT_SECONDS,
    SQL_TRANSACTION_TIMEOUT_SECONDS,
    OperationConflictError,
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkPermanentError,
    OperationSinkStaleLeaseError,
    OperationSinkTransientError,
    PostgresOperationSink,
    PurgeOutcome,
    RecordingPurgeLease,
)
from projetv0_voice.session import (
    CallIdentity,
    ControlWriter,
    RecordingBoundary,
    RecordingStartResult,
    RecordingStartState,
)
from projetv0_voice.telnyx.call_control import (
    ATTEMPT_DEADLINE_SECONDS,
    MAX_CALL_CONTROL_ID_CHARS,
    CallControlResult,
    RecordingCatalogInvalidError,
    RecordingCatalogTransientError,
    RecordingStartV1,
    RecordingStopV1,
)
from projetv0_voice.telnyx.webhooks import (
    MAX_PROVIDER_ID_CHARS,
    VerifiedWebhook,
    WebhookDisposition,
    WebhookDurableEffect,
)

RECORDING_NAMESPACE_V1 = UUID("8f0f6b4b-6194-4a43-87a0-649810b2760f")
RECORDING_WEBHOOK_OPERATION_NAMESPACE_V1 = UUID(
    "eeec9d6b-b4f8-4e40-b26e-62c7fe04e218"
)
MAX_CLIENT_STATE_B64_CHARS = 4_096
MAX_CLIENT_STATE_JSON_BYTES = 3_072
MAX_DEPLOYMENT_ID_UTF8_BYTES = 256
MAX_RETENTION_DAYS = 3_650
RECONCILE_TIMESTAMP_SLOP_SECONDS = 5.0
RECORDING_LIST_DEADLINE_SECONDS = 1.0
RECORDING_RETRIEVE_DEADLINE_SECONDS = 1.0
RECORDING_DELETE_DEADLINE_SECONDS = 1.0
REQUIRED_HANGUP_OVERHEAD_SECONDS = 0.1
REQUIRED_HANGUP_RESERVE_SECONDS = (
    2 * ATTEMPT_DEADLINE_SECONDS + REQUIRED_HANGUP_OVERHEAD_SECONDS
)
PURGE_ACK_OVERHEAD_SECONDS = 0.5
PURGE_ACK_RESERVE_SECONDS = (
    POOL_ACQUIRE_TIMEOUT_SECONDS
    + SQL_TRANSACTION_TIMEOUT_SECONDS
    + PURGE_ACK_OVERHEAD_SECONDS
)
PURGE_CLOCK_SKEW_RESERVE_SECONDS = 0.25

_CAPSULE_KEYS = frozenset(
    {
        "v",
        "deployment_id",
        "call_id",
        "recording_id",
        "call_control_id",
        "call_leg_id",
        "call_session_id",
        "retention_days",
        "required",
    }
)
RecordingAction = Literal["recording-start", "recording-stop", "hangup"]


class RecordingError(RuntimeError):
    """A recording lifecycle error carrying only a constant project code."""


class RecordingCapsuleError(RecordingError):
    """The redacted recording correlation capsule is invalid."""


class RecordingLifecycleError(RecordingError):
    """A bounded recording lifecycle operation failed closed."""


class RecordingWebhookError(RecordingError):
    """A signed recording callback violated the frozen V1 contract."""


class RecordingPurgeError(RecordingError):
    """A purge batch failed closed with only a constant project code."""


@dataclass(frozen=True, slots=True, repr=False)
class ProviderDeleteResultV1:
    outcome: Literal["deleted", "not_found", "retry", "failed"]
    recording_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            self.outcome not in {"deleted", "not_found", "retry", "failed"}
            or self.outcome == "deleted"
            and (
                not isinstance(self.recording_id, str)
                or not is_valid_provider_recording_id(self.recording_id)
            )
            or self.outcome != "deleted"
            and self.recording_id is not None
        ):
            raise ValueError("provider_delete_result_invalid")

    def __repr__(self) -> str:
        return f"ProviderDeleteResultV1(outcome={self.outcome!r})"


@dataclass(frozen=True, slots=True)
class PurgeBatchResult:
    leased: int
    deleted: int
    not_found: int
    retry: int
    failed: int
    stale: int
    expired: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.leased,
                self.deleted,
                self.not_found,
                self.retry,
                self.failed,
                self.stale,
                self.expired,
            )
        ):
            raise ValueError("purge_batch_result_invalid")


@dataclass(frozen=True, slots=True, repr=False)
class ProviderRecordingV1:
    recording_id: str | None = field(repr=False)
    call_control_id: str | None = field(repr=False)
    call_leg_id: str | None = field(repr=False)
    call_session_id: str | None = field(repr=False)
    channels: str | None
    status: str | None
    source: str | None
    initiated_by: str | None
    recording_started_at: datetime | None
    recording_ended_at: datetime | None

    def __post_init__(self) -> None:
        text_values = (
            self.recording_id,
            self.call_control_id,
            self.call_leg_id,
            self.call_session_id,
            self.channels,
            self.status,
            self.source,
            self.initiated_by,
        )
        if (
            any(value is not None and not isinstance(value, str) for value in text_values)
            or self.recording_id is not None
            and not is_valid_provider_recording_id(self.recording_id)
            or any(
                value is not None
                and (not value or len(value) > MAX_PROVIDER_ID_CHARS)
                for value in text_values
            )
            or any(
                value is not None
                and (
                    not isinstance(value, datetime)
                    or value.tzinfo is None
                    or value.utcoffset() is None
                )
                for value in (self.recording_started_at, self.recording_ended_at)
            )
        ):
            raise ValueError("provider_recording_invalid")

    def __repr__(self) -> str:
        return "ProviderRecordingV1()"


@dataclass(frozen=True, slots=True, repr=False)
class ProviderRecordingPageV1:
    page_number: int | None
    total_pages: int | None
    items: tuple[ProviderRecordingV1, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.page_number is not None
            and type(self.page_number) is not int
            or self.total_pages is not None
            and type(self.total_pages) is not int
            or not isinstance(self.items, tuple)
            or any(not isinstance(item, ProviderRecordingV1) for item in self.items)
        ):
            raise ValueError("provider_recording_page_invalid")

    def __repr__(self) -> str:
        return "ProviderRecordingPageV1()"


class TelnyxRecordingApi(Protocol):
    """Project testing seam over the one process-owned Telnyx client."""

    async def start_recording(
        self,
        call_control_id: str,
        request: RecordingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult: ...

    async def stop_recording(
        self,
        call_control_id: str,
        request: RecordingStopV1,
        *,
        command_id: UUID,
    ) -> CallControlResult: ...

    async def list_recordings_one_page(
        self,
        *,
        call_control_id: str,
        call_leg_id: str | None,
        call_session_id: str | None,
        start_gte_iso: str,
        start_lte_iso: str,
        end_gte_iso: str,
        end_lte_iso: str,
        timeout_seconds: float,
    ) -> ProviderRecordingPageV1: ...

    async def retrieve_recording(
        self,
        recording_id: str,
        *,
        timeout_seconds: float,
    ) -> ProviderRecordingV1: ...

    async def delete_recording(
        self,
        recording_id: str,
        *,
        timeout_seconds: float,
    ) -> ProviderDeleteResultV1: ...

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult: ...


def _raise_capsule_error() -> NoReturn:
    raise RecordingCapsuleError("recording_capsule_invalid") from None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("non_standard_json_constant")


def _valid_bounded_text(value: object, maximum: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum


def _valid_deployment_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= MAX_DEPLOYMENT_ID_UTF8_BYTES
    )


def _canonical_uuid(value: object) -> UUID | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        return None
    return parsed if str(parsed) == value else None


def derive_recording_id(deployment_id: str, call_id: UUID) -> UUID:
    if not _valid_deployment_id(deployment_id) or not isinstance(call_id, UUID):
        raise ValueError("recording_identity_invalid")
    name = _canonical_json_bytes(
        ["projetv0.voice.recording", 1, deployment_id, str(call_id)]
    ).decode("utf-8")
    return uuid5(RECORDING_NAMESPACE_V1, name)


def derive_recording_action_id(recording_id: UUID, action: RecordingAction) -> UUID:
    """Return a deterministic UUIDv4-shaped best-effort command token."""

    if not isinstance(recording_id, UUID) or action not in {
        "recording-start",
        "recording-stop",
        "hangup",
    }:
        raise ValueError("recording_action_invalid")
    digest = bytearray(
        hashlib.sha256(
            _canonical_json_bytes(
                ["projetv0.voice.recording.action", 1, str(recording_id), action]
            )
        ).digest()[:16]
    )
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(digest))


def _derive_webhook_operation_id(
    event: VerifiedWebhook,
    recording_id: UUID,
    purpose: Literal["base", "provider-id-enrichment"],
) -> UUID:
    name = _canonical_json_bytes(
        [event.event_type, event.event_id, str(recording_id), purpose]
    ).decode("utf-8")
    return uuid5(RECORDING_WEBHOOK_OPERATION_NAMESPACE_V1, name)


@dataclass(frozen=True, slots=True, repr=False)
class RecordingCorrelationV1:
    deployment_id: str = field(repr=False)
    call_id: UUID = field(repr=False)
    recording_id: UUID = field(repr=False)
    call_control_id: str = field(repr=False)
    call_leg_id: str | None = field(repr=False)
    call_session_id: str | None = field(repr=False)
    retention_days: int
    required: bool

    def __post_init__(self) -> None:
        if (
            not _valid_deployment_id(self.deployment_id)
            or not isinstance(self.call_id, UUID)
            or not isinstance(self.recording_id, UUID)
            or self.recording_id
            != derive_recording_id(self.deployment_id, self.call_id)
            or not _valid_bounded_text(
                self.call_control_id, MAX_CALL_CONTROL_ID_CHARS
            )
            or self.call_leg_id is not None
            and not _valid_bounded_text(self.call_leg_id, MAX_PROVIDER_ID_CHARS)
            or self.call_session_id is not None
            and not _valid_bounded_text(self.call_session_id, MAX_PROVIDER_ID_CHARS)
            or type(self.retention_days) is not int
            or not 1 <= self.retention_days <= MAX_RETENTION_DAYS
            or type(self.required) is not bool
        ):
            raise ValueError("recording_correlation_invalid") from None

    def __repr__(self) -> str:
        return "RecordingCorrelationV1()"

    def __str__(self) -> str:
        return "RecordingCorrelationV1()"


def build_recording_correlation(
    identity: CallIdentity,
    *,
    retention_days: int,
    required: bool,
) -> RecordingCorrelationV1:
    if not isinstance(identity, CallIdentity):
        raise ValueError("recording_correlation_invalid")
    return RecordingCorrelationV1(
        deployment_id=identity.deployment_id,
        call_id=identity.call_id,
        recording_id=derive_recording_id(identity.deployment_id, identity.call_id),
        call_control_id=identity.telnyx_call_control_id,
        call_leg_id=identity.telnyx_call_leg_id,
        call_session_id=identity.telnyx_call_session_id,
        retention_days=retention_days,
        required=required,
    )


def encode_recording_correlation(correlation: RecordingCorrelationV1) -> SecretStr:
    if not isinstance(correlation, RecordingCorrelationV1):
        raise ValueError("recording_correlation_invalid")
    raw = _canonical_json_bytes(
        {
            "v": 1,
            "deployment_id": correlation.deployment_id,
            "call_id": str(correlation.call_id),
            "recording_id": str(correlation.recording_id),
            "call_control_id": correlation.call_control_id,
            "call_leg_id": correlation.call_leg_id,
            "call_session_id": correlation.call_session_id,
            "retention_days": correlation.retention_days,
            "required": correlation.required,
        }
    )
    if len(raw) > MAX_CLIENT_STATE_JSON_BYTES:
        raise RecordingCapsuleError("recording_capsule_invalid") from None
    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > MAX_CLIENT_STATE_B64_CHARS:
        raise RecordingCapsuleError("recording_capsule_invalid") from None
    return SecretStr(encoded)


def decode_recording_correlation(client_state: SecretStr) -> RecordingCorrelationV1:
    if not isinstance(client_state, SecretStr):
        _raise_capsule_error()
    encoded = client_state.get_secret_value()
    if not 0 < len(encoded) <= MAX_CLIENT_STATE_B64_CHARS:
        _raise_capsule_error()
    try:
        raw = base64.b64decode(encoded, validate=True)
        if base64.b64encode(raw).decode("ascii") != encoded:
            _raise_capsule_error()
        if not 0 < len(raw) <= MAX_CLIENT_STATE_JSON_BYTES:
            _raise_capsule_error()
        parsed = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(parsed, dict) or set(parsed) != _CAPSULE_KEYS:
            _raise_capsule_error()
        if _canonical_json_bytes(parsed) != raw:
            _raise_capsule_error()
        if type(parsed.get("v")) is not int or parsed["v"] != 1:
            _raise_capsule_error()
        deployment_id = parsed.get("deployment_id")
        call_id = _canonical_uuid(parsed.get("call_id"))
        recording_id = _canonical_uuid(parsed.get("recording_id"))
        call_control_id = parsed.get("call_control_id")
        leg_id = parsed.get("call_leg_id")
        session_id = parsed.get("call_session_id")
        retention_days = parsed.get("retention_days")
        required = parsed.get("required")
        if (
            not _valid_deployment_id(deployment_id)
            or call_id is None
            or recording_id is None
            or not _valid_bounded_text(call_control_id, MAX_CALL_CONTROL_ID_CHARS)
            or leg_id is not None
            and not _valid_bounded_text(leg_id, MAX_PROVIDER_ID_CHARS)
            or session_id is not None
            and not _valid_bounded_text(session_id, MAX_PROVIDER_ID_CHARS)
            or type(retention_days) is not int
            or not 1 <= retention_days <= MAX_RETENTION_DAYS
            or type(required) is not bool
        ):
            _raise_capsule_error()
        return RecordingCorrelationV1(
            deployment_id=cast(str, deployment_id),
            call_id=call_id,
            recording_id=recording_id,
            call_control_id=cast(str, call_control_id),
            call_leg_id=cast(str | None, leg_id),
            call_session_id=cast(str | None, session_id),
            retention_days=retention_days,
            required=required,
        )
    except RecordingCapsuleError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, binascii.Error, TypeError, ValueError):
        _raise_capsule_error()


def _require_utc_now(utcnow: Callable[[], datetime]) -> datetime:
    now = utcnow()
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise RecordingLifecycleError("recording_clock_invalid")
    return now.astimezone(UTC)


def _recording_operation(
    *,
    operation_id: UUID,
    correlation: RecordingCorrelationV1,
    occurred_at: datetime,
    status: Literal["pending", "failed"],
) -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=operation_id,
        deployment_id=correlation.deployment_id,
        call_id=correlation.call_id,
        occurred_at=occurred_at,
        kind="recording.upsert",
        payload=RecordingUpsertPayloadV1(
            recording_id=correlation.recording_id,
            status=status,
            telnyx_recording_id=None,
            channels="dual",
            format="wav",
            started_at=None,
            ended_at=None,
            retention_until=None,
        ),
    )


class TelnyxRecordingBoundary(RecordingBoundary):
    """Task 8 recording boundary backed by the existing Telnyx SDK owner."""

    __slots__ = (
        "_operation_id",
        "_play_beep",
        "_required",
        "_retention_days",
        "_telnyx",
        "_utcnow",
        "_writer",
    )

    def __init__(
        self,
        *,
        telnyx: TelnyxRecordingApi,
        writer: ControlWriter,
        retention_days: int,
        required: bool,
        play_beep: bool,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        operation_id: Callable[[], UUID] = uuid4,
    ) -> None:
        if (
            type(retention_days) is not int
            or not 1 <= retention_days <= MAX_RETENTION_DAYS
            or type(required) is not bool
            or type(play_beep) is not bool
            or not callable(utcnow)
            or not callable(operation_id)
        ):
            raise ValueError("recording_boundary_config_invalid")
        self._telnyx = telnyx
        self._writer = writer
        self._retention_days = retention_days
        self._required = required
        self._play_beep = play_beep
        self._utcnow = utcnow
        self._operation_id = operation_id

    def __repr__(self) -> str:
        return "TelnyxRecordingBoundary()"

    async def _commit(self, operation: VoiceOperationV1) -> None:
        await self._writer.commit_control(
            PersistenceCommand("outbox", {"operation": operation}, None)
        )

    def _next_operation_id(self) -> UUID:
        operation_id = self._operation_id()
        if not isinstance(operation_id, UUID) or operation_id.version != 4:
            raise RecordingLifecycleError("recording_operation_id_invalid")
        return operation_id

    async def start(self, identity: CallIdentity) -> RecordingStartResult:
        correlation = build_recording_correlation(
            identity,
            retention_days=self._retention_days,
            required=self._required,
        )
        client_state = encode_recording_correlation(correlation)
        await self._commit(
            _recording_operation(
                operation_id=self._next_operation_id(),
                correlation=correlation,
                occurred_at=_require_utc_now(self._utcnow),
                status="pending",
            )
        )
        result = await self._telnyx.start_recording(
            correlation.call_control_id,
            RecordingStartV1(
                play_beep=self._play_beep,
                client_state=client_state,
            ),
            command_id=derive_recording_action_id(
                correlation.recording_id, "recording-start"
            ),
        )
        if not isinstance(result, CallControlResult):
            raise RecordingLifecycleError("recording_start_invalid")
        if result.outcome == "accepted":
            return RecordingStartResult(RecordingStartState.STARTED)
        if result.outcome in {"rejected", "rate_limited", "retryable_not_sent"}:
            await self._commit(
                _recording_operation(
                    operation_id=self._next_operation_id(),
                    correlation=correlation,
                    occurred_at=_require_utc_now(self._utcnow),
                    status="failed",
                )
            )
            return RecordingStartResult(RecordingStartState.DEFINITELY_NOT_STARTED)
        if result.outcome == "outcome_unknown":
            return RecordingStartResult(
                RecordingStartState.INDETERMINATE,
                gate_may_open=not self._required,
            )
        raise RecordingLifecycleError("recording_start_invalid")

    async def cleanup(
        self,
        identity: CallIdentity,
        *,
        recording_may_be_active: bool,
        reason: str,
    ) -> None:
        if type(recording_may_be_active) is not bool or not isinstance(reason, str):
            raise RecordingLifecycleError("recording_cleanup_failed")
        try:
            correlation = build_recording_correlation(
                identity,
                retention_days=self._retention_days,
                required=self._required,
            )
            client_state = encode_recording_correlation(correlation)
        except Exception:
            raise RecordingLifecycleError("recording_cleanup_failed") from None

        first_cancellation: asyncio.CancelledError | None = None
        failed = False
        if recording_may_be_active:
            stop_result: CallControlResult | None = None
            try:
                stop_result = await self._telnyx.stop_recording(
                    correlation.call_control_id,
                    RecordingStopV1(
                        client_state=client_state,
                        recording_id=None,
                    ),
                    command_id=derive_recording_action_id(
                        correlation.recording_id, "recording-stop"
                    ),
                )
            except asyncio.CancelledError as error:
                first_cancellation = error
            except Exception:
                failed = True
            failed = failed or not isinstance(stop_result, CallControlResult)
            if isinstance(stop_result, CallControlResult):
                failed = failed or stop_result.outcome != "accepted"

        hangup_result, cancellation, hangup_failed = await _join_owned_action(
            self._telnyx.hangup(
                correlation.call_control_id,
                command_id=derive_recording_action_id(
                    correlation.recording_id, "hangup"
                ),
                client_state=client_state,
            ),
            name="voice-recording-hangup",
        )
        if first_cancellation is None and cancellation is not None:
            first_cancellation = cancellation
        failed = failed or hangup_failed or not isinstance(
            hangup_result, CallControlResult
        )
        if isinstance(hangup_result, CallControlResult):
            failed = failed or hangup_result.outcome != "accepted"
        if first_cancellation is not None:
            raise first_cancellation
        if failed:
            raise RecordingLifecycleError("recording_cleanup_failed") from None


async def _join_owned_action(
    awaitable: Awaitable[CallControlResult],
    *,
    name: str,
) -> tuple[CallControlResult | None, asyncio.CancelledError | None, bool]:
    task: asyncio.Future[CallControlResult] = asyncio.ensure_future(awaitable)
    if isinstance(task, asyncio.Task):
        task.set_name(name)
    first_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), first_cancellation, False
        except asyncio.CancelledError as error:
            if task.done():
                return None, first_cancellation or error, True
            if first_cancellation is None:
                first_cancellation = error
        except Exception:
            return None, first_cancellation, True


async def _join_owned_with_deadline(
    awaitable: Awaitable[object],
    *,
    monotonic: Callable[[], float],
    deadline: float,
) -> tuple[object | None, asyncio.CancelledError | None, bool]:
    task: asyncio.Future[object] = asyncio.ensure_future(
        _await_with_deadline(
            awaitable,
            monotonic=monotonic,
            deadline=deadline,
        )
    )
    first_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), first_cancellation, False
        except asyncio.CancelledError as error:
            if task.done():
                return None, first_cancellation or error, True
            if first_cancellation is None:
                first_cancellation = error
        except Exception:
            return None, first_cancellation, True


def _recording_correlation_from_event(event: VerifiedWebhook) -> RecordingCorrelationV1:
    if not isinstance(event, VerifiedWebhook) or event.client_state is None:
        raise RecordingWebhookError("recording_webhook_invalid") from None
    try:
        correlation = decode_recording_correlation(event.client_state)
    except RecordingCapsuleError:
        raise RecordingWebhookError("recording_webhook_invalid") from None
    if (
        event.call_control_id is not None
        and event.call_control_id != correlation.call_control_id
        or event.call_leg_id is not None
        and correlation.call_leg_id is not None
        and event.call_leg_id != correlation.call_leg_id
        or event.call_session_id is not None
        and correlation.call_session_id is not None
        and event.call_session_id != correlation.call_session_id
    ):
        raise RecordingWebhookError("recording_webhook_invalid") from None
    return correlation


def resolve_recording_webhook(
    event: VerifiedWebhook,
) -> WebhookDurableEffect | None:
    """Resolve one signed recording callback without retaining provider URLs."""

    if not isinstance(event, VerifiedWebhook) or event.event_type not in {
        "call.recording.saved",
        "call.recording.error",
    }:
        return None
    try:
        correlation = _recording_correlation_from_event(event)
        started_at = event.recording_started_at
        ended_at = event.recording_ended_at
        if (started_at is None) != (ended_at is None):
            raise ValueError("half_timeline")
        if started_at is not None and ended_at is not None and ended_at < started_at:
            raise ValueError("timeline_order")
        if event.event_type == "call.recording.saved":
            if (
                started_at is None
                or ended_at is None
                or event.recording_channels != "dual"
            ):
                raise ValueError("saved_fields")
            status: Literal["saved", "failed"] = "saved"
            retention_until = ended_at + timedelta(days=correlation.retention_days)
        else:
            status = "failed"
            retention_until = None
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=_derive_webhook_operation_id(
                event, correlation.recording_id, "base"
            ),
            deployment_id=correlation.deployment_id,
            call_id=correlation.call_id,
            occurred_at=event.occurred_at,
            kind="recording.upsert",
            payload=RecordingUpsertPayloadV1(
                recording_id=correlation.recording_id,
                status=status,
                telnyx_recording_id=event.recording_id,
                channels="dual",
                format="wav",
                started_at=started_at,
                ended_at=ended_at,
                retention_until=retention_until,
            ),
        )
        return WebhookDurableEffect(operation=operation)
    except RecordingWebhookError:
        raise
    except Exception:
        raise RecordingWebhookError("recording_webhook_invalid") from None


async def _await_with_deadline(
    awaitable: Awaitable[object],
    *,
    monotonic: Callable[[], float],
    deadline: float,
) -> object:
    remaining = deadline - monotonic()
    if remaining <= 0:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError
    async with asyncio.timeout(remaining):
        return await awaitable


def _bounded_remaining(
    *,
    monotonic: Callable[[], float],
    deadline: float,
    maximum: float,
) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError
    return min(float(remaining), maximum)


def _catalog_match(
    item: ProviderRecordingV1,
    correlation: RecordingCorrelationV1,
    event: VerifiedWebhook,
) -> Literal["valid", "missing", "mismatch", "missing_id"]:
    if item.recording_id is None:
        return "missing_id"
    required_values: tuple[tuple[object, object], ...] = (
        (item.call_control_id, correlation.call_control_id),
        (item.channels, "dual"),
        (item.status, "completed"),
        (item.source, "call"),
        (item.initiated_by, "StartCallRecordingAPI"),
        (item.recording_started_at, event.recording_started_at),
        (item.recording_ended_at, event.recording_ended_at),
    )
    if correlation.call_leg_id is not None:
        required_values += ((item.call_leg_id, correlation.call_leg_id),)
    if correlation.call_session_id is not None:
        required_values += ((item.call_session_id, correlation.call_session_id),)
    if any(actual is None for actual, _ in required_values):
        return "missing"
    for actual, expected in required_values:
        if isinstance(actual, datetime) and isinstance(expected, datetime):
            if abs((actual - expected).total_seconds()) > RECONCILE_TIMESTAMP_SLOP_SECONDS:
                return "mismatch"
        elif actual != expected:
            return "mismatch"
    if (
        item.recording_started_at is None
        or item.recording_ended_at is None
        or item.recording_ended_at < item.recording_started_at
    ):
        return "mismatch"
    return "valid"


def _enrichment_operation(
    event: VerifiedWebhook,
    effect: WebhookDurableEffect,
    provider_recording_id: str,
) -> VoiceOperationV1:
    base = effect.operation
    if (
        base is None
        or base.kind != "recording.upsert"
        or not isinstance(base.payload, RecordingUpsertPayloadV1)
        or base.payload.status != "saved"
        or base.payload.telnyx_recording_id is not None
    ):
        raise RecordingWebhookError("recording_webhook_invalid")
    return base.model_copy(
        update={
            "operation_id": _derive_webhook_operation_id(
                event, base.payload.recording_id, "provider-id-enrichment"
            ),
            "payload": base.payload.model_copy(
                update={"telnyx_recording_id": provider_recording_id}
            ),
        }
    )


async def _reconcile_provider_recording(
    event: VerifiedWebhook,
    effect: WebhookDurableEffect,
    *,
    correlation: RecordingCorrelationV1,
    telnyx: TelnyxRecordingApi,
    writer: ControlWriter,
    monotonic: Callable[[], float],
    deadline: float,
) -> WebhookDisposition:
    started_at = event.recording_started_at
    ended_at = event.recording_ended_at
    if started_at is None or ended_at is None:
        return WebhookDisposition(500)
    slop = timedelta(seconds=RECONCILE_TIMESTAMP_SLOP_SECONDS)
    try:
        list_timeout = _bounded_remaining(
            monotonic=monotonic,
            deadline=deadline,
            maximum=RECORDING_LIST_DEADLINE_SECONDS,
        )
        page = await _await_with_deadline(
            telnyx.list_recordings_one_page(
                call_control_id=correlation.call_control_id,
                call_leg_id=correlation.call_leg_id,
                call_session_id=correlation.call_session_id,
                start_gte_iso=_utc_iso(started_at - slop),
                start_lte_iso=_utc_iso(started_at + slop),
                end_gte_iso=_utc_iso(ended_at - slop),
                end_lte_iso=_utc_iso(ended_at + slop),
                timeout_seconds=list_timeout,
            ),
            monotonic=monotonic,
            deadline=deadline,
        )
    except asyncio.CancelledError:
        raise
    except (RecordingCatalogTransientError, TimeoutError):
        return WebhookDisposition(503)
    except (RecordingCatalogInvalidError, Exception):
        return WebhookDisposition(500)
    if not isinstance(page, ProviderRecordingPageV1):
        return WebhookDisposition(500)
    if page.page_number != 1 or page.total_pages != 1:
        return WebhookDisposition(500)
    if not page.items:
        return WebhookDisposition(503)
    if len(page.items) != 1:
        return WebhookDisposition(500)
    item = page.items[0]
    match = _catalog_match(item, correlation, event)
    if match == "missing_id" or match == "mismatch":
        return WebhookDisposition(500)
    if match == "missing":
        assert item.recording_id is not None
        try:
            retrieve_timeout = _bounded_remaining(
                monotonic=monotonic,
                deadline=deadline,
                maximum=RECORDING_RETRIEVE_DEADLINE_SECONDS,
            )
            retrieved = await _await_with_deadline(
                telnyx.retrieve_recording(
                    item.recording_id,
                    timeout_seconds=retrieve_timeout,
                ),
                monotonic=monotonic,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except (RecordingCatalogTransientError, TimeoutError):
            return WebhookDisposition(503)
        except (RecordingCatalogInvalidError, Exception):
            return WebhookDisposition(500)
        if (
            not isinstance(retrieved, ProviderRecordingV1)
            or retrieved.recording_id != item.recording_id
            or _catalog_match(retrieved, correlation, event) != "valid"
        ):
            return WebhookDisposition(500)
        item = retrieved
    assert item.recording_id is not None
    try:
        operation = _enrichment_operation(event, effect, item.recording_id)
        enrichment_fingerprint = hashlib.sha256(
            canonical_operation_bytes(operation)
        ).digest()
        await writer.commit_control(
            PersistenceCommand(
                "webhook_enrichment",
                {
                    "receipt": {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "call_control_id": event.call_control_id,
                        "occurred_at": event.occurred_at,
                        "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
                    },
                    "enrichment_fingerprint_sha256": enrichment_fingerprint,
                    "operation": operation,
                },
                None,
            )
        )
    except asyncio.CancelledError:
        raise
    except CommandConflictError:
        return WebhookDisposition(500)
    except (PersistenceError, TimeoutError):
        return WebhookDisposition(503)
    except Exception:
        return WebhookDisposition(500)
    return WebhookDisposition(200)


async def after_recording_webhook_commit(
    event: VerifiedWebhook,
    effect: WebhookDurableEffect | None,
    *,
    telnyx: TelnyxRecordingApi,
    writer: ControlWriter,
    local_drain: Callable[[UUID, str], Awaitable[None]],
    monotonic: Callable[[], float],
    timeout_seconds: float,
) -> WebhookDisposition | None:
    if not isinstance(event, VerifiedWebhook) or event.event_type not in {
        "call.recording.saved",
        "call.recording.error",
    }:
        return None
    if (
        effect is None
        or not isinstance(effect, WebhookDurableEffect)
        or effect.operation is None
        or type(timeout_seconds) not in {float, int}
        or timeout_seconds <= 0
    ):
        return WebhookDisposition(500)
    try:
        started = monotonic()
        if not isinstance(started, (float, int)):
            return WebhookDisposition(500)
        deadline = float(started) + float(timeout_seconds)
        correlation = _recording_correlation_from_event(event)
    except Exception:
        return WebhookDisposition(500)
    if event.event_type == "call.recording.saved":
        if event.recording_id is not None:
            return WebhookDisposition(200)
        return await _reconcile_provider_recording(
            event,
            effect,
            correlation=correlation,
            telnyx=telnyx,
            writer=writer,
            monotonic=monotonic,
            deadline=deadline,
        )
    if not correlation.required:
        return WebhookDisposition(200)

    first_cancellation: asyncio.CancelledError | None = None
    drain_failed = False
    try:
        drain_result = local_drain(correlation.call_id, "recording_required_error")
        await _await_with_deadline(
            drain_result,
            monotonic=monotonic,
            deadline=deadline - REQUIRED_HANGUP_RESERVE_SECONDS,
        )
    except asyncio.CancelledError as error:
        first_cancellation = error
    except Exception:
        drain_failed = True

    hangup_result, hangup_cancellation, hangup_failed = (
        await _join_owned_with_deadline(
            telnyx.hangup(
                correlation.call_control_id,
                command_id=derive_recording_action_id(
                    correlation.recording_id, "hangup"
                ),
                client_state=event.client_state,
            ),
            monotonic=monotonic,
            deadline=deadline,
        )
    )
    if first_cancellation is None and hangup_cancellation is not None:
        first_cancellation = hangup_cancellation
    if first_cancellation is not None:
        raise first_cancellation
    if drain_failed:
        return WebhookDisposition(500)
    if hangup_failed or not isinstance(hangup_result, CallControlResult):
        return WebhookDisposition(503)
    if hangup_result.outcome == "accepted":
        return WebhookDisposition(200)
    return WebhookDisposition(503)


def _purge_now(utcnow: Callable[[], datetime]) -> datetime:
    value = utcnow()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RecordingPurgeError("recording_purge_clock_invalid")
    return value.astimezone(UTC)


async def purge_recordings_once(
    *,
    worker_id: str,
    lease_seconds: int,
    batch_size: int,
    telnyx: TelnyxRecordingApi,
    sink: PostgresOperationSink,
    utcnow: Callable[[], datetime],
) -> PurgeBatchResult:
    if (
        not isinstance(worker_id, str)
        or not worker_id
        or type(lease_seconds) is not int
        or not 1 <= lease_seconds <= 300
        or type(batch_size) is not int
        or not 1 <= batch_size <= 100
        or not callable(utcnow)
    ):
        raise RecordingPurgeError("recording_purge_input_invalid")
    try:
        leased_result = await sink.lease_recording_purges(
            worker_id, lease_seconds, batch_size
        )
        leases = tuple(leased_result)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise RecordingPurgeError("recording_purge_lease_failed") from None
    if len(leases) > batch_size or any(
        not isinstance(lease, RecordingPurgeLease) for lease in leases
    ):
        raise RecordingPurgeError("recording_purge_lease_failed")

    deleted = 0
    not_found = 0
    retry = 0
    failed = 0
    stale = 0
    expired = 0
    before_delete_reserve = (
        RECORDING_DELETE_DEADLINE_SECONDS
        + PURGE_ACK_RESERVE_SECONDS
        + PURGE_CLOCK_SKEW_RESERVE_SECONDS
    )
    before_ack_reserve = PURGE_ACK_RESERVE_SECONDS + PURGE_CLOCK_SKEW_RESERVE_SECONDS
    for lease in leases:
        now = _purge_now(utcnow)
        if (lease.lease_expires_at - now).total_seconds() < before_delete_reserve:
            expired += 1
            continue
        provider_id = lease.telnyx_recording_id
        outcome: PurgeOutcome
        if not is_valid_provider_recording_id(provider_id):
            outcome = "failed"
        else:
            try:
                result = await telnyx.delete_recording(
                    provider_id,
                    timeout_seconds=RECORDING_DELETE_DEADLINE_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                result = ProviderDeleteResultV1("retry")
            if not isinstance(result, ProviderDeleteResultV1):
                outcome = "retry"
            elif result.outcome == "deleted":
                outcome = "deleted" if result.recording_id == provider_id else "retry"
            else:
                outcome = result.outcome

        ack_at = _purge_now(utcnow)
        if (lease.lease_expires_at - ack_at).total_seconds() < before_ack_reserve:
            expired += 1
            continue
        try:
            async with asyncio.timeout(PURGE_ACK_RESERVE_SECONDS):
                await sink.ack_recording_purge(
                    lease.recording_id,
                    lease.lease_token,
                    outcome,
                    ack_at,
                )
        except asyncio.CancelledError:
            raise
        except OperationSinkStaleLeaseError:
            stale += 1
            continue
        except (
            OperationSinkTransientError,
            OperationSinkCommitAmbiguousError,
            OperationSinkPermanentError,
            OperationSinkContractError,
            OperationConflictError,
        ):
            raise RecordingPurgeError("recording_purge_ack_failed") from None
        except Exception:
            raise RecordingPurgeError("recording_purge_ack_failed") from None
        if outcome == "deleted":
            deleted += 1
        elif outcome == "not_found":
            not_found += 1
        elif outcome == "retry":
            retry += 1
        else:
            failed += 1
    return PurgeBatchResult(
        leased=len(leases),
        deleted=deleted,
        not_found=not_found,
        retry=retry,
        failed=failed,
        stale=stale,
        expired=expired,
    )
