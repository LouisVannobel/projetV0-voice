from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import SecretStr, ValidationError

from projetv0_voice.models import RecordingUpsertPayloadV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.persistence.postgres_sink import (
    OperationConflictError,
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkPermanentError,
    OperationSinkStaleLeaseError,
    OperationSinkTransientError,
    RecordingPurgeLease,
)
from projetv0_voice.session import CallIdentity, RecordingStartState
from projetv0_voice.telnyx.call_control import (
    CallControlResult,
    RecordingStartV1,
)
from projetv0_voice.telnyx.recordings import (
    MAX_CLIENT_STATE_B64_CHARS,
    MAX_CLIENT_STATE_JSON_BYTES,
    RECORDING_NAMESPACE_V1,
    RECORDING_WEBHOOK_OPERATION_NAMESPACE_V1,
    ProviderDeleteResultV1,
    PurgeBatchResult,
    RecordingCapsuleError,
    RecordingCorrelationV1,
    RecordingLifecycleError,
    RecordingPurgeError,
    RecordingWebhookError,
    TelnyxRecordingBoundary,
    after_recording_webhook_commit,
    build_recording_correlation,
    decode_recording_correlation,
    derive_recording_action_id,
    derive_recording_id,
    encode_recording_correlation,
    purge_recordings_once,
    resolve_recording_webhook,
)
from projetv0_voice.telnyx.webhooks import (
    VerifiedWebhook,
    WebhookDisposition,
    WebhookDurableEffect,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
CALL_ID = UUID("12345678-1234-4234-8234-1234567890ab")
DEPLOYMENT_ID = "agent-révision-7"
CALL_CONTROL_ID = "v3:RAW-CALL-CONTROL-SENTINEL"
LEG_ID = "RAW-LEG-SENTINEL"
SESSION_ID = "RAW-SESSION-SENTINEL"
_DEFAULT_CLIENT_STATE = object()


def identity(**updates: object) -> CallIdentity:
    values: dict[str, object] = {
        "call_id": CALL_ID,
        "durable_generation": "generation-ignored",
        "lease_identity": "lease-ignored",
        "lease_claim": object(),
        "deployment_id": DEPLOYMENT_ID,
        "registry_handle": object(),
        "telnyx_call_control_id": CALL_CONTROL_ID,
        "telnyx_call_leg_id": LEG_ID,
        "telnyx_call_session_id": SESSION_ID,
        "stream_id": "stream-ignored",
        "started_at": NOW,
        "retention_until": NOW + timedelta(days=30),
    }
    values.update(updates)
    return CallIdentity(**values)  # type: ignore[arg-type]


def test_recording_identity_action_and_capsule_golden_vectors_are_frozen() -> None:
    correlation = build_recording_correlation(
        identity(), retention_days=30, required=False
    )
    encoded = encode_recording_correlation(correlation)

    assert UUID("8f0f6b4b-6194-4a43-87a0-649810b2760f") == RECORDING_NAMESPACE_V1
    assert UUID(
        "eeec9d6b-b4f8-4e40-b26e-62c7fe04e218"
    ) == RECORDING_WEBHOOK_OPERATION_NAMESPACE_V1
    assert correlation.recording_id == UUID("7daaf468-59d5-58ef-9c74-cfedbfd07a3b")
    assert derive_recording_action_id(correlation.recording_id, "recording-start") == UUID(
        "0385dab8-10f3-42c0-a06b-52bcebf1dcb5"
    )
    assert derive_recording_action_id(correlation.recording_id, "recording-stop") == UUID(
        "106ffaf9-be8b-4fad-87b9-ec2787139aa5"
    )
    assert derive_recording_action_id(correlation.recording_id, "hangup") == UUID(
        "6bf8c2d1-99e3-432e-a318-2c586a48cdec"
    )
    assert encoded.get_secret_value() == (
        "eyJjYWxsX2NvbnRyb2xfaWQiOiJ2MzpSQVctQ0FMTC1DT05UUk9MLVNFTlRJTkVMIiwiY2FsbF9pZCI6"
        "IjEyMzQ1Njc4LTEyMzQtNDIzNC04MjM0LTEyMzQ1Njc4OTBhYiIsImNhbGxfbGVnX2lkIjoiUkFXLUxF"
        "Ry1TRU5USU5FTCIsImNhbGxfc2Vzc2lvbl9pZCI6IlJBVy1TRVNTSU9OLVNFTlRJTkVMIiwiZGVwbG95"
        "bWVudF9pZCI6ImFnZW50LXLDqXZpc2lvbi03IiwicmVjb3JkaW5nX2lkIjoiN2RhYWY0NjgtNTlkNS01"
        "OGVmLTljNzQtY2ZlZGJmZDA3YTNiIiwicmVxdWlyZWQiOmZhbHNlLCJyZXRlbnRpb25fZGF5cyI6MzAs"
        "InYiOjF9"
    )
    assert decode_recording_correlation(encoded) == correlation


def test_identity_excludes_mutable_call_runtime_values_and_actions_are_distinct() -> None:
    first = build_recording_correlation(identity(), retention_days=30, required=True)
    changed = build_recording_correlation(
        identity(
            durable_generation="changed-generation",
            lease_identity="changed-lease",
            lease_claim=object(),
            registry_handle=object(),
            stream_id="changed-stream",
            started_at=NOW + timedelta(hours=1),
            retention_until=NOW + timedelta(days=90),
        ),
        retention_days=30,
        required=True,
    )

    assert first.recording_id == changed.recording_id
    assert len(
        {
            derive_recording_action_id(first.recording_id, "recording-start"),
            derive_recording_action_id(first.recording_id, "recording-stop"),
            derive_recording_action_id(first.recording_id, "hangup"),
        }
    ) == 3
    assert derive_recording_action_id(first.recording_id, "hangup") == (
        derive_recording_action_id(changed.recording_id, "hangup")
    )


def test_capsule_is_exact_canonical_bounded_and_recomputes_recording_identity() -> None:
    correlation = build_recording_correlation(identity(), retention_days=30, required=False)
    encoded = encode_recording_correlation(correlation)
    raw = base64.b64decode(encoded.get_secret_value(), validate=True)
    decoded = json.loads(raw)

    assert len(encoded.get_secret_value()) <= MAX_CLIENT_STATE_B64_CHARS == 4096
    assert len(raw) <= MAX_CLIENT_STATE_JSON_BYTES == 3072
    assert raw == json.dumps(
        decoded,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert set(decoded) == {
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
    assert derive_recording_id(DEPLOYMENT_ID, CALL_ID) == correlation.recording_id


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: {**value, "extra": "forbidden"},
        lambda value: {**value, "recording_id": "00000000-0000-4000-8000-000000000000"},
        lambda value: {**value, "call_id": str(CALL_ID).upper()},
        lambda value: {**value, "retention_days": True},
        lambda value: {**value, "required": 0},
    ],
)
def test_capsule_rejects_extra_mismatch_noncanonical_uuid_and_coercion(
    mutate: object,
) -> None:
    original = build_recording_correlation(identity(), retention_days=30, required=False)
    values = json.loads(base64.b64decode(encode_recording_correlation(original).get_secret_value()))
    changed = mutate(values)  # type: ignore[operator]
    raw = json.dumps(
        changed,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded = SecretStr(base64.b64encode(raw).decode("ascii"))

    with pytest.raises(RecordingCapsuleError, match="recording_capsule_invalid") as raised:
        decode_recording_correlation(encoded)

    assert CALL_CONTROL_ID not in repr(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"v":1,"v":1}',
        b'{"v":NaN}',
        b"not-json",
    ],
)
def test_capsule_rejects_duplicate_keys_constants_and_non_json_without_echo(raw: bytes) -> None:
    encoded = SecretStr(base64.b64encode(raw).decode("ascii"))
    with pytest.raises(RecordingCapsuleError, match="recording_capsule_invalid") as raised:
        decode_recording_correlation(encoded)
    assert encoded.get_secret_value() not in repr(raised.value)


@pytest.mark.parametrize(
    "encoded",
    [
        "not base64!",
        base64.b64encode(b"{}").decode("ascii").rstrip("="),
        "A" * 4097,
    ],
)
def test_capsule_requires_canonical_padded_base64_and_fixed_bounds(encoded: str) -> None:
    with pytest.raises(RecordingCapsuleError, match="recording_capsule_invalid") as raised:
        decode_recording_correlation(SecretStr(encoded))
    assert encoded not in repr(raised.value)


def test_capsule_correlation_and_request_surfaces_are_input_redacting() -> None:
    correlation = build_recording_correlation(identity(), retention_days=30, required=False)
    capsule = encode_recording_correlation(correlation)
    request = RecordingStartV1(play_beep=True, client_state=capsule)

    assert isinstance(request.client_state, SecretStr)
    assert request.client_state.get_secret_value() == capsule.get_secret_value()
    assert request.model_dump() == {"play_beep": True}
    assert request.model_dump_json() == '{"play_beep":true}'
    rendered = repr(correlation) + str(correlation) + repr(request) + str(request)
    for secret in (
        CALL_CONTROL_ID,
        LEG_ID,
        SESSION_ID,
        capsule.get_secret_value(),
    ):
        assert secret not in rendered
    with pytest.raises(ValidationError) as raised:
        RecordingStartV1(play_beep=True, client_state="x" * 4097)
    assert "x" * 4097 not in str(raised.value)


def test_recording_correlation_rejects_direct_invalid_construction_without_leaks() -> None:
    with pytest.raises(ValueError, match="recording_correlation_invalid") as raised:
        RecordingCorrelationV1(
            deployment_id="",
            call_id=CALL_ID,
            recording_id=UUID(int=0),
            call_control_id=CALL_CONTROL_ID,
            call_leg_id=None,
            call_session_id=None,
            retention_days=30,
            required=False,
        )
    assert CALL_CONTROL_ID not in repr(raised.value)


def test_provider_recording_metadata_rejects_url_shaped_recording_id() -> None:
    from projetv0_voice.telnyx.recordings import ProviderRecordingV1

    with pytest.raises(ValueError, match="provider_recording_invalid"):
        ProviderRecordingV1(
            recording_id="https://RAW-URL-SENTINEL",
            call_control_id=CALL_CONTROL_ID,
            call_leg_id=LEG_ID,
            call_session_id=SESSION_ID,
            channels="dual",
            status="completed",
            source="call",
            initiated_by="StartCallRecordingAPI",
            recording_started_at=NOW,
            recording_ended_at=NOW + timedelta(minutes=1),
        )


def test_catalog_and_delete_results_accept_256_url_safe_opaque_recording_id() -> None:
    from projetv0_voice.telnyx.recordings import ProviderRecordingV1

    provider_id = "r." + "A" * 251 + "~_-"
    catalog = ProviderRecordingV1(
        recording_id=provider_id,
        call_control_id=CALL_CONTROL_ID,
        call_leg_id=LEG_ID,
        call_session_id=SESSION_ID,
        channels="dual",
        status="completed",
        source="call",
        initiated_by="StartCallRecordingAPI",
        recording_started_at=NOW,
        recording_ended_at=NOW + timedelta(minutes=1),
    )
    deleted = ProviderDeleteResultV1("deleted", provider_id)

    assert catalog.recording_id == provider_id
    assert deleted.recording_id == provider_id


@pytest.mark.parametrize("length", [257, 1024])
def test_provider_recording_accepts_call_control_id_through_1024(length: int) -> None:
    from projetv0_voice.telnyx.recordings import ProviderRecordingV1

    call_control_id = "c" * length
    item = ProviderRecordingV1(
        recording_id="recording_Ab-12",
        call_control_id=call_control_id,
        call_leg_id=LEG_ID,
        call_session_id=SESSION_ID,
        channels="dual",
        status="completed",
        source="call",
        initiated_by="StartCallRecordingAPI",
        recording_started_at=NOW,
        recording_ended_at=NOW + timedelta(minutes=1),
    )

    assert item.call_control_id == call_control_id
    assert call_control_id not in repr(item)


def test_provider_recording_rejects_call_control_id_over_1024() -> None:
    from projetv0_voice.telnyx.recordings import ProviderRecordingV1

    with pytest.raises(ValueError, match="provider_recording_invalid"):
        ProviderRecordingV1(
            recording_id="recording_Ab-12",
            call_control_id="c" * 1025,
            call_leg_id=LEG_ID,
            call_session_id=SESSION_ID,
            channels="dual",
            status="completed",
            source="call",
            initiated_by="StartCallRecordingAPI",
            recording_started_at=NOW,
            recording_ended_at=NOW + timedelta(minutes=1),
        )


@pytest.mark.parametrize("provider_id", [".", ".."])
def test_catalog_and_delete_results_reject_exact_dot_segments(provider_id: str) -> None:
    from projetv0_voice.telnyx.recordings import ProviderRecordingV1

    with pytest.raises(ValueError, match="provider_recording_invalid"):
        ProviderRecordingV1(
            recording_id=provider_id,
            call_control_id=CALL_CONTROL_ID,
            call_leg_id=LEG_ID,
            call_session_id=SESSION_ID,
            channels="dual",
            status="completed",
            source="call",
            initiated_by="StartCallRecordingAPI",
            recording_started_at=NOW,
            recording_ended_at=NOW + timedelta(minutes=1),
        )
    with pytest.raises(ValueError, match="provider_delete_result_invalid"):
        ProviderDeleteResultV1("deleted", provider_id)


class StubWriter:
    def __init__(self, results: list[object] | None = None) -> None:
        self.results = list(results or [])
        self.commands: list[PersistenceCommand] = []

    async def commit_control(self, command: PersistenceCommand) -> None:
        self.commands.append(command)
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            if callable(result):
                await result()


class StubRecordingApi:
    def __init__(self, result: object) -> None:
        self.result = result
        self.start_calls: list[tuple[str, RecordingStartV1, UUID]] = []

    async def start_recording(
        self,
        call_control_id: str,
        request: RecordingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        self.start_calls.append((call_control_id, request, command_id))
        if isinstance(self.result, BaseException):
            raise self.result
        if callable(self.result):
            return await self.result()
        return self.result  # type: ignore[return-value]


def boundary(
    api: StubRecordingApi,
    writer: StubWriter,
    *,
    required: bool = False,
) -> TelnyxRecordingBoundary:
    operation_ids = iter(
        (
            UUID("10000000-0000-4000-8000-000000000001"),
            UUID("10000000-0000-4000-8000-000000000002"),
        )
    )
    return TelnyxRecordingBoundary(
        telnyx=api,
        writer=writer,
        retention_days=30,
        required=required,
        play_beep=True,
        utcnow=lambda: NOW,
        operation_id=lambda: next(operation_ids),
    )


def recording_operations(writer: StubWriter) -> list[object]:
    return [
        command.payload["operation"]
        for command in writer.commands
        if command.kind == "outbox"
    ]


@pytest.mark.asyncio
async def test_start_commits_pending_before_exact_provider_start_and_leaves_pending() -> None:
    writer = StubWriter()
    api = StubRecordingApi(CallControlResult("accepted"))

    result = await boundary(api, writer).start(identity())

    assert result.state is RecordingStartState.STARTED
    assert result.gate_may_open is False
    assert len(api.start_calls) == 1
    control_id, request, command_id = api.start_calls[0]
    assert control_id == CALL_CONTROL_ID
    assert request.play_beep is True
    correlation = decode_recording_correlation(request.client_state)
    assert correlation == build_recording_correlation(
        identity(), retention_days=30, required=False
    )
    assert command_id == derive_recording_action_id(
        correlation.recording_id, "recording-start"
    )
    operations = recording_operations(writer)
    assert len(operations) == 1
    operation = operations[0]
    assert operation.operation_id == UUID("10000000-0000-4000-8000-000000000001")
    assert operation.deployment_id == DEPLOYMENT_ID
    assert operation.call_id == CALL_ID
    assert operation.occurred_at == NOW
    assert operation.kind == "recording.upsert"
    assert operation.payload == RecordingUpsertPayloadV1(
        recording_id=correlation.recording_id,
        status="pending",
        telnyx_recording_id=None,
        channels="dual",
        format="wav",
        started_at=None,
        ended_at=None,
        retention_until=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rejected", "rate_limited", "retryable_not_sent"])
async def test_definite_start_rejection_commits_failed_and_never_emits_active(
    outcome: str,
) -> None:
    writer = StubWriter()
    api = StubRecordingApi(CallControlResult(outcome))  # type: ignore[arg-type]

    result = await boundary(api, writer, required=True).start(identity())

    assert result.state is RecordingStartState.DEFINITELY_NOT_STARTED
    assert [operation.payload.status for operation in recording_operations(writer)] == [
        "pending",
        "failed",
    ]
    failed = recording_operations(writer)[1]
    assert failed.payload.started_at is None
    assert failed.payload.ended_at is None
    assert failed.payload.retention_until is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("required", "gate_may_open"), [(False, True), (True, False)])
async def test_unknown_start_outcome_remains_pending_with_policy_gate(
    required: bool, gate_may_open: bool
) -> None:
    writer = StubWriter()
    api = StubRecordingApi(CallControlResult("outcome_unknown"))

    result = await boundary(api, writer, required=required).start(identity())

    assert result.state is RecordingStartState.INDETERMINATE
    assert result.gate_may_open is gate_may_open
    assert [operation.payload.status for operation in recording_operations(writer)] == [
        "pending"
    ]


@pytest.mark.asyncio
async def test_pending_writer_failure_makes_zero_provider_calls() -> None:
    writer = StubWriter([RuntimeError("RAW-WRITER-SENTINEL")])
    api = StubRecordingApi(CallControlResult("accepted"))

    with pytest.raises(RuntimeError, match="RAW-WRITER-SENTINEL"):
        await boundary(api, writer).start(identity())

    assert api.start_calls == []


@pytest.mark.asyncio
async def test_malformed_provider_result_fails_closed_without_state_fabrication() -> None:
    writer = StubWriter()
    api = StubRecordingApi(object())

    with pytest.raises(RecordingLifecycleError, match="recording_start_invalid") as raised:
        await boundary(api, writer).start(identity())

    assert [operation.payload.status for operation in recording_operations(writer)] == [
        "pending"
    ]
    assert "object" not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["pending", "provider", "accepted", "failed"])
async def test_start_propagates_cancellation_at_every_owned_cut(cut: str) -> None:
    entered = asyncio.Event()

    async def block() -> object:
        entered.set()
        await asyncio.Future()

    async def accepted_then_cancel() -> CallControlResult:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        return CallControlResult("accepted")

    if cut == "pending":
        writer = StubWriter([block])
        api = StubRecordingApi(CallControlResult("accepted"))
    elif cut == "provider":
        writer = StubWriter()
        api = StubRecordingApi(block)
    elif cut == "accepted":
        writer = StubWriter()
        api = StubRecordingApi(accepted_then_cancel)
    else:
        writer = StubWriter([None, block])
        api = StubRecordingApi(CallControlResult("rejected"))
    task = asyncio.create_task(boundary(api, writer).start(identity()))
    if cut != "accepted":
        await entered.wait()
        task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    if cut == "pending":
        assert api.start_calls == []


def recording_event(
    event_type: str,
    *,
    provider_recording_id: str | None,
    required: bool = False,
    client_state: object = _DEFAULT_CLIENT_STATE,
    started_at: datetime | None = NOW,
    ended_at: datetime | None = NOW + timedelta(minutes=1),
    channels: str | None = "dual",
    call_control_id: str | None = None,
    call_leg_id: str | None = LEG_ID,
    call_session_id: str | None = SESSION_ID,
) -> VerifiedWebhook:
    correlation = build_recording_correlation(
        identity(), retention_days=30, required=required
    )
    return VerifiedWebhook(
        event_id=(
            "event-saved-1"
            if event_type == "call.recording.saved"
            else "event-error-1"
        ),
        event_type=event_type,
        occurred_at=NOW + timedelta(minutes=2),
        call_control_id=call_control_id,
        call_leg_id=call_leg_id,
        call_session_id=call_session_id,
        recording_id=provider_recording_id,
        stream_id=None,
        client_state=(
            encode_recording_correlation(correlation)
            if client_state is _DEFAULT_CLIENT_STATE
            else client_state
        ),  # type: ignore[arg-type]
        recording_started_at=started_at,
        recording_ended_at=ended_at,
        recording_channels=channels,
        semantic_fingerprint_sha256=b"w" * 32,
    )


def test_saved_resolver_builds_deterministic_url_free_atomic_operation() -> None:
    event = recording_event(
        "call.recording.saved",
        provider_recording_id="recording_Ab-12",
        call_control_id=None,
    )

    effect = resolve_recording_webhook(event)

    assert isinstance(effect, WebhookDurableEffect)
    assert effect.lease is None
    assert effect.operation is not None
    operation = effect.operation
    assert operation.operation_id == UUID("ff3ceaf2-1cd2-5686-8442-6c512ead9906")
    assert operation.deployment_id == DEPLOYMENT_ID
    assert operation.call_id == CALL_ID
    assert operation.occurred_at == NOW + timedelta(minutes=2)
    assert operation.kind == "recording.upsert"
    assert operation.payload == RecordingUpsertPayloadV1(
        recording_id=derive_recording_id(DEPLOYMENT_ID, CALL_ID),
        status="saved",
        telnyx_recording_id="recording_Ab-12",
        channels="dual",
        format="wav",
        started_at=NOW,
        ended_at=NOW + timedelta(minutes=1),
        retention_until=NOW + timedelta(days=30, minutes=1),
    )
    rendered = repr(effect)
    assert "recording_Ab-12" not in rendered
    assert "RAW-CLIENT-STATE" not in rendered


def test_error_resolver_builds_failed_without_inventing_timeline_or_reason() -> None:
    event = recording_event(
        "call.recording.error",
        provider_recording_id=None,
        started_at=None,
        ended_at=None,
        channels=None,
        call_leg_id=None,
        call_session_id=None,
    )

    effect = resolve_recording_webhook(event)

    assert effect is not None and effect.operation is not None
    payload = effect.operation.payload
    assert isinstance(payload, RecordingUpsertPayloadV1)
    assert payload.status == "failed"
    assert payload.telnyx_recording_id is None
    assert payload.started_at is None
    assert payload.ended_at is None
    assert payload.retention_until is None


@pytest.mark.parametrize(
    "updates",
    [
        {"client_state": None},
        {"call_control_id": "wrong-control"},
        {"call_leg_id": "wrong-leg"},
        {"call_session_id": "wrong-session"},
        {"channels": "single"},
        {"started_at": None},
        {"ended_at": NOW - timedelta(seconds=1)},
    ],
)
def test_saved_resolver_rejects_missing_capsule_identity_timeline_and_channels(
    updates: dict[str, object],
) -> None:
    event = recording_event(
        "call.recording.saved",
        provider_recording_id=None,
        **updates,  # type: ignore[arg-type]
    )

    with pytest.raises(RecordingWebhookError, match="recording_webhook_invalid") as raised:
        resolve_recording_webhook(event)

    assert CALL_CONTROL_ID not in repr(raised.value)


def test_unhandled_webhook_has_no_recording_effect() -> None:
    event = recording_event("call.answered", provider_recording_id=None)
    assert resolve_recording_webhook(event) is None


class TerminationApi:
    def __init__(self, result: object, events: list[str]) -> None:
        self.result = result
        self.events = events
        self.hangup_calls: list[tuple[str, UUID, SecretStr | None]] = []

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult:
        self.events.append("hangup")
        self.hangup_calls.append((call_control_id, command_id, client_state))
        if isinstance(self.result, BaseException):
            raise self.result
        if callable(self.result):
            return await self.result()
        return self.result  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_optional_recording_error_returns_200_without_termination() -> None:
    event = recording_event(
        "call.recording.error",
        provider_recording_id=None,
        required=False,
        started_at=None,
        ended_at=None,
        channels=None,
    )
    effect = resolve_recording_webhook(event)
    events: list[str] = []
    api = TerminationApi(CallControlResult("accepted"), events)

    result = await after_recording_webhook_commit(
        event,
        effect,
        telnyx=api,
        writer=StubWriter(),
        local_drain=lambda _call_id, _reason: asyncio.sleep(0),
        monotonic=lambda: 1.0,
        timeout_seconds=2.0,
    )

    assert result == WebhookDisposition(200)
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hangup_outcome", "expected"),
    [
        ("accepted", 200),
        ("rejected", 503),
        ("rate_limited", 503),
        ("retryable_not_sent", 503),
        ("outcome_unknown", 503),
    ],
)
async def test_required_error_drains_then_reuses_shared_hangup_with_deterministic_status(
    hangup_outcome: str, expected: int
) -> None:
    event = recording_event(
        "call.recording.error",
        provider_recording_id=None,
        required=True,
        started_at=None,
        ended_at=None,
        channels=None,
    )
    effect = resolve_recording_webhook(event)
    events: list[str] = []
    api = TerminationApi(CallControlResult(hangup_outcome), events)  # type: ignore[arg-type]

    async def local_drain(call_id: UUID, reason: str) -> None:
        assert call_id == CALL_ID
        assert reason == "recording_required_error"
        events.append("local_drain")

    result = await after_recording_webhook_commit(
        event,
        effect,
        telnyx=api,
        writer=StubWriter(),
        local_drain=local_drain,
        monotonic=lambda: 1.0,
        timeout_seconds=2.0,
    )

    assert result == WebhookDisposition(expected)
    assert events == ["local_drain", "hangup"]
    correlation = decode_recording_correlation(event.client_state)  # type: ignore[arg-type]
    assert api.hangup_calls == [
        (
            CALL_CONTROL_ID,
            derive_recording_action_id(correlation.recording_id, "hangup"),
            event.client_state,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["exception", "cancellation"])
async def test_local_drain_fault_cannot_suppress_shared_hangup(fault: str) -> None:
    event = recording_event(
        "call.recording.error",
        provider_recording_id=None,
        required=True,
        started_at=None,
        ended_at=None,
        channels=None,
    )
    effect = resolve_recording_webhook(event)
    events: list[str] = []
    api = TerminationApi(CallControlResult("accepted"), events)
    cancellation = asyncio.CancelledError("caller-cancelled")

    async def local_drain(_call_id: UUID, _reason: str) -> None:
        events.append("local_drain")
        if fault == "cancellation":
            raise cancellation
        raise RuntimeError("RAW-DRAIN-SENTINEL")

    if fault == "cancellation":
        with pytest.raises(asyncio.CancelledError) as raised:
            await after_recording_webhook_commit(
                event,
                effect,
                telnyx=api,
                writer=StubWriter(),
                local_drain=local_drain,
                monotonic=lambda: 1.0,
                timeout_seconds=2.0,
            )
        assert raised.value is cancellation
    else:
        assert await after_recording_webhook_commit(
            event,
            effect,
            telnyx=api,
            writer=StubWriter(),
            local_drain=local_drain,
            monotonic=lambda: 1.0,
            timeout_seconds=2.0,
        ) == WebhookDisposition(500)
    assert events == ["local_drain", "hangup"]


@pytest.mark.asyncio
async def test_required_error_drain_timeout_reserves_and_joins_provider_hangup() -> None:
    event = recording_event(
        "call.recording.error",
        provider_recording_id=None,
        required=True,
        started_at=None,
        ended_at=None,
        channels=None,
    )
    effect = resolve_recording_webhook(event)
    drain_started = asyncio.Event()
    drain_cancelled = asyncio.Event()
    hangup_started = asyncio.Event()
    hangup_release = asyncio.Event()
    events: list[str] = []

    async def blocked_drain(_call_id: UUID, _reason: str) -> None:
        drain_started.set()
        try:
            await asyncio.Future()
        finally:
            drain_cancelled.set()

    async def blocked_hangup() -> CallControlResult:
        hangup_started.set()
        await hangup_release.wait()
        return CallControlResult("accepted")

    api = TerminationApi(blocked_hangup, events)
    pending = asyncio.create_task(
        after_recording_webhook_commit(
            event,
            effect,
            telnyx=api,
            writer=StubWriter(),
            local_drain=blocked_drain,
            monotonic=time.monotonic,
            timeout_seconds=1.15,
        )
    )
    try:
        await drain_started.wait()
        await asyncio.wait_for(hangup_started.wait(), timeout=0.25)
        assert drain_cancelled.is_set()
        assert pending.done() is False
        hangup_release.set()
        assert await pending == WebhookDisposition(500)
        assert events == ["hangup"]
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_required_error_cancellation_during_hangup_joins_attempt_before_propagating() -> None:
    event = recording_event(
        "call.recording.error",
        provider_recording_id=None,
        required=True,
        started_at=None,
        ended_at=None,
        channels=None,
    )
    effect = resolve_recording_webhook(event)
    entered = asyncio.Event()
    release = asyncio.Event()
    events: list[str] = []

    async def blocked_hangup() -> CallControlResult:
        entered.set()
        await release.wait()
        return CallControlResult("accepted")

    api = TerminationApi(blocked_hangup, events)
    pending = asyncio.create_task(
        after_recording_webhook_commit(
            event,
            effect,
            telnyx=api,
            writer=StubWriter(),
            local_drain=lambda _call_id, _reason: asyncio.sleep(0),
            monotonic=lambda: 1.0,
            timeout_seconds=2.0,
        )
    )
    await entered.wait()
    pending.cancel("caller-cancelled")
    await asyncio.sleep(0)

    assert pending.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert events == ["hangup"]


class CleanupApi:
    def __init__(
        self,
        *,
        stop_result: object = CallControlResult("accepted"),
        hangup_result: object = CallControlResult("accepted"),
    ) -> None:
        self.stop_result = stop_result
        self.hangup_result = hangup_result
        self.events: list[str] = []
        self.stop_calls: list[tuple[str, object, UUID]] = []
        self.hangup_calls: list[tuple[str, UUID, SecretStr | None]] = []

    async def stop_recording(
        self,
        call_control_id: str,
        request: object,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        self.events.append("stop")
        self.stop_calls.append((call_control_id, request, command_id))
        if isinstance(self.stop_result, BaseException):
            raise self.stop_result
        if callable(self.stop_result):
            return await self.stop_result()
        return self.stop_result  # type: ignore[return-value]

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult:
        self.events.append("hangup")
        self.hangup_calls.append((call_control_id, command_id, client_state))
        if isinstance(self.hangup_result, BaseException):
            raise self.hangup_result
        if callable(self.hangup_result):
            return await self.hangup_result()
        return self.hangup_result  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_recording_off_cleanup_skips_stop_but_always_hangs_up_with_capsule() -> None:
    writer = StubWriter()
    api = CleanupApi()
    subject = boundary(api, writer)  # type: ignore[arg-type]

    await subject.cleanup(
        identity(),
        recording_may_be_active=False,
        reason="recording_off",
    )

    assert api.events == ["hangup"]
    assert api.stop_calls == []
    assert writer.commands == []
    correlation = build_recording_correlation(
        identity(), retention_days=30, required=False
    )
    assert api.hangup_calls[0][:2] == (
        CALL_CONTROL_ID,
        derive_recording_action_id(correlation.recording_id, "hangup"),
    )
    assert decode_recording_correlation(api.hangup_calls[0][2]) == correlation  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_may_active_cleanup_stops_then_hangs_up_without_fabricating_state() -> None:
    writer = StubWriter()
    api = CleanupApi()
    subject = boundary(api, writer)  # type: ignore[arg-type]

    await subject.cleanup(
        identity(),
        recording_may_be_active=True,
        reason="normal_shutdown",
    )

    assert api.events == ["stop", "hangup"]
    correlation = build_recording_correlation(
        identity(), retention_days=30, required=False
    )
    stop_request = api.stop_calls[0][1]
    assert stop_request.recording_id is None
    assert decode_recording_correlation(stop_request.client_state) == correlation
    assert api.stop_calls[0][2] == derive_recording_action_id(
        correlation.recording_id, "recording-stop"
    )
    assert api.hangup_calls[0][1] == derive_recording_action_id(
        correlation.recording_id, "hangup"
    )
    assert writer.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["rejected", "exception", "cancellation"])
async def test_stop_fault_never_suppresses_hangup_and_preserves_first_cancellation(
    fault: str,
) -> None:
    cancellation = asyncio.CancelledError("stop-cancelled")
    stop_result: object
    if fault == "rejected":
        stop_result = CallControlResult("rejected")
    elif fault == "exception":
        stop_result = RuntimeError("RAW-STOP-SENTINEL")
    else:
        stop_result = cancellation
    api = CleanupApi(stop_result=stop_result)
    subject = boundary(api, StubWriter())  # type: ignore[arg-type]

    if fault == "cancellation":
        with pytest.raises(asyncio.CancelledError) as raised:
            await subject.cleanup(
                identity(), recording_may_be_active=True, reason="shutdown"
            )
        assert raised.value is cancellation
    else:
        with pytest.raises(
            RecordingLifecycleError, match="recording_cleanup_failed"
        ) as raised:
            await subject.cleanup(
                identity(), recording_may_be_active=True, reason="shutdown"
            )
        assert "RAW-STOP-SENTINEL" not in repr(raised.value)
    assert api.events == ["stop", "hangup"]


@pytest.mark.asyncio
async def test_caller_cancellation_during_hangup_waits_for_owned_attempt_then_propagates() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_hangup() -> CallControlResult:
        entered.set()
        await release.wait()
        return CallControlResult("accepted")

    api = CleanupApi(hangup_result=blocked_hangup)
    subject = boundary(api, StubWriter())  # type: ignore[arg-type]
    pending = asyncio.create_task(
        subject.cleanup(
            identity(),
            recording_may_be_active=False,
            reason="caller_cancelled",
        )
    )
    await entered.wait()
    owned_hangups = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "voice-recording-hangup"
    ]
    assert len(owned_hangups) == 1
    assert owned_hangups[0].done() is False
    pending.cancel("caller-cancelled")
    await asyncio.sleep(0)

    assert pending.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert owned_hangups[0].done() is True
    assert api.events == ["hangup"]
    assert not any(
        task is not asyncio.current_task() and "hangup" in task.get_name()
        for task in asyncio.all_tasks()
    )


class DeleteApi:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, float]] = []

    async def delete_recording(
        self, recording_id: str, *, timeout_seconds: float
    ) -> ProviderDeleteResultV1:
        self.calls.append((recording_id, timeout_seconds))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]


class PurgeSink:
    def __init__(
        self,
        leases: list[RecordingPurgeLease],
        *,
        ack_results: list[BaseException | None] | None = None,
    ) -> None:
        self.leases = leases
        self.ack_results = list(ack_results or [])
        self.lease_calls: list[tuple[str, int, int]] = []
        self.ack_calls: list[tuple[UUID, UUID, str, datetime]] = []

    async def lease_recording_purges(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> list[RecordingPurgeLease]:
        self.lease_calls.append((worker_id, lease_seconds, batch_size))
        return self.leases

    async def ack_recording_purge(
        self,
        recording_id: UUID,
        lease_token: UUID,
        outcome: str,
        occurred_at: datetime,
    ) -> None:
        self.ack_calls.append((recording_id, lease_token, outcome, occurred_at))
        if self.ack_results:
            error = self.ack_results.pop(0)
            if error is not None:
                raise error


def purge_lease(number: int, **updates: object) -> RecordingPurgeLease:
    values: dict[str, object] = {
        "schema_version": 1,
        "recording_id": UUID(int=number),
        "lease_token": UUID(int=100 + number),
        "telnyx_recording_id": f"recording_{number}",
        "purge_attempt": 1,
        "lease_expires_at": NOW + timedelta(seconds=30),
    }
    values.update(updates)
    return RecordingPurgeLease(**values)  # type: ignore[arg-type]


def clock(*values: datetime):
    remaining = iter(values)
    return lambda: next(remaining)


@pytest.mark.asyncio
async def test_purge_maps_four_provider_outcomes_and_acks_sequentially_with_fresh_clocks() -> None:
    leases = [purge_lease(number) for number in range(1, 5)]
    sink = PurgeSink(leases)
    api = DeleteApi(
        [
            ProviderDeleteResultV1("deleted", "recording_1"),
            ProviderDeleteResultV1("not_found", None),
            ProviderDeleteResultV1("retry", None),
            ProviderDeleteResultV1("failed", None),
        ]
    )
    times = [NOW + timedelta(milliseconds=index) for index in range(8)]

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=4,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(*times),
    )

    assert result == PurgeBatchResult(
        leased=4,
        deleted=1,
        not_found=1,
        retry=1,
        failed=1,
        stale=0,
        expired=0,
    )
    assert api.calls == [(f"recording_{number}", 1.0) for number in range(1, 5)]
    assert [call[2] for call in sink.ack_calls] == [
        "deleted",
        "not_found",
        "retry",
        "failed",
    ]
    assert [call[3] for call in sink.ack_calls] == times[1::2]


@pytest.mark.asyncio
async def test_purge_expiry_before_delete_and_after_delete_never_acks_uncertain_state() -> None:
    before = purge_lease(1, lease_expires_at=NOW + timedelta(seconds=2))
    after = purge_lease(2, lease_expires_at=NOW + timedelta(seconds=30))
    sink = PurgeSink([before, after])
    api = DeleteApi([ProviderDeleteResultV1("deleted", "recording_2")])

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=2,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW, NOW + timedelta(seconds=29)),
    )

    assert result.expired == 2
    assert result.deleted == 0
    assert api.calls == [("recording_2", 1.0)]
    assert sink.ack_calls == []


@pytest.mark.asyncio
async def test_purge_does_not_delete_when_lease_cannot_cover_real_sink_ack_bound() -> None:
    lease = purge_lease(1, lease_expires_at=NOW + timedelta(seconds=8))
    sink = PurgeSink([lease])
    api = DeleteApi([ProviderDeleteResultV1("deleted", "recording_1")])

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=1,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW),
    )

    assert result.expired == 1
    assert api.calls == []
    assert sink.ack_calls == []


@pytest.mark.asyncio
async def test_purge_ack_timeout_stops_batch_without_same_token_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.telnyx.recordings as recordings_module

    entered = asyncio.Event()

    class BlockingAckSink(PurgeSink):
        async def ack_recording_purge(
            self,
            recording_id: UUID,
            lease_token: UUID,
            outcome: str,
            occurred_at: datetime,
        ) -> None:
            self.ack_calls.append((recording_id, lease_token, outcome, occurred_at))
            entered.set()
            await asyncio.Future()

    monkeypatch.setattr(recordings_module, "PURGE_ACK_RESERVE_SECONDS", 0.01)
    sink = BlockingAckSink([purge_lease(1), purge_lease(2)])
    api = DeleteApi([ProviderDeleteResultV1("deleted", "recording_1")])

    with pytest.raises(RecordingPurgeError, match="recording_purge_ack_failed"):
        await asyncio.wait_for(
            purge_recordings_once(
                worker_id="worker-1",
                lease_seconds=30,
                batch_size=2,
                telnyx=api,
                sink=sink,  # type: ignore[arg-type]
                utcnow=clock(NOW, NOW),
            ),
            timeout=0.1,
        )

    assert entered.is_set()
    assert len(api.calls) == 1
    assert len(sink.ack_calls) == 1


@pytest.mark.asyncio
async def test_purge_ack_cancellation_propagates_without_replay() -> None:
    cancellation = asyncio.CancelledError("ack-cancelled")
    sink = PurgeSink(
        [purge_lease(1), purge_lease(2)],
        ack_results=[cancellation],
    )
    api = DeleteApi([ProviderDeleteResultV1("deleted", "recording_1")])

    with pytest.raises(asyncio.CancelledError) as raised:
        await purge_recordings_once(
            worker_id="worker-1",
            lease_seconds=30,
            batch_size=2,
            telnyx=api,
            sink=sink,  # type: ignore[arg-type]
            utcnow=clock(NOW, NOW),
        )

    assert raised.value is cancellation
    assert len(api.calls) == 1
    assert len(sink.ack_calls) == 1


@pytest.mark.asyncio
async def test_purge_rejects_malformed_provider_id_without_provider_io_and_acks_failed() -> None:
    malformed = purge_lease(1, telnyx_recording_id="https://RAW-URL-SENTINEL")
    sink = PurgeSink([malformed])
    api = DeleteApi([])

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=1,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW + timedelta(milliseconds=1)),
    )

    assert result.failed == 1
    assert api.calls == []
    assert sink.ack_calls[0][2] == "failed"
    assert "RAW-URL-SENTINEL" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", [".", ".."])
async def test_purge_dot_segments_are_local_failed_without_provider_dispatch(
    provider_id: str,
) -> None:
    lease = purge_lease(1, telnyx_recording_id=provider_id)
    sink = PurgeSink([lease])
    api = DeleteApi([])

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=1,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW + timedelta(milliseconds=1)),
    )

    assert result.failed == 1
    assert result.retry == 0
    assert api.calls == []
    assert sink.ack_calls[0][2] == "failed"


@pytest.mark.asyncio
async def test_purge_accepts_256_url_safe_opaque_provider_id() -> None:
    provider_id = "r." + "A" * 251 + "~_-"
    lease = purge_lease(1, telnyx_recording_id=provider_id)
    sink = PurgeSink([lease])
    api = DeleteApi([ProviderDeleteResultV1("deleted", provider_id)])

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=1,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW + timedelta(milliseconds=1)),
    )

    assert result.deleted == 1
    assert api.calls == [(provider_id, 1.0)]
    assert sink.ack_calls[0][2] == "deleted"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ack_error",
    [
        OperationSinkTransientError("postgres_unavailable"),
        OperationSinkCommitAmbiguousError("postgres_commit_ambiguous"),
        OperationSinkPermanentError("purge_contract_failure"),
        OperationSinkContractError("purge_ack_result_invalid"),
        OperationConflictError("operation_hash_conflict"),
    ],
)
async def test_uncertain_or_permanent_ack_failure_stops_batch_without_replay(
    ack_error: BaseException,
) -> None:
    sink = PurgeSink([purge_lease(1), purge_lease(2)], ack_results=[ack_error])
    api = DeleteApi([ProviderDeleteResultV1("deleted", "recording_1")])

    with pytest.raises(RecordingPurgeError, match="recording_purge_ack_failed") as raised:
        await purge_recordings_once(
            worker_id="worker-1",
            lease_seconds=30,
            batch_size=2,
            telnyx=api,
            sink=sink,  # type: ignore[arg-type]
            utcnow=clock(NOW, NOW),
        )

    assert len(api.calls) == 1
    assert len(sink.ack_calls) == 1
    assert str(ack_error) not in repr(raised.value)


@pytest.mark.asyncio
async def test_stale_ack_is_counted_and_batch_continues_without_same_token_replay() -> None:
    sink = PurgeSink(
        [purge_lease(1), purge_lease(2)],
        ack_results=[OperationSinkStaleLeaseError("purge_stale_lease"), None],
    )
    api = DeleteApi(
        [
            ProviderDeleteResultV1("deleted", "recording_1"),
            ProviderDeleteResultV1("not_found", None),
        ]
    )

    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=2,
        telnyx=api,
        sink=sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW, NOW, NOW),
    )

    assert result.stale == 1
    assert result.deleted == 0
    assert result.not_found == 1
    assert len(sink.ack_calls) == 2
    assert len({call[1] for call in sink.ack_calls}) == 2


@pytest.mark.asyncio
async def test_delete_cancellation_propagates_without_ack() -> None:
    cancellation = asyncio.CancelledError("delete-cancelled")
    sink = PurgeSink([purge_lease(1)])
    api = DeleteApi([cancellation])

    with pytest.raises(asyncio.CancelledError) as raised:
        await purge_recordings_once(
            worker_id="worker-1",
            lease_seconds=30,
            batch_size=1,
            telnyx=api,
            sink=sink,  # type: ignore[arg-type]
            utcnow=clock(NOW),
        )

    assert raised.value is cancellation
    assert sink.ack_calls == []


@pytest.mark.asyncio
async def test_delete_success_crash_before_ack_then_new_lease_404_closes_as_not_found() -> None:
    first_sink = PurgeSink(
        [purge_lease(1)],
        ack_results=[OperationSinkCommitAmbiguousError("postgres_commit_ambiguous")],
    )
    first_api = DeleteApi([ProviderDeleteResultV1("deleted", "recording_1")])
    with pytest.raises(RecordingPurgeError):
        await purge_recordings_once(
            worker_id="worker-1",
            lease_seconds=30,
            batch_size=1,
            telnyx=first_api,
            sink=first_sink,  # type: ignore[arg-type]
            utcnow=clock(NOW, NOW),
        )

    second_sink = PurgeSink(
        [purge_lease(1, lease_token=UUID(int=999), purge_attempt=2)]
    )
    second_api = DeleteApi([ProviderDeleteResultV1("not_found", None)])
    result = await purge_recordings_once(
        worker_id="worker-1",
        lease_seconds=30,
        batch_size=1,
        telnyx=second_api,
        sink=second_sink,  # type: ignore[arg-type]
        utcnow=clock(NOW, NOW),
    )

    assert result.not_found == 1
    assert second_sink.ack_calls[0][2] == "not_found"
