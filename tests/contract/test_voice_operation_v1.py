from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from projetv0_voice.models import (
    CallUpsertPayloadV1,
    RecordingUpsertPayloadV1,
    TurnUpsertPayloadV1,
    VoiceOperationV1,
)

NOW = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)
OCCURRED = NOW + timedelta(minutes=5)
CALL_ID = UUID("22222222-2222-4222-8222-222222222222")
OPERATION_ID = UUID("33333333-3333-4333-8333-333333333333")


def call_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "telnyx_call_control_id": "v3:test-call-control-id",
        "telnyx_call_leg_id": "call-leg-id",
        "telnyx_call_session_id": "call-session-id",
        "status": "closed",
        "disclosure_state": "completed",
        "started_at": NOW.isoformat(),
        "ended_at": (NOW + timedelta(minutes=3)).isoformat(),
        "end_reason": "caller_hangup",
        "retention_until": (NOW + timedelta(days=7)).isoformat(),
    }
    payload.update(updates)
    return payload


def turn_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "turn_id": "44444444-4444-4444-8444-444444444444",
        "turn_no": 1,
        "role": "user",
        "source": "stt_final",
        "crypto_version": 1,
        "key_version": 1,
        "nonce_b64": base64.b64encode(b"n" * 24).decode("ascii"),
        "ciphertext_b64": base64.b64encode(b"ciphertext-and-tag").decode("ascii"),
        "started_at": NOW.isoformat(),
        "ended_at": (NOW + timedelta(seconds=2)).isoformat(),
        "interrupted": False,
    }
    payload.update(updates)
    return payload


def recording_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "recording_id": "55555555-5555-4555-8555-555555555555",
        "status": "saved",
        "telnyx_recording_id": "recording-test-id",
        "channels": "dual",
        "format": "wav",
        "started_at": NOW.isoformat(),
        "ended_at": (NOW + timedelta(minutes=3)).isoformat(),
        "retention_until": (NOW + timedelta(days=7)).isoformat(),
    }
    payload.update(updates)
    return payload


def operation(kind: str, payload: dict[str, object], **updates: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": 1,
        "operation_id": str(OPERATION_ID),
        "deployment_id": "voice-agent-a",
        "call_id": str(CALL_ID),
        "occurred_at": OCCURRED.isoformat(),
        "kind": kind,
        "payload": payload,
    }
    data.update(updates)
    return data


@pytest.mark.parametrize(
    ("kind", "payload", "payload_type"),
    [
        ("call.upsert", call_payload(), CallUpsertPayloadV1),
        ("turn.upsert", turn_payload(), TurnUpsertPayloadV1),
        ("recording.upsert", recording_payload(), RecordingUpsertPayloadV1),
    ],
)
def test_voice_operation_round_trips_to_the_matching_typed_payload(
    kind: str, payload: dict[str, object], payload_type: type[object]
) -> None:
    parsed = VoiceOperationV1.model_validate(operation(kind, payload))

    assert isinstance(parsed.payload, payload_type)
    assert VoiceOperationV1.model_validate_json(parsed.model_dump_json()) == parsed
    with pytest.raises(ValidationError, match="frozen"):
        parsed.deployment_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("call.upsert", turn_payload()),
        ("turn.upsert", recording_payload()),
        ("recording.upsert", call_payload()),
    ],
)
def test_voice_operation_rejects_kind_payload_mismatch(
    kind: str, payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="payload does not match kind"):
        VoiceOperationV1.model_validate(operation(kind, payload))


def test_call_snapshot_rejects_missing_fields_and_invalid_status_timestamps() -> None:
    incomplete = call_payload()
    incomplete.pop("retention_until")
    with pytest.raises(ValidationError, match="retention_until"):
        CallUpsertPayloadV1.model_validate(incomplete)

    with pytest.raises(ValidationError, match="active call"):
        CallUpsertPayloadV1.model_validate(
            call_payload(status="active", ended_at=None, end_reason=None, started_at=None)
        )
    with pytest.raises(ValidationError, match="ended_at"):
        CallUpsertPayloadV1.model_validate(
            call_payload(ended_at=(NOW - timedelta(seconds=1)).isoformat())
        )


def test_pending_and_pre_activation_failed_call_snapshots_allow_null_start() -> None:
    pending = CallUpsertPayloadV1.model_validate(
        call_payload(
            status="pending",
            disclosure_state="pending",
            started_at=None,
            ended_at=None,
            end_reason=None,
        )
    )
    failed = CallUpsertPayloadV1.model_validate(
        call_payload(status="failed", started_at=None, disclosure_state="failed")
    )
    assert pending.started_at is None
    assert failed.started_at is None


def test_recording_requires_internal_id_and_consistent_temporal_state() -> None:
    missing_id = recording_payload()
    missing_id.pop("recording_id")
    with pytest.raises(ValidationError, match="recording_id"):
        RecordingUpsertPayloadV1.model_validate(missing_id)

    with pytest.raises(ValidationError, match="ended_at"):
        RecordingUpsertPayloadV1.model_validate(
            recording_payload(ended_at=(NOW - timedelta(seconds=1)).isoformat())
        )


def test_off_recording_snapshot_accepts_all_nullable_metadata_as_null() -> None:
    parsed = RecordingUpsertPayloadV1.model_validate(
        recording_payload(
            status="off",
            telnyx_recording_id=None,
            channels=None,
            format=None,
            started_at=None,
            ended_at=None,
            retention_until=None,
        )
    )
    assert parsed.started_at is None
    with pytest.raises(ValidationError, match="off recording"):
        RecordingUpsertPayloadV1.model_validate(
            recording_payload(
                status="off",
                telnyx_recording_id=None,
                channels=None,
                format=None,
                started_at=NOW.isoformat(),
                ended_at=None,
                retention_until=None,
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nonce_b64", "not base64!"),
        ("nonce_b64", ""),
        ("ciphertext_b64", "not base64!"),
        ("ciphertext_b64", ""),
    ],
)
def test_turn_rejects_malformed_encrypted_envelope(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        TurnUpsertPayloadV1.model_validate(turn_payload(**{field: value}))


@pytest.mark.parametrize("nonce_length", [12, 24])
def test_turn_envelope_does_not_choose_task3_nonce_algorithm(nonce_length: int) -> None:
    parsed = TurnUpsertPayloadV1.model_validate(
        turn_payload(nonce_b64=base64.b64encode(b"n" * nonce_length).decode("ascii"))
    )
    assert len(base64.b64decode(parsed.nonce_b64)) == nonce_length


def test_turn_rejects_invalid_numbers_role_source_and_time_order() -> None:
    for updates in (
        {"turn_no": 0},
        {"key_version": 0},
        {"role": "system"},
        {"source": "partial"},
        {"ended_at": (NOW - timedelta(seconds=1)).isoformat()},
    ):
        with pytest.raises(ValidationError):
            TurnUpsertPayloadV1.model_validate(turn_payload(**updates))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_operation_contract_does_not_coerce_integer_fields(value: object) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        VoiceOperationV1.model_validate(
            operation("turn.upsert", turn_payload(), schema_version=value)
        )
    with pytest.raises(ValidationError, match="turn_no"):
        TurnUpsertPayloadV1.model_validate(turn_payload(turn_no=value))


def test_turn_contract_does_not_coerce_integer_to_boolean() -> None:
    with pytest.raises(ValidationError, match="interrupted"):
        TurnUpsertPayloadV1.model_validate(turn_payload(interrupted=1))


def test_all_contract_datetimes_reject_naive_values_and_normalize_offsets_to_utc() -> None:
    naive = operation("call.upsert", call_payload(), occurred_at="2026-08-25T10:00:00")
    with pytest.raises(ValidationError, match="timezone-aware"):
        VoiceOperationV1.model_validate(naive)

    offset = operation("turn.upsert", turn_payload(), occurred_at="2026-08-25T12:05:00+02:00")
    parsed = VoiceOperationV1.model_validate(offset)
    assert parsed.occurred_at == OCCURRED
    assert parsed.occurred_at.tzinfo is UTC


def test_contracts_reject_unknown_fields_and_unsupported_schema_versions() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VoiceOperationV1.model_validate(operation("call.upsert", call_payload(), extra="no"))
    with pytest.raises(ValidationError, match="schema_version"):
        VoiceOperationV1.model_validate(
            operation("call.upsert", call_payload(), schema_version=2)
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VoiceOperationV1.model_validate(
            operation("call.upsert", {**call_payload(), "natural_key": "forbidden"})
        )


def test_operation_rejects_impossible_occurrence_time_before_payload() -> None:
    with pytest.raises(ValidationError, match="occurred_at"):
        VoiceOperationV1.model_validate(
            operation(
                "turn.upsert",
                turn_payload(
                    started_at=(OCCURRED + timedelta(minutes=1)).isoformat(),
                    ended_at=(OCCURRED + timedelta(minutes=2)).isoformat(),
                ),
            )
        )
