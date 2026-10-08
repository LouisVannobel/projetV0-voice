"""Native identity constructor only; metadata does not claim Begin/SDK/carrier qualification."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from projetv0_voice.admission import CallGenerationHandle, ProcessLeaseClaim
from projetv0_voice.audio_contract import BeginCallSnapshotV2
from projetv0_voice.models import RoutingV1
from projetv0_voice.session import CallIdentity


def identity_values():
    admitted = datetime(2026, 10, 7, 12, tzinfo=UTC)
    media_started = admitted + timedelta(seconds=2)
    call, workspace, recording, generation = (UUID(int=value) for value in (1, 2, 3, 4))
    routing = RoutingV1(
        schema_version=1,
        direction="incoming",
        connection_id="clock-fixture",
        to_e164="+33102030405",
        from_e164=None,
        telnyx_call_control_id="clock-control",
        telnyx_call_leg_id="clock-leg",
        telnyx_call_session_id="clock-session",
        admitted_at=admitted,
    )
    snapshot = BeginCallSnapshotV2.model_validate(
        {
            "schema_version": 2,
            "workspace_id": workspace,
            "call_id": call,
            "configuration_revision": 1,
            "knowledge": {
                "business_name": "Clock fixture",
                "sector": "garage",
                "opening_hours": "",
                "services": "",
                "prices": "",
                "faq": "",
                "instructions": "",
            },
            "transfer_destination": None,
            "retention_until": admitted + timedelta(days=30),
            "recording_policy": "local_30d",
            "recording_contact_phone": "+33102030405",
            "audio_available": True,
            "recording_id": recording,
        }
    )
    return {
        "call_id": call,
        "generation": CallGenerationHandle("clock-control", generation),
        "lease_claim": ProcessLeaseClaim(
            call_control_id="clock-control",
            call_id=call,
            generation=generation,
            token_digest=b"t" * 32,
            claimed_at=media_started,
        ),
        "deployment_id": "clock-deployment",
        "telnyx_call_control_id": "clock-control",
        "telnyx_call_leg_id": "clock-leg",
        "telnyx_call_session_id": "clock-session",
        "stream_id": "clock-stream",
        "started_at": media_started,
        "retention_until": admitted + timedelta(days=30),
        "routing": routing,
        "begin_snapshot": snapshot,
    }


def test_native_identity_preserves_original_admission_and_later_media_start():
    values = identity_values()
    identity = CallIdentity(**values)
    assert identity.started_at == identity.lease_claim.claimed_at
    assert identity.routing.admitted_at + timedelta(seconds=2) == identity.started_at
    assert identity.retention_until == identity.routing.admitted_at + timedelta(days=30)


def skewed_values(seconds):
    values = identity_values()
    admitted = values["started_at"] + timedelta(seconds=seconds)
    retention = admitted + timedelta(days=30)
    values["routing"] = RoutingV1.model_validate(
        {
            **values["routing"].model_dump(mode="python"),
            "admitted_at": admitted,
        }
    )
    values["begin_snapshot"] = BeginCallSnapshotV2.model_validate(
        {
            **values["begin_snapshot"].model_dump(mode="python"),
            "retention_until": retention,
        }
    )
    values["retention_until"] = retention
    return values


@pytest.mark.parametrize("seconds", [1, 30])
def test_native_identity_uses_existing_provider_clock_skew_margin(seconds):
    identity = CallIdentity(**skewed_values(seconds))
    assert identity.routing.admitted_at == identity.started_at + timedelta(seconds=seconds)
    assert identity.retention_until == identity.routing.admitted_at + timedelta(days=30)


def test_native_identity_refuses_future_admission_outside_existing_margin():
    # All original-retention/snapshot/claim fields match. Only the existing
    # 30-second provider clock allowance is exceeded, so this is not vacuous.
    with pytest.raises(ValueError, match="call_identity_mismatch"):
        CallIdentity(**skewed_values(31))


@pytest.mark.parametrize(
    "changes",
    [
        {"call_id": UUID(int=9)},
        {"generation": CallGenerationHandle("clock-control", UUID(int=9))},
        {"telnyx_call_control_id": "foreign-control"},
        {"telnyx_call_leg_id": "foreign-leg"},
        {"telnyx_call_session_id": "foreign-session"},
        {"retention_until": datetime(2026, 11, 6, 12, 0, 0, 1000, tzinfo=UTC)},
        {"started_at": datetime(2026, 10, 7, 12, 0, 3, tzinfo=UTC)},
    ],
)
def test_native_identity_distinct_clocks_keep_foreign_and_retention_guards(changes):
    with pytest.raises(ValueError):
        CallIdentity(**{**identity_values(), **changes})
