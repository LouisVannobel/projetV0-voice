"""Full signed event precision at native V2 registry/SQLite admission, offline only."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from projetv0_voice.admission import CallAdmissionRejected
from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    canonical_candidate_profile_sha256,
)
from projetv0_voice.telnyx.webhooks import TelnyxWebhookVerifier
from tests.integration.test_fixed_v2_process import registry_case
from tests.unit.test_qualified_profile import candidate_data
from tests.unit.test_sparra_admission import DID, policy
from tests.unit.test_telnyx_webhooks import event_body, signing_fixture


def signed_event(instant, kind="call.initiated", event_id=None):
    body = event_body(event_id=event_id or uuid4().hex, event_type=kind,
        occurred_at=instant.isoformat(), payload={
            "call_control_id": "original", "call_leg_id": "original-leg",
            "call_session_id": "session", "connection_id": policy().connection_id,
            "to": DID, "from": None,
        })
    public_key, headers = signing_fixture(body, timestamp=int(time.time()))
    return TelnyxWebhookVerifier(public_key=public_key).verify(body=body, headers=headers)


def rows(path):
    with closing(sqlite3.connect(path)) as db:
        return (
            tuple(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("call_leases", "webhook_receipts", "outbox", "local_audio_pin")),
            db.execute("SELECT profile_sha256,total_calls,used_calls FROM qualification_runs")
                .fetchall(),
        )


async def commit(case, event, profile, *, duplicate=False):
    resolution = await (case.registry.resolve_duplicate_webhook(event) if duplicate
                        else case.registry.resolve_webhook(event))
    effect = resolution.effect
    ticket = case.writer.submit_webhook(
        receipt={"event_id": event.event_id, "event_type": event.event_type,
            "call_control_id": event.call_control_id, "occurred_at": event.occurred_at,
            "received_at": datetime.now(UTC),
            "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256},
        lease=None if effect is None else effect.lease,
        operation=None if effect is None else effect.operation,
        admission_facts=None if effect is None else effect.admission_facts,
        operation_generation=None if effect is None else effect.operation_generation,
        qualification_run_id=profile.run_id if event.event_type == "call.initiated" else None,
        qualification_profile_sha256=bytes.fromhex(canonical_candidate_profile_sha256(profile)),
    )
    try:
        result = await asyncio.wait_for(ticket.wait(), 2)
    except TimeoutError:
        fault = case.writer.fatal_fault
        pytest.fail("native_receipt_timeout:" + ("none" if fault is None else fault.code))
    return await case.registry.reconcile_after_commit(event, resolution, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("offset_us", [-1, 0, 1])
async def test_new_admission_uses_full_verified_floor_precision(tmp_path, offset_us):
    # All three values share one canonical routing millisecond. Truncating the
    # floor or event would admit the -1 us case or reject equality incorrectly.
    floor = (datetime.now(UTC) - timedelta(seconds=1)).replace(microsecond=123456)
    profile = QualificationCandidateProfileV1.model_validate({
        **candidate_data(datetime.now(UTC)), "admission_not_before": floor,
    })
    verified = signed_event(floor + timedelta(microseconds=offset_us))
    async with registry_case(tmp_path, utcnow=lambda: datetime.now(UTC),
        admission_not_before=profile.admission_not_before, candidate_run_id=profile.run_id,
        monotonic=time.monotonic) as case:
        before = rows(case.path)
        if offset_us < 0:
            with pytest.raises(CallAdmissionRejected, match="^call_event_invalid$"):
                await case.registry.resolve_webhook(verified)
            assert rows(case.path) == before == ((0, 0, 0, 0), [])
            assert await case.registry.snapshot("original") is None
            assert not case.begins and not case.provider.actions
        else:
            assert (await commit(case, verified, profile)).status_code == 200
            admitted = await case.registry.snapshot("original")
            assert admitted is not None and admitted.durable
            after = rows(case.path)
            assert after == ((1, 1, 0, 1), [(
                bytes.fromhex(canonical_candidate_profile_sha256(profile)), 1, 1)])
            assert len(case.begins) == 1 and len(case.provider.actions) == 1
            assert (await commit(case, verified, profile, duplicate=True)).status_code == 200
            assert rows(case.path) == after and len(case.begins) == 1


@pytest.mark.asyncio
async def test_new_admission_rejects_future_epoch_without_spending(tmp_path):
    floor = datetime.now(UTC) + timedelta(seconds=10)
    async with registry_case(tmp_path, utcnow=lambda: datetime.now(UTC),
        admission_not_before=floor, monotonic=time.monotonic) as case:
        with pytest.raises(CallAdmissionRejected, match="^call_event_invalid$"):
            await case.registry.resolve_webhook(signed_event(floor))
        assert rows(case.path) == ((0, 0, 0, 0), [])
        assert not case.begins and not case.provider.actions


@pytest.mark.asyncio
async def test_old_initiation_cannot_admit_through_answered_placeholder(tmp_path):
    floor = datetime.now(UTC) - timedelta(seconds=1)
    profile = QualificationCandidateProfileV1.model_validate({
        **candidate_data(datetime.now(UTC)), "admission_not_before": floor,
    })
    async with registry_case(tmp_path, utcnow=lambda: datetime.now(UTC),
        admission_not_before=floor, candidate_run_id=profile.run_id,
        monotonic=time.monotonic) as case:
        answered = signed_event(floor - timedelta(microseconds=1), "call.answered")
        assert (await commit(case, answered, profile)).status_code == 200
        before = rows(case.path)
        assert before == ((0, 1, 0, 0), [])
        with pytest.raises(CallAdmissionRejected, match="^call_event_invalid$"):
            await case.registry.resolve_webhook(signed_event(floor - timedelta(microseconds=1)))
        assert rows(case.path) == before
        assert await case.registry.snapshot("original") is None
        assert not case.begins and not case.provider.actions


@pytest.mark.asyncio
async def test_saved_epoch_rejects_same_old_body_after_registry_restart(tmp_path):
    floor = datetime.now(UTC) - timedelta(seconds=1)
    profile = QualificationCandidateProfileV1.model_validate({
        **candidate_data(datetime.now(UTC)), "admission_not_before": floor,
    })
    saved = profile.model_dump_json()
    fingerprint = canonical_candidate_profile_sha256(profile)
    old_event = signed_event(floor - timedelta(microseconds=1))
    for _ in range(2):
        restored = QualificationCandidateProfileV1.model_validate_json(saved)
        assert restored.admission_not_before == floor
        assert canonical_candidate_profile_sha256(restored) == fingerprint
        async with registry_case(tmp_path, utcnow=lambda: datetime.now(UTC),
            admission_not_before=restored.admission_not_before, candidate_run_id=restored.run_id,
            monotonic=time.monotonic) as case:
            with pytest.raises(CallAdmissionRejected, match="^call_event_invalid$"):
                await case.registry.resolve_webhook(old_event)
            assert rows(case.path) == ((0, 0, 0, 0), [])
            assert not case.begins and not case.provider.actions
