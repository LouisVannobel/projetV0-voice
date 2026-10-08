"""Native registry/SQLite clock regression; provider fixture does not prove carrier I/O."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from projetv0_voice.admission import CallAdmissionRejected, CallRegistry
from projetv0_voice.audio_contract import BeginCallSnapshotV2, VoiceOperationV2
from projetv0_voice.config import SparraManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, DisclosureEvidenceV1
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.telnyx.webhooks import TelnyxWebhookVerifier
from tests.unit.test_sparra_admission import DID, ControlledProvider, policy
from tests.unit.test_telnyx_webhooks import event_body, signing_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_original", [False, True], ids=["distinct-native-clocks", "forged-original-date"]
)
@pytest.mark.parametrize("microsecond", [123000, 123456, 123999, 456])
@pytest.mark.parametrize("recording_policy", ["off", "local_30d"])
async def test_v2_original_event_time_survives_later_native_lease_creation(
    tmp_path, forged_original, microsecond, recording_policy
):
    occurred = datetime(2026, 10, 7, 12, 0, 0, microsecond, tzinfo=UTC)
    admitted = occurred.replace(microsecond=microsecond // 1000 * 1000)
    received = occurred + timedelta(milliseconds=10)
    database = tmp_path / "event-clock.sqlite"
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: bytes(range(32))}, active_version=1),
        contract_version=2,
        process_agent_id="native-agent",
        process_deployment_id="native-deployment",
        utcnow=lambda: received,
    )
    owner = asyncio.create_task(writer.run())
    registry = resolved = webhook = result = None
    replayed = []
    primary = None
    run_id = UUID(int=99)

    def signed_event(event_id, instant):
        body = event_body(event_id=event_id, occurred_at=instant.isoformat(), payload={
            "call_control_id": "original", "call_leg_id": "original-leg",
            "call_session_id": "session", "connection_id": "fixture-connection",
            "to": DID, "from": None,
        })
        public_key, headers = signing_fixture(body, timestamp=int(time.time()))
        verified = TelnyxWebhookVerifier(public_key=public_key).verify(body=body, headers=headers)
        assert verified.occurred_at == instant
        return verified

    def submit(verified, selected_effect):
        return writer.submit_webhook(
            receipt={"event_id": verified.event_id, "event_type": verified.event_type,
                "call_control_id": verified.call_control_id,
                "occurred_at": verified.occurred_at, "received_at": received,
                "semantic_fingerprint_sha256": verified.semantic_fingerprint_sha256},
            lease=selected_effect.lease, operation=selected_effect.operation,
            admission_facts=selected_effect.admission_facts,
            operation_generation=selected_effect.operation_generation,
            qualification_run_id=run_id,
            qualification_total_calls=3,
            qualification_profile_sha256=b"p" * 32,
        )

    async def unconsumed_begin(_deployment, _call, _routing):
        # This component regression ends at actual acknowledged pending COMMIT.
        # Real Psycopg BeginV2 belongs to the separate joined native fixture.
        raise AssertionError("component test unexpectedly invoked remote Begin")

    try:
        assert await writer.wait_ready()
        await writer.assert_sparra_compatible()
        selected = SparraManifestV1.model_validate(
            {**policy().model_dump(), "operation_contract_version": 2}
        )
        registry = CallRegistry(
            writer=writer,
            call_control=ControlledProvider(),
            tenant_id=str(UUID(int=22)),
            agent_id="native-agent",
            deployment_id="native-deployment",
            capacity=1,
            lease_ttl_seconds=30,
            stream_url="wss://fixture.invalid/media",
            retention_days=30,
            utcnow=lambda: received,
            monotonic=lambda: 100.0,
            sparra=selected,
            called_did=DID,
            begin_call=unconsumed_begin,
        )
        webhook = signed_event("event-original", occurred)
        resolved = await registry.resolve_webhook(webhook)
        effect = resolved.effect
        assert (
            effect is not None and effect.lease is not None and effect.admission_facts is not None
        )
        assert effect.lease["created_at"] == received
        assert effect.admission_facts.admitted_at == admitted
        assert effect.admission_facts.retention_until == admitted + timedelta(days=30)
        assert effect.admission_facts.admission_generation is not None
        admission = (
            replace(
                effect.admission_facts,
                admitted_at=received.replace(microsecond=received.microsecond // 1000 * 1000),
                retention_until=received.replace(
                    microsecond=received.microsecond // 1000 * 1000
                ) + timedelta(days=30),
            )
            if forged_original
            else effect.admission_facts
        )
        ticket = writer.submit_webhook(
            receipt={
                "event_id": webhook.event_id,
                "event_type": webhook.event_type,
                "call_control_id": webhook.call_control_id,
                "occurred_at": webhook.occurred_at,
                "received_at": received,
                "semantic_fingerprint_sha256": webhook.semantic_fingerprint_sha256,
            },
            lease=effect.lease,
            operation=effect.operation,
            admission_facts=admission,
            operation_generation=effect.operation_generation,
            qualification_run_id=run_id,
            qualification_total_calls=3,
            qualification_profile_sha256=b"p" * 32,
        )
        committed = asyncio.create_task(ticket.wait())
        failed = asyncio.create_task(writer.fatal_event.wait())
        try:
            await asyncio.wait({committed, failed}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            fault = writer.fatal_fault
            if fault is not None:
                print("NATIVE_PENDING_COMMIT_FAULT " + fault.code)
            if forged_original:
                assert fault is not None and fault.code == "local_admission_identity_conflict"
                return
            assert fault is None, "native_pending_commit_fault:" + fault.code
            assert committed.done(), "native_pending_commit_ack_deadline"
            result = await committed
            assert result.receipt == "first"
        finally:
            for pending in (committed, failed):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(committed, failed, return_exceptions=True)
        facts = await writer.read_call_lifecycle(effect.admission_facts.call_id)
        assert facts is not None
        assert facts.admitted_at == admitted and facts.retention_until == admitted + timedelta(
            days=30
        )
        with sqlite3.connect(database) as connection:
            stored = connection.execute(
                "SELECT tenant_id,agent_id,created_at FROM call_leases WHERE call_id=?",
                (str(facts.call_id),),
            ).fetchone()
        assert stored is not None and stored[:2] == (str(UUID(int=22)), "native-agent")
        assert datetime.fromisoformat(stored[2].replace("Z", "+00:00")) == received
        assert not writer.is_degraded
        # Receipt identity remains raw even when business admission shares one millisecond.
        replay_instant = admitted + timedelta(microseconds=789)
        same_id_changed_time = signed_event(webhook.event_id, replay_instant)
        assert same_id_changed_time.semantic_fingerprint_sha256 != (
            webhook.semantic_fingerprint_sha256
        )
        assert await writer.classify_webhook_receipt(
            event_id=webhook.event_id,
            semantic_fingerprint_sha256=same_id_changed_time.semantic_fingerprint_sha256,
        ) == "conflict"
        assert await writer.classify_webhook_receipt(
            event_id=webhook.event_id,
            semantic_fingerprint_sha256=webhook.semantic_fingerprint_sha256,
        ) == "duplicate"
        exact_duplicate = await registry.resolve_duplicate_webhook(webhook)
        duplicate_result = None
        replayed.append((webhook, exact_duplicate, lambda: duplicate_result))
        duplicate_result = await submit(webhook, exact_duplicate.effect).wait()
        assert duplicate_result.receipt == "duplicate"
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM call_leases").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (1,)
            assert connection.execute(
                "SELECT profile_sha256,total_calls,used_calls FROM qualification_runs"
            ).fetchall() == [(b"p" * 32, 3, 1)]

        another_id = signed_event("event-same-millisecond", replay_instant)
        assert await writer.classify_webhook_receipt(
            event_id=another_id.event_id,
            semantic_fingerprint_sha256=another_id.semantic_fingerprint_sha256,
        ) == "missing"
        another_resolved = await registry.resolve_webhook(another_id)
        another_result = None
        replayed.append((another_id, another_resolved, lambda: another_result))
        assert another_resolved.effect.admission_facts.admitted_at == admitted
        another_result = await submit(another_id, another_resolved.effect).wait()
        assert another_result.receipt == "first"
        assert (await submit(another_id, another_resolved.effect).wait()).receipt == "duplicate"
        distinct_millisecond = signed_event(
            "event-distinct-millisecond", admitted + timedelta(milliseconds=1, microseconds=789)
        )
        with pytest.raises(CallAdmissionRejected, match="call_identity_conflict"):
            await registry.resolve_webhook(distinct_millisecond)
        assert not writer.is_degraded
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT event_id,occurred_at,semantic_fingerprint_sha256 "
                "FROM webhook_receipts ORDER BY event_id"
            ).fetchall()
            assert connection.execute("SELECT count(*) FROM call_leases").fetchone() == (1,)
            assert connection.execute(
                "SELECT profile_sha256,total_calls,used_calls FROM qualification_runs"
            ).fetchall() == [(b"p" * 32, 3, 1)]
        assert rows == sorted([
            (webhook.event_id, occurred.isoformat().replace("+00:00", "Z"),
             webhook.semantic_fingerprint_sha256),
            (another_id.event_id, replay_instant.isoformat().replace("+00:00", "Z"),
             another_id.semantic_fingerprint_sha256),
        ])
        generation = effect.admission_facts.admission_generation
        assert generation is not None
        # Native domain models are metadata fixtures for these SQLite guards;
        # no remote Begin reply, playable capture or qualification is asserted.
        pin = BeginCallSnapshotV2.model_validate(
            {
                "schema_version": 2,
                "workspace_id": str(UUID(int=22)),
                "call_id": str(facts.call_id),
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
                "retention_until": facts.retention_until,
                "recording_policy": recording_policy,
                "recording_contact_phone": DID,
                "audio_available": recording_policy == "local_30d",
                "recording_id": str(UUID(int=44)) if recording_policy == "local_30d" else None,
            }
        )
        for changes in (
            {"workspace_id": str(UUID(int=33))},
            {"retention_until": facts.retention_until + timedelta(milliseconds=1)},
        ):
            invalid = BeginCallSnapshotV2.model_validate(
                {**pin.model_dump(mode="python"), **changes}
            )
            with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
                await writer.bind_audio_snapshot(invalid, generation=generation)
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(pin, generation=uuid4())
        await writer.bind_audio_snapshot(pin, generation=generation)
        evidence = DisclosureEvidenceV1(
            schema_version=1,
            started_at=admitted,
            completed_at=received,
            failed_at=None,
            input_gate_opened_at=received,
        )
        gate = VoiceOperationV2(
            schema_version=2,
            operation_id=uuid4(),
            deployment_id="native-deployment",
            call_id=facts.call_id,
            occurred_at=received,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=webhook.call_control_id,
                telnyx_call_leg_id=webhook.call_leg_id,
                telnyx_call_session_id=webhook.call_session_id,
                status="active",
                disclosure_state="completed",
                started_at=admitted,
                ended_at=None,
                end_reason=None,
                retention_until=facts.retention_until,
                disclosure_evidence=evidence,
            ),
        )
        for changes in ({"deployment_id": "foreign-deployment"},):
            invalid = VoiceOperationV2.model_validate({**gate.model_dump(mode="python"), **changes})
            with pytest.raises(PersistenceError, match="control_v2_refused"):
                await writer.publish_control_v2(invalid, generation=generation)
        invalid_retention = VoiceOperationV2.model_validate(
            {
                **gate.model_dump(mode="python"),
                "payload": {
                    **gate.payload.model_dump(mode="python"),
                    "retention_until": facts.retention_until + timedelta(milliseconds=1),
                },
            }
        )
        with pytest.raises(PersistenceError, match="control_v2_refused"):
            await writer.publish_control_v2(invalid_retention, generation=generation)
        await writer.publish_control_v2(gate, generation=generation)
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT event_id,occurred_at,semantic_fingerprint_sha256 "
                "FROM webhook_receipts ORDER BY event_id"
            ).fetchall() == rows
        terminal = VoiceOperationV2.model_validate(
            {
                **gate.model_dump(mode="python"),
                "operation_id": uuid4(),
                "payload": {
                    **gate.payload.model_dump(mode="python"),
                    "status": "closed",
                    "ended_at": received,
                    "end_reason": "closed",
                },
            }
        )
        with pytest.raises(PersistenceError, match="final_call_v2_refused"):
            await writer.freeze_call_publication_v2(
                terminal, None, generation=uuid4(), provider_callback=None
            )
        frozen = await writer.freeze_call_publication_v2(
            terminal, None, generation=generation, provider_callback=None
        )
        assert frozen is not None and frozen.payload.retention_until == facts.retention_until
        assert (await writer.read_call_lifecycle(facts.call_id)).admitted_at == admitted
    except BaseException as error:
        primary = error
    finally:
        failures = []
        try:
            if registry is not None:
                await registry.begin_drain()
                if resolved is not None and resolved.reservation is not None:
                    if result is not None:
                        # Confirm the actual COMMIT through the native callback,
                        # after stopping this component's unconsumed remote leg.
                        await asyncio.wait_for(
                            registry.reconcile_after_commit(webhook, resolved, result), 2
                        )
                    else:
                        # Same exact reservation cleanup as native webhook
                        # finalization when its writer submission fails.
                        await asyncio.wait_for(
                            resolved.reservation.settle_after_submit_failure(), 2
                        )
                for replay_event, replay_resolution, replay_result in replayed:
                    if replay_resolution.reservation is not None:
                        if replay_result() is not None:
                            await asyncio.wait_for(registry.reconcile_after_commit(
                                replay_event, replay_resolution, replay_result()), 2)
                        else:
                            await asyncio.wait_for(
                                replay_resolution.reservation.settle_after_submit_failure(), 2)
                await asyncio.wait_for(registry.close_session_owner_registration(), 2)
                registry.close_registration()
                await asyncio.wait_for(registry.wait_background(), 2)
        except BaseException as error:
            failures.append(error)
        try:
            if writer.is_degraded:
                await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 2)
            else:
                await writer.drain(2)
                await asyncio.wait_for(owner, 2)
        except BaseException as error:
            failures.append(error)
        if primary is not None:
            if failures:
                raise BaseExceptionGroup("Native clock failure and cleanup", [primary, *failures])
            raise primary
        if failures:
            raise BaseExceptionGroup("Native clock cleanup", failures)
