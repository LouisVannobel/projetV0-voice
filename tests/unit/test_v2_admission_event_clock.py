"""Native registry/SQLite clock regression; provider fixture does not prove carrier I/O."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from projetv0_voice.admission import CallRegistry
from projetv0_voice.audio_contract import BeginCallSnapshotV2, VoiceOperationV2
from projetv0_voice.config import SparraManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, DisclosureEvidenceV1
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.writer import PersistenceWriter
from tests.unit.test_sparra_admission import DID, ControlledProvider, event, policy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_original", [False, True], ids=["distinct-native-clocks", "forged-original-date"]
)
async def test_v2_original_event_time_survives_later_native_lease_creation(
    tmp_path, forged_original
):
    admitted = datetime(2026, 10, 7, 12, 0, 0, 123000, tzinfo=UTC)
    received = admitted + timedelta(milliseconds=10)
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
    primary = None

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
        webhook = event(occurred_at=admitted)
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
                admitted_at=received,
                retention_until=received + timedelta(days=30),
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
                "recording_policy": "local_30d",
                "recording_contact_phone": DID,
                "audio_available": True,
                "recording_id": str(UUID(int=44)),
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
