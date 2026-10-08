"""Verified-event fixtures, native registry/SQLite and real codec; no live bridge proof."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4, uuid5

import pytest
from pydantic import SecretStr

from projetv0_voice.audio_contract import BeginCallSnapshotV2
from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.lifecycle import RuntimeSupervisor
from projetv0_voice.persistence.commands import (
    canonical_operation_bytes,
    decode_operation_v2,
    operation_aad_from_metadata,
)
from projetv0_voice.persistence.relay import OutboxRelay
from projetv0_voice.telnyx.webhooks import ResolvedWebhook
from tests.integration.test_fixed_v2_process import NOW
from tests.integration.test_v2_final_call_freeze import final_case, frozen_bytes
from tests.unit.test_sparra_admission import TARGET, event


@asynccontextmanager
async def phone_case(tmp_path, *, failpoint=None):
    def qualified_pin(pin, _call, _routing):
        return BeginCallSnapshotV2.model_validate({
            **pin.model_dump(mode="python"), "transfer_destination": TARGET,
        })

    async with final_case(tmp_path, failpoint=failpoint, reply=qualified_pin) as case:
        assert callable(getattr(case.writer, "commit_transfer_observation_v2", None)), (
            "missing atomic V2 transfer observation"
        )
        case.supervisor = RuntimeSupervisor(writer=case.writer, registry=case.registry,
            sink=case.sink, call_control=case.provider, sparra_enabled=True, retention_days=30,
            utcnow=lambda: case.clock[0])
        case.requested = asyncio.create_task(case.registry.request_human(case.grant.generation))
        try:
            await asyncio.wait_for(case.provider.transfer_entered.wait(), 2)
            case.transfer = await case.writer.read_call_lifecycle(case.grant.call_id)
            assert case.transfer.transfer_command_id is not None
            yield case
        finally:
            case.provider.transfer_release.set()
            await asyncio.gather(case.requested, return_exceptions=True)
            case.supervisor.webhook_finalizers.close_registration()
            await case.supervisor.webhook_finalizers.join_until_empty(
                asyncio.get_running_loop().time() + 2
            )


def target_event(case, kind="call.initiated", **updates):
    fields = {"occurred_at": case.clock[0], "call_control_id": "target",
        "call_leg_id": "target-leg", "call_session_id": "session", "to_e164": TARGET,
        "client_state": SecretStr(case.transfer.transfer_correlation),
        "direction": "outgoing" if kind == "call.initiated" else None, "call_state": None}
    fields.update(updates)
    return event(kind, **fields)


async def supervised(case, observed):
    classified = await case.supervisor.classify_webhook_receipt(observed)
    if classified == "duplicate":
        resolved = await case.registry.resolve_duplicate_webhook(observed)
    else:
        resolved = await case.registry.resolve_webhook(observed)
    return await case.supervisor.start_webhook_finalization(observed, resolved,
        receipt="duplicate" if classified == "duplicate" else "first").wait()


async def phone_operations(case):
    with sqlite3.connect(case.path) as db:
        rows = db.execute("SELECT op_id,deployment_id,call_id,schema_version,key_version,"
            "nonce,ciphertext FROM outbox WHERE kind='call.upsert' ORDER BY queue_id").fetchall()
    operations = []
    for operation_id, deployment, call, schema, key, nonce, cipher in rows:
        assert schema == 2
        aad = operation_aad_from_metadata({"schema_version": schema, "operation_id": operation_id,
            "deployment_id": deployment, "call_id": call, "kind": "call.upsert"})
        operation = decode_operation_v2(case.writer._keyring.decrypt(
            EncryptedValue(key, nonce, cipher), aad=aad
        ))
        if operation.payload.status in {"closing", "closed"}:
            operations.append(operation)
    return operations


async def bind_target(case):
    case.clock[0] = NOW + timedelta(seconds=1)
    assert (await supervised(case, target_event(case))).status_code == 200
    case.provider.transfer_release.set()
    await case.requested


@pytest.mark.asyncio
async def test_phone_facts_v2_verified_correlation_only_matching_bridge_publishes(tmp_path):
    async with phone_case(tmp_path) as case:
        assert await phone_operations(case) == []
        for bad in ({"client_state": SecretStr("owned-wrong-correlation")},
                    {"call_session_id": "wrong-session"}, {"connection_id": "wrong-connection"},
                    {"to_e164": "+33102030499"}):
            assert (await supervised(case, target_event(case, **bad))).status_code == 200
        assert await phone_operations(case) == []
        await bind_target(case)
        # Accepted command/target ringing is not bridge proof.
        assert await phone_operations(case) == []
        for bad in ({"call_control_id": "wrong-target"}, {"call_leg_id": "wrong-leg"}):
            observed = target_event(case, "call.bridged", **bad)
            assert (await supervised(case, observed)).status_code == 200
        assert await phone_operations(case) == []
        case.clock[0] = NOW + timedelta(seconds=2)
        bridge = target_event(case, "call.bridged")
        assert (await supervised(case, bridge)).status_code == 200
        closing = [operation for operation in await phone_operations(case)
                   if operation.payload.status == "closing"]
        assert len(closing) == 1 and closing[0].schema_version == 2
        assert closing[0].operation_id == uuid5(case.grant.call_id, "qualified-line-bridge")
        assert closing[0].occurred_at == bridge.occurred_at
        assert closing[0].payload.retention_until == case.grant.retention_until
        assert closing[0].payload.message_result is None
        facts = await case.writer.read_call_lifecycle(case.grant.call_id)
        assert facts.qualified_line_bridged_at == bridge.occurred_at
        assert await case.registry.live_call_count() == 1


@pytest.mark.asyncio
async def test_phone_facts_v2_original_hangup_atomic_closed_keeps_business_slot(
    tmp_path, monkeypatch,
):
    async with phone_case(tmp_path) as case:
        await bind_target(case)
        case.clock[0] = NOW + timedelta(seconds=2)
        bridge = target_event(case, "call.bridged")
        assert (await supervised(case, bridge)).status_code == 200
        authority = await case.session._registry_terminalizer.reserve_or_read(
            case.session._terminal_proposal("qualified_line_connected"))
        assert await case.session._commit_authoritative_terminal_call(status=authority.status,
            reason=authority.reason, disclosure_completed=True,
            operation_id=authority.completion_token, ended_at=authority._closed_at)
        before = frozen_bytes(case)
        await case.session._registry_terminalizer.complete(authority)
        case.clock[0] += timedelta(seconds=1)
        hangup = event("call.hangup", occurred_at=case.clock[0])
        assert (await supervised(case, hangup)).status_code == 200
        assert (await supervised(case, hangup)).status_code == 200
        operations = await phone_operations(case)
        closed = [operation for operation in operations if operation.payload.status == "closed"]
        assert len(closed) == 1 and closed[0].schema_version == 2
        assert closed[0].operation_id == uuid5(case.grant.call_id, hangup.event_id)
        assert closed[0].occurred_at == hangup.occurred_at
        assert closed[0].payload.message_result is None
        same_slot = frozen_bytes(case) == before
        assert same_slot and await case.registry.live_call_count() == 0
        native_ingest = case.sink.ingest_v2

        async def ingest(operation):
            case.connection.rows = [({"schema_version": 2, "status": "applied",
                "operation_id": str(operation.operation_id), "payload_sha256":
                hashlib.sha256(canonical_operation_bytes(operation)).hexdigest()},)]
            await native_ingest(operation)

        async def fail():
            pytest.fail("native phone-fact relay degraded")

        monkeypatch.setattr(case.sink, "ingest_v2", ingest)
        relay = OutboxRelay(case.writer, case.sink, utcnow=lambda: case.clock[0],
                            on_degraded=fail, drain=fail)
        assert (await relay.run_once()).status == "delivered"
        assert (await supervised(case, bridge)).status_code == 200
        assert (await supervised(case, hangup)).status_code == 200
        assert await case.writer.oldest_outbox_created_at() is None
        assert frozen_bytes(case) == before


@pytest.mark.asyncio
async def test_phone_facts_v2_held_signed_status_commit_and_wrong_generation_roll_back(
    tmp_path, monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with phone_case(tmp_path, failpoint=hold) as case:
        await bind_target(case)
        native_submit = case.writer.submit_webhook

        def submit(**options):
            nonlocal armed
            if options["receipt"]["event_type"] == "call.hangup":
                armed = True
            return native_submit(**options)

        monkeypatch.setattr(case.writer, "submit_webhook", submit)
        case.clock[0] = NOW + timedelta(seconds=3)
        hangup = event("call.hangup", occurred_at=case.clock[0])
        waiting = asyncio.create_task(supervised(case, hangup))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            with sqlite3.connect(case.path) as db:
                assert db.execute("SELECT count(*) FROM webhook_receipts WHERE event_id=?",
                                  (hangup.event_id,)).fetchone()[0] == 0
                assert db.execute("SELECT state FROM call_leases").fetchone()[0] == "active"
            assert await case.registry.live_call_count() == 1
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        finally:
            release.set()
        await case.writer.wait_until_idle()
        assert (await supervised(case, hangup)).status_code == 200
        closed = [op for op in await phone_operations(case) if op.payload.status == "closed"]
        assert len(closed) == 1 and closed[0].schema_version == 2
    directory = tmp_path / "wrong-generation"
    directory.mkdir()
    async with phone_case(directory) as case:
        await bind_target(case)
        case.clock[0] = NOW + timedelta(seconds=3)
        hangup = event("call.hangup", occurred_at=case.clock[0])
        resolved = await case.registry.resolve_webhook(hangup)
        effect = replace(resolved.effect, operation_generation=uuid4())
        invalid = ResolvedWebhook(effect=effect, reservation=resolved.reservation)
        finalized = await case.supervisor.start_webhook_finalization(hangup, invalid).wait()
        assert finalized.status_code == 503
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT count(*) FROM webhook_receipts WHERE event_id=?",
                              (hangup.event_id,)).fetchone()[0] == 0
            assert db.execute("SELECT state FROM call_leases").fetchone()[0] == "active"


@pytest.mark.asyncio
async def test_phone_facts_v2_expiry_erase_keep_verified_original_facts_without_content(tmp_path):
    for variant in ("expiry", "erase"):
        directory = tmp_path / variant
        directory.mkdir()
        async with phone_case(directory) as case:
            await bind_target(case)
            if variant == "expiry":
                case.clock[0] = case.grant.retention_until
            else:
                await case.writer.erase_call_content(
                    case.grant.call_id, lease_token=uuid4(), now=NOW
                )
            bridge = target_event(case, "call.bridged")
            assert (await supervised(case, bridge)).status_code == 200
            facts = await case.writer.read_call_lifecycle(case.grant.call_id)
            assert facts.qualified_line_bridged_at == bridge.occurred_at
            assert await phone_operations(case) == []
            assert await case.registry.live_call_count() == 1
            await case.session._prepare_partial_result()
            assert case.requests == [] and case.session._partial_result is None
            hangup = event("call.hangup", occurred_at=case.clock[0])
            assert (await supervised(case, hangup)).status_code == 200
            facts = await case.writer.read_call_lifecycle(case.grant.call_id)
            assert facts.original_ended_at == hangup.occurred_at
            assert facts.retention_until == case.grant.retention_until
            assert await case.registry.live_call_count() == 0
            assert (await case.writer.read_retained_call(case.grant.call_id)).erased
