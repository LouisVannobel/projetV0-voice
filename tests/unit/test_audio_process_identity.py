"""Native process identities are distinct from immutable lease and wire identities."""

from __future__ import annotations

import sqlite3
from uuid import UUID, uuid4

import pytest
from test_audio_writer import CALL, DEADLINE, GENERATION, NOW, WORKSPACE, owned, snapshot

from projetv0_voice.audio_contract import VoiceOperationV2
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, DisclosureEvidenceV1
from projetv0_voice.persistence.commands import PersistenceError, canonical_operation_bytes
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts, PersistenceWriter

AGENT = "native-agent"
DEPLOYMENT = "native-deployment"


@pytest.mark.parametrize(
    ("agent", "deployment"),
    [(None, None), (AGENT, None), (None, DEPLOYMENT), ("", DEPLOYMENT),
     (True, DEPLOYMENT), (AGENT, 1), (AGENT, "bad\ndeployment")],
    ids=["missing-pair", "missing-deployment", "missing-agent", "empty-agent",
         "boolean-agent", "numeric-deployment", "control-deployment"],
)
def test_process_identity_v2_invalid_pair_refuses_before_file_mutation(tmp_path, agent, deployment):
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    fresh = tmp_path / "absent" / "voice.sqlite"
    with pytest.raises(ValueError, match="^invalid_writer_process_identity$"):
        PersistenceWriter(fresh, keyring, contract_version=2,
                          process_agent_id=agent, process_deployment_id=deployment)
    assert not fresh.parent.exists()
    existing = tmp_path / "untouched.sqlite"
    existing.write_bytes(b"owned untouched fixture")
    with pytest.raises(ValueError, match="^invalid_writer_process_identity$"):
        PersistenceWriter(existing, keyring, contract_version=2,
                          process_agent_id=agent, process_deployment_id=deployment)
    assert existing.read_bytes() == b"owned untouched fixture"


async def seed_lease(writer, *, agent=AGENT):
    await writer.submit_webhook(receipt={"event_id": "identity-admission",
        "event_type": "call.initiated", "call_control_id": "identity-control", "occurred_at": NOW,
        "received_at": NOW, "semantic_fingerprint_sha256": b"i" * 32},
        lease={"action": "upsert", "call_control_id": "identity-control", "call_id": CALL,
            "tenant_id": str(WORKSPACE), "agent_id": agent, "state": "pending",
            "token_hash": b"t" * 32, "created_at": NOW, "expires_at": DEADLINE, "closed_at": None},
        operation=None, admission_facts=LocalCallAdmissionFacts(
            CALL, NOW, DEADLINE, "identity-leg", "identity-session",
            admission_generation=GENERATION,
        )).wait()


def control(*, deployment=DEPLOYMENT, closed=False):
    return VoiceOperationV2(schema_version=2, operation_id=uuid4(), deployment_id=deployment,
        call_id=CALL, occurred_at=NOW, kind="call.upsert", payload=CallUpsertPayloadV1(
            telnyx_call_control_id="identity-control", telnyx_call_leg_id="identity-leg",
            telnyx_call_session_id="identity-session", status="closed" if closed else "active",
            disclosure_state="completed", started_at=NOW, ended_at=NOW if closed else None,
            end_reason="closed" if closed else None, retention_until=DEADLINE,
            disclosure_evidence=DisclosureEvidenceV1(schema_version=1, started_at=NOW,
                completed_at=NOW, failed_at=None, input_gate_opened_at=NOW)))


def encrypted_rows(path):
    with sqlite3.connect(path) as db:
        return db.execute("SELECT op_id,schema_version,deployment_id,key_version,nonce,ciphertext "
                          "FROM outbox ORDER BY queue_id").fetchall()


@pytest.mark.asyncio
async def test_process_identity_v2_unequal_agent_deployment_pin_and_native_publication(tmp_path):
    path = tmp_path / "unequal.sqlite"
    async with owned(path, contract_version=2, process_agent_id=AGENT,
                     process_deployment_id=DEPLOYMENT) as (writer, _keyring):
        await seed_lease(writer)
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT agent_id FROM call_leases").fetchone()[0] == AGENT
            assert db.execute(
                "SELECT deployment_id FROM local_audio_pin"
            ).fetchone()[0] == DEPLOYMENT
            assert db.execute("PRAGMA user_version").fetchone()[0] == 8
        operation = control()
        await writer.publish_control_v2(operation, generation=GENERATION)
        with pytest.raises(PersistenceError):
            await writer.publish_control_v2(control(deployment=AGENT), generation=GENERATION)
        with pytest.raises(PersistenceError):
            await writer.publish_control_v2(control(), generation=UUID(int=999))
        final = control(closed=True)
        assert await writer.freeze_call_publication_v2(final, None, generation=GENERATION,
            provider_callback=None) == final.model_copy(update={"payload": final.payload.model_copy(
                update={"transcript_loss_count": 0})})
        assert await writer.quick_check() and not writer.is_degraded


@pytest.mark.asyncio
async def test_process_identity_v2_foreign_lease_agent_cannot_bind_or_rewrite(tmp_path):
    path = tmp_path / "foreign.sqlite"
    async with owned(path, contract_version=2, process_agent_id=AGENT,
                     process_deployment_id=DEPLOYMENT) as (writer, _keyring):
        await seed_lease(writer, agent="foreign-agent")
        with pytest.raises(PersistenceError):
            await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT agent_id FROM call_leases").fetchone()[0] == "foreign-agent"
            assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_process_identity_v2_restart_conflict_refuses_new_work_preserves_original_bytes(
    tmp_path,
):
    path = tmp_path / "restart.sqlite"
    async with owned(path, contract_version=2, process_agent_id=AGENT,
                     process_deployment_id=DEPLOYMENT) as (writer, _keyring):
        await seed_lease(writer)
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        original = control()
        await writer.publish_control_v2(original, generation=GENERATION)
    before = encrypted_rows(path)
    with sqlite3.connect(path) as db:
        pin_before = db.execute("SELECT * FROM local_audio_pin").fetchall()
    async with owned(path, contract_version=2, process_agent_id=AGENT,
                     process_deployment_id=DEPLOYMENT) as (writer, _keyring):
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        assert encrypted_rows(path) == before
    async with owned(path, contract_version=2, process_agent_id=AGENT,
                     process_deployment_id="replacement-deployment") as (writer, _keyring):
        with pytest.raises(PersistenceError):
            await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        with pytest.raises(PersistenceError):
            await writer.publish_control_v2(control(deployment="replacement-deployment"),
                                            generation=GENERATION)
        claimed = await writer.read_relay_batch(batch_size=100, now=NOW, lease_seconds=30)
        assert [canonical_operation_bytes(item.operation) for item in claimed] == [
            canonical_operation_bytes(original)]
        assert encrypted_rows(path) == before
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT * FROM local_audio_pin").fetchall() == pin_before
        assert not writer.is_degraded
    async with owned(tmp_path / "legacy.sqlite") as (writer, _keyring):
        assert writer.contract_version == 1
        with sqlite3.connect(tmp_path / "legacy.sqlite") as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 5
