"""First RED consumers for fixed V2 publication through the real SQLite writer."""

from __future__ import annotations

import asyncio
import base64
import math
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from projetv0_voice.audio_contract import (
    BeginCallSnapshotV2,
    VoiceOperationV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.models import (
    CallUpsertPayloadV1,
    DisclosureEvidenceV1,
    RecordingUpsertPayloadV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import (
    PersistenceCommand,
    PersistenceError,
    canonical_operation_bytes,
    decode_operation,
    decode_operation_v2,
    encrypt_audio_operation,
    operation_aad,
)
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts, PersistenceWriter

NOW = datetime(2026, 10, 6, 10, tzinfo=UTC)
DEADLINE = NOW + timedelta(days=30)
CALL, WORKSPACE, RECORDING, GENERATION = (UUID(int=value) for value in (1, 2, 3, 4))
KEY = bytes(range(32))


def snapshot():
    return BeginCallSnapshotV2.model_validate({
        "schema_version": 2, "workspace_id": str(WORKSPACE), "call_id": str(CALL),
        "configuration_revision": 7, "knowledge": {"business_name": "Fixture company",
            "sector": "garage", "opening_hours": "", "services": "", "prices": "",
            "faq": "", "instructions": ""}, "transfer_destination": None,
        "retention_until": DEADLINE.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "recording_policy": "local_30d", "recording_contact_phone": "+33123456789",
        "audio_available": True, "recording_id": str(RECORDING),
    })


def chunk(keyring, sequence=0):
    payload = {"schema_version": 2, "workspace_id": str(WORKSPACE),
        "recording_id": str(RECORDING), "sequence": sequence, "sample_count": 8000,
        "sample_rate": 8000, "channels": 2, "sample_format": "s16le",
        "configuration_revision": 7,
        "retention_until": snapshot().model_dump(mode="json")["retention_until"],
        "crypto_version": 1, "key_version": 1,
        "nonce_b64": base64.b64encode(bytes(12)).decode("ascii"),
        "ciphertext_b64": base64.b64encode(bytes(32016)).decode("ascii")}
    wire = {"schema_version": 2, "operation_id": str(UUID(int=100 + sequence)),
        "deployment_id": "agent-a", "call_id": str(CALL), "occurred_at": NOW,
        "kind": "audio.chunk", "payload": payload}
    metadata = VoiceOperationV2.model_validate(wire)
    inner = keyring.encrypt(b"\x01\x00\x02\x00" * 8000, aad=canonical_audio_chunk_aad(metadata))
    payload.update(nonce_b64=base64.b64encode(inner.nonce).decode("ascii"),
                   ciphertext_b64=base64.b64encode(inner.ciphertext).decode("ascii"))
    return VoiceOperationV2.model_validate(wire)


async def seed_admission(writer):
    ticket = writer.submit_webhook(
        receipt={"event_id": "audio-admission", "event_type": "call.initiated",
            "call_control_id": "control-a", "occurred_at": NOW, "received_at": NOW,
            "semantic_fingerprint_sha256": b"a" * 32},
        lease={"action": "upsert", "call_control_id": "control-a", "call_id": CALL,
            "tenant_id": str(WORKSPACE), "agent_id": "agent-a", "state": "pending",
            "token_hash": b"b" * 32, "created_at": NOW,
            "expires_at": NOW + timedelta(hours=1), "closed_at": None},
        operation=None,
        admission_facts=LocalCallAdmissionFacts(CALL, NOW, DEADLINE, "leg-a", "session-a",
                                               admission_generation=GENERATION),
    )
    await ticket.wait()
    await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)


@asynccontextmanager
async def owned(path, *, utcnow=lambda: NOW, **options):
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    writer = PersistenceWriter(path, keyring, utcnow=utcnow, **options)
    task = asyncio.create_task(writer.run())
    try:
        assert await writer.wait_ready()
        yield writer, keyring
    finally:
        if writer.is_degraded:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        else:
            await writer.drain(timeout_seconds=2)
            await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_v2_admitted_chunk_commits_real_ciphertext_and_claim_decodes_exact_bytes(tmp_path):
    path = tmp_path / "voice.sqlite"
    async with owned(path, contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        operation = chunk(keyring)
        assert writer.offer_audio_chunk(operation) is True
        await writer.wait_for_audio_commit(operation.operation_id)
        with sqlite3.connect(path) as db:
            version, kind, key_version, nonce, ciphertext = db.execute(
                "SELECT schema_version,kind,key_version,nonce,ciphertext FROM outbox"
            ).fetchone()
            assert db.execute("PRAGMA user_version").fetchone()[0] == 6
        assert (version, kind) == (2, "audio.chunk") and len(nonce) == 12
        plaintext = keyring.decrypt(EncryptedValue(key_version, nonce, ciphertext),
                                    aad=operation_aad(operation))
        assert plaintext == canonical_operation_bytes(operation)
        assert decode_operation_v2(plaintext) == operation
        assert len(operation_aad(operation)) + len(nonce) + len(ciphertext) <= 46812
        claimed = await writer.read_relay_batch(batch_size=1, now=NOW, lease_seconds=30)
        assert len(claimed) == 1 and claimed[0].operation == operation
        assert b"\x01\x00\x02\x00" * 8000 not in path.read_bytes()
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_one_audio_commit_and_low_queue_refusal_preserve_native_control(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold_commit(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(
        tmp_path / "voice.sqlite", contract_version=2, failpoint=hold_commit
    ) as (writer, keyring):
        await seed_admission(writer)
        first, second = chunk(keyring), chunk(keyring, 1)
        armed = True
        assert writer.offer_audio_chunk(first)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert writer.offer_audio_chunk(second) is False and not writer.is_degraded
            cancelled_wait = asyncio.create_task(writer.wait_for_audio_commit(first.operation_id))
            await asyncio.sleep(0)
            cancelled_wait.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_wait
            assert writer.offer_audio_chunk(second) is False
        finally:
            release.set()
        await writer.wait_for_audio_commit(first.operation_id)
        await writer.wait_until_idle()
        entered.clear()
        release.clear()
        armed = True
        tickets = []
        for index in range(242):
            tickets.append(writer.submit_webhook(receipt={"event_id": f"control-{index}",
                "event_type": "fixture.control", "call_control_id": None,
                "occurred_at": NOW, "received_at": NOW, "semantic_fingerprint_sha256": b"c" * 32},
                lease=None, operation=None))
            if index == 0:
                await asyncio.wait_for(entered.wait(), 1)
        try:
            assert writer.queue_size == 241
            assert writer.offer_audio_chunk(second) is False and not writer.is_degraded
            control = asyncio.create_task(writer.quick_check())
        finally:
            release.set()
        assert await asyncio.wait_for(control, 2)
        await tickets[-1].wait()
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_fixed_v2_refuses_legacy_outbox_before_mutation_and_default_v1_replays_bytes(
    tmp_path,
):
    path = tmp_path / "voice.sqlite"
    legacy = VoiceOperationV1(schema_version=1, operation_id=UUID(int=500), deployment_id="agent-a",
        call_id=CALL, occurred_at=NOW, kind="recording.upsert", payload=RecordingUpsertPayloadV1(
            recording_id=RECORDING, status="failed", telnyx_recording_id="fixture-recording",
            channels="dual", format="wav", started_at=None, ended_at=None, retention_until=None))
    async with owned(path) as (writer, _keyring):
        await writer.commit_control(PersistenceCommand("outbox", {"operation": legacy}, None))
    before = path.read_bytes()
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    refused = PersistenceWriter(path, keyring, utcnow=lambda: NOW, contract_version=2)
    task = asyncio.create_task(refused.run())
    assert await refused.wait_ready() is False
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert path.read_bytes() == before
    async with owned(path) as (writer, _keyring):
        claimed = await writer.read_relay_batch(batch_size=1, now=NOW, lease_seconds=30)
        assert len(claimed) == 1 and type(claimed[0].operation) is VoiceOperationV1
        assert claimed[0].operation == legacy and not writer.is_degraded


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["commit-lease", "stale-terminal"])
async def test_fixed_v2_public_legacy_terminal_rejects_before_queue_and_preserves_control(
    tmp_path, method,
):
    path = tmp_path / "voice.sqlite"
    async with owned(path, contract_version=2) as (writer, _keyring):
        await seed_admission(writer)
    mutations = []

    def observe(name):
        if name == "after_mutation_before_commit":
            mutations.append(name)

    end = NOW + timedelta(seconds=10)
    legacy = VoiceOperationV1(
        schema_version=1, operation_id=UUID(int=700), deployment_id="agent-a", call_id=CALL,
        occurred_at=end, kind="call.upsert", payload=CallUpsertPayloadV1(
            telnyx_call_control_id="control-a", telnyx_call_leg_id="leg-a",
            telnyx_call_session_id="session-a", status="closed", disclosure_state="failed",
            started_at=NOW, ended_at=end, end_reason="fixture-hangup", retention_until=DEADLINE,
        ),
    )
    async with owned(path, contract_version=2, failpoint=observe) as (writer, _keyring):
        stale = writer.take_stale_leases()
        assert len(stale) == 1 and stale[0].call_id == CALL
        with sqlite3.connect(path) as db:
            original = db.execute(
                "SELECT state,closed_at,lifecycle_json,token_hash FROM call_leases"
            ).fetchone()
        with pytest.raises(PersistenceError, match="writer_contract_mismatch"):
            if method == "commit-lease":
                await writer.commit_lease(
                    call_control_id="control-a", call_id=CALL, tenant_id=str(WORKSPACE),
                    agent_id="agent-a", state="terminal", token_hash=b"b" * 32, created_at=NOW,
                    expires_at=NOW + timedelta(hours=1), closed_at=end, operation=legacy,
                )
            else:
                await writer.terminalize_stale_lease(stale[0], closed_at=end, operation=legacy)
        assert writer.queue_size == 0 and mutations == []
        with sqlite3.connect(path) as db:
            assert db.execute(
                "SELECT state,closed_at,lifecycle_json,token_hash FROM call_leases"
            ).fetchone() == original
            assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
        assert not writer.is_degraded and await writer.quick_check()


@pytest.mark.asyncio
async def test_committed_audio_receipt_keeps_its_bounded_slot_until_exact_wait_consumes_it(
    tmp_path,
):
    async with owned(tmp_path / "voice.sqlite", contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        first, second = chunk(keyring), chunk(keyring, 1)
        assert writer.offer_audio_chunk(first)
        await writer.wait_until_idle()  # Actual native COMMIT, no receipt wait yet.
        assert writer.offer_audio_chunk(second) is False
        await writer.wait_for_audio_commit(first.operation_id)
        assert writer.offer_audio_chunk(second) is True
        await writer.wait_for_audio_commit(second.operation_id)
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_queued_erase_refuses_cached_audio_without_poisoning_native_control(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold_commit(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(
        tmp_path / "voice.sqlite", contract_version=2, failpoint=hold_commit
    ) as (writer, keyring):
        await seed_admission(writer)
        operation = chunk(keyring)
        armed = True
        control = writer.submit_webhook(receipt={"event_id": "held-control",
            "event_type": "fixture.control", "call_control_id": None,
            "occurred_at": NOW, "received_at": NOW, "semantic_fingerprint_sha256": b"d" * 32},
            lease=None, operation=None)
        await asyncio.wait_for(entered.wait(), 1)
        control_wait = asyncio.create_task(control.wait())
        erasing = asyncio.create_task(writer.erase_call_content(CALL, now=NOW))
        queued = asyncio.Event()
        loop = asyncio.get_running_loop()
        queue_check: asyncio.Handle | None = None

        def check_queue():
            nonlocal queue_check
            if writer.queue_size == 1:
                queued.set()
            else:
                queue_check = loop.call_soon(check_queue)

        queue_check = loop.call_soon(check_queue)
        try:
            await asyncio.wait_for(queued.wait(), 1)
            assert writer.queue_size == 1  # Real erase is queued first; its COMMIT is still held.
            assert writer.offer_audio_chunk(operation) is True
            release.set()
            await control_wait
            await erasing
            with pytest.raises(PersistenceError):
                await writer.wait_for_audio_commit(operation.operation_id)
            assert not writer.is_degraded
            assert await writer.quick_check()
            assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
        finally:
            release.set()
            if queue_check is not None:
                queue_check.cancel()
            for task in (control_wait, erasing):
                if not task.done():
                    task.cancel()
            await asyncio.gather(control_wait, erasing, return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_physical_growth_and_unknown_capacity_preserve_control(tmp_path):
    path = tmp_path / "voice.sqlite"
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False
    measurement = "actual"

    def file_size(candidate):
        if measurement == "unknown":
            raise OSError("owned-measurement-unavailable")
        if measurement == "reserve":
            return (268_435_456 - 33_554_432) // 2 if candidate == path else 0
        return candidate.stat().st_size if candidate.exists() else 0

    async def hold_commit(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(
        path, contract_version=2, file_size=file_size, failpoint=hold_commit
    ) as (writer, keyring):
        await seed_admission(writer)
        with sqlite3.connect(path) as db:
            page = db.execute("PRAGMA page_size").fetchone()[0]
            before_pages = db.execute("PRAGMA page_count").fetchone()[0]
        assert writer.pragma_state["secure_delete"] == 1
        before = path.stat().st_size
        operation = chunk(keyring)
        prepared = encrypt_audio_operation(operation, keyring)
        assert prepared.envelope_size <= 46812
        armed = True
        assert writer.offer_audio_chunk(operation)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            journal = path.with_name(path.name + "-journal").stat().st_size
            assert journal > 0
            # Adopted allowance includes native table/index/overflow/sequence pages.
            allowance = (math.ceil(46812 / page) + 32) * page
            assert path.stat().st_size + journal <= 2 * (before + allowance)
        finally:
            release.set()
        await writer.wait_for_audio_commit(operation.operation_id)
        with sqlite3.connect(path) as db:
            after_pages = db.execute("PRAGMA page_count").fetchone()[0]
        assert 0 < after_pages - before_pages <= math.ceil(46812 / page) + 32
        assert path.stat().st_size - before == (after_pages - before_pages) * page
        assert not path.with_name(path.name + "-journal").exists()
        for mode in ("unknown", "reserve"):
            measurement = mode
            assert writer.offer_audio_chunk(chunk(keyring, 1)) is False
            assert not writer.is_degraded
        measurement = "actual"
        assert await writer.quick_check()
        assert writer.offer_audio_chunk(chunk(keyring, 1)) is True
        await writer.wait_for_audio_commit(UUID(int=101))


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["default-v1", "missing-marker", "extra-table"])
async def test_fixed_v2_provenance_refusal_never_repairs_or_mutates_database(tmp_path, tamper):
    path = tmp_path / "voice.sqlite"
    async with owned(path, contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        assert writer.offer_audio_chunk(chunk(keyring))
        await writer.wait_for_audio_commit(UUID(int=100))
    if tamper != "default-v1":
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM local_audio_contract" if tamper == "missing-marker"
                       else "CREATE TABLE owned_unexpected(value INTEGER)")
    before = path.read_bytes()
    refused = PersistenceWriter(path, keyring, utcnow=lambda: NOW,
                                contract_version=1 if tamper == "default-v1" else 2)
    task = asyncio.create_task(refused.run())
    assert await refused.wait_ready() is False
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert path.read_bytes() == before


def active_control_v2(*, gate=False):
    started = NOW + timedelta(seconds=1)
    completed = NOW + timedelta(seconds=2)
    opened = NOW + timedelta(seconds=3) if gate else None
    return VoiceOperationV2(
        schema_version=2, operation_id=UUID(int=801 if gate else 800), deployment_id="agent-a",
        call_id=CALL, occurred_at=opened or completed, kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="control-a", telnyx_call_leg_id="leg-a",
            telnyx_call_session_id="session-a", status="active", disclosure_state="completed",
            started_at=started, ended_at=None, end_reason=None, retention_until=DEADLINE,
            disclosure_evidence=DisclosureEvidenceV1(
                schema_version=1, started_at=started, completed_at=completed, failed_at=None,
                input_gate_opened_at=opened,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_control_v2_fresh_disclosure_and_gate_commit_exact_crypto_and_immutable_replay(
    tmp_path,
):
    path = tmp_path / "voice.sqlite"
    now = NOW + timedelta(seconds=3)
    first, gated = active_control_v2(), active_control_v2(gate=True)
    async with owned(path, contract_version=2, utcnow=lambda: now) as (writer, keyring):
        await seed_admission(writer)
        await writer.publish_control_v2(first, generation=GENERATION)
        first_facts = await writer.read_call_lifecycle(CALL)
        assert first_facts.started_at == first.payload.started_at
        assert first_facts.disclosure_evidence == first.payload.disclosure_evidence
        assert first_facts.original_ended_at is None
        await writer.publish_control_v2(gated, generation=GENERATION)
        with sqlite3.connect(path) as db:
            frozen = db.execute(
                "SELECT op_id,schema_version,kind,key_version,nonce,ciphertext FROM outbox "
                "ORDER BY queue_id"
            ).fetchall()
            assert db.execute("PRAGMA user_version").fetchone()[0] == 6
        assert len(frozen) == 2
        for row, operation in zip(frozen, (first, gated), strict=True):
            assert row[:3] == (str(operation.operation_id), 2, "call.upsert")
            plaintext = keyring.decrypt(EncryptedValue(*row[3:]), aad=operation_aad(operation))
            assert plaintext == canonical_operation_bytes(operation)
            assert decode_operation_v2(plaintext) == operation
        await writer.publish_control_v2(first, generation=GENERATION)
        await writer.publish_control_v2(gated, generation=GENERATION)
        with sqlite3.connect(path) as db:
            assert db.execute(
                "SELECT op_id,schema_version,kind,key_version,nonce,ciphertext FROM outbox "
                "ORDER BY queue_id"
            ).fetchall() == frozen
        facts = await writer.read_call_lifecycle(CALL)
        assert facts.admission_generation == GENERATION
        assert facts.admitted_at == NOW and facts.retention_until == DEADLINE
        assert facts.disclosure_evidence.input_gate_opened_at == now
        assert facts.original_ended_at is None and facts.recording_enabled is False
        claimed = await writer.read_relay_batch(batch_size=2, now=now, lease_seconds=30)
        assert tuple(item.operation for item in claimed) == (first, gated)
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_control_v2_default_v1_refuses_before_queue_and_preserves_legacy_bytes(tmp_path):
    path = tmp_path / "voice.sqlite"
    operation = active_control_v2()
    legacy = VoiceOperationV1.model_validate({
        **operation.model_dump(mode="python"), "schema_version": 1,
    })
    async with owned(path) as (writer, keyring):
        with pytest.raises(PersistenceError):
            await writer.publish_control_v2(operation, generation=GENERATION)
        assert writer.queue_size == 0 and not writer.is_degraded
        await writer.commit_control(PersistenceCommand("outbox", {"operation": legacy}, None))
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 5
            row = db.execute(
                "SELECT schema_version,key_version,nonce,ciphertext FROM outbox"
            ).fetchone()
        assert row[0] == 1
        plaintext = keyring.decrypt(EncryptedValue(*row[1:]), aad=operation_aad(legacy))
        assert plaintext == canonical_operation_bytes(legacy)
        assert decode_operation(plaintext) == legacy
        assert await writer.quick_check()


@pytest.mark.asyncio
async def test_control_v2_rejects_v1_and_noncall_without_mutation_or_degradation(tmp_path):
    async with owned(tmp_path / "voice.sqlite", contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        legacy = VoiceOperationV1.model_validate({
            **active_control_v2().model_dump(mode="python"), "schema_version": 1,
        })
        before = await writer.read_call_lifecycle(CALL)
        for operation in (legacy, chunk(keyring)):
            with pytest.raises(PersistenceError):
                await writer.publish_control_v2(operation, generation=GENERATION)
            assert writer.queue_size == 0 and not writer.is_degraded
        assert await writer.read_call_lifecycle(CALL) == before
        assert await writer.oldest_outbox_created_at() is None
        assert await writer.quick_check()


@pytest.mark.asyncio
async def test_control_v2_requires_original_generation_retention_and_provider_identity(tmp_path):
    now = NOW + timedelta(seconds=3)
    async with owned(tmp_path / "voice.sqlite", contract_version=2, utcnow=lambda: now) as (
        writer, _keyring,
    ):
        await seed_admission(writer)
        original = active_control_v2()
        wire = original.model_dump(mode="python")
        wrong_retention = VoiceOperationV2.model_validate({
            **wire, "payload": {**wire["payload"], "retention_until": DEADLINE + timedelta(days=1)},
        })
        wrong_identity = VoiceOperationV2.model_validate({
            **wire, "payload": {**wire["payload"], "telnyx_call_session_id": "other-session"},
        })
        before = await writer.read_call_lifecycle(CALL)
        for operation, generation in (
            (original, UUID(int=999)), (wrong_retention, GENERATION), (wrong_identity, GENERATION),
        ):
            with pytest.raises(PersistenceError):
                await writer.publish_control_v2(operation, generation=generation)
            assert not writer.is_degraded
        assert await writer.read_call_lifecycle(CALL) == before
        assert await writer.oldest_outbox_created_at() is None
        assert await writer.quick_check()


@pytest.mark.asyncio
async def test_control_v2_keeps_the_original_configuration_pin_after_revision_conflict(tmp_path):
    path = tmp_path / "voice.sqlite"
    now = NOW + timedelta(seconds=3)
    async with owned(path, contract_version=2, utcnow=lambda: now) as (writer, _keyring):
        await seed_admission(writer)
        changed = BeginCallSnapshotV2.model_validate({
            **snapshot().model_dump(mode="python"), "configuration_revision": 8,
        })
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(changed, generation=GENERATION)
        with sqlite3.connect(path) as db:
            assert db.execute(
                "SELECT generation,configuration_revision,retention_until FROM local_audio_pin"
            ).fetchone() == (str(GENERATION), 7, DEADLINE.isoformat().replace("+00:00", "Z"))
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        assert not writer.is_degraded and await writer.quick_check()


@pytest.mark.asyncio
async def test_control_v2_expiry_and_committed_erase_refuse_without_false_phone_end(tmp_path):
    clock = [NOW + timedelta(seconds=3)]
    async with owned(tmp_path / "voice.sqlite", contract_version=2, utcnow=lambda: clock[0]) as (
        writer, _keyring,
    ):
        await seed_admission(writer)
        operation = active_control_v2()
        clock[0] = DEADLINE
        with pytest.raises(PersistenceError):
            await writer.publish_control_v2(operation, generation=GENERATION)
        assert not writer.is_degraded
        clock[0] = NOW
        await writer.erase_call_content(CALL, now=clock[0])
        with pytest.raises(PersistenceError):
            await writer.publish_control_v2(operation, generation=GENERATION)
        facts = await writer.read_call_lifecycle(CALL)
        assert facts.content_erased and facts.original_ended_at is None
        assert await writer.oldest_outbox_created_at() is None
        assert not writer.is_degraded and await writer.quick_check()
