"""Real SQLite terminal evidence.

Near-limit rows are engineering fixtures, not ten-minute capture.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import timedelta
from uuid import UUID

import pytest

from projetv0_voice.audio_contract import VoiceOperationV2
from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.persistence.commands import (
    PersistenceError,
    canonical_operation_bytes,
    decode_operation_v2,
    operation_aad,
)
from projetv0_voice.persistence.schema import LOCAL_AUDIO_CHOICE_MIGRATION_SQL
from tests.unit.test_audio_choice_writer import historical_v7, wait_for_queued
from tests.unit.test_audio_writer import (
    CALL,
    DEADLINE,
    GENERATION,
    NOW,
    RECORDING,
    WORKSPACE,
    authenticate_audio,
    chunk,
    owned,
    seed_admission,
    snapshot,
)


def terminal(*, revoke=False, last=0, total=8000, reason="complete", operation_id=2000):
    payload = {"schema_version": 2, "workspace_id": str(WORKSPACE),
        "recording_id": str(RECORDING), "configuration_revision": 7,
        "retention_until": DEADLINE.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "reason": "caller_declined" if revoke else reason}
    if not revoke:
        payload.update(last_sequence=last, total_samples=total)
    return VoiceOperationV2.model_validate({
        "schema_version": 2, "operation_id": str(UUID(int=operation_id)),
        "deployment_id": "agent-a", "call_id": str(CALL), "occurred_at": NOW,
        "kind": "audio.revoke" if revoke else "audio.finish", "payload": payload,
    })


def counters(path):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT committed_last_sequence,committed_total_samples FROM local_audio_pin"
        ).fetchone()


def terminal_slot(path, kind):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT op_id,fingerprint,key_version,nonce,ciphertext,acked "
            "FROM local_audio_terminal WHERE kind=?", (kind,),
        ).fetchone()


async def ack_controlled_successes(writer):
    """Native exact claim/ACK transaction fixture, not a live PostgreSQL success claim."""
    claimed = await writer.read_relay_batch(batch_size=100, now=NOW, lease_seconds=30)
    for item in claimed:
        assert (await writer.ack_outbox(queue_id=item.queue_id,
                                       expected_claim_attempt=item.claim_attempt)).applied


@pytest.mark.asyncio
async def test_terminal_real_chunk_counters_and_finish_include_only_known_commits(tmp_path):
    path = tmp_path / "terminal.sqlite"
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(path, contract_version=2, failpoint=hold) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 9
        assert counters(path) == (None, 0)
        operation = chunk(keyring)
        armed = True
        assert writer.offer_audio_chunk(operation)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert counters(path) == (None, 0)
            with pytest.raises(PersistenceError):
                await writer.publish_audio_terminal_v2(terminal(), generation=GENERATION)
        finally:
            release.set()
        await writer.wait_for_audio_commit(operation.operation_id)
        assert counters(path) == (0, 8000)
        if writer.offer_audio_chunk(operation):
            await writer.wait_for_audio_commit(operation.operation_id)
        assert counters(path) == (0, 8000)
        finished = terminal()
        await writer.publish_audio_terminal_v2(finished, generation=GENERATION)
        slot = terminal_slot(path, "audio.finish")
        assert slot[:2] == (str(finished.operation_id),
                            hashlib.sha256(canonical_operation_bytes(finished)).digest())
        plaintext = keyring.decrypt(EncryptedValue(*slot[2:5]), aad=operation_aad(finished))
        assert plaintext == canonical_operation_bytes(finished)
        assert decode_operation_v2(plaintext) == finished and slot[5] == 0
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
    # Explicit counter boundary fixture: no claim that 599 prior chunks were captured.
    near = tmp_path / "near-limit.sqlite"
    async with owned(near, contract_version=2) as (writer, _keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
    with sqlite3.connect(near) as db:
        db.execute("UPDATE local_audio_pin SET committed_last_sequence=598,"
                   "committed_total_samples=4792000")
    async with owned(near, contract_version=2) as (writer, keyring):
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        final_chunk = chunk(keyring, 599)
        assert writer.offer_audio_chunk(final_chunk)
        await writer.wait_for_audio_commit(final_chunk.operation_id)
        assert counters(near) == (599, 4_800_000)
        await writer.publish_audio_terminal_v2(terminal(last=599, total=4_800_000, reason="limit"),
                                               generation=GENERATION)
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
        assert await writer.quick_check() and not writer.is_degraded


@pytest.mark.asyncio
async def test_terminal_slots_survive_exact_ack_restart_and_immutable_replay(tmp_path):
    path = tmp_path / "replay.sqlite"
    finish, revoke = terminal(), terminal(revoke=True, operation_id=2001)
    async with owned(path, contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        first = chunk(keyring)
        assert writer.offer_audio_chunk(first)
        await writer.wait_for_audio_commit(first.operation_id)
        await writer.publish_audio_terminal_v2(finish, generation=GENERATION)
        frozen = terminal_slot(path, "audio.finish")
        await ack_controlled_successes(writer)
        assert terminal_slot(path, "audio.finish")[:5] == frozen[:5]
        assert terminal_slot(path, "audio.finish")[5] == 1
    async with owned(path, contract_version=2) as (writer, _keyring):
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        assert counters(path) == (0, 8000)
        if writer.offer_audio_chunk(first):
            with pytest.raises(PersistenceError):
                await writer.wait_for_audio_commit(first.operation_id)
        assert counters(path) == (0, 8000)
        await writer.publish_audio_terminal_v2(finish, generation=GENERATION)
        assert terminal_slot(path, "audio.finish") == (*frozen[:5], 1)
        for changed in (terminal(operation_id=2222), terminal(reason="failure")):
            with pytest.raises(PersistenceError):
                await writer.publish_audio_terminal_v2(changed, generation=GENERATION)
            assert not writer.is_degraded
        await writer.commit_audio_choice(
            CALL, generation=GENERATION, choice="off", occurred_at=None
        )
        await writer.publish_audio_terminal_v2(revoke, generation=GENERATION)
        revoked = terminal_slot(path, "audio.revoke")
        await ack_controlled_successes(writer)
        assert terminal_slot(path, "audio.revoke")[:5] == revoked[:5]
        assert terminal_slot(path, "audio.revoke")[5] == 1
    async with owned(path, contract_version=2) as (writer, _keyring):
        await writer.publish_audio_terminal_v2(revoke, generation=GENERATION)
        assert terminal_slot(path, "audio.revoke") == (*revoked[:5], 1)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT count(*) FROM local_audio_terminal").fetchone()[0] == 2
        assert counters(path) == (0, 8000) and not writer.is_degraded


@pytest.mark.asyncio
async def test_terminal_cancelled_commit_and_invalid_authority_never_expose_optimistic_success(
    tmp_path,
):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    empty = terminal(last=None, total=0)
    path = tmp_path / "held.sqlite"
    async with owned(path, contract_version=2, failpoint=hold) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        for operation, generation in ((chunk(keyring), GENERATION), (empty, UUID(int=999)),
                                       (terminal(), GENERATION)):
            with pytest.raises(PersistenceError):
                await writer.publish_audio_terminal_v2(operation, generation=generation)
            assert not writer.is_degraded
        bad_pin = VoiceOperationV2.model_validate({
            **empty.model_dump(mode="python"),
            "payload": {**empty.payload.model_dump(mode="python"), "configuration_revision": 8},
        })
        with pytest.raises(PersistenceError):
            await writer.publish_audio_terminal_v2(bad_pin, generation=GENERATION)
        armed = True
        publishing = asyncio.create_task(
            writer.publish_audio_terminal_v2(empty, generation=GENERATION)
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert terminal_slot(path, "audio.finish") is None
            publishing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await publishing
        finally:
            release.set()
        await writer.wait_until_idle()
        frozen = terminal_slot(path, "audio.finish")
        assert frozen is not None
        await writer.publish_audio_terminal_v2(empty, generation=GENERATION)
        assert terminal_slot(path, "audio.finish") == frozen
        assert await writer.quick_check() and not writer.is_degraded
    async with owned(tmp_path / "v1.sqlite") as (writer, _keyring):
        with pytest.raises(PersistenceError):
            await writer.publish_audio_terminal_v2(empty, generation=GENERATION)
        assert not writer.is_degraded and await writer.quick_check()


@pytest.mark.asyncio
async def test_terminal_revoke_after_off_and_expiry_preserves_phone_content_and_cleanup_replay(
    tmp_path,
):
    clock = [NOW]
    path = tmp_path / "revoke.sqlite"
    revoke = terminal(revoke=True, operation_id=2001)
    async with owned(path, contract_version=2, utcnow=lambda: clock[0]) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        first = chunk(keyring)
        assert writer.offer_audio_chunk(first)
        await writer.wait_for_audio_commit(first.operation_id)
        before = await writer.read_call_lifecycle(CALL)
        await writer.commit_audio_choice(
            CALL, generation=GENERATION, choice="off", occurred_at=None
        )
        clock[0] = DEADLINE + timedelta(seconds=1)
        await writer.publish_audio_terminal_v2(revoke, generation=GENERATION)
        frozen = terminal_slot(path, "audio.revoke")
        assert writer.offer_audio_chunk(chunk(keyring, 1)) is False
        assert await writer.read_call_lifecycle(CALL) == before
        with sqlite3.connect(path) as db:
            assert db.execute(
                "SELECT count(*) FROM outbox WHERE kind='audio.chunk'"
            ).fetchone()[0] == 0
            assert db.execute(
                "SELECT count(*) FROM outbox WHERE kind='call.upsert'"
            ).fetchone()[0] == 2
            assert db.execute("SELECT count(*) FROM sparra_content_fences").fetchone()[0] == 0
        await writer.publish_audio_terminal_v2(revoke, generation=GENERATION)
        assert terminal_slot(path, "audio.revoke") == frozen
        claimed = await writer.read_relay_batch(batch_size=100, now=clock[0], lease_seconds=30)
        assert tuple(
            item.operation for item in claimed if item.operation.kind == "audio.revoke"
        ) == (revoke,)
        assert frozen[5] == 0  # Local COMMIT is not known remote ACK/deletion.
        assert counters(path) == (0, 8000) and not writer.is_degraded


@pytest.mark.asyncio
async def test_terminal_historical_seven_migrates_atomically_with_unknown_accounting(tmp_path):
    path = tmp_path / "historical-seven.sqlite"
    historical_v7(path)
    with sqlite3.connect(path) as db:
        # Original qualified V8 migration, historical metadata only; no consent consumer claim.
        db.executescript(LOCAL_AUDIO_CHOICE_MIGRATION_SQL)
        original = db.execute("SELECT op_id,key_version,nonce,ciphertext FROM outbox").fetchone()
        lifecycle = json.loads(db.execute("SELECT lifecycle_json FROM call_leases").fetchone()[0])
        instant = NOW.isoformat().replace("+00:00", "Z")
        lifecycle.update(started_at=instant, disclosure_evidence={"schema_version": 1,
            "started_at": instant, "completed_at": instant, "failed_at": None,
            "input_gate_opened_at": instant})
        db.execute("UPDATE call_leases SET lifecycle_json=?", (json.dumps(lifecycle),))
        db.execute(
            "UPDATE local_audio_pin SET choice_state='accepted',choice_occurred_at=?", (instant,)
        )
        assert db.execute("PRAGMA user_version").fetchone()[0] == 8
    before = path.read_bytes()
    async with owned(path, contract_version=2) as (writer, keyring):
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        assert counters(path) == (None, None)
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 9
            assert db.execute(
                "SELECT op_id,key_version,nonce,ciphertext FROM outbox"
            ).fetchone() == original
            assert db.execute(
                "SELECT choice_state,choice_occurred_at,denied_at FROM local_audio_pin"
            ).fetchone() == ("accepted", instant, None)
            with pytest.raises(sqlite3.IntegrityError):
                db.execute("UPDATE local_audio_pin SET committed_last_sequence=0")
        original_sequence = chunk(keyring)
        fresh_identity = VoiceOperationV2.model_validate({
            **original_sequence.model_dump(mode="python"), "operation_id": str(UUID(int=9876)),
        })
        assert writer.offer_audio_chunk(fresh_identity) is False
        with pytest.raises(PersistenceError):
            await writer.publish_audio_terminal_v2(
                terminal(last=None, total=0), generation=GENERATION
            )
        await writer.publish_audio_terminal_v2(terminal(revoke=True, operation_id=2001),
                                               generation=GENERATION)
        assert not writer.is_degraded
    failed = tmp_path / "rollback-seven.sqlite"
    failed.write_bytes(before)

    def fail_upgrade(name):
        if name == "after_audio_terminal_migration_before_commit":
            raise RuntimeError("owned-terminal-migration-fault")

    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.unit.test_audio_writer import KEY

    refused = PersistenceWriter(failed, CryptoKeyring({1: KEY}, active_version=1),
                                contract_version=2, failpoint=fail_upgrade,
                                process_agent_id="agent-a", process_deployment_id="agent-a")
    task = asyncio.create_task(refused.run())
    ready = await refused.wait_ready()
    if ready:
        await refused.drain(2)
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert not ready and failed.read_bytes() == before
    async with owned(failed, contract_version=2) as (writer, _keyring):
        assert counters(failed) == (None, None)
    altered = tmp_path / "altered-seven.sqlite"
    altered.write_bytes(before)
    with sqlite3.connect(altered) as db:
        db.execute("CREATE TABLE owned_unexpected(value INTEGER)")
    frozen = altered.read_bytes()
    refused = PersistenceWriter(
        altered, CryptoKeyring({1: KEY}, active_version=1), contract_version=2,
        process_agent_id="agent-a", process_deployment_id="agent-a",
    )
    task = asyncio.create_task(refused.run())
    assert not await refused.wait_ready()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert altered.read_bytes() == frozen


@pytest.mark.asyncio
async def test_terminal_finish_seals_cached_and_queued_chunk_admission(tmp_path):
    path = tmp_path / "sealed.sqlite"
    async with owned(path, contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        first = chunk(keyring)
        assert writer.offer_audio_chunk(first)
        await writer.wait_for_audio_commit(first.operation_id)
        await writer.publish_audio_terminal_v2(terminal(), generation=GENERATION)
        assert writer.offer_audio_chunk(chunk(keyring, 1)) is False
        assert counters(path) == (0, 8000) and not writer.is_degraded
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(tmp_path / "queued-seal.sqlite", contract_version=2, failpoint=hold) as (
        writer, keyring,
    ):
        await seed_admission(writer)
        await authenticate_audio(writer)
        first = chunk(keyring)
        assert writer.offer_audio_chunk(first)
        await writer.wait_for_audio_commit(first.operation_id)
        armed = True
        control = writer.submit_webhook(
            receipt={"event_id": "held-terminal-control", "event_type": "fixture.control",
                "call_control_id": None, "occurred_at": NOW, "received_at": NOW,
                "semantic_fingerprint_sha256": b"t" * 32},
            lease=None,
            operation=None,
        )
        await asyncio.wait_for(entered.wait(), 1)
        finishing = asyncio.create_task(
            writer.publish_audio_terminal_v2(terminal(), generation=GENERATION)
        )
        try:
            await wait_for_queued(writer)
            next_chunk = chunk(keyring, 1)
            assert writer.offer_audio_chunk(next_chunk) is True
        finally:
            release.set()
        await control.wait()
        await finishing
        with pytest.raises(PersistenceError):
            await writer.wait_for_audio_commit(next_chunk.operation_id)
        assert writer.offer_audio_chunk(next_chunk) is False
        assert counters(tmp_path / "queued-seal.sqlite") == (0, 8000)
        assert await writer.quick_check() and not writer.is_degraded


@pytest.mark.asyncio
async def test_terminal_gc_preserves_unknown_ack_then_collects_exact_eligible_slots(tmp_path):
    from tests.contract.test_postgres_sink import sink_with_rows

    path = tmp_path / "terminal-gc.sqlite"
    clock = [NOW]
    token = UUID(int=900)
    sink, pool, connection, _factory = sink_with_rows([])
    try:
        async with owned(path, contract_version=2, utcnow=lambda: clock[0]) as (writer, keyring):
            await seed_admission(writer)
            await authenticate_audio(writer)
            await writer.assert_sparra_compatible()
            first = chunk(keyring)
            assert writer.offer_audio_chunk(first)
            await writer.wait_for_audio_commit(first.operation_id)
            await writer.publish_audio_terminal_v2(terminal(), generation=GENERATION)

            async def known_sink_acks():
                claimed = await writer.read_relay_batch(
                    batch_size=100, now=clock[0], lease_seconds=30
                )
                for item in claimed:
                    candidate = item.operation
                    connection.rows = [({"schema_version": 2, "status": "applied",
                        "operation_id": str(candidate.operation_id),
                        "payload_sha256": hashlib.sha256(
                            canonical_operation_bytes(candidate)
                        ).hexdigest()},)]
                    await sink.ingest_v2(candidate)
                    assert (await writer.ack_outbox(queue_id=item.queue_id,
                        expected_claim_attempt=item.claim_attempt)).applied

            await known_sink_acks()
            assert terminal_slot(path, "audio.finish")[5] == 1
            await writer.publish_audio_terminal_v2(terminal(revoke=True, operation_id=2001),
                                                   generation=GENERATION)
            assert terminal_slot(path, "audio.revoke")[5] == 0
            clock[0] = DEADLINE + timedelta(seconds=901)
            await writer.cleanup_local_state(now=clock[0])
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT count(*) FROM local_audio_terminal").fetchone()[0] == 2
                assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 1
            # Real sink codec validates the controlled known commit before exact local ACK.
            await known_sink_acks()
            assert terminal_slot(path, "audio.revoke")[5] == 1
            end = NOW + timedelta(seconds=10)
            ticket = writer.submit_webhook(
                receipt={"event_id": "terminal-original-hangup", "event_type": "call.hangup",
                    "call_control_id": "control-a", "call_leg_id": "leg-a",
                    "call_session_id": "session-a", "occurred_at": end, "received_at": clock[0],
                    "semantic_fingerprint_sha256": b"g" * 32},
                lease={"action": "upsert", "call_control_id": "control-a", "call_id": CALL,
                    "tenant_id": str(WORKSPACE), "agent_id": "agent-a", "state": "terminal",
                    "token_hash": b"b" * 32, "created_at": NOW,
                    "expires_at": NOW + timedelta(hours=1), "closed_at": end},
                operation=None,
            )
            await ticket.wait()
            cleaned = await writer.erase_call_content(CALL, lease_token=token, now=clock[0])
            assert cleaned is not None
            await writer.cleanup_local_state(now=clock[0])
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT count(*) FROM local_audio_terminal").fetchone()[0] == 2
                assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 1
            connection.rows = [(None,)]
            await sink.ack_call_erasure(CALL, token, cleaned)
            await writer.finish_erasure_ack(CALL, token, acknowledged=True)
            await writer.cleanup_local_state(now=clock[0])
            assert await writer.read_call_lifecycle(CALL) is None
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 0
                assert db.execute("SELECT count(*) FROM local_audio_terminal").fetchone()[0] == 0
            assert pool.active == 0 and connection.transaction_commits >= 6
            assert await writer.quick_check() and not writer.is_degraded
    finally:
        await sink.close()
