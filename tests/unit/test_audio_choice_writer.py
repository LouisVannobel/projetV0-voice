"""Native durable choice witnesses; historical V6 data is not caller consent."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta
from uuid import UUID

import pytest
from test_audio_writer import (
    CALL,
    DEADLINE,
    GENERATION,
    KEY,
    NOW,
    RECORDING,
    WORKSPACE,
    active_control_v2,
    chunk,
    owned,
    seed_admission,
    snapshot,
)

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.commands import PersistenceError, encrypt_audio_operation
from projetv0_voice.persistence.schema import LOCAL_AUDIO_SCHEMA_SQL, LOCAL_AUDIO_SCHEMA_VERSION
from projetv0_voice.persistence.writer import LocalCallLifecycleFacts, PersistenceWriter


def historical_v6(path):
    """Real qualified old DDL and AEAD, with no invented accepted/ready row."""
    assert LOCAL_AUDIO_SCHEMA_VERSION == 6
    prepared = encrypt_audio_operation(chunk(CryptoKeyring({1: KEY}, active_version=1)),
                                       CryptoKeyring({1: KEY}, active_version=1))
    facts = LocalCallLifecycleFacts(CALL, NOW, DEADLINE, "leg-a", "session-a",
                                   admission_generation=GENERATION)

    def wire(value):
        if isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        return str(value)

    with sqlite3.connect(path) as db:
        db.executescript(LOCAL_AUDIO_SCHEMA_SQL)
        db.execute(
            "INSERT INTO call_leases(call_control_id,call_id,tenant_id,agent_id,state,token_hash,"
            "created_at,expires_at,lifecycle_json) VALUES(?,?,?,?,?,?,?,?,?)",
            ("control-a", str(CALL), str(WORKSPACE), "agent-a", "pending", b"b" * 32,
             wire(NOW), wire(NOW + timedelta(hours=1)), json.dumps(asdict(facts), default=wire)),
        )
        db.execute(
            "INSERT INTO local_audio_pin(call_id,generation,workspace_id,deployment_id,"
            "recording_id,"
            "configuration_revision,recording_policy,audio_available,admitted_at,retention_until) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (str(CALL), str(GENERATION), str(WORKSPACE), "agent-a", str(RECORDING), 7,
             "local_30d", 1, wire(NOW), wire(DEADLINE)),
        )
        db.execute(
            "INSERT INTO outbox(op_id,deployment_id,kind,schema_version,call_id,recording_id,"
            "crypto_version,key_version,nonce,ciphertext,created_at,next_attempt_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(prepared.operation.operation_id), "agent-a", "audio.chunk", 2, str(CALL),
             str(RECORDING), 1, prepared.encrypted.key_version, prepared.encrypted.nonce,
             prepared.encrypted.ciphertext, wire(NOW), wire(NOW)),
        )


async def wait_for_queued(writer):
    queued = asyncio.Event()
    loop = asyncio.get_running_loop()
    handle = None

    def inspect_queue():
        nonlocal handle
        if writer.queue_size == 1:
            queued.set()
        else:
            handle = loop.call_soon(inspect_queue)

    handle = loop.call_soon(inspect_queue)
    try:
        await asyncio.wait_for(queued.wait(), 1)
    finally:
        handle.cancel()


@pytest.mark.asyncio
async def test_choice_exact_v6_upgrade_preserves_cipher_and_refuses_unknown_provenance(tmp_path):
    path = tmp_path / "history.sqlite"
    historical_v6(path)
    before = path.read_bytes()
    with sqlite3.connect(path) as db:
        original = db.execute("SELECT op_id,key_version,nonce,ciphertext FROM outbox").fetchone()
    async with owned(path, contract_version=2) as (writer, _keyring):
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 7
            assert db.execute(
                "SELECT op_id,key_version,nonce,ciphertext FROM outbox"
            ).fetchone() == original
            assert db.execute(
                "SELECT generation,retention_until,choice_state,choice_occurred_at "
                "FROM local_audio_pin"
            ).fetchone() == (str(GENERATION), DEADLINE.isoformat().replace("+00:00", "Z"),
                            "undecided", None)
        assert writer.offer_audio_chunk(chunk(_keyring)) is False
    upgraded = path.read_bytes()
    for label, data, mode in (("v1-six", before, 1), ("v1-seven", upgraded, 1),
                              ("altered-six", before, 2), ("unknown-version", before, 2)):
        candidate = tmp_path / f"{label}.sqlite"
        candidate.write_bytes(data)
        if label in {"altered-six", "unknown-version"}:
            with sqlite3.connect(candidate) as db:
                db.execute("CREATE TABLE owned_unexpected(value INTEGER)" if label == "altered-six"
                           else "PRAGMA user_version=1337")
        frozen = candidate.read_bytes()
        writer = PersistenceWriter(candidate, CryptoKeyring({1: KEY}, active_version=1),
                                   contract_version=mode)
        task = asyncio.create_task(writer.run())
        assert await writer.wait_ready() is False
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        assert candidate.read_bytes() == frozen
    failed = tmp_path / "failed-upgrade.sqlite"
    failed.write_bytes(before)

    def fail_migration(name):
        if name == "after_audio_choice_migration_before_commit":
            raise RuntimeError("owned-migration-fault")

    writer = PersistenceWriter(failed, CryptoKeyring({1: KEY}, active_version=1),
                               contract_version=2, failpoint=fail_migration)
    task = asyncio.create_task(writer.run())
    ready = await writer.wait_ready()
    if ready:
        await writer.drain(2)
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert ready is False and failed.read_bytes() == before
    async with owned(failed, contract_version=2) as (_writer, _keyring):
        with sqlite3.connect(failed) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 7


@pytest.mark.asyncio
async def test_choice_and_gate_each_require_actual_commit_before_native_chunk_admission(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(
        tmp_path / "fresh.sqlite", contract_version=2,
        utcnow=lambda: NOW + timedelta(seconds=3), failpoint=hold,
    ) as (writer, keyring):
        await seed_admission(writer)
        operation = chunk(keyring)
        assert writer.offer_audio_chunk(operation) is False
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        assert writer.offer_audio_chunk(operation) is False
        armed = True
        choosing = asyncio.create_task(writer.commit_audio_choice(
            CALL, generation=GENERATION, choice="accept", occurred_at=NOW + timedelta(seconds=3)
        ))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not choosing.done() and writer.offer_audio_chunk(operation) is False
        finally:
            release.set()
        accepted = await choosing
        assert type(accepted).__name__ == "AudioChoiceFacts"
        assert accepted.call_id == CALL and accepted.generation == GENERATION
        assert accepted.choice_state == "accepted" and accepted.denied_at is None
        assert writer.offer_audio_chunk(operation) is False
        entered.clear()
        release.clear()
        armed = True
        gating = asyncio.create_task(writer.publish_control_v2(active_control_v2(gate=True),
                                                                generation=GENERATION))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not gating.done() and writer.offer_audio_chunk(operation) is False
        finally:
            release.set()
        await gating
        assert writer.offer_audio_chunk(operation) is True
        await writer.wait_for_audio_commit(operation.operation_id)
        assert not writer.is_degraded and await writer.quick_check()


@pytest.mark.asyncio
async def test_choice_timestamp_window_replay_and_immutable_timeout_off(tmp_path):
    clock = [NOW + timedelta(seconds=3)]
    async with owned(tmp_path / "invalid.sqlite", contract_version=2,
                     utcnow=lambda: clock[0]) as (writer, _keyring):
        await seed_admission(writer)
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        for timestamp in (None, NOW + timedelta(seconds=1), NOW + timedelta(seconds=4)):
            with pytest.raises(PersistenceError):
                await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                                 occurred_at=timestamp)
            assert not writer.is_degraded
        clock[0] = NOW + timedelta(seconds=7)
        with pytest.raises(PersistenceError):
            await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                             occurred_at=NOW + timedelta(seconds=3))
        off = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="off",
                                               occurred_at=None)
        assert off.choice_state == "off" and off.choice_occurred_at is None
        assert not writer.is_degraded and await writer.quick_check()
    async with owned(tmp_path / "valid.sqlite", contract_version=2,
                     utcnow=lambda: NOW + timedelta(seconds=4)) as (writer, _keyring):
        await seed_admission(writer)
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        accepted = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                                    occurred_at=NOW + timedelta(seconds=3))
        replayed = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                                    occurred_at=NOW + timedelta(seconds=3))
        assert accepted == replayed and accepted.choice_state == "accepted"
        with pytest.raises(PersistenceError):
            await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                             occurred_at=NOW + timedelta(seconds=4))
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_choice_preannouncement_off_and_queued_later_denial_survive_restart(tmp_path):
    async with owned(tmp_path / "early-off.sqlite", contract_version=2,
                     utcnow=lambda: NOW + timedelta(seconds=3)) as (writer, keyring):
        await seed_admission(writer)
        off = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="off",
                                               occurred_at=None)
        assert off.choice_state == "off"
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        with pytest.raises(PersistenceError):
            await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                             occurred_at=NOW + timedelta(seconds=3))
        assert writer.offer_audio_chunk(chunk(keyring)) is False
        assert not (await writer.read_call_lifecycle(CALL)).content_erased
    path = tmp_path / "later-off.sqlite"
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(path, contract_version=2, utcnow=lambda: NOW + timedelta(seconds=3),
                     failpoint=hold) as (writer, keyring):
        await seed_admission(writer)
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        accepted = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                                    occurred_at=NOW + timedelta(seconds=3))
        await writer.publish_control_v2(active_control_v2(gate=True), generation=GENERATION)
        armed = True
        control = writer.submit_webhook(receipt={"event_id": "held-choice-control",
            "event_type": "fixture.control", "call_control_id": None, "occurred_at": NOW,
            "received_at": NOW, "semantic_fingerprint_sha256": b"q" * 32},
            lease=None, operation=None)
        await asyncio.wait_for(entered.wait(), 1)
        denying = asyncio.create_task(writer.commit_audio_choice(CALL, generation=GENERATION,
                                                                 choice="off", occurred_at=None))
        try:
            await wait_for_queued(writer)
            operation = chunk(keyring)
            assert writer.offer_audio_chunk(operation) is True
        finally:
            release.set()
        await control.wait()
        denied = await denying
        assert denied.choice_state == "off"
        assert denied.choice_occurred_at == accepted.choice_occurred_at
        assert denied.denied_at is not None
        with pytest.raises(PersistenceError):
            await writer.wait_for_audio_commit(operation.operation_id)
        assert writer.offer_audio_chunk(operation) is False
        facts = await writer.read_call_lifecycle(CALL)
        assert not facts.content_erased and facts.original_ended_at is None
        assert facts.disclosure_evidence.input_gate_opened_at is not None
        assert not writer.is_degraded and await writer.quick_check()
    async with owned(path, contract_version=2, utcnow=lambda: NOW + timedelta(seconds=3)) as (
        writer, keyring,
    ):
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        assert writer.offer_audio_chunk(chunk(keyring)) is False
        denied = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="off",
                                                  occurred_at=None)
        assert denied.choice_state == "off" and denied.choice_occurred_at is not None
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_choice_cancelled_commit_is_not_optimistic_and_queued_erase_wins(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with owned(
        tmp_path / "cancellation.sqlite", contract_version=2,
        utcnow=lambda: NOW + timedelta(seconds=3), failpoint=hold,
    ) as (writer, keyring):
        await seed_admission(writer)
        await writer.publish_control_v2(active_control_v2(), generation=GENERATION)
        with pytest.raises(PersistenceError):
            await writer.commit_audio_choice(CALL, generation=UUID(int=999), choice="accept",
                                             occurred_at=NOW + timedelta(seconds=3))
        armed = True
        choosing = asyncio.create_task(writer.commit_audio_choice(CALL, generation=GENERATION,
            choice="accept", occurred_at=NOW + timedelta(seconds=3)))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            choosing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await choosing
            assert writer.offer_audio_chunk(chunk(keyring)) is False
        finally:
            release.set()
        await writer.wait_until_idle()
        accepted = await writer.commit_audio_choice(CALL, generation=GENERATION, choice="accept",
                                                    occurred_at=NOW + timedelta(seconds=3))
        assert accepted.choice_state == "accepted"
        await writer.publish_control_v2(active_control_v2(gate=True), generation=GENERATION)
        entered.clear()
        release.clear()
        armed = True
        control = writer.submit_webhook(receipt={"event_id": "held-erase-control",
            "event_type": "fixture.control", "call_control_id": None, "occurred_at": NOW,
            "received_at": NOW, "semantic_fingerprint_sha256": b"r" * 32},
            lease=None, operation=None)
        await asyncio.wait_for(entered.wait(), 1)
        erasing = asyncio.create_task(
            writer.erase_call_content(CALL, now=NOW + timedelta(seconds=3))
        )
        try:
            await wait_for_queued(writer)
            operation = chunk(keyring)
            assert writer.offer_audio_chunk(operation) is True
        finally:
            release.set()
        await control.wait()
        await erasing
        with pytest.raises(PersistenceError):
            await writer.wait_for_audio_commit(operation.operation_id)
        assert writer.offer_audio_chunk(operation) is False
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
        assert not writer.is_degraded and await writer.quick_check()
