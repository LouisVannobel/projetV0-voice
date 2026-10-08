"""Offline V2 snapshot/sink contracts; pool doubles do not prove native SQL."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from datetime import timedelta
from uuid import UUID

import pytest
from psycopg.types.json import Jsonb, JsonbDumper
from pydantic import ValidationError

from projetv0_voice import audio_contract
from projetv0_voice.audio_contract import VoiceOperationV2
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.commands import (
    PersistenceError,
    canonical_operation_bytes,
    encrypt_audio_operation,
)
from projetv0_voice.persistence.postgres_sink import (
    OperationConflictError,
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkErasedError,
    OperationSinkPermanentError,
)
from projetv0_voice.persistence.relay import OutboxRelay
from tests.contract.test_postgres_sink import (
    NOW,
    SqlstateError,
    exception_graph,
    ingest_result,
    operation,
    sink_with_rows,
)
from tests.contract.test_sparra_sink import routing
from tests.contract.test_sparra_sink import snapshot as legacy_snapshot

CALL_ID = UUID(int=2)
WORKSPACE_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
RECORDING_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
RETENTION = (NOW + timedelta(days=30)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def snapshot(**updates):
    return {
        **legacy_snapshot(),
        "schema_version": 2,
        "workspace_id": str(WORKSPACE_ID),
        "retention_until": RETENTION,
        "recording_policy": "local_30d",
        "recording_contact_phone": "+33123456789",
        "audio_available": True,
        "recording_id": str(RECORDING_ID),
        **updates,
    }


def snapshot_type():
    assert hasattr(audio_contract, "BeginCallSnapshotV2"), "missing strict BeginCallSnapshotV2"
    return audio_contract.BeginCallSnapshotV2


def audio_operation():
    return VoiceOperationV2.model_validate({
        "schema_version": 2,
        "operation_id": str(UUID(int=1)),
        "deployment_id": "agent-a",
        "call_id": str(CALL_ID),
        "occurred_at": NOW,
        "kind": "audio.revoke",
        "payload": {
            "schema_version": 2,
            "workspace_id": str(WORKSPACE_ID),
            "recording_id": str(RECORDING_ID),
            "configuration_revision": 1,
            "retention_until": RETENTION,
            "reason": "caller_declined",
        },
    })


def receipt(**updates):
    return {**ingest_result(), "schema_version": 2, **updates}


def test_snapshot_v2_distinguishes_off_available_and_quota_unavailable():
    model = snapshot_type()
    for data in [
        snapshot(),
        snapshot(recording_policy="off", recording_contact_phone=None,
                 audio_available=False, recording_id=None),
        snapshot(audio_available=False, recording_id=None),
    ]:
        parsed = model.model_validate(data)
        assert parsed.model_dump(mode="json") == data
        assert parsed.workspace_id == WORKSPACE_ID
        assert parsed.retention_until == NOW + timedelta(days=30)
    maximum = model.model_validate(snapshot(configuration_revision=2_147_483_647))
    assert maximum.configuration_revision == 2_147_483_647


def test_snapshot_v2_rejects_inconsistent_policy_and_noncanonical_fields():
    model = snapshot_type()
    for change in [
        {"schema_version": True}, {"schema_version": 1},
        {"workspace_id": str(WORKSPACE_ID).upper()}, {"workspace_id": "foreign"},
        {"configuration_revision": True}, {"configuration_revision": 2_147_483_648},
        {"recording_policy": "telnyx_dual"}, {"recording_contact_phone": None},
        {"recording_contact_phone": "0033123456789"},
        {"audio_available": "true"}, {"audio_available": False},
        {"recording_id": None}, {"recording_policy": "off"},
        {"retention_until": (NOW + timedelta(days=30)).isoformat()},
        {"retention_until": NOW + timedelta(days=30, microseconds=1)},
        {"recording_enabled": True}, {"extra": 1},
    ]:
        with pytest.raises(ValidationError):
            model.model_validate(snapshot(**change))
    missing = snapshot()
    missing.pop("workspace_id")
    with pytest.raises(ValidationError):
        model.model_validate(missing)


@pytest.mark.asyncio
async def test_begin_v2_uses_fixed_sql_and_existing_pool_without_changing_v1():
    sink, pool, connection, factory = sink_with_rows([(snapshot(),)])
    assert callable(getattr(sink, "begin_call_v2", None)), "missing fixed begin_call_v2"
    admitted = routing()
    value = await sink.begin_call_v2("agent-a", CALL_ID, admitted)
    assert value.model_dump(mode="json") == snapshot()
    sql, params, prepare = connection.calls[0]
    assert sql == "SELECT voice.begin_call_v2(%s,%s,%s::jsonb)"
    assert params[:2] == ("agent-a", CALL_ID)
    assert isinstance(params[2], Jsonb) and params[2].obj == admitted.model_dump(mode="json")
    assert prepare is False and connection.transaction_commits == 1 and pool.active == 0
    connection.rows = [(legacy_snapshot(),)]
    assert (await sink.begin_call("agent-a", CALL_ID, admitted)).schema_version == 1
    connection.rows = [(ingest_result(),)]
    await sink.ingest(operation())
    assert [call[0] for call in connection.calls] == [
        "SELECT voice.begin_call_v2(%s,%s,%s::jsonb)",
        "SELECT voice.begin_call_v1(%s,%s,%s::jsonb)",
        "SELECT voice.ingest_operation_v1(%s::jsonb)",
    ]
    assert len(factory.calls) == 1 and pool.active == 0


@pytest.mark.asyncio
async def test_begin_v2_rejects_reply_shape_identity_workspace_and_changed_expiry():
    for rows in [
        [], [(None,)], [(snapshot(), "extra")], [(snapshot(),), (snapshot(),)],
        [(snapshot(call_id=str(UUID(int=99))),)],
        [(snapshot(workspace_id="foreign"),)],
        [(snapshot(retention_until=(NOW + timedelta(days=31)).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")),)],
        [(snapshot(schema_version=1),)], [(snapshot(extra=1),)],
    ]:
        sink, pool, connection, _ = sink_with_rows(rows)
        assert callable(getattr(sink, "begin_call_v2", None)), "missing fixed begin_call_v2"
        with pytest.raises(OperationSinkContractError):
            await sink.begin_call_v2("agent-a", CALL_ID, routing())
        assert connection.transaction_rollbacks == 1 and pool.active == 0
        assert len(connection.calls) == 1


@pytest.mark.asyncio
async def test_begin_v2_input_bounds_refuse_before_sql_and_preserve_unicode_ids():
    sink, _, connection, _ = sink_with_rows([(snapshot(),)])
    assert callable(getattr(sink, "begin_call_v2", None)), "missing fixed begin_call_v2"
    for arguments in [
        ("x" * 257, CALL_ID, routing()), ("bad\nid", CALL_ID, routing()),
        ("agent-a", str(CALL_ID), routing()), ("agent-a", CALL_ID, {}),
    ]:
        with pytest.raises(ValueError):
            await sink.begin_call_v2(*arguments)
    assert connection.calls == []
    await sink.begin_call_v2("🚙" * 256, CALL_ID, routing())
    assert connection.calls[0][1][0] == "🚙" * 256


@pytest.mark.asyncio
async def test_ingest_v2_fixed_sql_matches_real_codec_and_three_receipt_outcomes():
    candidate = audio_operation()
    prepared = encrypt_audio_operation(
        candidate, CryptoKeyring({1: bytes(range(32))}, active_version=1)
    )
    for status in ["applied", "duplicate", "conflict"]:
        sink, pool, connection, factory = sink_with_rows([(receipt(status=status),)])
        assert callable(getattr(sink, "ingest_v2", None)), "missing fixed ingest_v2"
        if status == "conflict":
            with pytest.raises(OperationConflictError):
                await sink.ingest_v2(candidate)
            assert connection.transaction_rollbacks == 1
        else:
            assert await sink.ingest_v2(candidate) is None
            assert connection.transaction_commits == 1
        sql, params, prepare = connection.calls[0]
        assert sql == "SELECT voice.ingest_operation_v2(%s::jsonb)" and prepare is False
        assert len(params) == 1 and isinstance(params[0], Jsonb)
        sent = JsonbDumper(Jsonb).dump(params[0])
        assert sent == prepared.plaintext
        assert len(connection.calls) == 1 and len(factory.calls) == 1 and pool.active == 0


@pytest.mark.asyncio
async def test_ingest_v2_rejects_wrong_version_type_and_exact_receipt_keys():
    for rows in [
        [], [(receipt(), "extra")], [(receipt(),), (receipt(),)],
        [(receipt(schema_version=1),)], [(receipt(schema_version=True),)],
        [(receipt(status="unknown"),)], [(receipt(operation_id=str(UUID(int=99))),)],
        [(receipt(payload_sha256="A" * 64),)], [(receipt(extra=True),)],
    ]:
        sink, pool, connection, _ = sink_with_rows(rows)
        assert callable(getattr(sink, "ingest_v2", None)), "missing fixed ingest_v2"
        with pytest.raises(OperationSinkContractError):
            await sink.ingest_v2(audio_operation())
        assert connection.transaction_rollbacks == 1 and pool.active == 0
        assert len(connection.calls) == 1
    sink, _, connection, _ = sink_with_rows([])
    assert callable(getattr(sink, "ingest_v2", None)), "missing fixed ingest_v2"
    with pytest.raises(ValueError):
        await sink.ingest_v2(operation())
    assert connection.calls == []


@pytest.mark.asyncio
async def test_v2_rpc_errors_keep_existing_safe_terminal_classification():
    for method, state, expected, args in [
        ("begin_call_v2", "PV202", OperationSinkPermanentError, ("agent-a", CALL_ID, routing())),
        ("ingest_v2", "PV301", OperationSinkErasedError, (audio_operation(),)),
    ]:
        sink, pool, connection, _ = sink_with_rows(
            [], execute_error=SqlstateError(state, "RAW-V2-DB-SENTINEL")
        )
        action = getattr(sink, method, None)
        assert callable(action), "missing fixed V2 sink method"
        with pytest.raises(expected) as captured:
            await action(*args)
        assert "RAW-V2-DB-SENTINEL" not in exception_graph(captured.value)
        assert connection.transaction_rollbacks == 1 and pool.active == 0
        assert len(connection.calls) == 1


@pytest.mark.asyncio
async def test_begin_v2_unknown_commit_preserves_exact_explicit_retry_identity():
    sink, pool, connection, _ = sink_with_rows(
        [(snapshot(),)], commit_error=OSError("RAW-V2-COMMIT-SENTINEL")
    )
    assert callable(getattr(sink, "begin_call_v2", None)), "missing fixed begin_call_v2"
    admitted = routing()
    with pytest.raises(OperationSinkCommitAmbiguousError) as captured:
        await sink.begin_call_v2("agent-a", CALL_ID, admitted)
    assert "RAW-V2-COMMIT-SENTINEL" not in exception_graph(captured.value)
    assert len(connection.calls) == 1 and pool.active == 0
    connection.commit_error = None
    replayed = await sink.begin_call_v2("agent-a", CALL_ID, admitted)
    assert replayed.recording_id == RECORDING_ID
    assert connection.calls[0][1][:2] == connection.calls[1][1][:2]
    assert connection.calls[0][1][2].obj == connection.calls[1][1][2].obj
    assert all(
        call[0] == "SELECT voice.begin_call_v2(%s,%s,%s::jsonb)" for call in connection.calls
    )


@pytest.mark.asyncio
async def test_v2_cancellation_rolls_back_returns_pool_and_allows_close():
    for method, rows, args in [
        ("begin_call_v2", [(snapshot(),)], ("agent-a", CALL_ID, routing())),
        ("ingest_v2", [(receipt(),)], (audio_operation(),)),
    ]:
        sink, pool, connection, _ = sink_with_rows(rows, execute_gate=asyncio.Event())
        action = getattr(sink, method, None)
        assert callable(action), "missing fixed V2 sink method"
        task = asyncio.create_task(action(*args))
        await asyncio.sleep(0)
        assert len(connection.calls) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert connection.transaction_exit_types == [asyncio.CancelledError]
        assert connection.transaction_rollbacks == 1 and pool.active == 0
        await sink.close()
        assert len(pool.close_calls) == 1


@pytest.mark.asyncio
async def test_fixed_v2_restart_replays_identical_cipher_and_real_sink_codec_before_ack(
    tmp_path, monkeypatch,
):
    from tests.unit.test_audio_writer import NOW as WRITER_NOW
    from tests.unit.test_audio_writer import authenticate_audio, chunk, owned, seed_admission

    path = tmp_path / "voice.sqlite"
    clock = [WRITER_NOW]
    async with owned(path, contract_version=2, utcnow=lambda: clock[0]) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        operation = chunk(keyring)
        assert writer.offer_audio_chunk(operation)
        await writer.wait_for_audio_commit(operation.operation_id)
    with sqlite3.connect(path) as db:
        frozen = db.execute(
            "SELECT op_id,created_at,key_version,nonce,ciphertext FROM outbox "
            "WHERE kind='audio.chunk'"
        ).fetchone()
    result = {
        "schema_version": 2, "status": "applied", "operation_id": str(operation.operation_id),
        "payload_sha256": hashlib.sha256(canonical_operation_bytes(operation)).hexdigest(),
    }
    sink, pool, connection, _factory = sink_with_rows(
        [(result,)], commit_error=OSError("owned-v2-response-lost")
    )
    original_execute = connection.execute
    audio_attempts = 0

    async def reply_for_operation(sql, params, *, prepare):
        nonlocal audio_attempts
        candidate = VoiceOperationV2.model_validate(params[0].obj)
        is_audio = candidate.kind == "audio.chunk"
        if is_audio:
            audio_attempts += 1
        connection.rows = [({
            "schema_version": 2,
            "status": "duplicate" if audio_attempts > 1 and is_audio else "applied",
            "operation_id": str(candidate.operation_id),
            "payload_sha256": hashlib.sha256(canonical_operation_bytes(candidate)).hexdigest(),
        },)]
        connection.commit_error = (
            OSError("owned-v2-response-lost") if is_audio and audio_attempts == 1 else None
        )
        return await original_execute(sql, params, prepare=prepare)

    monkeypatch.setattr(connection, "execute", reply_for_operation)

    async def unexpected_degradation():
        pytest.fail("unknown V2 commit must retain its durable claim")

    async with owned(path, contract_version=2, utcnow=lambda: clock[0]) as (writer, _keyring):
        relay = OutboxRelay(writer, sink, utcnow=lambda: clock[0],
                            on_degraded=unexpected_degradation, drain=unexpected_degradation)
        with pytest.raises(OperationSinkCommitAmbiguousError):
            await relay.run_once()
        with sqlite3.connect(path) as db:
            assert db.execute(
                "SELECT op_id,created_at,key_version,nonce,ciphertext FROM outbox "
                "WHERE kind='audio.chunk'"
            ).fetchone() == frozen
            claim = db.execute("SELECT attempts,next_attempt_at FROM outbox").fetchone()
            assert claim[0] == 1
        held = await relay.run_once()
        assert held.acked == 0 and len(connection.calls) == 3
        connection.commit_error = None
        connection.rows = [({**result, "status": "duplicate"},)]
        clock[0] = WRITER_NOW + timedelta(seconds=11)
        delivered = await relay.run_once()
        assert delivered.acked == 1 and not writer.is_degraded
        assert len(connection.calls) == 4
        assert all(call[0] == "SELECT voice.ingest_operation_v2(%s::jsonb)"
                   for call in connection.calls)
        assert connection.calls[2][1][0].obj == connection.calls[3][1][0].obj
        assert VoiceOperationV2.model_validate(connection.calls[3][1][0].obj) == operation
        assert await writer.oldest_outbox_created_at() is None
        assert pool.active == 0 and connection.transaction_commits == 3
    await sink.close()


@pytest.mark.asyncio
async def test_audio_expiry_and_erasure_preserve_end_and_ack_before_finite_pin_gc(tmp_path):
    from tests.unit.test_audio_writer import (
        CALL,
        DEADLINE,
        GENERATION,
        WORKSPACE,
        authenticate_audio,
        chunk,
        owned,
        seed_admission,
    )
    from tests.unit.test_audio_writer import NOW as WRITER_NOW
    from tests.unit.test_audio_writer import snapshot as writer_snapshot

    path = tmp_path / "voice.sqlite"
    clock = [WRITER_NOW]
    token = UUID(int=900)
    async with owned(path, contract_version=2, utcnow=lambda: clock[0]) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        await writer.assert_sparra_compatible()
        operation = chunk(keyring)
        assert writer.offer_audio_chunk(operation)
        await writer.wait_for_audio_commit(operation.operation_id)
        held = await writer.read_relay_batch(batch_size=3, now=WRITER_NOW, lease_seconds=30)
        assert tuple(item.operation.kind for item in held) == (
            "call.upsert", "call.upsert", "audio.chunk"
        )
        clock[0] = DEADLINE + timedelta(seconds=901)
        assert writer.offer_audio_chunk(chunk(keyring, 1)) is False
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
        await writer.cleanup_local_state(now=clock[0])
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 1
            assert db.execute(
                "SELECT count(*) FROM outbox WHERE kind='audio.chunk'"
            ).fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 3
        cleaned = await writer.erase_call_content(CALL, lease_token=token, now=clock[0])
        assert cleaned is not None
        assert writer.offer_audio_chunk(operation) is False
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT denied_at FROM local_audio_pin").fetchone()[0] is not None
            assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
        await writer.cleanup_local_state(now=clock[0])
        assert await writer.read_call_lifecycle(CALL) is not None
        end = WRITER_NOW + timedelta(seconds=10)
        ticket = writer.submit_webhook(
            receipt={"event_id": "original-hangup", "event_type": "call.hangup",
                "call_control_id": "control-a", "call_leg_id": "leg-a",
                "call_session_id": "session-a", "occurred_at": end, "received_at": clock[0],
                "semantic_fingerprint_sha256": b"e" * 32},
            lease={"action": "upsert", "call_control_id": "control-a", "call_id": CALL,
                "tenant_id": str(WORKSPACE), "agent_id": "agent-a", "state": "terminal",
                "token_hash": b"b" * 32, "created_at": WRITER_NOW,
                "expires_at": WRITER_NOW + timedelta(hours=1), "closed_at": end},
            operation=None,
        )
        await ticket.wait()
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at == end
        await writer.cleanup_local_state(now=clock[0])
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 1
        # Real sink codec + pool double: an unknown remote ACK is not local ACK evidence.
        sink, _pool, connection, _factory = sink_with_rows(
            [(None,)], commit_error=OSError("owned-erasure-ack-response-lost")
        )
        try:
            with pytest.raises(OperationSinkCommitAmbiguousError):
                await sink.ack_call_erasure(CALL, token, cleaned)
            await writer.cleanup_local_state(now=clock[0])
            assert await writer.read_call_lifecycle(CALL) is not None
            connection.commit_error = None
            await sink.ack_call_erasure(CALL, token, cleaned)
            await writer.finish_erasure_ack(CALL, token, acknowledged=True)
        finally:
            await sink.close()
        await writer.cleanup_local_state(now=clock[0])
        assert await writer.read_call_lifecycle(CALL) is None
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM sparra_content_fences").fetchone()[0] == 0
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(writer_snapshot(), generation=GENERATION)
        assert writer.offer_audio_chunk(operation) is False
        assert await writer.quick_check() and not writer.is_degraded
