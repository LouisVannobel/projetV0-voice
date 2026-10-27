"""Native capture/SQLite and real sink codec; pool receipts are controlled test data."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3

import pytest
from pipecat.frames.frames import EndFrame

from projetv0_voice import audio_capture
from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.persistence.commands import (
    canonical_operation_bytes,
    decode_operation_v2,
    operation_aad_from_metadata,
)
from projetv0_voice.persistence.relay import OutboxRelay
from projetv0_voice.persistence.writer import PersistenceWriter
from tests.contract.test_postgres_sink import sink_with_rows
from tests.integration.test_disclosure import _local_choice_wait
from tests.integration.test_local_audio_capture import (
    CALL,
    GENERATION,
    NOW,
    accept_local,
    caller_frames,
    capture_case,
)


def terminal_row(case, kind):
    with sqlite3.connect(case.path) as db:
        return db.execute(
            "SELECT op_id,deployment_id,key_version,nonce,ciphertext,acked "
            "FROM local_audio_terminal WHERE kind=?", (kind,),
        ).fetchone()


def terminal_operation(case, kind):
    row = terminal_row(case, kind)
    assert row is not None
    plaintext = case.keyring.decrypt(
        EncryptedValue(*row[2:5]), aad=operation_aad_from_metadata({
            "schema_version": 2, "operation_id": row[0], "deployment_id": row[1],
            "call_id": str(CALL), "kind": kind,
        }),
    )
    return decode_operation_v2(plaintext)


async def relay_known(case, monkeypatch):
    sink, pool, connection, _factory = sink_with_rows([])
    native_ingest = sink.ingest_v2
    delivered = []

    async def ingest(operation):
        connection.rows = [({
            "schema_version": 2, "status": "applied",
            "operation_id": str(operation.operation_id),
            "payload_sha256": hashlib.sha256(canonical_operation_bytes(operation)).hexdigest(),
        },)]
        await native_ingest(operation)
        delivered.append(operation)

    async def unexpected():
        pytest.fail("native terminal relay degraded or drained the phone")

    monkeypatch.setattr(sink, "ingest_v2", ingest)
    relay = OutboxRelay(case.writer, sink, utcnow=lambda: NOW,
                        on_degraded=unexpected, drain=unexpected)
    try:
        result = await relay.run_once()
        assert result.status == "delivered" and result.acked == len(delivered)
        assert pool.active == 0 and connection.transaction_commits == len(delivered)
        assert all(call[0] == "SELECT voice.ingest_operation_v2(%s::jsonb)"
                   for call in connection.calls)
        return delivered
    finally:
        await sink.close()


@pytest.mark.asyncio
async def test_capture_terminal_native_normal_tail_exact_finish_relay_and_restart(
    tmp_path, monkeypatch,
):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        await case.runtime.worker.queue_frame(EndFrame())
        await _local_choice_wait(lambda: case.capture.tap.state == "stopped")
        summary = await case.capture.finish("complete")
        assert not summary.pending and not summary.partial
        assert summary.committed_samples == 80 and summary.last_sequence == 0
        assert not case.controller.collection_audio_quiesced
        await case.controller.cleanup_termination("closed")
        assert case.controller.collection_audio_quiesced
        operation = terminal_operation(case, "audio.finish")
        assert (operation.payload.reason, operation.payload.last_sequence,
                operation.payload.total_samples) == ("complete", 0, 80)
        frozen = terminal_row(case, "audio.finish")
        delivered = await relay_known(case, monkeypatch)
        assert [item for item in delivered if item.kind == "audio.finish"] == [operation]
        assert terminal_row(case, "audio.finish") == (*frozen[:5], 1)
        await case.capture.finish("failure")
        assert terminal_row(case, "audio.finish") == (*frozen[:5], 1)
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None
        path, keyring = case.path, case.keyring
    writer = PersistenceWriter(
        path, keyring, contract_version=2, utcnow=lambda: NOW,
        process_agent_id="capture-deploy", process_deployment_id="capture-deploy",
    )
    task = asyncio.create_task(writer.run())
    try:
        assert await writer.wait_ready()
        await writer.publish_audio_terminal_v2(operation, generation=GENERATION)
        with sqlite3.connect(path) as db:
            assert db.execute(
                "SELECT op_id,deployment_id,key_version,nonce,ciphertext,acked "
                "FROM local_audio_terminal WHERE kind='audio.finish'"
            ).fetchone() == (*frozen[:5], 1)
            assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
    finally:
        await writer.drain(2)
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_capture_terminal_fixed_outcome_reasons_and_native_limit_keep_phone_active(
    tmp_path, monkeypatch,
):
    for reason in ("transfer", "interrupted", "failure"):
        async with capture_case(tmp_path / reason, terminal_publication=True) as case:
            await accept_local(case)
            await caller_frames(case, 1, samples=80)
            summary = await case.capture.finish(reason)
            assert not summary.pending and summary.committed_samples == 80
            assert terminal_operation(case, "audio.finish").payload.reason == reason
            await case.capture.finish("complete")
            assert terminal_operation(case, "audio.finish").payload.reason == reason
            assert case.controller.is_active()
    # Reduced native sample ceiling is an engineering boundary, not ten minutes of capture.
    monkeypatch.setattr(audio_capture, "_MAX_SAMPLES", 80)
    async with capture_case(tmp_path / "limit", terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        summary = await case.capture.finish("complete")
        assert summary.reason == "limit" and summary.committed_samples == 80
        finished = terminal_operation(case, "audio.finish")
        assert finished.payload.reason == "limit" and case.controller.is_active()
        frame = await case.serializer.deserialize(json.dumps({
            "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
            "occurred_at": NOW.isoformat().replace("+00:00", "Z"), "dtmf": {"digit": "2"},
        }))
        await case.runtime.worker.queue_frame(frame)
        await _local_choice_wait(lambda: terminal_row(case, "audio.revoke") is not None)
        revoked = terminal_operation(case, "audio.revoke")
        assert revoked.operation_id != finished.operation_id and case.controller.is_active()
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_capture_terminal_caller_two_revoke_retains_unknown_audio_receipt(
    tmp_path, monkeypatch,
):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        native_wait = case.writer.wait_for_audio_commit

        async def unknown(_operation_id):
            raise OSError("owned-unknown-capture-receipt")

        monkeypatch.setattr(case.writer, "wait_for_audio_commit", unknown)
        first = await case.capture.quiesce()
        assert first.pending and first.committed_samples == 0
        await case.writer.wait_until_idle()
        facts = await case.writer.read_call_lifecycle(CALL)
        frame = await case.serializer.deserialize(json.dumps({
            "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
            "occurred_at": NOW.isoformat().replace("+00:00", "Z"), "dtmf": {"digit": "2"},
        }))
        await case.runtime.worker.queue_frame(frame)
        await _local_choice_wait(lambda: terminal_row(case, "audio.revoke") is not None)
        assert not case.capture.holder.ready_for_native_event()
        assert not case.capture.holder.offer_native_pcm(b"\x11\x00" * 160, 8000, 2)
        summary = await case.capture.revoke()
        assert summary.pending and summary.partial
        assert summary.last_sequence is None and summary.committed_samples == 0
        frozen = terminal_row(case, "audio.revoke")
        assert terminal_operation(case, "audio.revoke").payload.reason == "caller_declined"
        assert await case.writer.read_call_lifecycle(CALL) == facts
        assert case.controller.is_active()
        await relay_known(case, monkeypatch)
        await case.capture.revoke()
        assert terminal_row(case, "audio.revoke") == (*frozen[:5], 1)
        monkeypatch.setattr(case.writer, "wait_for_audio_commit", native_wait)
        settled = await case.capture.quiesce()
        assert not settled.pending and settled.committed_samples == 80


@pytest.mark.asyncio
async def test_capture_terminal_cancelled_commit_freezes_first_body_until_exact_replay(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with capture_case(tmp_path, failpoint=hold, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        settled = await case.capture.quiesce()
        assert not settled.pending and settled.committed_samples == 80
        armed = True
        finishing = asyncio.create_task(case.capture.finish("complete"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert terminal_row(case, "audio.finish") is None
            finishing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await finishing
            assert terminal_row(case, "audio.finish") is None
        finally:
            release.set()
        await case.writer.wait_until_idle()
        frozen = terminal_row(case, "audio.finish")
        assert frozen is not None and frozen[5] == 0
        repeated = await case.capture.finish("failure")
        assert not repeated.pending and repeated.committed_samples == 80
        assert terminal_row(case, "audio.finish") == frozen
        assert terminal_operation(case, "audio.finish").payload.reason == "complete"
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT count(*) FROM local_audio_terminal").fetchone()[0] == 1
        assert await case.writer.quick_check() and not case.writer.is_degraded


@pytest.mark.asyncio
async def test_capture_terminal_superseded_finish_revoke_ack_retires_terminal_pending(
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

    async with capture_case(tmp_path, failpoint=hold, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        assert not (await case.capture.quiesce()).pending
        armed = True
        finishing = asyncio.create_task(case.capture.finish("complete"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            finishing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await finishing
        finally:
            release.set()
        await case.writer.wait_until_idle()
        frozen_finish = terminal_row(case, "audio.finish")
        assert frozen_finish is not None and frozen_finish[5] == 0
        frame = await case.serializer.deserialize(json.dumps({
            "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
            "occurred_at": NOW.isoformat().replace("+00:00", "Z"), "dtmf": {"digit": "2"},
        }))
        await case.runtime.worker.queue_frame(frame)
        await _local_choice_wait(lambda: terminal_row(case, "audio.revoke") is not None)
        frozen_revoke = terminal_row(case, "audio.revoke")
        delivered = await relay_known(case, monkeypatch)
        assert [item.kind for item in delivered].count("audio.revoke") == 1
        assert terminal_row(case, "audio.revoke") == (*frozen_revoke[:5], 1)
        settled = await case.capture.quiesce()
        assert not settled.pending and settled.committed_samples == 80
        assert terminal_row(case, "audio.finish") == frozen_finish
        assert terminal_operation(case, "audio.finish").payload.reason == "complete"
        await case.capture.finish("failure")
        await case.capture.revoke()
        assert terminal_row(case, "audio.finish") == frozen_finish
        assert terminal_row(case, "audio.revoke") == (*frozen_revoke[:5], 1)
        assert case.controller.is_active() and not case.writer.is_degraded


@pytest.mark.asyncio
async def test_capture_terminal_default_off_cleanup_has_no_rejected_finish_pending(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    clock = [10.0]
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with capture_case(tmp_path, failpoint=hold, terminal_publication=True,
                            monotonic=lambda: clock[0]) as case:
        armed = True
        try:
            # Hold the genuine disclosure COMMIT so its ACK-owned choice deadline expires.
            await asyncio.wait_for(entered.wait(), 2)
            clock[0] += 5.1
        finally:
            release.set()
        await asyncio.wait_for(case.active.wait(), 2)
        assert case.controller.is_active() and case.capture.tap.state != "recording"
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
        await caller_frames(case, 1, samples=80)
        assert case.stt.received and case.failure.code is None
        await case.controller.cleanup_termination("closed")
        summary = await case.capture.quiesce()
        assert not summary.pending and summary.committed_samples == 0
        assert summary.last_sequence is None and terminal_row(case, "audio.finish") is None
        assert case.failure.code is None and not case.writer.is_degraded
        facts = await case.writer.read_call_lifecycle(CALL)
        assert facts.original_ended_at is None and not facts.content_erased


@pytest.mark.asyncio
async def test_capture_terminal_accepted_external_cancel_preserves_interrupted_finish(tmp_path):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        await case.controller.terminalize_and_join(cancel_continuations=True)
        await case.controller.cleanup_termination("external_cancel")
        assert terminal_row(case, "audio.revoke") is None
        finished = terminal_operation(case, "audio.finish")
        assert (finished.payload.reason, finished.payload.last_sequence,
                finished.payload.total_samples) == ("interrupted", 0, 80)
        summary = await case.capture.quiesce()
        assert summary.partial and not summary.pending and summary.committed_samples == 80
        assert case.failure.code is None and not case.writer.is_degraded
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None

@pytest.mark.asyncio
async def test_collection_session_cleanup_distinguishes_settled_and_captured_pending(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from uuid import uuid4

    from projetv0_voice.calendar import CalendarPlayback
    from projetv0_voice.session import CallSession, TurnRecorder, _TerminalOutcome
    from tests.integration.test_disclosure import _TEST_RUNTIME_METRICS

    def recorder(identity, writer, keyring, failure):
        return TurnRecorder(identity=identity, writer=writer, keyring=keyring,
            first_failure=failure, runtime_metrics=_TEST_RUNTIME_METRICS, utcnow=lambda: NOW)

    async def inference(_context, **_options):
        return '{"result":null}'

    async def close(*_args, **_options):
        return None

    for settled in (False, True):
        async with capture_case(tmp_path / str(settled), terminal_publication=True,
                                recorder_factory=recorder) as case:
            await accept_local(case)
            await caller_frames(case, 1, samples=80)
            await case.runtime.worker.queue_frame(EndFrame())
            await asyncio.wait_for(case.runner, 2)
            if not settled:
                async def pending(_reason):
                    return replace(await case.capture.quiesce(), pending=True)
                case.controller._local_audio_finish = pending
            session = CallSession.__new__(CallSession)
            session._identity = case.controller._identity
            session._writer, session._controller = case.writer, case.controller
            session._utcnow, session._uuid_factory = lambda: NOW, uuid4
            session._terminal_publication = session._partial_result = None
            session._result_inference_task = session._registry_terminalizer = None
            session._metric_lease = session._end_call_playback = None
            session._calendar_task = session._program_action_task = None
            session._calendar_playback, session._action_authority = CalendarPlayback(), None
            session._no_new_ai = session._result_inference_fenced = False
            session._runtime_metrics = _TEST_RUNTIME_METRICS
            session._cleanup_phase_timeout_seconds = 2
            session._lease_terminalizer = SimpleNamespace(terminalize=close)
            session._services = SimpleNamespace(llm=SimpleNamespace(run_inference=inference),
                                                 aclose=close)
            await session._cleanup_owned_state(
                controller=case.controller, recorder=case.turn_recorder,
                first_failure=case.failure, transport=None, pipeline=None, runtime=case.runtime,
                runner_task=case.runner, failure_task=None,
                terminal_outcome=_TerminalOutcome('closed'),
                cancel_continuations=False)
            frozen = await case.writer.read_frozen_call_publication_v2(CALL,generation=GENERATION)
            assert session._collection_quiesced is settled
            if settled:
                assert frozen.payload.collection.audio.total_samples == 80
                assert frozen.payload.collection.result_outcome == 'absent'
            else:
                assert case.failure.code == 'recording_cleanup_failed'
                assert 'collection' not in frozen.payload.model_dump(mode='json')
