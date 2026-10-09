from __future__ import annotations

import asyncio
import base64
import json
from datetime import timedelta
from uuid import uuid4

import pytest

from projetv0_voice.models import CallUpsertPayloadV1, TurnUpsertPayloadV1, VoiceOperationV1
from tests.unit.test_sparra_admission import NOW, committed, event, start


def capture(writer, call_id, number, text="Pouvez-vous me rappeler ?"):
    turn_id = uuid4()
    encrypted = writer._keyring.encrypt(text.encode(), aad=f"turn:{turn_id}".encode())
    return VoiceOperationV1(
        schema_version=1,
        operation_id=uuid4(),
        deployment_id="fixture",
        call_id=call_id,
        occurred_at=NOW,
        kind="turn.upsert",
        payload=TurnUpsertPayloadV1(
            turn_id=turn_id,
            turn_no=number,
            role="user",
            source="stt_final",
            crypto_version=1,
            key_version=encrypted.key_version,
            nonce_b64=base64.b64encode(encrypted.nonce).decode(),
            ciphertext_b64=base64.b64encode(encrypted.ciphertext).decode(),
            started_at=NOW,
            ended_at=NOW,
            interrupted=False,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cutoff_microsecond", [0, 1, 1000, 123456])
async def test_zero_fraction_expiry_is_consumed_before_aged_fifo(tmp_path, cutoff_microsecond):
    import sqlite3

    from projetv0_voice.persistence.relay import OutboxRelay, maintain_call_content
    from projetv0_voice.persistence.writer import PersistenceWriter

    registry, original, worker, _ = await start(tmp_path)
    original._utcnow = lambda: NOW
    await committed(registry, original, event())
    call_id = (await registry.snapshot("original")).call_id
    await committed(registry, original, event("call.hangup"))
    await original.drain(2)
    await worker
    deadline = (NOW + timedelta(days=30)).replace(microsecond=0)
    with sqlite3.connect(original._database_path) as db:
        raw = db.execute("SELECT lifecycle_json FROM call_leases WHERE call_id=?",
                         (str(call_id),)).fetchone()[0]
        facts = json.loads(raw)
        facts["admitted_at"] = NOW.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        facts["retention_until"] = deadline.isoformat().replace("+00:00", "Z")
        db.execute("UPDATE call_leases SET lifecycle_json=? WHERE call_id=?",
                   (json.dumps(facts), str(call_id)))
    clock = [deadline.replace(microsecond=cutoff_microsecond)]
    writer = PersistenceWriter(original._database_path, original._keyring, utcnow=lambda: clock[0])
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    await writer.assert_sparra_compatible()
    stopped = []

    class Sink:
        async def lease_call_erasures(self, *args):
            return ()

        async def ingest(self, operation):
            pytest.fail("expired content cannot reach native ingest")

    async def stop(call):
        stopped.append(call)

    async def fail():
        pytest.fail("expired local content must not degrade the aged FIFO")

    async def maintain():
        await maintain_call_content(
            writer, Sink(), stop, utcnow=lambda: clock[0], timeout_seconds=2
        )

    try:
        relay = OutboxRelay(writer, Sink(), utcnow=lambda: clock[0], before_fifo=maintain,
                            on_degraded=fail, drain=fail)
        assert (await relay.run_once()).status == "empty"
        assert stopped == [call_id]
        assert (await writer.read_retained_call(call_id)).erased
        assert await writer.oldest_outbox_created_at() is None
        assert not writer.is_degraded
        # Native settled erasure permits GC only beyond the exact replay window.
        token = uuid4()
        await writer.erase_call_content(call_id, now=clock[0], lease_token=token)
        await writer.finish_erasure_ack(call_id, token, acknowledged=True)
        clock[0] = deadline + timedelta(seconds=900)
        await writer.cleanup_local_state(now=clock[0])
        assert await writer.read_call_lifecycle(call_id) is not None
        clock[0] += timedelta(microseconds=max(1, cutoff_microsecond))
        await writer.cleanup_local_state(now=clock[0])
        assert await writer.read_call_lifecycle(call_id) is None
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        if not task.done():
            await writer.drain(2)
        await task


@pytest.mark.asyncio
async def test_retained_dialogue_survives_ack_and_duplicate_capture(tmp_path):
    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        turn = capture(writer, call_id, 1)
        assert writer.try_enqueue_turn(turn)
        retained = await writer.read_retained_call(call_id)
        assert retained.turns[0].text == "Pouvez-vous me rappeler ?"
        rows = await writer.read_relay_batch(batch_size=100, now=writer._utcnow(), lease_seconds=10)
        assert any(row.operation.operation_id == turn.operation_id for row in rows)
        for row in rows:
            await writer.ack_outbox(queue_id=row.queue_id, expected_claim_attempt=row.claim_attempt)
        assert writer.try_enqueue_turn(turn)
        replay = await writer.read_retained_call(call_id)
        assert replay == retained
        assert not writer.is_degraded
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_full_map_budget_drops_without_global_degradation_and_counts_once(tmp_path):
    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        accepted = []
        dropped = None
        for number in range(1, 30):
            turn = capture(writer, call_id, number, "a" * 16384)
            assert writer.try_enqueue_turn(turn)
            retained = await writer.read_retained_call(call_id)
            if retained.loss_count:
                dropped = turn
                break
            accepted.append(turn)
        assert dropped is not None and len(accepted) < 200
        actual_map = {str(t.payload.turn_id): t.payload.model_dump(mode="json") for t in accepted}
        assert len(json.dumps(actual_map, ensure_ascii=False).encode()) <= 524288
        actual_map[str(dropped.payload.turn_id)] = dropped.payload.model_dump(mode="json")
        assert len(json.dumps(actual_map, ensure_ascii=False).encode()) > 524288
        assert writer.try_enqueue_turn(dropped)
        later = capture(writer, call_id, 99, "court")
        assert writer.try_enqueue_turn(later)
        retained = await writer.read_retained_call(call_id)
        assert retained.loss_count == 1
        assert retained.turns[-1].turn_no == 99
        assert not writer.is_degraded
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_content_tombstone_blocks_late_turn_and_preserves_phone_lease(tmp_path):
    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        assert writer.try_enqueue_turn(capture(writer, call_id, 1))
        token = uuid4()
        cleanup = await writer.erase_call_content(call_id, lease_token=token, now=NOW)
        assert cleanup == NOW
        assert writer.try_enqueue_turn(capture(writer, call_id, 2))
        retained = await writer.read_retained_call(call_id)
        assert retained.erased and retained.turns == ()
        assert (
            await writer.erase_call_content(
                call_id, lease_token=token, now=NOW + timedelta(seconds=1)
            )
            == NOW
        )
        assert await registry.live_call_count() == 1
        facts = await writer.read_call_lifecycle(call_id)
        assert facts is not None and facts.disclosure_evidence is None
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_frozen_partial_result_replays_after_ack_without_new_nonce(tmp_path):
    from projetv0_voice.models import MessageResultV1

    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        turn = capture(writer, call_id, 1)
        assert writer.try_enqueue_turn(turn)
        facts = await writer.read_call_lifecycle(call_id)
        base = VoiceOperationV1(
            schema_version=1,
            operation_id=uuid4(),
            deployment_id="fixture",
            call_id=call_id,
            occurred_at=NOW,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id="original",
                telnyx_call_leg_id="original-leg",
                telnyx_call_session_id="session",
                status="closed",
                disclosure_state="failed",
                started_at=NOW,
                ended_at=NOW,
                end_reason="call.hangup",
                retention_until=facts.retention_until,
            ),
        )
        result = MessageResultV1.model_validate(
            dict(
                schema_version=1,
                quality="partial",
                category="callback",
                summary="Rappel demandé.",
                next_action="Rappeler.",
                contact=dict(
                    name=None,
                    callback_e164=None,
                    preference=None,
                    callback_source="missing",
                    callback_confirmed=False,
                ),
                evidence=[dict(turn_id=turn.payload.turn_id, role="user")],
                request_confirmed=False,
            )
        )
        frozen = await writer.freeze_call_publication(base, result, provider_callback=None)
        assert frozen.payload.message_result is not None
        rows = await writer.read_relay_batch(batch_size=100, now=writer._utcnow(), lease_seconds=10)
        for row in rows:
            await writer.ack_outbox(queue_id=row.queue_id, expected_claim_attempt=row.claim_attempt)
        other = base.model_copy(update={"operation_id": uuid4()})
        replay = await writer.freeze_call_publication(other, None, provider_callback=None)
        assert replay == frozen
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_session_result_inference_is_joined_when_ai_stops(tmp_path):
    from types import SimpleNamespace

    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.session import CallSession

    registry, writer, worker, _ = await start(tmp_path)
    metrics = RuntimeMetrics.in_memory()
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def inference(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        assert writer.try_enqueue_turn(capture(writer, call_id, 1))
        session = CallSession.__new__(CallSession)
        session._runtime_metrics = metrics
        session._identity = SimpleNamespace(
            call_id=call_id, routing=SimpleNamespace(from_e164=None)
        )
        session._writer = writer
        session._no_new_ai = False
        session._controller = session._recorder = None
        session._services = SimpleNamespace(llm=SimpleNamespace(run_inference=inference))
        session._cleanup_phase_timeout_seconds = 1
        session._result_inference_task = None
        session._partial_result = None
        prepare = asyncio.create_task(session._prepare_partial_result())
        try:
            async with asyncio.timeout(2):
                await entered.wait()
                session.stop_new_ai()
                await prepare
        finally:
            if not prepare.done():
                prepare.cancel()
                await asyncio.gather(prepare, return_exceptions=True)
        assert cancelled.is_set()
        assert session._partial_result is None
    finally:
        await metrics.aclose()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_stopped_recorder_facts_reach_existing_native_llm_client(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import httpx
    from openai import DefaultAsyncHttpxClient
    from pydantic import SecretStr

    from projetv0_voice.inference import services
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.persistence.business_result import infer_partial_result
    from projetv0_voice.pipeline import FirstFailure
    from projetv0_voice.qualified_profile import InferenceProfileV1
    from projetv0_voice.session import TurnRecorder

    registry, writer, worker, _ = await start(tmp_path)
    metrics = RuntimeMetrics.in_memory()
    requests = []
    retained = None

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        turn_id = str(retained.turns[0].turn_id)
        result = dict(
            schema_version=1,
            quality="partial",
            category="callback",
            summary="Rappel demandé.",
            next_action="Rappeler.",
            contact=dict(
                name=None,
                callback_e164="+33102030407",
                preference=None,
                callback_source="provider",
                callback_confirmed=False,
            ),
            evidence=[dict(turn_id=turn_id, role="user")],
            request_confirmed=False,
        )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "test/llm",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
            },
        )

    http = DefaultAsyncHttpxClient(transport=httpx.MockTransport(handle), trust_env=False)
    monkeypatch.setattr(services, "DefaultAsyncHttpxClient", lambda **kwargs: http)
    profile = InferenceProfileV1.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/inference-profile-v1.json").read_text()
    )
    llm = services.build_llm(profile, SecretStr("owned-offline-key"))
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        recorder = TurnRecorder(
            identity=SimpleNamespace(
                call_id=call_id,
                deployment_id="fixture",
                routing=SimpleNamespace(from_e164="+33102030407"),
            ),
            writer=writer,
            keyring=writer._keyring,
            first_failure=FirstFailure(),
            runtime_metrics=metrics,
            utcnow=lambda: NOW,
        )
        recorder.record_user("Pouvez-vous me rappeler ?", NOW.isoformat())
        recorder.record_assistant("Je prends votre message.", NOW.isoformat(), True)
        recorder.close()
        retained = await writer.read_retained_call(call_id)
        assert len(retained.turns) == 2 and retained.turns[1].interrupted
        result = await infer_partial_result(llm, retained, "+33102030407")
        assert result.quality == "partial" and not result.request_confirmed
        assert (
            result.contact.callback_source == "provider" and not result.contact.callback_confirmed
        )
        assert requests[0]["stream"] is False
        assert requests[0]["provider"] == {"allow_fallbacks": True, "sort": "latency"}
        assert not requests[0].get("tools")
        assert requests[0]["model"] == "test/llm"
        assert "Pouvez-vous" in requests[0]["messages"][-1]["content"]
    finally:
        await llm._client.close()
        metrics._provider.shutdown()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_erase_departure_restart_keeps_capacity_until_real_original_hangup(tmp_path):
    from projetv0_voice.persistence.writer import PersistenceWriter

    registry, writer, worker, provider = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        await registry.stop_call_content(call_id)
        await writer.erase_call_content(call_id, now=NOW)
        facts = await writer.read_call_lifecycle(call_id)
        assert facts.content_erased and facts.content_departed_generation
        assert facts.transfer_command_id is None
        assert await registry.live_call_count() == 1
        assert not any(action[0] == "hangup" for action in provider.actions)
        await writer.drain(2)
        await worker
        reopened = PersistenceWriter(writer._database_path, writer._keyring)
        reopened_task = asyncio.create_task(reopened.run())
        assert await reopened.wait_ready()
        try:
            stale = reopened.take_stale_leases()
            assert stale[0].lifecycle.content_erased
            fresh_registry, scratch, scratch_worker, fresh_provider = await start(
                tmp_path / "fresh"
            )
            try:
                fresh_registry._writer = reopened
                await fresh_registry.restore_transfer_fence(stale[0])
                assert await fresh_registry.live_call_count() == 1
                await committed(fresh_registry, reopened, event("call.hangup"))
                assert await fresh_registry.live_call_count() == 0
                assert fresh_provider.actions == []
                assert (await reopened.read_retained_call(call_id)).erased
                assert await reopened.oldest_outbox_created_at() is None
            finally:
                await scratch.drain(2)
                await scratch_worker
        finally:
            await reopened.drain(2)
            await reopened_task
    finally:
        if not worker.done():
            await writer.drain(2)
            await worker


@pytest.mark.asyncio
async def test_truncation_and_budget_drop_are_one_loss_and_first200_survives_restart(tmp_path):
    from projetv0_voice.persistence.writer import PersistenceWriter

    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        for number in range(1, 202):
            turn = capture(writer, call_id, number, "a")
            assert writer.try_enqueue_turn(turn, truncated=number == 201)
            retained = await writer.read_retained_call(call_id)
        assert len(retained.turns) == 200 and retained.loss_count == 1
        assert writer.try_enqueue_turn(turn, truncated=True)
        await writer.read_retained_call(call_id)
        await writer.drain(2)
        await worker
        reopened = PersistenceWriter(writer._database_path, writer._keyring)
        owned = asyncio.create_task(reopened.run())
        assert await reopened.wait_ready()
        try:
            replay = await reopened.read_retained_call(call_id)
            assert replay == retained
            assert reopened.pragma_state["secure_delete"] == 1
        finally:
            await reopened.drain(2)
            await owned
    finally:
        if not worker.done():
            await writer.drain(2)
            await worker


@pytest.mark.asyncio
async def test_erasure_ack_identity_and_encrypted_byte_cleanup_survive_restart(tmp_path):
    from projetv0_voice.persistence.writer import PersistenceWriter

    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        turn = capture(writer, call_id, 1, "owned synthetic erasure witness" * 200)
        assert writer.try_enqueue_turn(turn)
        await writer.read_retained_call(call_id)
        import sqlite3

        # Only read owned encrypted fixture bytes; the native writer remains sole mutator.
        with sqlite3.connect(f"file:{writer._database_path.as_posix()}?mode=ro", uri=True) as db:
            ciphertext = db.execute(
                "SELECT ciphertext FROM sparra_turn_decisions WHERE call_id=?", (str(call_id),)
            ).fetchone()[0]
        marker = ciphertext[512:576]
        assert marker in writer._database_path.read_bytes()
        token = uuid4()
        await writer.erase_call_content(call_id, now=NOW, lease_token=token)
        assert marker not in writer._database_path.read_bytes()
        assert not writer._database_path.with_name(writer._database_path.name + "-journal").exists()
        await writer.drain(2)
        await worker
        reopened = PersistenceWriter(writer._database_path, writer._keyring)
        task = asyncio.create_task(reopened.run())
        assert await reopened.wait_ready()
        try:
            assert await reopened.pending_erasure_acks() == ((call_id, token, NOW),)
            assert (
                await reopened.erase_call_content(
                    call_id, now=NOW + timedelta(seconds=1), lease_token=token
                )
                == NOW
            )
            await reopened.finish_erasure_ack(call_id, token)
            assert await reopened.pending_erasure_acks() == ()
            replacement = uuid4()
            assert await reopened.erase_call_content(
                call_id, now=NOW + timedelta(seconds=2), lease_token=replacement
            ) == NOW + timedelta(seconds=2)
            assert await reopened.pending_erasure_acks() == (
                (call_id, replacement, NOW + timedelta(seconds=2)),
            )
        finally:
            await reopened.drain(2)
            await task
    finally:
        if not worker.done():
            await writer.drain(2)
            await worker


@pytest.mark.asyncio
async def test_invalid_stopped_capture_counts_durable_loss_without_poisoning_fifo(tmp_path):
    from types import SimpleNamespace

    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.pipeline import FirstFailure
    from projetv0_voice.session import TurnRecorder

    registry, writer, worker, _ = await start(tmp_path)
    metrics = RuntimeMetrics.in_memory()
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        recorder = TurnRecorder(
            identity=SimpleNamespace(
                call_id=call_id, deployment_id="fixture", routing=SimpleNamespace(from_e164=None)
            ),
            writer=writer,
            keyring=writer._keyring,
            first_failure=FirstFailure(),
            runtime_metrics=metrics,
            utcnow=lambda: NOW,
        )
        recorder.record_user("invalid\x00text", NOW.isoformat())
        retained = await writer.read_retained_call(call_id)
        assert retained.loss_count == 1 and retained.turns == ()
        assert not writer.is_degraded
    finally:
        await writer.drain(2)
        await worker
        metrics._provider.shutdown()


@pytest.mark.asyncio
async def test_retained_operation_authenticates_call_metadata_after_outbox_ack(tmp_path):
    import sqlite3

    from projetv0_voice.persistence.commands import FatalPersistenceError
    from projetv0_voice.persistence.writer import PersistenceWriter

    registry, writer, worker, _ = await start(tmp_path)
    await committed(registry, writer, event())
    call_id = (await registry.snapshot("original")).call_id
    assert writer.try_enqueue_turn(capture(writer, call_id, 1))
    await writer.read_retained_call(call_id)
    await writer.drain(2)
    await worker
    other_call = uuid4()
    # Owned on-disk corruption fixture, with the production writer fully stopped.
    with sqlite3.connect(writer._database_path) as db:
        db.execute(
            "UPDATE sparra_turn_decisions SET call_id=? WHERE call_id=?",
            (str(other_call), str(call_id)),
        )
    reopened = PersistenceWriter(writer._database_path, writer._keyring)
    task = asyncio.create_task(reopened.run())
    assert await reopened.wait_ready()
    try:
        with pytest.raises(FatalPersistenceError):
            await reopened.read_retained_call(other_call)
    finally:
        if not reopened.is_degraded:
            await reopened.drain(2)
        await task


@pytest.mark.asyncio
async def test_sparra_closes_existing_clients_when_terminal_publication_cannot_commit():
    from types import SimpleNamespace

    from projetv0_voice.pipeline import FirstFailure
    from projetv0_voice.session import CallSession, _TerminalOutcome

    closed = []

    async def close():
        closed.append(True)

    async def reserve(_proposal):
        return SimpleNamespace(
            status="closed",
            reason="telnyx_hangup",
            metric_class="closed",
            completion_token=uuid4(),
            _closed_at=NOW,
        )

    async def failed_commit(**kwargs):
        return False

    session = CallSession.__new__(CallSession)
    session._identity = SimpleNamespace(routing=object())
    session._writer = SimpleNamespace(fatal_event=asyncio.Event())
    session._registry_terminalizer = SimpleNamespace(reserve_or_read=reserve)
    session._metric_lease = SimpleNamespace(finish=lambda value: None)
    session._services = SimpleNamespace(aclose=close)
    session._cleanup_phase_timeout_seconds = 1
    session._commit_authoritative_terminal_call = failed_commit
    await session._finish_durable_boundaries(
        first_failure=FirstFailure(),
        terminal_outcome=_TerminalOutcome("closed"),
        disclosure_completed=False,
    )
    assert closed == [True]


@pytest.mark.asyncio
async def test_content_stop_reserves_phone_fence_before_first_writer_await(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    await committed(registry, writer, event())
    call_id = (await registry.snapshot("original")).call_id
    original_read, original_erase = writer.read_call_lifecycle, writer.erase_call_content

    async def paused_read(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_read(*args, **kwargs)

    async def paused_erase(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_erase(*args, **kwargs)

    writer.read_call_lifecycle, writer.erase_call_content = paused_read, paused_erase
    stopping = asyncio.create_task(registry.stop_call_content(call_id))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        assert registry._transfer_fenced(registry._by_control["original"])
        assert registry._by_control["original"].drain_intent
        await registry._hangup_unfenced("original", command_id=uuid4())
        assert not any(action[0] == "hangup" for action in provider.actions)
    finally:
        release.set()
        await stopping
        await writer.drain(2)
        await worker


@pytest.mark.parametrize(
    "reason", ["native_erased", "expired", "unavailable", "legacy", "held", "backlog"]
)
@pytest.mark.asyncio
async def test_startup_erasure_precedes_stale_recovery_and_keeps_original_generation(
    tmp_path, reason
):
    from projetv0_voice.admission import CallRegistry
    from projetv0_voice.lifecycle import RuntimeSupervisor
    from projetv0_voice.persistence.postgres_sink import CallErasureLease
    from projetv0_voice.persistence.relay import OutboxRelay, maintain_call_content
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.unit.test_sparra_admission import DID, ControlledProvider, policy, snapshot

    original_registry, original_writer, original_worker, _ = await start(tmp_path)
    await committed(original_registry, original_writer, event())
    call_id = (await original_registry.snapshot("original")).call_id
    original_generation = (await original_registry.generation_handle("original")).generation
    await original_writer.drain(2)
    await original_worker
    if reason == "legacy":
        import sqlite3

        with sqlite3.connect(original_writer._database_path) as db:
            raw = db.execute(
                "SELECT lifecycle_json FROM call_leases WHERE call_id=?", (str(call_id),)
            ).fetchone()[0]
            values = json.loads(raw)
            values.pop("admission_generation", None)
            db.execute(
                "UPDATE call_leases SET lifecycle_json=? WHERE call_id=?",
                (json.dumps(values), str(call_id)),
            )
    clock = NOW + (timedelta(days=31) if reason == "expired" else timedelta(seconds=40))
    writer = PersistenceWriter(
        original_writer._database_path, original_writer._keyring, utcnow=lambda: clock
    )
    chronology = []

    class Control(ControlledProvider):
        async def aclose(self):
            pass

    control = Control()

    async def begin(deployment, call, routing):
        return snapshot(call, routing)

    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="fixture",
        agent_id="fixture",
        deployment_id="fixture",
        capacity=1,
        lease_ttl_seconds=30,
        retention_days=30,
        stream_url="wss://fixture.invalid/media",
        utcnow=lambda: clock,
        monotonic=lambda: 100,
        sparra=policy(),
        called_did=DID,
        begin_call=begin,
    )
    lease = CallErasureLease(
        1, call_id, uuid4(), "fixture", NOW + timedelta(days=30), clock + timedelta(seconds=30)
    )

    class Sink:
        def __init__(self):
            # Native SQL excludes still-held leases and caps the ordered batch100.
            self.held_until = clock + timedelta(seconds=20) if reason == "held" else None
            self.leases = (
                (lease,)
                if reason in {"native_erased","held"}
                else (
                    tuple(
                        CallErasureLease(
                            1,
                            uuid4(),
                            uuid4(),
                            "fixture",
                            NOW + timedelta(days=30),
                            clock + timedelta(seconds=30),
                        )
                        for _ in range(100)
                    ) + (lease,)
                    if reason == "backlog"
                    else ()
                )
            )

        async def open(self):
            chronology.append("sink_open")

        async def close(self):
            pass

        async def lease_call_erasures(self, *args):
            if reason == "unavailable":
                from projetv0_voice.persistence.postgres_sink import OperationSinkTransientError

                raise OperationSinkTransientError("owned_native_knowledge_unavailable")
            chronology.append("maintenance")
            if self.held_until is not None and clock < self.held_until:
                return ()
            # The real function limits100; the actual call is obligation101.
            result, self.leases = self.leases[:100], self.leases[100:]
            return result

        async def ack_call_erasure(self, *args):
            assert (await writer.read_retained_call(args[0])).erased

        async def ingest(self, operation):
            pytest.fail("startup erased content cannot be relayed")

    sink = Sink()

    async def maintain():
        await maintain_call_content(
            writer, sink, registry.stop_call_content, utcnow=lambda: clock, timeout_seconds=2
        )

    async def fail():
        pytest.fail("erasure must not globally degrade")

    original_restore = registry.restore_transfer_fence

    async def restore(stale):
        chronology.append("recovery")
        if reason in {"held","backlog"}:
            assert stale.lifecycle.local_closing_at is not None
            assert not stale.lifecycle.content_erased
        await original_restore(stale)

    registry.restore_transfer_fence = restore
    relay = OutboxRelay(
        writer,
        sink,
        utcnow=lambda: clock,
        before_fifo=maintain,
        stop_erased_call=registry.stop_call_content,
        on_degraded=fail,
        drain=fail,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=control,
        relay=relay,
        registry=registry,
        sink=sink,
        sparra_enabled=True,
        utcnow=lambda: clock,
        loop_interval_seconds=100,
        deployment_id="fixture",
        retention_days=30,
        shutdown_timeout_seconds=10,
    )
    try:
        if reason in {"unavailable", "legacy"}:
            with pytest.raises(RuntimeError):
                await supervisor.startup()
            assert not any(action[0] == "hangup" for action in control.actions)
            return
        await supervisor.startup()
        assert not any(action[0] == "hangup" for action in control.actions)
        assert chronology.index("maintenance") < chronology.index("recovery")
        assert (await registry.generation_handle("original")).generation == original_generation
        assert await registry.live_call_count() == 1
        await committed(registry, writer, event("call.hangup", occurred_at=clock))
        assert await registry.live_call_count() == 0
        retained = await writer.read_retained_call(call_id)
        if reason in {"held", "backlog"}:
            facts = await writer.read_call_lifecycle(call_id)
            if reason=="held":
                assert not retained.erased and not facts.content_erased
            assert facts.local_closing_at is not None
        else:
            assert retained.erased
            assert await writer.oldest_outbox_created_at() is None
    finally:
        await supervisor.aclose()


@pytest.mark.parametrize("mismatch", ["role", "call", "provider", "caller"])
def test_partial_result_rejects_unretained_roles_or_unobserved_coordinates(mismatch):
    from projetv0_voice.models import MessageResultV1
    from projetv0_voice.persistence.business_result import (
        RetainedCall,
        RetainedTurn,
        validate_result_provenance,
    )

    turn_id = uuid4()
    retained = RetainedCall((RetainedTurn(turn_id, 1, "user", "Rappelez +33102030407", False),), 0)
    contact = dict(
        name=None,
        callback_e164=None,
        preference=None,
        callback_source="missing",
        callback_confirmed=False,
    )
    reference = dict(turn_id=str(turn_id), role="user")
    if mismatch == "role":
        reference["role"] = "assistant"
    if mismatch == "call":
        reference["turn_id"] = str(uuid4())
    if mismatch in {"provider", "caller"}:
        contact.update(callback_e164="+33102030408", callback_source=mismatch)
    result = MessageResultV1.model_validate(
        dict(
            schema_version=1,
            quality="partial",
            category="callback",
            summary="Rappel demandé.",
            contact=contact,
            next_action="Rappeler.",
            evidence=[reference],
            request_confirmed=False,
        )
    )
    with pytest.raises(ValueError):
        validate_result_provenance(result, retained, "+33102030407")


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["pending", "unknown", "bridged"])
async def test_result_never_calls_api_after_takeover(tmp_path, departure):
    from dataclasses import replace
    from types import SimpleNamespace

    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.session import CallSession

    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call_id = (await registry.snapshot("original")).call_id
        assert writer.try_enqueue_turn(capture(writer, call_id, 1))
        facts = await writer.read_call_lifecycle(call_id)
        facts = replace(
            facts,
            transfer_command_id=uuid4(),
            transfer_generation=facts.admission_generation,
            transfer_correlation="owned-correlation",
            transfer_connection_sha256="a" * 64,
            transfer_destination_sha256="b" * 64,
            qualified_line_bridged_at=NOW if departure == "bridged" else None,
        )
        await writer.commit_transfer_intent(facts)

        async def inference(*args, **kwargs):
            pytest.fail("takeover must never rearm result API")

        session = CallSession.__new__(CallSession)
        session._identity = SimpleNamespace(
            call_id=call_id, routing=SimpleNamespace(from_e164=None)
        )
        session._writer = writer
        session._runtime_metrics = RuntimeMetrics.in_memory()
        session._no_new_ai = departure == "bridged"
        session._result_inference_fenced = False
        session._services = SimpleNamespace(llm=SimpleNamespace(run_inference=inference))
        session._partial_result = session._result_inference_task = None
        try:
            await session._prepare_partial_result()
        finally:
            await session._runtime_metrics.aclose()
        assert session._partial_result is None and session._result_inference_task is None
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("pin_started", [False, True])
async def test_cleanup_ack_has_no_registry_caller_pin_or_completed_future(tmp_path, pin_started):
    from projetv0_voice.persistence.postgres_sink import CallErasureLease
    from projetv0_voice.persistence.relay import maintain_call_content

    registry, writer, worker, _ = await start(tmp_path)
    try:
        observed = event(from_e164="+33102030407")
        if pin_started:
            await committed(registry, writer, observed)
        else:
            resolved = await registry.resolve_webhook(observed)
            effect = resolved.effect
            ticket = writer.submit_webhook(
                receipt=dict(
                    event_id=observed.event_id,
                    event_type=observed.event_type,
                    call_control_id=observed.call_control_id,
                    occurred_at=observed.occurred_at,
                    received_at=NOW,
                    semantic_fingerprint_sha256=observed.semantic_fingerprint_sha256,
                ),
                lease=effect.lease,
                operation=effect.operation,
                admission_facts=effect.admission_facts,
            )
            await ticket.wait()
        entry = registry._by_control["original"]
        if pin_started:
            await registry._ensure_begin_snapshot("original")
            assert entry.begin_future.done() and entry.begin_snapshot is not None
        generation = entry.generation
        lease = CallErasureLease(
            1,
            entry.call_id,
            uuid4(),
            "fixture",
            NOW + timedelta(days=30),
            NOW + timedelta(seconds=30),
        )

        class Sink:
            async def lease_call_erasures(self, *args):
                return (lease,)

            async def ack_call_erasure(self, *args):
                assert entry.routing is None
                assert entry.begin_snapshot is None and entry.begin_future is None
                assert entry.begin_task is None and entry.construction_grant is None
                assert entry.session is None and entry.lifecycle_owner is None
                assert entry.generation == generation
                assert await registry.live_call_count() == 1

        await maintain_call_content(
            writer, Sink(), registry.stop_call_content, utcnow=lambda: NOW, timeout_seconds=2
        )
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_cleanup_ack_joins_inflight_begin_before_releasing_its_content(tmp_path):
    from projetv0_voice.persistence.postgres_sink import CallErasureLease
    from projetv0_voice.persistence.relay import maintain_call_content
    from tests.unit.test_sparra_admission import snapshot

    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def begin(deployment, call_id, routing):
        pin = snapshot(call_id, routing)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return pin

    registry, writer, worker, _ = await start(tmp_path, begin)
    admission = asyncio.create_task(committed(registry, writer, event(from_e164="+33102030407")))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        entry = registry._by_control["original"]
        lease = CallErasureLease(
            1,
            entry.call_id,
            uuid4(),
            "fixture",
            NOW + timedelta(days=30),
            NOW + timedelta(seconds=30),
        )

        class Sink:
            async def lease_call_erasures(self, *args):
                return (lease,)

            async def ack_call_erasure(self, *args):
                assert cancelled.is_set() and admission.done()
                assert entry.routing is None and entry.begin_snapshot is None
                assert entry.begin_future is None and entry.begin_task is None
                assert entry.lifecycle_owner is None and entry.session is None
                assert (await writer.read_retained_call(entry.call_id)).erased

        await maintain_call_content(
            writer, Sink(), registry.stop_call_content, utcnow=lambda: NOW, timeout_seconds=2
        )
        assert await registry.live_call_count() == 1
    finally:
        if not admission.done():
            admission.cancel()
        await asyncio.gather(admission, return_exceptions=True)
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_pv301_production_stop_preserves_replaced_claim_without_erasure(tmp_path):
    from projetv0_voice.persistence.postgres_sink import OperationSinkErasedError
    from projetv0_voice.persistence.relay import OutboxRelay
    from projetv0_voice.session_factory import ProcessSessionFactory

    registry, writer, worker, _ = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        entry = registry._by_control["original"]
        assert writer.try_enqueue_turn(capture(writer, entry.call_id, 1))
        await writer.read_retained_call(entry.call_id)
        clock = [writer._utcnow() + timedelta(seconds=1)]
        factory = ProcessSessionFactory.__new__(ProcessSessionFactory)
        factory._registry = registry
        deletions = []
        original_erase = writer.erase_call_content

        async def guarded_erase(*args, **kwargs):
            deletions.append(kwargs.get("expected_item"))
            return await original_erase(*args, **kwargs)

        writer.erase_call_content = guarded_erase

        async def production_stop(call_id):
            await factory.stop_call_content_by_id(call_id)

        async def fail():
            pytest.fail("stale claim must not degrade")

        replaced = []

        class Sink:
            async def ingest(self, operation):
                clock[0] += timedelta(seconds=11)
                row = (await writer.read_relay_batch(batch_size=1, now=clock[0], lease_seconds=10))[
                    0
                ]
                replaced.append(row)
                raise OperationSinkErasedError("owned_rejected_dispatch")

        relay = OutboxRelay(
            writer,
            Sink(),
            utcnow=lambda: clock[0],
            stop_erased_call=production_stop,
            on_degraded=fail,
            drain=fail,
        )
        result = await relay.run_once(batch_size=1)
        assert result.status == "stale_claim" and result.discarded == 0
        assert len(deletions) == 1 and deletions[0] is not None
        assert deletions[0].claim_attempt + 1 == replaced[0].claim_attempt
        assert not (await writer.read_retained_call(entry.call_id)).erased
        rows = await writer.read_relay_batch(
            batch_size=1, now=clock[0] + timedelta(seconds=11), lease_seconds=10
        )
        assert rows[0].queue_id == replaced[0].queue_id
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_erasure_preserves_pending_phone_transfer_without_inventing_bridge(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    requested = None
    try:
        await committed(registry, writer, event())
        generation = await registry.generation_handle("original")
        requested = asyncio.create_task(registry.request_human(generation))
        async with asyncio.timeout(2):
            await provider.transfer_entered.wait()
        entry = registry._by_control["original"]
        correlation = entry.transfer_facts.transfer_correlation
        await registry.stop_call_content(entry.call_id)
        provider.transfer_release.set()
        result = await requested
        assert result != "qualified_line_connected"
        facts = await writer.read_call_lifecycle(entry.call_id)
        assert facts.qualified_line_bridged_at is None and facts.transfer_correlation == correlation
        assert facts.transfer_fenced and facts.content_erased
        assert await registry.live_call_count() == 1
        assert not any(action[0] == "hangup" for action in provider.actions)
        await committed(registry, writer, event("call.hangup"))
        assert await registry.live_call_count() == 0
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.gather(requested, return_exceptions=True)
        await writer.drain(2)
        await worker
