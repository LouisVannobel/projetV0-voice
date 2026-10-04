from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.persistence.postgres_sink import OperationSinkErasedError
from projetv0_voice.persistence.relay import OutboxRelay, maintain_call_content
from projetv0_voice.persistence.writer import PersistenceWriter

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def publication(call_id):
    return VoiceOperationV1(
        schema_version=1,
        operation_id=uuid4(),
        deployment_id="fixture",
        call_id=call_id,
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id=str(call_id),
            telnyx_call_leg_id=None,
            telnyx_call_session_id=None,
            status="closed",
            disclosure_state="failed",
            started_at=NOW,
            ended_at=NOW,
            end_reason="fixture",
            retention_until=NOW + timedelta(days=30),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [False, True])
async def test_actual_writer_cleanup_precedes_fifo_age_and_pv301_continues(tmp_path, rejected):
    import asyncio

    clock = [NOW - timedelta(seconds=901)]
    writer = PersistenceWriter(
        tmp_path / "erasure.sqlite",
        CryptoKeyring({1: b"k" * 32}, active_version=1),
        utcnow=lambda: clock[0],
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    erased, other = uuid4(), uuid4()
    calls = []

    async def failed():
        pytest.fail("call-specific discard must not degrade unrelated FIFO")

    async def prepare():
        if not rejected:
            await writer.erase_call_content(erased, now=NOW)

    class Sink:
        async def ingest(self, operation):
            if operation.call_id == erased:
                raise OperationSinkErasedError("call_content_erased")
            calls.append(operation.call_id)

    try:
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": publication(erased)}, None)
        )
        clock[0] = NOW
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": publication(other)}, None)
        )
        if rejected:
            # PV301 is a dispatched rejection, so it remains inside the native age gate.
            clock[0] = NOW - timedelta(seconds=1)
            writer._utcnow = lambda: NOW
        relay = OutboxRelay(
            writer,
            Sink(),
            utcnow=lambda: clock[0],
            before_fifo=prepare,
            on_degraded=failed,
            drain=failed,
        )
        if rejected:
            # Use a fresh rejected row to test dispatch separately from expired cleanup.
            await writer.erase_call_content(erased, now=NOW)
            erased = uuid4()
            await writer.commit_control(
                PersistenceCommand("outbox", {"operation": publication(erased)}, None)
            )
            clock[0] = NOW
        result = await relay.run_once()
        assert calls == [other]
        assert result.acked == 1
        assert result.discarded == int(rejected)
        assert await writer.oldest_outbox_created_at() is None
        assert not writer.is_degraded
    finally:
        await writer.drain(2)
        await task


@pytest.mark.asyncio
async def test_actual_maintenance_freezes_cleanup_and_replays_lost_ack(tmp_path):
    import asyncio

    from projetv0_voice.persistence.postgres_sink import (
        CallErasureLease,
        OperationSinkCommitAmbiguousError,
    )

    writer = PersistenceWriter(
        tmp_path / "ack.sqlite", CryptoKeyring({1: b"k" * 32}, active_version=1), utcnow=lambda: NOW
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    call_id, token = uuid4(), uuid4()
    lease = CallErasureLease(
        1, call_id, token, "fixture", NOW + timedelta(days=30), NOW + timedelta(seconds=30)
    )
    acks, stopped = [], []

    async def stop(call):
        stopped.append(call)

    class Sink:
        def __init__(self):
            self.leases = (lease,)

        async def lease_call_erasures(self, *args):
            leases, self.leases = self.leases, ()
            return leases

        async def ack_call_erasure(self, call, lease_token, cleaned_at):
            assert (await writer.read_retained_call(call)).erased
            assert await writer.oldest_outbox_created_at() is None
            acks.append((call, lease_token, cleaned_at))
            if len(acks) == 1:
                raise OperationSinkCommitAmbiguousError("owned_ack_response_loss")

    try:
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": publication(call_id)}, None)
        )
        sink = Sink()
        with pytest.raises(OperationSinkCommitAmbiguousError):
            await maintain_call_content(writer, sink, stop, utcnow=lambda: NOW, timeout_seconds=2)
        assert await writer.pending_erasure_acks() == ((call_id, token, NOW),)
        await maintain_call_content(
            writer, sink, stop, utcnow=lambda: NOW + timedelta(seconds=1), timeout_seconds=2
        )
        assert acks == [(call_id, token, NOW)] * 2 and stopped == [call_id]
        assert await writer.pending_erasure_acks() == ()
    finally:
        await writer.drain(2)
        await task


@pytest.mark.asyncio
async def test_erasure_keeps_genuine_recording_operations_until_real_purge_handoff(tmp_path):
    import asyncio

    from projetv0_voice.models import RecordingUpsertPayloadV1

    writer = PersistenceWriter(
        tmp_path / "recording.sqlite",
        CryptoKeyring({1: b"k" * 32}, active_version=1),
        utcnow=lambda: NOW,
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    call_id = uuid4()
    recording = VoiceOperationV1(
        schema_version=1,
        operation_id=uuid4(),
        deployment_id="fixture",
        call_id=call_id,
        occurred_at=NOW,
        kind="recording.upsert",
        payload=RecordingUpsertPayloadV1(
            recording_id=uuid4(),
            status="saved",
            telnyx_recording_id="owned_recording-id",
            channels="dual",
            format="wav",
            started_at=NOW,
            ended_at=NOW,
            retention_until=NOW + timedelta(days=30),
        ),
    )
    try:
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": publication(call_id)}, None)
        )
        await writer.commit_control(PersistenceCommand("outbox", {"operation": recording}, None))
        await writer.erase_call_content(call_id, now=NOW)
        rows = await writer.read_relay_batch(batch_size=100, now=NOW, lease_seconds=10)
        assert [r.operation for r in rows] == [recording]
        await writer.ack_outbox(
            queue_id=rows[0].queue_id, expected_claim_attempt=rows[0].claim_attempt
        )
        # A genuinely observed late recording reopens native delivery despite the content tombstone.
        late = recording.model_copy(update={"operation_id": uuid4()})
        await writer.commit_control(PersistenceCommand("outbox", {"operation": late}, None))
        rows = await writer.read_relay_batch(batch_size=100, now=NOW, lease_seconds=10)
        assert [r.operation for r in rows] == [late]
        assert (await writer.read_retained_call(call_id)).erased
    finally:
        await writer.drain(2)
        await task


@pytest.mark.asyncio
async def test_expired_recording_handoff_precedes_age_gate_and_reaches_native_purger(
    tmp_path, monkeypatch
):
    import asyncio

    import httpx
    import telnyx

    from projetv0_voice.models import RecordingUpsertPayloadV1
    from projetv0_voice.persistence.postgres_sink import CallErasureLease, RecordingPurgeLease
    from projetv0_voice.telnyx import call_control
    from projetv0_voice.telnyx.recordings import purge_recordings_once

    clock = [NOW - timedelta(seconds=901)]
    writer = PersistenceWriter(
        tmp_path / "old-recording.sqlite",
        CryptoKeyring({1: b"k" * 32}, active_version=1),
        utcnow=lambda: clock[0],
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    call_id, other = uuid4(), uuid4()
    recording = VoiceOperationV1(
        schema_version=1,
        operation_id=uuid4(),
        deployment_id="fixture",
        call_id=call_id,
        occurred_at=NOW,
        kind="recording.upsert",
        payload=RecordingUpsertPayloadV1(
            recording_id=uuid4(),
            status="saved",
            telnyx_recording_id="owned_recording-id",
            channels="dual",
            format="wav",
            started_at=NOW,
            ended_at=NOW,
            retention_until=NOW + timedelta(days=30),
        ),
    )
    handoff, remote_ack, deletes = [], [], []

    class Sink:
        def __init__(self):
            self.erased = (
                CallErasureLease(
                    1,
                    call_id,
                    uuid4(),
                    "fixture",
                    NOW + timedelta(days=30),
                    NOW + timedelta(seconds=30),
                ),
            )

        async def lease_call_erasures(self, *args):
            result, self.erased = self.erased, ()
            return result

        async def ack_call_erasure(self, *args):
            pass

        async def ingest(self, operation):
            handoff.append(operation)

        async def lease_recording_purges(self, *args):
            assert any(op == recording for op in handoff)
            return (
                RecordingPurgeLease(
                    1,
                    recording.payload.recording_id,
                    uuid4(),
                    recording.payload.telnyx_recording_id,
                    1,
                    NOW + timedelta(seconds=30),
                ),
            )

        async def ack_recording_purge(self, recording_id, token, outcome, occurred_at):
            remote_ack.append((recording_id, outcome))

    sink = Sink()

    async def stop(call):
        pass

    async def maintain():
        await maintain_call_content(writer, sink, stop, utcnow=lambda: NOW, timeout_seconds=2)

    async def degraded():
        pass

    async def provider(request):
        deletes.append((request.method, request.url.path))
        return httpx.Response(200, json={"data": {"id": "owned_recording-id"}})

    http = telnyx.DefaultAsyncHttpxClient(transport=httpx.MockTransport(provider), trust_env=False)
    monkeypatch.setattr(call_control.telnyx, "DefaultAsyncHttpxClient", lambda **kwargs: http)
    client = call_control.CallControlClient(api_key="owned-offline-api-key")
    # Materialize the installed SDK resource outside the unchanged purge deadline.
    _recordings_resource = client._client.recordings
    try:
        await writer.commit_control(PersistenceCommand("outbox", {"operation": recording}, None))
        clock[0] = NOW
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": publication(other)}, None)
        )
        relay = OutboxRelay(
            writer,
            sink,
            utcnow=lambda: NOW,
            before_fifo=maintain,
            on_degraded=degraded,
            drain=degraded,
        )
        result = await relay.run_once()
        assert result.status == "delivered" and result.acked == 1
        assert handoff == [recording, next(op for op in handoff if op.call_id == other)]
        assert await writer.oldest_outbox_created_at() is None
        purged = await purge_recordings_once(
            worker_id="owned-purge",
            lease_seconds=30,
            batch_size=1,
            telnyx=client,
            sink=sink,
            utcnow=lambda: NOW,
        )
        assert deletes == [("DELETE", "/v2/recordings/owned_recording-id")]
        assert purged.deleted == 1
        assert remote_ack == [(recording.payload.recording_id, "deleted")]
    finally:
        await client.aclose()
        await writer.drain(2)
        await task


@pytest.mark.asyncio
async def test_known_recording_commit_unknown_restart_defers_held_claim_without_retiming(tmp_path):
    import asyncio
    import sqlite3

    from projetv0_voice.models import RecordingUpsertPayloadV1
    from projetv0_voice.persistence.postgres_sink import OperationSinkCommitAmbiguousError

    clock = [NOW - timedelta(seconds=901)]
    keyring = CryptoKeyring({1: b"k" * 32}, active_version=1)
    database = tmp_path / "recording-replay.sqlite"
    writer = PersistenceWriter(database, keyring, utcnow=lambda: clock[0])
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    call_id, other = uuid4(), uuid4()
    recording = VoiceOperationV1(
        schema_version=1,
        operation_id=uuid4(),
        deployment_id="fixture",
        call_id=call_id,
        occurred_at=NOW,
        kind="recording.upsert",
        payload=RecordingUpsertPayloadV1(
            recording_id=uuid4(),
            status="saved",
            telnyx_recording_id="owned_recording-id",
            channels="dual",
            format="wav",
            started_at=NOW,
            ended_at=NOW,
            retention_until=NOW + timedelta(days=30),
        ),
    )
    attempts = []

    class Sink:
        async def lease_call_erasures(self, *args):
            return ()

        async def ingest(self, operation):
            attempts.append(operation)
            if len(attempts) == 1:
                raise OperationSinkCommitAmbiguousError("owned_recording_commit_unknown")

    sink = Sink()

    async def stop(call):
        pass

    async def fail():
        pytest.fail("held recording claim cannot trigger global degradation")

    def relay_for(current):
        async def maintain():
            await maintain_call_content(
                current, sink, stop, utcnow=lambda: clock[0], timeout_seconds=2
            )

        return OutboxRelay(
            current,
            sink,
            utcnow=lambda: clock[0],
            before_fifo=maintain,
            on_degraded=fail,
            drain=fail,
        )

    try:
        await writer.commit_control(PersistenceCommand("outbox", {"operation": recording}, None))
        clock[0] = NOW
        await writer.erase_call_content(call_id, now=NOW)
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": publication(other)}, None)
        )
        with pytest.raises(OperationSinkCommitAmbiguousError):
            await relay_for(writer).run_once()
        await writer.drain(2)
        await task
        with sqlite3.connect(database) as db:
            before = db.execute(
                "SELECT created_at,key_version,nonce,ciphertext FROM outbox WHERE op_id=?",
                (str(recording.operation_id),),
            ).fetchone()
        reopened = PersistenceWriter(database, keyring, utcnow=lambda: clock[0])
        owned = asyncio.create_task(reopened.run())
        assert await reopened.wait_ready()
        try:
            relay = relay_for(reopened)
            held = await relay.run_once()
            assert held.status == "retry_scheduled" and len(attempts) == 1
            with sqlite3.connect(database) as db:
                assert (
                    db.execute(
                        "SELECT created_at,key_version,nonce,ciphertext FROM outbox WHERE op_id=?",
                        (str(recording.operation_id),),
                    ).fetchone()
                    == before
                )
            clock[0] = NOW + timedelta(seconds=31)
            resumed = await relay.run_once()
            assert resumed.status == "delivered" and resumed.acked == 1
            assert attempts[:2] == [recording, recording]
            assert attempts[-1].call_id == other
            assert await reopened.oldest_outbox_created_at() is None
        finally:
            await reopened.drain(2)
            await owned
    finally:
        if not task.done():
            await writer.drain(2)
            await task
