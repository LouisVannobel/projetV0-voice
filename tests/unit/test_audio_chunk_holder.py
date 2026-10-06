"""Holder RED consumers: real local writer/AEAD; no controller or live audio proof."""

from __future__ import annotations

import asyncio
import base64
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import InputAudioRawFrame, StartFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio import audio_buffer_processor as native_audio
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from test_audio_writer import authenticate_audio

from projetv0_voice import audio_capture
from projetv0_voice.audio_contract import (
    AudioChunkPayloadV2,
    BeginCallSnapshotV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts, PersistenceWriter

NOW = datetime(2026, 10, 6, 10, tzinfo=UTC)
DEADLINE = NOW + timedelta(days=30)
CALL, WORKSPACE, RECORDING, GENERATION = (UUID(int=value) for value in (1, 2, 3, 4))


def snapshot():
    return BeginCallSnapshotV2.model_validate({
        "schema_version": 2, "workspace_id": str(WORKSPACE), "call_id": str(CALL),
        "configuration_revision": 7, "knowledge": {"business_name": "Holder fixture",
            "sector": "garage", "opening_hours": "", "services": "", "prices": "",
            "faq": "", "instructions": ""}, "transfer_destination": None,
        "retention_until": DEADLINE.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "recording_policy": "local_30d", "recording_contact_phone": "+33123456789",
        "audio_available": True, "recording_id": str(RECORDING),
    })


def holder(writer, keyring):
    assert hasattr(audio_capture, "AudioChunkHolder"), "missing coalescing AudioChunkHolder"
    return audio_capture.AudioChunkHolder(
        snapshot=snapshot(), deployment_id="agent-a", keyring=keyring, writer=writer
    )


@asynccontextmanager
async def owned_writer(path, **options):
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    writer = PersistenceWriter(
        path, keyring, contract_version=2, utcnow=lambda: NOW,
        process_agent_id="agent-a", process_deployment_id="agent-a", **options,
    )
    task = asyncio.create_task(writer.run())
    try:
        assert await writer.wait_ready()
        ticket = writer.submit_webhook(
            receipt={"event_id": "holder-admission", "event_type": "call.initiated",
                "call_control_id": "holder-control", "occurred_at": NOW, "received_at": NOW,
                "semantic_fingerprint_sha256": b"a" * 32},
            lease={"action": "upsert", "call_control_id": "holder-control", "call_id": CALL,
                "tenant_id": str(WORKSPACE), "agent_id": "agent-a", "state": "pending",
                "token_hash": b"b" * 32, "created_at": NOW,
                "expires_at": NOW + timedelta(hours=1), "closed_at": None},
            operation=None,
            admission_facts=LocalCallAdmissionFacts(
                CALL, NOW, DEADLINE, "holder-leg", "holder-session",
                admission_generation=GENERATION,
            ),
        )
        await ticket.wait()
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        await authenticate_audio(writer, call_control_id="holder-control", call_leg_id="holder-leg",
                                 call_session_id="holder-session")
        yield writer, keyring
    finally:
        await writer.drain(timeout_seconds=2)
        await asyncio.wait_for(task, 2)


class Forwarded(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)


@asynccontextmanager
async def owned_tap(owner, timing, release_pending):
    manager = TaskManager(loop=asyncio.get_running_loop())
    clock = SystemClock()
    clock.start()
    worker = PipelineWorker(Pipeline([]), clock=clock, task_manager=manager,
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
        enable_rtvi=False, enable_turn_tracking=False)
    tap = audio_capture.BoundedAudioBufferTap(
        offer_chunk=owner.offer_native_pcm, on_event_join=owner.after_event_join,
        ready_for_native_event=owner.ready_for_native_event, monotonic=lambda: timing.now,
        on_capture_refused=owner.notify_capture_refused,
    )
    forwarded = Forwarded()
    tap.link(forwarded)
    setup = FrameProcessorSetup(clock=clock, task_manager=manager, pipeline_worker=worker,
                                observer=None)
    await tap.setup(setup)
    await forwarded.setup(setup)
    await tap.queue_frame(StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=8000))
    await tap.start_capture()
    try:
        yield tap, forwarded, manager
    finally:
        release_pending()
        await asyncio.wait_for(tap.quiesce(), 2)
        await tap.cleanup()
        await FrameProcessor.cleanup(forwarded)
        await worker.cleanup()
        await asyncio.sleep(0)
        assert not manager.current_tasks()


async def emit_event(tap, timing):
    for _ in range(9):
        timing.now += 0.1
        await tap.queue_frame(InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1))


@pytest.mark.asyncio
async def test_holder_coalesces_three_native_sized_events_into_exact_real_writer_chunks(tmp_path):
    async with owned_writer(tmp_path / "holder.sqlite") as (writer, keyring):
        owner = holder(writer, keyring)
        events = [b"\x01\x00\x02\x00" * 7200, b"\x03\x00\x04\x00" * 7200,
                  b"\x05\x00\x06\x00" * 7200]
        for pcm in events:
            assert owner.offer_native_pcm(pcm, 8000, 2) is True
            assert await owner.after_event_join() is True
        summary = await owner.finish()
        assert (summary.submitted_samples, summary.committed_samples, summary.last_sequence) == (
            21600, 21600, 2
        )
        assert summary.reason == "complete" and not summary.partial and not summary.pending
        claimed = await writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=30)
        assert tuple(item.operation.kind for item in claimed[:2]) == ("call.upsert", "call.upsert")
        chunks = tuple(item for item in claimed if item.operation.kind == "audio.chunk")
        assert len(claimed) == len(chunks) + 2
        decoded, lengths = [], []
        for item in chunks:
            assert item.operation.kind == "audio.chunk"
            payload = item.operation.payload
            assert isinstance(payload, AudioChunkPayloadV2)
            assert payload.sequence == len(decoded)
            raw = keyring.decrypt(
                EncryptedValue(payload.key_version, base64.b64decode(payload.nonce_b64),
                               base64.b64decode(payload.ciphertext_b64)),
                aad=canonical_audio_chunk_aad(item.operation),
            )
            decoded.append(raw)
            lengths.append(len(raw))
            assert payload.retention_until == DEADLINE
        assert lengths == [32000, 32000, 22400]
        assert b"".join(decoded) == b"".join(events)
        assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_native_join_precedes_held_sqlite_receipt_without_blocking_subthreshold_frames(
    tmp_path, monkeypatch,
):
    timing = SimpleNamespace(now=10.0)
    monkeypatch.setattr(native_audio, "time", SimpleNamespace(monotonic=lambda: timing.now))
    event_entered, event_release = asyncio.Event(), asyncio.Event()
    commit_entered, commit_release = asyncio.Event(), asyncio.Event()
    native_tasks, sizes = [], []
    original_register = AudioBufferProcessor.add_event_handler
    first = True
    armed = False

    def register(recorder, name, handler):
        async def observed(sender, pcm, rate, channels):
            nonlocal first
            native_tasks.append(asyncio.current_task())
            sizes.append(len(pcm))
            if first:
                first = False
                event_entered.set()
                await event_release.wait()
            return await handler(sender, pcm, rate, channels)
        return original_register(recorder, name, observed if name == "on_audio_data" else handler)

    async def hold_commit(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            commit_entered.set()
            await commit_release.wait()

    monkeypatch.setattr(AudioBufferProcessor, "add_event_handler", register)
    async with owned_writer(tmp_path / "held.sqlite", failpoint=hold_commit) as (writer, keyring):
        owner = holder(writer, keyring)
        try:
            async with owned_tap(
                owner, timing, lambda: (event_release.set(), commit_release.set())
            ) as (tap, forwarded, _manager):
                await emit_event(tap, timing)
                await asyncio.wait_for(event_entered.wait(), 1)
                control = TextFrame("holder control")
                await tap.queue_frame(control)
                assert control in forwarded.frames and len(native_tasks) == 1
                event_release.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                armed = True
                await emit_event(tap, timing)
                await asyncio.wait_for(commit_entered.wait(), 1)
                assert len(native_tasks) == 2 and all(task.done() for task in native_tasks)
                assert owner.ready_for_native_event() is False
                for _ in range(10):
                    timing.now += 0.04
                    await tap.queue_frame(InputAudioRawFrame(b"\x02\x00" * 320, 8000, 1))
                await tap.queue_frame(control)
                assert tap.state == "recording" and control in forwarded.frames
                assert len(native_tasks) == 2
                for _ in range(13):
                    timing.now += 0.04
                    await tap.queue_frame(InputAudioRawFrame(b"\x02\x00" * 320, 8000, 1))
                assert tap.state == "partial" and len(native_tasks) == 2
                assert not commit_release.is_set()
                commit_release.set()
                assert await tap.quiesce()
                summary = await owner.finish()
                assert summary.submitted_samples == summary.committed_samples
                assert not summary.pending and max(sizes) <= 32000
                assert summary.partial
        finally:
            event_release.set()
            commit_release.set()


class ControlledWriter:
    """Counts only metadata; this double is not SQLite or producer qualification."""

    def __init__(self, *, refuse=False, unknown=False):
        self.refuse, self.unknown = refuse, unknown
        self.pending = None
        self.offered = self.committed_samples = 0
        self.last_sequence = None
        self.retention_until = None

    def offer_audio_chunk(self, operation):
        if self.refuse or self.pending is not None:
            return False
        payload = operation.payload
        assert isinstance(payload, AudioChunkPayloadV2)
        if self.retention_until is None:
            self.retention_until = payload.retention_until
        assert payload.retention_until == self.retention_until
        self.offered += 1
        self.last_sequence = payload.sequence
        self.pending = (operation.operation_id, payload.sample_count)
        return True

    async def wait_for_audio_commit(self, operation_id):
        assert self.pending is not None and self.pending[0] == operation_id
        if self.unknown:
            raise OSError("fixture unknown commit")
        self.committed_samples += self.pending[1]
        self.pending = None


@pytest.mark.asyncio
async def test_holder_exact_sample_ceiling_never_creates_sequence600_or_extends_original_pin():
    writer = ControlledWriter()
    owner = holder(writer, CryptoKeyring({1: bytes(range(32))}, active_version=1))
    pcm = b"\x01\x00\x02\x00" * 8000
    for _ in range(600):
        assert owner.offer_native_pcm(pcm, 8000, 2)
        assert await owner.after_event_join()
    assert owner.offer_native_pcm(b"\x01\x00\x02\x00", 8000, 2) is False
    summary = await owner.finish()
    assert writer.offered == 600 and writer.last_sequence == 599
    assert summary.submitted_samples == summary.committed_samples == 4_800_000
    assert summary.last_sequence == 599 and summary.reason == "limit" and not summary.pending
    assert writer.retention_until == DEADLINE


@pytest.mark.asyncio
async def test_tail_flush_unknown_commit_and_offer_refusal_never_claim_terminal_completion():
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    uncertain = ControlledWriter(unknown=True)
    owner = holder(uncertain, keyring)
    assert owner.offer_native_pcm(b"\x01\x00\x02\x00" * 7200, 8000, 2)
    assert uncertain.offered == 0
    summary = await owner.finish()
    assert uncertain.offered == 1 and uncertain.committed_samples == 0
    assert summary.submitted_samples == 7200 and summary.committed_samples == 0
    assert summary.partial and summary.pending
    assert owner.ready_for_native_event() is False
    refused = ControlledWriter(refuse=True)
    stopped = holder(refused, keyring)
    assert stopped.offer_native_pcm(b"\x01\x00\x02\x00" * 8000, 8000, 2) is False
    rejected = await stopped.finish()
    assert refused.offered == 0 and rejected.committed_samples == 0
    assert rejected.partial and not rejected.pending


@pytest.mark.asyncio
async def test_known_queued_erase_refusal_is_partial_but_never_an_unknown_pending_commit(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold_commit(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    path = tmp_path / "known-refusal.sqlite"
    async with owned_writer(path, failpoint=hold_commit) as (writer, keyring):
        owner = holder(writer, keyring)
        armed = True
        control = writer.submit_webhook(
            receipt={"event_id": "holder-held-control", "event_type": "fixture.control",
                "call_control_id": None, "occurred_at": NOW, "received_at": NOW,
                "semantic_fingerprint_sha256": b"d" * 32},
            lease=None, operation=None,
        )
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
            assert writer.queue_size == 1
            assert owner.offer_native_pcm(b"\x01\x00\x02\x00" * 8000, 8000, 2)
            release.set()
            await control_wait
            await erasing
            assert await owner.after_event_join() is False
            summary = await owner.finish()
            assert summary.partial and not summary.pending
            assert summary.committed_samples == 0
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
            assert not writer.is_degraded and await writer.quick_check()
            assert (await writer.read_call_lifecycle(CALL)).original_ended_at is None
        finally:
            release.set()
            if queue_check is not None:
                queue_check.cancel()
            await asyncio.gather(control_wait, erasing, return_exceptions=True)
