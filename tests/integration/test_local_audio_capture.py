"""Native local composition; synthetic PCM and offline inference are test data."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from test_disclosure import (
    _TEST_RUNTIME_METRICS,
    _DisclosureRelay,
    _local_choice_wait,
    _PacedDisclosureOutput,
    _uuids,
    pipeline_module,
    session_module,
)

from projetv0_voice import audio_capture
from projetv0_voice.admission import CallGenerationHandle, ProcessLeaseClaim
from projetv0_voice.audio_contract import (
    AudioChunkPayloadV2,
    BeginCallSnapshotV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.models import RoutingV1
from projetv0_voice.persistence.commands import decode_operation_v2, operation_aad_from_metadata
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts, PersistenceWriter
from projetv0_voice.telnyx.serializer import AudioAdmission, ProjetV0TelnyxFrameSerializer

NOW = datetime.now(UTC).replace(microsecond=0)
CALL, WORKSPACE, RECORDING, GENERATION = (UUID(int=value) for value in (1, 2, 3, 4))


class OfflineNativeSTT(STTService):
    """Only inference is replaced; native default raw-audio passthrough stays active."""

    def __init__(self):
        super().__init__(sample_rate=8000, settings=STTSettings(model=None, language=None))
        self.received = []

    async def process_frame(self, frame, direction):
        if isinstance(frame, InputAudioRawFrame):
            self.received.append(frame.audio)
        await super().process_frame(frame, direction)

    async def run_stt(self, audio):
        yield None


class SentOutput(_PacedDisclosureOutput):
    def __init__(self):
        super().__init__()
        self.fail_next = False
        self.failed = []
        self.sent = []
        self.input_sent = []
        self.output_sent = []
        self.tts_stops = 0

        async def after_push(_output, frame):
            if isinstance(frame, InputAudioRawFrame):
                self.input_sent.append(frame.audio)
            elif isinstance(frame, OutputAudioRawFrame):
                self.output_sent.append(frame.audio)
            elif isinstance(frame, TTSStoppedFrame):
                self.tts_stops += 1

        self.add_event_handler("on_after_push_frame", after_push)

    async def write_audio_frame(self, frame):
        if self.fail_next:
            self.fail_next = False
            self.failed.append(frame.audio)
            return False
        result = await super().write_audio_frame(frame)
        if result:
            self.sent.append((frame.audio, frame.sample_rate, frame.num_channels))
        return result


@asynccontextmanager
async def capture_case(tmp_path, *, failpoint=None):
    assert hasattr(audio_capture, "LocalAudioCapture"), (
        "missing native LocalAudioCapture composition"
    )
    path = tmp_path / "capture.sqlite"
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    writer = PersistenceWriter(path, keyring, contract_version=2, utcnow=lambda: NOW,
                               failpoint=failpoint)
    writer_task = asyncio.create_task(writer.run())
    runtime = controller = capture = runner = None
    try:
        assert await writer.wait_ready()
        pin = BeginCallSnapshotV2.model_validate({
            "schema_version": 2, "workspace_id": str(WORKSPACE), "call_id": str(CALL),
            "configuration_revision": 7, "knowledge": {"business_name": "Capture fixture",
                "sector": "garage", "opening_hours": "", "services": "", "prices": "",
                "faq": "", "instructions": ""}, "transfer_destination": None,
            "retention_until": (NOW + timedelta(days=30)).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "recording_policy": "local_30d", "recording_contact_phone": "+33102030405",
            "audio_available": True, "recording_id": str(RECORDING),
        })
        routing = RoutingV1(schema_version=1, direction="incoming", connection_id="fixture",
            to_e164="+33102030405", from_e164=None, telnyx_call_control_id="capture-control",
            telnyx_call_leg_id="capture-leg", telnyx_call_session_id="capture-session",
            admitted_at=NOW)
        identity = session_module.CallIdentity(
            call_id=CALL, generation=CallGenerationHandle("capture-control", GENERATION),
            lease_claim=ProcessLeaseClaim(call_control_id="capture-control", call_id=CALL,
                generation=GENERATION, token_digest=b"k" * 32, claimed_at=NOW),
            deployment_id="capture-deploy", telnyx_call_control_id="capture-control",
            telnyx_call_leg_id="capture-leg", telnyx_call_session_id="capture-session",
            stream_id="capture-stream", started_at=NOW, retention_until=pin.retention_until,
            routing=routing, begin_snapshot=pin,
        )
        ticket = writer.submit_webhook(
            receipt={"event_id": "capture-admission", "event_type": "call.initiated",
                "call_control_id": "capture-control", "occurred_at": NOW, "received_at": NOW,
                "semantic_fingerprint_sha256": b"a" * 32},
            lease={"action": "upsert", "call_control_id": "capture-control", "call_id": CALL,
                "tenant_id": str(WORKSPACE), "agent_id": "capture-deploy", "state": "pending",
                "token_hash": b"k" * 32, "created_at": NOW,
                "expires_at": NOW + timedelta(hours=1), "closed_at": None},
            operation=None, admission_facts=LocalCallAdmissionFacts(
                CALL, NOW, pin.retention_until, "capture-leg", "capture-session",
                admission_generation=GENERATION),
        )
        await ticket.wait()
        await writer.bind_audio_snapshot(pin, generation=GENERATION)
        capture = audio_capture.LocalAudioCapture(snapshot=pin, deployment_id="capture-deploy",
                                                 keyring=keyring, writer=writer)
        active = asyncio.Event()

        async def became_active():
            active.set()

        async def quiesce():
            await capture.quiesce()

        class NoProviderRecording:
            async def start(self, identity):
                pytest.fail("V2 local capture invoked provider recording start")

            async def cleanup(self, identity, **options):
                pytest.fail("V2 local capture invoked provider recording cleanup")

        failure = pipeline_module.FirstFailure()
        controller = session_module.DisclosureController(
            identity=identity, writer=writer, first_failure=failure,
            recording=NoProviderRecording(),
            recording_enabled=False, recording_required=False, mark_timeout_seconds=2,
            runtime_metrics=_TEST_RUNTIME_METRICS, utcnow=lambda: NOW, uuid_factory=_uuids(500),
            on_active=became_active, local_audio_start=capture.start,
            local_audio_refuse=capture.refuse,
            local_audio_close=capture.close_admission,
            local_audio_quiesce=quiesce,
        )
        admission = AudioAdmission()
        admission.bind(controller.is_active)
        serializer = ProjetV0TelnyxFrameSerializer("capture-stream",
            expected_call_control_id="capture-control", audio_admission=admission)
        output = SentOutput()
        output.ack_during_send = controller
        stt = OfflineNativeSTT()
        services = SimpleNamespace(
            stt=stt, llm=_DisclosureRelay(), tts=_DisclosureRelay(synthesize=True)
        )
        input_processor = _DisclosureRelay()
        transport = SimpleNamespace(input=lambda: input_processor, output=lambda: output)
        pipeline = pipeline_module.build_pipeline(
            transport=transport, services=services, controller=controller,
            turn_recorder=SimpleNamespace(record_user=lambda *_args: None,
                                         record_assistant=lambda *_args: None),
            first_failure=failure, begin_snapshot=pin, capture_tap=capture.tap,
        )
        runtime = pipeline_module.build_runtime(pipeline=pipeline, first_failure=failure,
            greeting=controller.announcement_text, mark_name=controller.mark_name,
            idle_timeout_seconds=60, observers=pipeline_module._CallObservers(
                runtime_metrics=_TEST_RUNTIME_METRICS, stt=services.stt,
                llm=services.llm, tts=services.tts,
            ))
        await runtime.runner.add_workers(runtime.worker)
        runner = asyncio.create_task(runtime.runner.run(auto_end=True))
        yield SimpleNamespace(writer=writer, keyring=keyring, path=path, capture=capture,
            controller=controller, active=active, output=output, stt=stt, tts=services.tts,
            runtime=runtime,
            serializer=serializer, failure=failure, pin=pin)
    finally:
        if controller is not None:
            await controller.terminalize_and_join(cancel_continuations=True)
        if capture is not None:
            await capture.quiesce()
        if runner is not None and not runner.done():
            await runtime.runner.cancel(reason="capture_fixture_cleanup")
            await asyncio.wait_for(runner, 2)
        if writer.is_degraded:
            await asyncio.wait_for(asyncio.gather(writer_task, return_exceptions=True), 2)
        else:
            await writer.drain(2)
            await asyncio.wait_for(writer_task, 2)


async def accept_local(case):
    await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
    frame = await case.serializer.deserialize(json.dumps({
        "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "1",
        "occurred_at": NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "dtmf": {"digit": "1"},
    }))
    await case.runtime.worker.queue_frame(frame)
    await asyncio.wait_for(case.active.wait(), 2)


async def caller_frames(case, count, samples=800):
    pcm = b"\x10\x00" * samples
    for _ in range(count):
        await case.runtime.worker.queue_frame(InputAudioRawFrame(pcm, 8000, 1))
    await _local_choice_wait(lambda: len(case.stt.received) >= count)
    await _local_choice_wait(lambda: len(case.output.input_sent) >= count)
    return pcm * count


def captured_pcm(case):
    decoded = []
    with sqlite3.connect(case.path) as db:
        rows = db.execute(
            "SELECT schema_version,op_id,deployment_id,call_id,kind,key_version,nonce,ciphertext "
            "FROM outbox WHERE kind='audio.chunk' ORDER BY queue_id"
        ).fetchall()
    for schema, operation_id, deployment, call, kind, key, nonce, cipher in rows:
        operation = decode_operation_v2(case.keyring.decrypt(EncryptedValue(key, nonce, cipher),
            aad=operation_aad_from_metadata({"schema_version": schema, "operation_id": operation_id,
                "deployment_id": deployment, "call_id": call, "kind": kind})))
        payload = operation.payload
        assert isinstance(payload, AudioChunkPayloadV2)
        assert payload.retention_until == case.pin.retention_until
        decoded.append(case.keyring.decrypt(EncryptedValue(payload.key_version,
            base64.b64decode(payload.nonce_b64), base64.b64decode(payload.ciphertext_b64)),
            aad=canonical_audio_chunk_aad(operation)))
    return b"".join(decoded)


def track(pcm, channel):
    return b"".join(pcm[index + channel * 2:index + channel * 2 + 2]
                    for index in range(0, len(pcm), 4))


@pytest.mark.asyncio
async def test_local_capture_caller_passthrough_starts_only_after_real_choice_and_gate(tmp_path):
    async with capture_case(tmp_path) as case:
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        assert captured_pcm(case) == b"" and case.capture.tap.state == "off"
        await case.runtime.worker.queue_frame(InputAudioRawFrame(b"\x7f\x00" * 80, 8000, 1))
        await asyncio.sleep(0.01)
        assert case.stt.received == [] and captured_pcm(case) == b""
        await accept_local(case)
        expected = await caller_frames(case, 10)
        await _local_choice_wait(lambda: case.capture.tap.state == "recording")
        summary = await case.capture.quiesce()
        assert summary.submitted_samples == summary.committed_samples == 8000
        assert not summary.pending and not summary.partial
        assert track(captured_pcm(case), 0) == expected
        assert b"\x7f\x00" * 80 not in track(captured_pcm(case), 0)
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_local_capture_uses_actual_resampled_sent_output_and_omits_failed_write(tmp_path):
    async with capture_case(tmp_path) as case:
        await accept_local(case)
        announcement_count = len(case.output.sent)
        stop_count = case.output.tts_stops
        case.output.fail_next = True
        await case.tts.queue_frame(TTSAudioRawFrame(
            b"\x40\x00" * 12000, 24000, 1, context_id="captured-native-output"
        ))
        await case.tts.queue_frame(TTSStoppedFrame(context_id="captured-native-output"))
        await _local_choice_wait(lambda: case.output.tts_stops > stop_count)
        assert len(case.output.sent) > announcement_count and case.output.failed
        assert len(case.output.output_sent) == len(case.output.sent)
        summary = await case.capture.quiesce()
        sent = case.output.sent[announcement_count:]
        assert sent and all(rate == 8000 and channels == 1 for _pcm, rate, channels in sent)
        assert len(case.output.failed) == 1
        assert track(captured_pcm(case), 1) == b"".join(pcm for pcm, _rate, _channels in sent)
        assert summary.committed_samples > 0 and not summary.pending


@pytest.mark.asyncio
async def test_local_capture_held_native_event_and_sql_receipt_keep_phone_progress(
    tmp_path, monkeypatch,
):
    event_entered, event_release = asyncio.Event(), asyncio.Event()
    commit_entered, commit_release = asyncio.Event(), asyncio.Event()
    events = []
    armed = False
    native_register = AudioBufferProcessor.add_event_handler

    def observe(recorder, name, handler):
        async def held(sender, pcm, rate, channels):
            events.append(asyncio.current_task())
            event_entered.set()
            await event_release.wait()
            return await handler(sender, pcm, rate, channels)
        return native_register(recorder, name, held if name == "on_audio_data" else handler)

    async def hold_sql(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            commit_entered.set()
            await commit_release.wait()

    monkeypatch.setattr(AudioBufferProcessor, "add_event_handler", observe)
    try:
        async with capture_case(tmp_path, failpoint=hold_sql) as case:
            await accept_local(case)
            await caller_frames(case, 9)
            await asyncio.wait_for(event_entered.wait(), 2)
            await case.runtime.worker.queue_frame(InputAudioRawFrame(b"\x20\x00" * 320, 8000, 1))
            await case.runtime.worker.queue_frame(TextFrame("phone control"))
            await _local_choice_wait(lambda: len(case.stt.received) == 10)
            assert case.controller.is_active() and len(events) == 1
            assert case.capture.tap.state == "partial"
            event_release.set()
            armed = True
            closing = asyncio.create_task(case.capture.quiesce())
            try:
                await asyncio.wait_for(commit_entered.wait(), 2)
                await case.runtime.worker.queue_frame(InputAudioRawFrame(b"\x30\x00" * 80, 8000, 1))
                await _local_choice_wait(lambda: len(case.stt.received) == 11)
                assert case.controller.is_active() and len(events) == 1
            finally:
                commit_release.set()
            summary = await closing
            assert summary.partial and not summary.pending
            assert summary.committed_samples == summary.submitted_samples
    finally:
        event_release.set()
        commit_release.set()


@pytest.mark.asyncio
async def test_local_capture_normal_tail_opposition_unknown_receipt_has_no_remote_completion(
    tmp_path, monkeypatch,
):
    async with capture_case(tmp_path / "normal") as case:
        await accept_local(case)
        expected = await caller_frames(case, 1, samples=80)
        case.capture.tap.close_admission()
        assert case.capture.tap.state == "stopped"
        summary = await case.capture.quiesce()
        assert not summary.partial and not summary.pending and summary.committed_samples == 80
        assert track(captured_pcm(case), 0) == expected
        assert await case.capture.quiesce() == summary
    async with capture_case(tmp_path / "opposition") as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        case.capture.refuse()
        assert case.capture.tap.state == "partial"
        summary = await case.capture.quiesce()
        assert summary.partial and not summary.pending
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None
    async with capture_case(tmp_path / "unknown") as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        native_wait = case.writer.wait_for_audio_commit

        async def unknown(operation_id):
            raise OSError("owned-unknown-audio-receipt")

        monkeypatch.setattr(case.writer, "wait_for_audio_commit", unknown)
        first = await case.capture.quiesce()
        await case.writer.wait_until_idle()
        with sqlite3.connect(case.path) as db:
            frozen = db.execute(
                "SELECT op_id,nonce,ciphertext FROM outbox WHERE kind='audio.chunk'"
            ).fetchall()
        assert first.pending and first.partial and first.committed_samples == 0
        assert (await case.capture.quiesce()).pending
        with sqlite3.connect(case.path) as db:
            assert db.execute(
                "SELECT op_id,nonce,ciphertext FROM outbox WHERE kind='audio.chunk'"
            ).fetchall() == frozen
            assert db.execute(
                "SELECT count(*) FROM outbox WHERE kind IN ('audio.finish','audio.revoke')"
            ).fetchone()[0] == 0
        monkeypatch.setattr(case.writer, "wait_for_audio_commit", native_wait)
        settled = await case.capture.quiesce()
        assert not settled.pending and settled.committed_samples == 80
        await case.runtime.worker.queue_frame(EndFrame())


@pytest.mark.asyncio
async def test_local_capture_normal_controller_retirement_preserves_complete_tail(tmp_path):
    async with capture_case(tmp_path) as case:
        await accept_local(case)
        expected = await caller_frames(case, 1, samples=80)
        assert case.capture.tap.state == "recording"
        # The actual CallSession cleanup supplies this only for its known "closed" outcome.
        await case.controller.terminalize_and_join(
            cancel_continuations=False, normal_completion=True
        )
        assert not case.controller.is_active() and case.controller.pending_task_count == 0
        summary = await case.capture.quiesce()
        assert summary.submitted_samples == summary.committed_samples == 80
        assert not summary.pending and not summary.partial and summary.reason == "complete"
        assert track(captured_pcm(case), 0) == expected
        assert await case.capture.quiesce() == summary
        facts = await case.writer.read_call_lifecycle(CALL)
        assert facts.original_ended_at is None and not facts.content_erased
