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
from tests.integration.test_disclosure import (
    _TEST_RUNTIME_METRICS,
    _DisclosureRelay,
    _local_choice_wait,
    _PacedDisclosureOutput,
    _uuids,
    pipeline_module,
    session_module,
)

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
async def capture_case(
    tmp_path, *, failpoint=None, terminal_publication=False, monotonic=None,
    recorder_factory=None, final_playback=False, runtime_metrics=None,
):
    assert hasattr(audio_capture, "LocalAudioCapture"), (
        "missing native LocalAudioCapture composition"
    )
    path = tmp_path / "capture.sqlite"
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    writer = PersistenceWriter(path, keyring, contract_version=2, utcnow=lambda: NOW,
                               failpoint=failpoint, process_agent_id="capture-deploy",
                               process_deployment_id="capture-deploy")
    writer_task = asyncio.create_task(writer.run())
    runtime = controller = capture = runner = None
    runtime_metrics = runtime_metrics or _TEST_RUNTIME_METRICS
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
        capture = audio_capture.LocalAudioCapture(
            snapshot=pin, deployment_id="capture-deploy", generation=GENERATION,
            keyring=keyring, writer=writer,
        )
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
        playback = pipeline_module.EndCallPlayback(
            generation=GENERATION, first_failure=failure
        ) if final_playback else None
        terminal_callbacks = {
            "local_audio_revoke": capture.revoke,
            "local_audio_finish": capture.finish,
            "local_audio_transfer_ready": capture.ready_for_transfer,
        } if terminal_publication else {}
        controller = session_module.DisclosureController(
            identity=identity, writer=writer, first_failure=failure,
            recording=NoProviderRecording(),
            recording_enabled=False, recording_required=False, mark_timeout_seconds=2,
            runtime_metrics=runtime_metrics, utcnow=lambda: NOW, uuid_factory=_uuids(500),
            on_active=became_active, local_audio_start=capture.start,
            local_audio_refuse=capture.refuse,
            local_audio_close=capture.close_admission,
            local_audio_quiesce=quiesce,
            **terminal_callbacks,
            **({"monotonic": monotonic} if monotonic is not None else {}),
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
        turn_recorder = (
            SimpleNamespace(record_user=lambda *_args: None, record_assistant=lambda *_args: None)
            if recorder_factory is None
            else recorder_factory(identity, writer, keyring, failure)
        )
        pipeline = pipeline_module.build_pipeline(
            transport=transport, services=services, controller=controller,
            turn_recorder=turn_recorder,
            first_failure=failure, begin_snapshot=pin, capture_tap=capture.tap,
            end_call_playback=playback,
        )
        runtime = pipeline_module.build_runtime(pipeline=pipeline, first_failure=failure,
            greeting=controller.announcement_text, mark_name=controller.mark_name,
            idle_timeout_seconds=60, observers=pipeline_module._CallObservers(
                runtime_metrics=runtime_metrics, stt=services.stt,
                llm=services.llm, tts=services.tts,
            ))
        await runtime.runner.add_workers(runtime.worker)
        runner = asyncio.create_task(runtime.runner.run(auto_end=True))
        yield SimpleNamespace(writer=writer, keyring=keyring, path=path, capture=capture,
            controller=controller, active=active, output=output, stt=stt, tts=services.tts,
            runtime=runtime,
            serializer=serializer, failure=failure, pin=pin, turn_recorder=turn_recorder,
            playback=playback, runner=runner)
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
    async with capture_case(tmp_path, failpoint=hold_sql) as case:
        try:
            await accept_local(case)
            await caller_frames(case, 9)
            await asyncio.wait_for(event_entered.wait(), 2)
            await case.runtime.worker.queue_frame(InputAudioRawFrame(b"\x20\x00" * 320, 8000, 1))
            await case.runtime.worker.queue_frame(TextFrame("phone control"))
            await _local_choice_wait(lambda: len(case.stt.received) == 10)
            assert case.controller.is_active() and len(events) == 1
            assert not event_release.is_set() and not commit_release.is_set()
            await _local_choice_wait(lambda: case.capture.tap.state == "partial")
            assert not event_release.is_set() and not commit_release.is_set()
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
            # Release held native/SQL work before the context joins its owned teardown.
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


@pytest.mark.asyncio
async def test_local_capture_farewell_waits_for_native_mark_and_drains_complete_sent_pcm(tmp_path):
    from pipecat.frames.frames import InputTransportMessageFrame

    from projetv0_voice.telnyx.frames import TelnyxMarkFrame

    async with capture_case(tmp_path, final_playback=True) as case:
        await accept_local(case)
        expected = await caller_frames(case, 1, samples=80)
        attempt = case.playback.begin()
        assert attempt is not None
        case.playback.before_tts_frame(attempt.speak_frame)
        case.playback.bind_context("farewell-context")
        case.playback.after_tts_frame(attempt.speak_frame)
        final_sent = asyncio.Event()

        async def send_final(frame):
            if isinstance(frame, TelnyxMarkFrame) and frame.mark_name == attempt.mark_name:
                final_sent.set()
            else:
                await original_send(frame)

        original_send = case.output.send_message
        case.output.send_message = send_final
        pcm = b"\x22\x00" * 320
        await case.runtime.worker.queue_frame(TTSAudioRawFrame(
            pcm, 8000, 1, context_id="farewell-context"))
        await case.runtime.worker.queue_frame(TelnyxMarkFrame(attempt.mark_name))
        await asyncio.wait_for(final_sent.wait(), 2)
        waiting = asyncio.create_task(case.playback.wait_for_ack(attempt, phase_timeout=2))
        await asyncio.sleep(0)
        assert not waiting.done() and not case.playback.acknowledged
        assert case.capture.tap.state == "recording"
        assert any(pcm in wire[0] for wire in case.output.sent) or pcm in b"".join(
            wire[0] for wire in case.output.sent)
        await case.runtime.worker.queue_frame(InputTransportMessageFrame(
            message={"event": "mark", "mark": {"name": attempt.mark_name}}))
        assert await waiting
        await case.controller.terminalize_and_join(
            cancel_continuations=False, normal_completion=True)
        summary = await case.capture.quiesce()
        assert not summary.pending and not summary.partial
        assert summary.reason == "complete"
        audio = captured_pcm(case)
        assert track(audio, 0).startswith(expected)
        assert pcm in track(audio, 1)
        assert case.failure.code is None


@pytest.mark.asyncio
async def test_native_session_conversation_end_keeps_local_finish_complete(tmp_path):
    from tests.integration.test_call_session import _session

    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        session, _, _, _, _ = _session(events=[])
        session._identity = case.controller._identity
        session._writer = case.writer
        session._utcnow = lambda: NOW + timedelta(seconds=5)
        session._controller = case.controller
        session.stop_result_inference()
        recorder = session_module.TurnRecorder(
            identity=session._identity, writer=case.writer, keyring=case.keyring,
            first_failure=case.failure, runtime_metrics=_TEST_RUNTIME_METRICS)
        await case.runtime.worker.queue_frame(EndFrame(reason="conversation_ended"))
        await asyncio.wait_for(case.runner, 2)
        terminal = session_module._TerminalOutcome("conversation_ended")
        await session._cleanup_owned_state(
            controller=case.controller, recorder=recorder, first_failure=case.failure,
            transport=SimpleNamespace(input=lambda: None, output=lambda: case.output),
            pipeline=case.runtime.pipeline, runtime=case.runtime, runner_task=case.runner,
            failure_task=None, terminal_outcome=terminal, cancel_continuations=False)
        assert case.failure.code is None
        assert terminal.reason == "conversation_ended"
        batch = await case.writer.read_relay_batch(
            batch_size=64, now=NOW + timedelta(seconds=5), lease_seconds=30)
        finish = [item.operation for item in batch if item.operation.kind == "audio.finish"]
        assert len(finish) == 1
        assert finish[0].payload.reason == "complete"
        assert finish[0].payload.total_samples >= 80


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "timeout", "transport", "segment_limit", "text_limit", "text_invalid", "drain_timeout"
])
async def test_refused_capture_keeps_first_producer_stt_diagnostic_and_no_late_restart(
    tmp_path, reason,
):
    from pipecat.frames.frames import FatalErrorFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from projetv0_voice.metrics import RuntimeMetrics
    from tests.unit.test_pipeline import _metric_map

    owner = RuntimeMetrics.in_memory()
    try:
        async with capture_case(tmp_path, terminal_publication=True, runtime_metrics=owner) as case:
            await accept_local(case)
            two = await case.serializer.deserialize(json.dumps({
                "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
                "dtmf": {"digit": "2"}}))
            await case.runtime.worker.queue_frame(two)
            await _local_choice_wait(lambda: case.controller._local_revoke_requested)
            await case.controller.join_continuations()
            assert case.capture.tap.state != "recording"
            late_one = await case.serializer.deserialize(json.dumps({
                "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "3",
                "occurred_at": NOW.isoformat().replace("+00:00", "Z"),
                "dtmf": {"digit": "1"}}))
            await case.runtime.worker.queue_frame(late_one)
            await case.controller.join_continuations()
            await caller_frames(case, 1, samples=80)
            assert captured_pcm(case) == b""
            observer = pipeline_module.RuntimeMetricsObserver(
                runtime_metrics=owner, stt=case.stt,
                llm=case.runtime.pipeline.processors[6], tts=case.tts)
            frame = FatalErrorFrame(error=f"openrouter_stt_{reason}", processor=case.stt)
            await observer.on_push_frame(FramePushed(
                case.stt, case.tts, frame, FrameDirection.UPSTREAM, 1, first_push=False))
            assert "projetv0.voice.stt.failures" not in _metric_map(owner)
            # Native producer dispatch supplies first_push=True before the sanitizing boundary.
            await case.stt.push_error_frame(frame)
            await _local_choice_wait(lambda: "projetv0.voice.stt.failures" in _metric_map(owner))
            points = list(_metric_map(owner)["projetv0.voice.stt.failures"].data.data_points)
            assert [(dict(point.attributes), point.value) for point in points] == [
                ({"reason": reason}, 1)]
            assert case.capture.tap.state != "recording"
            assert captured_pcm(case) == b""
            assert owner.failure_code is None
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_transfer_capture_closes_synchronously_and_persists_native_tail(tmp_path):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        expected = await caller_frames(case, 1, samples=80)
        assert case.capture.tap.state == "recording"
        assert case.controller.close_local_audio_for_transfer() is True
        assert case.capture.tap.state == "stopped"
        assert case.controller.is_active()
        # An actual late pipeline frame still reaches native STT but cannot join capture.
        await caller_frames(case, 2, samples=80)
        assert await case.controller.prepare_audio_for_transfer() is True
        assert track(captured_pcm(case), 0) == expected
        summary = case.capture.holder.summary
        assert summary.submitted_samples == summary.committed_samples == 80
        assert not summary.pending and not case.capture.tap.pending_join
        with sqlite3.connect(case.path) as db:
            first = db.execute(
                "SELECT op_id,fingerprint FROM local_audio_terminal WHERE kind='audio.finish'"
            ).fetchone()
        assert first is not None
        facts = await case.writer.read_call_lifecycle(CALL)
        assert not facts.transfer_fenced and facts.original_ended_at is None
        assert facts.retention_until == case.pin.retention_until
        assert await case.controller.prepare_audio_for_transfer() is True
        await case.controller.cleanup_termination("qualified_line_connected")
        with sqlite3.connect(case.path) as db:
            assert db.execute(
                "SELECT op_id,fingerprint FROM local_audio_terminal WHERE kind='audio.finish'"
            ).fetchall() == [first]


@pytest.mark.asyncio
async def test_transfer_capture_waits_native_tail_receipt_before_returning(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(name):
        if name == "after_mutation_before_commit" and armed:
            entered.set()
            await release.wait()

    armed = False
    async with capture_case(tmp_path, terminal_publication=True, failpoint=hold) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        assert case.controller.close_local_audio_for_transfer() is True
        armed = True
        prepared = asyncio.create_task(case.controller.prepare_audio_for_transfer())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert not prepared.done()
            assert case.capture.tap.state == "stopped"
            assert case.controller.is_active()
            assert case.capture.holder.summary.pending
        finally:
            armed = False
            release.set()
            assert await asyncio.wait_for(prepared, 2) is True


@pytest.mark.asyncio
async def test_transfer_capture_missing_native_finish_is_unavailable(tmp_path):
    # Actual native recorder/writer; the deliberately missing finish composition is the fault.
    async with capture_case(tmp_path, terminal_publication=False) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        assert case.controller.close_local_audio_for_transfer() is True
        assert await case.controller.prepare_audio_for_transfer() is False
        assert case.controller.is_active()
        assert case.capture.tap.state == "stopped"
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT count(*) FROM local_audio_terminal").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_transfer_capture_does_not_certify_existing_call_failure(tmp_path):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        case.failure.signal("pipeline_task_failed")
        assert case.controller.close_local_audio_for_transfer() is False
        assert case.capture.tap.state == "stopped"
        assert await case.controller.prepare_audio_for_transfer() is False


@pytest.mark.asyncio
async def test_transfer_capture_opposition_during_actual_finish_joins_real_revoke(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    remaining = 0

    async def hold_finish(name):
        nonlocal remaining
        if name == "after_mutation_before_commit" and remaining:
            remaining -= 1
            if remaining == 0:
                entered.set()
                await release.wait()

    async with capture_case(tmp_path, terminal_publication=True, failpoint=hold_finish) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        assert case.controller.close_local_audio_for_transfer() is True
        # The actual first transaction commits the flushed tail; the second is audio.finish.
        remaining = 2
        prepared = asyncio.create_task(case.controller.prepare_audio_for_transfer())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert not prepared.done()
            assert case.capture.holder.summary.committed_samples == 80
            frame = await case.serializer.deserialize(json.dumps({
                "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
                "occurred_at": NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "dtmf": {"digit": "2"},
            }))
            await case.runtime.worker.queue_frame(frame)
            await _local_choice_wait(lambda: case.controller._local_revoke_requested)
            assert not prepared.done() and case.controller.is_active()
        finally:
            release.set()
        assert await asyncio.wait_for(prepared, 2) is True
        await case.controller.join_continuations()
        assert not case.capture.tap.pending_join and not case.capture.holder.summary.pending
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
            assert db.execute(
                "SELECT kind FROM outbox WHERE kind IN "
                "('audio.chunk','audio.finish','audio.revoke')"
            ).fetchall() == [("audio.revoke",)]
            assert db.execute(
                "SELECT count(*) FROM local_audio_terminal WHERE kind='audio.revoke'"
            ).fetchone()[0] == 1
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_transfer_partial_capture_does_not_certify_normal_tail(tmp_path):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        case.capture.refuse()
        assert case.capture.tap.state == "partial"
        assert await case.controller.prepare_audio_for_transfer() is False
        assert not case.capture.ready_for_transfer()
        assert not case.controller.audio_ready_for_transfer()
        assert not case.capture.tap.pending_join and not case.capture.holder.summary.pending
        assert case.capture.holder.summary.partial
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_transfer_earlier_frozen_limit_does_not_become_transfer(tmp_path):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        assert not (await case.capture.finish("limit")).pending
        with sqlite3.connect(case.path) as db:
            original = db.execute(
                "SELECT op_id,fingerprint FROM local_audio_terminal WHERE kind='audio.finish'"
            ).fetchone()
        assert original is not None
        assert await case.controller.prepare_audio_for_transfer() is False
        assert not case.capture.ready_for_transfer()
        with sqlite3.connect(case.path) as db:
            assert db.execute(
                "SELECT op_id,fingerprint FROM local_audio_terminal WHERE kind='audio.finish'"
            ).fetchall() == [original]
        assert (await case.writer.read_call_lifecycle(CALL)).original_ended_at is None


@pytest.mark.asyncio
async def test_transfer_readiness_invalidates_before_held_denial_lock(tmp_path):
    async with capture_case(tmp_path, terminal_publication=True) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        assert await case.controller.prepare_audio_for_transfer() is True
        assert case.controller.audio_ready_for_transfer()
        frame = await case.serializer.deserialize(json.dumps({
            "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
            "occurred_at": NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "dtmf": {"digit": "2"},
        }))
        await case.controller._lock.acquire()
        denial = asyncio.create_task(case.controller.accept_dtmf(frame))
        try:
            await _local_choice_wait(lambda: case.controller._local_revoke_requested)
            assert not denial.done()
            assert case.controller.pending_task_count == 0
            assert not case.controller.audio_ready_for_transfer()
        finally:
            case.controller._lock.release()
        assert await asyncio.wait_for(denial, 2)
        await case.controller.join_continuations()
        assert case.controller.audio_ready_for_transfer()


@pytest.mark.asyncio
async def test_cancelled_transfer_waiter_preserves_real_owned_denial(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        if name == "after_mutation_before_commit" and armed:
            entered.set()
            await release.wait()

    async with capture_case(tmp_path, terminal_publication=True, failpoint=hold) as case:
        await accept_local(case)
        await caller_frames(case, 1, samples=80)
        armed = True
        frame = await case.serializer.deserialize(json.dumps({
            "event": "dtmf", "stream_id": "capture-stream", "sequence_number": "2",
            "occurred_at": NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "dtmf": {"digit": "2"},
        }))
        await case.runtime.worker.queue_frame(frame)
        await asyncio.wait_for(entered.wait(), 2)
        owned = case.controller._local_denial_task
        assert owned is not None and not owned.done()
        joined = asyncio.Event()
        original_join = case.controller._join_audio_transfer_continuations

        async def observe_join():
            joined.set()
            await original_join()

        case.controller._join_audio_transfer_continuations = observe_join
        prepared = asyncio.create_task(case.controller.prepare_audio_for_transfer())
        try:
            await asyncio.wait_for(joined.wait(), 2)
            prepared.cancel()
            with pytest.raises(asyncio.CancelledError):
                await prepared
            assert not owned.done() and owned.cancelling() == 0
            assert not case.controller.audio_ready_for_transfer()
        finally:
            armed = False
            release.set()
        await case.controller.join_continuations()
        assert not owned.cancelled() and owned.exception() is None
        assert case.controller.audio_ready_for_transfer()
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
            assert db.execute(
                "SELECT count(*) FROM local_audio_terminal WHERE kind='audio.revoke'"
            ).fetchone()[0] == 1
