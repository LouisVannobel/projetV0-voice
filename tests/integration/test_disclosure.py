from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from types import SimpleNamespace
from uuid import UUID

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from starlette.websockets import WebSocket, WebSocketState

from projetv0_voice.admission import CallGenerationHandle, ProcessLeaseClaim
from projetv0_voice.audio_contract import BeginCallSnapshotV2
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.models import BeginCallSnapshotV1, RoutingV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts, PersistenceWriter
from projetv0_voice.telnyx.frames import TelnyxMarkFrame
from projetv0_voice.telnyx.serializer import AudioAdmission, ProjetV0TelnyxFrameSerializer

pipeline_module = import_module("projetv0_voice.pipeline")
session_module = import_module("projetv0_voice.session")

NOW = datetime(2026, 8, 28, 18, 0, tzinfo=UTC)
_TEST_RUNTIME_METRICS = RuntimeMetrics.in_memory()


class _Writer:
    def __init__(self, *, fail: bool = False, block: bool = False) -> None:
        self.fail = fail
        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.commands: list[PersistenceCommand] = []

    async def commit_control(self, command: PersistenceCommand) -> None:
        self.commands.append(command)
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("storage-secret")


class _TimeoutWriter(_Writer):
    async def commit_control(self, command: PersistenceCommand) -> None:
        self.commands.append(command)
        self.started.set()
        raise TimeoutError("storage-timeout-secret")


class _Recording:
    def __init__(
        self,
        outcome: object,
        *,
        block_after_accept: bool = False,
    ) -> None:
        self.outcome = outcome
        self.block_after_accept = block_after_accept
        self.starts = 0
        self.accepted = asyncio.Event()
        self.never = asyncio.Event()
        self.cleanups = 0

    async def start(self, _identity: object) -> object:
        self.starts += 1
        self.accepted.set()
        if self.block_after_accept:
            await self.never.wait()
        return self.outcome

    async def cleanup(
        self,
        _identity: object,
        *,
        recording_may_be_active: bool,
        reason: str,
    ) -> None:
        del recording_may_be_active, reason
        self.cleanups += 1


def _identity(call_int: int = 1) -> object:
    generation = UUID(int=call_int)
    return session_module.CallIdentity(
        call_id=UUID(int=call_int),
        generation=CallGenerationHandle(f"call-control-{call_int}", generation),
        lease_claim=ProcessLeaseClaim(
            call_control_id=f"call-control-{call_int}",
            call_id=UUID(int=call_int),
            generation=generation,
            token_digest=bytes([call_int]) * 32,
            claimed_at=NOW,
        ),
        deployment_id=f"deployment-{call_int}",
        telnyx_call_control_id=f"call-control-{call_int}",
        telnyx_call_leg_id=f"call-leg-{call_int}",
        telnyx_call_session_id=f"call-session-{call_int}",
        stream_id=f"stream-{call_int}",
        started_at=NOW,
        retention_until=NOW + timedelta(days=7),
    )


def _uuids(start: int = 100):
    current = start

    def factory() -> UUID:
        nonlocal current
        value = UUID(int=current)
        current += 1
        return value

    return factory


def _controller(
    *,
    writer: _Writer | None = None,
    recording: _Recording | None = None,
    recording_enabled: bool = False,
    recording_required: bool = False,
    timeout: float = 0.05,
    call_int: int = 1,
    identity: object | None = None,
) -> tuple[object, _Writer, _Recording, object]:
    selected_writer = writer or _Writer()
    selected_recording = recording or _Recording(
        session_module.RecordingStartResult(
            session_module.RecordingStartState.DEFINITELY_NOT_STARTED
        )
    )
    failure = pipeline_module.FirstFailure()
    controller = session_module.DisclosureController(
        identity=identity or _identity(call_int),
        writer=selected_writer,
        first_failure=failure,
        recording=selected_recording,
        recording_enabled=recording_enabled,
        recording_required=recording_required,
        mark_timeout_seconds=timeout,
        runtime_metrics=_TEST_RUNTIME_METRICS,
        utcnow=lambda: NOW + timedelta(seconds=1),
        uuid_factory=_uuids(100 + call_int * 10),
    )
    return controller, selected_writer, selected_recording, failure


def _pinned_identity(enabled: bool):
    snapshot = BeginCallSnapshotV1.model_validate({
        "schema_version": 1, "call_id": str(UUID(int=1)), "configuration_revision": 7,
        "knowledge": {"business_name": "Garage", "sector": "garage", "opening_hours": "",
                      "services": "", "prices": "", "faq": "", "instructions": ""},
        "transfer_destination": None, "retention_until": "2026-09-27T18:00:00.000Z",
        "recording_enabled": enabled,
    })
    routing = RoutingV1(
        schema_version=1, direction="incoming", connection_id="fixture",
        to_e164="+33102030405", from_e164=None, telnyx_call_control_id="call-control-1",
        telnyx_call_leg_id="call-leg-1", telnyx_call_session_id="call-session-1",
        admitted_at=NOW,
    )
    return replace(_identity(), begin_snapshot=snapshot, retention_until=snapshot.retention_until,
                   routing=routing)


class _DisclosureRelay(FrameProcessor):
    def __init__(self, *, synthesize: bool = False) -> None:
        super().__init__(enable_direct_mode=True)
        self.synthesize = synthesize
        self.input_audio: list[InputAudioRawFrame] = []
        self.input_dtmf: list[InputDTMFFrame] = []
        self.spoken: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.input_audio.append(frame)
        if isinstance(frame, InputDTMFFrame):
            self.input_dtmf.append(frame)
        if isinstance(frame, TTSSpeakFrame):
            self.spoken.append(frame.text)
        if self.synthesize and isinstance(frame, TTSSpeakFrame):
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=b"\x01\x00" * 1920,
                    sample_rate=8000,
                    num_channels=1,
                    context_id="paced-disclosure",
                ),
                direction,
            )
            await self.push_frame(TTSStoppedFrame(context_id="paced-disclosure"), direction)
            return
        await self.push_frame(frame, direction)


class _PacedDisclosureOutput(BaseOutputTransport):
    """Keep native chunking/MediaSender; replace only the offline wire sinks."""

    def __init__(self, *, ack_during_send: object | None = None) -> None:
        super().__init__(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=8000,
                audio_out_channels=1,
                audio_out_10ms_chunks=2,
                audio_out_end_silence_secs=0,
            )
        )
        self.ack_during_send = ack_during_send
        self.mark_enqueued = asyncio.Event()
        self.mark_sent = asyncio.Event()
        self.mark_pushed = asyncio.Event()
        self.audio_written = bytearray()

        async def after_process(_output: FrameProcessor, frame: Frame) -> None:
            if isinstance(frame, TelnyxMarkFrame):
                self.mark_enqueued.set()

        async def after_push(_output: FrameProcessor, frame: Frame) -> None:
            if isinstance(frame, TelnyxMarkFrame):
                self.mark_pushed.set()

        self.add_event_handler("on_after_process_frame", after_process)
        self.add_event_handler("on_after_push_frame", after_push)

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        await asyncio.sleep(len(frame.audio) / (frame.sample_rate * frame.num_channels * 2))
        self.audio_written.extend(frame.audio)
        return True

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ) -> None:
        if isinstance(frame, TelnyxMarkFrame):
            if self.ack_during_send is not None:
                assert await self.ack_during_send.accept_mark(frame.mark_name)
            self.mark_sent.set()


def _paced_disclosure_runtime(controller, failure, output, *, greeting=None, begin_snapshot=None):
    input_processor = _DisclosureRelay()
    services = SimpleNamespace(
        stt=_DisclosureRelay(),
        llm=_DisclosureRelay(),
        tts=_DisclosureRelay(synthesize=True),
    )
    transport = SimpleNamespace(input=lambda: input_processor, output=lambda: output)
    pipeline = pipeline_module.build_pipeline(
        transport=transport,
        services=services,
        controller=controller,
        turn_recorder=SimpleNamespace(
            record_user=lambda *_args: None,
            record_assistant=lambda *_args: None,
        ),
        first_failure=failure,
        begin_snapshot=begin_snapshot,
    )
    runtime = pipeline_module.build_runtime(
        pipeline=pipeline,
        first_failure=failure,
        greeting=pipeline_module.SPARRA_DISCLOSURE if greeting is None else greeting,
        mark_name=controller.mark_name,
        idle_timeout_seconds=60.0,
        observers=pipeline_module._CallObservers(  # noqa: SLF001
            runtime_metrics=_TEST_RUNTIME_METRICS,
            stt=services.stt,
            llm=services.llm,
            tts=services.tts,
        ),
    )
    return runtime, services


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_during_send", [False, True])
async def test_paced_disclosure_waits_for_native_output_mark_before_deadline_and_commit(
    ack_during_send: bool,
) -> None:
    writer = _Writer(block=True)
    controller, _, recording, failure = _controller(
        writer=writer, timeout=0.02, identity=_pinned_identity(False)
    )
    output = _PacedDisclosureOutput(
        ack_during_send=controller if ack_during_send else None
    )
    runtime, services = _paced_disclosure_runtime(controller, failure, output)
    await runtime.runner.add_workers(runtime.worker)
    runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
    try:
        await asyncio.wait_for(output.mark_enqueued.wait(), timeout=2)
        # The actual native audio queue is still playing a 240 ms greeting.
        await asyncio.sleep(0.06)
        assert not output.mark_sent.is_set()
        assert failure.code is None
        assert controller.state is session_module.DisclosureState.MARK_PENDING
        assert controller.pending_task_count == 0
        await runtime.worker.queue_frame(
            InputAudioRawFrame(audio=b"\x02\x00" * 80, sample_rate=8000, num_channels=1)
        )

        await asyncio.wait_for(output.mark_pushed.wait(), timeout=2)
        assert output.mark_sent.is_set()
        assert len(output.audio_written) == 3840
        if not ack_during_send:
            # Output completion arms a deadline; it cannot itself open input.
            assert controller.pending_task_count == 1
            await runtime.worker.queue_frame(
                InputTransportMessageFrame(
                    message={"event": "mark", "mark": {"name": controller.mark_name}}
                )
            )
        await asyncio.wait_for(writer.started.wait(), timeout=2)
        assert controller.state is session_module.DisclosureState.ACK_COMMITTING
        assert not controller.is_active()
        assert recording.starts == 0
        assert services.stt.input_audio == []
        assert len(writer.commands) == 1

        writer.release.set()
        await controller.join_continuations()
        assert controller.is_active()
        assert failure.code is None
        assert len(writer.commands) == 2
        await runtime.worker.queue_frame(EndFrame())
        await asyncio.wait_for(runner_task, timeout=2)
    finally:
        writer.release.set()
        await controller.terminalize_and_join(cancel_continuations=True)
        if not runner_task.done():
            await runtime.runner.cancel(reason="test_cleanup")
            await asyncio.wait_for(runner_task, timeout=2)


@pytest.mark.asyncio
async def test_paced_disclosure_output_mark_starts_missing_ack_deadline() -> None:
    controller, writer, recording, failure = _controller(timeout=0.02)
    output = _PacedDisclosureOutput()
    runtime, _services = _paced_disclosure_runtime(controller, failure, output)
    await runtime.runner.add_workers(runtime.worker)
    runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
    try:
        await asyncio.wait_for(output.mark_pushed.wait(), timeout=2)
        assert output.mark_sent.is_set()
        assert failure.code is None
        assert controller.state is session_module.DisclosureState.MARK_PENDING
        assert await asyncio.wait_for(failure.wait(), timeout=1) == "disclosure_timeout"
        assert controller.state is session_module.DisclosureState.ABORTED
        assert not controller.is_active()
        assert not await controller.accept_mark(controller.mark_name)
        assert writer.commands == []
        assert recording.starts == 0
    finally:
        await controller.terminalize_and_join(cancel_continuations=True)
        if not runner_task.done():
            await runtime.runner.cancel(reason="test_cleanup")
            await asyncio.wait_for(runner_task, timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_company_controller_pin_overrides_legacy_flags_and_waits_for_durable_disclosure(
    enabled,
):
    writer = _Writer(block=True)
    recording = _Recording(session_module.RecordingStartResult(
        session_module.RecordingStartState.STARTED))
    controller, _, _, failure = _controller(
        writer=writer, recording=recording, identity=_pinned_identity(enabled),
        recording_enabled=not enabled, recording_required=not enabled,
    )
    await controller.note_disclosure_audio()
    await controller.arm_expected_mark()
    await controller.mark_forwarded()
    assert await controller.accept_mark(controller.mark_name)
    await writer.started.wait()
    assert recording.starts == 0 and not controller.is_active()
    operations = [command.payload["operation"] for command in writer.commands]
    assert operations[0].payload.retention_until == NOW + timedelta(days=30)
    writer.release.set()
    await controller.join_continuations()
    assert recording.starts == int(enabled)
    assert controller.is_active() and failure.code is None
    assert controller.evidence.input_gate_opened_at == NOW + timedelta(seconds=1)
    assert len(writer.commands) == 2
    operations = [command.payload["operation"] for command in writer.commands]
    assert operations[0].payload.disclosure_evidence.input_gate_opened_at is None
    assert operations[1].payload.disclosure_evidence.input_gate_opened_at == (
        NOW + timedelta(seconds=1)
    )
    assert not await controller.accept_mark(controller.mark_name)
    await controller.terminalize_and_join(cancel_continuations=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["DEFINITELY_NOT_STARTED", "INDETERMINATE"])
async def test_company_on_pin_requires_recording_even_when_legacy_flags_are_off(state):
    recording = _Recording(session_module.RecordingStartResult(
        getattr(session_module.RecordingStartState, state),
        gate_may_open=state == "INDETERMINATE"))
    controller, _, _, failure = _controller(
        recording=recording, identity=_pinned_identity(True))
    await controller.note_disclosure_audio()
    await controller.arm_expected_mark()
    await controller.mark_forwarded()
    await controller.accept_mark(controller.mark_name)
    await controller.join_continuations()
    assert recording.starts == 1
    assert not controller.is_active() and failure.code == "recording_failed"
    assert controller.evidence.input_gate_opened_at is None
    await controller.terminalize_and_join(cancel_continuations=True)


@pytest.mark.asyncio
async def test_fast_ack_before_mark_forward_commits_once_without_timeout_or_recording(
) -> None:
    controller, writer, recording, failure = _controller()
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True

    assert await controller.accept_mark(controller.mark_name) is True
    await controller.mark_forwarded()
    await controller.join_continuations()
    assert await controller.accept_mark(controller.mark_name) is False

    assert controller.state is session_module.DisclosureState.ACTIVE
    assert controller.disclosure_completed is True
    assert controller.is_active() is True
    assert len(writer.commands) == 1
    operation = writer.commands[0].payload["operation"]
    assert operation.call_id == UUID(int=1)
    assert operation.deployment_id == "deployment-1"
    assert operation.payload.telnyx_call_control_id == "call-control-1"
    assert operation.payload.telnyx_call_leg_id == "call-leg-1"
    assert operation.payload.telnyx_call_session_id == "call-session-1"
    assert operation.payload.disclosure_state == "completed"
    assert recording.starts == 0
    assert failure.code is None
    assert controller.pending_task_count == 0


@pytest.mark.asyncio
async def test_no_audio_clear_timeout_and_foreign_or_late_acks_never_commit_or_open() -> None:
    no_audio, writer, _, _ = _controller()
    assert await no_audio.arm_expected_mark() is False
    assert no_audio.state is session_module.DisclosureState.ABORTED
    assert await no_audio.accept_mark(no_audio.mark_name) is False
    assert writer.commands == []

    cleared, cleared_writer, _, _ = _controller(call_int=2)
    await cleared.note_disclosure_audio()
    assert await cleared.arm_expected_mark() is True
    await cleared.abort("disclosure_failed")
    assert await cleared.accept_mark(cleared.mark_name) is False
    assert cleared_writer.commands == []

    timed, timed_writer, _, _ = _controller(timeout=0.001, call_int=3)
    await timed.note_disclosure_audio()
    assert await timed.arm_expected_mark() is True
    await timed.mark_forwarded()
    await asyncio.sleep(0.01)
    assert timed.state is session_module.DisclosureState.ABORTED
    assert await timed.accept_mark(timed.mark_name) is False
    assert timed_writer.commands == []

    foreign, foreign_writer, _, _ = _controller(call_int=4)
    other, _, _, _ = _controller(call_int=5)
    await foreign.note_disclosure_audio()
    assert await foreign.arm_expected_mark() is True
    assert await foreign.accept_mark("marque-étrangère") is False
    assert await foreign.accept_mark(other.mark_name) is False
    assert foreign_writer.commands == []
    await foreign.abort("disclosure_failed")


@pytest.mark.asyncio
async def test_clear_during_blocked_commit_preserves_completed_history_but_never_opens() -> None:
    writer = _Writer(block=True)
    controller, _, recording, failure = _controller(writer=writer)
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await writer.started.wait()

    await controller.abort("disclosure_failed")
    writer.release.set()
    await controller.join_continuations()

    assert controller.state is session_module.DisclosureState.ABORTED
    assert controller.disclosure_completed is True
    assert controller.is_active() is False
    assert recording.starts == 0
    assert len(writer.commands) == 1
    assert failure.code == "disclosure_failed"


@pytest.mark.asyncio
async def test_local_failure_before_mark_or_ack_blocks_completion_and_open() -> None:
    controller, writer, recording, failure = _controller()
    await controller.note_disclosure_audio()
    failure.signal("writer_failed")

    assert await controller.arm_expected_mark() is False
    assert await controller.accept_mark(controller.mark_name) is False
    await controller.join_continuations()

    assert controller.is_active() is False
    assert writer.commands == []
    assert recording.starts == 0


@pytest.mark.asyncio
async def test_local_failure_during_ack_commit_retains_history_but_never_opens() -> None:
    writer = _Writer(block=True)
    controller, _, recording, failure = _controller(writer=writer)
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await writer.started.wait()

    failure.signal("writer_failed")
    writer.release.set()
    await controller.join_continuations()

    assert controller.disclosure_completed is True
    assert controller.is_active() is False
    assert recording.starts == 0
    assert len(writer.commands) == 1


@pytest.mark.asyncio
async def test_commit_failure_is_constant_safe_and_starts_nothing() -> None:
    writer = _Writer(fail=True)
    controller, _, recording, failure = _controller(writer=writer)
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await controller.join_continuations()

    assert controller.state is session_module.DisclosureState.ABORTED
    assert controller.disclosure_completed is False
    assert controller.is_active() is False
    assert recording.starts == 0
    assert failure.code == "disclosure_commit_failed"


@pytest.mark.asyncio
async def test_commit_timeout_is_constant_safe_and_never_opens() -> None:
    writer = _TimeoutWriter()
    controller, _, recording, failure = _controller(writer=writer)
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await controller.join_continuations()

    assert controller.state is session_module.DisclosureState.ABORTED
    assert controller.is_active() is False
    assert recording.starts == 0
    assert failure.code == "disclosure_commit_failed"


@pytest.mark.asyncio
async def test_ack_continuation_failure_before_commit_is_consumed_and_signalled() -> None:
    values = iter((UUID(int=900), UUID(int=901)))

    def failing_uuid_factory() -> UUID:
        try:
            return next(values)
        except StopIteration:
            raise RuntimeError("uuid-provider-secret") from None

    writer = _Writer()
    failure = pipeline_module.FirstFailure()
    recording = _Recording(
        session_module.RecordingStartResult(
            session_module.RecordingStartState.DEFINITELY_NOT_STARTED
        )
    )
    controller = session_module.DisclosureController(
        identity=_identity(),
        writer=writer,
        first_failure=failure,
        recording=recording,
        recording_enabled=False,
        recording_required=False,
        mark_timeout_seconds=0.05,
        runtime_metrics=_TEST_RUNTIME_METRICS,
        utcnow=lambda: NOW,
        uuid_factory=failing_uuid_factory,
    )
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        await controller.note_disclosure_audio()
        assert await controller.arm_expected_mark() is True
        assert await controller.accept_mark(controller.mark_name) is True
        await controller.join_continuations()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert failure.code == "disclosure_commit_failed"
    assert writer.commands == []
    assert controller.pending_task_count == 0
    assert unhandled == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("required", "result", "active", "cleanup_required"),
    [
        (
            True,
            lambda: session_module.RecordingStartResult(
                session_module.RecordingStartState.STARTED
            ),
            True,
            True,
        ),
        (
            True,
            lambda: session_module.RecordingStartResult(
                session_module.RecordingStartState.DEFINITELY_NOT_STARTED
            ),
            False,
            False,
        ),
        (
            False,
            lambda: session_module.RecordingStartResult(
                session_module.RecordingStartState.DEFINITELY_NOT_STARTED
            ),
            True,
            False,
        ),
        (
            False,
            lambda: session_module.RecordingStartResult(
                session_module.RecordingStartState.INDETERMINATE,
                gate_may_open=True,
            ),
            True,
            True,
        ),
        (
            True,
            lambda: session_module.RecordingStartResult(
                session_module.RecordingStartState.INDETERMINATE
            ),
            False,
            True,
        ),
    ],
)
async def test_recording_policy_matrix(
    required: bool,
    result: object,
    active: bool,
    cleanup_required: bool,
) -> None:
    recording = _Recording(result())
    controller, _, _, failure = _controller(
        recording=recording,
        recording_enabled=True,
        recording_required=required,
    )
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await controller.join_continuations()

    assert controller.is_active() is active
    assert controller.recording_may_be_active is cleanup_required
    assert recording.starts == 1
    assert (failure.code is None) is active


@pytest.mark.asyncio
@pytest.mark.parametrize("company_pin", [False, True])
async def test_cancel_after_recording_provider_acceptance_keeps_cleanup_ownership(
    company_pin,
) -> None:
    recording = _Recording(
        session_module.RecordingStartResult(session_module.RecordingStartState.STARTED),
        block_after_accept=True,
    )
    controller, _, _, _ = _controller(
        recording=recording,
        recording_enabled=True,
        recording_required=True,
        identity=_pinned_identity(True) if company_pin else None,
    )
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await recording.accepted.wait()
    await controller.abort("recording_failed")
    await controller.cancel_and_join_continuations()

    assert controller.state is session_module.DisclosureState.ABORTED
    assert controller.recording_may_be_active is True
    assert controller.pending_task_count == 0


class _TransportGateController:
    def __init__(self) -> None:
        self.active = False
        self.mark_name = "open"

    def is_active(self) -> bool:
        return self.active

    async def accept_mark(self, mark_name: str) -> bool:
        if mark_name == self.mark_name:
            self.active = True
            return True
        return False

    async def abort(self, _code: str) -> None:
        self.active = False

    async def note_disclosure_audio(self) -> None:
        return None

    async def arm_expected_mark(self) -> bool:
        return True

    async def mark_forwarded(self) -> None:
        return None


def _media(byte: int) -> str:
    return json.dumps(
        {
            "event": "media",
            "stream_id": "stream-one",
            "media": {
                "track": "inbound",
                "payload": base64.b64encode(bytes([byte]) * 80).decode("ascii"),
            },
        }
    )


@pytest.mark.asyncio
async def test_active_local_failure_closes_serializer_and_drops_admitted_audio_at_gate() -> None:
    controller, _writer, _recording, failure = _controller()
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    assert await controller.accept_mark(controller.mark_name) is True
    await controller.join_continuations()
    assert controller.is_active() is True

    admission = AudioAdmission()
    admission.bind(controller.is_active)
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    await serializer.setup(StartFrame(audio_in_sample_rate=8000))
    admitted = await serializer.deserialize(_media(6))
    assert isinstance(admitted, InputAudioRawFrame)

    failure.signal("writer_failed")

    assert controller.state is session_module.DisclosureState.ACTIVE
    assert admission.allows_audio() is False
    assert await serializer.deserialize(_media(7)) is None
    downstream, _ = await run_test(
        pipeline_module.build_input_gate(
            controller=controller,
            first_failure=failure,
        ),
        frames_to_send=[admitted],
        expected_down_frames=[],
    )
    assert downstream == []


def _websocket_messages(
    messages: list[str],
    sent: list[dict[str, object]] | None = None,
) -> WebSocket:
    queued = list(messages)

    async def receive() -> dict[str, object]:
        if queued:
            return {"type": "websocket.receive", "text": queued.pop(0)}
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message: dict[str, object]) -> None:
        if sent is not None:
            sent.append(message)

    websocket = WebSocket(
        {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "wss",
            "server": ("voice.invalid", 443),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
            "path": "/media",
            "raw_path": b"/media",
            "query_string": b"",
            "headers": [],
            "subprotocols": [],
        },
        receive=receive,
        send=send,
    )
    websocket.application_state = WebSocketState.CONNECTED
    websocket.client_state = WebSocketState.CONNECTED
    return websocket


@pytest.mark.asyncio
async def test_real_fastapi_output_mark_starts_deadline_after_paced_wire_send() -> None:
    controller, writer, recording, failure = _controller(timeout=0.02)
    sent: list[dict[str, object]] = []
    transport = FastAPIWebsocketTransport(
        _websocket_messages([], sent),
        FastAPIWebsocketParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            audio_out_10ms_chunks=2,
            audio_out_end_silence_secs=0,
            serializer=ProjetV0TelnyxFrameSerializer(
                "stream-one", expected_call_control_id="call-one"
            ),
        ),
    )
    runtime, _services = _paced_disclosure_runtime(controller, failure, transport.output())
    mark_enqueued = asyncio.Event()
    mark_pushed = asyncio.Event()

    async def after_process(_output: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, TelnyxMarkFrame):
            mark_enqueued.set()

    async def after_push(_output: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, TelnyxMarkFrame):
            mark_pushed.set()

    transport.output().add_event_handler("on_after_process_frame", after_process)
    transport.output().add_event_handler("on_after_push_frame", after_push)
    await runtime.runner.add_workers(runtime.worker)
    runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
    try:
        await asyncio.wait_for(mark_enqueued.wait(), timeout=2)
        await asyncio.sleep(0.06)
        assert not mark_pushed.is_set()
        assert failure.code is None
        await asyncio.wait_for(mark_pushed.wait(), timeout=2)
        payloads = [
            json.loads(message["text"])
            for message in sent
            if message.get("type") == "websocket.send" and isinstance(message.get("text"), str)
        ]
        assert payloads[0]["event"] == "media"
        assert payloads[-1] == {"event": "mark", "mark": {"name": controller.mark_name}}
        assert failure.code is None
        assert controller.pending_task_count == 1
        assert await asyncio.wait_for(failure.wait(), timeout=1) == "disclosure_timeout"
        assert not controller.is_active()
        assert writer.commands == []
        assert recording.starts == 0
    finally:
        await controller.terminalize_and_join(cancel_continuations=True)
        if not runner_task.done():
            await runtime.runner.cancel(reason="test_cleanup")
            await asyncio.wait_for(runner_task, timeout=2)


@pytest.mark.asyncio
async def test_real_fastapi_transport_drops_pre_ack_audio_then_passes_post_active_once() -> None:
    controller = _TransportGateController()
    admission = AudioAdmission()
    admission.bind(controller.is_active)
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    mark = json.dumps(
        {
            "event": "mark",
            "stream_id": "stream-one",
            "mark": {"name": "open"},
        }
    )
    transport = FastAPIWebsocketTransport(
        _websocket_messages([_media(1), mark, _media(2)]),
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            serializer=serializer,
        ),
    )
    gate = pipeline_module.build_input_gate(
        controller=controller,
        first_failure=pipeline_module.FirstFailure(),
    )

    downstream, _ = await run_test(
        Pipeline([transport.input(), gate]),
        frames_to_send=[],
        pipeline_params=pipeline_module.pipeline_params(),
    )

    audio = [frame for frame in downstream if isinstance(frame, InputAudioRawFrame)]
    assert len(audio) == 1
    assert audio[0].audio != b""


@pytest.mark.asyncio
async def test_real_fastapi_transport_rechecks_admitted_audio_after_immediate_abort() -> None:
    controller = _TransportGateController()
    controller.active = True
    admission = AudioAdmission()
    admission.bind(controller.is_active)
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    error = json.dumps(
        {
            "event": "error",
            "stream_id": "stream-one",
            "payload": {"code": 100005, "title": "media failure"},
        }
    )
    transport = FastAPIWebsocketTransport(
        _websocket_messages([_media(3), error]),
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            serializer=serializer,
        ),
    )
    gate = pipeline_module.build_input_gate(
        controller=controller,
        first_failure=pipeline_module.FirstFailure(),
    )

    downstream, _ = await run_test(
        Pipeline([transport.input(), gate]),
        frames_to_send=[],
        pipeline_params=pipeline_module.pipeline_params(),
    )

    assert controller.active is False
    assert not any(isinstance(frame, InputAudioRawFrame) for frame in downstream)


@pytest.mark.asyncio
async def test_real_transport_uses_actual_durable_controller_before_audio_admission() -> None:
    writer = _Writer(block=True)
    controller, _, _recording, failure = _controller(writer=writer)
    admission = AudioAdmission()
    admission.bind(controller.is_active)
    await controller.note_disclosure_audio()
    assert await controller.arm_expected_mark() is True
    mark = json.dumps(
        {
            "event": "mark",
            "stream_id": "stream-one",
            "mark": {"name": controller.mark_name},
        }
    )
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    transport = FastAPIWebsocketTransport(
        _websocket_messages([mark, _media(4)]),
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            serializer=serializer,
        ),
    )
    gate = pipeline_module.build_input_gate(controller=controller, first_failure=failure)

    before_commit, _ = await run_test(
        Pipeline([transport.input(), gate]),
        frames_to_send=[],
        pipeline_params=pipeline_module.pipeline_params(),
    )
    await asyncio.wait_for(writer.started.wait(), timeout=1)
    assert not any(isinstance(frame, InputAudioRawFrame) for frame in before_commit)
    assert controller.is_active() is False

    writer.release.set()
    await controller.join_continuations()
    assert controller.is_active() is True

    active_transport = FastAPIWebsocketTransport(
        _websocket_messages([_media(5)]),
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            serializer=ProjetV0TelnyxFrameSerializer(
                "stream-one",
                expected_call_control_id="call-one",
                audio_admission=admission,
            ),
        ),
    )
    after_commit, _ = await run_test(
        Pipeline(
            [
                active_transport.input(),
                pipeline_module.build_input_gate(
                    controller=controller,
                    first_failure=failure,
                ),
            ]
        ),
        frames_to_send=[],
        pipeline_params=pipeline_module.pipeline_params(),
    )
    assert sum(isinstance(frame, InputAudioRawFrame) for frame in after_commit) == 1


@pytest.mark.asyncio
async def test_pre_active_interruption_aborts_then_reaches_real_telnyx_clear() -> None:
    controller, writer, _recording, failure = _controller()
    sent: list[dict[str, object]] = []
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
    )
    transport = FastAPIWebsocketTransport(
        _websocket_messages([], sent),
        FastAPIWebsocketParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )
    gate = pipeline_module.build_input_gate(controller=controller, first_failure=failure)

    downstream, _ = await run_test(
        Pipeline([gate, transport.output()]),
        frames_to_send=[InterruptionFrame()],
        pipeline_params=pipeline_module.pipeline_params(),
    )

    clear_payloads = [
        json.loads(message["text"])
        for message in sent
        if message.get("type") == "websocket.send" and isinstance(message.get("text"), str)
    ]
    assert any(payload.get("event") == "clear" for payload in clear_payloads)
    assert controller.state is session_module.DisclosureState.ABORTED
    assert controller.is_active() is False
    assert await controller.accept_mark(controller.mark_name) is False
    assert writer.commands == []
    assert sum(isinstance(frame, InterruptionFrame) for frame in downstream) == 1


async def _local_choice_wait(predicate):
    reached = asyncio.Event()
    loop = asyncio.get_running_loop()
    handle = None

    def observe():
        nonlocal handle
        if predicate():
            reached.set()
        else:
            handle = loop.call_later(0.001, observe)

    handle = loop.call_soon(observe)
    try:
        await asyncio.wait_for(reached.wait(), 2)
    finally:
        handle.cancel()


@asynccontextmanager
async def _local_choice_case(
    tmp_path, *, policy="local_30d", available=True, start_success=True,
    contact_phone="+33102030405",
):
    """Real writer/pipeline; local start/refusal/quiesce callbacks are test doubles."""
    clock = SimpleNamespace(utc=NOW + timedelta(seconds=1), mono=10.0)
    path = tmp_path / "local-choice.sqlite"
    probe = SimpleNamespace(
        remaining=0, entered=asyncio.Event(), release=asyncio.Event(), starts=0,
        refusals=0, quiesces=0, allows_offers=False, active=asyncio.Event(), activations=0,
        start_entered=asyncio.Event(), start_release=asyncio.Event(), block_start=False,
        quiesce_entered=asyncio.Event(), quiesce_release=asyncio.Event(), block_quiesce=False,
        quiesce_active=0, quiesce_peak=0,
    )

    async def hold_commit(name):
        if name == "after_mutation_before_commit" and probe.remaining:
            probe.remaining -= 1
            if probe.remaining == 0:
                probe.entered.set()
                await probe.release.wait()

    pin = BeginCallSnapshotV2.model_validate({
        "schema_version": 2, "workspace_id": str(UUID(int=2)), "call_id": str(UUID(int=1)),
        "configuration_revision": 7, "knowledge": {"business_name": "Garage local fixture",
            "sector": "garage", "opening_hours": "", "services": "", "prices": "",
            "faq": "", "instructions": ""}, "transfer_destination": None,
        "retention_until": (NOW + timedelta(days=30)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "recording_policy": policy,
        "recording_contact_phone": contact_phone if policy == "local_30d" else None,
        "audio_available": available, "recording_id": str(UUID(int=3)) if available else None,
    })
    routing = RoutingV1(schema_version=1, direction="incoming", connection_id="fixture",
        to_e164="+33102030405", from_e164=None, telnyx_call_control_id="call-control-1",
        telnyx_call_leg_id="call-leg-1", telnyx_call_session_id="call-session-1", admitted_at=NOW)
    identity = replace(_identity(), routing=routing, begin_snapshot=pin,
                       retention_until=pin.retention_until, stream_id="stream-one")
    writer = PersistenceWriter(path, CryptoKeyring({1: bytes(range(32))}, active_version=1),
                               contract_version=2, utcnow=lambda: clock.utc, failpoint=hold_commit,
                               process_agent_id=identity.deployment_id,
                               process_deployment_id=identity.deployment_id)
    writer_task = asyncio.create_task(writer.run())
    controller = runtime = runner_task = None
    legacy = _Recording(session_module.RecordingStartResult(
        session_module.RecordingStartState.DEFINITELY_NOT_STARTED
    ))
    try:
        assert await writer.wait_ready()
        ticket = writer.submit_webhook(
            receipt={"event_id": "local-choice-admission", "event_type": "call.initiated",
                "call_control_id": identity.telnyx_call_control_id, "occurred_at": NOW,
                "received_at": NOW, "semantic_fingerprint_sha256": b"l" * 32},
            lease={"action": "upsert", "call_control_id": identity.telnyx_call_control_id,
                "call_id": identity.call_id, "tenant_id": str(pin.workspace_id),
                "agent_id": identity.deployment_id, "state": "pending", "token_hash": b"k" * 32,
                "created_at": NOW, "expires_at": NOW + timedelta(hours=1), "closed_at": None},
            operation=None, admission_facts=LocalCallAdmissionFacts(
                identity.call_id, NOW, pin.retention_until, "call-leg-1", "call-session-1",
                admission_generation=identity.generation.generation),
        )
        await ticket.wait()
        await writer.bind_audio_snapshot(pin, generation=identity.generation.generation)

        async def start_local():
            probe.starts += 1
            facts = await writer.read_call_lifecycle(identity.call_id)
            with sqlite3.connect(path) as db:
                assert db.execute(
                    "SELECT choice_state FROM local_audio_pin"
                ).fetchone()[0] == "accepted"
            assert facts.disclosure_evidence.input_gate_opened_at is not None
            probe.start_entered.set()
            if probe.block_start:
                await probe.start_release.wait()
            probe.allows_offers = start_success
            return start_success

        def refuse_local():
            probe.refusals += 1
            probe.allows_offers = False

        async def quiesce_local():
            probe.quiesces += 1
            probe.quiesce_active += 1
            probe.quiesce_peak = max(probe.quiesce_peak, probe.quiesce_active)
            probe.quiesce_entered.set()
            try:
                if probe.block_quiesce:
                    await probe.quiesce_release.wait()
                probe.allows_offers = False
            finally:
                probe.quiesce_active -= 1

        async def active():
            probe.activations += 1
            probe.active.set()

        failure = pipeline_module.FirstFailure()
        controller = session_module.DisclosureController(
            identity=identity, writer=writer, first_failure=failure, recording=legacy,
            recording_enabled=True, recording_required=True, mark_timeout_seconds=2,
            runtime_metrics=_TEST_RUNTIME_METRICS, monotonic=lambda: clock.mono,
            utcnow=lambda: clock.utc, uuid_factory=_uuids(500), on_active=active,
            local_audio_start=start_local, local_audio_refuse=refuse_local,
            local_audio_quiesce=quiesce_local,
        )
        admission = AudioAdmission()
        admission.bind(controller.is_active)
        serializer = ProjetV0TelnyxFrameSerializer("stream-one",
            expected_call_control_id=identity.telnyx_call_control_id, audio_admission=admission)
        await serializer.setup(StartFrame(audio_in_sample_rate=8000))
        output = _PacedDisclosureOutput(ack_during_send=controller)
        runtime, services = _paced_disclosure_runtime(controller, failure, output,
            greeting=controller.announcement_text, begin_snapshot=pin)
        await runtime.runner.add_workers(runtime.worker)
        runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
        yield SimpleNamespace(writer=writer, path=path, probe=probe, clock=clock,
            controller=controller, identity=identity, pin=pin, legacy=legacy, failure=failure,
            serializer=serializer, runtime=runtime, services=services, output=output)
    finally:
        probe.release.set()
        probe.start_release.set()
        probe.quiesce_release.set()
        if controller is not None:
            await controller.terminalize_and_join(cancel_continuations=True)
        if runner_task is not None and not runner_task.done():
            await runtime.runner.cancel(reason="local_choice_test_cleanup")
            await asyncio.wait_for(runner_task, 2)
        if writer.is_degraded:
            await asyncio.wait_for(asyncio.gather(writer_task, return_exceptions=True), 2)
        else:
            await writer.drain(2)
            await asyncio.wait_for(writer_task, 2)


async def _local_choice_digit(case, digit, occurred_at, *, sequence="1"):
    payload = {"event": "dtmf", "stream_id": "stream-one", "sequence_number": sequence,
               "dtmf": {"digit": digit}}
    if occurred_at is not None:
        payload["occurred_at"] = occurred_at.isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
    frame = await case.serializer.deserialize(json.dumps(payload))
    assert isinstance(frame, InputDTMFFrame)
    await case.runtime.worker.queue_frame(frame)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy, available", [("off", False), ("local_30d", True)])
async def test_sparra_notice_explicitly_announces_ia_for_off_and_local_on(
    tmp_path, policy, available,
):
    async with _local_choice_case(tmp_path, policy=policy, available=available) as case:
        assert "un assistant vocal IA." in case.controller.announcement_text
        assert "un assistant vocal IA." in pipeline_module.SPARRA_RECORDING_DISCLOSURE


@pytest.mark.asyncio
@pytest.mark.parametrize("contact_phone", [None, "+33102030405"])
async def test_local_choice_paced_ack_keeps_input_closed_until_disclosure_commit(
    tmp_path, contact_phone,
):
    async with _local_choice_case(tmp_path, contact_phone=contact_phone) as case:
        case.probe.remaining = 1
        await asyncio.wait_for(case.output.mark_enqueued.wait(), 2)
        assert not case.output.mark_sent.is_set()
        await asyncio.wait_for(case.probe.entered.wait(), 2)
        assert case.output.mark_sent.is_set() and len(case.output.audio_written) == 3840
        assert not case.controller.is_active() and case.probe.starts == 0
        assert await case.serializer.deserialize(_media(1)) is None
        await _local_choice_digit(case, "1", case.clock.utc)
        await case.runtime.worker.queue_frame(InputAudioRawFrame(b"\x02\x00" * 80, 8000, 1))
        await asyncio.sleep(0.01)
        assert case.services.stt.input_audio == [] and case.probe.starts == 0
        case.clock.mono += 4
        case.clock.utc += timedelta(seconds=4)
        case.probe.release.set()
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        assert not case.controller.is_active() and case.legacy.starts == 0
        facts = await case.writer.read_call_lifecycle(case.identity.call_id)
        assert facts.disclosure_evidence.completed_at == NOW + timedelta(seconds=1)
        assert facts.disclosure_evidence.input_gate_opened_at is None
        assert facts.retention_until == NOW + timedelta(days=30)
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT schema_version,kind FROM outbox").fetchall() == [
                (2, "call.upsert")
            ]
        spoken = case.services.tts.spoken[0]
        assert "Garage local fixture" in spoken
        assert "+33102030405" not in spoken and "None" not in spoken
        assert "30" in spoken and "2" in spoken
        assert (
            "Le texte de cet échange est conservé trente jours, même sans enregistrement audio."
            in spoken
        )
        assert "Sans choix, l'appel continue sans enregistrement." in spoken
        assert "Pendant l'appel, tapez 2 pour arrêter l'enregistrement." in spoken
        assert spoken.endswith(
            "Après cette annonce, tapez 1 pour accepter l'enregistrement audio, "
            "ou 2 pour continuer sans."
        )
        assert "supprimer" not in spoken


@pytest.mark.asyncio
@pytest.mark.parametrize("contact_phone", [None, "+33102030405"])
async def test_local_choice_rejects_early_unknown_late_one_and_latches_early_two(
    tmp_path, contact_phone,
):
    async with _local_choice_case(tmp_path / "one", contact_phone=contact_phone) as case:
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        await _local_choice_digit(case, "1", NOW)
        await _local_choice_digit(case, "1", None, sequence="2")
        await asyncio.sleep(0.01)
        assert case.probe.starts == 0 and not case.controller.is_active()
        case.clock.mono += 5.1
        case.clock.utc += timedelta(seconds=5.1)
        await _local_choice_digit(case, "1", NOW + timedelta(seconds=2), sequence="3")
        await asyncio.wait_for(case.probe.active.wait(), 2)
        assert case.probe.starts == 0 and case.legacy.starts == 0
        assert case.services.stt.input_dtmf == []
    async with _local_choice_case(tmp_path / "two", contact_phone=contact_phone) as case:
        await _local_choice_digit(case, "2", None)
        await _local_choice_wait(lambda: case.probe.refusals > 0)
        assert not case.controller.is_active() and case.probe.starts == 0
        await asyncio.wait_for(case.probe.active.wait(), 2)
        await _local_choice_digit(case, "1", case.clock.utc, sequence="2")
        await asyncio.sleep(0.01)
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
        assert case.probe.starts == 0 and case.services.stt.input_dtmf == []
        facts = await case.writer.read_call_lifecycle(case.identity.call_id)
        assert facts.original_ended_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("contact_phone", [None, "+33102030405"])
async def test_local_choice_valid_one_waits_for_choice_and_gate_before_local_start(
    tmp_path, contact_phone,
):
    async with _local_choice_case(tmp_path, contact_phone=contact_phone) as case:
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        case.probe.remaining = 1
        await _local_choice_digit(case, "1", case.clock.utc)
        await asyncio.wait_for(case.probe.entered.wait(), 2)
        assert case.controller.state.name == "CHOICE_COMMITTING"
        assert case.probe.starts == 0 and not case.controller.is_active()
        assert await case.serializer.deserialize(_media(1)) is None
        case.probe.entered.clear()
        case.probe.remaining = 1
        case.probe.release.set()
        await _local_choice_wait(lambda: case.controller.state.name == "GATE_COMMITTING")
        case.probe.release.clear()
        await asyncio.wait_for(case.probe.entered.wait(), 2)
        assert case.probe.starts == 0 and not case.controller.is_active()
        case.probe.release.set()
        await asyncio.wait_for(case.probe.active.wait(), 2)
        assert case.probe.starts == 1 and case.probe.allows_offers and case.legacy.starts == 0
        frame = await case.serializer.deserialize(_media(2))
        assert isinstance(frame, InputAudioRawFrame)
        await case.runtime.worker.queue_frame(frame)
        await _local_choice_wait(lambda: len(case.services.stt.input_audio) == 1)
        assert case.services.stt.input_dtmf == []
        facts = await case.writer.read_call_lifecycle(case.identity.call_id)
        assert facts.admission_generation == case.identity.generation.generation
        assert facts.retention_until == case.pin.retention_until and facts.original_ended_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("contact_phone", [None, "+33102030405"])
async def test_local_choice_timeout_off_unavailable_and_failed_start_keep_ordinary_phone_input(
    tmp_path, contact_phone,
):
    for label in ("silence", "unknown", "off", "unavailable", "failed-start"):
        async with _local_choice_case(tmp_path / label,
            policy="off" if label == "off" else "local_30d",
            available=label not in {"off", "unavailable"},
            start_success=label != "failed-start",
            contact_phone=contact_phone,
        ) as case:
            if label == "silence":
                case.probe.remaining = 1
                await asyncio.wait_for(case.probe.entered.wait(), 2)
                case.clock.mono += 5.1
                case.clock.utc += timedelta(seconds=5.1)
                case.probe.release.set()
            elif label in {"unknown", "failed-start"}:
                await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
                await _local_choice_digit(case, "9" if label == "unknown" else "1", case.clock.utc)
            await asyncio.wait_for(case.probe.active.wait(), 2)
            with sqlite3.connect(case.path) as db:
                assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
            assert case.probe.starts == (1 if label == "failed-start" else 0)
            assert not case.probe.allows_offers and case.legacy.starts == 0
            assert case.failure.code is None
            frame = await case.serializer.deserialize(_media(2))
            assert isinstance(frame, InputAudioRawFrame)
            await case.runtime.worker.queue_frame(frame)
            await _local_choice_wait(lambda: len(case.services.stt.input_audio) == 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("contact_phone", [None, "+33102030405"])
async def test_local_choice_two_refuses_before_denial_commit_and_never_resurrects_capture(
    tmp_path, contact_phone,
):
    for stage in ("choice", "gate", "start", "active"):
        async with _local_choice_case(tmp_path / stage, contact_phone=contact_phone) as case:
            await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
            if stage in {"choice", "gate"}:
                case.probe.remaining = 1 if stage == "choice" else 2
            if stage == "start":
                case.probe.block_start = True
            await _local_choice_digit(case, "1", case.clock.utc)
            if stage in {"choice", "gate"}:
                await asyncio.wait_for(case.probe.entered.wait(), 2)
            elif stage == "start":
                await asyncio.wait_for(case.probe.start_entered.wait(), 2)
            else:
                await asyncio.wait_for(case.probe.active.wait(), 2)
                case.probe.remaining = 1
                case.probe.entered.clear()
                case.probe.release.clear()
            frame = await case.serializer.deserialize(json.dumps({
                "event": "dtmf", "stream_id": "stream-one", "sequence_number": "2",
                "dtmf": {"digit": "2"},
            }))
            assert await case.controller.accept_dtmf(frame) is True
            assert case.probe.refusals >= 1 and not case.probe.allows_offers
            if stage == "active":
                await asyncio.wait_for(case.probe.entered.wait(), 2)
                assert case.controller.is_active()
            case.probe.release.set()
            case.probe.start_release.set()
            await asyncio.wait_for(case.probe.active.wait(), 2)
            await case.controller.join_continuations()
            assert not case.probe.allows_offers and case.probe.quiesces == 1
            with sqlite3.connect(case.path) as db:
                assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
            facts = await case.writer.read_call_lifecycle(case.identity.call_id)
            assert not facts.content_erased and facts.original_ended_at is None
            assert case.legacy.starts == 0 and case.services.stt.input_dtmf == []
            await case.controller.terminalize_and_join(cancel_continuations=True)
            assert case.controller.pending_task_count == 0


@pytest.mark.asyncio
async def test_local_choice_post_terminal_two_cannot_create_denial_sql_or_owned_work(tmp_path):
    async with _local_choice_case(tmp_path) as case:
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        await _local_choice_digit(case, "1", case.clock.utc)
        await asyncio.wait_for(case.probe.active.wait(), 2)
        assert case.probe.starts == 1 and case.probe.allows_offers
        await case.controller.terminalize_and_join(cancel_continuations=True)
        assert case.controller.pending_task_count == 0
        assert not case.controller.is_active() and not case.probe.allows_offers
        with sqlite3.connect(case.path) as db:
            before = db.execute(
                "SELECT choice_state,choice_occurred_at,denied_at FROM local_audio_pin"
            ).fetchone()
        assert before[0] == "accepted" and before[2] is None
        case.probe.entered.clear()
        case.probe.release.clear()
        case.probe.remaining = 1
        try:
            # The real runner is still alive; metadata/control bypasses audio admission.
            await _local_choice_digit(case, "2", None, sequence="2")
            with suppress(TimeoutError):
                await asyncio.wait_for(case.probe.entered.wait(), 0.2)
            assert not case.probe.entered.is_set(), "terminal controller queued denial SQL"
            assert case.controller.pending_task_count == 0
            with sqlite3.connect(case.path) as db:
                assert db.execute(
                    "SELECT choice_state,choice_occurred_at,denied_at FROM local_audio_pin"
                ).fetchone() == before
            case.probe.remaining = 0
            facts = await case.writer.read_call_lifecycle(case.identity.call_id)
            assert facts.original_ended_at is None
        finally:
            case.probe.release.set()
            await case.controller.join_continuations()


@pytest.mark.asyncio
async def test_local_choice_oracle_choice_waiting_two_opens_ordinary_input_without_another_digit(
    tmp_path,
):
    async with _local_choice_case(tmp_path) as case:
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        assert case.controller.disclosure_completed and not case.controller.is_active()
        await _local_choice_digit(case, "2", None)
        await _local_choice_wait(lambda: case.probe.quiesces == 1)
        await case.writer.wait_until_idle()
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
        await asyncio.wait_for(case.probe.active.wait(), 2)
        assert case.controller.is_active() and case.probe.starts == case.legacy.starts == 0
        assert case.probe.activations == 1
        assert not case.probe.allows_offers and case.probe.quiesces == 1
        frame = await case.serializer.deserialize(_media(2))
        assert isinstance(frame, InputAudioRawFrame)
        await case.runtime.worker.queue_frame(frame)
        await _local_choice_wait(lambda: len(case.services.stt.input_audio) == 1)
        assert case.services.stt.input_dtmf == []
        facts = await case.writer.read_call_lifecycle(case.identity.call_id)
        assert not facts.content_erased and facts.original_ended_at is None


@pytest.mark.asyncio
async def test_local_choice_oracle_choice_timeout_and_two_share_one_held_quiesce(tmp_path):
    async with _local_choice_case(tmp_path) as case:
        case.probe.block_quiesce = True
        case.probe.remaining = 1
        try:
            await asyncio.wait_for(case.probe.entered.wait(), 2)
            case.clock.mono += 5.1
            case.clock.utc += timedelta(seconds=5.1)
            case.probe.release.set()
            await asyncio.wait_for(case.probe.quiesce_entered.wait(), 2)
            assert case.probe.quiesces == case.probe.quiesce_active == case.probe.quiesce_peak == 1
            assert not case.controller.is_active()
            case.probe.entered.clear()
            case.probe.release.clear()
            case.probe.remaining = 1
            await _local_choice_digit(case, "2", None)
            await asyncio.wait_for(case.probe.entered.wait(), 2)
            case.probe.release.set()
            assert await case.writer.quick_check()  # Queued behind the actual denial COMMIT.
            await case.writer.wait_until_idle()
            await asyncio.sleep(0)
            assert case.probe.quiesces == 1
            assert case.probe.quiesce_active == case.probe.quiesce_peak == 1
            case.probe.quiesce_release.set()
            await asyncio.wait_for(case.probe.active.wait(), 2)
            await case.controller.join_continuations()
            assert case.probe.quiesces == 1 and case.probe.quiesce_active == 0
            assert not case.probe.allows_offers and case.probe.starts == case.legacy.starts == 0
            with sqlite3.connect(case.path) as db:
                assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
            await case.controller.terminalize_and_join(cancel_continuations=True)
            assert case.controller.pending_task_count == 0
            facts = await case.writer.read_call_lifecycle(case.identity.call_id)
            assert facts.original_ended_at is None
        finally:
            case.probe.release.set()
            case.probe.quiesce_release.set()
            await case.controller.join_continuations()


@pytest.mark.asyncio
async def test_transfer_prepare_does_not_pass_failed_start_event_before_real_off_commit(tmp_path):
    """Actual controller/pipeline/SQLite; only the existing local-start callback is controlled."""
    async with _local_choice_case(tmp_path, start_success=False) as case:
        case.probe.block_start = True
        case.controller._local_audio_close = lambda: setattr(case.probe, "allows_offers", False)
        await _local_choice_wait(lambda: case.controller.state.name == "WAITING_CHOICE")
        await _local_choice_digit(case, "1", case.clock.utc)
        await asyncio.wait_for(case.probe.start_entered.wait(), 2)
        case.probe.remaining = 1
        case.probe.start_release.set()
        await asyncio.wait_for(case.probe.entered.wait(), 2)
        assert case.controller._local_start_done.is_set()
        assert case.controller.pending_task_count > 0
        joined = asyncio.Event()
        original_join = case.controller._join_audio_transfer_continuations

        async def observe_join():
            joined.set()
            await original_join()

        case.controller._join_audio_transfer_continuations = observe_join
        prepared = asyncio.create_task(case.controller.prepare_audio_for_transfer())
        try:
            await asyncio.wait_for(joined.wait(), 2)
            assert not prepared.done()
            assert not case.controller.audio_ready_for_transfer()
        finally:
            case.probe.release.set()
        # This fixture intentionally has no native finish/readiness callbacks.
        assert await asyncio.wait_for(prepared, 2) is False
        await case.controller.join_continuations()
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT choice_state FROM local_audio_pin").fetchone()[0] == "off"
        assert case.probe.starts == 1 and not case.probe.allows_offers
