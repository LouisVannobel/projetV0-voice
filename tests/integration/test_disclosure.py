from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from uuid import UUID

import pytest
from pipecat.frames.frames import InputAudioRawFrame, InterruptionFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from starlette.websockets import WebSocket, WebSocketState

from projetv0_voice.admission import CallGenerationHandle, ProcessLeaseClaim
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.models import BeginCallSnapshotV1, RoutingV1
from projetv0_voice.persistence.commands import PersistenceCommand
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
            "media": {"payload": base64.b64encode(bytes([byte]) * 80).decode("ascii")},
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
