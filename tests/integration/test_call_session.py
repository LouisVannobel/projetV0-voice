from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from pipecat.bus.bus import WorkerBus
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.registry.registry import WorkerRegistry
from pipecat.runner.types import TelnyxCallData
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner
from pydantic import SecretStr
from starlette.websockets import WebSocket, WebSocketState

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.inference.openrouter_tts import OpenRouterTTSService
from projetv0_voice.inference.services import build_llm
from projetv0_voice.models import VoiceOperationV1
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.telnyx.frames import TelnyxMarkFrame
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake
from projetv0_voice.telnyx.serializer import AudioAdmission, ProjetV0TelnyxFrameSerializer

pipeline_module = import_module("projetv0_voice.pipeline")
session_module = import_module("projetv0_voice.session")

NOW = datetime(2026, 8, 28, 18, 0, tzinfo=UTC)


class _SttClient:
    def __init__(self, *, failures: int = 0, block: bool = False) -> None:
        self.failures = failures
        self.block = block
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.never = asyncio.Event()

    async def aclose(self) -> None:
        self.calls += 1
        self.started.set()
        if self.block:
            try:
                await self.never.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.calls <= self.failures:
            raise RuntimeError("stt-client-secret")


class _LlmClient:
    def __init__(self, *, failures: int = 0, block: bool = False) -> None:
        self.failures = failures
        self.block = block
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.never = asyncio.Event()

    async def close(self) -> None:
        self.calls += 1
        self.started.set()
        if self.block:
            try:
                await self.never.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.calls <= self.failures:
            raise RuntimeError("llm-client-secret")


def _identity(lease_claim: object | None = None, *, call_int: int = 1) -> object:
    return session_module.CallIdentity(
        call_id=UUID(int=call_int),
        durable_generation=f"generation-{call_int}",
        lease_identity=f"lease-{call_int}",
        lease_claim=lease_claim or object(),
        deployment_id="deployment-1",
        registry_handle=f"registry-{call_int}",
        telnyx_call_control_id=f"call-control-{call_int}",
        telnyx_call_leg_id=f"call-leg-{call_int}",
        telnyx_call_session_id=f"call-session-{call_int}",
        stream_id=f"stream-{call_int}",
        started_at=NOW,
        retention_until=NOW + timedelta(days=7),
    )


@pytest.mark.asyncio
async def test_service_bundle_closes_public_stt_and_pinned_llm_once_even_concurrently() -> None:
    stt_client = _SttClient()
    llm_client = _LlmClient()
    tts = SimpleNamespace(cleanup_calls=0)
    bundle = session_module.ServiceBundle(
        stt=SimpleNamespace(),
        llm=SimpleNamespace(_client=llm_client),
        tts=tts,
        stt_http_client=stt_client,
        close_timeout_seconds=0.1,
    )
    assert repr(bundle) == "ServiceBundle()"

    await asyncio.gather(bundle.aclose(), bundle.aclose())
    await bundle.aclose()

    assert stt_client.calls == 1
    assert llm_client.calls == 1
    assert tts.cleanup_calls == 0


@pytest.mark.asyncio
async def test_service_bundle_attempts_both_and_retries_only_failed_resource() -> None:
    stt_client = _SttClient(failures=1)
    llm_client = _LlmClient()
    bundle = session_module.ServiceBundle(
        stt=SimpleNamespace(),
        llm=SimpleNamespace(_client=llm_client),
        tts=SimpleNamespace(),
        stt_http_client=stt_client,
        close_timeout_seconds=0.1,
    )

    with pytest.raises(session_module.ServiceLifecycleError) as raised:
        await bundle.aclose()
    assert str(raised.value) == "service_close_failed"
    assert "stt-client-secret" not in repr(raised.value)
    assert stt_client.calls == 1
    assert llm_client.calls == 1

    await bundle.aclose()
    assert stt_client.calls == 2
    assert llm_client.calls == 1


@pytest.mark.asyncio
async def test_service_bundle_cancellation_joins_close_tasks_and_is_retryable() -> None:
    stt_client = _SttClient(block=True)
    llm_client = _LlmClient(block=True)
    bundle = session_module.ServiceBundle(
        stt=SimpleNamespace(),
        llm=SimpleNamespace(_client=llm_client),
        tts=SimpleNamespace(),
        stt_http_client=stt_client,
        close_timeout_seconds=1.0,
    )
    closing = asyncio.create_task(bundle.aclose())
    await asyncio.gather(stt_client.started.wait(), llm_client.started.wait())
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert stt_client.cancelled.is_set()
    assert llm_client.cancelled.is_set()

    stt_client.block = False
    llm_client.block = False
    await bundle.aclose()
    assert stt_client.calls == 2
    assert llm_client.calls == 2


@pytest.mark.asyncio
async def test_pinned_llm_close_shim_is_the_only_accepted_private_client_contract() -> None:
    assert version("pipecat-ai") == "1.7.0"
    llm = build_llm(_profile().inference, SecretStr("offline-secret"))
    stt_client = _SttClient()
    bundle = session_module.ServiceBundle(
        stt=FrameProcessor(),
        llm=llm,
        tts=FrameProcessor(),
        stt_http_client=stt_client,
        close_timeout_seconds=0.1,
    )

    assert callable(llm._client.close)
    await bundle.aclose()
    assert stt_client.calls == 1


class _TurnWriter:
    def __init__(self, *, accepting: bool = True) -> None:
        self.accepting = accepting
        self.operations: list[object] = []

    def try_enqueue_turn(self, operation: object) -> bool:
        self.operations.append(operation)
        return self.accepting


def _uuid_factory():
    value = 100

    def factory() -> UUID:
        nonlocal value
        result = UUID(int=value)
        value += 1
        return result

    return factory


def test_turn_recorder_allocates_shared_fifo_encrypts_exact_aad_and_never_keeps_plaintext() -> None:
    writer = _TurnWriter()
    failure = pipeline_module.FirstFailure()
    key = b"k" * 32
    keyring = CryptoKeyring(
        {1: key},
        active_version=1,
        nonce_factory=lambda size: b"n" * size,
    )
    recorder = session_module.TurnRecorder(
        identity=_identity(),
        writer=writer,
        keyring=keyring,
        first_failure=failure,
        uuid_factory=_uuid_factory(),
        utcnow=lambda: NOW + timedelta(seconds=2),
    )

    recorder.record_user("bonjour-secret", NOW.isoformat())
    recorder.record_assistant("reponse-secret", NOW.isoformat(), False)
    recorder.record_assistant("interrompu-secret", NOW.isoformat(), True)
    recorder.record_user(None, NOW.isoformat())
    recorder.record_user("", NOW.isoformat())

    assert len(writer.operations) == 3
    for turn_no, (operation, plaintext) in enumerate(
        zip(
            writer.operations,
            [b"bonjour-secret", b"reponse-secret", b"interrompu-secret"],
            strict=True,
        ),
        start=1,
    ):
        payload = operation.payload
        assert payload.turn_no == turn_no
        assert payload.interrupted is (turn_no == 3)
        encrypted = EncryptedValue(
            key_version=payload.key_version,
            nonce=base64.b64decode(payload.nonce_b64),
            ciphertext=base64.b64decode(payload.ciphertext_b64),
        )
        aad = b"turn:" + str(payload.turn_id).encode("ascii")
        assert keyring.decrypt(encrypted, aad=aad) == plaintext
        rendered = repr(operation)
        assert plaintext.decode() not in rendered
    assert failure.code is None

    recorder.close()
    recorder.record_user("late-secret", NOW.isoformat())
    assert len(writer.operations) == 3


def test_turn_recorder_bounds_utf8_and_false_enqueue_signals_shared_fatal() -> None:
    writer = _TurnWriter(accepting=False)
    failure = pipeline_module.FirstFailure()
    keyring = CryptoKeyring(
        {1: b"k" * 32},
        active_version=1,
        nonce_factory=lambda size: b"n" * size,
    )
    recorder = session_module.TurnRecorder(
        identity=_identity(),
        writer=writer,
        keyring=keyring,
        first_failure=failure,
        uuid_factory=_uuid_factory(),
        utcnow=lambda: NOW,
    )
    recorder.record_user("e" * (session_module.MAX_TURN_TEXT_BYTES + 1), NOW.isoformat())

    assert failure.code == "writer_failed"
    operation = writer.operations[0]
    payload = operation.payload
    encrypted = EncryptedValue(
        key_version=payload.key_version,
        nonce=base64.b64decode(payload.nonce_b64),
        ciphertext=base64.b64decode(payload.ciphertext_b64),
    )
    plaintext = keyring.decrypt(
        encrypted,
        aad=b"turn:" + str(payload.turn_id).encode("ascii"),
    )
    assert len(plaintext) == session_module.MAX_TURN_TEXT_BYTES


class _PassProcessor(FrameProcessor):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        nested_failure: bool = False,
    ) -> None:
        super().__init__(name=name, enable_direct_mode=True)
        self.events = events
        self.started = asyncio.Event()
        self.nested_failure = nested_failure

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.started.set()
            if self.nested_failure:
                self.create_task(self._fail_nested(), "nested-session-failure")
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        await super().cleanup()
        self.events.append(f"pipeline:{self.name}")

    async def _fail_nested(self) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("nested-provider-secret")


class _SegmentedSessionStt(SegmentedSTTService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.transcribed = asyncio.Event()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self.started.set()

    async def run_stt(self, _audio: bytes) -> AsyncGenerator[Frame | None]:
        yield TranscriptionFrame(
            text="tour-retarde",
            user_id="",
            timestamp=NOW.isoformat(),
            language=Language.FR,
        )
        self.transcribed.set()


class _OfflineTts(_PassProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await FrameProcessor.process_frame(self, frame, direction)
        if isinstance(frame, StartFrame):
            self.started.set()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=b"\x01\x00" * 80,
                    sample_rate=8000,
                    num_channels=1,
                    context_id="disclosure",
                ),
                direction,
            )
            await self.push_frame(TTSStoppedFrame(context_id="disclosure"), direction)
        else:
            await self.push_frame(frame, direction)


class _PartialFailingTts(_OfflineTts):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, TTSSpeakFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=b"\x01\x00" * 80,
                    sample_rate=8000,
                    num_channels=1,
                    context_id="disclosure",
                ),
                direction,
            )
            await self.push_error_frame(
                ErrorFrame(
                    error="tts-provider-secret",
                    fatal=True,
                    exception=RuntimeError("tts-provider-secret"),
                )
            )
            return
        await super().process_frame(frame, direction)


class _FailingPcmStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(0.1)
        raise RuntimeError("tts-provider-secret")

    async def aclose(self) -> None:
        self.closed = True


class _Transport:
    def __init__(self, events: list[str], *, echo_ack: bool = False) -> None:
        self._input = _PassProcessor("transport-input", events)
        self._output = (
            _AckingOutput("transport-output", events)
            if echo_ack
            else _PassProcessor("transport-output", events)
        )

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    def add_event_handler(self, _event_name: str, _handler: object) -> None:
        return None


class _AckingOutput(_PassProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await FrameProcessor.process_frame(self, frame, direction)
        if isinstance(frame, TelnyxMarkFrame):
            await self.push_frame(
                InputTransportMessageFrame(
                    message={"event": "mark", "mark": {"name": frame.mark_name}}
                ),
                FrameDirection.UPSTREAM,
            )
        await self.push_frame(frame, direction)


async def _apply_phase_fault(kind: str | None, secret: str) -> None:
    if kind == "ordinary":
        raise RuntimeError(secret)
    if kind == "cancel":
        raise asyncio.CancelledError(secret)
    if kind == "timeout":
        await asyncio.Event().wait()


class _SessionWriter:
    def __init__(
        self,
        events: list[str],
        *,
        cleanup_fault: str | None = None,
        terminal_started: asyncio.Event | None = None,
        terminal_release: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.fatal_event = asyncio.Event()
        self.disclosure_committed = asyncio.Event()
        self.commands: list[object] = []
        self.cleanup_fault = cleanup_fault
        self.terminal_started = terminal_started
        self.terminal_release = terminal_release

    def try_enqueue_turn(self, operation: object) -> bool:
        self.commands.append(operation)
        return True

    async def commit_control(self, command: object) -> None:
        self.commands.append(command)
        operation = command.payload["operation"]
        if operation.kind == "call.upsert" and operation.payload.disclosure_state == "completed":
            self.disclosure_committed.set()
        if operation.kind == "call.upsert" and operation.payload.status in {"closed", "failed"}:
            self.events.append("call-terminal-attempt")
            if self.terminal_started is not None:
                self.terminal_started.set()
            if self.terminal_release is not None:
                await self.terminal_release.wait()
            await _apply_phase_fault(self.cleanup_fault, "writer-cleanup-secret")
            self.events.append("call-terminal")


class _RecordingBoundary:
    def __init__(self, events: list[str], *, cleanup_fault: str | None = None) -> None:
        self.events = events
        self.cleanup_fault = cleanup_fault

    async def start(self, _identity: object) -> object:
        return session_module.RecordingStartResult(
            session_module.RecordingStartState.DEFINITELY_NOT_STARTED
        )

    async def cleanup(
        self,
        _identity: object,
        *,
        recording_may_be_active: bool,
        reason: str,
    ) -> None:
        self.events.append("recording-cleanup")
        self.events.append(
            f"termination:{recording_may_be_active}:{reason}"
        )
        await _apply_phase_fault(self.cleanup_fault, "recording-cleanup-secret")


class _LeaseTerminalizer:
    def __init__(
        self,
        events: list[str],
        *,
        cleanup_fault: str | None = None,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.calls: list[tuple[str, str]] = []
        self.cleanup_fault = cleanup_fault
        self.started = started
        self.release = release

    async def terminalize(self, _identity: object, *, status: str, reason: str) -> None:
        self.events.append("lease-attempt")
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        await _apply_phase_fault(self.cleanup_fault, "lease-cleanup-secret")
        self.calls.append((status, reason))
        self.events.append("lease-terminal")


class _SessionServices:
    def __init__(
        self,
        *,
        stt: FrameProcessor,
        llm: FrameProcessor,
        tts: FrameProcessor,
        events: list[str],
        close_started: asyncio.Event | None = None,
        close_release: asyncio.Event | None = None,
        close_cancelled: asyncio.Event | None = None,
        cleanup_fault: str | None = None,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self._events = events
        self._close_started = close_started
        self._close_release = close_release
        self._close_cancelled = close_cancelled
        self._cleanup_fault = cleanup_fault

    async def aclose(self) -> None:
        self._events.append("services-attempt")
        if self._close_started is not None:
            self._close_started.set()
        if self._close_release is not None:
            try:
                await self._close_release.wait()
            except asyncio.CancelledError:
                if self._close_cancelled is not None:
                    self._close_cancelled.set()
                raise
        await _apply_phase_fault(self._cleanup_fault, "services-cleanup-secret")
        self._events.extend(["services:stt", "services:llm"])


def _manifest() -> AgentManifestV1:
    return AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "agent_id": "agent-1",
            "revision": "revision-1",
            "tenant_id": "tenant-1",
            "dids": ["+33123456789"],
            "language": "fr-FR",
            "prompt_path": "prompt.md",
            "prompt_revision": "prompt-revision-1",
            "greeting": "Bonjour, appel automatise.",
            "conversation_mode": "freeform",
            "max_concurrent_calls": 1,
            "direction": "inbound_only",
            "transport_codec": "PCMU",
            "transport_sample_rate_hz": 8000,
            "transcript_retention_days": 7,
            "recording_mode": "off",
            "recording_format": "wav",
            "recording_retention_days": None,
            "recording_required": False,
            "recording_play_beep": False,
        }
    )


def _profile() -> QualifiedDeploymentProfileV1:
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(encoding="utf-8")
    )
    return profile.model_copy(update={"deployment_id": "deployment-1"})


def _handshake(
    transport: object,
    lease_claim: object,
    *,
    call_int: int = 1,
) -> AuthenticatedTelnyxHandshake:
    admission = AudioAdmission()
    return AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id=f"stream-{call_int}",
            call_id=f"call-control-{call_int}",
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=lease_claim,
        transport=transport,  # type: ignore[arg-type]
        audio_admission=admission,
    )


def _session(
    *,
    events: list[str],
    nested_failure: bool = False,
    echo_ack: bool = False,
    service_close_started: asyncio.Event | None = None,
    service_close_release: asyncio.Event | None = None,
    service_close_cancelled: asyncio.Event | None = None,
    task_factory: object | None = None,
    partial_tts_failure: bool = False,
    segmented_stt: bool = False,
    call_int: int = 1,
    writer_override: _SessionWriter | None = None,
    cleanup_fault_phase: str | None = None,
    cleanup_fault_kind: str | None = None,
    terminal_started: asyncio.Event | None = None,
    terminal_release: asyncio.Event | None = None,
    lease_started: asyncio.Event | None = None,
    lease_release: asyncio.Event | None = None,
) -> tuple[object, FrameProcessor, _LeaseTerminalizer, _SessionWriter, _Transport]:
    transport = _Transport(events, echo_ack=echo_ack)
    stt: FrameProcessor = (
        _SegmentedSessionStt()
        if segmented_stt
        else _PassProcessor("stt", events, nested_failure=nested_failure)
    )
    llm = _PassProcessor("llm", events)
    tts = (
        _PartialFailingTts("tts", events)
        if partial_tts_failure
        else _OfflineTts("tts", events)
    )
    bundle = _SessionServices(
        stt=stt,
        llm=llm,
        tts=tts,
        events=events,
        close_started=service_close_started,
        close_release=service_close_release,
        close_cancelled=service_close_cancelled,
        cleanup_fault=(
            cleanup_fault_kind if cleanup_fault_phase == "services" else None
        ),
    )
    writer = writer_override or _SessionWriter(
        events,
        cleanup_fault=(
            cleanup_fault_kind if cleanup_fault_phase == "writer" else None
        ),
        terminal_started=terminal_started,
        terminal_release=terminal_release,
    )
    lease = _LeaseTerminalizer(
        events,
        cleanup_fault=(
            cleanup_fault_kind if cleanup_fault_phase == "lease" else None
        ),
        started=lease_started,
        release=lease_release,
    )
    lease_claim = object()
    identity = _identity(lease_claim, call_int=call_int)
    session = session_module.CallSession(
        identity=identity,
        manifest=_manifest(),
        profile=_profile(),
        services=bundle,
        writer=writer,
        keyring=CryptoKeyring(
            {1: b"k" * 32},
            active_version=1,
            nonce_factory=lambda size: b"n" * size,
        ),
        recording=_RecordingBoundary(
            events,
            cleanup_fault=(
                cleanup_fault_kind if cleanup_fault_phase == "recording" else None
            ),
        ),
        lease_terminalizer=lease,
        idle_timeout_seconds=60.0,
        cleanup_phase_timeout_seconds=0.05,
        task_factory=task_factory,
        utcnow=lambda: NOW + timedelta(seconds=5),
        uuid_factory=_uuid_factory(),
    )
    session.handshake = _handshake(transport, lease_claim, call_int=call_int)
    return session, stt, lease, writer, transport


def _real_websocket_session(
    *,
    timeout: bool,
    active_before_disconnect: bool = False,
) -> tuple[
    object,
    _LeaseTerminalizer,
    _SessionWriter,
    AudioAdmission,
    asyncio.Event,
]:
    events: list[str] = []
    never = asyncio.Event()
    disconnect = asyncio.Event()
    ack_ready = asyncio.Event()
    ack_text: str | None = None
    ack_delivered = False

    async def receive() -> dict[str, object]:
        nonlocal ack_delivered
        if timeout:
            await never.wait()
        if active_before_disconnect:
            if not ack_delivered:
                await ack_ready.wait()
                ack_delivered = True
                return {"type": "websocket.receive", "text": ack_text}
            await disconnect.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message: dict[str, object]) -> None:
        nonlocal ack_text
        text = message.get("text")
        if active_before_disconnect and isinstance(text, str):
            payload = json.loads(text)
            if payload.get("event") == "mark":
                ack_text = json.dumps(
                    {
                        "event": "mark",
                        "stream_id": "stream-1",
                        "mark": {"name": payload["mark"]["name"]},
                    }
                )
                ack_ready.set()

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
    admission = AudioAdmission()
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-1",
        expected_call_control_id="call-control-1",
        audio_admission=admission,
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
            session_timeout=1 if timeout else None,
        ),
    )
    stt = _PassProcessor("stt", events)
    services = _SessionServices(
        stt=stt,
        llm=_PassProcessor("llm", events),
        tts=_OfflineTts("tts", events),
        events=events,
    )
    writer = _SessionWriter(events)
    lease = _LeaseTerminalizer(events)
    lease_claim = object()
    session = session_module.CallSession(
        identity=_identity(lease_claim),
        manifest=_manifest(),
        profile=_profile(),
        services=services,
        writer=writer,
        keyring=CryptoKeyring(
            {1: b"k" * 32},
            active_version=1,
            nonce_factory=lambda size: b"n" * size,
        ),
        recording=_RecordingBoundary(events),
        lease_terminalizer=lease,
        idle_timeout_seconds=60.0,
        utcnow=lambda: NOW + timedelta(seconds=5),
        uuid_factory=_uuid_factory(),
    )
    session.handshake = AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id="stream-1",
            call_id="call-control-1",
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=lease_claim,
        transport=transport,
        audio_admission=admission,
    )
    return session, lease, writer, admission, disconnect


def _actual_partial_tts_telnyx_session() -> tuple[
    object,
    _LeaseTerminalizer,
    _SessionWriter,
    AudioAdmission,
    list[dict[str, object]],
    httpx.AsyncClient,
    _FailingPcmStream,
]:
    events: list[str] = []
    never = asyncio.Event()
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        await never.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message: dict[str, object]) -> None:
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
    admission = AudioAdmission()
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-1",
        expected_call_control_id="call-control-1",
        audio_admission=admission,
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )
    stream = _FailingPcmStream(
        [
            b"\x01",
            b"\x00" + b"\x02\x00" * 12000 + b"\x03",
            b"\x00" + b"\x04\x00" * 12000,
        ]
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/pcm"},
            stream=stream,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = OpenRouterTTSService(
        profile=_profile().inference,
        api_key=SecretStr("offline-secret"),
        http_client=client,
    )
    services = _SessionServices(
        stt=_PassProcessor("stt", events),
        llm=_PassProcessor("llm", events),
        tts=tts,
        events=events,
    )
    writer = _SessionWriter(events)
    lease = _LeaseTerminalizer(events)
    lease_claim = object()
    session = session_module.CallSession(
        identity=_identity(lease_claim),
        manifest=_manifest(),
        profile=_profile(),
        services=services,
        writer=writer,
        keyring=CryptoKeyring(
            {1: b"k" * 32},
            active_version=1,
            nonce_factory=lambda size: b"n" * size,
        ),
        recording=_RecordingBoundary(events),
        lease_terminalizer=lease,
        idle_timeout_seconds=60.0,
        cleanup_phase_timeout_seconds=0.25,
        utcnow=lambda: NOW + timedelta(seconds=5),
        uuid_factory=_uuid_factory(),
    )
    session.handshake = AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id="stream-1",
            call_id="call-control-1",
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=lease_claim,
        transport=transport,
        audio_admission=admission,
    )
    return session, lease, writer, admission, sent, client, stream


@pytest.mark.asyncio
async def test_public_call_session_cancellation_waits_for_ordered_bounded_cleanup() -> None:
    events: list[str] = []
    session, stt, lease, _writer, _transport = _session(events=events)
    running = asyncio.create_task(session.run(session.handshake))
    await stt.started.wait()
    running.cancel("caller-cancel")
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await running
    assert cancelled.value.args == ("caller-cancel",)

    pipeline_last = max(index for index, item in enumerate(events) if item.startswith("pipeline:"))
    services_first = min(index for index, item in enumerate(events) if item.startswith("services:"))
    assert pipeline_last < services_first
    assert services_first < events.index("call-terminal") < events.index("lease-terminal")
    assert "recording-cleanup" in events
    assert lease.calls == [("failed", "external_cancel")]


@pytest.mark.asyncio
async def test_cancellation_during_cleanup_is_shielded_and_first_cancel_wins() -> None:
    events: list[str] = []
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_cancelled = asyncio.Event()
    session, _stt, lease, writer, _transport = _session(
        events=events,
        echo_ack=True,
        service_close_started=close_started,
        service_close_release=close_release,
        service_close_cancelled=close_cancelled,
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=5)
    await session.request_drain()
    await asyncio.wait_for(close_started.wait(), timeout=5)

    running.cancel("first-cancel")
    await asyncio.sleep(0)
    running.cancel("second-cancel")
    close_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(running, timeout=5)

    assert cancelled.value.args == ("first-cancel",)
    assert close_cancelled.is_set() is False
    assert "call-terminal" in events
    assert lease.calls == [("failed", "external_cancel")]
    terminal_operation = next(
        command.payload["operation"]
        for command in writer.commands
        if not isinstance(command, VoiceOperationV1)
        and hasattr(command, "payload")
        and command.payload["operation"].kind == "call.upsert"
        and command.payload["operation"].payload.status == "failed"
    )
    assert terminal_operation.payload.end_reason == "external_cancel"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["terminal", "lease"])
async def test_terminal_outcome_is_frozen_before_irreversible_boundaries(
    blocked_phase: str,
) -> None:
    events: list[str] = []
    terminal_started = asyncio.Event()
    terminal_release = asyncio.Event()
    lease_started = asyncio.Event()
    lease_release = asyncio.Event()
    session, _stt, lease, writer, _transport = _session(
        events=events,
        echo_ack=True,
        terminal_started=terminal_started if blocked_phase == "terminal" else None,
        terminal_release=terminal_release if blocked_phase == "terminal" else None,
        lease_started=lease_started if blocked_phase == "lease" else None,
        lease_release=lease_release if blocked_phase == "lease" else None,
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=3)
    await session.request_drain()
    if blocked_phase == "terminal":
        await asyncio.wait_for(terminal_started.wait(), timeout=3)
    else:
        await asyncio.wait_for(lease_started.wait(), timeout=3)

    running.cancel("cancel-after-freeze")
    await asyncio.sleep(0)
    terminal_release.set()
    lease_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(running, timeout=3)

    terminal_operations = [
        command.payload["operation"]
        for command in writer.commands
        if not isinstance(command, VoiceOperationV1)
        and hasattr(command, "payload")
        and command.payload["operation"].kind == "call.upsert"
        and command.payload["operation"].payload.status in {"closed", "failed"}
    ]
    assert cancelled.value.args == ("cancel-after-freeze",)
    assert len(terminal_operations) == 1
    assert terminal_operations[0].payload.status == "closed"
    assert terminal_operations[0].payload.end_reason == "closed"
    assert lease.calls == [("closed", "closed")]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["recording", "services", "writer", "lease"])
@pytest.mark.parametrize("fault", ["ordinary", "cancel", "timeout"])
async def test_cleanup_attempts_all_phases_after_failure_timeout_or_cancellation(
    phase: str,
    fault: str,
) -> None:
    events: list[str] = []
    session, stt, _lease, _writer, _transport = _session(
        events=events,
        cleanup_fault_phase=phase,
        cleanup_fault_kind=fault,
    )
    assert isinstance(stt, _PassProcessor)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(stt.started.wait(), timeout=3)
    running.cancel("matrix-cancel")

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(running, timeout=3)

    assert cancelled.value.args == ("matrix-cancel",)
    assert "recording-cleanup" in events
    assert "services-attempt" in events
    assert "call-terminal-attempt" in events
    assert "lease-attempt" in events
    assert events.index("recording-cleanup") < events.index("services-attempt")
    assert events.index("services-attempt") < events.index("call-terminal-attempt")
    assert events.index("call-terminal-attempt") < events.index("lease-attempt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "code"),
    [
        ("recording", "recording_cleanup_failed"),
        ("services", "service_close_failed"),
        ("writer", "persistence_failed"),
        ("lease", "lease_terminalization_failed"),
    ],
)
async def test_first_cleanup_failure_changes_normal_close_to_failed_terminal_state(
    phase: str,
    code: str,
) -> None:
    events: list[str] = []
    session, _stt, lease, writer, _transport = _session(
        events=events,
        echo_ack=True,
        cleanup_fault_phase=phase,
        cleanup_fault_kind="ordinary",
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=3)
    await session.request_drain()

    with pytest.raises(session_module.CallSessionError, match=code):
        await asyncio.wait_for(running, timeout=3)

    if phase == "lease":
        assert lease.calls == []
    elif phase == "writer":
        assert lease.calls == [("closed", "closed")]
    else:
        assert lease.calls == [("failed", code)]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["controller", "runner"])
@pytest.mark.parametrize("fault", ["ordinary", "cancel", "timeout"])
async def test_controller_and_runner_cleanup_faults_do_not_skip_later_phases(
    phase: str,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    if phase == "controller":
        async def fail_controller(_controller: object, **_kwargs: object) -> None:
            events.append("controller-cleanup-attempt")
            await _apply_phase_fault(fault, "controller-cleanup-secret")

        monkeypatch.setattr(
            session_module.DisclosureController,
            "terminalize_and_join",
            fail_controller,
        )
    else:
        async def fail_runner_cancel(_runner: object, reason: str | None = None) -> None:
            del reason
            events.append("runner-cancel-attempt")
            await _apply_phase_fault(fault, "runner-cancel-secret")

        monkeypatch.setattr(pipeline_module.WorkerRunner, "cancel", fail_runner_cancel)

    session, stt, _lease, _writer, _transport = _session(events=events)
    assert isinstance(stt, _PassProcessor)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(stt.started.wait(), timeout=3)
    running.cancel("cleanup-owner-cancel")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=3)

    assert "recording-cleanup" in events
    assert "services-attempt" in events
    assert "call-terminal-attempt" in events
    assert "lease-attempt" in events
    if phase == "runner":
        assert any(item.startswith("pipeline:") for item in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["ordinary", "cancel"])
async def test_turn_recorder_close_fault_is_safe_and_failure_waiter_is_joined(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    waiter_joined = asyncio.Event()

    def fail_close(_recorder: object) -> None:
        events.append("turn-recorder-close-attempt")
        if fault == "cancel":
            raise asyncio.CancelledError("turn-recorder-close-secret")
        raise RuntimeError("turn-recorder-close-secret")

    def task_factory(coroutine: object, name: str) -> asyncio.Task[object]:
        if name != "call-first-failure":
            return asyncio.create_task(coroutine, name=name)  # type: ignore[arg-type]

        async def observe_waiter() -> object:
            try:
                return await coroutine  # type: ignore[misc]
            finally:
                waiter_joined.set()

        return asyncio.create_task(observe_waiter(), name=name)

    monkeypatch.setattr(session_module.TurnRecorder, "close", fail_close)
    session, _stt, lease, writer, _transport = _session(
        events=events,
        echo_ack=True,
        task_factory=task_factory,
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=3)
    await session.request_drain()

    with pytest.raises(session_module.CallSessionError, match="persistence_failed"):
        await asyncio.wait_for(running, timeout=3)

    assert waiter_joined.is_set()
    assert "services-attempt" in events
    assert "call-terminal-attempt" in events
    assert "lease-attempt" in events
    assert lease.calls == [("failed", "persistence_failed")]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["ordinary", "cancel"])
async def test_internal_cleanup_owner_failure_is_not_reported_as_caller_cancellation(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def fail_cleanup(_session: object, **_kwargs: object) -> None:
        if fault == "cancel":
            raise asyncio.CancelledError("internal-cleanup-secret")
        raise RuntimeError("internal-cleanup-secret")

    monkeypatch.setattr(session_module.CallSession, "_cleanup_owned_state", fail_cleanup)
    session, _stt, _lease, writer, _transport = _session(
        events=events,
        echo_ack=True,
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=3)
    await session.request_drain()

    with pytest.raises(session_module.CallSessionError, match="call_failed"):
        await asyncio.wait_for(running, timeout=3)


@pytest.mark.asyncio
async def test_same_turn_caller_and_internal_cleanup_cancellation_preserves_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def cancel_cleanup(_session: object, **_kwargs: object) -> None:
        cleanup_started.set()
        await cleanup_release.wait()
        raise asyncio.CancelledError("internal-cleanup-secret")

    monkeypatch.setattr(session_module.CallSession, "_cleanup_owned_state", cancel_cleanup)
    session, _stt, _lease, writer, _transport = _session(
        events=events,
        echo_ack=True,
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=3)
    await session.request_drain()
    await asyncio.wait_for(cleanup_started.wait(), timeout=3)

    cleanup_release.set()
    asyncio.get_running_loop().call_soon(running.cancel, "caller-same-turn")
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(running, timeout=3)

    assert cancelled.value.args == ("caller-same-turn",)


@pytest.mark.asyncio
async def test_nested_native_task_failure_cancels_and_joins_real_runner_before_return() -> None:
    events: list[str] = []
    session, _stt, lease, _writer, _transport = _session(
        events=events, nested_failure=True
    )

    with pytest.raises(session_module.CallSessionError, match="pipeline_task_failed"):
        await asyncio.wait_for(session.run(session.handshake), timeout=1)

    assert lease.calls == [("failed", "pipeline_task_failed")]
    assert events[-1] == "lease-terminal"


@pytest.mark.asyncio
async def test_partial_tts_failure_invokes_injected_termination_cleanup() -> None:
    events: list[str] = []
    session, _stt, lease, _writer, _transport = _session(
        events=events,
        partial_tts_failure=True,
    )

    with pytest.raises(session_module.CallSessionError, match="tts_failed"):
        await asyncio.wait_for(session.run(session.handshake), timeout=3)

    assert "termination:False:tts_failed" in events
    assert lease.calls == [("failed", "tts_failed")]


@pytest.mark.parametrize(
    "abort_raises",
    [False, True],
    ids=("normal-abort", "ordinary-abort-error"),
)
@pytest.mark.asyncio
async def test_pre_active_tts_failure_waits_for_exact_clear_before_public_runner_cancel(
    monkeypatch: pytest.MonkeyPatch,
    abort_raises: bool,
) -> None:
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    abort_attempted = asyncio.Event()
    upstream_error_reached = asyncio.Event()
    pipeline_finished = asyncio.Event()
    upstream_errors: list[ErrorFrame] = []
    cancel_reasons: list[str | None] = []
    runtime_box: list[object] = []
    original_build_runtime = session_module.build_runtime
    original_request_clear = pipeline_module.CallRuntime.request_clear
    original_runner_cancel = WorkerRunner.cancel
    original_controller_abort = session_module.DisclosureController.abort

    def build_observed_runtime(**kwargs: object) -> object:
        runtime = original_build_runtime(**kwargs)
        runtime_box.append(runtime)

        async def observe_upstream_error(_worker: object, frame: Frame) -> None:
            assert isinstance(frame, ErrorFrame)
            upstream_errors.append(frame)
            upstream_error_reached.set()

        async def observe_pipeline_finished(_worker: object, _frame: Frame) -> None:
            pipeline_finished.set()

        runtime.worker.add_reached_upstream_filter((ErrorFrame,))
        runtime.worker.add_event_handler(
            "on_frame_reached_upstream", observe_upstream_error
        )
        runtime.worker.add_event_handler(
            "on_pipeline_finished", observe_pipeline_finished
        )
        return runtime

    async def block_exact_clear(runtime: object) -> None:
        clear_started.set()
        await clear_release.wait()
        await original_request_clear(runtime)

    async def observe_runner_cancel(
        runner: WorkerRunner,
        reason: str | None = None,
    ) -> None:
        cancel_reasons.append(reason)
        await original_runner_cancel(runner, reason=reason)

    async def raise_on_tts_abort(
        controller: session_module.DisclosureController,
        code: str,
    ) -> None:
        if code == "tts_failed":
            abort_attempted.set()
            raise RuntimeError("controller-abort-secret")
        await original_controller_abort(controller, code)

    monkeypatch.setattr(session_module, "build_runtime", build_observed_runtime)
    monkeypatch.setattr(pipeline_module.CallRuntime, "request_clear", block_exact_clear)
    monkeypatch.setattr(WorkerRunner, "cancel", observe_runner_cancel)
    if abort_raises:
        monkeypatch.setattr(
            session_module.DisclosureController,
            "abort",
            raise_on_tts_abort,
        )
    session, lease, writer, admission, sent, client, stream = (
        _actual_partial_tts_telnyx_session()
    )
    running = asyncio.create_task(session.run(session.handshake))
    try:
        await asyncio.wait_for(clear_started.wait(), timeout=5)
        await asyncio.wait_for(upstream_error_reached.wait(), timeout=5)
        assert abort_attempted.is_set() is abort_raises
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pipeline_finished.wait(), timeout=0.2)

        assert len(runtime_box) == 1
        assert runtime_box[0].worker.has_finished() is False
        assert running.done() is False
        assert cancel_reasons == []
        assert len(upstream_errors) == 1
        assert upstream_errors[0].error == "tts_failed"
        assert upstream_errors[0].fatal is False
        assert upstream_errors[0].exception is None
        assert upstream_errors[0].processor is None
        assert "tts-provider-secret" not in repr(upstream_errors[0])
        assert "controller-abort-secret" not in repr(upstream_errors[0])

        blocked_payloads = [
            json.loads(message["text"])
            for message in sent
            if message.get("type") == "websocket.send"
            and isinstance(message.get("text"), str)
        ]
        blocked_events = [payload.get("event") for payload in blocked_payloads]
        assert "media" in blocked_events
        assert "clear" not in blocked_events
        assert "mark" not in blocked_events
        assert not any(
            not isinstance(command, VoiceOperationV1)
            and hasattr(command, "payload")
            and command.payload["operation"].payload.disclosure_state == "completed"
            for command in writer.commands
        )
        assert admission.allows_audio() is False

        clear_release.set()
        with pytest.raises(session_module.CallSessionError, match="tts_failed"):
            await asyncio.wait_for(running, timeout=5)
    finally:
        clear_release.set()
        if not running.done():
            await asyncio.wait_for(
                asyncio.gather(running, return_exceptions=True),
                timeout=5,
            )
        await client.aclose()

    payloads = [
        json.loads(message["text"])
        for message in sent
        if message.get("type") == "websocket.send" and isinstance(message.get("text"), str)
    ]
    events = [payload.get("event") for payload in payloads]
    assert "media" in events
    assert events.count("clear") == 1
    assert "mark" not in events
    assert events.index("media") < events.index("clear")
    assert not any(
        not isinstance(command, VoiceOperationV1)
        and hasattr(command, "payload")
        and command.payload["operation"].payload.disclosure_state == "completed"
        for command in writer.commands
    )
    assert admission.allows_audio() is False
    assert cancel_reasons == ["local_failure"]
    assert pipeline_finished.is_set()
    assert lease.calls == [("failed", "tts_failed")]
    assert stream.closed is True


@pytest.mark.asyncio
async def test_completed_disclosure_remains_completed_in_later_cancel_terminal_snapshot() -> None:
    events: list[str] = []
    session, _stt, lease, writer, _transport = _session(events=events, echo_ack=True)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    terminal_operations = [
        command.payload["operation"]
        for command in writer.commands
        if hasattr(command, "payload")
        and command.payload["operation"].kind == "call.upsert"
        and command.payload["operation"].payload.status == "failed"
    ]
    assert len(terminal_operations) == 1
    assert terminal_operations[0].payload.disclosure_state == "completed"
    assert lease.calls == [("failed", "external_cancel")]


@pytest.mark.asyncio
async def test_request_drain_uses_public_runner_and_closes_active_session() -> None:
    events: list[str] = []
    session, _stt, lease, writer, _transport = _session(events=events, echo_ack=True)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=5)

    await session.request_drain()
    await asyncio.wait_for(running, timeout=5)

    assert lease.calls == [("closed", "closed")]
    assert events[-1] == "lease-terminal"


@pytest.mark.asyncio
async def test_request_drain_during_add_workers_is_replayed_after_runner_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    add_blocked = asyncio.Event()
    add_release = asyncio.Event()
    cancel_calls = 0
    original_add_workers = WorkerRunner.add_workers
    original_cancel = WorkerRunner.cancel

    async def blocked_add(runner: WorkerRunner, *workers: object) -> None:
        await original_add_workers(runner, *workers)
        add_blocked.set()
        await add_release.wait()

    async def observe_cancel(runner: WorkerRunner, reason: str | None = None) -> None:
        nonlocal cancel_calls
        cancel_calls += 1
        await original_cancel(runner, reason=reason)

    monkeypatch.setattr(WorkerRunner, "add_workers", blocked_add)
    monkeypatch.setattr(WorkerRunner, "cancel", observe_cancel)
    session, _stt, lease, _writer, _transport = _session(events=events)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(add_blocked.wait(), timeout=3)

    await session.request_drain()
    add_release.set()
    with pytest.raises(session_module.CallSessionError, match="call_failed"):
        await asyncio.wait_for(running, timeout=3)

    assert cancel_calls == 1
    assert lease.calls == [("failed", "call_failed")]


@pytest.mark.asyncio
async def test_real_delayed_turn_event_finishes_before_recorder_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    delayed_started = asyncio.Event()
    delayed_release = asyncio.Event()
    original_build_pipeline = session_module.build_pipeline

    def build_with_delayed_handler(**kwargs: object) -> object:
        pipeline = original_build_pipeline(**kwargs)
        aggregator = next(
            processor
            for processor in pipeline.processors
            if isinstance(processor, LLMUserAggregator)
        )

        async def delayed_handler(*_args: object) -> None:
            delayed_started.set()
            await delayed_release.wait()

        aggregator.add_event_handler("on_user_turn_stopped", delayed_handler)
        return pipeline

    monkeypatch.setattr(session_module, "build_pipeline", build_with_delayed_handler)
    session, stt, lease, writer, transport = _session(
        events=events,
        echo_ack=True,
        segmented_stt=True,
    )
    assert isinstance(stt, _SegmentedSessionStt)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=5)
    await transport.input().queue_frame(VADUserStartedSpeakingFrame())
    await transport.input().queue_frame(
        InputAudioRawFrame(
            audio=b"\x01\x00" * 400,
            sample_rate=8000,
            num_channels=1,
        )
    )
    await transport.input().queue_frame(VADUserStoppedSpeakingFrame())
    await asyncio.wait_for(delayed_started.wait(), timeout=5)
    await session.request_drain()
    await asyncio.sleep(0)
    assert running.done() is False
    delayed_release.set()
    await asyncio.wait_for(running, timeout=5)

    turns = [
        operation
        for operation in writer.commands
        if isinstance(operation, VoiceOperationV1) and operation.kind == "turn.upsert"
    ]
    assert len(turns) == 1
    assert turns[0].payload.turn_no == 1
    terminal_index = next(
        index
        for index, command in enumerate(writer.commands)
        if not isinstance(command, VoiceOperationV1)
        and hasattr(command, "payload")
        and command.payload["operation"].kind == "call.upsert"
        and command.payload["operation"].payload.status == "closed"
    )
    assert writer.commands.index(turns[0]) < terminal_index
    assert lease.calls == [("closed", "closed")]


@pytest.mark.asyncio
async def test_two_running_sessions_isolate_local_failure_then_share_writer_fatal() -> None:
    events: list[str] = []
    shared_writer = _SessionWriter(events)
    session_a, stt_a, lease_a, _writer_a, _transport_a = _session(
        events=events,
        nested_failure=True,
        call_int=1,
        writer_override=shared_writer,
    )
    session_b, stt_b, lease_b, _writer_b, _transport_b = _session(
        events=events,
        call_int=2,
        writer_override=shared_writer,
    )
    assert isinstance(stt_a, _PassProcessor)
    assert isinstance(stt_b, _PassProcessor)
    running_a = asyncio.create_task(session_a.run(session_a.handshake))
    running_b = asyncio.create_task(session_b.run(session_b.handshake))
    await asyncio.gather(stt_a.started.wait(), stt_b.started.wait())

    with pytest.raises(session_module.CallSessionError, match="pipeline_task_failed"):
        await asyncio.wait_for(running_a, timeout=3)
    assert running_b.done() is False
    assert session_b.handshake.audio_admission.allows_audio() is False
    assert lease_a.calls == [("failed", "pipeline_task_failed")]
    assert lease_b.calls == []

    shared_writer.fatal_event.set()
    with pytest.raises(session_module.CallSessionError, match="writer_failed"):
        await asyncio.wait_for(running_b, timeout=3)
    assert lease_b.calls == [("failed", "writer_failed")]


@pytest.mark.asyncio
async def test_call_session_rejects_mismatched_authenticated_lease_claim() -> None:
    events: list[str] = []
    session, _stt, _lease, _writer, _transport = _session(events=events)
    mismatch = replace(session.handshake, lease_claim=object())

    with pytest.raises(session_module.CallSessionError, match="call_identity_mismatch"):
        await session.run(mismatch)

    assert mismatch.audio_admission.is_bound is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_point",
    ["pipeline", "worker", "add_workers", "runtime", "transport_handlers"],
)
async def test_post_bind_startup_failure_runs_partial_state_cleanup(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session, _stt, lease, _writer, _transport = _session(events=events)
    worker_finished = asyncio.Event()

    if failure_point == "pipeline":
        def fail_pipeline(**_kwargs: object) -> None:
            raise RuntimeError("pipeline-construction-secret")

        monkeypatch.setattr(session_module, "build_pipeline", fail_pipeline)
    elif failure_point == "worker":
        def fail_worker(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("worker-construction-secret")

        monkeypatch.setattr(pipeline_module, "PipelineWorker", fail_worker)
    elif failure_point == "add_workers":
        original_add_workers = pipeline_module.WorkerRunner.add_workers

        async def fail_add_workers(runner: object, *workers: object) -> None:
            worker = workers[0]

            async def on_finished(_worker: object, _frame: object) -> None:
                worker_finished.set()

            worker.add_event_handler("on_pipeline_finished", on_finished)
            await original_add_workers(runner, *workers)
            raise RuntimeError("worker-registration-secret")

        monkeypatch.setattr(pipeline_module.WorkerRunner, "add_workers", fail_add_workers)
    elif failure_point == "runtime":
        def fail_runtime(**_kwargs: object) -> None:
            raise RuntimeError("worker-registration-secret")

        monkeypatch.setattr(session_module, "build_runtime", fail_runtime)
    else:
        def fail_handler_setup(_event_name: str, _handler: object) -> None:
            raise RuntimeError("transport-handler-secret")

        monkeypatch.setattr(session.handshake.transport, "add_event_handler", fail_handler_setup)

    with pytest.raises(session_module.CallSessionError, match="call_failed"):
        await asyncio.wait_for(session.run(session.handshake), timeout=2)

    assert "recording-cleanup" in events
    assert "services:stt" in events
    assert events.index("call-terminal") < events.index("lease-terminal")
    assert lease.calls == [("failed", "call_failed")]
    assert session.handshake.audio_admission.allows_audio() is False
    if failure_point == "add_workers":
        assert worker_finished.is_set() is False
        assert any(item.startswith("pipeline:") for item in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", [1, 2])
async def test_runner_and_failure_task_creation_failures_run_cleanup(
    fail_on: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    calls = 0
    worker_finished = asyncio.Event()
    original_build_runtime = session_module.build_runtime

    def build_observed_runtime(**kwargs: object) -> object:
        runtime = original_build_runtime(**kwargs)

        async def on_finished(_worker: object, _frame: object) -> None:
            worker_finished.set()

        runtime.worker.add_event_handler("on_pipeline_finished", on_finished)
        return runtime

    monkeypatch.setattr(session_module, "build_runtime", build_observed_runtime)

    def task_factory(coroutine: object, name: str) -> asyncio.Task[object]:
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError(f"{name}-creation-secret")
        return asyncio.create_task(coroutine, name=name)  # type: ignore[arg-type]

    session, _stt, lease, _writer, _transport = _session(
        events=events,
        task_factory=task_factory,
    )

    with pytest.raises(session_module.CallSessionError, match="call_failed"):
        await asyncio.wait_for(session.run(session.handshake), timeout=3)

    assert "recording-cleanup" in events
    assert "services:stt" in events
    assert events.index("call-terminal") < events.index("lease-terminal")
    assert lease.calls == [("failed", "call_failed")]
    assert any(item.startswith("pipeline:") for item in events)
    assert worker_finished.is_set() is (fail_on == 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["watch", "runner_task"])
async def test_direct_no_run_cleanup_covers_owned_public_surfaces(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    worker_cleanup = asyncio.Event()
    runner_cleanup = asyncio.Event()
    bus_stop = asyncio.Event()
    bus_cleanup = asyncio.Event()
    original_worker_cleanup = PipelineWorker.cleanup
    original_runner_cleanup = WorkerRunner.cleanup
    original_bus_stop = WorkerBus.stop
    original_bus_cleanup = WorkerBus.cleanup

    async def observe_worker_cleanup(worker: PipelineWorker) -> None:
        worker_cleanup.set()
        await original_worker_cleanup(worker)

    async def observe_runner_cleanup(runner: WorkerRunner) -> None:
        runner_cleanup.set()
        await original_runner_cleanup(runner)

    async def observe_bus_stop(bus: WorkerBus) -> None:
        bus_stop.set()
        await original_bus_stop(bus)

    async def observe_bus_cleanup(bus: WorkerBus) -> None:
        bus_cleanup.set()
        await original_bus_cleanup(bus)

    monkeypatch.setattr(PipelineWorker, "cleanup", observe_worker_cleanup)
    monkeypatch.setattr(WorkerRunner, "cleanup", observe_runner_cleanup)
    monkeypatch.setattr(WorkerBus, "stop", observe_bus_stop)
    monkeypatch.setattr(WorkerBus, "cleanup", observe_bus_cleanup)

    task_factory: object | None = None
    if failure_point == "watch":
        async def fail_watch(_registry: object, *_args: object) -> None:
            raise RuntimeError("registry-watch-secret")

        monkeypatch.setattr(WorkerRegistry, "watch", fail_watch)
    else:
        def fail_runner_task(coroutine: object, name: str) -> None:
            del coroutine, name
            raise RuntimeError("runner-task-secret")

        task_factory = fail_runner_task

    session, _stt, lease, _writer, _transport = _session(
        events=events,
        task_factory=task_factory,
    )
    with pytest.raises(session_module.CallSessionError, match="call_failed"):
        await asyncio.wait_for(session.run(session.handshake), timeout=3)

    assert any(item.startswith("pipeline:") for item in events)
    assert worker_cleanup.is_set()
    assert runner_cleanup.is_set()
    assert bus_stop.is_set()
    assert bus_cleanup.is_set()
    assert "services-attempt" in events
    assert lease.calls == [("failed", "call_failed")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger_timeout", "code"),
    [
        (False, "transport_disconnected"),
        (True, "transport_session_timeout"),
    ],
)
async def test_real_transport_disconnect_and_timeout_end_call_promptly(
    trigger_timeout: bool,
    code: str,
) -> None:
    session, lease, _writer, admission, _disconnect = _real_websocket_session(
        timeout=trigger_timeout
    )

    with pytest.raises(session_module.CallSessionError, match=code):
        await asyncio.wait_for(session.run(session.handshake), timeout=3)

    assert admission.allows_audio() is False
    assert lease.calls == [("failed", code)]


@pytest.mark.asyncio
async def test_real_active_websocket_disconnect_does_not_wait_for_idle_timeout() -> None:
    session, lease, writer, admission, disconnect = _real_websocket_session(
        timeout=False,
        active_before_disconnect=True,
    )
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=3)
    assert admission.allows_audio() is True
    disconnect.set()

    with pytest.raises(session_module.CallSessionError, match="transport_disconnected"):
        await asyncio.wait_for(running, timeout=3)

    assert admission.allows_audio() is False
    assert lease.calls == [("failed", "transport_disconnected")]
