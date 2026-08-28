from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pipecat.frames.frames import (
    Frame,
    InputTransportMessageFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import TelnyxCallData
from pydantic import SecretStr

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.inference.services import build_llm
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.telnyx.frames import TelnyxMarkFrame
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake
from projetv0_voice.telnyx.serializer import AudioAdmission

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


def _identity(lease_claim: object | None = None) -> object:
    return session_module.CallIdentity(
        call_id=UUID(int=1),
        durable_generation="generation-1",
        lease_identity="lease-1",
        lease_claim=lease_claim or object(),
        deployment_id="deployment-1",
        registry_handle="registry-1",
        telnyx_call_control_id="call-control-1",
        telnyx_call_leg_id="call-leg-1",
        telnyx_call_session_id="call-session-1",
        stream_id="stream-1",
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


class _SessionWriter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fatal_event = asyncio.Event()
        self.disclosure_committed = asyncio.Event()
        self.commands: list[object] = []

    def try_enqueue_turn(self, operation: object) -> bool:
        self.commands.append(operation)
        return True

    async def commit_control(self, command: object) -> None:
        self.commands.append(command)
        operation = command.payload["operation"]
        if operation.kind == "call.upsert" and operation.payload.disclosure_state == "completed":
            self.disclosure_committed.set()
        if operation.kind == "call.upsert" and operation.payload.status in {"closed", "failed"}:
            self.events.append("call-terminal")


class _RecordingBoundary:
    def __init__(self, events: list[str]) -> None:
        self.events = events

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
        del recording_may_be_active, reason
        self.events.append("recording-cleanup")


class _LeaseTerminalizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, str]] = []

    async def terminalize(self, _identity: object, *, status: str, reason: str) -> None:
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
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self._events = events

    async def aclose(self) -> None:
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


def _handshake(transport: object, lease_claim: object) -> AuthenticatedTelnyxHandshake:
    admission = AudioAdmission()
    return AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id="stream-1",
            call_id="call-control-1",
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
) -> tuple[object, _PassProcessor, _LeaseTerminalizer, _SessionWriter]:
    transport = _Transport(events, echo_ack=echo_ack)
    stt = _PassProcessor("stt", events, nested_failure=nested_failure)
    llm = _PassProcessor("llm", events)
    tts = _OfflineTts("tts", events)
    bundle = _SessionServices(
        stt=stt,
        llm=llm,
        tts=tts,
        events=events,
    )
    writer = _SessionWriter(events)
    lease = _LeaseTerminalizer(events)
    lease_claim = object()
    identity = _identity(lease_claim)
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
        recording=_RecordingBoundary(events),
        lease_terminalizer=lease,
        idle_timeout_seconds=60.0,
        utcnow=lambda: NOW + timedelta(seconds=5),
        uuid_factory=_uuid_factory(),
    )
    session.handshake = _handshake(transport, lease_claim)
    return session, stt, lease, writer


@pytest.mark.asyncio
async def test_public_call_session_cancellation_waits_for_ordered_bounded_cleanup() -> None:
    events: list[str] = []
    session, stt, lease, _writer = _session(events=events)
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
async def test_nested_native_task_failure_cancels_and_joins_real_runner_before_return() -> None:
    events: list[str] = []
    session, _stt, lease, _writer = _session(events=events, nested_failure=True)

    with pytest.raises(session_module.CallSessionError, match="pipeline_task_failed"):
        await asyncio.wait_for(session.run(session.handshake), timeout=1)

    assert lease.calls == [("failed", "pipeline_task_failed")]
    assert events[-1] == "lease-terminal"


@pytest.mark.asyncio
async def test_completed_disclosure_remains_completed_in_later_cancel_terminal_snapshot() -> None:
    events: list[str] = []
    session, _stt, lease, writer = _session(events=events, echo_ack=True)
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
    session, _stt, lease, writer = _session(events=events, echo_ack=True)
    running = asyncio.create_task(session.run(session.handshake))
    await asyncio.wait_for(writer.disclosure_committed.wait(), timeout=5)

    await session.request_drain()
    await asyncio.wait_for(running, timeout=5)

    assert lease.calls == [("closed", "closed")]
    assert events[-1] == "lease-terminal"


@pytest.mark.asyncio
async def test_local_failure_isolated_until_shared_writer_fatal_wakes_all_waiters() -> None:
    shared = asyncio.Event()
    first = pipeline_module.FirstFailure(shared_failure_event=shared)
    sibling = pipeline_module.FirstFailure(shared_failure_event=shared)
    first.signal("pipeline_task_failed")

    assert await first.wait() == "pipeline_task_failed"
    sibling_wait = asyncio.create_task(sibling.wait())
    await asyncio.sleep(0)
    assert sibling_wait.done() is False
    shared.set()
    assert await sibling_wait == "writer_failed"


@pytest.mark.asyncio
async def test_call_session_rejects_mismatched_authenticated_lease_claim() -> None:
    events: list[str] = []
    session, _stt, _lease, _writer = _session(events=events)
    mismatch = replace(session.handshake, lease_claim=object())

    with pytest.raises(session_module.CallSessionError, match="call_identity_mismatch"):
        await session.run(mismatch)

    assert mismatch.audio_admission.is_bound is False
