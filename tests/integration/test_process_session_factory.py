from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import TelnyxCallData
from pydantic import SecretStr

from projetv0_voice.admission import (
    CallConstructionGrant,
    CallGenerationHandle,
    CallRegistry,
    ProcessLeaseAuthority,
    ProcessLeaseClaim,
    _TerminalCapability,
)
from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.persistence.writer import WebhookCommitResult
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.session_factory import ProcessSessionFactory
from projetv0_voice.telnyx.call_control import CallControlResult
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake
from projetv0_voice.telnyx.webhooks import VerifiedWebhook

NOW = datetime(2026, 8, 31, 9, tzinfo=UTC)
CALL_ID = UUID("00000000-0000-4000-8000-000000000001")
GENERATION_ID = UUID("11111111-1111-4111-8111-111111111111")


def _claim() -> ProcessLeaseClaim:
    return ProcessLeaseClaim(
        call_control_id="control-a",
        call_id=CALL_ID,
        generation=GENERATION_ID,
        token_digest=b"d" * 32,
        claimed_at=NOW,
    )


def _grant(claim: ProcessLeaseClaim) -> CallConstructionGrant:
    return CallConstructionGrant(
        call_id=CALL_ID,
        generation=CallGenerationHandle("control-a", GENERATION_ID),
        lease_claim=claim,
        deployment_id="deployment-a",
        telnyx_call_control_id="control-a",
        telnyx_call_leg_id="leg-a",
        telnyx_call_session_id="session-a",
        stream_id="stream-a",
        started_at=NOW,
        retention_until=NOW + timedelta(days=7),
    )


class _AudioAdmission:
    def bind(self, _gate: object) -> None:
        return None


def _handshake(claim: ProcessLeaseClaim) -> AuthenticatedTelnyxHandshake:
    return AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id="stream-a",
            call_id="control-a",
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=claim,
        transport=object(),  # type: ignore[arg-type]
        audio_admission=_AudioAdmission(),  # type: ignore[arg-type]
    )


def _mismatched_handshake(
    claim: ProcessLeaseClaim,
    events: list[str],
) -> AuthenticatedTelnyxHandshake:
    return AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id="stream-a",
            call_id="wrong-control",
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=claim,
        transport=_OfflineTransport(events),  # type: ignore[arg-type]
        audio_admission=_AudioAdmission(),  # type: ignore[arg-type]
    )


def _exact_real_handshake(
    claim: ProcessLeaseClaim,
    events: list[str],
) -> AuthenticatedTelnyxHandshake:
    return AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id="stream-a",
            call_id="control-a",
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=claim,
        transport=_OfflineTransport(events),  # type: ignore[arg-type]
        audio_admission=_AudioAdmission(),  # type: ignore[arg-type]
    )


def _claim_handshake(
    claim: ProcessLeaseClaim,
    events: list[str],
) -> AuthenticatedTelnyxHandshake:
    suffix = claim.call_control_id.removeprefix("control-")
    return AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(
            stream_id=f"stream-{suffix}",
            call_id=claim.call_control_id,
            outbound_encoding="PCMU",
        ),
        token_locator_id="telnyx-header-connected-v1",
        lease_claim=claim,
        transport=_OfflineTransport(events),  # type: ignore[arg-type]
        audio_admission=_AudioAdmission(),  # type: ignore[arg-type]
    )


class _Registrar:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.tasks: list[asyncio.Task[None]] = []
        self.started = asyncio.Event()

    def try_start(
        self,
        coroutine: Any,
        *,
        name: str,
    ) -> asyncio.Task[None] | None:
        if self.reject:
            coroutine.close()
            return None
        task = asyncio.create_task(coroutine, name=name)
        self.tasks.append(task)
        self.started.set()
        return task


class _EagerRegistrar(_Registrar):
    def try_start(
        self,
        coroutine: Any,
        *,
        name: str,
    ) -> asyncio.Task[None] | None:
        task = asyncio.Task(
            coroutine,
            loop=asyncio.get_running_loop(),
            name=name,
            eager_start=True,
        )
        self.tasks.append(task)
        self.started.set()
        return task


class _Registry:
    def __init__(self, grant: CallConstructionGrant) -> None:
        self.grant = grant
        self.consume_calls: list[tuple[object, str, object, asyncio.Task[None]]] = []
        self.live_checks = 0
        self.activated = asyncio.Event()
        self.required_drains: list[UUID] = []
        self.activation_error: BaseException | None = None
        self.consume_error: BaseException | None = None
        self.persist_call = False
        self.terminal_failures: list[str] = []
        self.preconsume_probe: Any = None

    async def consume_claim_for_construction(
        self,
        claim: ProcessLeaseClaim,
        stream_id: str,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> CallConstructionGrant | None:
        self.consume_calls.append((claim, stream_id, owner, owner_task))
        assert owner_task.done() is False
        if self.preconsume_probe is not None:
            self.preconsume_probe()
        if self.consume_error is not None:
            raise self.consume_error
        owner._terminal_capability = _TerminalCapability(  # type: ignore[attr-defined]  # noqa: SLF001
            grant=self.grant,
            owner=owner,
            owner_task=owner_task,
        )
        return self.grant

    async def construction_is_live(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> bool:
        self.live_checks += 1
        return grant is self.grant and owner_task is asyncio.current_task()

    async def activate_session(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
        session: object,
    ) -> bool:
        assert grant is self.grant
        assert owner_task is asyncio.current_task()
        assert owner._session is session
        if self.activation_error is not None:
            raise self.activation_error
        self.activated.set()
        return True

    async def reserve_or_read_terminal(
        self,
        grant: CallConstructionGrant,
        capability: _TerminalCapability,
        proposed: object,
    ) -> Any:
        del grant, capability
        return SimpleNamespace(
            status=proposed.status,
            reason=proposed.reason,
            metric_class=proposed.metric_class,
            cleanup_hangup=proposed.cleanup_hangup,
            persist_call=self.persist_call,
            persist_lease=False,
            completion_token=UUID(int=9),
            _closed_at=NOW,
        )

    async def complete_reserved_terminal(self, _authority: object) -> bool:
        return True

    async def prepare_required_recording_drain(self, call_id: UUID) -> None:
        self.required_drains.append(call_id)
        return None

    def _note_terminal_failure(self, code: str) -> None:
        self.terminal_failures.append(code)


class _Processor(FrameProcessor):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__()
        self._test_name = name
        self.events = events

    async def cleanup(self) -> None:
        self.events.append(f"{self._test_name}-cleanup")
        await super().cleanup()


class _SttClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def aclose(self) -> None:
        self.events.append("stt-client-close")


class _LlmClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def close(self) -> None:
        self.events.append("llm-client-close")


class _Llm(_Processor):
    def __init__(self, events: list[str]) -> None:
        super().__init__("llm", events)
        self._client = _LlmClient(events)


class _PassingProcessor(_Processor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _PassingLlm(_PassingProcessor):
    def __init__(self, events: list[str]) -> None:
        super().__init__("llm", events)
        self._client = _LlmClient(events)


class _PassingTts(_PassingProcessor):
    async def process_frame(self, frame, direction):
        from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, TTSStoppedFrame

        if isinstance(frame, TTSSpeakFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            await self.push_frame(TTSAudioRawFrame(
                audio=b"\x01\x00" * 80, sample_rate=8000, num_channels=1,
                context_id="disclosure",
            ), direction)
            await self.push_frame(TTSStoppedFrame(context_id="disclosure"), direction)
            return
        await super().process_frame(frame, direction)


class _Recording:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self, _identity: object) -> object:
        raise AssertionError("not started by factory")

    async def cleanup(self, *_args: object, **_kwargs: object) -> None:
        self.events.append("recording-cleanup")


class _OfflineTransport:
    def __init__(self, events: list[str]) -> None:
        self._input = _Processor("transport-input", events)
        self._output = _Processor("transport-output", events)

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    def add_event_handler(self, _event_name: str, _handler: object) -> None:
        return None


class _RealWriter:
    def __init__(
        self,
        *,
        terminal_faults: list[BaseException] | None = None,
        terminal_started: asyncio.Event | None = None,
        terminal_release: asyncio.Event | None = None,
    ) -> None:
        self.fatal_event = asyncio.Event()
        self.lease_commits: list[dict[str, object]] = []
        self.control_commits: list[object] = []
        self.terminal_faults = list(terminal_faults or [])
        self.terminal_started = terminal_started
        self.terminal_release = terminal_release

    async def commit_lease(self, **values: object) -> None:
        self.lease_commits.append(values)

    async def commit_control(self, command: object) -> None:
        self.control_commits.append(command)
        operation = command.payload["operation"]  # type: ignore[union-attr]
        if operation.kind == "call.upsert" and operation.payload.status in {"closed", "failed"}:
            if self.terminal_started is not None:
                self.terminal_started.set()
            if self.terminal_release is not None:
                await self.terminal_release.wait()
            if self.terminal_faults:
                raise self.terminal_faults.pop(0)


class _Task9SequencedWriter(_RealWriter):
    def __init__(
        self,
        outcomes: list[BaseException | None],
        *,
        blocked_attempts: frozenset[int],
    ) -> None:
        super().__init__()
        self.outcomes = outcomes
        self.blocked_attempts = blocked_attempts
        self.started = [asyncio.Event() for _ in outcomes]
        self.releases = [asyncio.Event() for _ in outcomes]

    async def commit_control(self, command: object) -> None:
        index = len(self.control_commits)
        self.control_commits.append(command)
        self.started[index].set()
        if index in self.blocked_attempts:
            await self.releases[index].wait()
        outcome = self.outcomes[index]
        if outcome is not None:
            raise outcome

    def try_enqueue_turn(self, _operation: object) -> bool:
        return True


class _RealControl:
    def __init__(self) -> None:
        self.hangups: list[tuple[str, UUID]] = []

    async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
        return CallControlResult("accepted")

    async def start_streaming(
        self, *_args: object, **_kwargs: object
    ) -> CallControlResult:
        return CallControlResult("accepted")

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult:
        del client_state
        self.hangups.append((call_control_id, command_id))
        return CallControlResult("accepted")


def _webhook(
    event_type: str,
    event_id: str,
    *,
    call_control_id: str = "control-a",
    call_leg_id: str = "leg-a",
    call_session_id: str = "session-a",
) -> VerifiedWebhook:
    return VerifiedWebhook(
        event_id=event_id,
        event_type=event_type,
        occurred_at=NOW,
        call_control_id=call_control_id,
        call_leg_id=call_leg_id,
        call_session_id=call_session_id,
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=(event_id.encode() + b"_" * 32)[:32],
        direction="incoming" if event_type == "call.initiated" else None,
        call_state="parked" if event_type == "call.initiated" else "answered",
    )


def _real_manifest() -> AgentManifestV1:
    return AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "agent_id": "agent-a",
            "revision": "revision-a",
            "tenant_id": "tenant-a",
            "dids": ["+33123456789"],
            "language": "fr-FR",
            "prompt_path": "prompt.md",
            "prompt_revision": "prompt-a",
            "greeting": "Bonjour.",
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


def _real_profile() -> QualifiedDeploymentProfileV1:
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(
            encoding="utf-8"
        )
    )
    return profile.model_copy(update={"deployment_id": "deployment-a"})


async def _real_claim(
    writer: _RealWriter,
    control: _RealControl,
) -> tuple[CallRegistry, ProcessLeaseClaim]:
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=lambda: 100.0,
        token_factory=lambda _size: "A" * 43,
        prefix_factory=lambda: 0,
        uuid_factory=iter(
            UUID(int=(4 << 76) | (0b10 << 62) | index)
            for index in range(1, 100)
        ).__next__,
    )
    initiated = _webhook("call.initiated", "initiated-a")
    resolution = await registry.resolve_webhook(initiated)
    await registry.reconcile_after_commit(
        initiated, resolution, WebhookCommitResult("first", "applied")
    )
    answered = _webhook("call.answered", "answered-a")
    resolution = await registry.resolve_webhook(answered)
    await registry.reconcile_after_commit(
        answered, resolution, WebhookCommitResult("first", "applied")
    )
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a",
        token_digest=bytes.fromhex(
            "0f007385b6f9d4b7eeb2748605afe1a984a0a3bfa3f014d09e2a784ce9e5cd1a"
        ),
    )
    assert claim is not None
    return registry, claim


async def _real_claims(
    writer: _RealWriter,
    control: _RealControl,
    count: int,
) -> tuple[CallRegistry, list[ProcessLeaseClaim]]:
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=count,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=lambda: 100.0,
        token_factory=lambda _size: "A" * 43,
        prefix_factory=lambda: 1,
        uuid_factory=iter(
            UUID(int=(4 << 76) | (0b10 << 62) | index)
            for index in range(1, 1000)
        ).__next__,
    )
    authority = ProcessLeaseAuthority(registry)
    claims: list[ProcessLeaseClaim] = []
    digest = bytes.fromhex(
        "0f007385b6f9d4b7eeb2748605afe1a984a0a3bfa3f014d09e2a784ce9e5cd1a"
    )
    for index in range(count):
        control_id = f"control-{index}"
        leg_id = f"leg-{index}"
        session_id = f"session-{index}"
        initiated = _webhook(
            "call.initiated",
            f"initiated-{index}",
            call_control_id=control_id,
            call_leg_id=leg_id,
            call_session_id=session_id,
        )
        resolution = await registry.resolve_webhook(initiated)
        await registry.reconcile_after_commit(
            initiated,
            resolution,
            WebhookCommitResult("first", "applied"),
        )
        answered = _webhook(
            "call.answered",
            f"answered-{index}",
            call_control_id=control_id,
            call_leg_id=leg_id,
            call_session_id=session_id,
        )
        resolution = await registry.resolve_webhook(answered)
        await registry.reconcile_after_commit(
            answered,
            resolution,
            WebhookCommitResult("first", "applied"),
        )
        claim = await authority.claim_once(
            call_control_id=control_id,
            token_digest=digest,
        )
        assert claim is not None
        claims.append(claim)
    return registry, claims


def _real_process_factory(
    *,
    registry: CallRegistry,
    writer: _RealWriter,
    metrics: RuntimeMetrics,
    events: list[str],
    registrar: _Registrar | None = None,
    pass_frames: bool = False,
) -> ProcessSessionFactory:
    processor = _PassingProcessor if pass_frames else _Processor
    return ProcessSessionFactory(
        registry=registry,
        registrar=registrar or _Registrar(),
        runtime_metrics=metrics,
        manifest=_real_manifest(),
        profile=_real_profile(),
        writer=writer,
        keyring=CryptoKeyring(
            {1: b"k" * 32},
            active_version=1,
            nonce_factory=lambda size: b"n" * size,
        ),
        stt_http_client_factory=lambda: _SttClient(events),
        stt_factory=lambda _client: processor("stt", events),
        llm_factory=lambda: _PassingLlm(events) if pass_frames else _Llm(events),
        tts_factory=lambda: (
            _PassingTts("tts", events) if pass_frames else _Processor("tts", events)
        ),
        recording_factory=lambda _identity: _Recording(events),
        idle_timeout_seconds=30.0,
    )


class _Session:
    def __init__(self, events: list[str], **values: object) -> None:
        self.events = events
        self.values = values
        self.handshakes: list[AuthenticatedTelnyxHandshake] = []

    async def run(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        self.events.append("session-run")
        self.handshakes.append(handshake)
        self.values["metric_lease"].finish("closed")  # type: ignore[union-attr]

    async def aclose_unstarted(self) -> None:
        self.events.append("session-aclose-unstarted")
        self.values["metric_lease"].finish("failed")  # type: ignore[union-attr]

    async def request_drain(self, reason: str) -> None:
        self.events.append(f"session-drain:{reason}")


@pytest.mark.parametrize(("clear_behavior", "operation_version", "early_bridge"), [
    ("stall", 1, False), ("raise", 1, False), ("cancel", 1, False),
    ("waiter_cancel", 1, False), ("erase", 1, False),
    ("raise", 1, True), ("raise", 2, False), ("raise", 2, True),
])
@pytest.mark.asyncio
async def test_correlated_bridge_cancels_native_ai_despite_clear_failure(
    tmp_path, monkeypatch, clear_behavior, operation_version, early_bridge
):
    import dataclasses
    from datetime import timedelta
    from uuid import uuid4

    from pydantic import SecretStr

    from projetv0_voice.audio_contract import BeginCallSnapshotV2
    from projetv0_voice.config import SparraManifestV1
    from projetv0_voice.models import BeginCallSnapshotV1
    from projetv0_voice.persistence.writer import PersistenceWriter
    from projetv0_voice.pipeline import CallRuntime
    from projetv0_voice.session import CallSession

    destination = "+33102030406"
    did = "+33102030405"
    policy = SparraManifestV1(
        schema_version=1,
        operation_contract_version=operation_version,
        connection_id="fixture",
        original_forward_line_e164=None,
        qualified_transfer_destination_e164=destination,
    )
    writer = PersistenceWriter(
        tmp_path / "takeover.sqlite", CryptoKeyring({1: b"k" * 32}, active_version=1),
        contract_version=operation_version,
        **({"process_agent_id": "agent-a", "process_deployment_id": "deployment-a"}
           if operation_version == 2 else {}),
        utcnow=lambda: NOW,
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()

    class Control(_RealControl):
        async def transfer(self, control_id, request, *, command_id):
            return CallControlResult("accepted")

    control = Control()

    async def begin(deployment, call_id, routing):
        original = BeginCallSnapshotV1(
            schema_version=1,
            call_id=call_id,
            configuration_revision=1,
            knowledge=dict(
                business_name="Garage",
                sector="garage",
                opening_hours="",
                services="",
                prices="",
                faq="",
                instructions="",
            ),
            transfer_destination=destination,
            retention_until=routing.admitted_at + timedelta(days=30),
        )
        if operation_version == 1:
            return original
        return BeginCallSnapshotV2.model_validate({
            **original.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(UUID(int=22)), "recording_policy": "off",
            "recording_contact_phone": None, "audio_available": False, "recording_id": None,
        })

    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id=str(UUID(int=22)) if operation_version == 2 else "tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        retention_days=30,
        stream_url="wss://fixture.invalid/media",
        utcnow=lambda: NOW,
        monotonic=lambda: 100.0,
        token_factory=lambda size: "A" * 43,
        sparra=policy,
        original_call_limit_seconds=300,
        called_did=did,
        begin_call=begin,
    )

    async def observed(kind, **changes):
        received = dataclasses.replace(
            _webhook(kind, str(uuid4())),
            connection_id="fixture",
            to_e164=did,
            from_e164=None,
            **changes,
        )
        resolution = await registry.resolve_webhook(received)
        effect = resolution.effect
        ticket = writer.submit_webhook(
            receipt=dict(
                event_id=received.event_id,
                event_type=kind,
                call_control_id=received.call_control_id,
                occurred_at=received.occurred_at,
                received_at=NOW,
                semantic_fingerprint_sha256=received.semantic_fingerprint_sha256,
            ),
            lease=None if effect is None else effect.lease,
            operation=None if effect is None else effect.operation,
            admission_facts=None if effect is None else effect.admission_facts,
            operation_generation=None if effect is None else effect.operation_generation,
        )
        await registry.reconcile_after_commit(received, resolution, await ticket.wait())

    await observed("call.initiated")
    await observed("call.answered")
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a",
        token_digest=bytes.fromhex(
            "0f007385b6f9d4b7eeb2748605afe1a984a0a3bfa3f014d09e2a784ce9e5cd1a"
        ),
    )
    assert claim is not None
    entry = registry._by_control["control-a"]
    facts = await writer.read_call_lifecycle(claim.call_id)
    assert facts is not None and facts.admitted_at == NOW
    assert facts.retention_until == NOW + timedelta(days=30)
    assert facts.admission_generation == claim.generation
    assert facts.recording_policy_revision == (1 if operation_version == 1 else None)
    assert facts.recording_enabled is False
    assert entry.begin_snapshot is not None
    assert entry.begin_snapshot.call_id == claim.call_id
    assert entry.begin_snapshot.retention_until == facts.retention_until
    assert writer._utcnow() < facts.retention_until
    events = []
    metrics = RuntimeMetrics.in_memory()
    factory = _real_process_factory(
        registry=registry, writer=writer, metrics=metrics, events=events
    )
    factory._manifest = _real_manifest().model_copy(
        update={"sparra": policy, "dids": (did,), "transcript_retention_days": 30}
    )
    factory._session_factory = lambda **kwargs: CallSession(
        **{**kwargs, "cleanup_phase_timeout_seconds": 0.1, "utcnow": lambda: NOW}
    )

    async def forward_native_frame(current, frame, direction):
        await FrameProcessor.process_frame(current, frame, direction)
        await current.push_frame(frame, direction)

    monkeypatch.setattr(_Processor, "process_frame", forward_native_frame)
    from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame

    class NativeTtsBoundary(_Processor):
        async def process_frame(self, frame, direction):
            await FrameProcessor.process_frame(self, frame, direction)
            if isinstance(frame, TTSSpeakFrame):
                await self.push_frame(
                    TTSAudioRawFrame(audio=bytes(320), sample_rate=8000, num_channels=1), direction
                )
            else:
                await self.push_frame(frame, direction)

    factory._tts_factory = lambda: NativeTtsBoundary("tts", events)
    ready = asyncio.Event()
    original_replay = CallSession._replay_pending_drain

    async def replay(current):
        await original_replay(current)
        ready.set()

    monkeypatch.setattr(CallSession, "_replay_pending_drain", replay)
    clear_started = asyncio.Event()

    async def clear(current):
        clear_started.set()
        if clear_behavior == "raise":
            raise RuntimeError("owned clear fixture failure")
        if clear_behavior == "cancel":
            raise asyncio.CancelledError("owned clear fixture cancellation")
        await asyncio.Event().wait()

    monkeypatch.setattr(CallRuntime, "request_clear", clear)
    handshake = _exact_real_handshake(claim, events)
    handshake = dataclasses.replace(
        handshake, call_data=handshake.call_data.model_copy(update={"to_number": did})
    )
    running = asyncio.create_task(factory.run(handshake))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        session = registry._by_control["control-a"].session
        for _ in range(100):
            if session._controller.state.name == "MARK_PENDING":
                break
            await asyncio.sleep(0.01)
        assert session._controller.state.name == "MARK_PENDING"
        assert await session._controller.accept_mark(session._controller.mark_name)
        await session._controller.join_continuations()
        if clear_behavior == "erase":
            from projetv0_voice.persistence.postgres_sink import CallErasureLease
            from projetv0_voice.persistence.relay import maintain_call_content

            before = registry._by_control["control-a"]
            owner = before.lifecycle_owner
            assert before.begin_future.done()
            lease = CallErasureLease(
                1,
                claim.call_id,
                uuid4(),
                "deployment-a",
                NOW + timedelta(days=30),
                NOW + timedelta(seconds=30),
            )

            class ErasureSink:
                async def lease_call_erasures(self, *args):
                    return (lease,)

                async def ack_call_erasure(self, *args):
                    assert before.routing is None and before.begin_snapshot is None
                    assert before.begin_future is None and before.begin_task is None
                    assert before.construction_grant is None and before.session is None
                    assert before.lifecycle_owner is None and before.lifecycle_owner_task is None
                    assert (
                        owner._grant is None and owner._handshake is None and owner._session is None
                    )
                    assert "stt-client-close" in events and "llm-client-close" in events

            await asyncio.wait_for(
                maintain_call_content(
                    writer,
                    ErasureSink(),
                    factory.erase_call_by_id,
                    utcnow=lambda: NOW,
                    timeout_seconds=5,
                ),
                timeout=5,
            )
            await asyncio.wait_for(running, timeout=5)
            facts = await writer.read_call_lifecycle(claim.call_id)
            assert facts.content_erased and facts.disclosure_evidence is None
            assert facts.content_departed_generation == claim.generation
            assert (await writer.read_retained_call(claim.call_id)).erased
            assert await writer.oldest_outbox_created_at() is None
            assert control.hangups == []
            assert await registry.live_call_count() == 1
            assert "stt-client-close" in events and "llm-client-close" in events
            return
        generation = await registry.generation_handle("control-a")
        assert await registry.request_human(generation) == "ringing"
        facts = await writer.read_call_lifecycle(claim.call_id)
        target = dict(
            call_control_id="target",
            call_leg_id="target-leg",
            client_state=SecretStr(facts.transfer_correlation),
            direction="outgoing",
            call_state=None,
        )

        # Rebind the destination through the signed target event, never a model argument.
        async def target_event(kind):
            received = dataclasses.replace(
                _webhook(kind, str(uuid4())),
                connection_id="fixture",
                to_e164=destination,
                from_e164=None,
                **{**target, "direction": "outgoing" if kind == "call.initiated" else None},
            )
            return await registry.resolve_webhook(received)

        if not early_bridge:
            await target_event("call.initiated")
        takeover = asyncio.create_task(target_event("call.bridged"))
        if clear_behavior == "waiter_cancel":
            await clear_started.wait()
            takeover.cancel()
        await asyncio.wait_for(takeover, timeout=1)
        assert session.no_new_ai
        if early_bridge:
            await target_event("call.initiated")
        await session.request_drain("qualified_line_connected")
        await asyncio.wait_for(running, timeout=5)
        assert clear_started.is_set()
        facts = await writer.read_call_lifecycle(claim.call_id)
        assert facts.qualified_line_bridged_at == NOW
        assert facts.bridge_operation_id is not None
        assert session._drain_task.done()
        assert registry._by_control["control-a"].begin_future is None
        items = await writer.read_relay_batch(
            batch_size=100, now=datetime.now(UTC) + timedelta(seconds=1), lease_seconds=60
        )
        call_payloads = [
            item.operation.payload for item in items if item.operation.kind == "call.upsert"
        ]
        assert any(payload.status == "closing" for payload in call_payloads)
        assert all(payload.ended_at is None for payload in call_payloads)
        assert control.hangups == []
        assert await registry.live_call_count() == 1
        assert "stt-client-close" in events and "llm-client-close" in events
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await writer.drain(2)
        await writer_task
        metrics._provider.shutdown(timeout_millis=10000.0)


def _metric_points(owner: RuntimeMetrics, name: str) -> list[Any]:
    data = owner._metric_reader.get_metrics_data()  # noqa: SLF001
    if data is None:
        return []
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def _factory(
    *,
    registry: _Registry,
    registrar: _Registrar,
    runtime_metrics: RuntimeMetrics,
    events: list[str],
    session_error: bool = False,
    writer: object | None = None,
) -> Any:
    from projetv0_voice.session_factory import ProcessSessionFactory

    def stt_http_client_factory() -> _SttClient:
        events.append("stt-client")
        return _SttClient(events)

    def stt_factory(_client: _SttClient) -> _Processor:
        events.append("stt")
        return _Processor("stt", events)

    def llm_factory() -> _Llm:
        events.append("llm")
        return _Llm(events)

    def tts_factory() -> _Processor:
        events.append("tts")
        return _Processor("tts", events)

    def recording_factory(_identity: object) -> _Recording:
        events.append("recording")
        return _Recording(events)

    def session_factory(**values: object) -> _Session:
        events.append("session-constructor")
        if session_error:
            raise RuntimeError("constructor-secret")
        return _Session(events, **values)

    return ProcessSessionFactory(
        registry=registry,
        registrar=registrar,
        runtime_metrics=runtime_metrics,
        manifest=object(),
        profile=object(),
        writer=writer or object(),
        keyring=object(),
        stt_http_client_factory=stt_http_client_factory,
        stt_factory=stt_factory,
        llm_factory=llm_factory,
        tts_factory=tts_factory,
        recording_factory=recording_factory,
        session_factory=session_factory,
        idle_timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_registrar_rejection_closes_candidate_before_claim_or_metric() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    registrar = _Registrar(reject=True)
    metrics = RuntimeMetrics.in_memory(monotonic=lambda: 1.0)
    events: list[str] = []
    registry.preconsume_probe = lambda: (
        events == [] or pytest.fail("eager owner crossed closed gate before grant")
    )
    factory = _factory(
        registry=registry,
        registrar=registrar,
        runtime_metrics=metrics,
        events=events,
    )

    with pytest.raises(RuntimeError, match="^session_owner_registration_failed$"):
        await factory.run(_handshake(claim))

    assert registry.consume_calls == []
    assert events == []
    assert _metric_points(metrics, "projetv0.voice.calls.active") == []
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_pregrant_exception_cancels_and_joins_registered_candidate() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    registry.consume_error = RuntimeError("consume-secret")
    registrar = _Registrar()
    metrics = RuntimeMetrics.in_memory()
    factory = _factory(
        registry=registry,
        registrar=registrar,
        runtime_metrics=metrics,
        events=[],
    )

    try:
        with pytest.raises(RuntimeError, match="consume-secret"):
            await factory.run(_handshake(claim))
        assert len(registrar.tasks) == 1
        assert registrar.tasks[0].done()
    finally:
        for task in registrar.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*registrar.tasks, return_exceptions=True)
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_eager_registrar_reaches_only_closed_owner_gate_before_grant() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    registrar = _EagerRegistrar()
    metrics = RuntimeMetrics.in_memory(monotonic=iter((1.0, 2.0)).__next__)
    events: list[str] = []
    registry.preconsume_probe = lambda: (
        events == [] or pytest.fail("eager owner crossed closed gate before grant")
    )
    factory = _factory(
        registry=registry,
        registrar=registrar,
        runtime_metrics=metrics,
        events=events,
    )

    await factory.run(_handshake(claim))

    assert len(registrar.tasks) == 1
    assert registrar.tasks[0].done()
    assert events[:2] == ["stt-client", "stt"]
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_factory_transfers_exact_grant_resources_observers_and_metric_to_owner() -> None:
    claim = _claim()
    grant = _grant(claim)
    registry = _Registry(grant)
    registrar = _Registrar()
    samples = iter((5.0, 8.0))
    metrics = RuntimeMetrics.in_memory(monotonic=lambda: next(samples))
    events: list[str] = []
    factory = _factory(
        registry=registry,
        registrar=registrar,
        runtime_metrics=metrics,
        events=events,
    )
    handshake = _handshake(claim)

    await factory.run(handshake)

    assert len(registry.consume_calls) == 1
    consumed_claim, stream_id, owner, retained_task = registry.consume_calls[0]
    assert consumed_claim is claim
    assert stream_id == "stream-a"
    assert retained_task is registrar.tasks[0]
    assert owner._phase == "done"
    assert registry.live_checks == 9
    assert events == [
        "stt-client",
        "stt",
        "llm",
        "tts",
        "recording",
        "session-constructor",
        "session-run",
    ]
    duration = _metric_points(metrics, "projetv0.voice.sessions.duration")
    assert len(duration) == 1
    assert (duration[0].sum, duration[0].attributes["session"]) == (3.0, "closed")
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_constructor_failure_retains_outer_reverse_cleanup_and_metric_authority() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    registrar = _Registrar()
    samples = iter((10.0, 10.25))
    metrics = RuntimeMetrics.in_memory(monotonic=lambda: next(samples))
    events: list[str] = []
    factory = _factory(
        registry=registry,
        registrar=registrar,
        runtime_metrics=metrics,
        events=events,
        session_error=True,
    )

    await factory.run(_handshake(claim))

    assert events.count("recording-cleanup") == 1
    assert events.count("stt-client-close") == 1
    assert events.count("llm-client-close") == 1
    cleanup = [event for event in events if event.endswith("-cleanup")]
    assert cleanup == [
        "recording-cleanup",
        "tts-cleanup",
        "llm-cleanup",
        "stt-cleanup",
    ]
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert len(total) == 1
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_required_drain_validates_exact_types_before_idempotent_lookup() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    metrics = RuntimeMetrics.in_memory()
    factory = _factory(
        registry=registry,
        registrar=_Registrar(),
        runtime_metrics=metrics,
        events=[],
    )

    class ReasonSubclass(str):
        pass

    class UuidSubclass(UUID):
        pass

    invalid = (
        (CALL_ID, "wrong", "call_drain_reason_invalid"),
        (CALL_ID, ReasonSubclass("recording_required_error"), "call_drain_reason_invalid"),
        ("not-a-uuid", "recording_required_error", "call_drain_call_id_invalid"),
        (UuidSubclass(str(CALL_ID)), "recording_required_error", "call_drain_call_id_invalid"),
    )
    for call_id, reason, code in invalid:
        with pytest.raises(ValueError, match=f"^{code}$") as raised:
            await factory.drain_call_by_id(call_id, reason)
        assert raised.value.__cause__ is None
    assert registry.required_drains == []

    await factory.drain_call_by_id(CALL_ID, "recording_required_error")
    await factory.drain_call_by_id(CALL_ID, "recording_required_error")

    assert registry.required_drains == [CALL_ID, CALL_ID]
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_activation_failure_uses_transferred_session_cleanup_once() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    registry.activation_error = RuntimeError("activation-secret")
    metrics = RuntimeMetrics.in_memory(monotonic=iter((2.0, 3.0)).__next__)
    events: list[str] = []
    factory = _factory(
        registry=registry,
        registrar=_Registrar(),
        runtime_metrics=metrics,
        events=events,
    )

    await factory.run(_handshake(claim))

    assert events.count("session-aclose-unstarted") == 1
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert len(total) == 1
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_post_freeze_construction_persistence_fault_latches_process_failure() -> None:
    claim = _claim()
    registry = _Registry(_grant(claim))
    registry.persist_call = True
    metrics = RuntimeMetrics.in_memory(monotonic=iter((3.0, 4.0)).__next__)

    class FailingWriter:
        fatal_event = asyncio.Event()

        async def commit_control(self, _command: object) -> None:
            raise RuntimeError("persistence-secret")

    factory = _factory(
        registry=registry,
        registrar=_Registrar(),
        runtime_metrics=metrics,
        events=[],
        session_error=True,
        writer=FailingWriter(),
    )

    await factory.run(_handshake(claim))

    assert registry.terminal_failures == ["terminal_persistence_failed"]
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert len(total) == 1
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_real_composition_mismatched_handshake_cleans_terminalizes_and_balances_metric() -> (
    None
):
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((10.0, 11.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )
    handshake = _mismatched_handshake(claim, events)

    await factory.run(handshake)

    snapshot = await registry.snapshot("control-a")
    assert snapshot is None
    assert events.count("recording-cleanup") == 1
    assert len(writer.control_commits) == 1
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.end_reason == "call_identity_mismatch"
    assert [commit["state"] for commit in writer.lease_commits] == [
        "active",
        "terminal",
    ]
    assert len(control.hangups) == 1
    active = _metric_points(metrics, "projetv0.voice.calls.active")
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert active[0].value == 0
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_ten_same_claim_factory_contenders_retain_one_real_winner() -> None:
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    registrar = _Registrar()
    metrics = RuntimeMetrics.in_memory(monotonic=iter((60.0, 61.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
        registrar=registrar,
    )

    results = await asyncio.gather(
        *(
            factory.run(_mismatched_handshake(claim, events))
            for _index in range(10)
        ),
        return_exceptions=True,
    )

    assert sum(result is None for result in results) == 1
    losers = [result for result in results if isinstance(result, RuntimeError)]
    assert len(losers) == 9
    assert all(str(error) == "process_lease_claim_unavailable" for error in losers)
    assert len(registrar.tasks) == 10
    assert all(task.done() for task in registrar.tasks)
    assert len(writer.control_commits) == 1
    assert await registry.snapshot("control-a") is None
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert len(total) == 1
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_blocked_registry_lock_route_cancellation_joins_candidate_without_metric() -> None:
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    registrar = _Registrar()
    metrics = RuntimeMetrics.in_memory()
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
        registrar=registrar,
    )
    await registry._lock.acquire()  # noqa: SLF001
    running = asyncio.create_task(factory.run(_mismatched_handshake(claim, events)))
    await registrar.started.wait()

    running.cancel("blocked-consume-cancel")
    registry._lock.release()  # noqa: SLF001
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await running

    assert cancelled.value.args == ("blocked-consume-cancel",)
    assert len(registrar.tasks) == 1
    assert registrar.tasks[0].done()
    assert _metric_points(metrics, "projetv0.voice.calls.active") == []
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.lease_state == "active"
    await registry.close_session_owner_registration()
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_terminal_call_persistence_cancellation_retries_same_authority_work() -> None:
    cancellation = asyncio.CancelledError("terminal-write-cancel")
    writer = _RealWriter(terminal_faults=[cancellation])
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((20.0, 21.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )

    await factory.run(_mismatched_handshake(claim, events))

    assert await registry.snapshot("control-a") is None
    assert len(writer.control_commits) == 2
    operations = [
        command.payload["operation"]  # type: ignore[union-attr]
        for command in writer.control_commits
    ]
    assert operations[0].operation_id == operations[1].operation_id
    assert operations[0].occurred_at == operations[1].occurred_at
    assert registry.internal_failure_code is None
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_terminal_call_persistence_failure_latches_even_after_earlier_failure() -> None:
    writer = _RealWriter(
        terminal_faults=[RuntimeError("terminal-persistence-secret")]
    )
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((30.0, 31.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )

    await factory.run(_mismatched_handshake(claim, events))

    assert await registry.snapshot("control-a") is not None
    assert len(writer.control_commits) == 1
    assert registry.internal_failure_code == "terminal_persistence_failed"
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert len(total) == 1
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_unconsumed_task9_writer_failure_retains_authority_and_capacity() -> None:
    writer = _RealWriter(
        terminal_faults=[RuntimeError("task9-terminal-persistence-secret")]
    )
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory()
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=[],
    )

    await factory.drain_call_by_id(
        claim.call_id,
        "recording_required_error",
    )

    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.lease_state == "terminal"
    entry = registry._by_control["control-a"]  # noqa: SLF001
    assert entry.terminal_state == "reserved"
    assert entry.terminal_authority is not None
    assert entry.terminal_authority.reason == "recording_required_error"
    assert await registry.live_call_count() == 1
    assert registry.internal_failure_code == "terminal_persistence_failed"
    assert [commit["state"] for commit in writer.lease_commits] == ["active"]
    assert control.hangups == []
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize("ordinary_failure", [False, True])
async def test_task9_cancellation_rethrows_after_success_or_latched_failure(
    ordinary_failure: bool,
) -> None:
    writer = _Task9SequencedWriter(
        (
            [None, RuntimeError("task9-retry-secret")]
            if ordinary_failure
            else [None, None, None]
        ),
        blocked_attempts=(
            frozenset({0}) if ordinary_failure else frozenset({0, 1, 2})
        ),
    )
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory()
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=[],
    )
    draining = asyncio.create_task(
        factory.drain_call_by_id(
            claim.call_id,
            "recording_required_error",
        )
    )
    await writer.started[0].wait()
    draining.cancel("task9-first-cancel")
    await writer.started[1].wait()
    if not ordinary_failure:
        draining.cancel("task9-second-cancel")
        await writer.started[2].wait()
        writer.releases[2].set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await draining

    assert cancelled.value.args == ("task9-first-cancel",)
    operations = [
        command.payload["operation"]  # type: ignore[union-attr]
        for command in writer.control_commits
    ]
    assert len({operation.operation_id for operation in operations}) == 1
    assert len({operation.occurred_at for operation in operations}) == 1
    assert all(operation.payload == operations[0].payload for operation in operations)
    assert control.hangups == []
    if ordinary_failure:
        assert registry.internal_failure_code == "terminal_persistence_failed"
        assert await registry.snapshot("control-a") is not None
        assert [commit["state"] for commit in writer.lease_commits] == ["active"]
    else:
        assert await registry.snapshot("control-a") is None
        assert [commit["state"] for commit in writer.lease_commits] == [
            "active",
            "terminal",
        ]
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_repeated_route_cancellation_preserves_first_and_waits_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.session_factory import _CallLifecycleOwner

    terminal_started = asyncio.Event()
    terminal_release = asyncio.Event()
    writer = _RealWriter(
        terminal_started=terminal_started,
        terminal_release=terminal_release,
    )
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((35.0, 36.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )
    drain_seen = asyncio.Event()
    request_drain = _CallLifecycleOwner.request_drain

    def observe_drain(owner: object, cause: str) -> None:
        request_drain(owner, cause)  # type: ignore[arg-type]
        drain_seen.set()

    monkeypatch.setattr(_CallLifecycleOwner, "request_drain", observe_drain)
    running = asyncio.create_task(factory.run(_mismatched_handshake(claim, events)))
    await terminal_started.wait()

    running.cancel("first-route-cancel")
    await drain_seen.wait()
    running.cancel("second-route-cancel")
    terminal_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await running

    assert cancelled.value.args == ("first-route-cancel",)
    assert await registry.snapshot("control-a") is None
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_process_close_cancels_constructing_owner_as_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((40.0, 41.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )
    construction_entered = asyncio.Event()
    never = asyncio.Event()

    async def block_first_construction_check(*_args: object) -> bool:
        construction_entered.set()
        await never.wait()
        return True

    monkeypatch.setattr(
        registry,
        "construction_is_live",
        block_first_construction_check,
    )
    running = asyncio.create_task(factory.run(_handshake(claim)))
    await construction_entered.wait()

    await registry.close_session_owner_registration()
    await running

    assert await registry.snapshot("control-a") is None
    assert len(writer.control_commits) == 1
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.status == "closed"
    assert operation.payload.end_reason == "process_draining"
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert total[0].attributes["session"] == "drained"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_required_drain_cancels_constructing_owner_without_local_hangup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((45.0, 46.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )
    construction_entered = asyncio.Event()
    never = asyncio.Event()

    async def block_first_construction_check(*_args: object) -> bool:
        construction_entered.set()
        await never.wait()
        return True

    monkeypatch.setattr(
        registry,
        "construction_is_live",
        block_first_construction_check,
    )
    running = asyncio.create_task(factory.run(_handshake(claim)))
    await construction_entered.wait()

    await factory.drain_call_by_id(
        claim.call_id,
        "recording_required_error",
    )
    await running

    assert await registry.snapshot("control-a") is None
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.status == "failed"
    assert operation.payload.end_reason == "recording_required_error"
    assert control.hangups == []
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_native_session_timeout_uses_one_owned_hangup_and_releases_lease() -> None:
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
    from starlette.websockets import WebSocket, WebSocketState

    from projetv0_voice.telnyx.serializer import AudioAdmission, ProjetV0TelnyxFrameSerializer

    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    hangup_command_id = registry._by_control["control-a"].hangup_command_id  # noqa: SLF001
    events: list[str] = []
    metrics = RuntimeMetrics.in_memory()
    factory = _real_process_factory(
        registry=registry, writer=writer, metrics=metrics, events=events,
        pass_frames=True,
    )
    never = asyncio.Event()

    async def receive() -> dict[str, object]:
        await never.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(_message: dict[str, object]) -> None:
        return None

    websocket = WebSocket(
        {"type": "websocket", "asgi": {"version": "3.0"}, "scheme": "wss",
         "server": ("voice.invalid", 443), "client": ("127.0.0.1", 12345),
         "root_path": "", "path": "/media", "raw_path": b"/media",
         "query_string": b"", "headers": [], "subprotocols": []},
        receive=receive, send=send,
    )
    websocket.application_state = WebSocketState.CONNECTED
    websocket.client_state = WebSocketState.CONNECTED
    admission = AudioAdmission()
    transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True, session_timeout=1,
        serializer=ProjetV0TelnyxFrameSerializer(
            "stream-a", expected_call_control_id="control-a", audio_admission=admission,
        ),
    ))
    handshake = AuthenticatedTelnyxHandshake(
        call_data=TelnyxCallData(stream_id="stream-a", call_id="control-a",
                                outbound_encoding="PCMU"),
        token_locator_id="telnyx-header-connected-v1", lease_claim=claim,
        transport=transport, audio_admission=admission,
    )
    try:
        await asyncio.wait_for(factory.run(handshake), timeout=3)

        terminal = [command.payload["operation"] for command in writer.control_commits
                    if command.payload["operation"].kind == "call.upsert"
                    and command.payload["operation"].payload.status == "failed"]
        assert len(terminal) == 1
        assert terminal[0].payload.end_reason == "transport_session_timeout"
        assert control.hangups == [("control-a", hangup_command_id)]
        assert await registry.snapshot("control-a") is None
        assert [commit["state"] for commit in writer.lease_commits] == ["active", "terminal"]
        assert admission.allows_audio() is False
        assert "recording-cleanup" in events
        assert "stt-client-close" in events and "llm-client-close" in events
    finally:
        await metrics.aclose()


@pytest.mark.asyncio
async def test_process_close_drains_real_running_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((50.0, 51.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )
    activated = asyncio.Event()
    activate_session = registry.activate_session

    async def observe_activation(*args: object) -> bool:
        result = await activate_session(*args)  # type: ignore[arg-type]
        if result:
            activated.set()
        return result

    monkeypatch.setattr(registry, "activate_session", observe_activation)
    running = asyncio.create_task(
        factory.run(_exact_real_handshake(claim, events))
    )
    await activated.wait()

    await registry.close_session_owner_registration()
    await running

    assert await registry.snapshot("control-a") is None
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.status == "closed"
    assert operation.payload.end_reason == "process_draining"
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert total[0].attributes["session"] == "drained"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_required_drain_real_running_session_leaves_hangup_to_task9(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _RealWriter()
    control = _RealControl()
    registry, claim = await _real_claim(writer, control)
    metrics = RuntimeMetrics.in_memory(monotonic=iter((55.0, 56.0)).__next__)
    events: list[str] = []
    factory = _real_process_factory(
        registry=registry,
        writer=writer,
        metrics=metrics,
        events=events,
    )
    activated = asyncio.Event()
    activate_session = registry.activate_session

    async def observe_activation(*args: object) -> bool:
        result = await activate_session(*args)  # type: ignore[arg-type]
        if result:
            activated.set()
        return result

    monkeypatch.setattr(registry, "activate_session", observe_activation)
    running = asyncio.create_task(
        factory.run(_exact_real_handshake(claim, events))
    )
    await activated.wait()

    await factory.drain_call_by_id(
        claim.call_id,
        "recording_required_error",
    )
    await running

    assert await registry.snapshot("control-a") is None
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.status == "failed"
    assert operation.payload.end_reason == "recording_required_error"
    assert control.hangups == []
    total = _metric_points(metrics, "projetv0.voice.calls.total")
    assert total[0].attributes["session"] == "failed"
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_ten_real_calls_own_distinct_resources_runtimes_and_four_observers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_module = import_module("projetv0_voice.session")
    writer = _RealWriter()
    control = _RealControl()
    registry, claims = await _real_claims(writer, control, 10)
    samples = iter(float(index) for index in range(100, 120))
    metrics = RuntimeMetrics.in_memory(monotonic=samples.__next__)
    events: list[str] = []
    stt_clients: list[_SttClient] = []
    stt_services: list[_Processor] = []
    llm_services: list[_Llm] = []
    tts_services: list[_Processor] = []
    recordings: list[_Recording] = []
    sessions: list[object] = []
    runtimes: list[object] = []
    runtimes_ready = asyncio.Event()
    build_runtime = session_module.build_runtime

    def observe_runtime(**values: object) -> object:
        runtime = build_runtime(**values)
        runtimes.append(runtime)
        if len(runtimes) == 10:
            runtimes_ready.set()
        return runtime

    monkeypatch.setattr(session_module, "build_runtime", observe_runtime)

    def stt_client_factory() -> _SttClient:
        client = _SttClient(events)
        stt_clients.append(client)
        return client

    def stt_factory(_client: _SttClient) -> _Processor:
        service = _Processor("stt", events)
        stt_services.append(service)
        return service

    def llm_factory() -> _Llm:
        service = _Llm(events)
        llm_services.append(service)
        return service

    def tts_factory() -> _Processor:
        service = _Processor("tts", events)
        tts_services.append(service)
        return service

    def recording_factory(_identity: object) -> _Recording:
        recording = _Recording(events)
        recordings.append(recording)
        return recording

    def session_factory(**values: object) -> object:
        session = session_module.CallSession(**values)
        sessions.append(session)
        return session

    factory = ProcessSessionFactory(
        registry=registry,
        registrar=_Registrar(),
        runtime_metrics=metrics,
        manifest=_real_manifest(),
        profile=_real_profile(),
        writer=writer,
        keyring=CryptoKeyring(
            {1: b"k" * 32},
            active_version=1,
            nonce_factory=lambda size: b"n" * size,
        ),
        stt_http_client_factory=stt_client_factory,
        stt_factory=stt_factory,
        llm_factory=llm_factory,
        tts_factory=tts_factory,
        recording_factory=recording_factory,
        session_factory=session_factory,  # type: ignore[arg-type]
        idle_timeout_seconds=30.0,
    )
    running = [
        asyncio.create_task(factory.run(_claim_handshake(claim, events)))
        for claim in claims
    ]
    await runtimes_ready.wait()

    await registry.close_session_owner_registration()
    await asyncio.gather(*running)

    collections = (
        stt_clients,
        stt_services,
        llm_services,
        tts_services,
        recordings,
        sessions,
        runtimes,
    )
    assert all(len(collection) == 10 for collection in collections)
    assert all(len({id(item) for item in collection}) == 10 for collection in collections)
    assert len({id(runtime.pipeline) for runtime in runtimes}) == 10
    assert len({id(runtime.worker) for runtime in runtimes}) == 10
    assert len({id(runtime.runner) for runtime in runtimes}) == 10
    inventories = [runtime.worker._observer._observers for runtime in runtimes]  # noqa: SLF001
    assert all(len(inventory) == 4 for inventory in inventories)
    assert len({id(observer) for inventory in inventories for observer in inventory}) == 40
    assert len(writer.control_commits) == 10
    assert len(control.hangups) == 10
    assert await registry.live_call_count() == 0
    assert events.count("stt-client-close") == 10
    assert events.count("llm-client-close") == 10
    assert events.count("tts-cleanup") == 10
    totals = _metric_points(metrics, "projetv0.voice.calls.total")
    assert sum(point.value for point in totals) == 10
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001
