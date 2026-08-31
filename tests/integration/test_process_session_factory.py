from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import TelnyxCallData

from projetv0_voice.admission import (
    CallConstructionGrant,
    CallGenerationHandle,
    ProcessLeaseClaim,
)
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake

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


class _Registrar:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.tasks: list[asyncio.Task[None]] = []

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
        return task


class _Registry:
    def __init__(self, grant: CallConstructionGrant) -> None:
        self.grant = grant
        self.consume_calls: list[tuple[object, str, object, asyncio.Task[None]]] = []
        self.live_checks = 0
        self.activated = asyncio.Event()
        self.required_drains: list[UUID] = []
        self.activation_error: BaseException | None = None
        self.persist_call = False
        self.terminal_failures: list[str] = []

    async def consume_claim_for_construction(
        self,
        claim: ProcessLeaseClaim,
        stream_id: str,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> CallConstructionGrant | None:
        self.consume_calls.append((claim, stream_id, owner, owner_task))
        assert owner_task.done() is False
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
        owner: object,
        owner_task: asyncio.Task[None],
        proposed: object,
    ) -> Any:
        del grant, owner, owner_task
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


class _Recording:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self, _identity: object) -> object:
        raise AssertionError("not started by factory")

    async def cleanup(self, *_args: object, **_kwargs: object) -> None:
        self.events.append("recording-cleanup")


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
