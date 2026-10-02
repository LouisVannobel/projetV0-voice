"""Production Voice Cell process ownership and lifecycle supervision."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import math
import os
import threading
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, NoReturn, Protocol, cast
from uuid import UUID

from pipecat.processors.frame_processor import FrameProcessor
from pydantic import SecretStr

from projetv0_voice.admission import (
    CallRegistry,
    ProcessLeaseAuthority,
    SynchronousUnauthenticatedGate,
    select_call_capacity,
)
from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import (
    RuntimeMetrics,
    RuntimePublication,
    RuntimePublishedSnapshot,
)
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.relay import OutboxRelay, maintain_call_content
from projetv0_voice.persistence.writer import (
    PersistenceWriter,
    QualificationRunConsumed,
    StaleLease,
    WebhookCommitResult,
    WebhookCommitValue,
    WriterRuntimeObservation,
)
from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    QualificationOverrideV1,
    QualifiedDeploymentProfileV1,
    RuntimeDeploymentProfileV1,
    canonical_inference_profile_sha256,
    canonical_qualified_profile_sha256,
)
from projetv0_voice.runtime_config import RuntimeSettingsV1
from projetv0_voice.session import (
    CallIdentity,
    PublicSttHttpClient,
    RecordingBoundary,
)
from projetv0_voice.session_factory import ProcessSessionFactory
from projetv0_voice.telnyx.call_control import CallControlResult, StreamingStartV1
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshakeService
from projetv0_voice.telnyx.recordings import (
    PurgeBatchResult,
    RecordingPurgeError,
    TelnyxRecordingBoundary,
    after_recording_webhook_commit,
    purge_recordings_once,
    recording_metric_transition,
    resolve_recording_webhook,
)
from projetv0_voice.telnyx.webhooks import (
    ResolvedWebhook,
    TelnyxWebhookProcessor,
    TelnyxWebhookVerifier,
    VerifiedWebhook,
    WebhookDisposition,
    WebhookDurableEffect,
)

_MAX_STORAGE_BYTES = 268_435_456
_MAX_WRITER_QUEUE_AGE_SECONDS = 1.0
_MAX_OUTBOX_AGE_SECONDS = 900.0
_RUNTIME_HARD_EXIT_CODE = 72

NON_INGRESS_METRIC_OWNERS = MappingProxyType(
    {
        "actions.total": "call_control",
        "relay.runs": "relay_supervisor",
        "writer.queue_depth": "writer_snapshot",
        "writer.queue_oldest_age": "writer_snapshot",
        "writer.quick_check": "writer_snapshot",
        "outbox.depth": "writer_snapshot",
        "outbox.oldest_age": "writer_snapshot",
        "outbox.bytes": "writer_snapshot",
        "recordings.total": "recording_transitions",
        "runtime.event_loop_lag": "lag_supervisor",
        "ready": "runtime_publication",
    }
)

type WebhookMetricClass = Literal[
    "initiated", "answered", "terminal", "recording", "unsupported", "invalid"
]
type WebhookMetricReceipt = Literal["none", "first", "duplicate"]
type WebhookMetricDisposition = Literal[
    "ok", "bad_request", "forbidden", "too_large", "unavailable", "internal_error"
]
type AdmissionMetricRejection = Literal[
    "capacity", "draining", "qualification", "persistence", "invalid"
]


@dataclass(frozen=True, slots=True, repr=False)
class IngressMetricOutcome:
    """Closed terminal values transferred with one webhook delivery owner."""

    webhook_class: WebhookMetricClass
    receipt: WebhookMetricReceipt
    disposition: WebhookMetricDisposition
    admission_rejection: AdmissionMetricRejection | None = None

    def __post_init__(self) -> None:
        if (
            self.webhook_class
            not in {"initiated", "answered", "terminal", "recording", "unsupported", "invalid"}
            or self.receipt not in {"none", "first", "duplicate"}
            or self.disposition
            not in {
                "ok",
                "bad_request",
                "forbidden",
                "too_large",
                "unavailable",
                "internal_error",
            }
            or self.admission_rejection is not None
            and self.admission_rejection
            not in {"capacity", "draining", "qualification", "persistence", "invalid"}
        ):
            raise ValueError("ingress_metric_outcome_invalid") from None


class IngressMetricObservation:
    """One-finish delivery observation for the later Task 10C-I ingress wiring."""

    __slots__ = ("_finished", "_lock", "_metrics")

    def __init__(self, metrics: RuntimeMetrics) -> None:
        if not isinstance(metrics, RuntimeMetrics):
            raise ValueError("ingress_metric_observation_invalid") from None
        self._metrics = metrics
        self._lock = threading.Lock()
        self._finished = False

    def __repr__(self) -> str:
        return "IngressMetricObservation()"

    def finish(self, outcome: IngressMetricOutcome) -> None:
        if type(outcome) is not IngressMetricOutcome:
            return
        with self._lock:
            if self._finished:
                return
            self._finished = True
        if outcome.admission_rejection is not None:
            self._metrics.record_admission_rejection(outcome.admission_rejection)
        self._metrics.record_webhook(
            outcome.webhook_class,
            outcome.receipt,
            outcome.disposition,
        )


class _CallControl(Protocol):
    async def answer(
        self, call_control_id: str, *, command_id: UUID
    ) -> CallControlResult: ...

    async def start_streaming(
        self,
        call_control_id: str,
        request: StreamingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult: ...

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult: ...


class _ProcessCallControl(_CallControl, Protocol):
    async def aclose(self) -> None: ...


class _RuntimeRegistry(Protocol):
    @property
    def candidate_run_id(self) -> UUID | None: ...

    @property
    def internal_failure_code(self) -> str | None: ...

    @property
    def internal_failure_event(self) -> asyncio.Event: ...

    async def begin_drain(self) -> None: ...

    async def reap_expired(self) -> int: ...

    async def qualification_state_valid(self) -> bool: ...

    async def qualification_state(
        self,
    ) -> Literal["valid", "consumed", "expired"]: ...

    async def close_session_owner_registration(self) -> None: ...

    def close_registration(self) -> None: ...

    async def join_until_empty(self) -> None: ...

    async def reconcile_after_commit(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        result: WebhookCommitValue,
    ) -> WebhookDisposition: ...

    async def confirm_late_after_fail_closed(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        result: WebhookCommitValue,
    ) -> object: ...


class _RuntimeSink(Protocol):
    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def begin_call(self, deployment_id: str, call_id: UUID, routing: Any) -> Any: ...


class _CloseableGate(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeProfileSelection:
    """One validated runtime profile plus an optional qualification override."""

    profile: RuntimeDeploymentProfileV1
    override: QualificationOverrideV1 | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.profile,
            QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1,
        ) or self.override is not None and not isinstance(
            self.override, QualificationOverrideV1
        ):
            raise ValueError("runtime_profile_selection_invalid") from None


@dataclass(frozen=True, slots=True)
class RuntimeInferenceFactories:
    """Typed external inference constructors retained by the session factory."""

    stt_http_client_factory: Callable[[], PublicSttHttpClient]
    stt_factory: Callable[[PublicSttHttpClient], FrameProcessor]
    llm_factory: Callable[[], FrameProcessor]
    tts_factory: Callable[[], FrameProcessor]

    def __post_init__(self) -> None:
        if not all(
            callable(value)
            for value in (
                self.stt_http_client_factory,
                self.stt_factory,
                self.llm_factory,
                self.tts_factory,
            )
        ):
            raise ValueError("runtime_inference_factories_invalid") from None


@dataclass(frozen=True, slots=True)
class RuntimeProductionFactories:
    """Typed filesystem and external-client seams used by production composition."""

    validate_artifacts: Callable[[RuntimeSettingsV1], None]
    load_manifest: Callable[[RuntimeSettingsV1], AgentManifestV1]
    load_profile: Callable[
        [RuntimeSettingsV1, AgentManifestV1, datetime], RuntimeProfileSelection
    ]
    read_secret: Callable[[PurePosixPath], SecretStr]
    load_keyring: Callable[[RuntimeSettingsV1], CryptoKeyring]
    sink_factory: Callable[[SecretStr], _RuntimeSink]
    call_control_factory: Callable[[SecretStr], _ProcessCallControl]
    inference_factory: Callable[
        [SecretStr, RuntimeDeploymentProfileV1, str], RuntimeInferenceFactories
    ]

    def __post_init__(self) -> None:
        if not all(
            callable(value)
            for value in (
                self.validate_artifacts,
                self.load_manifest,
                self.load_profile,
                self.read_secret,
                self.load_keyring,
                self.sink_factory,
                self.call_control_factory,
                self.inference_factory,
            )
        ):
            raise ValueError("runtime_production_factories_invalid") from None


class _MeasuredCallControl:
    """Single semantic metric owner around the process Call Control facade."""

    __slots__ = ("_control", "_metrics")

    def __init__(self, control: _CallControl, metrics: RuntimeMetrics) -> None:
        self._control = control
        self._metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._control, name)

    async def answer(
        self, call_control_id: str, *, command_id: UUID
    ) -> CallControlResult:
        return await self._invoke(
            "answer",
            self._control.answer(call_control_id, command_id=command_id),
        )

    async def start_streaming(
        self,
        call_control_id: str,
        request: StreamingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        return await self._invoke(
            "streaming_start",
            self._control.start_streaming(
                call_control_id,
                request,
                command_id=command_id,
            ),
        )

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult:
        return await self._invoke(
            "hangup",
            self._control.hangup(
                call_control_id,
                command_id=command_id,
                client_state=client_state,
            ),
        )

    async def _invoke(
        self,
        action: Literal["answer", "streaming_start", "hangup"],
        operation: Coroutine[Any, Any, CallControlResult],
    ) -> CallControlResult:
        try:
            result = await operation
        except BaseException:
            self._metrics.record_action(action, "internal_error")
            raise
        if not isinstance(result, CallControlResult):
            self._metrics.record_action(action, "internal_error")
            raise RuntimeError("call_control_result_invalid") from None
        self._metrics.record_action(action, result.outcome)
        return result


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeProductionGraph:
    """Fully composed Task 10C-L graph consumed later by the FastAPI app."""

    settings: RuntimeSettingsV1
    manifest: AgentManifestV1
    profile: RuntimeDeploymentProfileV1
    override: QualificationOverrideV1 | None
    keyring: CryptoKeyring
    supervisor: RuntimeSupervisor
    metrics: RuntimeMetrics
    writer: PersistenceWriter
    sink: _RuntimeSink
    relay: OutboxRelay
    raw_call_control: _ProcessCallControl
    measured_call_control: _CallControl
    registry: CallRegistry
    lease_authority: ProcessLeaseAuthority
    unauthenticated_gate: SynchronousUnauthenticatedGate
    handshake: AuthenticatedTelnyxHandshakeService
    session_factory: ProcessSessionFactory
    webhook_processor: TelnyxWebhookProcessor
    recording_factory: Callable[[CallIdentity], RecordingBoundary]
    recording_call_control_identity: object

    def __repr__(self) -> str:
        return "RuntimeProductionGraph()"


def publish_runtime_readiness(
    publication: RuntimePublication,
    *,
    writer: WriterRuntimeObservation,
    startup_profile_and_stale_recovery_complete: bool,
    draining: bool,
    admission_open: bool,
    qualification_state_valid: bool,
    writer_owner_alive_and_ready: bool,
    no_writer_fatal_or_degradation: bool,
    relay_supervisor_alive: bool,
    no_permanent_relay_or_sink_degradation: bool,
) -> RuntimePublishedSnapshot:
    """Validate and atomically replace one complete readiness publication."""

    predicates = (
        startup_profile_and_stale_recovery_complete,
        draining,
        admission_open,
        qualification_state_valid,
        writer_owner_alive_and_ready,
        no_writer_fatal_or_degradation,
        relay_supervisor_alive,
        no_permanent_relay_or_sink_degradation,
    )
    if (
        type(publication) is not RuntimePublication
        or type(writer) is not WriterRuntimeObservation
        or any(type(value) is not bool for value in predicates)
    ):
        raise RuntimeError("runtime_readiness_invalid") from None
    ready = (
        startup_profile_and_stale_recovery_complete
        and not draining
        and admission_open
        and qualification_state_valid
        and writer_owner_alive_and_ready
        and writer.writer_quick_check
        and no_writer_fatal_or_degradation
        and writer.storage_bytes <= _MAX_STORAGE_BYTES
        and writer.writer_queue_oldest_age <= _MAX_WRITER_QUEUE_AGE_SECONDS
        and relay_supervisor_alive
        and no_permanent_relay_or_sink_degradation
        and writer.outbox_oldest_age <= _MAX_OUTBOX_AGE_SECONDS
    )
    current = publication.snapshot()
    snapshot = RuntimePublishedSnapshot(
        generation=current.generation + 1,
        ready=ready,
        draining=draining,
        writer_queue_depth=writer.writer_queue_depth,
        writer_queue_oldest_age=writer.writer_queue_oldest_age,
        writer_quick_check=writer.writer_quick_check,
        outbox_depth=writer.outbox_depth,
        outbox_oldest_age=writer.outbox_oldest_age,
        outbox_bytes=writer.outbox_bytes,
        storage_bytes=writer.storage_bytes,
        startup_profile_and_stale_recovery_complete=(
            startup_profile_and_stale_recovery_complete
        ),
        admission_open=admission_open,
        qualification_state_valid=qualification_state_valid,
        writer_owner_alive_and_ready=writer_owner_alive_and_ready,
        no_writer_fatal_or_degradation=no_writer_fatal_or_degradation,
        relay_supervisor_alive=relay_supervisor_alive,
        no_permanent_relay_or_sink_degradation=(
            no_permanent_relay_or_sink_degradation
        ),
    )
    publication.publish(snapshot)
    return snapshot


class _OwnedTaskSet:
    """One synchronous closed registration gate for same-loop process tasks."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._closed = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._terminal_failure = False

    def __repr__(self) -> str:
        return "OwnedTaskSet()"

    def try_start(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> asyncio.Task[None] | None:
        if asyncio.get_running_loop() is not self._loop:
            coroutine.close()
            return None
        start_gate = asyncio.Event()

        async def owned() -> None:
            await start_gate.wait()
            await coroutine

        wrapper = owned()
        try:
            with self._lock:
                if self._closed:
                    wrapper.close()
                    coroutine.close()
                    return None
                task = asyncio.create_task(wrapper, name=name)
                self._tasks.add(task)
        except BaseException:
            wrapper.close()
            coroutine.close()
            return None

        def completed(done: asyncio.Task[None]) -> None:
            failed = False
            if not done.cancelled():
                try:
                    failed = done.exception() is not None
                except BaseException:
                    failed = True
            with self._lock:
                if failed:
                    self._terminal_failure = True
                self._tasks.discard(done)

        task.add_done_callback(completed)
        start_gate.set()
        return task

    def spawn_required(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> asyncio.Task[None]:
        task = self.try_start(coroutine, name=name)
        if task is None:
            raise RuntimeError("owned_task_registration_failed") from None
        return task

    def close_registration(self) -> None:
        with self._lock:
            self._closed = True

    async def join_until_empty(self, deadline: float) -> None:
        if not isinstance(deadline, int | float) or not math.isfinite(deadline):
            raise RuntimeError("owned_task_deadline_invalid") from None
        while True:
            with self._lock:
                tasks = tuple(self._tasks)
            if not tasks:
                break
            remaining = float(deadline) - self._loop.time()
            if remaining <= 0:
                raise RuntimeError("owned_task_deadline_exceeded") from None
            done, pending = await asyncio.wait(tasks, timeout=remaining)
            for task in done:
                if not task.cancelled():
                    with contextlib.suppress(BaseException):
                        task.exception()
            if pending:
                raise RuntimeError("owned_task_deadline_exceeded") from None
        with self._lock:
            failed = self._terminal_failure
        if failed:
            raise RuntimeError("owned_task_failed") from None


class _WebhookFinalizationHandle:
    __slots__ = ("_completion",)

    def __init__(self, completion: asyncio.Future[WebhookDisposition]) -> None:
        self._completion = completion

    def __repr__(self) -> str:
        return "WebhookFinalizationHandle()"

    async def wait(self) -> WebhookDisposition:
        return await asyncio.shield(self._completion)


class RuntimeSupervisor:
    """Own production startup, fixed supervisors, publication, and shutdown."""

    def __init__(
        self,
        *,
        writer: PersistenceWriter | None = None,
        call_control: _ProcessCallControl | None = None,
        metrics: RuntimeMetrics | None = None,
        publication: RuntimePublication | None = None,
        relay: OutboxRelay | None = None,
        registry: _RuntimeRegistry | None = None,
        sink: _RuntimeSink | None = None,
        unauthenticated_gate: _CloseableGate | None = None,
        purge_once: Callable[[], Awaitable[PurgeBatchResult]] | None = None,
        recording_after_commit: Callable[
            [VerifiedWebhook, WebhookDurableEffect | None],
            Awaitable[WebhookDisposition | None],
        ]
        | None = None,
        candidate_run_id: UUID | None = None,
        deployment_id: str = "projetv0-voice",
        retention_days: int = 7,
        sparra_enabled: bool = False,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        loop_interval_seconds: float = 0.25,
        startup_phase_timeout_seconds: float = 5.0,
        startup_phase_timeouts: Mapping[str, float] | None = None,
        shutdown_timeout_seconds: float = 30.0,
        hard_exit: Callable[[int], NoReturn] = os._exit,
    ) -> None:
        self.webhook_finalizers = _OwnedTaskSet()
        self.call_lifecycle_owners = _OwnedTaskSet()
        self.fixed_supervisors = _OwnedTaskSet()
        if (
            type(retention_days) is not int
            or retention_days <= 0
            or isinstance(loop_interval_seconds, bool)
            or not isinstance(loop_interval_seconds, int | float)
            or not math.isfinite(float(loop_interval_seconds))
            or loop_interval_seconds <= 0
            or isinstance(shutdown_timeout_seconds, bool)
            or not isinstance(shutdown_timeout_seconds, int | float)
            or not math.isfinite(float(shutdown_timeout_seconds))
            or shutdown_timeout_seconds <= 0
            or isinstance(startup_phase_timeout_seconds, bool)
            or not isinstance(startup_phase_timeout_seconds, int | float)
            or not math.isfinite(float(startup_phase_timeout_seconds))
            or startup_phase_timeout_seconds <= 0
            or not isinstance(deployment_id, str)
            or not deployment_id
            or not callable(hard_exit)
        ):
            raise ValueError("runtime_supervisor_config_invalid") from None
        selected_phase_timeouts = dict(startup_phase_timeouts or {})
        if any(
            key
            not in {
                "writer_startup_failed",
                "writer_quick_check_failed",
                "qualification_status_failed",
                "operation_sink_open_failed",
                "stale_recovery_failed",
                "runtime_publication_failed",
                "runtime_begin_drain_failed",
            }
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or value <= 0
            for key, value in selected_phase_timeouts.items()
        ):
            raise ValueError("runtime_supervisor_config_invalid") from None
        self._writer = writer
        self._call_control = call_control
        self._metrics = metrics
        self._measured_call_control = (
            None
            if call_control is None or metrics is None
            else _MeasuredCallControl(call_control, metrics)
        )
        self._publication = publication or (
            metrics.publication if metrics is not None else _new_publication()
        )
        self._relay = relay
        self._registry = registry
        self._sink = sink
        self._unauthenticated_gate = unauthenticated_gate
        self._purge_once = purge_once
        self._recording_after_commit = recording_after_commit
        self._candidate_run_id = candidate_run_id
        self._deployment_id = deployment_id
        self._retention_days = retention_days
        self._sparra_enabled = sparra_enabled
        self._utcnow = utcnow
        self._loop_interval_seconds = float(loop_interval_seconds)
        self._startup_phase_timeout_seconds = float(startup_phase_timeout_seconds)
        self._startup_phase_timeouts = MappingProxyType(selected_phase_timeouts)
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._hard_exit = hard_exit
        self._writer_task: asyncio.Task[None] | None = None
        self._startup_phase_tasks: set[asyncio.Task[Any]] = set()
        self._startup_phase_cancel_requested: set[asyncio.Task[Any]] = set()
        self._startup_cleanup_task: asyncio.Task[None] | None = None
        self._startup_phase_timed_out = False
        self._fixed_tasks: dict[str, asyncio.Task[None]] = {}
        self._state_stop = asyncio.Event()
        self._relay_stop = asyncio.Event()
        self._drain_event = asyncio.Event()
        self._started = False
        self._startup_complete = False
        self._draining = False
        self._admission_open = False
        self._qualification_valid = False
        self._relay_healthy = relay is not None and sink is not None
        self._last_writer_observation = WriterRuntimeObservation(
            writer_queue_depth=0,
            writer_queue_oldest_age=0.0,
            writer_quick_check=False,
            outbox_depth=0,
            outbox_oldest_age=0.0,
            outbox_bytes=0,
            storage_bytes=0,
        )
        self._close_task: asyncio.Task[None] | None = None
        self._deadline_tasks: set[asyncio.Future[object]] = set()
        self._shutdown_failure_code: str | None = None
        self._closed = False

    async def startup(self) -> None:
        if self._started or self._closed or self._writer is None:
            raise RuntimeError("runtime_startup_invalid") from None
        self._started = True
        failure_code: str | None = None
        cancellation: asyncio.CancelledError | None = None
        try:
            self._writer_task = asyncio.create_task(
                self._writer.run(), name="voice-persistence-writer"
            )
            if not await self._startup_await(
                self._writer.wait_ready(), code="writer_startup_failed"
            ):
                raise RuntimeError("writer_startup_failed")
            if not await self._startup_await(
                self._writer.quick_check(), code="writer_quick_check_failed"
            ):
                raise RuntimeError("writer_quick_check_failed")
            if self._candidate_run_id is not None and await self._startup_await(
                self._writer.qualification_run_consumed(self._candidate_run_id),
                code="qualification_status_failed",
            ):
                raise RuntimeError("qualification_run_consumed")
            self._qualification_valid = True
            if self._sparra_enabled:
                await self._startup_await(
                    self._writer.assert_sparra_compatible(), code="writer_startup_failed"
                )
            if self._sink is not None:
                await self._startup_await(self._sink.open(), code="operation_sink_open_failed")
            if self._sparra_enabled:
                if self._relay is None:
                    raise RuntimeError("stale_recovery_failed")
                await self._startup_await(
                    self._relay.prepare_before_fifo(), code="stale_recovery_failed"
                )
            await self._recover_stale_leases()
            self._register_fixed_supervisors()
            self.fixed_supervisors.close_registration()
            self._startup_complete = True
            self._admission_open = True
            self._last_writer_observation = await self._startup_await(
                self._writer.runtime_observation(),
                code="runtime_publication_failed",
            )
            self._publish_current()
            if self._writer_health_breached():
                await self._startup_await(self.begin_drain(), code="runtime_begin_drain_failed")
        except asyncio.CancelledError as error:
            cancellation = error
        except BaseException as error:
            candidate = str(error)
            failure_code = (
                candidate
                if candidate
                in {
                    "stale_recovery_failed",
                    "qualification_run_consumed",
                    "writer_startup_failed",
                    "writer_quick_check_failed",
                    "owned_task_registration_failed",
                }
                else "runtime_startup_failed"
            )
        if cancellation is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._finish_startup_unwind()
            raise cancellation
        if failure_code is not None:
            if self._startup_phase_timed_out:
                self._ensure_startup_cleanup()
            else:
                await self._finish_startup_unwind()
            raise RuntimeError(failure_code) from None

    async def _startup_await[ResultT](
        self,
        awaitable: Awaitable[ResultT],
        *,
        code: str,
    ) -> ResultT:
        start_gate = asyncio.Event()

        async def run_phase() -> ResultT:
            await start_gate.wait()
            return await awaitable

        runner = run_phase()
        create_failed = False
        try:
            task = asyncio.create_task(runner, name=f"voice-startup-{code}")
        except BaseException:
            create_failed = True
        if create_failed:
            runner.close()
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise RuntimeError(code) from None

        self._startup_phase_tasks.add(task)

        def finished(done: asyncio.Task[ResultT]) -> None:
            self._startup_phase_tasks.discard(done)
            self._startup_phase_cancel_requested.discard(done)
            if not done.cancelled():
                with contextlib.suppress(BaseException):
                    done.exception()

        task.add_done_callback(finished)
        start_gate.set()
        timeout_seconds = self._startup_phase_timeouts.get(
            code, self._startup_phase_timeout_seconds
        )
        parent_cancellation: asyncio.CancelledError | None = None
        done: set[asyncio.Task[ResultT]] = set()
        try:
            completed, _pending = await asyncio.wait(
                (task,), timeout=timeout_seconds
            )
            done.update(completed)
        except asyncio.CancelledError as error:
            parent_cancellation = error
        if parent_cancellation is not None:
            self._cancel_startup_phase_once(task)
            raise parent_cancellation
        if task not in done:
            self._startup_phase_timed_out = True
            self._startup_complete = False
            self._admission_open = False
            self._qualification_valid = False
            self.webhook_finalizers.close_registration()
            self.fixed_supervisors.close_registration()
            if self._unauthenticated_gate is not None:
                self._unauthenticated_gate.close()
            self._cancel_startup_phase_once(task)
            raise RuntimeError(code) from None
        phase_failed = False
        result: ResultT | None = None
        try:
            result = task.result()
        except asyncio.CancelledError:
            raise
        except BaseException:
            phase_failed = True
        if phase_failed:
            raise RuntimeError(code) from None
        return cast(ResultT, result)

    def _cancel_startup_phase_once(self, task: asyncio.Task[Any]) -> None:
        if task.done() or task in self._startup_phase_cancel_requested:
            return
        self._startup_phase_cancel_requested.add(task)
        task.cancel()

    @property
    def call_control_facade(self) -> _CallControl:
        """Return the one measured facade for registry and recording composition."""

        if self._measured_call_control is None:
            raise RuntimeError("call_control_facade_unavailable") from None
        return self._measured_call_control

    def bind_runtime_graph(
        self,
        *,
        relay: OutboxRelay,
        registry: _RuntimeRegistry,
        sink: _RuntimeSink,
        unauthenticated_gate: _CloseableGate,
        purge_once: Callable[[], Awaitable[PurgeBatchResult]],
        recording_after_commit: Callable[
            [VerifiedWebhook, WebhookDurableEffect | None],
            Awaitable[WebhookDisposition | None],
        ],
    ) -> None:
        """Bind the finite post-client runtime graph exactly once before startup."""

        if self._started or any(
            value is not None
            for value in (
                self._relay,
                self._registry,
                self._sink,
                self._unauthenticated_gate,
                self._purge_once,
                self._recording_after_commit,
            )
        ):
            raise RuntimeError("runtime_graph_already_bound") from None
        self._relay = relay
        self._registry = registry
        self._sink = sink
        self._unauthenticated_gate = unauthenticated_gate
        self._purge_once = purge_once
        self._recording_after_commit = recording_after_commit
        self._relay_healthy = True

    async def begin_drain(self) -> None:
        if self._draining:
            return
        self._draining = True
        self._admission_open = False
        current = self._publication.snapshot()
        self._publication.publish(
            replace(
                current,
                generation=current.generation + 1,
                ready=False,
                draining=True,
                admission_open=False,
            )
        )
        self.webhook_finalizers.close_registration()
        self.fixed_supervisors.close_registration()
        if self._unauthenticated_gate is not None:
            self._unauthenticated_gate.close()
        drain_failed = False
        if self._registry is not None:
            try:
                await self._registry.begin_drain()
            except asyncio.CancelledError:
                raise
            except BaseException:
                drain_failed = True
        self._drain_event.set()
        if drain_failed:
            raise RuntimeError("runtime_begin_drain_failed") from None

    def readiness_snapshot(self) -> RuntimePublishedSnapshot:
        return self._publication.snapshot()

    def observe_qualification_state(
        self, state: Literal["valid", "consumed", "expired"]
    ) -> None:
        """Consume the registry's synchronous post-COMMIT state notification."""

        if state == "valid":
            self._qualification_valid = True
            return
        if state != "consumed":
            return
        self._qualification_valid = False
        self._admission_open = False
        self._publish_current()

    async def classify_webhook_receipt(
        self, event: VerifiedWebhook
    ) -> Literal["missing", "duplicate", "conflict"]:
        if self._writer is None or not isinstance(event, VerifiedWebhook):
            raise RuntimeError("webhook_classification_unavailable") from None
        return await self._writer.classify_webhook_receipt(
            event_id=event.event_id,
            semantic_fingerprint_sha256=event.semantic_fingerprint_sha256,
            legacy_v1_semantic_fingerprint_sha256=(
                event.legacy_v1_semantic_fingerprint_sha256
            ),
        )

    def start_webhook_finalization(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        receipt: Literal["first", "duplicate"] = "first",
    ) -> _WebhookFinalizationHandle:
        if (
            self._writer is None
            or not isinstance(event, VerifiedWebhook)
            or not isinstance(resolution, ResolvedWebhook)
            or receipt not in {"first", "duplicate"}
        ):
            raise RuntimeError("webhook_finalizer_invalid") from None
        completion: asyncio.Future[WebhookDisposition] = (
            asyncio.get_running_loop().create_future()
        )
        task = self.webhook_finalizers.try_start(
            self._finalize_webhook(event, resolution, receipt, completion),
            name="voice-webhook-finalizer",
        )
        if task is None:
            if resolution.reservation is not None:
                with contextlib.suppress(Exception):
                    resolution.reservation.abandon_before_submit()
            raise RuntimeError("webhook_finalizer_registration_closed") from None
        return _WebhookFinalizationHandle(completion)

    async def _finalize_webhook(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        receipt: Literal["first", "duplicate"],
        completion: asyncio.Future[WebhookDisposition],
    ) -> None:
        assert self._writer is not None
        try:
            effect = resolution.effect
            ticket = self._writer.submit_webhook(
                receipt={
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "call_control_id": event.call_control_id,
                    "occurred_at": event.occurred_at,
                    "received_at": _aware_utc(self._utcnow()),
                    "semantic_fingerprint_sha256": (event.semantic_fingerprint_sha256),
                    **(
                        {
                            "call_leg_id": event.call_leg_id,
                            "call_session_id": event.call_session_id,
                        }
                        if self._sparra_enabled and event.event_type == "call.hangup"
                        else {}
                    ),
                },
                lease=None if effect is None else effect.lease,
                operation=None if effect is None else effect.operation,
                **(
                    {"admission_facts": effect.admission_facts}
                    if effect is not None and effect.admission_facts is not None
                    else {}
                ),
                legacy_v1_semantic_fingerprint_sha256=(event.legacy_v1_semantic_fingerprint_sha256),
                qualification_run_id=(
                    self._candidate_run_id
                    if receipt == "first"
                    and event.event_type == "call.initiated"
                    and resolution.effect is not None
                    and resolution.effect.lease is not None
                    and resolution.effect.lease.get("state") == "pending"
                    else None
                ),
            )
        except BaseException:
            if resolution.reservation is not None:
                with contextlib.suppress(Exception):
                    resolution.reservation.abandon_before_submit()
            if not completion.done():
                completion.set_result(WebhookDisposition(503))
            return

        timed_out = False
        try:
            try:
                async with asyncio.timeout(self._writer.control_commit_timeout_seconds):
                    result = await ticket.wait()
            except TimeoutError:
                timed_out = True
                self._writer.latch_control_commit_timeout()
                if not completion.done():
                    completion.set_result(WebhookDisposition(503))
                result = await ticket.wait()
        except asyncio.CancelledError:
            raise
        except BaseException:
            if resolution.reservation is not None:
                try:
                    await resolution.reservation.settle_after_submit_failure()
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    pass
            if not completion.done():
                completion.set_result(WebhookDisposition(503))
            return

        if timed_out:
            if self._registry is not None:
                with contextlib.suppress(Exception):
                    await self._registry.confirm_late_after_fail_closed(event, resolution, result)
            return

        disposition = WebhookDisposition(
            503 if isinstance(result, QualificationRunConsumed) else 200,
            admission_rejection=(
                "qualification" if isinstance(result, QualificationRunConsumed) else None
            ),
        )
        if self._registry is not None:
            try:
                disposition = await self._registry.reconcile_after_commit(event, resolution, result)
            except BaseException:
                disposition = WebhookDisposition(500)
        if isinstance(result, WebhookCommitResult) and self._metrics is not None:
            transition = recording_metric_transition(event, resolution.effect, result)
            if transition is not None:
                self._metrics.record_recording(transition)
        if self._recording_after_commit is not None:
            try:
                recording_disposition = await self._recording_after_commit(event, resolution.effect)
            except asyncio.CancelledError:
                raise
            except BaseException:
                recording_disposition = WebhookDisposition(500)
            if recording_disposition is not None:
                disposition = recording_disposition
        if not completion.done():
            completion.set_result(disposition)

    async def aclose(self) -> None:
        if self._startup_cleanup_task is not None:
            await asyncio.shield(self._startup_cleanup_task)
            if self._closed:
                return
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._aclose_core(), name="voice-runtime-close"
            )
        await asyncio.shield(self._close_task)

    async def _aclose_core(self) -> None:
        if self._closed:
            return
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout_seconds
        terminal_owner_failed = False

        async def join_or_remember(awaitable: Awaitable[object]) -> None:
            nonlocal terminal_owner_failed
            try:
                await awaitable
            except RuntimeError as error:
                if str(error) != "owned_task_failed":
                    raise
                terminal_owner_failed = True

        await self._await_shutdown_deadline(self.begin_drain(), deadline)
        await join_or_remember(self.webhook_finalizers.join_until_empty(deadline))
        if self._registry is not None:
            await self._await_shutdown_deadline(
                self._registry.close_session_owner_registration(), deadline
            )
        self.call_lifecycle_owners.close_registration()
        await join_or_remember(
            self.call_lifecycle_owners.join_until_empty(deadline)
        )

        self._state_stop.set()
        if await self._join_state_supervisors(deadline):
            terminal_owner_failed = True
        if self._registry is not None:
            self._registry.close_registration()
            await self._await_shutdown_deadline(
                self._registry.join_until_empty(), deadline
            )

        self._relay_stop.set()
        await join_or_remember(self._join_task("relay", deadline))
        await join_or_remember(self.fixed_supervisors.join_until_empty(deadline))
        await self._close_dependencies(deadline)
        self._closed = True
        if terminal_owner_failed:
            raise RuntimeError("runtime_shutdown_failed") from None

    async def _unwind_startup(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout_seconds
        await self._await_shutdown_deadline(self.begin_drain(), deadline)
        await self._join_startup_phase_tasks(deadline)
        await self._join_empty_owner_for_unwind(
            self.webhook_finalizers, deadline
        )
        if self._registry is not None:
            await self._await_shutdown_deadline(
                self._registry.close_session_owner_registration(), deadline
            )
        self.call_lifecycle_owners.close_registration()
        await self._join_empty_owner_for_unwind(
            self.call_lifecycle_owners, deadline
        )

        self._state_stop.set()
        await self._join_state_supervisors(deadline)
        if self._registry is not None:
            self._registry.close_registration()
            await self._await_shutdown_deadline(
                self._registry.join_until_empty(), deadline
            )

        self._relay_stop.set()
        await self._join_task("relay", deadline)
        await self._join_empty_owner_for_unwind(self.fixed_supervisors, deadline)
        await self._unwind_startup_dependencies(deadline)
        self._closed = True

    async def _join_startup_phase_tasks(self, deadline: float) -> None:
        while self._startup_phase_tasks:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
            tasks = tuple(self._startup_phase_tasks)
            _done, pending = await asyncio.wait(tasks, timeout=remaining)
            if pending:
                raise RuntimeError("runtime_shutdown_deadline_exceeded") from None

    def _ensure_startup_cleanup(self) -> asyncio.Task[None]:
        if self._startup_cleanup_task is None:
            self._startup_cleanup_task = asyncio.create_task(
                self._startup_cleanup_runner(), name="voice-startup-cleanup"
            )
        return self._startup_cleanup_task

    async def _startup_cleanup_runner(self) -> None:
        hard_exit_required = False
        try:
            await self._unwind_startup()
        except BaseException as error:
            hard_exit_required = str(error) == "runtime_shutdown_deadline_exceeded"
        if hard_exit_required:
            self._invoke_hard_exit()

    def _invoke_hard_exit(self) -> NoReturn:
        self._hard_exit(_RUNTIME_HARD_EXIT_CODE)
        os._exit(_RUNTIME_HARD_EXIT_CODE)

    async def _finish_startup_unwind(self) -> None:
        task = self._ensure_startup_cleanup()
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException:
                break
        await task
        if cancellation is not None:
            raise cancellation

    @staticmethod
    async def _join_empty_owner_for_unwind(
        owner: _OwnedTaskSet,
        deadline: float,
    ) -> None:
        try:
            await owner.join_until_empty(deadline)
        except RuntimeError as error:
            if str(error) != "owned_task_failed":
                raise

    async def _close_dependencies(self, deadline: float) -> None:
        await self._close_writer(deadline)
        if self._call_control is not None:
            await self._await_shutdown_deadline(
                self._call_control.aclose(), deadline
            )
        if self._sink is not None:
            await self._await_shutdown_deadline(self._sink.close(), deadline)
        if self._metrics is not None:
            await self._await_shutdown_deadline(self._metrics.aclose(), deadline)

    async def _unwind_startup_dependencies(self, deadline: float) -> None:
        deadline_failure: RuntimeError | None = None
        if self._call_control is not None:
            try:
                await self._await_shutdown_deadline(
                    self._call_control.aclose(), deadline
                )
            except RuntimeError as error:
                if str(error) == "runtime_shutdown_deadline_exceeded":
                    deadline_failure = error
        if self._sink is not None:
            try:
                await self._await_shutdown_deadline(self._sink.close(), deadline)
            except RuntimeError as error:
                if str(error) == "runtime_shutdown_deadline_exceeded":
                    deadline_failure = error
        try:
            await self._close_writer(deadline)
        except RuntimeError as error:
            if str(error) == "runtime_shutdown_deadline_exceeded":
                deadline_failure = error
        if self._metrics is not None:
            try:
                await self._await_shutdown_deadline(self._metrics.aclose(), deadline)
            except RuntimeError as error:
                if str(error) == "runtime_shutdown_deadline_exceeded":
                    deadline_failure = error
        if deadline_failure is not None:
            raise deadline_failure

    async def _await_shutdown_deadline[ResultT](
        self,
        awaitable: Coroutine[Any, Any, ResultT],
        deadline: float,
    ) -> ResultT:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            awaitable.close()
            self._shutdown_failure_code = "runtime_shutdown_deadline_exceeded"
            raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
        task = asyncio.create_task(awaitable)
        retained = cast(asyncio.Future[object], task)
        self._deadline_tasks.add(retained)

        def finished(done: asyncio.Task[ResultT]) -> None:
            self._deadline_tasks.discard(cast(asyncio.Future[object], done))
            if not done.cancelled():
                with contextlib.suppress(BaseException):
                    done.exception()

        task.add_done_callback(finished)
        done, _pending = await asyncio.wait((task,), timeout=remaining)
        if task not in done:
            task.cancel()
            self._shutdown_failure_code = "runtime_shutdown_deadline_exceeded"
            raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
        cancelled = False
        failed = False
        result: ResultT | None = None
        try:
            result = task.result()
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            failed = True
        if cancelled:
            raise asyncio.CancelledError
        if failed:
            self._shutdown_failure_code = "runtime_shutdown_failed"
            raise RuntimeError("runtime_shutdown_failed") from None
        return cast(ResultT, result)

    async def _close_writer(self, deadline: float) -> None:
        if self._writer is not None and self._writer_task is not None:
            remaining = max(0.001, deadline - asyncio.get_running_loop().time())
            if not self._writer_task.done() and not self._writer.fatal_event.is_set():
                with contextlib.suppress(Exception):
                    await self._writer.drain(timeout_seconds=remaining)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0 and not self._writer_task.done():
                raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
            timed_out = False
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._writer_task),
                    timeout=max(0.001, remaining),
                )
            except TimeoutError:
                timed_out = True
            if timed_out:
                raise RuntimeError("runtime_shutdown_deadline_exceeded") from None

    async def _recover_stale_leases(self) -> None:
        assert self._writer is not None
        stale_leases = self._writer.take_stale_leases()
        if stale_leases and self._call_control is None:
            raise RuntimeError("stale_recovery_failed") from None
        measured = self._measured_call_control
        for stale in stale_leases:
            if self._sparra_enabled or stale.lifecycle is not None:
                lifecycle = await self._startup_await(
                    self._writer.read_call_lifecycle(stale.call_id), code="stale_recovery_failed"
                )
                stale = replace(stale, lifecycle=lifecycle)
            if stale.lifecycle is not None and not stale.lifecycle.transfer_fenced:
                # Leasing is not a negative witness: held leases and later batches
                # remain invisible. Actual AI departure preserves the original phone.
                lifecycle = await self._startup_await(
                    self._writer.mark_call_departed(stale.call_id, now=_aware_utc(self._utcnow())),
                    code="stale_recovery_failed",
                )
                stale = replace(stale, lifecycle=lifecycle)
            if stale.lifecycle is not None and stale.lifecycle.transfer_fenced:
                if self._registry is None:
                    raise RuntimeError("transfer_recovery_registry_missing")
                await cast(Any, self._registry).restore_transfer_fence(stale)
                continue
            command_id = _stale_cleanup_id(stale.call_id)
            try:
                result = (
                    await self._startup_await(
                        measured.hangup(
                            stale.call_control_id,
                            command_id=command_id,
                        ),
                        code="stale_recovery_failed",
                    )
                    if measured is not None
                    else await self._startup_await(
                        self._call_control.hangup(  # type: ignore[union-attr]
                            stale.call_control_id,
                            command_id=command_id,
                        ),
                        code="stale_recovery_failed",
                    )
                )
            except asyncio.CancelledError:
                raise
            except BaseException:
                raise RuntimeError("stale_recovery_failed") from None
            if result.outcome != "accepted":
                raise RuntimeError("stale_recovery_failed") from None
            ended_at = _aware_utc(self._utcnow())
            await self._startup_await(
                self._writer.terminalize_stale_lease(
                    stale,
                    closed_at=ended_at,
                    operation=_stale_terminal_operation(
                        stale,
                        deployment_id=self._deployment_id,
                        ended_at=ended_at,
                        retention_days=self._retention_days,
                    ),
                ),
                code="stale_recovery_failed",
            )

    def _register_fixed_supervisors(self) -> None:
        inventory: tuple[
            tuple[str, Callable[[], Coroutine[Any, Any, None]]], ...
        ] = (
            ("relay", self._relay_loop),
            ("purge", self._purge_loop),
            ("reaper", self._reaper_loop),
            ("qualification-expiry", self._qualification_expiry_loop),
            ("writer-fatal", self._writer_fatal_loop),
            ("readiness", self._readiness_loop),
            ("event-loop-lag", self._event_loop_lag_loop),
        )
        for name, create_coroutine in inventory:
            task = self.fixed_supervisors.spawn_required(
                create_coroutine(), name=f"voice-{name}-supervisor"
            )
            self._fixed_tasks[name] = task

    async def _relay_loop(self) -> None:
        if self._relay is None:
            await self._relay_stop.wait()
            return
        from projetv0_voice.persistence.postgres_sink import (
            OperationSinkCommitAmbiguousError,
        )

        while not self._relay_stop.is_set():
            try:
                result = await self._relay.run_once()
            except asyncio.CancelledError:
                raise
            except OperationSinkCommitAmbiguousError:
                if await _wait_or_stop(
                    self._relay_stop, float(self._relay.claim_lease_seconds)
                ):
                    return
                continue
            except BaseException:
                await self.begin_drain()
                return
            if self._metrics is not None:
                self._metrics.record_relay_run(result.status)
            if result.status == "degraded":
                self._relay_healthy = False
                await self.begin_drain()
                return
            if await _wait_or_stop(self._relay_stop, self._loop_interval_seconds):
                return

    async def _purge_loop(self) -> None:
        while not self._state_stop.is_set():
            if self._purge_once is not None:
                try:
                    result = await self._purge_once()
                except asyncio.CancelledError:
                    raise
                except RecordingPurgeError:
                    result = None
                except BaseException:
                    await self.begin_drain()
                    return
                if result is not None and self._metrics is not None:
                    for _ in range(result.deleted + result.not_found):
                        self._metrics.record_recording("purged")
            if await _wait_or_stop(self._state_stop, self._loop_interval_seconds):
                return

    async def _reaper_loop(self) -> None:
        while not self._state_stop.is_set():
            if self._registry is not None:
                try:
                    await self._registry.reap_expired()
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    await self.begin_drain()
                    return
            if await _wait_or_stop(self._state_stop, self._loop_interval_seconds):
                return

    async def _qualification_expiry_loop(self) -> None:
        while not self._state_stop.is_set():
            if self._registry is not None:
                try:
                    state = await self._registry.qualification_state()
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    state = "expired"
                if state == "consumed":
                    self._qualification_valid = False
                    self._admission_open = False
                    self._publish_current()
                    return
                if state == "expired":
                    self._qualification_valid = False
                    await self.begin_drain()
                    return
            if await _wait_or_stop(self._state_stop, self._loop_interval_seconds):
                return

    async def _writer_fatal_loop(self) -> None:
        if self._writer is None:
            await self._state_stop.wait()
            return
        stop = asyncio.create_task(self._state_stop.wait())
        fatal = asyncio.create_task(self._writer.fatal_event.wait())
        registry_fatal = (
            None
            if self._registry is None
            else asyncio.create_task(self._registry.internal_failure_event.wait())
        )
        watched = (
            (stop, fatal)
            if registry_fatal is None
            else (stop, fatal, registry_fatal)
        )
        done, pending = await asyncio.wait(
            watched, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if fatal in done and self._writer.fatal_event.is_set() and not self._state_stop.is_set():
            await self.begin_drain()
            return
        if (
            registry_fatal is not None
            and registry_fatal in done
            and self._registry is not None
            and self._registry.internal_failure_code is not None
            and not self._state_stop.is_set()
        ):
            await self.begin_drain()

    async def _readiness_loop(self) -> None:
        while not self._state_stop.is_set():
            if await _wait_or_stop(self._state_stop, self._loop_interval_seconds):
                return
            if self._writer is None:
                continue
            try:
                self._last_writer_observation = await self._writer.runtime_observation()
                self._publish_current()
                if self._writer_health_breached():
                    await self.begin_drain()
                    return
            except asyncio.CancelledError:
                raise
            except BaseException:
                await self.begin_drain()
                return

    async def _event_loop_lag_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._state_stop.is_set():
            deadline = loop.time() + self._loop_interval_seconds
            if await _wait_or_stop(self._state_stop, self._loop_interval_seconds):
                return
            if self._metrics is not None:
                self._metrics.record_event_loop_lag(max(0.0, loop.time() - deadline))

    def _publish_current(self) -> RuntimePublishedSnapshot:
        return publish_runtime_readiness(
            self._publication,
            writer=self._last_writer_observation,
            startup_profile_and_stale_recovery_complete=self._startup_complete,
            draining=self._draining,
            admission_open=self._admission_open,
            qualification_state_valid=self._qualification_valid,
            writer_owner_alive_and_ready=(
                self._writer_task is not None and not self._writer_task.done()
            ),
            no_writer_fatal_or_degradation=(
                self._writer is not None
                and not self._writer.fatal_event.is_set()
                and not self._writer.is_degraded
            ),
            relay_supervisor_alive=(
                self._relay is not None
                and "relay" in self._fixed_tasks
                and not self._fixed_tasks["relay"].done()
            ),
            no_permanent_relay_or_sink_degradation=self._relay_healthy,
        )

    def _writer_health_breached(self) -> bool:
        observation = self._last_writer_observation
        return (
            not observation.writer_quick_check
            or observation.storage_bytes > _MAX_STORAGE_BYTES
            or observation.writer_queue_oldest_age > _MAX_WRITER_QUEUE_AGE_SECONDS
            or observation.outbox_oldest_age > _MAX_OUTBOX_AGE_SECONDS
            or self._writer is None
            or self._writer.fatal_event.is_set()
            or self._writer.is_degraded
        )

    async def _join_state_supervisors(self, deadline: float) -> bool:
        terminal_owner_failed = False
        for name in (
            "purge",
            "reaper",
            "qualification-expiry",
            "writer-fatal",
            "readiness",
            "event-loop-lag",
        ):
            try:
                await self._join_task(name, deadline)
            except RuntimeError as error:
                if str(error) != "owned_task_failed":
                    raise
                terminal_owner_failed = True
        return terminal_owner_failed

    async def _join_task(self, name: str, deadline: float) -> None:
        task = self._fixed_tasks.get(name)
        if task is None:
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
        timed_out = False
        failed = False
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except TimeoutError:
            timed_out = True
        except BaseException:
            failed = True
        if timed_out:
            raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
        if failed:
            raise RuntimeError("owned_task_failed") from None


async def build_production_runtime(
    settings: RuntimeSettingsV1,
    *,
    factories: RuntimeProductionFactories,
    utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = lambda: asyncio.get_running_loop().time(),
    startup_phase_timeout_seconds: float = 5.0,
    startup_phase_timeouts: Mapping[str, float] | None = None,
    shutdown_timeout_seconds: float | None = None,
    hard_exit: Callable[[int], NoReturn] = os._exit,
) -> RuntimeProductionGraph:
    """Validate and compose the complete non-ASGI production runtime graph."""

    if (
        type(settings) is not RuntimeSettingsV1
        or type(factories) is not RuntimeProductionFactories
        or not callable(utcnow)
        or not callable(monotonic)
        or isinstance(startup_phase_timeout_seconds, bool)
        or not isinstance(startup_phase_timeout_seconds, int | float)
        or not math.isfinite(float(startup_phase_timeout_seconds))
        or startup_phase_timeout_seconds <= 0
        or not callable(hard_exit)
    ):
        raise ValueError("runtime_production_composition_invalid") from None
    selected_shutdown_timeout = (
        float(settings.shutdown_grace_seconds)
        if shutdown_timeout_seconds is None
        else shutdown_timeout_seconds
    )
    if (
        isinstance(selected_shutdown_timeout, bool)
        or not isinstance(selected_shutdown_timeout, int | float)
        or not math.isfinite(float(selected_shutdown_timeout))
        or selected_shutdown_timeout <= 0
    ):
        raise ValueError("runtime_production_composition_invalid") from None

    metrics: RuntimeMetrics | None = None
    sink: _RuntimeSink | None = None
    raw_call_control: _ProcessCallControl | None = None
    composition_failed = False
    try:
        factories.validate_artifacts(settings)
        manifest = factories.load_manifest(settings)
        now = _aware_utc(utcnow())
        selection = factories.load_profile(settings, manifest, now)
        if not isinstance(manifest, AgentManifestV1) or not isinstance(
            selection, RuntimeProfileSelection
        ):
            raise RuntimeError("runtime_artifacts_invalid")
        _validate_profile_selection(settings, manifest, selection, now=now)

        telnyx_api_key = factories.read_secret(settings.telnyx_api_key_file)
        webhook_public_key = factories.read_secret(settings.telnyx_webhook_public_key_file)
        openrouter_api_key = factories.read_secret(settings.openrouter_api_key_file)
        postgres_dsn = factories.read_secret(settings.postgres_dsn_file)
        if not all(
            isinstance(value, SecretStr) and bool(value.get_secret_value())
            for value in (
                telnyx_api_key,
                webhook_public_key,
                openrouter_api_key,
                postgres_dsn,
            )
        ):
            raise RuntimeError("runtime_secret_invalid")
        api_key_sha256 = hashlib.sha256(
            telnyx_api_key.get_secret_value().encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(selection.profile.telnyx_api_key_sha256, api_key_sha256):
            raise RuntimeError("runtime_profile_api_key_mismatch")

        keyring = factories.load_keyring(settings)
        if not isinstance(keyring, CryptoKeyring):
            raise RuntimeError("runtime_keyring_invalid")
        token = settings.observability_token()
        metrics = RuntimeMetrics.production(
            token,
            endpoint=settings.otlp_http_endpoint,
        )
        if not isinstance(metrics, RuntimeMetrics):
            raise RuntimeError("runtime_metrics_composition_invalid")
        writer = PersistenceWriter(Path(str(settings.sqlite_path)), keyring)
        sink = factories.sink_factory(postgres_dsn)

        supervisor_ref: RuntimeSupervisor | None = None

        async def begin_drain() -> None:
            if supervisor_ref is None:
                raise RuntimeError("runtime_supervisor_unavailable") from None
            await supervisor_ref.begin_drain()

        async def prepare_sparra_fifo() -> None:
            await maintain_call_content(
                writer, cast(Any, sink), session_factory.erase_call_by_id,
                utcnow=utcnow, timeout_seconds=settings.call_cleanup_phase_timeout_seconds + 20,
            )
        relay = OutboxRelay(
            writer,
            cast(Any, sink),
            on_degraded=begin_drain,
            drain=begin_drain,
            before_fifo=prepare_sparra_fifo if manifest.sparra is not None else None,
            stop_erased_call=(lambda call_id: session_factory.stop_call_content_by_id(call_id))
            if manifest.sparra is not None
            else None,
        )
        raw_call_control = factories.call_control_factory(telnyx_api_key)
        inference = factories.inference_factory(
            openrouter_api_key,
            selection.profile,
            manifest.language,
        )
        if not isinstance(inference, RuntimeInferenceFactories):
            raise RuntimeError("runtime_inference_composition_invalid")
        candidate_run_id = (
            settings.qualification_run_id
            if settings.runtime_mode == "qualification_candidate"
            else None
        )
        supervisor = RuntimeSupervisor(
            writer=writer,
            call_control=raw_call_control,
            metrics=metrics,
            candidate_run_id=candidate_run_id,
            deployment_id=settings.deployment_id,
            retention_days=manifest.transcript_retention_days,
            sparra_enabled=manifest.sparra is not None,
            utcnow=utcnow,
            loop_interval_seconds=0.25,
            startup_phase_timeout_seconds=float(startup_phase_timeout_seconds),
            startup_phase_timeouts=startup_phase_timeouts,
            shutdown_timeout_seconds=float(selected_shutdown_timeout),
            hard_exit=hard_exit,
        )
        supervisor_ref = supervisor
        measured_call_control = supervisor.call_control_facade
        capacity = select_call_capacity(
            profile=selection.profile,
            manifest=manifest,
            deployment_max_calls=settings.deployment_max_calls,
            override=selection.override,
        )
        admission_expires_at = (
            selection.profile.expires_at
            if isinstance(selection.profile, QualificationCandidateProfileV1)
            else None
            if selection.override is None
            else selection.override.expires_at
        )
        registry = CallRegistry(
            writer=writer,
            call_control=cast(Any, measured_call_control),
            tenant_id=manifest.tenant_id,
            agent_id=manifest.agent_id,
            deployment_id=settings.deployment_id,
            capacity=capacity,
            lease_ttl_seconds=selection.profile.call_lease_ttl_seconds,
            stream_url=settings.telnyx_media_wss_url,
            retention_days=manifest.transcript_retention_days,
            utcnow=utcnow,
            monotonic=monotonic,
            candidate_run_id=candidate_run_id,
            admission_expires_at=admission_expires_at,
            qualification_observer=supervisor.observe_qualification_state,
            sparra=manifest.sparra,
            called_did=manifest.dids[0],
            begin_call=getattr(sink, "begin_call", None),
        )
        lease_authority = ProcessLeaseAuthority(registry)
        gate = SynchronousUnauthenticatedGate(capacity)
        handshake = AuthenticatedTelnyxHandshakeService(
            profile=selection.profile,
            lease_authority=lease_authority,
            unauthenticated_gate=gate,
            timeout_seconds=settings.handshake_timeout_seconds,
        )
        recording_retention_days = (
            manifest.recording_retention_days or manifest.transcript_retention_days
        )

        def recording_factory(identity: CallIdentity) -> RecordingBoundary:
            del identity
            return TelnyxRecordingBoundary(
                telnyx=cast(Any, measured_call_control),
                writer=writer,
                retention_days=recording_retention_days,
                required=manifest.recording_required,
                play_beep=manifest.recording_play_beep,
                utcnow=utcnow,
            )

        session_factory = ProcessSessionFactory(
            registry=cast(Any, registry),
            registrar=supervisor.call_lifecycle_owners,
            runtime_metrics=metrics,
            manifest=manifest,
            profile=selection.profile,
            writer=writer,
            keyring=keyring,
            stt_http_client_factory=inference.stt_http_client_factory,
            stt_factory=inference.stt_factory,
            llm_factory=inference.llm_factory,
            tts_factory=inference.tts_factory,
            recording_factory=recording_factory,
            recording_call_control_identity=measured_call_control,
            idle_timeout_seconds=settings.call_idle_timeout_seconds,
            cleanup_phase_timeout_seconds=(settings.call_cleanup_phase_timeout_seconds),
        )
        verifier = TelnyxWebhookVerifier(
            public_key=webhook_public_key.get_secret_value(),
            call_control_required_types=(
                "call.initiated",
                "call.answered",
                "call.hangup",
                "call.recording.saved",
                "call.recording.error",
            ),
        )
        async def resolve_webhook(event: VerifiedWebhook) -> ResolvedWebhook:
            if event.event_type in {"call.recording.saved", "call.recording.error"}:
                return ResolvedWebhook(resolve_recording_webhook(event))
            return await registry.resolve_webhook(event)

        async def resolve_duplicate_webhook(event: VerifiedWebhook) -> ResolvedWebhook:
            if event.event_type in {"call.recording.saved", "call.recording.error"}:
                return ResolvedWebhook(resolve_recording_webhook(event))
            return await registry.resolve_duplicate_webhook(event)

        webhook_processor = TelnyxWebhookProcessor(
            verifier=verifier,
            resolver=resolve_webhook,
            duplicate_resolver=resolve_duplicate_webhook,
            finalizer_owner=supervisor,
        )

        async def recording_after_commit(
            event: VerifiedWebhook,
            effect: WebhookDurableEffect | None,
        ) -> WebhookDisposition | None:
            return await after_recording_webhook_commit(
                event,
                effect,
                telnyx=cast(Any, measured_call_control),
                writer=writer,
                local_drain=session_factory.drain_call_by_id,
                monotonic=monotonic,
                timeout_seconds=settings.call_cleanup_phase_timeout_seconds,
            )

        async def purge_once() -> PurgeBatchResult:
            return await purge_recordings_once(
                worker_id="voice-recording-purge",
                lease_seconds=30,
                batch_size=100,
                telnyx=cast(Any, measured_call_control),
                sink=cast(Any, sink),
                utcnow=utcnow,
            )

        supervisor.bind_runtime_graph(
            relay=relay,
            registry=registry,
            sink=sink,
            unauthenticated_gate=gate,
            purge_once=purge_once,
            recording_after_commit=recording_after_commit,
        )
        return RuntimeProductionGraph(
            settings=settings,
            manifest=manifest,
            profile=selection.profile,
            override=selection.override,
            keyring=keyring,
            supervisor=supervisor,
            metrics=metrics,
            writer=writer,
            sink=sink,
            relay=relay,
            raw_call_control=raw_call_control,
            measured_call_control=measured_call_control,
            registry=registry,
            lease_authority=lease_authority,
            unauthenticated_gate=gate,
            handshake=handshake,
            session_factory=session_factory,
            webhook_processor=webhook_processor,
            recording_factory=recording_factory,
            recording_call_control_identity=measured_call_control,
        )
    except asyncio.CancelledError:
        await _close_failed_composition(
            raw_call_control,
            sink,
            metrics,
            timeout_seconds=float(startup_phase_timeout_seconds),
            hard_exit=hard_exit,
        )
        raise
    except BaseException:
        await _close_failed_composition(
            raw_call_control,
            sink,
            metrics,
            timeout_seconds=float(startup_phase_timeout_seconds),
            hard_exit=hard_exit,
        )
        composition_failed = True
    if composition_failed:
        raise RuntimeError("runtime_production_composition_failed") from None
    raise RuntimeError("runtime_production_composition_failed") from None


def _validate_profile_selection(
    settings: RuntimeSettingsV1,
    manifest: AgentManifestV1,
    selection: RuntimeProfileSelection,
    *,
    now: datetime,
) -> None:
    profile = selection.profile
    expected = (
        profile.deployment_id == settings.deployment_id
        and profile.runtime_contract_sha256 == settings.runtime_contract_sha256
        and profile.image_digest == settings.image_digest
        and profile.agent_bundle_sha256 == settings.agent_bundle_sha256
        and profile.inference_profile_sha256
        == settings.inference_profile_sha256
        and profile.inference_profile_sha256
        == canonical_inference_profile_sha256(profile.inference)
    )
    if not expected:
        raise RuntimeError("runtime_profile_binding_invalid")
    if settings.runtime_mode == "strict":
        valid_mode = (
            isinstance(profile, QualifiedDeploymentProfileV1)
            and selection.override is None
            and profile.telnyx_data_locality == "EU"
        )
    elif settings.runtime_mode == "qualification_candidate":
        valid_mode = (
            isinstance(profile, QualificationCandidateProfileV1)
            and selection.override is None
            and settings.qualification_run_id == profile.run_id
            and settings.benchmark_did_sha256 == profile.benchmark_did_hash
            and now < profile.expires_at
            and manifest.recording_mode == "off"
        )
    else:
        override = selection.override
        valid_mode = (
            isinstance(profile, QualifiedDeploymentProfileV1)
            and isinstance(override, QualificationOverrideV1)
            and settings.qualification_run_id == override.run_id
            and override.deployment_id == settings.deployment_id
            and override.qualified_profile_sha256
            == canonical_qualified_profile_sha256(profile)
            and profile.qualified_at <= override.created_at < now < override.expires_at
            and profile.telnyx_data_locality == "EU"
        )
    if not valid_mode:
        raise RuntimeError("runtime_profile_mode_invalid")


async def _close_failed_composition(
    call_control: _ProcessCallControl | None,
    sink: _RuntimeSink | None,
    metrics: RuntimeMetrics | None,
    *,
    timeout_seconds: float,
    hard_exit: Callable[[int], NoReturn] = os._exit,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    retained: set[asyncio.Task[None]] = set()
    cancellation: asyncio.CancelledError | None = None
    inventory = (
        ("call-control", None if call_control is None else call_control.aclose),
        ("sink", None if sink is None else sink.close),
        ("metrics", None if metrics is None else metrics.aclose),
    )
    for name, close in inventory:
        if close is None:
            continue
        remaining = deadline - loop.time()
        if remaining <= 0:
            _hard_exit_now(hard_exit)
        start_gate = asyncio.Event()

        async def run_close(
            operation: Callable[[], Coroutine[Any, Any, None]] = close,
            gate: asyncio.Event = start_gate,
        ) -> None:
            await gate.wait()
            await operation()

        runner = run_close()
        create_failed = False
        try:
            task = asyncio.create_task(
                runner, name=f"voice-composition-close-{name}"
            )
        except BaseException:
            create_failed = True
        if create_failed:
            with contextlib.suppress(BaseException):
                runner.close()
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    _hard_exit_now(hard_exit)
                try:
                    await asyncio.sleep(remaining)
                except asyncio.CancelledError:
                    continue
        retained.add(task)

        def consume(completed: asyncio.Task[None]) -> None:
            retained.discard(completed)
            if not completed.cancelled():
                with contextlib.suppress(BaseException):
                    completed.exception()

        task.add_done_callback(consume)
        start_gate.set()
        done: set[asyncio.Task[None]] = set()
        while task not in done:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                completed, _pending = await asyncio.wait(
                    (task,), timeout=remaining
                )
                done.update(completed)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if task not in done:
            task.cancel()
            _hard_exit_now(hard_exit)
        with contextlib.suppress(BaseException):
            task.result()
    if cancellation is not None:
        raise cancellation


def _hard_exit_now(hard_exit: Callable[[int], NoReturn]) -> NoReturn:
    hard_exit(_RUNTIME_HARD_EXIT_CODE)
    os._exit(_RUNTIME_HARD_EXIT_CODE)


def _new_publication() -> RuntimePublication:
    return RuntimePublication(
        RuntimePublishedSnapshot(
            generation=0,
            ready=False,
            draining=False,
            writer_queue_depth=0,
            writer_queue_oldest_age=0.0,
            writer_quick_check=False,
            outbox_depth=0,
            outbox_oldest_age=0.0,
            outbox_bytes=0,
            storage_bytes=0,
            startup_profile_and_stale_recovery_complete=False,
            admission_open=False,
            qualification_state_valid=False,
            writer_owner_alive_and_ready=False,
            no_writer_fatal_or_degradation=False,
            relay_supervisor_alive=False,
            no_permanent_relay_or_sink_degradation=False,
        )
    )


def _stale_cleanup_id(call_id: UUID) -> UUID:
    digest = hashlib.sha256(
        b"projetv0.voice.stale-cleanup.v1\x00" + call_id.bytes
    ).digest()
    return UUID(bytes=digest[:16], version=4)


def _stale_terminal_operation(
    stale: StaleLease,
    *,
    deployment_id: str,
    ended_at: datetime,
    retention_days: int,
) -> VoiceOperationV1:
    operation_digest = hashlib.sha256(
        b"projetv0.voice.stale-terminal.v1\x00" + stale.call_id.bytes
    ).digest()
    facts = stale.lifecycle
    if facts is not None and facts.transfer_fenced:
        raise RuntimeError("transfer_fence_cannot_stale_terminalize")
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(bytes=operation_digest[:16], version=4),
        deployment_id=deployment_id,
        call_id=stale.call_id,
        occurred_at=ended_at,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id=stale.call_control_id,
            telnyx_call_leg_id=None if facts is None else facts.telnyx_call_leg_id,
            telnyx_call_session_id=None if facts is None else facts.telnyx_call_session_id,
            status="failed",
            disclosure_state="completed"
            if facts is not None
            and facts.disclosure_evidence is not None
            and facts.disclosure_evidence.completed_at is not None
            else "failed",
            started_at=None if facts is None else facts.started_at,
            ended_at=ended_at,
            end_reason="stale_process_recovery",
            retention_until=ended_at + timedelta(days=retention_days)
            if facts is None
            else facts.retention_until,
            **(
                cast(
                    Any,
                    (
                        {"disclosure_evidence": facts.disclosure_evidence}
                        if facts is not None and facts.disclosure_evidence is not None
                        else {}
                    ),
                )
            ),
        ),
    )


def _aware_utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeError("runtime_clock_invalid") from None
    return value.astimezone(UTC)


async def _wait_or_stop(stop: asyncio.Event, delay_seconds: float) -> bool:
    try:
        async with asyncio.timeout(delay_seconds):
            await stop.wait()
    except TimeoutError:
        return False
    return True


__all__ = [
    "RuntimeInferenceFactories",
    "RuntimeProductionFactories",
    "RuntimeProductionGraph",
    "RuntimeProfileSelection",
    "RuntimeSupervisor",
    "build_production_runtime",
    "publish_runtime_readiness",
]
