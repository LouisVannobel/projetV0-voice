"""Production Voice Cell process ownership and lifecycle supervision."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import SecretStr

from projetv0_voice.metrics import (
    RuntimeMetrics,
    RuntimePublication,
    RuntimePublishedSnapshot,
)
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.relay import OutboxRelay
from projetv0_voice.persistence.writer import (
    PersistenceWriter,
    QualificationRunConsumed,
    StaleLease,
    WebhookCommitResult,
    WebhookCommitValue,
    WriterRuntimeObservation,
)
from projetv0_voice.telnyx.call_control import CallControlResult, StreamingStartV1
from projetv0_voice.telnyx.recordings import (
    PurgeBatchResult,
    RecordingPurgeError,
    recording_metric_transition,
)
from projetv0_voice.telnyx.webhooks import (
    ResolvedWebhook,
    VerifiedWebhook,
    WebhookDisposition,
    WebhookDurableEffect,
)

_MAX_STORAGE_BYTES = 268_435_456
_MAX_WRITER_QUEUE_AGE_SECONDS = 1.0
_MAX_OUTBOX_AGE_SECONDS = 900.0

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


class _CloseableGate(Protocol):
    def close(self) -> None: ...


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
            with self._lock:
                self._terminal_failure = True
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
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        loop_interval_seconds: float = 0.25,
        shutdown_timeout_seconds: float = 30.0,
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
            or not isinstance(deployment_id, str)
            or not deployment_id
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
        self._utcnow = utcnow
        self._loop_interval_seconds = float(loop_interval_seconds)
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._writer_task: asyncio.Task[None] | None = None
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
        self._closed = False

    async def startup(self) -> None:
        if self._started or self._closed or self._writer is None:
            raise RuntimeError("runtime_startup_invalid") from None
        self._started = True
        try:
            self._writer_task = asyncio.create_task(
                self._writer.run(), name="voice-persistence-writer"
            )
            async with asyncio.timeout(self._shutdown_timeout_seconds):
                if not await self._writer.wait_ready():
                    raise RuntimeError("writer_startup_failed")
                if not await self._writer.quick_check():
                    raise RuntimeError("writer_quick_check_failed")
                if (
                    self._candidate_run_id is not None
                    and await self._writer.qualification_run_consumed(
                        self._candidate_run_id
                    )
                ):
                    raise RuntimeError("qualification_run_consumed")
                self._qualification_valid = True
                if self._sink is not None:
                    await self._sink.open()
                await self._recover_stale_leases()
                self._register_fixed_supervisors()
                self.fixed_supervisors.close_registration()
                self._startup_complete = True
                self._admission_open = True
                self._last_writer_observation = await self._writer.runtime_observation()
                self._publish_current()
                if self._writer_health_breached():
                    await self.begin_drain()
        except asyncio.CancelledError:
            await self._unwind_startup()
            raise
        except BaseException as error:
            code = (
                str(error)
                if str(error)
                in {
                    "stale_recovery_failed",
                    "qualification_run_consumed",
                    "writer_startup_failed",
                    "writer_quick_check_failed",
                }
                else "runtime_startup_failed"
            )
            await self._unwind_startup()
            raise RuntimeError(code) from None

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
        if self._registry is not None:
            await self._registry.begin_drain()
        self._drain_event.set()

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
    ) -> _WebhookFinalizationHandle:
        if (
            self._writer is None
            or not isinstance(event, VerifiedWebhook)
            or not isinstance(resolution, ResolvedWebhook)
        ):
            raise RuntimeError("webhook_finalizer_invalid") from None
        completion: asyncio.Future[WebhookDisposition] = (
            asyncio.get_running_loop().create_future()
        )
        task = self.webhook_finalizers.try_start(
            self._finalize_webhook(event, resolution, completion),
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
                    "semantic_fingerprint_sha256": (
                        event.semantic_fingerprint_sha256
                    ),
                },
                lease=None if effect is None else effect.lease,
                operation=None if effect is None else effect.operation,
                legacy_v1_semantic_fingerprint_sha256=(
                    event.legacy_v1_semantic_fingerprint_sha256
                ),
                qualification_run_id=(
                    None
                    if self._registry is None
                    else self._registry.candidate_run_id
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
                async with asyncio.timeout(
                    self._writer.control_commit_timeout_seconds
                ):
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
            if not completion.done():
                completion.set_result(WebhookDisposition(503))
            return

        if timed_out:
            if self._registry is not None:
                with contextlib.suppress(Exception):
                    await self._registry.confirm_late_after_fail_closed(
                        event, resolution, result
                    )
            return

        disposition = WebhookDisposition(
            503 if isinstance(result, QualificationRunConsumed) else 200
        )
        if self._registry is not None:
            try:
                disposition = await self._registry.reconcile_after_commit(
                    event, resolution, result
                )
            except BaseException:
                disposition = WebhookDisposition(500)
        if isinstance(result, WebhookCommitResult) and self._metrics is not None:
            transition = recording_metric_transition(event, resolution.effect, result)
            if transition is not None:
                self._metrics.record_recording(transition)
        if self._recording_after_commit is not None:
            try:
                recording_disposition = await self._recording_after_commit(
                    event, resolution.effect
                )
            except asyncio.CancelledError:
                raise
            except BaseException:
                recording_disposition = WebhookDisposition(500)
            if recording_disposition is not None:
                disposition = recording_disposition
        if not completion.done():
            completion.set_result(disposition)

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._aclose_core(), name="voice-runtime-close"
            )
        await asyncio.shield(self._close_task)

    async def _aclose_core(self) -> None:
        if self._closed:
            return
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout_seconds
        await self.begin_drain()
        await self.webhook_finalizers.join_until_empty(deadline)
        if self._registry is not None:
            await self._registry.close_session_owner_registration()
        self.call_lifecycle_owners.close_registration()
        await self.call_lifecycle_owners.join_until_empty(deadline)

        self._state_stop.set()
        await self._join_state_supervisors(deadline)
        if self._registry is not None:
            self._registry.close_registration()
            await self._registry.join_until_empty()

        self._relay_stop.set()
        await self._join_task("relay", deadline)
        await self.fixed_supervisors.join_until_empty(deadline)
        await self._close_dependencies(deadline)
        self._closed = True

    async def _unwind_startup(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout_seconds
        await self.begin_drain()
        await self.webhook_finalizers.join_until_empty(deadline)
        if self._registry is not None:
            await self._registry.close_session_owner_registration()
        self.call_lifecycle_owners.close_registration()
        await self.call_lifecycle_owners.join_until_empty(deadline)

        self._state_stop.set()
        await self._join_state_supervisors(deadline)
        if self._registry is not None:
            self._registry.close_registration()
            await self._registry.join_until_empty()

        self._relay_stop.set()
        await self._join_task("relay", deadline)
        await self.fixed_supervisors.join_until_empty(deadline)
        await self._unwind_startup_dependencies(deadline)
        self._closed = True

    async def _close_dependencies(self, deadline: float) -> None:
        await self._close_writer(deadline)
        if self._call_control is not None:
            with contextlib.suppress(Exception):
                await self._call_control.aclose()
        if self._sink is not None:
            with contextlib.suppress(Exception):
                await self._sink.close()
        if self._metrics is not None:
            with contextlib.suppress(Exception):
                await self._metrics.aclose()

    async def _unwind_startup_dependencies(self, deadline: float) -> None:
        if self._call_control is not None:
            with contextlib.suppress(Exception):
                await self._call_control.aclose()
        if self._sink is not None:
            with contextlib.suppress(Exception):
                await self._sink.close()
        await self._close_writer(deadline)
        if self._metrics is not None:
            with contextlib.suppress(Exception):
                await self._metrics.aclose()

    async def _close_writer(self, deadline: float) -> None:
        if self._writer is not None and self._writer_task is not None:
            remaining = max(0.001, deadline - asyncio.get_running_loop().time())
            with contextlib.suppress(Exception):
                await self._writer.drain(timeout_seconds=remaining)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0 and not self._writer_task.done():
                raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._writer_task),
                    timeout=max(0.001, remaining),
                )
            except TimeoutError:
                raise RuntimeError("runtime_shutdown_deadline_exceeded") from None

    async def _recover_stale_leases(self) -> None:
        assert self._writer is not None
        stale_leases = self._writer.take_stale_leases()
        if stale_leases and self._call_control is None:
            raise RuntimeError("stale_recovery_failed") from None
        measured = self._measured_call_control
        for stale in stale_leases:
            command_id = _stale_cleanup_id(stale.call_id)
            try:
                result = (
                    await measured.hangup(
                        stale.call_control_id,
                        command_id=command_id,
                    )
                    if measured is not None
                    else await self._call_control.hangup(  # type: ignore[union-attr]
                        stale.call_control_id,
                        command_id=command_id,
                    )
                )
            except asyncio.CancelledError:
                raise
            except BaseException:
                raise RuntimeError("stale_recovery_failed") from None
            if result.outcome != "accepted":
                raise RuntimeError("stale_recovery_failed") from None
            ended_at = _aware_utc(self._utcnow())
            await self._writer.terminalize_stale_lease(
                stale,
                closed_at=ended_at,
                operation=_stale_terminal_operation(
                    stale,
                    deployment_id=self._deployment_id,
                    ended_at=ended_at,
                    retention_days=self._retention_days,
                ),
            )

    def _register_fixed_supervisors(self) -> None:
        inventory = (
            ("relay", self._relay_loop()),
            ("purge", self._purge_loop()),
            ("reaper", self._reaper_loop()),
            ("qualification-expiry", self._qualification_expiry_loop()),
            ("writer-fatal", self._writer_fatal_loop()),
            ("readiness", self._readiness_loop()),
            ("event-loop-lag", self._event_loop_lag_loop()),
        )
        for name, coroutine in inventory:
            task = self.fixed_supervisors.spawn_required(
                coroutine, name=f"voice-{name}-supervisor"
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

    async def _join_state_supervisors(self, deadline: float) -> None:
        for name in (
            "purge",
            "reaper",
            "qualification-expiry",
            "writer-fatal",
            "readiness",
            "event-loop-lag",
        ):
            await self._join_task(name, deadline)

    async def _join_task(self, name: str, deadline: float) -> None:
        task = self._fixed_tasks.get(name)
        if task is None:
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError("runtime_shutdown_deadline_exceeded") from None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except TimeoutError:
            raise RuntimeError("runtime_shutdown_deadline_exceeded") from None


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
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(bytes=operation_digest[:16], version=4),
        deployment_id=deployment_id,
        call_id=stale.call_id,
        occurred_at=ended_at,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id=stale.call_control_id,
            telnyx_call_leg_id=None,
            telnyx_call_session_id=None,
            status="failed",
            disclosure_state="failed",
            started_at=None,
            ended_at=ended_at,
            end_reason="stale_process_recovery",
            retention_until=ended_at + timedelta(days=retention_days),
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


__all__ = ["RuntimeSupervisor", "publish_runtime_readiness"]
