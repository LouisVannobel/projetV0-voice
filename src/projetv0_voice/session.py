"""Per-call lifecycle, disclosure, persistence, and recording ownership."""

from __future__ import annotations

import asyncio
import base64
import hmac
import math
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from importlib.metadata import version
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid4

from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, TurnUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.pipeline import (
    CallRuntime,
    FirstFailure,
    ObservedPipeline,
    PipelineTransport,
    build_pipeline,
    build_runtime,
)
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake


class DisclosureState(Enum):
    PLAYING = auto()
    MARK_PENDING = auto()
    ACK_COMMITTING = auto()
    DISCLOSURE_DURABLE = auto()
    RECORDING_STARTING = auto()
    ACTIVE = auto()
    ABORTED = auto()


class RecordingStartState(Enum):
    STARTED = auto()
    DEFINITELY_NOT_STARTED = auto()
    INDETERMINATE = auto()


@dataclass(frozen=True, slots=True)
class RecordingStartResult:
    state: RecordingStartState
    gate_may_open: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, RecordingStartState) or type(self.gate_may_open) is not bool:
            raise ValueError("recording_result_invalid")
        if self.gate_may_open and self.state is not RecordingStartState.INDETERMINATE:
            raise ValueError("recording_result_invalid")


@dataclass(frozen=True, slots=True)
class CallIdentity:
    call_id: UUID = field(repr=False)
    durable_generation: str = field(repr=False)
    lease_identity: str = field(repr=False)
    lease_claim: object = field(repr=False)
    deployment_id: str
    registry_handle: object = field(repr=False)
    telnyx_call_control_id: str = field(repr=False)
    telnyx_call_leg_id: str | None = field(repr=False)
    telnyx_call_session_id: str | None = field(repr=False)
    stream_id: str = field(repr=False)
    started_at: datetime
    retention_until: datetime

    def __post_init__(self) -> None:
        required_text = (
            self.durable_generation,
            self.lease_identity,
            self.deployment_id,
            self.telnyx_call_control_id,
            self.stream_id,
        )
        if (
            not isinstance(self.call_id, UUID)
            or any(not isinstance(value, str) or not value for value in required_text)
            or self.started_at.tzinfo is None
            or self.started_at.utcoffset() is None
            or self.retention_until.tzinfo is None
            or self.retention_until.utcoffset() is None
            or self.retention_until <= self.started_at
        ):
            raise ValueError("call_identity_invalid")

    def __repr__(self) -> str:
        return f"CallIdentity(deployment_id={self.deployment_id!r})"


class ControlWriter(Protocol):
    async def commit_control(self, command: PersistenceCommand) -> None: ...


class RecordingBoundary(Protocol):
    async def start(self, identity: CallIdentity) -> RecordingStartResult: ...

    async def cleanup(
        self,
        identity: CallIdentity,
        *,
        recording_may_be_active: bool,
        reason: str,
    ) -> None: ...


class ServiceLifecycleError(RuntimeError):
    """A constant-safe inference resource lifecycle error."""


class PublicSttHttpClient(Protocol):
    async def aclose(self) -> None: ...


@dataclass(slots=True)
class ServiceBundle:
    """Per-call inference services and the two retained client owners."""

    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor
    stt_http_client: PublicSttHttpClient = field(repr=False)
    close_timeout_seconds: float = field(default=5.0, repr=False)
    _lock: asyncio.Lock = field(init=False, repr=False)
    _stt_closed: bool = field(default=False, init=False, repr=False)
    _llm_closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.close_timeout_seconds, int | float)
            or isinstance(self.close_timeout_seconds, bool)
            or not math.isfinite(self.close_timeout_seconds)
            or self.close_timeout_seconds <= 0
        ):
            raise ValueError("service_close_config_invalid")
        self.close_timeout_seconds = float(self.close_timeout_seconds)
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "ServiceBundle()"

    async def aclose(self) -> None:
        async with self._lock:
            close_jobs: list[asyncio.Task[None]] = []
            if not self._stt_closed:
                close_jobs.append(
                    asyncio.create_task(
                        self._close_stt_client(),
                        name="close-stt-http-client",
                    )
                )
            if not self._llm_closed:
                close_jobs.append(
                    asyncio.create_task(
                        self._close_llm_client(),
                        name="close-llm-client",
                    )
                )
            if not close_jobs:
                return
            try:
                await asyncio.wait(close_jobs)
            except asyncio.CancelledError as cancellation:
                for task in close_jobs:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*close_jobs, return_exceptions=True)
                raise cancellation
            results = await asyncio.gather(*close_jobs, return_exceptions=True)

            child_cancellation = next(
                (
                    result
                    for result in results
                    if isinstance(result, asyncio.CancelledError)
                ),
                None,
            )
            if child_cancellation is not None:
                raise child_cancellation
            if any(isinstance(result, BaseException) for result in results):
                raise ServiceLifecycleError("service_close_failed")

    async def _close_stt_client(self) -> None:
        await self._bounded_close(self.stt_http_client.aclose)
        self._stt_closed = True

    async def _close_llm_client(self) -> None:
        await self._bounded_close(self._close_pinned_llm_client)
        self._llm_closed = True

    async def _bounded_close(self, close: Callable[[], Awaitable[None]]) -> None:
        try:
            async with asyncio.timeout(self.close_timeout_seconds):
                await close()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ServiceLifecycleError("service_close_failed") from None

    async def _close_pinned_llm_client(self) -> None:
        if version("pipecat-ai") != "1.7.0":
            raise ServiceLifecycleError("service_close_failed")
        client = getattr(self.llm, "_client", None)
        close = getattr(client, "close", None)
        if not callable(close):
            raise ServiceLifecycleError("service_close_failed")
        result = close()
        if not hasattr(result, "__await__"):
            raise ServiceLifecycleError("service_close_failed")
        await cast(Awaitable[None], result)


# TurnUpsertPayloadV1 permits 65,536 decoded ciphertext bytes; AES-GCM appends
# a 16-byte tag, so the plaintext copy must remain below that structural bound.
MAX_TURN_TEXT_BYTES = 65_520


class TurnWriter(Protocol):
    def try_enqueue_turn(self, operation: VoiceOperationV1) -> bool: ...


class TurnRecorder:
    """Synchronous, atomic turn encryption and shared-writer admission."""

    def __init__(
        self,
        *,
        identity: CallIdentity,
        writer: TurnWriter,
        keyring: CryptoKeyring,
        first_failure: FirstFailure,
        uuid_factory: Callable[[], UUID] = uuid4,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._identity = identity
        self._writer = writer
        self._keyring = keyring
        self._first_failure = first_failure
        self._uuid_factory = uuid_factory
        self._utcnow = utcnow
        self._accepting = True
        self._turn_no = 0

    def close(self) -> None:
        self._accepting = False

    def record_user(self, content: str | None, timestamp: str) -> None:
        self._record(
            role="user",
            source="stt_final",
            content=content,
            timestamp=timestamp,
            interrupted=False,
        )

    def record_assistant(self, content: str, timestamp: str, interrupted: bool) -> None:
        self._record(
            role="assistant",
            source="pipecat_assistant",
            content=content,
            timestamp=timestamp,
            interrupted=interrupted,
        )

    def _record(
        self,
        *,
        role: Literal["user", "assistant"],
        source: Literal["stt_final", "pipecat_assistant"],
        content: str | None,
        timestamp: str,
        interrupted: bool,
    ) -> None:
        if not self._accepting or content is None or content == "":
            return
        try:
            plaintext = self._bounded_copy(content)
            if not plaintext:
                return
            self._turn_no += 1
            turn_id = self._uuid_factory()
            encrypted = self._keyring.encrypt(
                plaintext,
                aad=b"turn:" + str(turn_id).encode("ascii"),
            )
            event_time = self._parse_timestamp(timestamp)
            occurred_at = max(self._utcnow().astimezone(UTC), event_time)
            payload = TurnUpsertPayloadV1(
                turn_id=turn_id,
                turn_no=self._turn_no,
                role=role,
                source=source,
                crypto_version=1,
                key_version=encrypted.key_version,
                nonce_b64=base64.b64encode(encrypted.nonce).decode("ascii"),
                ciphertext_b64=base64.b64encode(encrypted.ciphertext).decode("ascii"),
                started_at=event_time,
                ended_at=event_time,
                interrupted=interrupted,
            )
            operation = VoiceOperationV1(
                schema_version=1,
                operation_id=self._uuid_factory(),
                deployment_id=self._identity.deployment_id,
                call_id=self._identity.call_id,
                occurred_at=occurred_at,
                kind="turn.upsert",
                payload=payload,
            )
            if not self._writer.try_enqueue_turn(operation):
                self._first_failure.signal("writer_failed")
        except Exception:
            self._first_failure.signal("persistence_failed")

    @staticmethod
    def _bounded_copy(content: str) -> bytes:
        encoded = content.encode("utf-8")
        if len(encoded) <= MAX_TURN_TEXT_BYTES:
            return bytes(encoded)
        return encoded[:MAX_TURN_TEXT_BYTES].decode("utf-8", errors="ignore").encode("utf-8")

    def _parse_timestamp(self, value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
            return parsed.astimezone(UTC)
        except (AttributeError, ValueError):
            return self._utcnow().astimezone(UTC)


class LeaseTerminalizer(Protocol):
    async def terminalize(
        self,
        identity: CallIdentity,
        *,
        status: str,
        reason: str,
    ) -> None: ...


class SessionWriter(ControlWriter, TurnWriter, Protocol):
    fatal_event: asyncio.Event


class CallSessionError(RuntimeError):
    """A constant-safe per-call runtime failure."""


class _TransportEvents(Protocol):
    def add_event_handler(self, event_name: str, handler: object) -> None: ...


SessionTaskFactory = Callable[[Coroutine[Any, Any, Any], str], asyncio.Task[Any]]


@dataclass(frozen=True, slots=True)
class _FrozenTerminalOutcome:
    status: Literal["closed", "failed"]
    reason: str


class _TerminalOutcome:
    """Same-event-loop terminal reason owner with one irreversible freeze."""

    def __init__(self, reason: str) -> None:
        self._reason = reason
        self._frozen: _FrozenTerminalOutcome | None = None

    @property
    def reason(self) -> str:
        frozen = self._frozen
        return frozen.reason if frozen is not None else self._reason

    def note_caller_cancellation(self) -> None:
        if self._frozen is None and self._reason == "closed":
            self._reason = "external_cancel"

    def promote_failure(self, code: str | None) -> None:
        if self._frozen is None and self._reason == "closed" and code is not None:
            self._reason = code

    def freeze(self) -> _FrozenTerminalOutcome:
        if self._frozen is None:
            self._frozen = _FrozenTerminalOutcome(
                status="closed" if self._reason == "closed" else "failed",
                reason=self._reason,
            )
        return self._frozen


class _NativeFatalObservation:
    """First constant-safe fatal evidence observed on Pipecat's native path."""

    _SAFE_CODES = frozenset(
        {"stt_failed", "llm_failed", "tts_failed", "inference_failed", "call_failed"}
    )

    def __init__(self) -> None:
        self._code: str | None = None

    @property
    def code(self) -> str | None:
        return self._code

    def record(self, error: ErrorFrame) -> None:
        if self._code is not None:
            return
        try:
            if not isinstance(error, ErrorFrame) or not error.fatal:
                return
            code = error.error if error.error in self._SAFE_CODES else "call_failed"
        except Exception:
            code = "call_failed"
        self._code = code


class CallSession:
    """Own and supervise one authenticated Pipecat call runtime."""

    def __init__(
        self,
        *,
        identity: CallIdentity,
        manifest: AgentManifestV1,
        profile: QualifiedDeploymentProfileV1,
        services: ServiceBundle,
        writer: SessionWriter,
        keyring: CryptoKeyring,
        recording: RecordingBoundary,
        lease_terminalizer: LeaseTerminalizer,
        idle_timeout_seconds: float,
        cleanup_phase_timeout_seconds: float = 5.0,
        task_factory: SessionTaskFactory | None = None,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if (
            profile.deployment_id != identity.deployment_id
            or not isinstance(idle_timeout_seconds, int | float)
            or isinstance(idle_timeout_seconds, bool)
            or not math.isfinite(idle_timeout_seconds)
            or idle_timeout_seconds <= 0
            or not isinstance(cleanup_phase_timeout_seconds, int | float)
            or isinstance(cleanup_phase_timeout_seconds, bool)
            or not math.isfinite(cleanup_phase_timeout_seconds)
            or cleanup_phase_timeout_seconds <= 0
        ):
            raise ValueError("call_session_config_invalid")
        self._identity = identity
        self._manifest = manifest
        self._profile = profile
        self._services = services
        self._writer = writer
        self._keyring = keyring
        self._recording = recording
        self._lease_terminalizer = lease_terminalizer
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._cleanup_phase_timeout_seconds = float(cleanup_phase_timeout_seconds)
        self._task_factory = task_factory or self._default_task_factory
        self._utcnow = utcnow
        self._uuid_factory = uuid_factory
        self._run_started = False
        self._active_runner: WorkerRunner | None = None
        self._drain_requested = False
        self._drain_claimed = False
        self._drain_lock = asyncio.Lock()

    async def run(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        if self._run_started:
            raise CallSessionError("call_session_already_run")
        self._run_started = True
        self._verify_handshake(handshake)

        first_failure = FirstFailure(shared_failure_event=self._writer.fatal_event)
        native_fatal = _NativeFatalObservation()
        controller = DisclosureController(
            identity=self._identity,
            writer=self._writer,
            first_failure=first_failure,
            recording=self._recording,
            recording_enabled=self._manifest.recording_mode != "off",
            recording_required=self._manifest.recording_required,
            mark_timeout_seconds=self._profile.disclosure_mark_timeout_ms / 1000,
            utcnow=self._utcnow,
            uuid_factory=self._uuid_factory,
        )
        recorder = TurnRecorder(
            identity=self._identity,
            writer=self._writer,
            keyring=self._keyring,
            first_failure=first_failure,
            uuid_factory=self._uuid_factory,
            utcnow=self._utcnow,
        )
        pipeline: ObservedPipeline | None = None
        runtime: CallRuntime | None = None
        runner_task: asyncio.Task[None] | None = None
        failure_task: asyncio.Task[str] | None = None
        reason: str | None = None
        cancellation: asyncio.CancelledError | None = None

        try:
            handshake.audio_admission.bind(controller.is_active)
            self._register_transport_handlers(
                transport=cast(_TransportEvents, handshake.transport),
                controller=controller,
                first_failure=first_failure,
            )
            pipeline = build_pipeline(
                transport=handshake.transport,
                services=self._services,
                controller=controller,
                turn_recorder=recorder,
                first_failure=first_failure,
            )
            runtime = build_runtime(
                pipeline=pipeline,
                first_failure=first_failure,
                greeting=self._manifest.greeting,
                mark_name=controller.mark_name,
                idle_timeout_seconds=self._idle_timeout_seconds,
            )

            async def observe_native_fatal(
                _worker: object,
                error: ErrorFrame,
            ) -> None:
                native_fatal.record(error)

            runtime.worker.add_event_handler(
                "on_pipeline_error",
                observe_native_fatal,
            )
            await runtime.runner.add_workers(runtime.worker)
            runner_task = cast(
                asyncio.Task[None],
                self._create_session_task(
                    runtime.runner.run(auto_end=True),
                    "call-runner",
                ),
            )

            await asyncio.sleep(0)
            self._active_runner = runtime.runner
            await self._replay_pending_drain()
            failure_task = cast(
                asyncio.Task[str],
                self._create_session_task(
                    first_failure.wait(),
                    "call-first-failure",
                ),
            )
            done, _pending = await asyncio.wait(
                (runner_task, failure_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_task in done:
                reason = failure_task.result()
            else:
                await runner_task
                reason = first_failure.code or native_fatal.code
                if reason is None and not controller.is_active():
                    reason = first_failure.code or "call_failed"
        except asyncio.CancelledError as error:
            cancellation = error
            reason = "external_cancel"
        except Exception:
            reason = first_failure.code or "call_failed"
            first_failure.signal(reason)
        terminal_outcome = _TerminalOutcome(reason or "closed")
        cleanup = self._cleanup_owned_state(
            controller=controller,
            recorder=recorder,
            first_failure=first_failure,
            transport=handshake.transport,
            pipeline=pipeline,
            runtime=runtime,
            runner_task=runner_task,
            failure_task=failure_task,
            terminal_outcome=terminal_outcome,
            cancel_continuations=cancellation is not None,
        )
        cancellation = await self._await_owned_cleanup(
            cleanup,
            cancellation=cancellation,
            terminal_outcome=terminal_outcome,
            first_failure=first_failure,
        )
        self._active_runner = None
        if cancellation is not None:
            raise cancellation
        if terminal_outcome.reason != "closed":
            raise CallSessionError(terminal_outcome.reason)
        final_error = first_failure.code
        if final_error is not None:
            raise CallSessionError(final_error)

    async def request_drain(self) -> None:
        """End the active per-call runner through its public cancellation surface."""

        runner: WorkerRunner | None = None
        async with self._drain_lock:
            self._drain_requested = True
            if self._active_runner is not None and not self._drain_claimed:
                self._drain_claimed = True
                runner = self._active_runner
        if runner is not None:
            await runner.cancel(reason="drain")

    async def _replay_pending_drain(self) -> None:
        runner: WorkerRunner | None = None
        async with self._drain_lock:
            if (
                self._drain_requested
                and not self._drain_claimed
                and self._active_runner is not None
            ):
                self._drain_claimed = True
                runner = self._active_runner
        if runner is not None:
            await runner.cancel(reason="drain")

    @staticmethod
    def _default_task_factory(
        coroutine: Coroutine[Any, Any, Any],
        name: str,
    ) -> asyncio.Task[Any]:
        return asyncio.create_task(coroutine, name=name)

    def _create_session_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        name: str,
    ) -> asyncio.Task[Any]:
        try:
            return self._task_factory(coroutine, name)
        except BaseException:
            coroutine.close()
            raise

    @staticmethod
    def _register_transport_handlers(
        *,
        transport: _TransportEvents,
        controller: DisclosureController,
        first_failure: FirstFailure,
    ) -> None:
        def handler(code: str) -> Callable[[object, object], Awaitable[None]]:
            async def fail_transport(_transport: object, _websocket: object) -> None:
                first_failure.signal(code)
                try:
                    await controller.abort(code)
                except asyncio.CancelledError:
                    first_failure.signal(code)
                except Exception:
                    first_failure.signal(code)

            return fail_transport

        transport.add_event_handler(
            "on_client_disconnected",
            handler("transport_disconnected"),
        )
        transport.add_event_handler(
            "on_session_timeout",
            handler("transport_session_timeout"),
        )

    async def _await_owned_cleanup(
        self,
        cleanup: Coroutine[Any, Any, None],
        *,
        cancellation: asyncio.CancelledError | None,
        terminal_outcome: _TerminalOutcome,
        first_failure: FirstFailure,
    ) -> asyncio.CancelledError | None:
        caller_task = asyncio.current_task()
        cleanup_task = asyncio.create_task(cleanup, name="call-cleanup")
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as error:
                caller_is_cancelling = (
                    caller_task is not None and caller_task.cancelling() > 0
                )
                if caller_is_cancelling and cancellation is None:
                    cancellation = error
                    terminal_outcome.note_caller_cancellation()
                if cleanup_task.done() and cleanup_task.cancelled():
                    first_failure.signal("call_failed")
                    self._promote_cleanup_failure(terminal_outcome, first_failure)
                    break
            except Exception:
                first_failure.signal("call_failed")
                self._promote_cleanup_failure(terminal_outcome, first_failure)
                break
        if cleanup_task.cancelled():
            first_failure.signal("call_failed")
            self._promote_cleanup_failure(terminal_outcome, first_failure)
        elif cleanup_task.done():
            try:
                cleanup_task.result()
            except Exception:
                first_failure.signal("call_failed")
                self._promote_cleanup_failure(terminal_outcome, first_failure)
        return cancellation

    async def _cleanup_owned_state(
        self,
        *,
        controller: DisclosureController,
        recorder: TurnRecorder,
        first_failure: FirstFailure,
        transport: PipelineTransport,
        pipeline: ObservedPipeline | None,
        runtime: CallRuntime | None,
        runner_task: asyncio.Task[None] | None,
        failure_task: asyncio.Task[str] | None,
        terminal_outcome: _TerminalOutcome,
        cancel_continuations: bool,
    ) -> None:
        await self._attempt(
            lambda: controller.terminalize_and_join(
                cancel_continuations=cancel_continuations
            ),
            first_failure,
            "call_failed",
        )

        if runtime is not None and runner_task is not None:
            if not runner_task.done():
                await asyncio.sleep(0)
                if terminal_outcome.reason != "closed":
                    await self._attempt(
                        lambda: self._queue_interruption_and_wait(runtime),
                        first_failure,
                        "call_failed",
                    )
                await self._attempt(
                    lambda: runtime.runner.cancel(
                        reason=(
                            "external_cancel"
                            if terminal_outcome.reason == "external_cancel"
                            else "local_failure"
                        )
                    ),
                    first_failure,
                    "call_failed",
                )
            await self._attempt(
                lambda: self._join_task(runner_task),
                first_failure,
                "call_failed",
            )
            if not runtime.worker.has_finished():
                await self._attempt(
                    runtime.worker.pipeline.cleanup,
                    first_failure,
                    "pipeline_cleanup_failed",
                )
                await self._attempt(
                    runtime.worker.cleanup,
                    first_failure,
                    "pipeline_cleanup_failed",
                )
        elif runtime is not None:
            await self._attempt(
                runtime.pipeline.cleanup,
                first_failure,
                "pipeline_cleanup_failed",
            )
            await self._attempt(
                runtime.worker.cleanup,
                first_failure,
                "pipeline_cleanup_failed",
            )
            await self._attempt(
                runtime.runner.cleanup,
                first_failure,
                "call_failed",
            )
            await self._attempt(
                runtime.runner.bus.stop,
                first_failure,
                "call_failed",
            )
            await self._attempt(
                runtime.runner.bus.cleanup,
                first_failure,
                "call_failed",
            )
        elif pipeline is not None:
            await self._attempt(
                pipeline.cleanup,
                first_failure,
                "pipeline_cleanup_failed",
            )
        else:
            processor_factories: tuple[Callable[[], FrameProcessor], ...] = (
                transport.input,
                lambda: self._services.stt,
                lambda: self._services.llm,
                lambda: self._services.tts,
                transport.output,
            )
            cleaned_processor_ids: set[int] = set()
            for processor_factory in processor_factories:
                try:
                    processor = processor_factory()
                except asyncio.CancelledError:
                    first_failure.signal("pipeline_cleanup_failed")
                    continue
                except Exception:
                    first_failure.signal("pipeline_cleanup_failed")
                    continue
                processor_id = id(processor)
                if processor_id in cleaned_processor_ids:
                    continue
                cleaned_processor_ids.add(processor_id)
                await self._attempt(
                    processor.cleanup,
                    first_failure,
                    "pipeline_cleanup_failed",
                )

        if failure_task is not None:
            failure_task.cancel()
            await asyncio.gather(failure_task, return_exceptions=True)

        try:
            recorder.close()
        except asyncio.CancelledError:
            first_failure.signal("persistence_failed")
        except Exception:
            first_failure.signal("persistence_failed")
        await self._attempt(
            lambda: controller.cleanup_termination(terminal_outcome.reason),
            first_failure,
            "recording_cleanup_failed",
        )
        await self._attempt(self._services.aclose, first_failure, "service_close_failed")
        self._promote_cleanup_failure(terminal_outcome, first_failure)
        await self._finish_durable_boundaries(
            first_failure=first_failure,
            terminal_outcome=terminal_outcome,
            disclosure_completed=controller.disclosure_completed,
        )

    @staticmethod
    async def _join_task(task: asyncio.Task[None]) -> None:
        await task

    @staticmethod
    async def _queue_interruption_and_wait(runtime: CallRuntime) -> None:
        await runtime.request_clear()

    def _verify_handshake(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        if (
            not isinstance(handshake, AuthenticatedTelnyxHandshake)
            or handshake.call_data.call_id != self._identity.telnyx_call_control_id
            or handshake.call_data.stream_id != self._identity.stream_id
            or handshake.lease_claim is not self._identity.lease_claim
        ):
            raise CallSessionError("call_identity_mismatch")

    async def _finish_durable_boundaries(
        self,
        *,
        first_failure: FirstFailure,
        terminal_outcome: _TerminalOutcome,
        disclosure_completed: bool,
    ) -> None:
        self._promote_cleanup_failure(terminal_outcome, first_failure)
        frozen = terminal_outcome.freeze()
        if not self._writer.fatal_event.is_set():
            await self._attempt(
                lambda: self._commit_terminal_call(
                    status=frozen.status,
                    reason=frozen.reason,
                    disclosure_completed=disclosure_completed,
                ),
                first_failure,
                "persistence_failed",
            )
        await self._attempt(
            lambda: self._lease_terminalizer.terminalize(
                self._identity,
                status=frozen.status,
                reason=frozen.reason,
            ),
            first_failure,
            "lease_terminalization_failed",
        )

    @staticmethod
    def _promote_cleanup_failure(
        terminal_outcome: _TerminalOutcome,
        first_failure: FirstFailure,
    ) -> None:
        terminal_outcome.promote_failure(first_failure.code)

    async def _commit_terminal_call(
        self,
        *,
        status: Literal["closed", "failed"],
        reason: str,
        disclosure_completed: bool,
    ) -> None:
        ended_at = self._utcnow().astimezone(UTC)
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=self._uuid_factory(),
            deployment_id=self._identity.deployment_id,
            call_id=self._identity.call_id,
            occurred_at=ended_at,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=self._identity.telnyx_call_control_id,
                telnyx_call_leg_id=self._identity.telnyx_call_leg_id,
                telnyx_call_session_id=self._identity.telnyx_call_session_id,
                status=status,
                disclosure_state="completed" if disclosure_completed else "failed",
                started_at=self._identity.started_at,
                ended_at=ended_at,
                end_reason=reason,
                retention_until=self._identity.retention_until,
            ),
        )
        await self._writer.commit_control(
            PersistenceCommand("outbox", {"operation": operation}, None)
        )

    async def _attempt(
        self,
        phase: Callable[[], Awaitable[None]],
        first_failure: FirstFailure,
        code: str,
    ) -> None:
        try:
            async with asyncio.timeout(self._cleanup_phase_timeout_seconds):
                await phase()
        except asyncio.CancelledError:
            first_failure.signal(code)
        except TimeoutError:
            first_failure.signal(code)
        except Exception:
            first_failure.signal(code)


class DisclosureController:
    """Call-owned disclosure, durable commit, and recording gate."""

    def __init__(
        self,
        *,
        identity: CallIdentity,
        writer: ControlWriter,
        first_failure: FirstFailure,
        recording: RecordingBoundary,
        recording_enabled: bool,
        recording_required: bool,
        mark_timeout_seconds: float,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if (
            type(recording_enabled) is not bool
            or type(recording_required) is not bool
            or recording_required and not recording_enabled
            or not isinstance(mark_timeout_seconds, int | float)
            or isinstance(mark_timeout_seconds, bool)
            or mark_timeout_seconds <= 0
        ):
            raise ValueError("disclosure_config_invalid")
        self._identity = identity
        self._writer = writer
        self._first_failure = first_failure
        self._recording = recording
        self._recording_enabled = recording_enabled
        self._recording_required = recording_required
        self._mark_timeout_seconds = float(mark_timeout_seconds)
        self._utcnow = utcnow
        self._uuid_factory = uuid_factory
        self.disclosure_generation = uuid_factory()
        self.mark_name = f"pv0-disclosure-{uuid_factory().hex}"
        self._lock = asyncio.Lock()
        self.state = DisclosureState.PLAYING
        self.disclosure_completed = False
        self.recording_may_be_active = False
        self._input_closed = False
        self._audio_observed = False
        self._timeout_task: asyncio.Task[None] | None = None
        self._continuations: set[asyncio.Task[None]] = set()

    @property
    def pending_task_count(self) -> int:
        return sum(not task.done() for task in self._continuations)

    def is_active(self) -> bool:
        return (
            self._first_failure.code is None
            and self.state is DisclosureState.ACTIVE
            and not self._input_closed
        )

    async def note_disclosure_audio(self) -> None:
        async with self._lock:
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                return
            if self.state is DisclosureState.PLAYING:
                self._audio_observed = True

    async def arm_expected_mark(self) -> bool:
        failed = False
        async with self._lock:
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                return False
            if self.state is not DisclosureState.PLAYING or not self._audio_observed:
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                    self._input_closed = True
                failed = True
            else:
                self.state = DisclosureState.MARK_PENDING
                return True
        if failed:
            self._first_failure.signal("disclosure_failed")
        return False

    async def mark_forwarded(self) -> None:
        async with self._lock:
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                return
            if self.state is not DisclosureState.MARK_PENDING or self._timeout_task is not None:
                return
            task = asyncio.create_task(self._mark_timeout(), name="disclosure-mark-timeout")
            self._timeout_task = task
            self._track(task)

    async def accept_mark(self, mark_name: str) -> bool:
        try:
            received_mark = mark_name.encode("utf-8")
        except (AttributeError, UnicodeEncodeError):
            return False
        async with self._lock:
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                timeout_task = self._timeout_task
                self._timeout_task = None
                if timeout_task is not None:
                    timeout_task.cancel()
                return False
            if (
                not isinstance(mark_name, str)
                or not hmac.compare_digest(received_mark, self.mark_name.encode("ascii"))
                or self.state is not DisclosureState.MARK_PENDING
            ):
                return False
            self.state = DisclosureState.ACK_COMMITTING
            timeout_task = self._timeout_task
            self._timeout_task = None
            if timeout_task is not None:
                timeout_task.cancel()
            task = asyncio.create_task(
                self._run_owned_continuation(
                    lambda: self._complete_disclosure(self._utcnow()),
                    "disclosure_commit_failed",
                ),
                name="disclosure-ack-continuation",
            )
            self._track(task)
            return True

    async def abort(self, code: str) -> None:
        timeout_task: asyncio.Task[None] | None
        async with self._lock:
            self._input_closed = True
            if self.state is not DisclosureState.ACTIVE:
                self.state = DisclosureState.ABORTED
            timeout_task = self._timeout_task
            self._timeout_task = None
            if timeout_task is not None:
                timeout_task.cancel()
        self._first_failure.signal(code)

    async def join_continuations(self) -> None:
        while True:
            pending = [task for task in self._continuations if not task.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def cancel_and_join_continuations(self) -> None:
        pending = [task for task in self._continuations if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def terminalize_and_join(self, *, cancel_continuations: bool) -> None:
        async with self._lock:
            self._input_closed = True
            if self.state is not DisclosureState.ACTIVE:
                self.state = DisclosureState.ABORTED
            timeout_task = self._timeout_task
            self._timeout_task = None
            if timeout_task is not None:
                timeout_task.cancel()
        if cancel_continuations:
            await self.cancel_and_join_continuations()
        else:
            await self.join_continuations()

    async def cleanup_termination(self, reason: str) -> None:
        try:
            await self._recording.cleanup(
                self._identity,
                recording_may_be_active=self.recording_may_be_active,
                reason=reason,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._first_failure.signal("recording_cleanup_failed")

    def _track(self, task: asyncio.Task[None]) -> None:
        self._continuations.add(task)
        task.add_done_callback(self._consume_continuation_result)

    def _consume_continuation_result(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            self._first_failure.signal("disclosure_failed")
        finally:
            self._continuations.discard(task)

    async def _run_owned_continuation(
        self,
        continuation: Callable[[], Awaitable[None]],
        failure_code: str,
    ) -> None:
        try:
            await continuation()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.abort(failure_code)

    async def _mark_timeout(self) -> None:
        try:
            await asyncio.sleep(self._mark_timeout_seconds)
            async with self._lock:
                if self.state is not DisclosureState.MARK_PENDING:
                    return
                self.state = DisclosureState.ABORTED
                self._input_closed = True
                self._timeout_task = None
            self._first_failure.signal("disclosure_timeout")
        except asyncio.CancelledError:
            raise

    async def _complete_disclosure(self, acknowledged_at: datetime) -> None:
        async with self._lock:
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                return

        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=self._uuid_factory(),
            deployment_id=self._identity.deployment_id,
            call_id=self._identity.call_id,
            occurred_at=acknowledged_at,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=self._identity.telnyx_call_control_id,
                telnyx_call_leg_id=self._identity.telnyx_call_leg_id,
                telnyx_call_session_id=self._identity.telnyx_call_session_id,
                status="active",
                disclosure_state="completed",
                started_at=self._identity.started_at,
                ended_at=None,
                end_reason=None,
                retention_until=self._identity.retention_until,
            ),
        )
        try:
            await self._writer.commit_control(
                PersistenceCommand("outbox", {"operation": operation}, None)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._lock:
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                    self._input_closed = True
            self._first_failure.signal("disclosure_commit_failed")
            return

        async with self._lock:
            self.disclosure_completed = True
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                return
            if self.state is not DisclosureState.ACK_COMMITTING:
                return
            self.state = DisclosureState.DISCLOSURE_DURABLE
            if not self._recording_enabled:
                self.state = DisclosureState.ACTIVE
                return
            self.state = DisclosureState.RECORDING_STARTING
            self.recording_may_be_active = True

        result: RecordingStartResult
        try:
            received = await self._recording.start(self._identity)
            result = (
                received
                if isinstance(received, RecordingStartResult)
                else RecordingStartResult(RecordingStartState.INDETERMINATE)
            )
        except asyncio.CancelledError:
            async with self._lock:
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                    self._input_closed = True
            raise
        except Exception:
            result = RecordingStartResult(RecordingStartState.INDETERMINATE)

        failed = False
        async with self._lock:
            if result.state is RecordingStartState.DEFINITELY_NOT_STARTED:
                self.recording_may_be_active = False
            if self._first_failure.code is not None:
                self._input_closed = True
                if self.state is not DisclosureState.ACTIVE:
                    self.state = DisclosureState.ABORTED
                return
            if self.state is not DisclosureState.RECORDING_STARTING:
                return
            if result.state is RecordingStartState.STARTED:
                self.state = DisclosureState.ACTIVE
            elif result.state is RecordingStartState.DEFINITELY_NOT_STARTED:
                if self._recording_required:
                    self.state = DisclosureState.ABORTED
                    self._input_closed = True
                    failed = True
                else:
                    self.state = DisclosureState.ACTIVE
            elif not self._recording_required and result.gate_may_open:
                self.state = DisclosureState.ACTIVE
            else:
                self.state = DisclosureState.ABORTED
                self._input_closed = True
                failed = True
        if failed:
            self._first_failure.signal("recording_failed")


__all__ = [
    "CallIdentity",
    "CallSession",
    "CallSessionError",
    "DisclosureController",
    "DisclosureState",
    "RecordingBoundary",
    "RecordingStartResult",
    "RecordingStartState",
    "MAX_TURN_TEXT_BYTES",
    "ServiceBundle",
    "ServiceLifecycleError",
    "TurnRecorder",
]
