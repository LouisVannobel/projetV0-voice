"""Per-call lifecycle, disclosure, persistence, and recording ownership."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from importlib.metadata import version
from typing import Literal, Protocol, cast
from uuid import UUID, uuid4

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, TurnUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.pipeline import FirstFailure, build_pipeline, build_runtime
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
            close_jobs: list[tuple[str, asyncio.Task[None]]] = []
            if not self._stt_closed:
                close_jobs.append(
                    (
                        "stt",
                        asyncio.create_task(
                            self._bounded_close(self.stt_http_client.aclose),
                            name="close-stt-http-client",
                        ),
                    )
                )
            if not self._llm_closed:
                close_jobs.append(
                    (
                        "llm",
                        asyncio.create_task(
                            self._bounded_close(self._close_pinned_llm_client),
                            name="close-llm-client",
                        ),
                    )
                )
            if not close_jobs:
                return
            tasks = [task for _, task in close_jobs]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            failed = False
            for (resource, _task), result in zip(close_jobs, results, strict=True):
                if isinstance(result, BaseException):
                    failed = True
                elif resource == "stt":
                    self._stt_closed = True
                else:
                    self._llm_closed = True
            if failed:
                raise ServiceLifecycleError("service_close_failed")

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
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if (
            profile.deployment_id != identity.deployment_id
            or not isinstance(idle_timeout_seconds, int | float)
            or isinstance(idle_timeout_seconds, bool)
            or not math.isfinite(idle_timeout_seconds)
            or idle_timeout_seconds <= 0
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
        self._utcnow = utcnow
        self._uuid_factory = uuid_factory
        self._run_started = False
        self._active_runner: WorkerRunner | None = None

    async def run(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        if self._run_started:
            raise CallSessionError("call_session_already_run")
        self._run_started = True
        self._verify_handshake(handshake)

        first_failure = FirstFailure(shared_failure_event=self._writer.fatal_event)
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
        try:
            handshake.audio_admission.bind(controller.is_active)
        except Exception:
            first_failure.signal("call_failed")
            await self._cleanup_without_runtime(
                controller=controller,
                recorder=recorder,
                first_failure=first_failure,
                reason="call_failed",
            )
            raise CallSessionError("call_failed") from None

        pipeline = build_pipeline(
            transport=handshake.transport,
            services=self._services,
            controller=controller,
            turn_recorder=recorder,
            first_failure=first_failure,
        )
        runtime = await build_runtime(
            pipeline=pipeline,
            first_failure=first_failure,
            greeting=self._manifest.greeting,
            mark_name=controller.mark_name,
            idle_timeout_seconds=self._idle_timeout_seconds,
        )
        self._active_runner = runtime.runner

        runner_task = asyncio.create_task(
            runtime.runner.run(auto_end=True),
            name="call-runner",
        )
        failure_task = asyncio.create_task(
            first_failure.wait(),
            name="call-first-failure",
        )
        reason: str | None = None
        external_cancelled = False
        cancellation: asyncio.CancelledError | None = None
        try:
            done, _pending = await asyncio.wait(
                (runner_task, failure_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_task in done:
                reason = failure_task.result()
                await controller.abort(reason)
                await runtime.runner.cancel(reason="local_failure")
                await runner_task
            else:
                await runner_task
                if not controller.is_active():
                    reason = first_failure.code or "call_failed"
        except asyncio.CancelledError as error:
            external_cancelled = True
            cancellation = error
            reason = "external_cancel"
            await controller.terminalize_and_join(cancel_continuations=True)
            await runtime.runner.cancel(reason="external_cancel")
            try:
                await runner_task
            except Exception:
                first_failure.signal("call_failed")
        except Exception:
            reason = first_failure.code or "call_failed"
            first_failure.signal(reason)
            await runtime.runner.cancel(reason="local_failure")
            with contextlib.suppress(Exception):
                await runner_task
        finally:
            failure_task.cancel()
            await asyncio.gather(failure_task, return_exceptions=True)

        final_reason = reason or "closed"
        await self._cleanup_after_runtime(
            controller=controller,
            recorder=recorder,
            first_failure=first_failure,
            reason=final_reason,
            cancel_continuations=external_cancelled,
        )
        self._active_runner = None
        if cancellation is not None:
            raise cancellation
        if reason is not None:
            raise CallSessionError(reason)

    async def request_drain(self) -> None:
        """End the active per-call runner through its public cancellation surface."""

        runner = self._active_runner
        if runner is not None:
            await runner.cancel(reason="drain")

    def _verify_handshake(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        if (
            not isinstance(handshake, AuthenticatedTelnyxHandshake)
            or handshake.call_data.call_id != self._identity.telnyx_call_control_id
            or handshake.call_data.stream_id != self._identity.stream_id
            or handshake.lease_claim is not self._identity.lease_claim
        ):
            raise CallSessionError("call_identity_mismatch")

    async def _cleanup_without_runtime(
        self,
        *,
        controller: DisclosureController,
        recorder: TurnRecorder,
        first_failure: FirstFailure,
        reason: str,
    ) -> None:
        await controller.terminalize_and_join(cancel_continuations=True)
        recorder.close()
        await controller.cleanup_termination(reason)
        await self._attempt(self._services.aclose, first_failure, "service_close_failed")
        await self._finish_durable_boundaries(
            first_failure=first_failure,
            reason=reason,
            disclosure_completed=controller.disclosure_completed,
        )

    async def _cleanup_after_runtime(
        self,
        *,
        controller: DisclosureController,
        recorder: TurnRecorder,
        first_failure: FirstFailure,
        reason: str,
        cancel_continuations: bool,
    ) -> None:
        await self._attempt(
            lambda: controller.terminalize_and_join(
                cancel_continuations=cancel_continuations
            ),
            first_failure,
            "call_failed",
        )
        recorder.close()
        await self._attempt(
            lambda: controller.cleanup_termination(reason),
            first_failure,
            "recording_cleanup_failed",
        )
        await self._attempt(self._services.aclose, first_failure, "service_close_failed")
        await self._finish_durable_boundaries(
            first_failure=first_failure,
            reason=reason,
            disclosure_completed=controller.disclosure_completed,
        )

    async def _finish_durable_boundaries(
        self,
        *,
        first_failure: FirstFailure,
        reason: str,
        disclosure_completed: bool,
    ) -> None:
        status: Literal["closed", "failed"] = "closed" if reason == "closed" else "failed"
        if not self._writer.fatal_event.is_set():
            await self._attempt(
                lambda: self._commit_terminal_call(
                    status=status,
                    reason=reason,
                    disclosure_completed=disclosure_completed,
                ),
                first_failure,
                "persistence_failed",
            )
        await self._attempt(
            lambda: self._lease_terminalizer.terminalize(
                self._identity,
                status=status,
                reason=reason,
            ),
            first_failure,
            "lease_terminalization_failed",
        )

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

    @staticmethod
    async def _attempt(
        phase: Callable[[], Awaitable[None]],
        first_failure: FirstFailure,
        code: str,
    ) -> None:
        try:
            await phase()
        except asyncio.CancelledError:
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
        return self.state is DisclosureState.ACTIVE and not self._input_closed

    async def note_disclosure_audio(self) -> None:
        async with self._lock:
            if self.state is DisclosureState.PLAYING:
                self._audio_observed = True

    async def arm_expected_mark(self) -> bool:
        failed = False
        async with self._lock:
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
            if self.state is not DisclosureState.MARK_PENDING or self._timeout_task is not None:
                return
            task = asyncio.create_task(self._mark_timeout(), name="disclosure-mark-timeout")
            self._timeout_task = task
            self._track(task)

    async def accept_mark(self, mark_name: str) -> bool:
        async with self._lock:
            if (
                not isinstance(mark_name, str)
                or not hmac.compare_digest(mark_name, self.mark_name)
                or self.state is not DisclosureState.MARK_PENDING
            ):
                return False
            self.state = DisclosureState.ACK_COMMITTING
            timeout_task = self._timeout_task
            self._timeout_task = None
            if timeout_task is not None:
                timeout_task.cancel()
            task = asyncio.create_task(
                self._complete_disclosure(self._utcnow()),
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
        task.add_done_callback(self._continuations.discard)

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
