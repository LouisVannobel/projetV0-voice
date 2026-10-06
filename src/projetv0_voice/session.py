"""Per-call lifecycle, disclosure, persistence, and recording ownership."""

from __future__ import annotations

import asyncio
import base64
import hmac
import math
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from importlib.metadata import version
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid4

from pipecat.frames.frames import ErrorFrame, FunctionCallResultProperties, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.runner import WorkerRunner

from projetv0_voice.admission import (
    CallGenerationHandle,
    ProcessLeaseClaim,
    TerminalAuthority,
    TerminalProposal,
)
from projetv0_voice.audio_capture import AudioCaptureSummary, AudioFinishReason, LocalAudioCapture
from projetv0_voice.audio_contract import BeginCallSnapshotV2, VoiceOperationV2
from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimeMetrics, _CallMetricLease
from projetv0_voice.models import (
    BeginCallSnapshotV1,
    CallUpsertPayloadV1,
    DisclosureEvidenceV1,
    MessageResultV1,
    RoutingV1,
    TurnUpsertPayloadV1,
    VoiceOperationV1,
    _e164,
)
from projetv0_voice.persistence.business_contract import validate_turn_text
from projetv0_voice.persistence.business_result import infer_partial_result
from projetv0_voice.persistence.commands import (
    FatalPersistenceError,
    PersistenceCommand,
    PersistenceError,
)
from projetv0_voice.persistence.writer import AudioChoiceFacts, PersistenceWriter
from projetv0_voice.pipeline import (
    SPARRA_DISCLOSURE,
    SPARRA_RECORDING_DISCLOSURE,
    CallRuntime,
    FirstFailure,
    ObservedPipeline,
    PipelineTransport,
    _CallObservers,
    build_pipeline,
    build_runtime,
)
from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    QualifiedDeploymentProfileV1,
    RuntimeDeploymentProfileV1,
)
from projetv0_voice.telnyx.frames import TelnyxInputDTMFFrame
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake


class DisclosureState(Enum):
    PLAYING = auto()
    MARK_PENDING = auto()
    ACK_COMMITTING = auto()
    DISCLOSURE_DURABLE = auto()
    WAITING_CHOICE = auto()
    CHOICE_COMMITTING = auto()
    GATE_COMMITTING = auto()
    LOCAL_AUDIO_STARTING = auto()
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
    generation: CallGenerationHandle = field(repr=False)
    lease_claim: ProcessLeaseClaim = field(repr=False)
    deployment_id: str = field(repr=False)
    telnyx_call_control_id: str = field(repr=False)
    telnyx_call_leg_id: str | None = field(repr=False)
    telnyx_call_session_id: str | None = field(repr=False)
    stream_id: str = field(repr=False)
    started_at: datetime
    retention_until: datetime
    routing: RoutingV1 | None = field(default=None, repr=False)
    begin_snapshot: BeginCallSnapshotV1 | BeginCallSnapshotV2 | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        required_text = (self.deployment_id, self.telnyx_call_control_id, self.stream_id)
        if (
            not isinstance(self.call_id, UUID)
            or not isinstance(self.generation, CallGenerationHandle)
            or not isinstance(self.lease_claim, ProcessLeaseClaim)
            or self.generation.call_control_id != self.telnyx_call_control_id
            or self.generation.generation != self.lease_claim.generation
            or self.lease_claim.call_control_id != self.telnyx_call_control_id
            or self.lease_claim.call_id != self.call_id
            or self.lease_claim.claimed_at != self.started_at
            or any(not isinstance(value, str) or not value for value in required_text)
            or self.telnyx_call_leg_id is not None
            and (type(self.telnyx_call_leg_id) is not str or not self.telnyx_call_leg_id)
            or self.telnyx_call_session_id is not None
            and (
                type(self.telnyx_call_session_id) is not str
                or not self.telnyx_call_session_id
            )
            or self.started_at.tzinfo is None
            or self.started_at.utcoffset() is None
            or self.retention_until.tzinfo is None
            or self.retention_until.utcoffset() is None
            or self.retention_until <= self.started_at
        ):
            raise ValueError("call_identity_invalid")
        if self.begin_snapshot is not None and (
            self.begin_snapshot.call_id != self.call_id
            or self.begin_snapshot.retention_until != self.retention_until
            or self.routing is not None and (
                self.routing.admitted_at != self.started_at
                or self.routing.admitted_at + timedelta(days=30) != self.retention_until
                or self.routing.telnyx_call_control_id != self.telnyx_call_control_id
                or self.routing.telnyx_call_leg_id != self.telnyx_call_leg_id
                or self.routing.telnyx_call_session_id != self.telnyx_call_session_id
            )
        ):
            raise ValueError("call_identity_mismatch")

    def __repr__(self) -> str:
        return "CallIdentity()"


class ControlWriter(Protocol):
    async def commit_control(self, command: PersistenceCommand) -> None: ...

    async def publish_control_v2(
        self, operation: VoiceOperationV2, *, generation: UUID
    ) -> None: ...

    async def commit_audio_choice(
        self, call_id: UUID, *, generation: UUID, choice: Literal["accept", "off"],
        occurred_at: datetime | None,
    ) -> AudioChoiceFacts: ...


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


_CloseOutcome = Literal[True] | BaseException


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
            close_jobs: list[asyncio.Task[_CloseOutcome]] = []
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
                await asyncio.gather(*close_jobs)
                raise cancellation
            results = await asyncio.gather(*close_jobs)

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

    async def _close_stt_client(self) -> _CloseOutcome:
        try:
            await self._bounded_close(self.stt_http_client.aclose)
        except BaseException as error:
            return error
        self._stt_closed = True
        return True

    async def _close_llm_client(self) -> _CloseOutcome:
        try:
            await self._bounded_close(self._close_pinned_llm_client)
        except BaseException as error:
            return error
        self._llm_closed = True
        return True

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
    def try_enqueue_turn(self, operation: VoiceOperationV1, *, truncated: bool = False) -> bool: ...

    def try_enqueue_turn_v2(
        self, operation: VoiceOperationV2, *, generation: UUID, truncated: bool = False,
    ) -> bool: ...


class TurnRecorder:
    """Synchronous, atomic turn encryption and shared-writer admission."""

    def __init__(
        self,
        *,
        identity: CallIdentity,
        writer: TurnWriter,
        keyring: CryptoKeyring,
        first_failure: FirstFailure,
        runtime_metrics: RuntimeMetrics,
        uuid_factory: Callable[[], UUID] = uuid4,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._identity = identity
        self._writer = writer
        self._keyring = keyring
        self._first_failure = first_failure
        self._runtime_metrics = runtime_metrics
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
        accepted = False
        capture_id = None
        try:
            if self._identity.routing is not None:
                self._turn_no += 1
                capture_id = self._uuid_factory()
                accepted = True
            bound = 16384 if self._identity.routing is not None else MAX_TURN_TEXT_BYTES
            encoded = content.encode("utf-8")
            plaintext = encoded[:bound].decode("utf-8", errors="ignore").encode("utf-8")
            if not plaintext:
                return
            accepted = True
            if capture_id is None:
                self._turn_no += 1
                capture_id = self._uuid_factory()
            else:
                validate_turn_text(plaintext.decode("utf-8"))
            turn_id = capture_id
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
            if isinstance(getattr(self._identity, "begin_snapshot", None), BeginCallSnapshotV2):
                fresh = VoiceOperationV2(
                    schema_version=2, operation_id=self._uuid_factory(),
                    deployment_id=self._identity.deployment_id, call_id=self._identity.call_id,
                    occurred_at=occurred_at, kind="turn.upsert", payload=payload,
                )
                enqueued = self._writer.try_enqueue_turn_v2(
                    fresh, generation=self._identity.generation.generation,
                    truncated=len(encoded) > bound,
                )
            else:
                operation = VoiceOperationV1(
                    schema_version=1, operation_id=self._uuid_factory(),
                    deployment_id=self._identity.deployment_id, call_id=self._identity.call_id,
                    occurred_at=occurred_at, kind="turn.upsert", payload=payload,
                )
                enqueued = (
                    self._writer.try_enqueue_turn(operation, truncated=len(encoded) > bound)
                    if self._identity.routing is not None
                    else self._writer.try_enqueue_turn(operation)
                )
            if not enqueued:
                self._record_turn_lost()
                self._first_failure.signal("writer_failed")
        except Exception:
            if self._identity.routing is not None and capture_id is not None:
                cast(Any, self._writer).try_enqueue_capture_loss(self._identity.call_id, capture_id)
            if accepted:
                self._record_turn_lost()
            self._first_failure.signal("persistence_failed")

    def _record_turn_lost(self) -> None:
        try:
            self._runtime_metrics.record_transcript_turn_lost()
        except Exception:
            return

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


class RegistryTerminalizer(Protocol):
    async def reserve_or_read(
        self,
        proposed: TerminalProposal,
    ) -> TerminalAuthority: ...

    async def complete(self, authority: TerminalAuthority) -> bool: ...

    def note_failure(self, code: str) -> None: ...

    def transfer_to_cleanup(self, cleanup_task: asyncio.Task[None]) -> None: ...


class SessionWriter(ControlWriter, TurnWriter, Protocol):
    fatal_event: asyncio.Event


class CallSessionError(RuntimeError):
    """A constant-safe per-call runtime failure."""


class _TransportEvents(Protocol):
    def add_event_handler(self, event_name: str, handler: object) -> None: ...


SessionTaskFactory = Callable[[Coroutine[Any, Any, Any], str], asyncio.Task[Any]]


@dataclass(frozen=True, slots=True)
class _FrozenTerminalOutcome:
    status: Literal["closing", "closed", "failed"]
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

    def note_runtime_reason(self, reason: str | None) -> None:
        if self._frozen is None and self._reason == "closed" and reason is not None:
            self._reason = reason

    def request_reason(self, reason: str) -> None:
        if self._frozen is None and (
            reason == "recording_required_error" or self._reason == "closed"
        ):
            self._reason = reason

    def promote_failure(self, code: str | None) -> None:
        if self._frozen is None and self._reason == "closed" and code is not None:
            self._reason = code

    def freeze(self) -> _FrozenTerminalOutcome:
        if self._frozen is None:
            self._frozen = _FrozenTerminalOutcome(
                status="closing"
                if self._reason == "qualified_line_connected"
                else "closed"
                if self._reason == "closed"
                else "failed",
                reason=self._reason,
            )
        return self._frozen

    def freeze_authoritative(
        self,
        *,
        status: Literal["closing", "closed", "failed"],
        reason: str,
    ) -> _FrozenTerminalOutcome:
        if self._frozen is None:
            self._frozen = _FrozenTerminalOutcome(status=status, reason=reason)
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
        profile: RuntimeDeploymentProfileV1,
        services: ServiceBundle,
        writer: SessionWriter,
        keyring: CryptoKeyring,
        recording: RecordingBoundary,
        lease_terminalizer: LeaseTerminalizer,
        registry_terminalizer: RegistryTerminalizer | None = None,
        metric_lease: _CallMetricLease | None = None,
        runtime_metrics: RuntimeMetrics,
        observers: _CallObservers,
        idle_timeout_seconds: float,
        cleanup_phase_timeout_seconds: float = 5.0,
        task_factory: SessionTaskFactory | None = None,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if (
            not isinstance(profile, QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1)
            or profile.deployment_id != identity.deployment_id
            or type(runtime_metrics) is not RuntimeMetrics
            or type(observers) is not _CallObservers
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
        try:
            observers._bind_session(  # noqa: SLF001
                runtime_metrics=runtime_metrics,
                services=services,
            )
        except ValueError:
            raise ValueError("call_session_config_invalid") from None
        self._identity = identity
        self._manifest = manifest
        self._profile = profile
        self._services = services
        self._writer = writer
        self._keyring = keyring
        self._recording = recording
        self._lease_terminalizer = lease_terminalizer
        self._registry_terminalizer = registry_terminalizer
        self._metric_lease = metric_lease
        self._runtime_metrics = runtime_metrics
        self._observers = observers
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._cleanup_phase_timeout_seconds = float(cleanup_phase_timeout_seconds)
        self._task_factory = task_factory or self._default_task_factory
        self._utcnow = utcnow
        self._uuid_factory = uuid_factory
        self._run_started = False
        self._active_runner: WorkerRunner | None = None
        self._drain_requested = False
        self._drain_claimed = False
        self._drain_task: asyncio.Task[None] | None = None
        self._drain_lock = asyncio.Lock()
        self._terminal_outcome = _TerminalOutcome("closed")
        self._no_new_ai = False
        self._caller_started = False
        self._invitation_queued = False
        self._controller: DisclosureController | None = None
        self._recorder: TurnRecorder | None = None
        self._active_runtime: CallRuntime | None = None
        self._terminal_publication: VoiceOperationV1 | VoiceOperationV2 | None = None
        self._partial_result: MessageResultV1 | None = None
        self._result_inference_task: asyncio.Task[MessageResultV1 | None] | None = None
        self._result_inference_fenced = False

    @property
    def no_new_ai(self) -> bool:
        return self._no_new_ai

    def stop_new_ai(self) -> None:
        self._no_new_ai = True
        self.stop_result_inference()
        if self._controller is not None:
            self._controller.stop_input()
        if self._recorder is not None:
            self._recorder.close()

    def stop_result_inference(self) -> None:
        """A takeover intent irreversibly revokes this call's result request."""
        self._result_inference_fenced = True
        self._partial_result = None
        if self._result_inference_task is not None:
            self._result_inference_task.cancel()

    async def _prepare_partial_result(self) -> None:
        if self._identity.routing is None:
            return
        frozen = (
            await cast(Any, self._writer).read_frozen_call_publication_v2(
                self._identity.call_id, generation=self._identity.generation.generation
            ) if isinstance(getattr(self._identity, "begin_snapshot", None), BeginCallSnapshotV2)
            else await cast(Any, self._writer).read_frozen_call_publication(self._identity.call_id)
        )
        if frozen is not None:
            self._terminal_publication = frozen
            return
        retained = await cast(Any, self._writer).read_retained_call(self._identity.call_id)
        facts = await cast(Any, self._writer).read_call_lifecycle(self._identity.call_id)
        if (
            self._no_new_ai
            or getattr(self, "_result_inference_fenced", False)
            or retained.erased
            or (facts is not None and facts.transfer_fenced)
        ):
            return
        task = asyncio.create_task(
            infer_partial_result(
                cast(Any, self._services.llm), retained, self._identity.routing.from_e164
            ),
            name="call-result-inference-owned",
        )
        self._result_inference_task = task
        try:
            async with asyncio.timeout(self._cleanup_phase_timeout_seconds):
                result = await task
            if not self._no_new_ai and not self._result_inference_fenced:
                self._partial_result = result
        except (Exception, asyncio.CancelledError):
            self._partial_result = None
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _request_human_tool(self, params: FunctionCallParams) -> None:
        if (
            not isinstance(params.arguments, dict)
            or params.arguments
            or self._registry_terminalizer is None
            or self._no_new_ai
        ):
            result = "unavailable_collect_message"
        else:
            result = await cast(Any, self._registry_terminalizer).request_human()
        await params.result_callback(
            {"status": result, "human_identity_verified": False},
            properties=FunctionCallResultProperties(run_llm=not self._no_new_ai),
        )

    def _note_caller_started(self) -> None:
        self._caller_started = True

    async def _queue_opening_invitation(self) -> None:
        controller, runtime = self._controller, self._active_runtime
        if (
            self._invitation_queued
            or self._caller_started
            or self._no_new_ai
            or self._drain_requested
            or self._writer.fatal_event.is_set()
            or controller is None
            or not controller.is_active()
            or runtime is None
            or runtime.worker.has_finished()
        ):
            return
        self._invitation_queued = True
        await runtime.worker.queue_frame(
            TTSSpeakFrame("Comment puis-je vous aider ?", append_to_context=True)
        )

    async def run(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        if self._run_started:
            raise CallSessionError("call_session_already_run")
        self._run_started = True

        first_failure = FirstFailure(shared_failure_event=self._writer.fatal_event)
        native_fatal = _NativeFatalObservation()
        local_capture = (
            LocalAudioCapture(
                snapshot=self._identity.begin_snapshot, deployment_id=self._identity.deployment_id,
                generation=self._identity.generation.generation,
                keyring=self._keyring, writer=self._writer,
            )
            if isinstance(self._identity.begin_snapshot, BeginCallSnapshotV2)
            and self._identity.begin_snapshot.audio_available
            and self._identity.begin_snapshot.recording_policy == "local_30d"
            and isinstance(self._writer, PersistenceWriter)
            else None
        )

        async def quiesce_local_capture() -> None:
            if local_capture is not None and (await local_capture.quiesce()).pending:
                raise RuntimeError("local_audio_quiesce_pending")

        controller = DisclosureController(
            identity=self._identity,
            writer=self._writer,
            first_failure=first_failure,
            recording=self._recording,
            recording_enabled=self._identity.begin_snapshot.recording_enabled
            if isinstance(self._identity.begin_snapshot, BeginCallSnapshotV1)
            else self._manifest.recording_mode != "off",
            recording_required=self._identity.begin_snapshot.recording_enabled
            if isinstance(self._identity.begin_snapshot, BeginCallSnapshotV1)
            else self._manifest.recording_required,
            mark_timeout_seconds=self._profile.disclosure_mark_timeout_ms / 1000,
            runtime_metrics=self._runtime_metrics,
            utcnow=self._utcnow,
            uuid_factory=self._uuid_factory,
            on_active=self._queue_opening_invitation
            if self._identity.begin_snapshot is not None
            else None,
            local_audio_start=None if local_capture is None else local_capture.start,
            local_audio_refuse=None if local_capture is None else local_capture.refuse,
            local_audio_close=None if local_capture is None else local_capture.close_admission,
            local_audio_quiesce=None if local_capture is None else quiesce_local_capture,
            local_audio_finish=None if local_capture is None else local_capture.finish,
            local_audio_revoke=None if local_capture is None else local_capture.revoke,
        )
        recorder = TurnRecorder(
            identity=self._identity,
            writer=self._writer,
            keyring=self._keyring,
            first_failure=first_failure,
            runtime_metrics=self._runtime_metrics,
            uuid_factory=self._uuid_factory,
            utcnow=self._utcnow,
        )
        self._controller, self._recorder = controller, recorder
        if self._no_new_ai:
            self.stop_new_ai()
        pipeline: ObservedPipeline | None = None
        runtime: CallRuntime | None = None
        runner_task: asyncio.Task[None] | None = None
        failure_task: asyncio.Task[str] | None = None
        reason: str | None = None
        cancellation: asyncio.CancelledError | None = None

        try:
            self._verify_handshake(handshake)
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
                begin_snapshot=self._identity.begin_snapshot,
                capture_tap=None if local_capture is None else local_capture.tap,
                transfer_handler=self._request_human_tool
                if self._identity.begin_snapshot is not None
                and self._registry_terminalizer is not None
                and await cast(Any, self._registry_terminalizer).human_tool_available()
                else None,
                on_user_turn_started=self._note_caller_started
                if self._identity.begin_snapshot is not None
                else None,
            )
            runtime = build_runtime(
                pipeline=pipeline,
                first_failure=first_failure,
                greeting=controller.announcement_text
                if self._identity.begin_snapshot is not None
                else self._manifest.greeting,
                mark_name=controller.mark_name,
                idle_timeout_seconds=self._idle_timeout_seconds,
                observers=self._observers,
            )
            self._active_runtime = runtime

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
        except CallSessionError as error:
            reason = (
                error.args[0]
                if error.args and error.args[0] == "call_identity_mismatch"
                else "call_failed"
            )
        except Exception:
            reason = first_failure.code or "call_failed"
            first_failure.signal(reason)
        terminal_outcome = self._terminal_outcome
        terminal_outcome.note_runtime_reason(reason)
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
        if terminal_outcome.reason == "qualified_line_connected":
            return
        if terminal_outcome.reason != "closed":
            raise CallSessionError(terminal_outcome.reason)
        final_error = first_failure.code
        if final_error is not None:
            raise CallSessionError(final_error)

    async def request_drain(self, reason: str | None = None) -> None:
        """End the active per-call runner through its public cancellation surface."""

        if reason is not None:
            self._latch_drain_reason(reason)

        async with self._drain_lock:
            self._drain_requested = True
            if self._active_runner is not None and not self._drain_claimed:
                self._drain_claimed = True
                self._drain_task = asyncio.create_task(
                    self._drain_runner_owned(self._active_runner),
                    name="call-drain-owned",
                )
            task = self._drain_task
        if task is not None:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A cancelled waiter cannot revoke the call-owned departure.
                    continue
            await task

    async def _drain_runner_owned(self, runner: WorkerRunner) -> None:
        clear_task = None
        if self._no_new_ai and self._active_runtime is not None:
            clear_task = asyncio.create_task(
                self._clear_takeover_owned(self._active_runtime), name="call-takeover-clear"
            )
        try:
            # Cancellation is independent of the optional output acknowledgement.
            await runner.cancel(reason="drain")
        finally:
            if self._result_inference_task is not None and self._no_new_ai:
                self._result_inference_task.cancel()
                await asyncio.gather(self._result_inference_task, return_exceptions=True)
            if clear_task is not None:
                await clear_task

    async def _clear_takeover_owned(self, runtime: CallRuntime) -> None:
        try:
            async with asyncio.timeout(self._cleanup_phase_timeout_seconds):
                await runtime.request_clear()
        except (Exception, asyncio.CancelledError):
            return

    def _latch_drain_reason(self, reason: str) -> None:
        if type(reason) is not str or reason not in {
            "recording_required_error",
            "process_draining",
            "external_cancel",
            "token_deadline",
            "session_construction_failed",
            "telnyx_hangup",
            "qualified_line_connected",
            "content_erased",
        }:
            raise ValueError("call_drain_reason_invalid") from None
        self._terminal_outcome.request_reason(reason)

    async def aclose_unstarted(self) -> None:
        """Close transferred resources and terminalize before `run` starts."""

        if self._run_started:
            return
        self._run_started = True
        first_failure = FirstFailure(shared_failure_event=self._writer.fatal_event)
        terminal_outcome = self._terminal_outcome
        terminal_outcome.note_runtime_reason("session_construction_failed")
        processors = (self._services.tts, self._services.llm, self._services.stt)
        for processor in processors:
            await self._attempt(
                processor.cleanup,
                first_failure,
                "pipeline_cleanup_failed",
            )
        await self._attempt(
            lambda: self._recording.cleanup(
                self._identity,
                recording_may_be_active=False,
                reason=terminal_outcome.reason,
            ),
            first_failure,
            "recording_cleanup_failed",
        )
        await self._attempt(
            self._services.aclose,
            first_failure,
            "service_close_failed",
        )
        await self._finish_durable_boundaries(
            first_failure=first_failure,
            terminal_outcome=terminal_outcome,
            disclosure_completed=False,
        )

    async def _replay_pending_drain(self) -> None:
        async with self._drain_lock:
            requested = self._drain_requested
        if requested:
            await self.request_drain()

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
        cleanup_gate = asyncio.Event()

        async def run_cleanup() -> None:
            await cleanup_gate.wait()
            await cleanup

        cleanup_task = asyncio.create_task(run_cleanup(), name="call-cleanup")
        if self._registry_terminalizer is not None:
            self._registry_terminalizer.transfer_to_cleanup(cleanup_task)
        cleanup_gate.set()
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as error:
                caller_is_cancelling = caller_task is not None and caller_task.cancelling() > 0
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
                cancel_continuations=cancel_continuations,
                normal_completion=terminal_outcome.reason == "closed",
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
        await self._attempt(self._prepare_partial_result, first_failure, "persistence_failed")
        if self._identity.routing is None:
            await self._attempt(self._services.aclose, first_failure, "service_close_failed")
        self._promote_cleanup_failure(terminal_outcome, first_failure)
        try:
            await self._finish_durable_boundaries(
                first_failure=first_failure,
                terminal_outcome=terminal_outcome,
                disclosure_completed=controller.disclosure_completed,
            )
        finally:
            if self._identity.routing is not None:
                await self._attempt(self._services.aclose, first_failure, "service_close_failed")

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
        routing = self._identity.routing
        if routing is not None:
            data = handshake.call_data.model_dump(by_alias=True)
            caller = data.get("from")
            try:
                caller = None if caller is None else _e164(caller)
            except (ValueError, TypeError):
                caller = None
            if data.get("to") != routing.to_e164 or caller != routing.from_e164:
                raise CallSessionError("call_identity_mismatch")

    async def _finish_durable_boundaries(
        self,
        *,
        first_failure: FirstFailure,
        terminal_outcome: _TerminalOutcome,
        disclosure_completed: bool,
    ) -> None:
        self._promote_cleanup_failure(terminal_outcome, first_failure)
        if self._registry_terminalizer is not None and self._metric_lease is not None:
            proposed = self._terminal_proposal(terminal_outcome.reason)
            authority = await self._registry_terminalizer.reserve_or_read(proposed)
            frozen = terminal_outcome.freeze_authoritative(
                status=authority.status,
                reason=authority.reason,
            )
            self._metric_lease.finish(authority.metric_class)
            if not self._writer.fatal_event.is_set():
                persisted = await self._commit_authoritative_terminal_call(
                    status=frozen.status,
                    reason=frozen.reason,
                    disclosure_completed=disclosure_completed,
                    operation_id=authority.completion_token,
                    ended_at=authority._closed_at,  # noqa: SLF001
                )
                if not persisted:
                    if self._identity.routing is not None:
                        await self._attempt(
                            self._services.aclose, first_failure, "service_close_failed"
                        )
                    return
            if self._identity.routing is not None:
                await self._attempt(self._services.aclose, first_failure, "service_close_failed")
            await self._registry_terminalizer.complete(authority)
            return
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
        if self._identity.routing is not None:
            await self._attempt(self._services.aclose, first_failure, "service_close_failed")
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
    def _terminal_proposal(reason: str) -> TerminalProposal:
        if reason == "qualified_line_connected":
            return TerminalProposal(
                status="closing", reason=reason, metric_class="drained", cleanup_hangup=False
            )
        if reason == "recording_required_error":
            return TerminalProposal(
                status="failed",
                reason=reason,
                metric_class="failed",
                cleanup_hangup=False,
            )
        if reason == "process_draining":
            return TerminalProposal(
                status="closed",
                reason=reason,
                metric_class="drained",
                cleanup_hangup=True,
            )
        return TerminalProposal(
            status="closed" if reason == "closed" else "failed",
            reason=reason,
            metric_class="closed" if reason == "closed" else "failed",
            cleanup_hangup=True,
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
        status: Literal["closing", "closed", "failed"],
        reason: str,
        disclosure_completed: bool,
        operation_id: UUID | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        terminal_at = (
            self._utcnow().astimezone(UTC) if ended_at is None else ended_at.astimezone(UTC)
        )
        if terminal_at >= self._identity.retention_until:
            return
        evidence = None if self._controller is None else self._controller.evidence
        operation = self._terminal_publication
        if operation is None:
            payload = CallUpsertPayloadV1(
                telnyx_call_control_id=self._identity.telnyx_call_control_id,
                telnyx_call_leg_id=self._identity.telnyx_call_leg_id,
                telnyx_call_session_id=self._identity.telnyx_call_session_id,
                status=status,
                disclosure_state="completed"
                if disclosure_completed
                or evidence is not None
                and evidence.completed_at is not None
                else "failed",
                started_at=self._identity.started_at,
                ended_at=None if status == "closing" else terminal_at,
                end_reason=reason,
                retention_until=self._identity.retention_until,
                **(
                    cast(
                        Any,
                        (
                            {"disclosure_evidence": evidence}
                            if self._identity.routing is not None and evidence is not None
                            else {}
                        ),
                    )
                ),
            )
            if isinstance(getattr(self._identity, "begin_snapshot", None), BeginCallSnapshotV2):
                operation = VoiceOperationV2(schema_version=2,
                    operation_id=operation_id or self._uuid_factory(),
                    deployment_id=self._identity.deployment_id, call_id=self._identity.call_id,
                    occurred_at=terminal_at, kind="call.upsert", payload=payload)
            else:
                operation = VoiceOperationV1(schema_version=1,
                    operation_id=operation_id or self._uuid_factory(),
                    deployment_id=self._identity.deployment_id, call_id=self._identity.call_id,
                    occurred_at=terminal_at, kind="call.upsert", payload=payload)
        self._terminal_publication = operation
        if self._identity.routing is not None:
            if isinstance(operation, VoiceOperationV2):
                frozen = await cast(Any, self._writer).freeze_call_publication_v2(
                    operation, self._partial_result,
                    generation=self._identity.generation.generation,
                    provider_callback=self._identity.routing.from_e164,
                    result_permitted=lambda: (
                        not self._no_new_ai and not self._result_inference_fenced
                    ),
                )
            else:
                frozen = await cast(Any, self._writer).freeze_call_publication(
                    operation, self._partial_result,
                    provider_callback=self._identity.routing.from_e164,
                    result_permitted=lambda: (
                        not self._no_new_ai and not self._result_inference_fenced
                    ),
                )
            if frozen is not None:
                self._terminal_publication = frozen
            return
        await self._writer.commit_control(
            PersistenceCommand("outbox", {"operation": operation}, None)
        )

    async def _commit_authoritative_terminal_call(
        self,
        *,
        status: Literal["closing", "closed", "failed"],
        reason: str,
        disclosure_completed: bool,
        operation_id: UUID,
        ended_at: datetime,
    ) -> bool:
        while True:
            try:
                await self._commit_terminal_call(
                    status=status,
                    reason=reason,
                    disclosure_completed=disclosure_completed,
                    operation_id=operation_id,
                    ended_at=ended_at,
                )
            except asyncio.CancelledError:
                continue
            except BaseException:
                if self._registry_terminalizer is not None:
                    self._registry_terminalizer.note_failure("terminal_persistence_failed")
                return False
            return True

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
        runtime_metrics: RuntimeMetrics,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        uuid_factory: Callable[[], UUID] = uuid4,
        on_active: Callable[[], Awaitable[None]] | None = None,
        local_audio_start: Callable[[], Awaitable[bool]] | None = None,
        local_audio_refuse: Callable[[], None] | None = None,
        local_audio_close: Callable[[], None] | None = None,
        local_audio_quiesce: Callable[[], Awaitable[None]] | None = None,
        local_audio_finish: (
            Callable[[AudioFinishReason], Awaitable[AudioCaptureSummary]] | None
        ) = None,
        local_audio_revoke: Callable[[], Awaitable[AudioCaptureSummary]] | None = None,
    ) -> None:
        if isinstance(identity.begin_snapshot, BeginCallSnapshotV1):
            recording_enabled = identity.begin_snapshot.recording_enabled
            recording_required = recording_enabled
        elif isinstance(identity.begin_snapshot, BeginCallSnapshotV2):
            recording_enabled = recording_required = False
        if (
            type(recording_enabled) is not bool
            or type(recording_required) is not bool
            or recording_required
            and not recording_enabled
            or not isinstance(mark_timeout_seconds, int | float)
            or isinstance(mark_timeout_seconds, bool)
            or mark_timeout_seconds <= 0
            or type(runtime_metrics) is not RuntimeMetrics
            or not callable(monotonic)
        ):
            raise ValueError("disclosure_config_invalid")
        self._identity = identity
        self._writer = writer
        self._first_failure = first_failure
        self._recording = recording
        self._recording_enabled = recording_enabled
        self._recording_required = recording_required
        self._mark_timeout_seconds = float(mark_timeout_seconds)
        self._runtime_metrics = runtime_metrics
        self._monotonic = monotonic
        self._utcnow = utcnow
        self._uuid_factory = uuid_factory
        self._on_active = on_active
        self._local_snapshot = (
            identity.begin_snapshot if isinstance(identity.begin_snapshot, BeginCallSnapshotV2)
            else None
        )
        self._local_enabled = (
            self._local_snapshot is not None and self._local_snapshot.audio_available
            and self._local_snapshot.recording_policy == "local_30d"
        )
        self._local_audio_start = local_audio_start
        self._local_audio_refuse = local_audio_refuse
        self._local_audio_close = local_audio_close
        self._local_audio_quiesce = local_audio_quiesce
        self._local_audio_finish = local_audio_finish
        self._local_audio_revoke = local_audio_revoke
        self._local_revoke_requested = False
        self._local_choice_off = False
        self._local_refused = False
        self._local_quiesced = False
        self._local_quiesce_lock = asyncio.Lock()
        self._local_start_done = asyncio.Event()
        self._local_start_done.set()
        self._local_denial_task: asyncio.Task[None] | None = None
        self._choice_timeout_task: asyncio.Task[None] | None = None
        self._choice_deadline: float | None = None
        self.disclosure_generation = uuid_factory()
        self.mark_name = f"pv0-disclosure-{uuid_factory().hex}"
        self._lock = asyncio.Lock()
        self.state = DisclosureState.PLAYING
        self.disclosure_completed = False
        self.recording_may_be_active = False
        self._input_closed = False
        self._audio_observed = False
        self._timeout_task: asyncio.Task[None] | None = None
        self._mark_armed_at: float | None = None
        self._continuations: set[asyncio.Task[None]] = set()
        self._disclosure_started_at: datetime | None = None
        self._disclosure_completed_at: datetime | None = None
        self._disclosure_failed_at: datetime | None = None
        self._input_gate_opened_at: datetime | None = None
        self._gate_publication: VoiceOperationV1 | VoiceOperationV2 | None = None

    @property
    def announcement_text(self) -> str:
        pin = self._local_snapshot
        if pin is not None and self._local_enabled:
            return (
                "Bonjour. Je suis Sparra, un assistant vocal automatisé. "
                "Pour prendre votre message, l'audio de notre conversation peut être conservé "
                f"30 jours pour {pin.knowledge.business_name}. "
                f"Vous pouvez contacter cet établissement au {pin.recording_contact_phone}. "
                "Le texte de cet échange est conservé trente jours, "
                "même sans enregistrement audio. "
                "Sans choix, l'appel continue sans enregistrement. "
                "Pendant l'appel, tapez 2 pour arrêter l'enregistrement. "
                "Après cette annonce, tapez 1 pour accepter l'enregistrement audio, "
                "ou 2 pour continuer sans."
            )
        return SPARRA_RECORDING_DISCLOSURE if self._recording_enabled else SPARRA_DISCLOSURE

    @property
    def evidence(self) -> DisclosureEvidenceV1:
        return DisclosureEvidenceV1(
            schema_version=1,
            started_at=self._disclosure_started_at,
            completed_at=self._disclosure_completed_at,
            failed_at=self._disclosure_failed_at,
            input_gate_opened_at=self._input_gate_opened_at,
        )

    def stop_input(self) -> None:
        self._input_closed = True
        if self._local_snapshot is not None:
            self._refuse_local_audio()

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
                if self._disclosure_started_at is None:
                    self._disclosure_started_at = self._utcnow().astimezone(UTC)

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
                try:
                    armed_at = self._monotonic()
                    self._mark_armed_at = (
                        float(armed_at)
                        if isinstance(armed_at, int | float)
                        and not isinstance(armed_at, bool)
                        and math.isfinite(armed_at)
                        else None
                    )
                except Exception:
                    self._mark_armed_at = None
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
            acknowledged_at = self._utcnow().astimezone(UTC)
            self._disclosure_completed_at = acknowledged_at
            if self._local_enabled:
                self._choice_deadline = self._monotonic() + 5.0
            armed_at = self._mark_armed_at
            self._mark_armed_at = None
            if armed_at is not None:
                with suppress(Exception):
                    self._runtime_metrics.record_disclosure_ack(self._monotonic() - armed_at)
            timeout_task = self._timeout_task
            self._timeout_task = None
            self._mark_armed_at = None
            if timeout_task is not None:
                timeout_task.cancel()
            task = asyncio.create_task(
                self._run_owned_continuation(
                    lambda: self._complete_disclosure(acknowledged_at),
                    "disclosure_commit_failed",
                ),
                name="disclosure-ack-continuation",
            )
            self._track(task)
            return True

    def _refuse_local_audio(self) -> None:
        if self._local_refused:
            return
        self._local_refused = True
        if self._local_audio_refuse is not None:
            try:
                self._local_audio_refuse()
            except Exception:
                self._first_failure.signal("local_audio_refusal_failed")

    def _cancel_choice_timeout(self) -> None:
        timer = self._choice_timeout_task
        self._choice_timeout_task = None
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()

    async def _quiesce_local_audio(self) -> None:
        async with self._local_quiesce_lock:
            await self._local_start_done.wait()
            if not self._local_quiesced:
                if self._local_audio_quiesce is not None:
                    await self._local_audio_quiesce()
                self._local_quiesced = True

    async def _deny_local_audio(self) -> None:
        decided = await self._writer.commit_audio_choice(
            self._identity.call_id, generation=self._identity.generation.generation,
            choice="off", occurred_at=None,
        )
        self._local_choice_off = decided.choice_state == "off"
        await self._revoke_local_audio()

    async def _revoke_local_audio(self) -> None:
        if self._local_audio_revoke is None:
            await self._quiesce_local_audio()
            return
        async with self._local_quiesce_lock:
            await self._local_start_done.wait()
            summary = await self._local_audio_revoke()
            self._local_quiesced = not summary.pending

    def _schedule_local_denial(self) -> None:
        if self._local_denial_task is not None:
            return
        task = asyncio.create_task(
            self._run_owned_continuation(self._deny_local_audio, "local_audio_denial_failed"),
            name="local-audio-denial",
        )
        self._local_denial_task = task
        self._track(task)

    def _local_control_closed(self) -> bool:
        return (
            self._input_closed or self._first_failure.code is not None
            or self.state is DisclosureState.ABORTED
        )

    async def accept_dtmf(self, frame: TelnyxInputDTMFFrame) -> bool:
        if not self._local_enabled or not isinstance(frame, TelnyxInputDTMFFrame):
            return False
        if self._local_control_closed():
            return True
        digit = frame.button.value
        if digit == "2":
            # Close the concrete capture offer latch before any lock/SQL await.
            self._refuse_local_audio()
            self._local_revoke_requested = True
            self._cancel_choice_timeout()
            async with self._lock:
                if self._local_control_closed():
                    return True
                if self.state is DisclosureState.WAITING_CHOICE:
                    self.state = DisclosureState.CHOICE_COMMITTING
                    task = asyncio.create_task(
                        self._run_owned_continuation(
                            lambda: self._finish_local_choice("off", None),
                            "local_audio_choice_failed",
                        ), name="local-audio-choice-off",
                    )
                    self._track(task)
                else:
                    self._schedule_local_denial()
            return True
        async with self._lock:
            if self.state is not DisclosureState.WAITING_CHOICE:
                return True
            deadline = self._choice_deadline
            timed_out = deadline is None or self._monotonic() >= deadline
            timestamp = frame.occurred_at
            if digit == "1" and not timed_out:
                if (
                    timestamp is None or frame.sequence_number is None
                    or self._disclosure_completed_at is None
                    or not self._disclosure_completed_at <= timestamp <= self._utcnow()
                ):
                    return True
                choice: Literal["accept", "off"] = "accept"
            else:
                choice = "off"
                timestamp = None
                self._refuse_local_audio()
            self._cancel_choice_timeout()
            self.state = DisclosureState.CHOICE_COMMITTING
            task = asyncio.create_task(
                self._run_owned_continuation(
                    lambda: self._finish_local_choice(choice, timestamp),
                    "local_audio_choice_failed",
                ), name="local-audio-choice",
            )
            self._track(task)
            return True

    def _local_control_operation(self, occurred_at: datetime) -> VoiceOperationV2:
        return VoiceOperationV2(
            schema_version=2, operation_id=self._uuid_factory(),
            deployment_id=self._identity.deployment_id, call_id=self._identity.call_id,
            occurred_at=occurred_at, kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=self._identity.telnyx_call_control_id,
                telnyx_call_leg_id=self._identity.telnyx_call_leg_id,
                telnyx_call_session_id=self._identity.telnyx_call_session_id,
                status="active", disclosure_state="completed", started_at=self._identity.started_at,
                ended_at=None, end_reason=None, retention_until=self._identity.retention_until,
                disclosure_evidence=self.evidence,
            ),
        )

    async def _complete_local_disclosure(self, acknowledged_at: datetime) -> None:
        await self._writer.publish_control_v2(
            self._local_control_operation(acknowledged_at),
            generation=self._identity.generation.generation,
        )
        async with self._lock:
            self.disclosure_completed = True
            if (
                self._first_failure.code is not None or self._input_closed
                or self.state is not DisclosureState.ACK_COMMITTING
            ):
                return
            deadline = self._choice_deadline
            if (
                self._local_refused or not self._local_enabled
                or deadline is None or self._monotonic() >= deadline
            ):
                self.state = DisclosureState.CHOICE_COMMITTING
                choose_off = True
            else:
                self.state = DisclosureState.WAITING_CHOICE
                choose_off = False
                task = asyncio.create_task(
                    self._local_choice_timeout(), name="local-audio-choice-timeout"
                )
                self._choice_timeout_task = task
                self._track(task)
        if choose_off:
            await self._finish_local_choice("off", None)

    async def _local_choice_timeout(self) -> None:
        deadline = self._choice_deadline
        if deadline is None:
            return
        await asyncio.sleep(max(0.0, deadline - self._monotonic()))
        async with self._lock:
            if self.state is not DisclosureState.WAITING_CHOICE or self._input_closed:
                return
            self.state = DisclosureState.CHOICE_COMMITTING
            self._choice_timeout_task = None
            self._refuse_local_audio()
        await self._run_owned_continuation(
            lambda: self._finish_local_choice("off", None), "local_audio_choice_failed"
        )

    async def _finish_local_choice(
        self, choice: Literal["accept", "off"], timestamp: datetime | None
    ) -> None:
        if choice == "off":
            self._refuse_local_audio()
        try:
            decided = await self._writer.commit_audio_choice(
                self._identity.call_id, generation=self._identity.generation.generation,
                choice=choice, occurred_at=timestamp,
            )
        except FatalPersistenceError:
            raise
        except PersistenceError:
            self._refuse_local_audio()
            decided = await self._writer.commit_audio_choice(
                self._identity.call_id, generation=self._identity.generation.generation,
                choice="off", occurred_at=None,
            )
        self._local_choice_off = decided.choice_state == "off"
        accepted = decided.choice_state == "accepted" and not self._local_refused
        if self._local_denial_task is not None:
            await self._local_denial_task
            accepted = False
        async with self._lock:
            if self._first_failure.code is not None or self._input_closed:
                return
            self.state = DisclosureState.GATE_COMMITTING
            self._input_gate_opened_at = self._utcnow().astimezone(UTC)
        await self._publish_gate_evidence()
        if accepted and not self._local_refused:
            async with self._lock:
                if self._first_failure.code is not None or self._input_closed:
                    return
                self.state = DisclosureState.LOCAL_AUDIO_STARTING
                self._local_start_done.clear()
            try:
                started = (
                    self._local_audio_start is not None and await self._local_audio_start() is True
                )
            except asyncio.CancelledError:
                self._refuse_local_audio()
                raise
            except Exception:
                started = False
            finally:
                self._local_start_done.set()
            if not started:
                self._refuse_local_audio()
                decided = await self._writer.commit_audio_choice(
                    self._identity.call_id, generation=self._identity.generation.generation,
                    choice="off", occurred_at=None,
                )
                self._local_choice_off = decided.choice_state == "off"
        if self._local_refused:
            if self._local_denial_task is not None:
                await self._local_denial_task
            else:
                if self._local_revoke_requested:
                    await self._revoke_local_audio()
                else:
                    await self._quiesce_local_audio()
        async with self._lock:
            if self._first_failure.code is not None or self._input_closed:
                return
            self.state = DisclosureState.ACTIVE
        if self._on_active is not None and self.is_active():
            await self._on_active()

    async def abort(self, code: str) -> None:
        timeout_task: asyncio.Task[None] | None
        async with self._lock:
            self._input_closed = True
            if self._disclosure_completed_at is None and self._disclosure_failed_at is None:
                self._disclosure_failed_at = self._utcnow().astimezone(UTC)
            if self.state is not DisclosureState.ACTIVE:
                self.state = DisclosureState.ABORTED
            timeout_task = self._timeout_task
            self._timeout_task = None
            self._mark_armed_at = None
            if timeout_task is not None:
                timeout_task.cancel()
        self._first_failure.signal(code)
        if self._local_snapshot is not None:
            self._refuse_local_audio()
            self._cancel_choice_timeout()

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

    async def terminalize_and_join(
        self, *, cancel_continuations: bool, normal_completion: bool = False
    ) -> None:
        self._input_closed = True
        if self._local_snapshot is not None:
            if (
                normal_completion and not cancel_continuations and not self._local_refused
                and self._first_failure.code is None and self._local_audio_close is not None
            ):
                self._local_audio_close()
            else:
                self._refuse_local_audio()
            self._cancel_choice_timeout()
        async with self._lock:
            self._input_closed = True
            if self.state is not DisclosureState.ACTIVE:
                self.state = DisclosureState.ABORTED
            timeout_task = self._timeout_task
            self._timeout_task = None
            self._mark_armed_at = None
            if timeout_task is not None:
                timeout_task.cancel()
        if cancel_continuations:
            await self.cancel_and_join_continuations()
        else:
            await self.join_continuations()

    async def cleanup_termination(self, reason: str) -> None:
        try:
            if self._local_snapshot is not None:
                if (
                    (self._local_revoke_requested or self._local_choice_off)
                    and self._local_audio_revoke is not None
                ):
                    await self._revoke_local_audio()
                    if not self._local_quiesced:
                        raise RuntimeError("local_audio_terminal_pending")
                elif self._local_audio_finish is None:
                    await self._quiesce_local_audio()
                else:
                    audio_reason: AudioFinishReason = (
                        "complete" if reason == "closed"
                        else "transfer" if reason == "qualified_line_connected"
                        else "interrupted" if reason in {"external_cancel", "process_draining"}
                        else "failure"
                    )
                    summary = await self._local_audio_finish(audio_reason)
                    if summary.pending:
                        raise RuntimeError("local_audio_terminal_pending")
                    self._local_quiesced = True
                return
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
                self._mark_armed_at = None
            with suppress(Exception):
                self._runtime_metrics.record_disclosure_timeout()
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

        if self._local_snapshot is not None:
            await self._complete_local_disclosure(acknowledged_at)
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
                **(
                    cast(
                        Any,
                        (
                            {"disclosure_evidence": self.evidence}
                            if self._identity.routing is not None
                            else {}
                        ),
                    )
                ),
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
                self._input_gate_opened_at = self._utcnow().astimezone(UTC)
            else:
                self.state = DisclosureState.RECORDING_STARTING
                self.recording_may_be_active = True

        if not self._recording_enabled:
            if self._identity.routing is not None:
                await self._publish_gate_evidence()
            if self.is_active() and self._on_active is not None:
                await self._on_active()
            return

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
        elif self._identity.begin_snapshot is not None:
            async with self._lock:
                if not self.is_active():
                    return
                self._input_gate_opened_at = self._utcnow().astimezone(UTC)
            if self._identity.routing is not None:
                await self._publish_gate_evidence()
            if self.is_active() and self._on_active is not None:
                await self._on_active()

    async def _publish_gate_evidence(self) -> None:
        assert self._input_gate_opened_at is not None
        if self._local_snapshot is not None:
            if self._gate_publication is None:
                self._gate_publication = self._local_control_operation(self._input_gate_opened_at)
            if not isinstance(self._gate_publication, VoiceOperationV2):
                raise RuntimeError("local_audio_control_invalid")
            await self._writer.publish_control_v2(
                self._gate_publication, generation=self._identity.generation.generation
            )
            return
        if self._gate_publication is None:
            self._gate_publication = VoiceOperationV1(
                schema_version=1,
                operation_id=self._uuid_factory(),
                deployment_id=self._identity.deployment_id,
                call_id=self._identity.call_id,
                occurred_at=self._input_gate_opened_at,
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
                    disclosure_evidence=self.evidence,
                ),
            )
        await self._writer.commit_control(
            PersistenceCommand("outbox", {"operation": self._gate_publication}, None)
        )


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
