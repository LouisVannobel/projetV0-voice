"""Process-local admission and ownership contracts for the production Voice Cell."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import re
import secrets
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast
from uuid import UUID, uuid4, uuid5

from pydantic import SecretStr

from projetv0_voice.audio_contract import BeginCallSnapshotV2, VoiceOperationV2
from projetv0_voice.config import AgentManifestV1, SparraManifestV1
from projetv0_voice.models import (
    BeginCallSnapshotV1,
    CallUpsertPayloadV1,
    DisclosureEvidenceV1,
    RoutingV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import PersistenceCommand, PersistenceError
from projetv0_voice.persistence.postgres_sink import OperationSinkCommitAmbiguousError
from projetv0_voice.persistence.writer import (
    LocalCallAdmissionFacts,
    LocalCallLifecycleFacts,
    PersistenceWriter,
    QualificationRunConsumed,
    WebhookCommitResult,
    WebhookCommitValue,
)
from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    QualificationOverrideV1,
    QualifiedDeploymentProfileV1,
    RuntimeDeploymentProfileV1,
    canonical_qualified_profile_sha256,
)
from projetv0_voice.telnyx.call_control import (
    CallControlResult,
    StreamingStartV1,
    TransferRequestV1,
)

if TYPE_CHECKING:
    from projetv0_voice.telnyx.webhooks import (
        ResolvedWebhook,
        VerifiedWebhook,
        WebhookDisposition,
        WebhookDurableEffect,
    )

PinnedBeginSnapshot = BeginCallSnapshotV1 | BeginCallSnapshotV2


class WebhookFinalizationHandle(Protocol):
    """Opaque process-owned completion handle exposed to one HTTP waiter."""

    async def wait(self) -> WebhookDisposition: ...


class WebhookFinalizerOwner(Protocol):
    """Narrow Task 10A seam implemented by the Task 10C runtime supervisor."""

    async def classify_webhook_receipt(
        self, event: VerifiedWebhook
    ) -> Literal["missing", "duplicate", "conflict"]: ...

    def start_webhook_finalization(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        receipt: Literal["first", "duplicate"] = "first",
    ) -> WebhookFinalizationHandle: ...


class CallAdmissionRejected(RuntimeError):
    """A constant-safe local policy rejection before receipt persistence."""

    _POLICY_CODES = frozenset(
        {
            "call_capacity_reached",
            "placeholder_capacity_reached",
            "qualification_run_consumed",
            "qualification_window_expired",
            "call_draining",
            "sparra_recording_unqualified",
        }
    )
    _EVENT_CODES = frozenset({"call_event_invalid", "call_identity_conflict"})

    @property
    def status_code(self) -> Literal[400, 500, 503]:
        code = self.args[0] if self.args and isinstance(self.args[0], str) else ""
        if code in self._POLICY_CODES:
            return 503
        if code in self._EVENT_CODES:
            return 400
        return 500


def select_call_capacity(
    *,
    profile: RuntimeDeploymentProfileV1,
    manifest: AgentManifestV1,
    deployment_max_calls: int,
    override: QualificationOverrideV1 | None = None,
) -> int:
    """Select the literal strict, candidate, or bound-override capacity."""

    if (
        not isinstance(
            profile, QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1
        )
        or not isinstance(manifest, AgentManifestV1)
        or type(deployment_max_calls) is not int
    ):
        raise ValueError("call_capacity_invalid")
    if override is not None:
        if (
            not isinstance(profile, QualifiedDeploymentProfileV1)
            or not isinstance(override, QualificationOverrideV1)
            or override.deployment_id != profile.deployment_id
            or override.qualified_profile_sha256
            != canonical_qualified_profile_sha256(profile)
            or deployment_max_calls != override.benchmark_max_calls
        ):
            raise ValueError("call_capacity_invalid")
        return override.benchmark_max_calls
    if isinstance(profile, QualificationCandidateProfileV1):
        if deployment_max_calls < 1:
            raise ValueError("call_capacity_invalid")
        return 1
    if deployment_max_calls != manifest.max_concurrent_calls:
        raise ValueError("call_capacity_invalid")
    return manifest.max_concurrent_calls


class _LeaseWriter(Protocol):
    async def read_call_lifecycle(self, call_id: UUID) -> LocalCallLifecycleFacts | None: ...

    async def commit_lease(
        self,
        *,
        call_control_id: str,
        call_id: UUID,
        tenant_id: str,
        agent_id: str,
        state: Literal["pending", "active", "terminal"],
        token_hash: bytes,
        created_at: datetime,
        expires_at: datetime,
        closed_at: datetime | None,
        operation: VoiceOperationV1 | None = None,
    ) -> None: ...


class _DisclosureFields(TypedDict, total=False):
    disclosure_evidence: DisclosureEvidenceV1


class _CallControl(Protocol):
    async def answer(self, call_control_id: str, *, command_id: UUID) -> CallControlResult: ...

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


class _SessionLifecycleOwner(Protocol):
    _phase: str
    _session: object | None
    _task: asyncio.Task[None] | None
    _terminal_capability: _TerminalCapability | None

    def request_drain(self, cause: str) -> None: ...

    async def wait(self) -> None: ...


ActionState = Literal["idle", "in_flight", "retryable", "accepted", "rejected", "unknown"]
LeaseState = Literal["provisional", "pending", "claiming", "active", "terminal"]
SessionPhase = Literal[
    "active_unconsumed",
    "constructing",
    "preactivated",
    "running",
    "terminal",
    "removed",
]
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


@dataclass(slots=True, repr=False)
class _ActionSlot:
    command_id: UUID
    state: ActionState = "idle"
    completion: asyncio.Future[WebhookDisposition] | None = field(default=None, repr=False)
    owner_task: asyncio.Task[object] | None = field(default=None, repr=False)
    disposition: int = 200


@dataclass(frozen=True, slots=True, repr=False)
class _TerminalWork:
    entry: _CallEntry = field(repr=False)
    generation: UUID
    action_owners: tuple[asyncio.Task[object], ...] = field(repr=False)
    lifecycle_owners: tuple[_SessionLifecycleOwner, ...] = field(repr=False)
    reason: str
    persist_terminal: bool
    cleanup_hangup: bool


@dataclass(slots=True, repr=False)
class _CallEntry:
    generation: UUID
    call_control_id: str
    call_id: UUID
    call_leg_id: str | None
    call_session_id: str | None
    initiated_at: datetime
    created_at: datetime
    expires_at: datetime
    token_deadline: float
    token_digest: bytes = field(repr=False)
    raw_token: str | None = field(repr=False)
    answer: _ActionSlot = field(repr=False)
    streaming: _ActionSlot = field(repr=False)
    hangup_command_id: UUID = field(repr=False)
    precommit_refcount: int = 0
    durable: bool = False
    answer_evidence: bool = False
    streaming_evidence: bool = False
    lease_state: LeaseState = "provisional"
    claim: ProcessLeaseClaim | None = field(default=None, repr=False)
    claimed_at: datetime | None = field(default=None, repr=False)
    construction_grant: CallConstructionGrant | None = field(default=None, repr=False)
    session_phase: SessionPhase = "active_unconsumed"
    lifecycle_owner: object | None = field(default=None, repr=False)
    lifecycle_owner_task: asyncio.Task[None] | None = field(default=None, repr=False)
    terminal_capability: _TerminalCapability | None = field(default=None, repr=False)
    terminal_state: Literal["open", "external_pending", "reserved", "completing", "removed"] = (
        "open"
    )
    terminal_authority: TerminalAuthority | None = field(default=None, repr=False)
    terminal_completion_owner: asyncio.Task[object] | None = field(
        default=None,
        repr=False,
    )
    terminal_completion_event: asyncio.Event = field(
        default_factory=asyncio.Event,
        repr=False,
    )
    terminal_pending_count: int = 0
    terminal_settled_event: asyncio.Event = field(
        default_factory=asyncio.Event,
        repr=False,
    )
    terminal_event: str | None = None
    capacity_released: bool = False
    cleanup_hangup_started: bool = False
    attached: bool = False
    drain_intent: bool = False
    session: object | None = field(default=None, repr=False)
    resources_released: bool = False
    routing: RoutingV1 | None = None
    begin_snapshot: PinnedBeginSnapshot | None = None
    begin_task: asyncio.Task[None] | None = field(default=None, repr=False)
    begin_future: asyncio.Future[PinnedBeginSnapshot] | None = field(default=None, repr=False)
    answered_at: datetime | None = None
    transfer_facts: LocalCallLifecycleFacts | None = field(default=None, repr=False)
    transfer_task: asyncio.Task[None] | None = field(default=None, repr=False)
    content_stop_task: asyncio.Task[None] | None = field(default=None, repr=False)
    transfer_future: asyncio.Future[str] | None = field(default=None, repr=False)
    bridge_publication: VoiceOperationV1 | VoiceOperationV2 | None = field(default=None, repr=False)
    no_new_ai: bool = False
    abort_target_clearers: list[tuple[_AbortTarget, Callable[[_AbortTarget], None]]] = field(
        default_factory=list, repr=False
    )


@dataclass(frozen=True, slots=True, repr=False)
class _LinkedAbortIdentity:
    call_control_id: str = field(repr=False)
    generation: UUID = field(repr=False)
    token_digest: bytes = field(repr=False)

    def __repr__(self) -> str:
        return "LinkedAbortIdentity()"


@dataclass(slots=True, repr=False)
class _AnsweredPlaceholder:
    call_control_id: str
    call_leg_id: str | None
    call_session_id: str | None
    first_seen: float
    deadline: float
    answered_at: datetime | None = None
    precommit_refcount: int = 1
    durable: bool = False
    linked_entry: _CallEntry | None = field(default=None, repr=False)
    linked_abort_identity: _LinkedAbortIdentity | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class CallSnapshot:
    generation: UUID
    call_id: UUID
    durable: bool
    precommit_refcount: int
    answer_state: ActionState
    streaming_state: ActionState
    lease_state: LeaseState
    raw_token_retained: bool
    token_digest: bytes = field(repr=False)

    def __repr__(self) -> str:
        return (
            "CallSnapshot("
            f"durable={self.durable!r}, precommit_refcount={self.precommit_refcount!r}, "
            f"answer_state={self.answer_state!r}, streaming_state={self.streaming_state!r}, "
            f"lease_state={self.lease_state!r}, raw_token_retained={self.raw_token_retained!r})"
        )


class CallReservation:
    """One pre-COMMIT reference to an exact registry generation."""

    __slots__ = (
        "_abandoned",
        "_abandon_event",
        "_abandonment_task",
        "_abandon_requested",
        "_entry",
        "_event_type",
        "_generation",
        "_registry",
        "_registry_applied",
        "_settlement_task",
        "_settled",
        "_terminal_occurred_at",
    )

    def __init__(
        self,
        registry: CallRegistry,
        entry: _CallEntry,
        event_type: str,
        *,
        terminal_occurred_at: datetime | None = None,
    ) -> None:
        self._registry = registry
        self._entry = entry
        self._event_type = event_type
        self._terminal_occurred_at = terminal_occurred_at
        self._generation = entry.generation
        self._settled = False
        self._abandoned = False
        self._abandon_requested = False
        self._abandon_event = asyncio.Event()
        self._abandonment_task: asyncio.Task[None] | None = None
        self._registry_applied = False
        self._settlement_task: asyncio.Task[None] | None = None

    def __repr__(self) -> str:
        return "CallReservation()"

    def abandon_before_submit(self) -> None:
        if self._settled or self._abandon_requested:
            return
        self._abandon_requested = True
        self._abandon_event.set()

    async def settle_after_submit_failure(self) -> None:
        self.abandon_before_submit()
        await _join_owned_abandonment(self._abandonment_task)

    async def confirm(
        self,
        result: WebhookCommitValue,
    ) -> asyncio.CancelledError | None:
        if self._settled:
            return None
        if self._event_type == "call.hangup":
            settlement = self._settlement_task
            if settlement is None:
                settlement = asyncio.create_task(
                    self._registry._settle_reservation(self, result),
                    name="voice-provider-terminal-settlement",
                )
                self._settlement_task = settlement
            cancellation: asyncio.CancelledError | None = None
            while not settlement.done():
                try:
                    await asyncio.shield(settlement)
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
            settlement.result()
        else:
            await self._registry._settle_reservation(self, result)
            cancellation = None
        self._settled = True
        self._abandon_event.set()
        return cancellation

    async def _run_abandonment(self) -> None:
        await self._abandon_event.wait()
        if self._settled:
            return
        cancellation_seen = False
        while not self._registry_applied:
            try:
                await self._registry._settle_reservation(self, None)
            except asyncio.CancelledError:
                cancellation_seen = True
        self._abandoned = True
        if cancellation_seen:
            raise asyncio.CancelledError


class _PlaceholderReservation:
    __slots__ = (
        "_abandoned",
        "_abandon_event",
        "_abandonment_task",
        "_abandon_requested",
        "_placeholder",
        "_registry",
        "_registry_applied",
        "_settled",
    )

    def __init__(self, registry: CallRegistry, placeholder: _AnsweredPlaceholder) -> None:
        self._registry = registry
        self._placeholder = placeholder
        self._settled = False
        self._abandoned = False
        self._abandon_requested = False
        self._abandon_event = asyncio.Event()
        self._abandonment_task: asyncio.Task[None] | None = None
        self._registry_applied = False

    def __repr__(self) -> str:
        return "AnsweredPlaceholderReservation()"

    def abandon_before_submit(self) -> None:
        if self._settled or self._abandon_requested:
            return
        self._abandon_requested = True
        self._abandon_event.set()

    async def settle_after_submit_failure(self) -> None:
        self.abandon_before_submit()
        await _join_owned_abandonment(self._abandonment_task)

    async def confirm(self, result: WebhookCommitValue) -> None:
        if self._settled:
            return
        await self._registry._settle_placeholder(self, result)
        self._settled = True
        self._abandon_event.set()

    async def confirm_fail_closed(
        self,
        result: WebhookCommitValue,
        *,
        linked_abort_identity: _LinkedAbortIdentity | None = None,
    ) -> _AbortTarget | None:
        if self._settled:
            return None
        target = await self._registry._settle_placeholder(
            self,
            result,
            promote_linked_evidence=False,
            linked_abort_identity=linked_abort_identity,
        )
        self._settled = True
        self._abandon_event.set()
        return target

    async def _run_abandonment(self) -> None:
        await self._abandon_event.wait()
        if self._settled:
            return
        cancellation_seen = False
        while not self._registry_applied:
            try:
                await self._registry._settle_placeholder(self, None)
            except asyncio.CancelledError:
                cancellation_seen = True
        self._abandoned = True
        if cancellation_seen:
            raise asyncio.CancelledError


async def _join_owned_abandonment(task: asyncio.Task[None] | None) -> None:
    """Join the already-owned reservation descendant without creating another owner."""

    if task is None:
        raise RuntimeError("reservation_settlement_unavailable") from None
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except BaseException:
            break
    if task.cancelled():
        raise RuntimeError("reservation_settlement_failed") from None
    task.result()
    if cancellation is not None:
        raise cancellation


@dataclass(frozen=True, slots=True, repr=False)
class ProcessLeaseClaim:
    """Opaque proof that the exact local pending lease reached durable active."""

    call_control_id: str = field(repr=False)
    call_id: UUID = field(repr=False)
    generation: UUID = field(repr=False)
    token_digest: bytes = field(repr=False)
    claimed_at: datetime = field(repr=False)

    def __repr__(self) -> str:
        return "ProcessLeaseClaim()"


@dataclass(frozen=True, slots=True, repr=False)
class _AbortTarget:
    entry: _CallEntry = field(repr=False)
    generation: UUID
    call_control_id: str = field(repr=False)
    token_digest: bytes = field(repr=False)

    def __repr__(self) -> str:
        return "AbortTarget()"


@dataclass(slots=True, repr=False)
class _AbortHandoff:
    target: _AbortTarget = field(repr=False)
    scheduled: bool = False

    def __repr__(self) -> str:
        return "AbortHandoff()"


@dataclass(frozen=True, slots=True, repr=False)
class CallGenerationHandle:
    """Opaque generation-checked registry handle for later lifecycle owners."""

    call_control_id: str = field(repr=False)
    generation: UUID = field(repr=False)

    def __post_init__(self) -> None:
        if not self.call_control_id or not isinstance(self.generation, UUID):
            raise ValueError("call_generation_handle_invalid")

    def __repr__(self) -> str:
        return "CallGenerationHandle()"


@dataclass(frozen=True, slots=True, repr=False)
class CallConstructionGrant:
    """Immutable publication of one atomic claim-to-owner consume."""

    call_id: UUID = field(repr=False)
    generation: CallGenerationHandle = field(repr=False)
    lease_claim: ProcessLeaseClaim = field(repr=False)
    deployment_id: str = field(repr=False)
    telnyx_call_control_id: str = field(repr=False)
    telnyx_call_leg_id: str | None = field(repr=False)
    telnyx_call_session_id: str | None = field(repr=False)
    stream_id: str = field(repr=False)
    started_at: datetime = field(repr=False)
    retention_until: datetime = field(repr=False)
    routing: RoutingV1 | None = field(default=None, repr=False)
    begin_snapshot: PinnedBeginSnapshot | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.call_id, UUID)
            or not isinstance(self.generation, CallGenerationHandle)
            or not isinstance(self.lease_claim, ProcessLeaseClaim)
            or self.generation.call_control_id != self.telnyx_call_control_id
            or self.generation.generation != self.lease_claim.generation
            or self.lease_claim.call_control_id != self.telnyx_call_control_id
            or self.lease_claim.call_id != self.call_id
            or self.lease_claim.claimed_at != self.started_at
            or any(
                type(value) is not str or not value
                for value in (
                    self.deployment_id,
                    self.telnyx_call_control_id,
                    self.stream_id,
                )
            )
            or self.telnyx_call_leg_id is not None
            and (
                type(self.telnyx_call_leg_id) is not str
                or not self.telnyx_call_leg_id
            )
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
            raise ValueError("call_construction_grant_invalid") from None

    def __repr__(self) -> str:
        return "CallConstructionGrant()"


class _TerminalCapability:
    """One-shot exact authority transfer from lifecycle owner to cleanup child."""

    __slots__ = ("_cleanup_task", "_grant", "_owner", "_owner_task")

    def __init__(
        self,
        *,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> None:
        self._grant = grant
        self._owner = owner
        self._owner_task = owner_task
        self._cleanup_task: asyncio.Task[None] | None = None

    def __repr__(self) -> str:
        return "TerminalCapability()"

    def transfer_to_cleanup(self, cleanup_task: asyncio.Task[None]) -> None:
        if (
            asyncio.current_task() is not self._owner_task
            or not isinstance(cleanup_task, asyncio.Task)
            or cleanup_task.done()
            or cleanup_task.cancelling() != 0
        ):
            raise RuntimeError("terminal_capability_transfer_failed") from None
        if self._cleanup_task is None:
            self._cleanup_task = cleanup_task
            return
        if self._cleanup_task is not cleanup_task:
            raise RuntimeError("terminal_capability_transfer_failed") from None

    def permits_current_task(self) -> bool:
        current = asyncio.current_task()
        if self._cleanup_task is not None:
            return current is self._cleanup_task
        return current is self._owner_task


@dataclass(frozen=True, slots=True, repr=False)
class TerminalProposal:
    status: Literal["closing", "closed", "failed"]
    reason: str
    metric_class: Literal["closed", "failed", "drained"]
    cleanup_hangup: bool

    def __post_init__(self) -> None:
        if (
            self.status not in {"closing", "closed", "failed"}
            or type(self.reason) is not str
            or not self.reason
            or self.metric_class not in {"closed", "failed", "drained"}
            or type(self.cleanup_hangup) is not bool
        ):
            raise ValueError("terminal_proposal_invalid") from None

    def __repr__(self) -> str:
        return "TerminalProposal()"


@dataclass(frozen=True, slots=True, repr=False)
class TerminalAuthority:
    status: Literal["closing", "closed", "failed"]
    reason: str
    metric_class: Literal["closed", "failed", "drained"]
    cleanup_hangup: bool
    persist_call: bool
    persist_lease: bool
    completion_token: UUID = field(repr=False)
    _entry: _CallEntry = field(repr=False)
    _generation: UUID = field(repr=False)
    _completion_owner: asyncio.Task[object] = field(repr=False)
    _closed_at: datetime = field(repr=False)

    def __repr__(self) -> str:
        return "TerminalAuthority()"


@dataclass(frozen=True, slots=True)
class CommittedWebhookConfirmation:
    disposition: WebhookDisposition
    generation: CallGenerationHandle | None


@dataclass(frozen=True, slots=True)
class FailClosedWebhookConfirmation:
    generation: CallGenerationHandle | None
    abort_scheduled: bool


class _UnauthenticatedPermit:
    __slots__ = ("_gate", "_released")

    def __init__(self, gate: SynchronousUnauthenticatedGate) -> None:
        self._gate = gate
        self._released = False

    def __repr__(self) -> str:
        return "UnauthenticatedPermit()"

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._gate._release()


class SynchronousUnauthenticatedGate:
    """A synchronous closeable permit set bounding only unauthenticated WSS work."""

    __slots__ = ("_capacity", "_closed", "_in_use", "_lock")

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("unauthenticated_gate_config_invalid")
        self._capacity = capacity
        self._closed = False
        self._in_use = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return "SynchronousUnauthenticatedGate()"

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    def try_acquire(self) -> _UnauthenticatedPermit | None:
        with self._lock:
            if self._closed or self._in_use >= self._capacity:
                return None
            self._in_use += 1
        return _UnauthenticatedPermit(self)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _release(self) -> None:
        with self._lock:
            if self._in_use > 0:
                self._in_use -= 1


BackgroundTaskFactory = Callable[[Coroutine[object, object, None], str], asyncio.Task[None]]


class _ProcessCallIdAllocator:
    """Constant-space process-local UUIDv4 allocator for live call identities."""

    __slots__ = ("_counter", "_prefix")

    def __init__(self, *, prefix_factory: Callable[[], int]) -> None:
        prefix = prefix_factory()
        if type(prefix) is not int or not 0 <= prefix < (1 << 90):
            raise ValueError("call_identifier_prefix_invalid") from None
        self._prefix = prefix
        self._counter = 0

    def __repr__(self) -> str:
        return "ProcessCallIdAllocator()"

    def next(self) -> UUID:
        counter = self._counter
        if counter >= 1 << 32:
            raise RuntimeError("call_identifier_exhausted") from None
        self._counter = counter + 1
        payload = (self._prefix << 32) | counter
        high48 = payload >> 74
        mid12 = (payload >> 62) & ((1 << 12) - 1)
        low62 = payload & ((1 << 62) - 1)
        uuid_int = (high48 << 80) | (4 << 76) | (mid12 << 64) | (0b10 << 62) | low62
        return UUID(int=uuid_int)


def _default_background_task_factory(
    coroutine: Coroutine[object, object, None], name: str
) -> asyncio.Task[None]:
    return asyncio.create_task(coroutine, name=name)


class _BackgroundTaskOwner:
    """Synchronous closed-gate task registration for registry cleanup work."""

    __slots__ = ("_closed", "_lock", "_task_factory", "_tasks")

    def __init__(self, task_factory: BackgroundTaskFactory) -> None:
        self._closed = False
        self._lock = threading.Lock()
        self._task_factory = task_factory
        self._tasks: set[asyncio.Task[None]] = set()

    def start(
        self, coroutine: Coroutine[object, object, None], *, name: str
    ) -> asyncio.Task[None] | None:
        start_gate = asyncio.Event()

        async def run() -> None:
            await start_gate.wait()
            await coroutine

        runner = run()
        try:
            with self._lock:
                if self._closed:
                    runner.close()
                    coroutine.close()
                    return None
                task = self._task_factory(runner, name)
                self._tasks.add(task)
        except Exception:
            runner.close()
            coroutine.close()
            return None

        def discard(completed: asyncio.Task[None]) -> None:
            with self._lock:
                self._tasks.discard(completed)

        task.add_done_callback(discard)
        start_gate.set()
        return task

    def spawn(self, coroutine: Coroutine[object, object, None], *, name: str) -> bool:
        return self.start(coroutine, name=name) is not None

    def close_registration(self) -> None:
        with self._lock:
            self._closed = True

    async def join_until_empty(self) -> None:
        while True:
            with self._lock:
                tasks = tuple(self._tasks)
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)


class CallRegistry:
    """One-lock process-local authority for admission, actions, and lease claims."""

    def __init__(
        self,
        *,
        writer: _LeaseWriter,
        call_control: _CallControl,
        tenant_id: str,
        agent_id: str,
        deployment_id: str,
        capacity: int,
        lease_ttl_seconds: int,
        stream_url: str,
        retention_days: int,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        token_factory: Callable[[int], str] = secrets.token_urlsafe,
        prefix_factory: Callable[[], int] = lambda: secrets.randbits(90),
        uuid_factory: Callable[[], UUID] = uuid4,
        candidate_run_id: UUID | None = None,
        candidate_consumed: bool = False,
        admission_expires_at: datetime | None = None,
        qualification_observer: Callable[[Literal["valid", "consumed", "expired"]], None]
        | None = None,
        background_task_factory: BackgroundTaskFactory = _default_background_task_factory,
        sparra: SparraManifestV1 | None = None,
        called_did: str | None = None,
        begin_call: Callable[[str, UUID, RoutingV1], Awaitable[PinnedBeginSnapshot]] | None = None,
    ) -> None:
        if (
            type(capacity) is not int
            or capacity <= 0
            or type(lease_ttl_seconds) is not int
            or lease_ttl_seconds <= 0
            or type(retention_days) is not int
            or retention_days <= 0
            or not all(
                isinstance(value, str) and value for value in (tenant_id, agent_id, deployment_id)
            )
            or not isinstance(stream_url, str)
            or not stream_url.startswith("wss://")
            or candidate_run_id is not None
            and not isinstance(candidate_run_id, UUID)
            or type(candidate_consumed) is not bool
            or candidate_consumed
            and candidate_run_id is None
            or not callable(background_task_factory)
            or qualification_observer is not None
            and not callable(qualification_observer)
            or not callable(prefix_factory)
            or admission_expires_at is not None
            and (
                not isinstance(admission_expires_at, datetime)
                or admission_expires_at.tzinfo is None
                or admission_expires_at.utcoffset() is None
            )
        ):
            raise ValueError("call_registry_config_invalid")
        self._writer = writer
        if sparra is not None and (
            capacity != 1 or retention_days != 30 or called_did is None or not callable(begin_call)
        ):
            raise ValueError("sparra_admission_config_invalid")
        self._operation_contract_version: Literal[1, 2] = (
            1 if sparra is None else sparra.operation_contract_version
        )
        if (
            isinstance(writer, PersistenceWriter)
            and writer.contract_version != self._operation_contract_version
            or self._operation_contract_version == 2 and not isinstance(writer, PersistenceWriter)
        ):
            raise ValueError("sparra_writer_contract_mismatch")
        self._sparra = sparra
        self._called_did = called_did
        self._begin_call = begin_call
        self._call_control = call_control
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._deployment_id = deployment_id
        self._capacity = capacity
        self._lease_ttl_seconds = lease_ttl_seconds
        self._stream_url = stream_url
        self._retention_days = retention_days
        self._retention_delta = timedelta(days=retention_days)
        self._utcnow = utcnow
        self._monotonic = monotonic
        self._token_factory = token_factory
        self._call_id_allocator = _ProcessCallIdAllocator(prefix_factory=prefix_factory)
        self._uuid_factory = uuid_factory
        self._candidate_run_id = candidate_run_id
        self._candidate_consumed = candidate_consumed
        self._admission_expires_at = (
            None if admission_expires_at is None else admission_expires_at.astimezone(UTC)
        )
        self._qualification_observer = qualification_observer
        self._lock = asyncio.Lock()
        self._by_control: dict[str, _CallEntry] = {}
        self._by_call_id: dict[UUID, _CallEntry] = {}
        self._answered_placeholders: dict[str, _AnsweredPlaceholder] = {}
        self._permits_used = 0
        self._background_owner = _BackgroundTaskOwner(background_task_factory)
        self._session_owner_registration_open = True
        self._session_close_task: asyncio.Task[None] | None = None
        self._internal_failure_code: str | None = None
        self._internal_failure_event = asyncio.Event()
        self._draining = False

    def __repr__(self) -> str:
        return "CallRegistry()"

    @property
    def candidate_run_id(self) -> UUID | None:
        return self._candidate_run_id

    @property
    def internal_failure_code(self) -> str | None:
        return self._internal_failure_code

    @property
    def call_control_identity(self) -> object:
        """Expose the process facade identity for composition verification."""

        return self._call_control

    @property
    def internal_failure_event(self) -> asyncio.Event:
        return self._internal_failure_event

    def _set_internal_failure(self, code: str) -> None:
        self._internal_failure_code = code
        self._internal_failure_event.set()

    def _note_terminal_failure(self, code: str) -> None:
        if code in {
            "terminal_persistence_failed",
            "terminal_hangup_failed",
            "terminal_removal_failed",
        }:
            self._set_internal_failure(code)

    def _qualification_expired(self) -> bool:
        if self._admission_expires_at is None:
            return False
        return self._require_aware(self._utcnow()) >= self._admission_expires_at

    @staticmethod
    def _require_aware(value: datetime) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise CallAdmissionRejected("call_event_invalid")
        return value.astimezone(UTC)

    def _new_entry(
        self, event: VerifiedWebhook, *, minted_monotonic: float, routing: RoutingV1 | None = None
    ) -> _CallEntry:
        if event.call_control_id is None:
            raise CallAdmissionRejected("call_event_invalid")
        created_at = self._require_aware(self._utcnow())
        if not isinstance(minted_monotonic, int | float):
            raise CallAdmissionRejected("call_clock_invalid")
        call_id = self._call_id_allocator.next()
        raw_token = self._token_factory(32)
        if not isinstance(raw_token, str) or _TOKEN_PATTERN.fullmatch(raw_token) is None:
            raise CallAdmissionRejected("stream_token_invalid")
        answer_id = self._uuid_factory()
        streaming_id = self._uuid_factory()
        hangup_id = self._uuid_factory()
        identifiers = (answer_id, streaming_id, hangup_id)
        if any(not isinstance(value, UUID) or value.version != 4 for value in identifiers):
            raise CallAdmissionRejected("call_identifier_invalid")
        digest = hashlib.sha256(raw_token.encode("utf-8")).digest()
        expires_at = created_at + timedelta(seconds=self._lease_ttl_seconds)
        return _CallEntry(
            generation=uuid4(),
            call_control_id=event.call_control_id,
            call_id=call_id,
            call_leg_id=event.call_leg_id,
            call_session_id=event.call_session_id,
            initiated_at=routing.admitted_at
            if routing is not None
            else self._require_aware(event.occurred_at),
            created_at=created_at,
            expires_at=expires_at,
            token_deadline=float(minted_monotonic) + self._lease_ttl_seconds,
            token_digest=digest,
            raw_token=raw_token,
            answer=_ActionSlot(answer_id),
            streaming=_ActionSlot(streaming_id),
            hangup_command_id=hangup_id,
            precommit_refcount=1,
            routing=routing,
        )

    def _validate_sparra_routing(self, event: VerifiedWebhook) -> RoutingV1:
        try:
            routing = RoutingV1(
                schema_version=1,
                direction="incoming",
                connection_id=cast(str, event.connection_id),
                to_e164=cast(str, event.to_e164),
                from_e164=event.from_e164,
                telnyx_call_control_id=cast(str, event.call_control_id),
                telnyx_call_leg_id=event.call_leg_id,
                telnyx_call_session_id=event.call_session_id,
                admitted_at=event.occurred_at,
            )
        except ValueError:
            raise CallAdmissionRejected("call_event_invalid") from None
        if (
            self._sparra is None
            or event.connection_id != self._sparra.connection_id
            or event.to_e164 != self._called_did
            or event.direction != "incoming"
            or not -30
            <= (self._require_aware(self._utcnow()) - routing.admitted_at).total_seconds()
            <= 300
        ):
            raise CallAdmissionRejected("call_event_invalid")
        return routing

    async def _ensure_begin_snapshot(self, control_id: str) -> PinnedBeginSnapshot:
        async with self._lock:
            entry = self._by_control.get(control_id)
            if (
                entry is None
                or entry.routing is None
                or entry.terminal_event is not None
                or entry.drain_intent
                or self._draining
                or self._qualification_expired()
                or float(self._monotonic()) >= entry.token_deadline
                or not self._admission_fresh(entry)
                or self._transfer_fenced(entry)
            ):
                raise CallAdmissionRejected("call_event_invalid")
            if entry.begin_snapshot is not None:
                return entry.begin_snapshot
            if entry.begin_task is None:
                entry.begin_future = asyncio.get_running_loop().create_future()
                task = self._background_owner.start(
                    self._begin_future_owned(entry, entry.begin_future), name="voice-begin-call"
                )
                if task is None:
                    raise CallAdmissionRejected("owner_registration_failed")
                entry.begin_task = task
            begin_future = entry.begin_future
        assert begin_future is not None
        return await asyncio.shield(begin_future)

    async def _begin_future_owned(
        self, entry: _CallEntry, future: asyncio.Future[PinnedBeginSnapshot]
    ) -> None:
        try:
            result = await self._begin_owned(entry)
        except BaseException as error:
            if not future.done():
                future.set_exception(error)
                future.exception()
        else:
            if not future.done():
                future.set_result(result)

    async def _begin_owned(self, entry: _CallEntry) -> PinnedBeginSnapshot:
        assert self._begin_call is not None and entry.routing is not None
        remaining = min(
            entry.token_deadline - float(self._monotonic()),
            300 - (self._require_aware(self._utcnow()) - entry.initiated_at).total_seconds(),
        )
        async with asyncio.timeout(max(0.0, remaining)):
            try:
                snapshot = await self._begin_call(self._deployment_id, entry.call_id, entry.routing)
            except OperationSinkCommitAmbiguousError:
                snapshot = await self._begin_call(self._deployment_id, entry.call_id, entry.routing)
        if (
            not isinstance(snapshot, BeginCallSnapshotV2)
            if self._operation_contract_version == 2
            else not isinstance(snapshot, BeginCallSnapshotV1)
        ):
            raise CallAdmissionRejected("call_identity_conflict")
        if (
            snapshot.call_id != entry.call_id
            or snapshot.retention_until != entry.initiated_at + timedelta(days=30)
        ):
            raise CallAdmissionRejected("call_identity_conflict")
        try:
            if isinstance(snapshot, BeginCallSnapshotV2):
                if not isinstance(self._writer, PersistenceWriter):
                    raise CallAdmissionRejected("call_identity_conflict")
                await self._writer.bind_audio_snapshot(snapshot, generation=entry.generation)
            else:
                await cast(Any, self._writer).bind_recording_policy(
                    snapshot, generation=entry.generation
                )
        except PersistenceError:
            raise CallAdmissionRejected("call_identity_conflict") from None
        # Company preference is pinned; native audio activation stays closed
        # until disclosure, recording and retention qualification are complete.
        if isinstance(snapshot, BeginCallSnapshotV1) and snapshot.recording_enabled:
            # Exercise actual owned capacity/readiness before the immutable
            # capability gate. A rejected ON admission must not retain copy authority.
            try:
                await cast(Any, self._writer).reserve_recording_audio(
                    entry.call_id, generation=entry.generation
                )
            except PersistenceError:
                pass
            finally:
                await cast(Any, self._writer).release_recording_audio(
                    entry.call_id, generation=entry.generation
                )
            raise CallAdmissionRejected("sparra_recording_unqualified")
        async with self._lock:
            if (
                self._by_control.get(entry.call_control_id) is not entry
                or entry.terminal_event is not None
                or entry.drain_intent
                or self._draining
                or self._qualification_expired()
                or float(self._monotonic()) >= entry.token_deadline
                or not self._admission_fresh(entry)
            ):
                raise CallAdmissionRejected("call_event_invalid")
            entry.begin_snapshot = snapshot
        return snapshot

    def _admission_fresh(self, entry: _CallEntry) -> bool:
        return (
            -30 <= (self._require_aware(self._utcnow()) - entry.initiated_at).total_seconds() <= 300
        )

    def _qualified_destination(self, entry: _CallEntry) -> str | None:
        if self._sparra is None or entry.begin_snapshot is None:
            return None
        if entry.call_session_id is None:
            return None
        destination = entry.begin_snapshot.transfer_destination
        if (
            destination is None
            or destination != self._sparra.qualified_transfer_destination_e164
            or destination in {self._called_did, self._sparra.original_forward_line_e164}
        ):
            return None
        return destination

    @staticmethod
    def _transfer_fenced(entry: _CallEntry) -> bool:
        return entry.transfer_facts is not None and entry.transfer_facts.transfer_fenced

    async def _hangup_unfenced(self, control_id: str, *, command_id: UUID) -> CallControlResult:
        if self._sparra is None:
            return await self._call_control.hangup(control_id, command_id=command_id)
        async with self._lock:
            entry = self._by_control.get(control_id)
            if entry is not None:
                if self._transfer_fenced(entry):
                    return CallControlResult("accepted")
                entry.cleanup_hangup_started = True
        return await self._call_control.hangup(control_id, command_id=command_id)

    async def restore_transfer_fence(self, stale: Any) -> None:
        facts = stale.lifecycle
        if (
            facts is None
            or not facts.transfer_fenced
            or (facts.admission_generation is None and facts.transfer_generation is None
                and facts.content_departed_generation is None)
        ):
            raise RuntimeError("transfer_recovery_facts_invalid")
        async with self._lock:
            if self._permits_used >= self._capacity:
                raise RuntimeError("transfer_recovery_capacity_conflict")
            entry = _CallEntry(
                generation=(facts.admission_generation or facts.transfer_generation
                            or facts.content_departed_generation),
                call_control_id=stale.call_control_id,
                call_id=stale.call_id,
                call_leg_id=facts.telnyx_call_leg_id,
                call_session_id=facts.telnyx_call_session_id,
                initiated_at=facts.admitted_at,
                created_at=stale.created_at,
                expires_at=stale.expires_at,
                token_deadline=0.0,
                token_digest=stale.token_hash,
                raw_token=None,
                answer=_ActionSlot(uuid4()),
                streaming=_ActionSlot(uuid4()),
                hangup_command_id=uuid4(),
                durable=True,
                lease_state="active",
                answered_at=facts.started_at,
                transfer_facts=facts,
                answer_evidence=facts.started_at is not None,
                no_new_ai=True,
                attached=True,
            )
            self._by_control[entry.call_control_id] = entry
            self._by_call_id[entry.call_id] = entry
            self._permits_used += 1

    async def human_tool_available(self, generation: CallGenerationHandle) -> bool:
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            return (
                entry is not None
                and entry.generation == generation.generation
                and self._qualified_destination(entry) is not None
            )

    async def stop_call_content(self, call_id: UUID, *, delete_content: bool = True) -> None:
        """Revoke AI content while keeping actual telephone capacity/correlation."""
        async with self._lock:
            entry = self._by_call_id.get(call_id)
            if entry is None or entry.lease_state == "terminal":
                return
            entry.no_new_ai = True
            entry.drain_intent = True
            stop = getattr(entry.session, "stop_new_ai", None)
            if stop is not None:
                stop()
            facts = entry.transfer_facts or LocalCallLifecycleFacts(
                entry.call_id, entry.initiated_at, entry.initiated_at + timedelta(days=30),
                entry.call_leg_id, entry.call_session_id,
            )
            entry.transfer_facts = replace(
                facts,
                content_departed_generation=entry.generation,
                local_closing_at=facts.local_closing_at or self._require_aware(self._utcnow()),
            )
            if entry.content_stop_task is None:
                entry.content_stop_task = self._background_owner.start(
                    self._stop_call_content_owned(entry), name="voice-call-content-stop"
                )
                if entry.content_stop_task is None:
                    raise RuntimeError("content_stop_owner_closed")
            task = entry.content_stop_task
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = error
        await task
        if delete_content:
            await cast(Any, self._writer).erase_call_content(
                entry.call_id, now=self._require_aware(self._utcnow()), generation=entry.generation
            )
        if cancellation is not None:
            raise cancellation

    async def _stop_call_content_owned(self, entry: _CallEntry) -> None:
        async with self._lock:
            owner = entry.lifecycle_owner
            tasks = tuple(
                dict.fromkeys(
                    task
                    for task in (
                        entry.begin_task,
                        entry.answer.owner_task,
                        entry.streaming.owner_task,
                    )
                    if task is not None and task is not asyncio.current_task()
                )
            )
        try:
            await cast(Any, self._writer).mark_call_departed(
                entry.call_id, now=self._require_aware(self._utcnow())
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # Even a persistence failure must close the already-revoked AI resources.
            if owner is not None and self._valid_session_owner(owner):
                cast(_SessionLifecycleOwner, owner).request_drain("content_erased")
                await cast(_SessionLifecycleOwner, owner).wait()
            async with self._lock:
                self._clear_content_holders_locked(entry, clear_cached_content=True)

    @staticmethod
    def _clear_content_holders_locked(
        entry: _CallEntry, *, clear_cached_content: bool = False
    ) -> None:
        if clear_cached_content:
            entry.bridge_publication = None
            if (
                entry.transfer_facts is not None
                and entry.transfer_facts.disclosure_evidence is not None
            ):
                entry.transfer_facts = replace(entry.transfer_facts, disclosure_evidence=None)
        entry.begin_snapshot = None
        entry.routing = None
        entry.begin_future = None
        entry.begin_task = None
        entry.construction_grant = None
        entry.terminal_capability = None
        entry.claim = None
        entry.raw_token = None
        entry.session = None
        entry.lifecycle_owner = None
        entry.lifecycle_owner_task = None

    async def request_human(self, generation: CallGenerationHandle) -> str:
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            if (
                entry is None
                or entry.generation != generation.generation
                or entry.cleanup_hangup_started
                or entry.terminal_authority is not None
                or entry.terminal_event is not None
                or self._draining
            ):
                return "unavailable_collect_message"
            destination = self._qualified_destination(entry)
            if destination is None:
                return "unavailable_collect_message"
            if entry.transfer_task is None:
                stop_result = getattr(entry.session, "stop_result_inference", None)
                if stop_result is not None:
                    stop_result()
                pin = entry.begin_snapshot
                if isinstance(pin, BeginCallSnapshotV2) and (
                    pin.audio_available and pin.recording_policy == "local_30d"
                ):
                    close_audio = getattr(entry.session, "close_audio_for_transfer", None)
                    if not callable(close_audio) or close_audio() is not True:
                        return "unavailable_collect_message"
                assert entry.routing is not None
                command_id = uuid4()
                correlation = base64.b64encode(uuid4().bytes + command_id.bytes).decode("ascii")
                # Reservation precedes the first writer await; terminal cleanup observes this fence.
                entry.transfer_facts = LocalCallLifecycleFacts(
                    entry.call_id,
                    entry.initiated_at,
                    entry.initiated_at + timedelta(days=30),
                    entry.call_leg_id,
                    entry.call_session_id,
                    transfer_command_id=command_id,
                    transfer_correlation=correlation,
                    transfer_generation=entry.generation,
                    transfer_connection_sha256=hashlib.sha256(
                        entry.routing.connection_id.encode()
                    ).hexdigest(),
                    transfer_destination_sha256=hashlib.sha256(destination.encode()).hexdigest(),
                )
                entry.transfer_future = asyncio.get_running_loop().create_future()
                task = self._background_owner.start(
                    self._transfer_future_owned(entry, destination, entry.transfer_future),
                    name="voice-qualified-transfer",
                )
                if task is None:
                    entry.transfer_facts = None
                    return "unavailable_collect_message"
                entry.transfer_task = task
            transfer_future = entry.transfer_future
        assert transfer_future is not None
        return await asyncio.shield(transfer_future)

    async def _transfer_future_owned(
        self, entry: _CallEntry, destination: str, future: asyncio.Future[str]
    ) -> None:
        try:
            result = await self._transfer_owned(entry, destination)
        except BaseException as error:
            if not future.done():
                future.set_exception(error)
                future.exception()
        else:
            if not future.done():
                future.set_result(result)

    async def _transfer_owned(self, entry: _CallEntry, destination: str) -> str:
        facts = entry.transfer_facts
        assert facts is not None and facts.transfer_command_id is not None
        assert facts.transfer_correlation is not None
        pin = entry.begin_snapshot
        requires_local_audio = isinstance(pin, BeginCallSnapshotV2) and (
            pin.audio_available and pin.recording_policy == "local_30d"
        )
        commit_entered = False
        try:
            if requires_local_audio:
                prepare_audio = getattr(entry.session, "prepare_audio_for_transfer", None)
                if not callable(prepare_audio):
                    return "unavailable_collect_message"
                try:
                    prepared = await prepare_audio()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return "unavailable_collect_message"
                if prepared is not True:
                    return "unavailable_collect_message"
                async with self._lock:
                    if (
                        self._by_control.get(entry.call_control_id) is not entry
                        or entry.transfer_facts is not facts
                        or entry.terminal_authority is not None
                        or entry.terminal_event is not None
                        or entry.cleanup_hangup_started
                        or entry.drain_intent
                        or self._draining
                    ):
                        return "unavailable_collect_message"
                    audio_ready = getattr(entry.session, "audio_ready_for_transfer", None)
                    if not callable(audio_ready) or audio_ready() is not True:
                        return "unavailable_collect_message"
            commit_entered = True
            await cast(Any, self._writer).commit_transfer_intent(facts)
        finally:
            if not commit_entered:
                async with self._lock:
                    if (
                        self._by_control.get(entry.call_control_id) is entry
                        and entry.transfer_facts is facts
                        and entry.terminal_authority is None
                    ):
                        entry.transfer_facts = None
        async with self._lock:
            if (
                self._by_control.get(entry.call_control_id) is not entry
                or entry.transfer_facts is not facts
                or entry.terminal_event == "call.hangup"
                or not getattr(self._call_control, "dispatch_available", True)
            ):
                return "unavailable_collect_message"
            if requires_local_audio:
                if (
                    entry.terminal_authority is not None or entry.terminal_event is not None
                    or entry.cleanup_hangup_started or entry.drain_intent or self._draining
                ):
                    return "unavailable_collect_message"
                audio_ready = getattr(entry.session, "audio_ready_for_transfer", None)
                if not callable(audio_ready) or audio_ready() is not True:
                    return "unavailable_collect_message"
        # A durable pending intent fences cleanup even after local failure/drain.
        try:
            result = await cast(Any, self._call_control).transfer(
                entry.call_control_id,
                TransferRequestV1(
                    to_e164=destination, target_leg_client_state=facts.transfer_correlation
                ),
                command_id=facts.transfer_command_id,
            )
        except Exception:
            return "outcome_unknown"
        observed = entry.transfer_facts
        if entry.no_new_ai:
            return (
                "qualified_line_connected"
                if observed is not None and observed.qualified_line_bridged_at is not None
                else "outcome_unknown"
            )
        if observed is not None and observed.transfer_failed_at is not None:
            return f"{observed.transfer_failure_cause or 'target_hangup'}_collect_message"
        return "ringing" if result.outcome == "accepted" else "outcome_unknown"

    async def _resolve_transfer_event(self, event: VerifiedWebhook) -> ResolvedWebhook | None:
        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

        if self._sparra is None or event.event_type not in {
            "call.initiated",
            "call.answered",
            "call.bridged",
            "call.hangup",
        }:
            return None
        entry: _CallEntry | None = None
        session: object | None = None
        async with self._lock:
            for candidate in self._by_control.values():
                facts = candidate.transfer_facts
                if (
                    facts is None
                    or not facts.transfer_fenced
                    or event.call_control_id == candidate.call_control_id
                ):
                    continue
                token = (
                    None if event.client_state is None else event.client_state.get_secret_value()
                )
                if (
                    token != facts.transfer_correlation
                    or event.call_session_id != candidate.call_session_id
                    or event.connection_id is None
                    or event.to_e164 is None
                    or hashlib.sha256(event.connection_id.encode()).hexdigest()
                    != facts.transfer_connection_sha256
                    or hashlib.sha256(event.to_e164.encode()).hexdigest()
                    != facts.transfer_destination_sha256
                ):
                    continue
                if facts.target_call_control_id is None:
                    if (
                        event.event_type != "call.initiated"
                        or event.direction != "outgoing"
                        or event.call_leg_id is None
                    ):
                        return ResolvedWebhook(None)
                    facts = replace(
                        facts,
                        target_call_control_id=event.call_control_id,
                        target_call_leg_id=event.call_leg_id,
                    )
                elif (
                    event.call_control_id != facts.target_call_control_id
                    or event.call_leg_id != facts.target_call_leg_id
                ):
                    return ResolvedWebhook(None)
                if event.event_type == "call.bridged":
                    if facts.qualified_line_bridged_at is None:
                        facts = replace(facts, qualified_line_bridged_at=event.occurred_at)
                    candidate.no_new_ai = True
                    session = candidate.session
                    stop = getattr(session, "stop_new_ai", None)
                    if callable(stop):
                        stop()
                elif event.event_type == "call.hangup" and facts.qualified_line_bridged_at is None:
                    cause = (
                        event.hangup_cause
                        if event.hangup_cause
                        in {"user_busy", "no_answer", "timeout", "call_rejected"}
                        else "target_hangup"
                    )
                    facts = replace(
                        facts, transfer_failed_at=event.occurred_at, transfer_failure_cause=cause
                    )
                    if (
                        candidate.terminal_authority is not None
                        and candidate.terminal_authority.status == "closing"
                    ):
                        facts = replace(
                            facts, local_closing_at=candidate.terminal_authority._closed_at
                        )
                candidate.transfer_facts = facts
                entry = candidate
                break
        if entry is None:
            if event.direction == "outgoing" or event.event_type == "call.bridged":
                return ResolvedWebhook(None)
            return None
        operation: VoiceOperationV1 | VoiceOperationV2 | None = None
        assert facts is not None
        if session is not None:
            await cast(Any, session).request_drain("qualified_line_connected")
        if facts.qualified_line_bridged_at is not None:
            bridged_at = facts.qualified_line_bridged_at
            lifecycle = await cast(Any, self._writer).read_call_lifecycle(entry.call_id)
            async with self._lock:
                started = lifecycle.started_at if lifecycle is not None else None
                started = started or entry.answered_at or entry.claimed_at
                if started is None:
                    raise CallAdmissionRejected("call_identity_conflict")
                # A writer read/drain can cross explicit content stop. Recheck the
                # registered entry under its lock before retaining any payload.
                facts = entry.transfer_facts or facts
                content_stopped = (
                    entry.content_stop_task is not None
                    or facts.content_erased
                    or lifecycle is not None and lifecycle.content_erased
                )
                if content_stopped:
                    entry.bridge_publication = None
                    facts = replace(facts, disclosure_evidence=None)
                if (
                    not content_stopped
                    and lifecycle is not None
                    and lifecycle.bridge_operation_id is None
                    and facts.bridge_operation_id is None
                ):
                    payload = CallUpsertPayloadV1(
                            telnyx_call_control_id=entry.call_control_id,
                            telnyx_call_leg_id=entry.call_leg_id,
                            telnyx_call_session_id=entry.call_session_id,
                            status="closing",
                            disclosure_state="completed"
                            if lifecycle is not None
                            and lifecycle.disclosure_evidence is not None
                            and lifecycle.disclosure_evidence.completed_at is not None
                            else "pending",
                            started_at=started,
                            ended_at=None,
                            end_reason="qualified_line_connected",
                            retention_until=entry.initiated_at + timedelta(days=30),
                            **(
                                cast(
                                    Any,
                                    (
                                        {"disclosure_evidence": lifecycle.disclosure_evidence}
                                        if lifecycle is not None
                                        and lifecycle.disclosure_evidence is not None
                                        else {}
                                    ),
                                )
                            ),
                    )
                    operation = entry.bridge_publication
                    if operation is None:
                        if self._operation_contract_version == 2:
                            operation = VoiceOperationV2(schema_version=2,
                                operation_id=uuid5(entry.call_id, "qualified-line-bridge"),
                                deployment_id=self._deployment_id, call_id=entry.call_id,
                                occurred_at=bridged_at, kind="call.upsert", payload=payload)
                        else:
                            operation = VoiceOperationV1(schema_version=1,
                                operation_id=uuid5(entry.call_id, "qualified-line-bridge"),
                                deployment_id=self._deployment_id, call_id=entry.call_id,
                                occurred_at=bridged_at, kind="call.upsert", payload=payload)
                    entry.bridge_publication = operation
                    facts = replace(facts, bridge_operation_id=operation.operation_id)
                elif lifecycle is not None and lifecycle.bridge_operation_id is not None:
                    facts = replace(facts, bridge_operation_id=lifecycle.bridge_operation_id)
                entry.transfer_facts = facts
        if self._operation_contract_version == 2:
            if not isinstance(self._writer, PersistenceWriter):
                raise CallAdmissionRejected("call_identity_conflict")
            if isinstance(operation, VoiceOperationV1):
                raise CallAdmissionRejected("call_identity_conflict")
            await self._writer.commit_transfer_observation_v2(
                facts, operation, generation=entry.generation
            )
        else:
            await cast(Any, self._writer).commit_transfer_observation(facts, operation)
        return ResolvedWebhook(None)

    def _pending_effect(self, entry: _CallEntry) -> WebhookDurableEffect:
        from projetv0_voice.telnyx.webhooks import WebhookDurableEffect

        lease: Mapping[str, object] = {
            "action": "upsert",
            "call_control_id": entry.call_control_id,
            "call_id": entry.call_id,
            "tenant_id": self._tenant_id,
            "agent_id": self._agent_id,
            "state": "pending",
            "token_hash": entry.token_digest,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "closed_at": None,
        }
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=entry.call_id,
            deployment_id=self._deployment_id,
            call_id=entry.call_id,
            occurred_at=entry.initiated_at,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=entry.call_control_id,
                telnyx_call_leg_id=entry.call_leg_id,
                telnyx_call_session_id=entry.call_session_id,
                status="pending",
                disclosure_state="pending",
                started_at=None,
                ended_at=None,
                end_reason=None,
                retention_until=entry.initiated_at + timedelta(days=self._retention_days),
            ),
        )
        facts = (
            None
            if entry.routing is None
            else LocalCallAdmissionFacts(
                entry.call_id,
                entry.initiated_at,
                entry.initiated_at + timedelta(days=30),
                entry.call_leg_id,
                entry.call_session_id,
                admission_generation=entry.generation,
            )
        )
        return WebhookDurableEffect(
            lease=lease,
            operation=operation if self._operation_contract_version == 1 else None,
            admission_facts=facts,
        )

    def _terminal_effect(self, entry: _CallEntry, event: VerifiedWebhook) -> WebhookDurableEffect:
        from projetv0_voice.telnyx.webhooks import WebhookDurableEffect

        closed_at = self._require_aware(event.occurred_at)
        lease: Mapping[str, object] = {
            "action": "upsert",
            "call_control_id": entry.call_control_id,
            "call_id": entry.call_id,
            "tenant_id": self._tenant_id,
            "agent_id": self._agent_id,
            "state": "terminal",
            "token_hash": entry.token_digest,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "closed_at": closed_at,
        }
        if (
            entry.routing is not None or entry.transfer_facts is not None
        ) and closed_at >= entry.initiated_at + self._retention_delta:
            return WebhookDurableEffect(
                lease=lease,
                operation_generation=(
                    entry.generation if self._operation_contract_version == 2 else None
                ),
            )
        actual_start = entry.answered_at or entry.claimed_at
        answered = actual_start is not None
        payload = CallUpsertPayloadV1(
                telnyx_call_control_id=entry.call_control_id,
                telnyx_call_leg_id=entry.call_leg_id,
                telnyx_call_session_id=entry.call_session_id,
                status="closed" if answered else "failed",
                disclosure_state="failed",
                started_at=actual_start,
                ended_at=closed_at,
                end_reason="telnyx_hangup",
                retention_until=entry.initiated_at + timedelta(days=self._retention_days),
        )
        operation: VoiceOperationV1 | VoiceOperationV2
        if self._operation_contract_version == 2:
            operation = VoiceOperationV2(schema_version=2,
                operation_id=uuid5(entry.call_id, event.event_id),
                deployment_id=self._deployment_id,
                call_id=entry.call_id, occurred_at=closed_at, kind="call.upsert", payload=payload)
        else:
            operation = VoiceOperationV1(schema_version=1,
                operation_id=uuid5(entry.call_id, event.event_id),
                deployment_id=self._deployment_id,
                call_id=entry.call_id, occurred_at=closed_at, kind="call.upsert", payload=payload)
        return WebhookDurableEffect(
            lease=lease, operation=operation,
            operation_generation=(
                entry.generation if self._operation_contract_version == 2 else None
            ),
        )

    async def resolve_webhook(self, event: VerifiedWebhook) -> ResolvedWebhook:
        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

        target = await self._resolve_transfer_event(event)
        if target is not None:
            return target
        routing = (
            self._validate_sparra_routing(event)
            if event.event_type == "call.initiated" and self._sparra is not None
            else None
        )
        if event.event_type in {"call.initiated", "call.answered"}:
            async with self._lock:
                if self._draining:
                    raise CallAdmissionRejected("call_draining")
        if event.event_type == "call.answered":
            return await self._resolve_answered(event)
        if event.event_type == "call.hangup":
            return await self._resolve_hangup(event)
        if event.event_type != "call.initiated":
            return ResolvedWebhook(None)
        if (
            event.call_control_id is None
            or event.direction != "incoming"
            or event.call_state != "parked"
        ):
            raise CallAdmissionRejected("call_event_invalid")
        entry: _CallEntry
        async with self._lock:
            existing = self._by_control.get(event.call_control_id)
            if existing is not None:
                if (
                    existing.terminal_event is not None
                    or self._transfer_fenced(existing)
                    or existing.call_leg_id != event.call_leg_id
                    or existing.call_session_id != event.call_session_id
                    or existing.initiated_at
                    != (
                        routing.admitted_at
                        if routing is not None
                        else event.occurred_at.astimezone(UTC)
                    )
                    or existing.routing is not None
                    and (
                        existing.routing.connection_id != event.connection_id
                        or existing.routing.to_e164 != event.to_e164
                        or existing.routing.from_e164 != event.from_e164
                    )
                ):
                    raise CallAdmissionRejected("call_identity_conflict")
                existing.precommit_refcount += 1
                entry = existing
            else:
                if self._qualification_expired():
                    raise CallAdmissionRejected("qualification_window_expired")
                if self._candidate_consumed:
                    raise CallAdmissionRejected("qualification_run_consumed")
                if self._permits_used >= self._capacity:
                    raise CallAdmissionRejected("call_capacity_reached")
                placeholder = self._answered_placeholders.get(event.call_control_id)
                if placeholder is not None and (
                    placeholder.call_leg_id != event.call_leg_id
                    or placeholder.call_session_id != event.call_session_id
                ):
                    raise CallAdmissionRejected("call_identity_conflict")
                minted_monotonic = self._monotonic()
                if not isinstance(minted_monotonic, int | float):
                    raise CallAdmissionRejected("call_clock_invalid")
                now = float(minted_monotonic)
                if placeholder is not None and now >= placeholder.deadline:
                    if self._answered_placeholders.get(event.call_control_id) is placeholder:
                        self._answered_placeholders.pop(event.call_control_id, None)
                    placeholder.linked_entry = None
                    placeholder = None
                entry = self._new_entry(event, minted_monotonic=now, routing=routing)
                if placeholder is not None:
                    if placeholder.durable:
                        entry.answer_evidence = True
                        entry.answered_at = placeholder.answered_at
                        entry.answer.state = "accepted"
                        if placeholder.precommit_refcount == 0:
                            self._answered_placeholders.pop(event.call_control_id, None)
                    else:
                        placeholder.linked_entry = entry
                        placeholder.linked_abort_identity = _LinkedAbortIdentity(
                            call_control_id=entry.call_control_id,
                            generation=entry.generation,
                            token_digest=entry.token_digest,
                        )
                self._by_control[entry.call_control_id] = entry
                self._by_call_id[entry.call_id] = entry
                self._permits_used += 1
        reservation = await self._register_call_reservation(entry, event.event_type)
        return ResolvedWebhook(self._pending_effect(entry), reservation)

    async def resolve_duplicate_webhook(self, event: VerifiedWebhook) -> ResolvedWebhook:
        """Resolve a classified duplicate without creating process-local identity."""

        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

        transfer = await self._resolve_transfer_event(event)
        if transfer is not None:
            return transfer

        async with self._lock:
            if self._draining:
                raise CallAdmissionRejected("call_draining")
        if event.call_control_id is None or event.event_type not in {
            "call.initiated",
            "call.answered",
            "call.hangup",
        }:
            return ResolvedWebhook(None)
        entry: _CallEntry | None = None
        placeholder: _AnsweredPlaceholder | None = None
        async with self._lock:
            entry = self._by_control.get(event.call_control_id)
            if entry is not None:
                if event.event_type == "call.hangup" and not self._session_owner_registration_open:
                    return ResolvedWebhook(None)
                if (
                    entry.call_leg_id != event.call_leg_id
                    or entry.call_session_id != event.call_session_id
                    or event.event_type == "call.initiated"
                    and entry.initiated_at != event.occurred_at.astimezone(UTC)
                ):
                    raise CallAdmissionRejected("call_identity_conflict")
                if entry.terminal_event is not None:
                    return ResolvedWebhook(None)
                entry.precommit_refcount += 1
                if event.event_type == "call.hangup":
                    entry.terminal_pending_count += 1
                    entry.terminal_state = "external_pending"
                    entry.terminal_settled_event.clear()
            elif event.event_type == "call.answered":
                placeholder = self._answered_placeholders.get(event.call_control_id)
                if placeholder is not None:
                    if (
                        placeholder.call_leg_id != event.call_leg_id
                        or placeholder.call_session_id != event.call_session_id
                    ):
                        raise CallAdmissionRejected("call_identity_conflict")
                    placeholder.precommit_refcount += 1
        if entry is not None:
            effect = (
                self._terminal_effect(entry, event)
                if event.event_type == "call.hangup"
                else self._pending_effect(entry)
                if event.event_type == "call.initiated"
                else None
            )
            return ResolvedWebhook(
                effect,
                await self._register_call_reservation(
                    entry,
                    event.event_type,
                    terminal_occurred_at=(
                        event.occurred_at if event.event_type == "call.hangup" else None
                    ),
                ),
            )
        if placeholder is not None:
            return ResolvedWebhook(None, await self._register_placeholder_reservation(placeholder))
        return ResolvedWebhook(None)

    async def _resolve_answered(self, event: VerifiedWebhook) -> ResolvedWebhook:
        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

        if event.call_control_id is None or event.call_state != "answered":
            raise CallAdmissionRejected("call_event_invalid")
        call_entry: _CallEntry | None = None
        placeholder: _AnsweredPlaceholder | None = None
        async with self._lock:
            if self._qualification_expired():
                raise CallAdmissionRejected("qualification_window_expired")
            entry = self._by_control.get(event.call_control_id)
            if entry is not None:
                if (
                    entry.terminal_event is not None
                    or entry.call_leg_id != event.call_leg_id
                    or entry.call_session_id != event.call_session_id
                ):
                    raise CallAdmissionRejected("call_identity_conflict")
                entry.precommit_refcount += 1
                call_entry = entry
            else:
                existing = self._answered_placeholders.get(event.call_control_id)
            if entry is None and existing is not None:
                if (
                    existing.call_leg_id != event.call_leg_id
                    or existing.call_session_id != event.call_session_id
                ):
                    raise CallAdmissionRejected("call_identity_conflict")
                existing.precommit_refcount += 1
                placeholder = existing
            elif entry is None:
                if len(self._answered_placeholders) >= self._capacity:
                    raise CallAdmissionRejected("placeholder_capacity_reached")
                first_seen = float(self._monotonic())
                placeholder = _AnsweredPlaceholder(
                    call_control_id=event.call_control_id,
                    call_leg_id=event.call_leg_id,
                    call_session_id=event.call_session_id,
                    first_seen=first_seen,
                    deadline=first_seen + self._lease_ttl_seconds,
                    answered_at=event.occurred_at.astimezone(UTC),
                )
                self._answered_placeholders[event.call_control_id] = placeholder
        if call_entry is not None:
            return ResolvedWebhook(
                None, await self._register_call_reservation(call_entry, event.event_type)
            )
        if placeholder is None:
            raise RuntimeError("placeholder_resolution_failed")
        return ResolvedWebhook(None, await self._register_placeholder_reservation(placeholder))

    async def _resolve_hangup(self, event: VerifiedWebhook) -> ResolvedWebhook:
        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

        if event.call_control_id is None:
            raise CallAdmissionRejected("call_event_invalid")
        entry: _CallEntry | None
        async with self._lock:
            entry = self._by_control.get(event.call_control_id)
            if (
                entry is None
                or entry.terminal_event is not None
                and not self._transfer_fenced(entry)
                or not self._session_owner_registration_open
                and not self._transfer_fenced(entry)
            ):
                return ResolvedWebhook(None)
            if (
                entry.call_leg_id != event.call_leg_id
                or entry.call_session_id != event.call_session_id
            ):
                raise CallAdmissionRejected("call_identity_conflict")
            entry.precommit_refcount += 1
            entry.terminal_pending_count += 1
            entry.terminal_state = "external_pending"
            entry.terminal_settled_event.clear()
        reservation = await self._register_call_reservation(
            entry,
            event.event_type,
            terminal_occurred_at=event.occurred_at,
        )
        effect = self._terminal_effect(entry, event)
        if entry.routing is not None or entry.transfer_facts is not None:
            facts = await cast(Any, self._writer).read_call_lifecycle(entry.call_id)
            if facts is not None and effect.operation is not None:
                payload = cast(CallUpsertPayloadV1, effect.operation.payload)
                starts = [
                    value
                    for value in (facts.started_at, entry.answered_at, entry.claimed_at)
                    if value is not None
                ]
                started = min(starts) if starts else None
                payload = payload.model_copy(
                    update={
                        "started_at": started,
                        "status": "closed" if started is not None else "failed",
                        "disclosure_state": "completed"
                        if facts.disclosure_evidence is not None
                        and facts.disclosure_evidence.completed_at is not None
                        else "failed",
                        **(
                            {"disclosure_evidence": facts.disclosure_evidence}
                            if facts.disclosure_evidence is not None
                            else {}
                        ),
                    }
                )
                effect = replace(
                    effect,
                    operation=effect.operation.model_copy(update={"payload": payload}),
                )
        return ResolvedWebhook(effect, reservation)

    async def _register_call_reservation(
        self,
        entry: _CallEntry,
        event_type: str,
        *,
        terminal_occurred_at: datetime | None = None,
    ) -> CallReservation:
        reservation = CallReservation(
            self,
            entry,
            event_type,
            terminal_occurred_at=terminal_occurred_at,
        )
        abandonment = self._background_owner.start(
            reservation._run_abandonment(), name="voice-reservation-owner"
        )
        if abandonment is not None:
            reservation._abandonment_task = abandonment
            return reservation
        self._set_internal_failure("background_task_registration_failed")
        await self._settle_failed_call_registration(reservation)
        raise CallAdmissionRejected("owner_registration_failed")

    async def _settle_failed_call_registration(
        self, reservation: CallReservation
    ) -> None:
        cancellation_seen = False
        while not reservation._registry_applied:
            try:
                await self._settle_reservation(reservation, None)
            except asyncio.CancelledError:
                cancellation_seen = True
        if cancellation_seen:
            raise asyncio.CancelledError

    async def _register_placeholder_reservation(
        self, placeholder: _AnsweredPlaceholder
    ) -> _PlaceholderReservation:
        reservation = _PlaceholderReservation(self, placeholder)
        abandonment = self._background_owner.start(
            reservation._run_abandonment(), name="voice-placeholder-owner"
        )
        if abandonment is not None:
            reservation._abandonment_task = abandonment
            return reservation
        self._set_internal_failure("background_task_registration_failed")
        await self._settle_failed_placeholder_registration(reservation)
        raise CallAdmissionRejected("owner_registration_failed")

    async def _settle_failed_placeholder_registration(
        self, reservation: _PlaceholderReservation
    ) -> None:
        cancellation_seen = False
        while not reservation._registry_applied:
            try:
                await self._settle_placeholder(reservation, None)
            except asyncio.CancelledError:
                cancellation_seen = True
        if cancellation_seen:
            raise asyncio.CancelledError

    def close_registration(self) -> None:
        self._background_owner.close_registration()

    async def join_until_empty(self) -> None:
        await self._background_owner.join_until_empty()

    async def wait_background(self) -> None:
        await self.join_until_empty()

    async def _settle_reservation(
        self,
        reservation: CallReservation,
        result: WebhookCommitValue | None,
    ) -> None:
        terminal_action_owners: tuple[asyncio.Task[object], ...] = ()
        qualification_consumed = False
        async with self._lock:
            if reservation._registry_applied:
                return
            entry = self._by_control.get(reservation._entry.call_control_id)
            if entry is not reservation._entry or entry.generation != reservation._generation:
                reservation._registry_applied = True
                return
            if entry.precommit_refcount > 0:
                entry.precommit_refcount -= 1
            if reservation._event_type == "call.hangup":
                if entry.terminal_pending_count > 0:
                    entry.terminal_pending_count -= 1
                if (
                    isinstance(result, WebhookCommitResult)
                    and result.effect in {"applied", "duplicate", "existing_terminal"}
                    and (
                        entry.terminal_authority is None
                        or entry.terminal_authority.status == "closing"
                    )
                ):
                    completion_owner = asyncio.current_task()
                    if completion_owner is None:
                        raise RuntimeError("terminal_owner_unavailable")
                    closed_at = self._require_aware(
                        reservation._terminal_occurred_at or self._utcnow()
                    )
                    authority = TerminalAuthority(
                        status="closed" if entry.answer_evidence else "failed",
                        reason="telnyx_hangup",
                        metric_class=("closed" if entry.answer_evidence else "failed"),
                        cleanup_hangup=False,
                        persist_call=False,
                        persist_lease=False,
                        completion_token=self._uuid_factory(),
                        _entry=entry,
                        _generation=entry.generation,
                        _completion_owner=cast(asyncio.Task[object], completion_owner),
                        _closed_at=closed_at,
                    )
                    entry.terminal_authority = authority
                    entry.terminal_completion_owner = cast(asyncio.Task[object], completion_owner)
                    entry.terminal_state = "reserved"
                    entry.terminal_event = "call.hangup"
                    entry.lease_state = "terminal"
                    entry.session_phase = "terminal"
                    entry.raw_token = None
                    entry.drain_intent = True
                    entry.terminal_settled_event.set()
                    current_task = asyncio.current_task()
                    action_owners: list[asyncio.Task[object]] = []
                    from projetv0_voice.telnyx.webhooks import WebhookDisposition

                    for slot in (entry.answer, entry.streaming):
                        if slot.state == "in_flight":
                            slot.state = "unknown"
                            slot.disposition = 200
                        if slot.completion is not None and not slot.completion.done():
                            slot.completion.set_result(WebhookDisposition(slot.disposition))
                        if slot.owner_task is not None and slot.owner_task is not current_task:
                            action_owners.append(slot.owner_task)
                    terminal_action_owners = tuple(dict.fromkeys(action_owners))
                if entry.terminal_pending_count == 0:
                    if entry.terminal_authority is None:
                        entry.terminal_state = "open"
                    entry.terminal_settled_event.set()
            if (
                reservation._event_type == "call.initiated"
                and isinstance(result, WebhookCommitResult)
                and (
                    result.receipt,
                    result.effect,
                )
                == ("first", "applied")
            ):
                entry.durable = True
                entry.lease_state = "pending"
                if self._candidate_run_id is not None and not self._candidate_consumed:
                    self._candidate_consumed = True
                    qualification_consumed = True
            elif isinstance(result, QualificationRunConsumed) and not self._candidate_consumed:
                self._candidate_consumed = True
                qualification_consumed = True
            if entry.precommit_refcount == 0 and not entry.durable:
                entry.raw_token = None
                if not entry.capacity_released:
                    entry.capacity_released = True
                    self._permits_used -= 1
                self._by_control.pop(entry.call_control_id, None)
                self._by_call_id.pop(entry.call_id, None)
            reservation._registry_applied = True
        if qualification_consumed and self._qualification_observer is not None:
            with contextlib.suppress(Exception):
                self._qualification_observer("consumed")
        for owner_task in terminal_action_owners:
            owner_task.cancel()
        if terminal_action_owners:
            await asyncio.gather(*terminal_action_owners, return_exceptions=True)

    async def _settle_placeholder(
        self,
        reservation: _PlaceholderReservation,
        result: WebhookCommitValue | None,
        *,
        promote_linked_evidence: bool = True,
        linked_abort_identity: _LinkedAbortIdentity | None = None,
    ) -> _AbortTarget | None:
        completion: asyncio.Future[WebhookDisposition] | None = None
        abort_target: _AbortTarget | None = None
        async with self._lock:
            if reservation._registry_applied:
                return None
            placeholder = reservation._placeholder
            mapped_placeholder = self._answered_placeholders.get(placeholder.call_control_id)
            is_live_placeholder = mapped_placeholder is placeholder
            if placeholder.precommit_refcount > 0:
                placeholder.precommit_refcount -= 1
            if isinstance(result, WebhookCommitResult) and (
                result.receipt,
                result.effect,
            ) == ("first", "applied"):
                now = float(self._monotonic())
                if is_live_placeholder and now >= placeholder.deadline:
                    if self._answered_placeholders.get(placeholder.call_control_id) is placeholder:
                        self._answered_placeholders.pop(placeholder.call_control_id, None)
                    placeholder.linked_entry = None
                    is_live_placeholder = False
                elif is_live_placeholder:
                    placeholder.durable = True
                    linked = placeholder.linked_entry
                    if (
                        promote_linked_evidence
                        and linked is not None
                        and now < linked.token_deadline
                        and self._by_control.get(linked.call_control_id) is linked
                        and linked.terminal_event is None
                    ):
                        linked.answer_evidence = True
                        linked.answered_at = placeholder.answered_at
                        linked.answer.state = "accepted"
                        linked.answer.disposition = 200
                        completion = linked.answer.completion
            if placeholder.precommit_refcount == 0:
                if (
                    not placeholder.durable
                    or placeholder.linked_entry is not None
                    or not is_live_placeholder
                ):
                    if self._answered_placeholders.get(placeholder.call_control_id) is placeholder:
                        self._answered_placeholders.pop(placeholder.call_control_id, None)
                    placeholder.linked_entry = None
                placeholder.linked_abort_identity = None
            if (
                linked_abort_identity is not None
                and isinstance(result, WebhookCommitResult)
                and (result.receipt, result.effect) == ("first", "applied")
            ):
                current = self._by_control.get(linked_abort_identity.call_control_id)
                if (
                    current is not None
                    and current.call_control_id == linked_abort_identity.call_control_id
                    and current.generation == linked_abort_identity.generation
                    and current.terminal_event is None
                    and hmac.compare_digest(
                        current.token_digest,
                        linked_abort_identity.token_digest,
                    )
                ):
                    abort_target = _AbortTarget(
                        entry=current,
                        generation=current.generation,
                        call_control_id=current.call_control_id,
                        token_digest=current.token_digest,
                    )
            reservation._registry_applied = True
        if completion is not None and not completion.done():
            from projetv0_voice.telnyx.webhooks import WebhookDisposition

            completion.set_result(WebhookDisposition(200))
        return abort_target

    async def reconcile_after_commit(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        result: WebhookCommitValue,
    ) -> WebhookDisposition:
        from projetv0_voice.telnyx.webhooks import WebhookDisposition

        reservation = resolution.reservation
        reservation_cancellation: asyncio.CancelledError | None = None
        if isinstance(reservation, CallReservation):
            reservation_cancellation = await reservation.confirm(result)
        elif isinstance(reservation, _PlaceholderReservation):
            await reservation.confirm_fail_closed(result)
        if isinstance(result, QualificationRunConsumed):
            return WebhookDisposition(503, admission_rejection="qualification")
        if result.effect == "existing_terminal":
            if event.event_type == "call.hangup" and event.call_control_id is not None:
                await self._finish_provider_terminal_envelope(
                    event.call_control_id,
                    reservation_cancellation,
                )
            return WebhookDisposition(200)
        if event.call_control_id is None:
            return WebhookDisposition(200)
        if event.event_type == "call.answered":
            completion: asyncio.Future[WebhookDisposition] | None = None
            start_streaming = False
            placeholder_deadline = (
                reservation._placeholder.deadline
                if isinstance(reservation, _PlaceholderReservation)
                else None
            )
            async with self._lock:
                entry = self._by_control.get(event.call_control_id)
                if entry is None or entry.terminal_event is not None:
                    return WebhookDisposition(200)
                now = float(self._monotonic())
                if (
                    placeholder_deadline is not None
                    and now >= placeholder_deadline
                    or now >= entry.token_deadline
                ):
                    return WebhookDisposition(200)
                entry.answer_evidence = True
                entry.answered_at = event.occurred_at.astimezone(UTC)
                entry.answer.state = "accepted"
                entry.answer.disposition = 200
                completion = entry.answer.completion
                start_streaming = entry.durable
            if completion is not None and not completion.done():
                completion.set_result(WebhookDisposition(200))
            if start_streaming:
                return await self._run_action(event.call_control_id, "streaming")
            return WebhookDisposition(200)
        if event.event_type == "call.hangup":
            await self._finish_provider_terminal_envelope(
                event.call_control_id,
                reservation_cancellation,
            )
            return WebhookDisposition(200)
        if event.event_type != "call.initiated":
            return WebhookDisposition(200)
        if not isinstance(reservation, CallReservation):
            # A committed transfer-target observation owns no original AI admission.
            return WebhookDisposition(200)
        async with self._lock:
            entry = self._by_control.get(event.call_control_id)
            answered_early = entry is not None and entry.answer_evidence
        if answered_early:
            return await self._run_action(event.call_control_id, "streaming")
        return await self._run_action(event.call_control_id, "answer")

    async def confirm_committed(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        result: WebhookCommitValue,
    ) -> CommittedWebhookConfirmation:
        disposition = await self.reconcile_after_commit(event, resolution, result)
        reservation = resolution.reservation
        generation = (
            CallGenerationHandle(
                reservation._entry.call_control_id, reservation._generation
            )
            if isinstance(reservation, CallReservation)
            else None
        )
        return CommittedWebhookConfirmation(disposition, generation)

    async def confirm_late_after_fail_closed(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        result: WebhookCommitValue,
    ) -> FailClosedWebhookConfirmation:
        del event
        reservation = resolution.reservation
        linked_abort_identity: _LinkedAbortIdentity | None = None
        linked_abort_target: _AbortTarget | None = None
        if isinstance(reservation, CallReservation):
            await reservation.confirm(result)
        elif isinstance(reservation, _PlaceholderReservation):
            async with self._lock:
                linked_abort_identity = reservation._placeholder.linked_abort_identity
            linked_abort_target = await reservation.confirm_fail_closed(
                result,
                linked_abort_identity=linked_abort_identity,
            )
        generation: CallGenerationHandle | None = None
        abort_scheduled = False
        if isinstance(reservation, CallReservation):
            generation = CallGenerationHandle(
                reservation._entry.call_control_id, reservation._generation
            )
            if isinstance(result, WebhookCommitResult) and (
                result.receipt,
                result.effect,
            ) == ("first", "applied"):
                abort_scheduled = self._schedule_abort_exact(
                    reservation._entry,
                    reservation._generation,
                )
        elif isinstance(reservation, _PlaceholderReservation) and linked_abort_identity is not None:
            identity = linked_abort_identity
            generation = CallGenerationHandle(identity.call_control_id, identity.generation)
            if (
                isinstance(result, WebhookCommitResult)
                and (
                    result.receipt,
                    result.effect,
                )
                == ("first", "applied")
                and linked_abort_target is not None
            ):
                abort_scheduled = self._schedule_abort_target(
                    linked_abort_target,
                    name="voice-fail-closed-lease-abort",
                )
        elif isinstance(reservation, _PlaceholderReservation):
            async with self._lock:
                placeholder = self._answered_placeholders.get(
                    reservation._placeholder.call_control_id
                )
                if placeholder is reservation._placeholder:
                    self._answered_placeholders.pop(
                        placeholder.call_control_id, None
                    )
        return FailClosedWebhookConfirmation(generation, abort_scheduled)

    async def _run_action(
        self, call_control_id: str, action: Literal["answer", "streaming"]
    ) -> WebhookDisposition:
        from projetv0_voice.telnyx.webhooks import WebhookDisposition

        if self._sparra is not None:
            try:
                await self._ensure_begin_snapshot(call_control_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                return WebhookDisposition(503, admission_rejection="invalid")

        owner = False
        completion: asyncio.Future[WebhookDisposition] | None = None
        command_id: UUID | None = None
        raw_token: str | None = None
        streaming_request: StreamingStartV1 | None = None
        entry: _CallEntry | None = None
        generation: UUID | None = None
        deadline_work: _TerminalWork | None = None
        async with self._lock:
            if self._draining:
                return WebhookDisposition(503)
            if self._qualification_expired():
                return WebhookDisposition(503)
            entry = self._by_control.get(call_control_id)
            if (
                entry is None
                or entry.terminal_event is not None
                or not entry.durable
                or (self._transfer_fenced(entry))
            ):
                return WebhookDisposition(200)
            now = float(self._monotonic())
            if now >= entry.token_deadline and not (
                entry.attached and entry.lease_state == "active"
            ):
                deadline_work = self._mark_terminal_locked(
                    entry,
                    reason="token_deadline",
                    persist_terminal=True,
                    cleanup_hangup=True,
                )
            else:
                slot = entry.answer if action == "answer" else entry.streaming
                if slot.state == "in_flight" and slot.completion is not None:
                    completion = slot.completion
                elif slot.state in {"accepted", "unknown", "rejected"}:
                    return WebhookDisposition(slot.disposition)
                else:
                    if action == "streaming" and entry.raw_token is None:
                        return WebhookDisposition(200)
                    owner = True
                    generation = entry.generation
                    command_id = slot.command_id
                    if action == "streaming":
                        raw_token = entry.raw_token
                    completion = asyncio.get_running_loop().create_future()
                    slot.state = "in_flight"
                    slot.completion = completion
                    slot.owner_task = cast(asyncio.Task[object], asyncio.current_task())
        if deadline_work is not None:
            await self._run_terminal_cleanup(deadline_work)
            return WebhookDisposition(200)
        if not owner:
            if completion is None:
                return WebhookDisposition(500)
            return await asyncio.shield(completion)

        outcome: str
        response_failure = False
        cancelled = False
        try:
            try:
                if action == "answer":
                    call_result = await self._call_control.answer(
                        call_control_id, command_id=cast(UUID, command_id)
                    )
                else:
                    streaming_request = StreamingStartV1(
                        stream_url=self._stream_url,
                        stream_auth_token=SecretStr(cast(str, raw_token)),
                    )
                    call_result = await self._call_control.start_streaming(
                        call_control_id,
                        streaming_request,
                        command_id=cast(UUID, command_id),
                    )
            finally:
                streaming_request = None
                raw_token = None
            outcome = call_result.outcome
        except asyncio.CancelledError:
            outcome = "outcome_unknown"
            cancelled = True
        except Exception:
            outcome = "outcome_unknown"
            response_failure = True

        disposition = WebhookDisposition(
            500
            if response_failure
            else 503
            if outcome in {"rate_limited", "retryable_not_sent"}
            else 200
        )
        if outcome not in {
            "accepted",
            "rate_limited",
            "retryable_not_sent",
            "rejected",
            "outcome_unknown",
        }:
            outcome = "outcome_unknown"
            disposition = WebhookDisposition(500)
        future_to_finish: asyncio.Future[WebhookDisposition] | None = None
        terminalize_rejected = False
        result_deadline_work: _TerminalWork | None = None
        async with self._lock:
            current = self._by_control.get(call_control_id)
            if current is entry and current.generation == generation:
                slot = current.answer if action == "answer" else current.streaming
                future_to_finish = slot.completion
                if slot.owner_task is asyncio.current_task():
                    slot.owner_task = None
                positive_evidence = (
                    current.answer_evidence if action == "answer" else current.streaming_evidence
                )
                now = float(self._monotonic())
                if current.terminal_event is not None:
                    disposition = WebhookDisposition(slot.disposition)
                elif positive_evidence:
                    slot.state = "accepted"
                    slot.disposition = 200
                    disposition = WebhookDisposition(200)
                    if action == "streaming":
                        current.raw_token = None
                elif now >= current.token_deadline and not (
                    current.attached and current.lease_state == "active"
                ):
                    result_deadline_work = self._mark_terminal_locked(
                        current,
                        reason="token_deadline",
                        persist_terminal=True,
                        cleanup_hangup=True,
                    )
                    disposition = WebhookDisposition(slot.disposition)
                else:
                    next_state: dict[str, ActionState] = {
                        "accepted": "accepted",
                        "rate_limited": "retryable",
                        "retryable_not_sent": "retryable",
                        "rejected": "rejected",
                        "outcome_unknown": "unknown",
                    }
                    slot.state = next_state[outcome]
                    slot.disposition = disposition.status_code
                    if action == "streaming" and slot.state in {
                        "accepted",
                        "unknown",
                        "rejected",
                    }:
                        current.raw_token = None
                    terminalize_rejected = slot.state == "rejected"
        if future_to_finish is not None and not future_to_finish.done():
            future_to_finish.set_result(disposition)
        if result_deadline_work is not None:
            if cancelled:
                cleanup = self._background_owner.start(
                    self._run_terminal_cleanup(result_deadline_work),
                    name="voice-action-deadline-cleanup",
                )
                if cleanup is None:
                    self._set_internal_failure("background_task_registration_failed")
                    while True:
                        try:
                            await self._run_terminal_cleanup(
                                result_deadline_work,
                                retry_cancelled_io=True,
                            )
                            break
                        except asyncio.CancelledError:
                            continue
                    self._set_internal_failure("background_task_registration_failed")
                else:
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            continue
                    await cleanup
            else:
                await self._run_terminal_cleanup(result_deadline_work)
        if terminalize_rejected and entry is not None and generation is not None:
            await self._terminalize_rejected(entry, generation)
        if cancelled:
            raise asyncio.CancelledError
        return disposition

    def _mark_terminal_locked(
        self,
        entry: _CallEntry,
        *,
        reason: str,
        persist_terminal: bool,
        cleanup_hangup: bool,
    ) -> _TerminalWork | None:
        from projetv0_voice.telnyx.webhooks import WebhookDisposition

        if self._transfer_fenced(entry) or entry.terminal_event is not None:
            return None
        entry.terminal_event = reason
        entry.lease_state = "terminal"
        entry.raw_token = None
        entry.cleanup_hangup_started = cleanup_hangup
        entry.drain_intent = True
        current_task = asyncio.current_task()
        action_owners: list[asyncio.Task[object]] = []
        for slot in (entry.answer, entry.streaming):
            if slot.state == "in_flight":
                slot.state = "unknown"
                slot.disposition = 200
            if slot.completion is not None and not slot.completion.done():
                slot.completion.set_result(WebhookDisposition(slot.disposition))
            if slot.owner_task is not None and slot.owner_task is not current_task:
                action_owners.append(slot.owner_task)
        lifecycle_owners = (
            (cast(_SessionLifecycleOwner, entry.lifecycle_owner),)
            if entry.lifecycle_owner is not None
            and self._valid_session_owner(entry.lifecycle_owner)
            else ()
        )
        return _TerminalWork(
            entry=entry,
            generation=entry.generation,
            action_owners=tuple(dict.fromkeys(action_owners)),
            lifecycle_owners=lifecycle_owners,
            reason=reason,
            persist_terminal=persist_terminal,
            cleanup_hangup=cleanup_hangup,
        )

    async def _run_terminal_cleanup(
        self, work: _TerminalWork, *, retry_cancelled_io: bool = False
    ) -> None:
        for owner in work.action_owners:
            owner.cancel()
        if work.action_owners:
            await asyncio.gather(*work.action_owners, return_exceptions=True)
        lifecycle_waits: list[Awaitable[None]] = []
        for lifecycle_owner in work.lifecycle_owners:
            try:
                lifecycle_owner.request_drain(work.reason)
                if lifecycle_owner._task is not asyncio.current_task():
                    lifecycle_waits.append(lifecycle_owner.wait())
            except BaseException:
                self._set_internal_failure("lifecycle_drain_failed")
        if lifecycle_waits:
            results = await asyncio.gather(*lifecycle_waits, return_exceptions=True)
            if any(isinstance(result, BaseException) for result in results):
                self._set_internal_failure("lifecycle_drain_failed")
        if work.lifecycle_owners:
            return
        entry = work.entry
        closed_at = self._require_aware(self._utcnow())
        operation: VoiceOperationV1 | None = None
        cancellation: asyncio.CancelledError | None = None
        persistence_failed = False
        native_terminal = (
            work.persist_terminal and self._sparra is not None and entry.routing is not None
        )
        if native_terminal:
            facts: LocalCallLifecycleFacts | None
            while True:
                try:
                    facts = await self._writer.read_call_lifecycle(entry.call_id)
                    break
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
                except BaseException:
                    self._set_internal_failure("terminal_persistence_failed")
                    persistence_failed = True
                    facts = None
                    break
            if not persistence_failed and (
                facts is None
                or facts.call_id != entry.call_id
                or facts.admission_generation != work.generation
                or facts.admitted_at != entry.initiated_at
                or facts.retention_until != entry.initiated_at + self._retention_delta
                or facts.telnyx_call_leg_id != entry.call_leg_id
                or facts.telnyx_call_session_id != entry.call_session_id
            ):
                self._set_internal_failure("terminal_persistence_failed")
                persistence_failed = True
            if not persistence_failed and facts is not None and closed_at < facts.retention_until:
                starts = [
                    value
                    for value in (
                        facts.started_at,
                        entry.answered_at,
                        entry.claimed_at,
                    )
                    if value is not None
                ]
                evidence = facts.disclosure_evidence
                disclosure_fields: _DisclosureFields = {}
                if evidence is not None:
                    disclosure_fields["disclosure_evidence"] = evidence
                operation = VoiceOperationV1(
                    schema_version=1,
                    operation_id=uuid5(
                        entry.call_id, f"local-terminal:{work.generation}:{work.reason}"
                    ),
                    deployment_id=self._deployment_id,
                    call_id=entry.call_id,
                    occurred_at=closed_at,
                    kind="call.upsert",
                    payload=CallUpsertPayloadV1(
                        telnyx_call_control_id=entry.call_control_id,
                        telnyx_call_leg_id=entry.call_leg_id,
                        telnyx_call_session_id=entry.call_session_id,
                        status="failed",
                        disclosure_state="completed"
                        if evidence is not None and evidence.completed_at is not None
                        else "failed",
                        started_at=min(starts) if starts else None,
                        ended_at=closed_at,
                        end_reason=work.reason,
                        retention_until=facts.retention_until,
                        **disclosure_fields,
                    ),
                )
        if work.persist_terminal:
            while True:
                if persistence_failed:
                    break
                try:
                    await self._writer.commit_lease(
                        call_control_id=entry.call_control_id,
                        call_id=entry.call_id,
                        tenant_id=self._tenant_id,
                        agent_id=self._agent_id,
                        state="terminal",
                        token_hash=entry.token_digest,
                        created_at=entry.created_at,
                        expires_at=entry.expires_at,
                        closed_at=closed_at,
                        **({"operation": operation} if operation is not None else {}),
                    )
                except asyncio.CancelledError as error:
                    if native_terminal:
                        if cancellation is None:
                            cancellation = error
                        continue
                    if retry_cancelled_io:
                        continue
                    self._set_internal_failure("terminal_persistence_failed")
                except BaseException:
                    self._set_internal_failure("terminal_persistence_failed")
                    persistence_failed = native_terminal
                break
        if work.cleanup_hangup:
            while True:
                try:
                    await self._hangup_unfenced(
                        entry.call_control_id,
                        command_id=entry.hangup_command_id,
                    )
                except asyncio.CancelledError:
                    if retry_cancelled_io:
                        continue
                except BaseException:
                    pass
                break
        if persistence_failed:
            # Keep the exact generation/capacity until the durable call and lease
            # are accounted for; readiness is already failed closed above.
            if cancellation is not None:
                raise cancellation
            return
        generation = CallGenerationHandle(entry.call_control_id, work.generation)
        while True:
            try:
                await self.complete_terminal_cleanup(generation)
                break
            except asyncio.CancelledError:
                continue
        if cancellation is not None:
            raise cancellation

    async def complete_terminal_cleanup(self, generation: CallGenerationHandle) -> bool:
        abort_target_clearers: tuple[tuple[_AbortTarget, Callable[[_AbortTarget], None]], ...] = ()
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            if (
                entry is None
                or entry.generation != generation.generation
                or entry.terminal_event is None
                or self._transfer_fenced(entry)
                and entry.terminal_event != "call.hangup"
            ):
                return False
            if not entry.capacity_released:
                entry.capacity_released = True
                self._permits_used -= 1
            entry.resources_released = True
            abort_target_clearers = tuple(entry.abort_target_clearers)
            entry.abort_target_clearers.clear()
            self._by_control.pop(entry.call_control_id, None)
            self._by_call_id.pop(entry.call_id, None)
        for abort_target, clearer in abort_target_clearers:
            with contextlib.suppress(BaseException):
                clearer(abort_target)
        return True

    async def _terminalize_after_hangup(self, call_control_id: str) -> None:
        async with self._lock:
            entry = self._by_control.get(call_control_id)
            if entry is None:
                return
            work = self._mark_terminal_locked(
                entry,
                reason="call.hangup",
                persist_terminal=False,
                cleanup_hangup=False,
            )
        if work is not None:
            await self._run_terminal_cleanup(work)

    async def _terminalize_rejected(
        self, entry: _CallEntry, generation: UUID
    ) -> None:
        async with self._lock:
            current = self._by_control.get(entry.call_control_id)
            if (
                current is not entry
                or current.generation != generation
                or current.terminal_event is not None
            ):
                return
            work = self._mark_terminal_locked(
                current,
                reason="provider_rejected",
                persist_terminal=True,
                cleanup_hangup=True,
            )
        if work is not None:
            await self._run_terminal_cleanup(work)

    async def live_call_count(self) -> int:
        async with self._lock:
            return self._permits_used

    async def generation_handle(
        self, call_control_id: str
    ) -> CallGenerationHandle | None:
        async with self._lock:
            entry = self._by_control.get(call_control_id)
            if entry is None:
                return None
            return CallGenerationHandle(entry.call_control_id, entry.generation)

    async def snapshot(self, call_control_id: str) -> CallSnapshot | None:
        async with self._lock:
            entry = self._by_control.get(call_control_id)
            if entry is None:
                return None
            return CallSnapshot(
                generation=entry.generation,
                call_id=entry.call_id,
                durable=entry.durable,
                precommit_refcount=entry.precommit_refcount,
                answer_state=entry.answer.state,
                streaming_state=entry.streaming.state,
                lease_state=entry.lease_state,
                raw_token_retained=entry.raw_token is not None,
                token_digest=entry.token_digest,
            )

    async def placeholder_count(self) -> int:
        async with self._lock:
            return len(self._answered_placeholders)

    async def placeholder_deadline(self, call_control_id: str) -> float | None:
        async with self._lock:
            placeholder = self._answered_placeholders.get(call_control_id)
            return None if placeholder is None else placeholder.deadline

    async def claim_once(
        self,
        *,
        call_control_id: str,
        token_digest: bytes,
        abort_target_publisher: Callable[[_AbortTarget], bool],
        abort_target_clearer: Callable[[_AbortTarget], None],
    ) -> ProcessLeaseClaim | None:
        if (
            not isinstance(call_control_id, str)
            or not call_control_id
            or type(token_digest) is not bytes
            or len(token_digest) != 32
        ):
            return None
        entry: _CallEntry | None = None
        generation: UUID | None = None
        async with self._lock:
            entry = self._by_control.get(call_control_id)
            now = float(self._monotonic())
            expected_digest = b"\0" * 32 if entry is None else entry.token_digest
            digest_matches = hmac.compare_digest(expected_digest, token_digest)
            if (
                entry is None
                or self._draining
                or self._qualification_expired()
                or entry.terminal_event is not None
                or not entry.durable
                or entry.lease_state != "pending"
                or entry.streaming.state not in {"in_flight", "accepted", "unknown"}
                or now >= entry.token_deadline
                or not digest_matches
            ):
                return None
            entry.lease_state = "claiming"
            claimed_at = self._require_aware(self._utcnow())
            entry.claimed_at = claimed_at
            generation = entry.generation
            abort_target = _AbortTarget(
                entry=entry,
                generation=entry.generation,
                call_control_id=entry.call_control_id,
                token_digest=entry.token_digest,
            )
            if not abort_target_publisher(abort_target):
                entry.lease_state = "pending"
                return None
            entry.abort_target_clearers.append(
                (abort_target, abort_target_clearer)
            )
        await self._writer.commit_lease(
            call_control_id=entry.call_control_id,
            call_id=entry.call_id,
            tenant_id=self._tenant_id,
            agent_id=self._agent_id,
            state="active",
            token_hash=entry.token_digest,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
            closed_at=None,
        )
        completion: asyncio.Future[WebhookDisposition] | None = None
        async with self._lock:
            current = self._by_control.get(call_control_id)
            if (
                current is not entry
                or current.generation != generation
                or current.lease_state != "claiming"
                or current.terminal_event is not None
            ):
                return None
            claim = ProcessLeaseClaim(
                call_control_id=current.call_control_id,
                call_id=current.call_id,
                generation=current.generation,
                token_digest=current.token_digest,
                claimed_at=claimed_at,
            )
            current.claim = claim
            current.lease_state = "active"
            current.streaming_evidence = True
            current.streaming.state = "accepted"
            current.streaming.disposition = 200
            completion = current.streaming.completion
            current.raw_token = None
        if completion is not None and not completion.done():
            from projetv0_voice.telnyx.webhooks import WebhookDisposition

            completion.set_result(WebhookDisposition(200))
        return claim

    async def begin_drain(self) -> None:
        """Close new admissions, action ownership, and ordinary WSS claims."""

        async with self._lock:
            self._draining = True

    async def qualification_state_valid(self) -> bool:
        """Return the owner-locked candidate/override readiness predicate."""

        return await self.qualification_state() == "valid"

    async def qualification_state(
        self,
    ) -> Literal["valid", "consumed", "expired"]:
        """Distinguish candidate consumption from expiry for WSS eligibility."""

        async with self._lock:
            if self._candidate_consumed:
                return "consumed"
            if self._qualification_expired():
                return "expired"
            return "valid"

    async def close_session_owner_registration(self) -> None:
        cancellation: asyncio.CancelledError | None = None
        close_task: asyncio.Task[None]
        while True:
            try:
                async with self._lock:
                    self._session_owner_registration_open = False
                    retained = self._session_close_task
                    if retained is None:
                        coroutine = self._close_session_owner_registration_core()
                        try:
                            retained = asyncio.create_task(
                                coroutine,
                                name="voice-session-owner-close",
                            )
                        except BaseException:
                            coroutine.close()
                            raise
                        self._session_close_task = retained
                    close_task = retained
                break
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException:
                break
        close_error: BaseException | None = None
        try:
            close_task.result()
        except BaseException as error:
            close_error = error
        if cancellation is not None:
            raise cancellation
        if close_error is not None:
            raise close_error

    async def _close_session_owner_registration_core(self) -> None:
        authorities: list[TerminalAuthority] = []
        provider_authorities: list[tuple[TerminalAuthority, _CallEntry, UUID, asyncio.Event]] = []
        owners: list[tuple[_SessionLifecycleOwner, str]] = []
        cancellation: asyncio.CancelledError | None = None
        failure: BaseException | None = None
        async with self._lock:
            self._session_owner_registration_open = False
            pending_events = tuple(
                dict.fromkeys(
                    entry.terminal_settled_event
                    for entry in self._by_control.values()
                    if entry.terminal_authority is None
                    and entry.terminal_state == "external_pending"
                    and entry.terminal_pending_count > 0
                )
            )
        for pending_event in pending_events:
            while not pending_event.is_set():
                try:
                    await asyncio.shield(pending_event.wait())
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
                    continue
        while True:
            try:
                async with self._lock:
                    self._session_owner_registration_open = False
                    completion_owner = asyncio.current_task()
                    if completion_owner is None:
                        raise RuntimeError("terminal_owner_unavailable") from None
                    for entry in tuple(self._by_control.values()):
                        if self._transfer_fenced(entry):
                            continue
                        authority = entry.terminal_authority
                        if (
                            authority is not None
                            and entry.terminal_event == "call.hangup"
                            and entry.terminal_state in {"reserved", "completing"}
                            and authority._entry is entry
                            and authority._generation == entry.generation
                            and not authority.persist_call
                            and not authority.persist_lease
                            and not authority.cleanup_hangup
                        ):
                            provider_authorities.append(
                                (
                                    authority,
                                    entry,
                                    entry.generation,
                                    entry.terminal_completion_event,
                                )
                            )
                            if entry.lifecycle_owner is not None and self._valid_session_owner(
                                entry.lifecycle_owner
                            ):
                                entry.drain_intent = True
                                owners.append(
                                    (
                                        cast(
                                            _SessionLifecycleOwner,
                                            entry.lifecycle_owner,
                                        ),
                                        "telnyx_hangup",
                                    )
                                )
                            continue
                        if (
                            entry.terminal_event is not None
                            or authority is not None
                            or entry.session_phase in {"terminal", "removed"}
                        ):
                            continue
                        if entry.session_phase == "active_unconsumed":
                            authority = TerminalAuthority(
                                status="closed",
                                reason="process_draining",
                                metric_class="drained",
                                cleanup_hangup=True,
                                persist_call=True,
                                persist_lease=True,
                                completion_token=self._uuid_factory(),
                                _entry=entry,
                                _generation=entry.generation,
                                _completion_owner=cast(asyncio.Task[object], completion_owner),
                                _closed_at=self._require_aware(self._utcnow()),
                            )
                            entry.terminal_authority = authority
                            entry.terminal_completion_owner = cast(
                                asyncio.Task[object], completion_owner
                            )
                            entry.terminal_state = "reserved"
                            entry.terminal_event = "process_draining"
                            entry.lease_state = "terminal"
                            entry.session_phase = "terminal"
                            entry.raw_token = None
                            entry.drain_intent = True
                            entry.terminal_settled_event.set()
                            authorities.append(authority)
                            continue
                        if entry.lifecycle_owner is not None and self._valid_session_owner(
                            entry.lifecycle_owner
                        ):
                            entry.drain_intent = True
                            owners.append(
                                (
                                    cast(
                                        _SessionLifecycleOwner,
                                        entry.lifecycle_owner,
                                    ),
                                    "process_draining",
                                )
                            )
                break
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        unique_owners = tuple(dict.fromkeys(owners))
        for owner, cause in unique_owners:
            owner.request_drain(cause)
        for owner, _cause in unique_owners:
            try:
                await self._wait_lifecycle_owner(owner)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException as error:
                self._set_internal_failure("lifecycle_drain_failed")
                if failure is None:
                    failure = error
        for authority, entry, generation, completion_event in provider_authorities:
            if authority._entry is not entry or authority._generation != generation:
                if failure is None:
                    failure = RuntimeError("terminal_authority_unavailable")
                continue
            completed = False
            try:
                completed = await self.complete_reserved_terminal(authority)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException as error:
                if failure is None:
                    failure = error
                continue
            if not completed:
                while not completion_event.is_set():
                    try:
                        await asyncio.shield(completion_event.wait())
                    except asyncio.CancelledError as error:
                        if cancellation is None:
                            cancellation = error
        for authority in authorities:
            try:
                persistence_cancellation = await self._persist_authority_call(authority)
                if cancellation is None and persistence_cancellation is not None:
                    cancellation = persistence_cancellation
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                continue
            except BaseException as error:
                if failure is None:
                    failure = error
                continue
            try:
                await self.complete_reserved_terminal(authority)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException as error:
                if failure is None:
                    failure = error
        if cancellation is not None:
            raise cancellation
        if failure is not None:
            raise failure

    async def _persist_authority_call(
        self,
        authority: TerminalAuthority,
    ) -> asyncio.CancelledError | None:
        entry = authority._entry
        if authority._closed_at >= entry.initiated_at + self._retention_delta:
            return None
        if self._operation_contract_version == 2 and not isinstance(
            entry.begin_snapshot, BeginCallSnapshotV2
        ):
            return None
        started_at = entry.claimed_at or entry.answered_at
        read_facts = getattr(self._writer, "read_call_lifecycle", None)
        facts = await read_facts(entry.call_id) if callable(read_facts) else None
        if facts is not None:
            started_at = facts.started_at or started_at
        payload = CallUpsertPayloadV1(
                telnyx_call_control_id=entry.call_control_id,
                telnyx_call_leg_id=entry.call_leg_id,
                telnyx_call_session_id=entry.call_session_id,
                status="failed"
                if authority.status == "closed" and started_at is None
                else authority.status,
                disclosure_state="completed"
                if facts is not None
                and facts.disclosure_evidence is not None
                and facts.disclosure_evidence.completed_at is not None
                else "failed",
                started_at=started_at,
                ended_at=None if authority.status == "closing" else authority._closed_at,
                end_reason=authority.reason,
                retention_until=entry.initiated_at + self._retention_delta,
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
        )
        operation: VoiceOperationV1 | VoiceOperationV2
        if self._operation_contract_version == 2:
            operation = VoiceOperationV2(schema_version=2, operation_id=authority.completion_token,
                deployment_id=self._deployment_id, call_id=entry.call_id,
                occurred_at=authority._closed_at, kind="call.upsert", payload=payload)
        else:
            operation = VoiceOperationV1(schema_version=1, operation_id=authority.completion_token,
                deployment_id=self._deployment_id, call_id=entry.call_id,
                occurred_at=authority._closed_at, kind="call.upsert", payload=payload)
        commit_control = getattr(self._writer, "commit_control", None)
        if not callable(commit_control):
            self._note_terminal_failure("terminal_persistence_failed")
            raise RuntimeError("terminal_persistence_failed") from None
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                if isinstance(operation, VoiceOperationV2):
                    if not isinstance(self._writer, PersistenceWriter):
                        raise RuntimeError("terminal_persistence_failed")
                    await self._writer.freeze_call_publication_v2(operation, None,
                        generation=authority._generation, provider_callback=None)
                else:
                    await commit_control(
                        PersistenceCommand("outbox", {"operation": operation}, None)
                    )
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                continue
            except BaseException:
                self._note_terminal_failure("terminal_persistence_failed")
                if cancellation is not None:
                    raise cancellation from None
                raise RuntimeError("terminal_persistence_failed") from None
            return cancellation

    @staticmethod
    def _valid_session_owner(owner: object) -> bool:
        return callable(getattr(owner, "request_drain", None)) and callable(
            getattr(owner, "wait", None)
        )

    async def consume_claim_for_construction(
        self,
        claim: ProcessLeaseClaim,
        stream_id: str,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> CallConstructionGrant | None:
        if (
            not isinstance(claim, ProcessLeaseClaim)
            or type(stream_id) is not str
            or not stream_id
            or not self._valid_session_owner(owner)
            or not isinstance(owner_task, asyncio.Task)
        ):
            return None
        async with self._lock:
            entry = self._by_control.get(claim.call_control_id)
            now = float(self._monotonic())
            if (
                not self._session_owner_registration_open
                or entry is None
                or entry.generation != claim.generation
                or entry.claim is not claim
                or entry.claimed_at != claim.claimed_at
                or entry.lease_state != "active"
                or entry.terminal_event is not None
                or entry.drain_intent
                or entry.attached
                or entry.session_phase != "active_unconsumed"
                or entry.lifecycle_owner is not None
                or entry.lifecycle_owner_task is not None
                or now >= entry.token_deadline
                or owner_task.done()
                or owner_task.cancelling() != 0
                or self._sparra is not None
                and (entry.routing is None or entry.begin_snapshot is None)
            ):
                return None
            generation = CallGenerationHandle(entry.call_control_id, entry.generation)
            grant = CallConstructionGrant(
                call_id=entry.call_id,
                generation=generation,
                lease_claim=claim,
                deployment_id=self._deployment_id,
                telnyx_call_control_id=entry.call_control_id,
                telnyx_call_leg_id=entry.call_leg_id,
                telnyx_call_session_id=entry.call_session_id,
                stream_id=stream_id,
                started_at=claim.claimed_at,
                retention_until=entry.initiated_at + timedelta(days=self._retention_days),
                routing=entry.routing,
                begin_snapshot=entry.begin_snapshot,
            )
            entry.construction_grant = grant
            entry.lifecycle_owner = owner
            entry.lifecycle_owner_task = owner_task
            capability = _TerminalCapability(
                grant=grant,
                owner=owner,
                owner_task=owner_task,
            )
            entry.terminal_capability = capability
            cast(_SessionLifecycleOwner, owner)._terminal_capability = capability
            entry.session_phase = "constructing"
            return grant

    async def construction_is_live(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> bool:
        if (
            not isinstance(grant, CallConstructionGrant)
            or not self._valid_session_owner(owner)
            or not isinstance(owner_task, asyncio.Task)
        ):
            return False
        async with self._lock:
            entry = self._by_control.get(grant.telnyx_call_control_id)
            now = float(self._monotonic())
            return (
                self._session_owner_registration_open
                and entry is not None
                and entry.generation == grant.generation.generation
                and entry.claim is grant.lease_claim
                and entry.construction_grant is grant
                and entry.lifecycle_owner is owner
                and entry.lifecycle_owner_task is owner_task
                and entry.session_phase == "constructing"
                and entry.terminal_event is None
                and not entry.drain_intent
                and not entry.attached
                and now < entry.token_deadline
                and asyncio.current_task() is owner_task
                and not owner_task.done()
                and owner_task.cancelling() == 0
            )

    async def activate_session(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
        session: object,
    ) -> bool:
        if (
            not isinstance(grant, CallConstructionGrant)
            or not self._valid_session_owner(owner)
            or not isinstance(owner_task, asyncio.Task)
        ):
            return False
        async with self._lock:
            entry = self._by_control.get(grant.telnyx_call_control_id)
            now = float(self._monotonic())
            if (
                not self._session_owner_registration_open
                or entry is None
                or entry.generation != grant.generation.generation
                or entry.claim is not grant.lease_claim
                or entry.construction_grant is not grant
                or entry.lifecycle_owner is not owner
                or entry.lifecycle_owner_task is not owner_task
                or entry.session_phase != "constructing"
                or entry.terminal_event is not None
                or entry.drain_intent
                or entry.attached
                or now >= entry.token_deadline
                or asyncio.current_task() is not owner_task
                or owner_task.done()
                or owner_task.cancelling() != 0
                or cast(_SessionLifecycleOwner, owner)._phase != "preactivated"
                or cast(_SessionLifecycleOwner, owner)._session is not session
            ):
                return False
            entry.session_phase = "preactivated"
            entry.session = session
            cast(_SessionLifecycleOwner, owner)._phase = "running"
            entry.session_phase = "running"
            entry.attached = True
            return True

    async def reserve_or_read_terminal(
        self,
        grant: CallConstructionGrant,
        capability: _TerminalCapability,
        proposed: TerminalProposal,
    ) -> TerminalAuthority:
        if (
            not isinstance(grant, CallConstructionGrant)
            or not isinstance(capability, _TerminalCapability)
            or not isinstance(proposed, TerminalProposal)
        ):
            raise RuntimeError("terminal_authority_unavailable") from None
        while True:
            pending_event: asyncio.Event | None = None
            async with self._lock:
                entry = self._by_control.get(grant.telnyx_call_control_id)
                if (
                    entry is None
                    or entry.generation != grant.generation.generation
                    or entry.claim is not grant.lease_claim
                    or entry.construction_grant is not grant
                    or entry.terminal_capability is not capability
                    or capability._grant is not grant
                    or capability._owner is not entry.lifecycle_owner
                    or capability._owner_task is not entry.lifecycle_owner_task
                    or not capability.permits_current_task()
                    or entry.session_phase
                    not in {"constructing", "preactivated", "running", "terminal"}
                ):
                    raise RuntimeError("terminal_authority_unavailable") from None
                if entry.terminal_authority is not None:
                    return entry.terminal_authority
                if entry.terminal_state == "external_pending" and entry.terminal_pending_count > 0:
                    pending_event = entry.terminal_settled_event
                else:
                    fenced = self._transfer_fenced(entry)
                    completion_owner = asyncio.current_task()
                    if completion_owner is None:
                        raise RuntimeError("terminal_owner_unavailable") from None
                    authority = TerminalAuthority(
                        status="closing" if fenced else proposed.status,
                        reason=(
                            "content_erased"
                            if entry.transfer_facts is not None
                            and entry.transfer_facts.content_erased
                            else "qualified_line_connected"
                            if entry.transfer_facts is not None
                            and entry.transfer_facts.qualified_line_bridged_at is not None
                            else "transfer_outcome_unknown"
                            if entry.transfer_facts is not None
                            and entry.transfer_facts.transfer_command_id is not None
                            else "local_ai_departure"
                        )
                        if fenced
                        else proposed.reason,
                        metric_class=proposed.metric_class,
                        cleanup_hangup=False if fenced else proposed.cleanup_hangup,
                        persist_call=True,
                        persist_lease=not fenced,
                        completion_token=self._uuid_factory(),
                        _entry=entry,
                        _generation=entry.generation,
                        _completion_owner=cast(asyncio.Task[object], completion_owner),
                        _closed_at=self._require_aware(self._utcnow()),
                    )
                    entry.terminal_authority = authority
                    entry.terminal_completion_owner = cast(asyncio.Task[object], completion_owner)
                    entry.terminal_state = "reserved"
                    entry.terminal_event = proposed.reason
                    entry.lease_state = "active" if fenced else "terminal"
                    entry.session_phase = "terminal"
                    entry.raw_token = None
                    entry.drain_intent = True
                    entry.terminal_settled_event.set()
                    return authority
            if pending_event is not None:
                while not pending_event.is_set():
                    try:
                        await asyncio.shield(pending_event.wait())
                    except asyncio.CancelledError:
                        continue

    async def complete_reserved_terminal(self, authority: object) -> bool:
        if not isinstance(authority, TerminalAuthority):
            return False
        if authority.status == "closing":
            async with self._lock:
                closing_entry = self._by_control.get(authority._entry.call_control_id)
                if (
                    closing_entry is not authority._entry
                    or closing_entry.terminal_authority is not authority
                ):
                    return False
                closing_entry.resources_released = True
                self._clear_content_holders_locked(closing_entry)
                closing_entry.terminal_completion_event.set()
            return True
        cancellation: asyncio.CancelledError | None = None
        entry: _CallEntry
        while True:
            try:
                async with self._lock:
                    entry = self._by_control.get(  # type: ignore[assignment]
                        authority._entry.call_control_id
                    )
                    if (
                        entry is not authority._entry
                        or entry.generation != authority._generation
                        or entry.terminal_authority is not authority
                        or entry.terminal_state not in {"reserved", "completing"}
                    ):
                        return False
                    current_task = asyncio.current_task()
                    if current_task is None:
                        return False
                    completion_owner = entry.terminal_completion_owner
                    if current_task is not completion_owner:
                        provider_takeover = (
                            not authority.persist_call
                            and not authority.persist_lease
                            and not authority.cleanup_hangup
                            and (
                                completion_owner is None
                                or completion_owner.done()
                                or completion_owner.cancelling() != 0
                            )
                        )
                        if not provider_takeover:
                            return False
                        entry.terminal_completion_owner = cast(asyncio.Task[object], current_task)
                    entry.terminal_state = "completing"
                break
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if authority.persist_lease:
            while True:
                try:
                    await self._writer.commit_lease(
                        call_control_id=entry.call_control_id,
                        call_id=entry.call_id,
                        tenant_id=self._tenant_id,
                        agent_id=self._agent_id,
                        state="terminal",
                        token_hash=entry.token_digest,
                        created_at=entry.created_at,
                        expires_at=entry.expires_at,
                        closed_at=authority._closed_at,
                    )
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
                    continue
                except BaseException:
                    self._note_terminal_failure("terminal_persistence_failed")
                    raise RuntimeError("terminal_persistence_failed") from None
                break
        if authority.cleanup_hangup:
            while True:
                try:
                    result = await self._hangup_unfenced(
                        entry.call_control_id,
                        command_id=entry.hangup_command_id,
                    )
                    if not isinstance(result, CallControlResult) or result.outcome != "accepted":
                        self._note_terminal_failure("terminal_hangup_failed")
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
                    continue
                except BaseException:
                    self._set_internal_failure("terminal_hangup_failed")
                break
        abort_target_clearers: tuple[tuple[_AbortTarget, Callable[[_AbortTarget], None]], ...] = ()
        while True:
            try:
                async with self._lock:
                    current = self._by_control.get(entry.call_control_id)
                    if (
                        current is not entry
                        or current.generation != authority._generation
                        or current.terminal_authority is not authority
                    ):
                        return False
                    if not current.capacity_released:
                        current.capacity_released = True
                        self._permits_used -= 1
                    current.resources_released = True
                    current.session_phase = "removed"
                    current.terminal_state = "removed"
                    current.terminal_settled_event.set()
                    current.terminal_completion_event.set()
                    abort_target_clearers = tuple(current.abort_target_clearers)
                    current.abort_target_clearers.clear()
                    self._by_control.pop(current.call_control_id, None)
                    self._by_call_id.pop(current.call_id, None)
                break
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        for abort_target, clearer in abort_target_clearers:
            with contextlib.suppress(BaseException):
                clearer(abort_target)
        if cancellation is not None:
            raise cancellation
        return True

    async def prepare_required_recording_drain(
        self,
        call_id: UUID,
    ) -> TerminalAuthority | None:
        if type(call_id) is not UUID:
            return None
        owner: object | None = None
        session: object | None = None
        while True:
            pending_event: asyncio.Event | None = None
            async with self._lock:
                entry = self._by_call_id.get(call_id)
                if (
                    entry is None
                    or entry.terminal_authority is not None
                    or entry.terminal_event is not None
                    or entry.session_phase in {"terminal", "removed"}
                ):
                    return None
                if (
                    entry.session_phase == "active_unconsumed"
                    and entry.terminal_state == "external_pending"
                    and entry.terminal_pending_count > 0
                ):
                    pending_event = entry.terminal_settled_event
                elif entry.session_phase == "active_unconsumed":
                    completion_owner = asyncio.current_task()
                    if completion_owner is None:
                        raise RuntimeError("terminal_owner_unavailable") from None
                    authority = TerminalAuthority(
                        status="failed",
                        reason="recording_required_error",
                        metric_class="failed",
                        cleanup_hangup=False,
                        persist_call=True,
                        persist_lease=True,
                        completion_token=self._uuid_factory(),
                        _entry=entry,
                        _generation=entry.generation,
                        _completion_owner=cast(
                            asyncio.Task[object], completion_owner
                        ),
                        _closed_at=self._require_aware(self._utcnow()),
                    )
                    entry.terminal_authority = authority
                    entry.terminal_completion_owner = cast(
                        asyncio.Task[object], completion_owner
                    )
                    entry.terminal_state = "reserved"
                    entry.terminal_event = "recording_required_error"
                    entry.lease_state = "terminal"
                    entry.session_phase = "terminal"
                    entry.raw_token = None
                    entry.drain_intent = True
                    entry.terminal_settled_event.set()
                    return authority
                elif entry.session_phase in {"constructing", "preactivated"}:
                    owner = entry.lifecycle_owner
                elif entry.session_phase == "running":
                    session = entry.session
            if pending_event is None:
                break
            while not pending_event.is_set():
                try:
                    await asyncio.shield(pending_event.wait())
                except asyncio.CancelledError:
                    continue
        if owner is not None and self._valid_session_owner(owner):
            lifecycle_owner = cast(_SessionLifecycleOwner, owner)
            lifecycle_owner.request_drain("recording_required_error")
            if asyncio.current_task() is lifecycle_owner._task:
                return None
            await self._wait_lifecycle_owner(lifecycle_owner)
            return None
        if session is not None:
            request_drain = getattr(session, "request_drain", None)
            if callable(request_drain):
                await request_drain("recording_required_error")
        return None

    @staticmethod
    async def _wait_lifecycle_owner(owner: _SessionLifecycleOwner) -> None:
        wait_task = asyncio.create_task(owner.wait())
        cancellation: asyncio.CancelledError | None = None
        while not wait_task.done():
            try:
                await asyncio.shield(wait_task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException:
                break
        try:
            await wait_task
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        if cancellation is not None:
            raise cancellation

    async def _finish_provider_terminal_envelope(
        self,
        call_control_id: str,
        cancellation: asyncio.CancelledError | None,
    ) -> None:
        try:
            await self._finish_provider_terminal(call_control_id)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        if cancellation is not None:
            raise cancellation

    async def _finish_provider_terminal(self, call_control_id: str) -> None:
        authority: TerminalAuthority | None = None
        owner: object | None = None
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                async with self._lock:
                    entry = self._by_control.get(call_control_id)
                    if entry is not None:
                        authority = entry.terminal_authority
                        owner = entry.lifecycle_owner
                break
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if authority is None:
            if cancellation is not None:
                raise cancellation
            return
        if owner is not None and self._valid_session_owner(owner):
            lifecycle_owner = cast(_SessionLifecycleOwner, owner)
            lifecycle_owner.request_drain("telnyx_hangup")
            if asyncio.current_task() is not lifecycle_owner._task:
                try:
                    await self._wait_lifecycle_owner(lifecycle_owner)
                except asyncio.CancelledError as error:
                    cancellation = error
        completed = await self.complete_reserved_terminal(authority)
        if not completed:
            completion_event = authority._entry.terminal_completion_event
            while not completion_event.is_set():
                try:
                    await asyncio.shield(completion_event.wait())
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
        if cancellation is not None:
            raise cancellation

    def _schedule_abort_exact(self, entry: _CallEntry, generation: UUID) -> bool:
        return self._schedule_abort_target(
            _AbortTarget(
                entry=entry,
                generation=generation,
                call_control_id=entry.call_control_id,
                token_digest=entry.token_digest,
            ),
            name="voice-fail-closed-lease-abort",
        )

    def _schedule_abort_target(self, target: _AbortTarget, *, name: str) -> bool:
        coroutine = self._abort_target(target)
        scheduled = self._background_owner.spawn(coroutine, name=name)
        if not scheduled:
            self._set_internal_failure("background_task_registration_failed")
        return scheduled

    async def _abort_target(self, target: _AbortTarget) -> None:
        async with self._lock:
            entry = self._by_control.get(target.call_control_id)
            if (
                entry is not target.entry
                or entry.generation != target.generation
                or entry.terminal_event is not None
                or not hmac.compare_digest(entry.token_digest, target.token_digest)
            ):
                return
            work = self._mark_terminal_locked(
                entry,
                reason="lease_abort",
                persist_terminal=True,
                cleanup_hangup=True,
            )
        if work is not None:
            await self._run_terminal_cleanup(work)

    async def mark_attached(self, claim: ProcessLeaseClaim) -> bool:
        if not isinstance(claim, ProcessLeaseClaim):
            return False
        async with self._lock:
            entry = self._by_control.get(claim.call_control_id)
            if (
                entry is None
                or entry.generation != claim.generation
                or entry.claim is not claim
                or entry.lease_state != "active"
            ):
                return False
            entry.attached = True
            return True

    async def reap_expired(self) -> int:
        expired: list[_TerminalWork] = []
        expired_placeholders = 0
        async with self._lock:
            now = float(self._monotonic())
            qualification_expired = self._qualification_expired()
            for call_control_id, placeholder in tuple(
                self._answered_placeholders.items()
            ):
                if now >= placeholder.deadline:
                    if self._answered_placeholders.get(call_control_id) is placeholder:
                        self._answered_placeholders.pop(call_control_id, None)
                    placeholder.linked_entry = None
                    expired_placeholders += 1
            for entry in tuple(self._by_control.values()):
                if not qualification_expired and entry.attached and entry.lease_state == "active":
                    continue
                if (
                    not qualification_expired
                    and now < entry.token_deadline
                    or entry.terminal_event is not None
                ):
                    continue
                work = self._mark_terminal_locked(
                    entry,
                    reason="token_deadline",
                    persist_terminal=True,
                    cleanup_hangup=True,
                )
                if work is not None:
                    expired.append(work)
        for work in expired:
            await self._run_terminal_cleanup(work)
        return len(expired) + expired_placeholders


class ProcessLeaseAuthority:
    """Concrete Task 6 authority backed by one process-local CallRegistry."""

    __slots__ = (
        "_abort_handoff_capacity",
        "_abort_handoffs",
        "_abort_handoffs_lock",
        "_internal_failure_code",
        "_registry",
    )

    def __init__(self, registry: CallRegistry) -> None:
        if not isinstance(registry, CallRegistry):
            raise ValueError("lease_authority_config_invalid")
        self._registry = registry
        self._abort_handoff_capacity = registry._capacity
        self._abort_handoffs: dict[tuple[str, bytes], _AbortHandoff] = {}
        self._abort_handoffs_lock = threading.Lock()
        self._internal_failure_code: str | None = None

    def __repr__(self) -> str:
        return "ProcessLeaseAuthority()"

    @property
    def internal_failure_code(self) -> str | None:
        return self._internal_failure_code

    def _publish_abort_target(self, target: _AbortTarget) -> bool:
        key = (target.call_control_id, target.token_digest)
        with self._abort_handoffs_lock:
            handoff = self._abort_handoffs.get(key)
            if handoff is not None:
                if handoff.target is target:
                    return True
                self._internal_failure_code = "abort_target_unavailable"
                return False
            if len(self._abort_handoffs) >= self._abort_handoff_capacity:
                self._internal_failure_code = "abort_target_unavailable"
                return False
            self._abort_handoffs[key] = _AbortHandoff(target)
        return True

    def _clear_abort_target(self, target: _AbortTarget) -> None:
        key = (target.call_control_id, target.token_digest)
        with self._abort_handoffs_lock:
            handoff = self._abort_handoffs.get(key)
            if handoff is not None and handoff.target is target:
                self._abort_handoffs.pop(key, None)

    def _capture_abort_target_once(
        self,
        *,
        call_control_id: str,
        token_digest: bytes,
    ) -> _AbortTarget | None:
        if (
            not isinstance(call_control_id, str)
            or not call_control_id
            or type(token_digest) is not bytes
            or len(token_digest) != 32
        ):
            self._internal_failure_code = "abort_target_unavailable"
            return None
        key = (call_control_id, token_digest)
        with self._abort_handoffs_lock:
            handoff = self._abort_handoffs.get(key)
            if handoff is None:
                self._internal_failure_code = "abort_target_unavailable"
                return None
            if handoff.scheduled:
                return None
            handoff.scheduled = True
            return handoff.target

    async def claim_once(
        self, *, call_control_id: str, token_digest: bytes
    ) -> ProcessLeaseClaim | None:
        return await self._registry.claim_once(
            call_control_id=call_control_id,
            token_digest=token_digest,
            abort_target_publisher=self._publish_abort_target,
            abort_target_clearer=self._clear_abort_target,
        )

    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        try:
            target = self._capture_abort_target_once(
                call_control_id=call_control_id,
                token_digest=token_digest,
            )
            if target is None:
                return
            if not self._registry._schedule_abort_target(
                target,
                name="voice-matching-lease-abort",
            ):
                self._internal_failure_code = "abort_target_unavailable"
        except BaseException:
            self._internal_failure_code = "abort_target_unavailable"
