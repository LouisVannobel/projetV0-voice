"""Process-local admission and ownership contracts for the production Voice Cell."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import re
import secrets
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, Protocol, cast
from uuid import UUID, uuid4, uuid5

from pydantic import SecretStr

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.writer import (
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
)

if TYPE_CHECKING:
    from projetv0_voice.telnyx.webhooks import (
        ResolvedWebhook,
        VerifiedWebhook,
        WebhookDisposition,
        WebhookDurableEffect,
    )


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
    ) -> WebhookFinalizationHandle: ...


class CallAdmissionRejected(RuntimeError):
    """A constant-safe local policy rejection before receipt persistence."""

    _POLICY_CODES = frozenset(
        {
            "call_capacity_reached",
            "placeholder_capacity_reached",
            "qualification_run_consumed",
            "qualification_window_expired",
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
    ) -> None: ...


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


class RegistryLifecycleOwner(Protocol):
    def request_drain(self) -> None: ...

    async def wait(self) -> None: ...


ActionState = Literal["idle", "in_flight", "retryable", "accepted", "rejected", "unknown"]
LeaseState = Literal["provisional", "pending", "claiming", "active", "terminal"]
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
    lifecycle_owners: tuple[RegistryLifecycleOwner, ...] = field(repr=False)
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
    terminal_event: str | None = None
    capacity_released: bool = False
    cleanup_hangup_started: bool = False
    attached: bool = False
    construction_owner: object | None = field(default=None, repr=False)
    drain_intent: bool = False
    session: object | None = field(default=None, repr=False)
    session_owner: object | None = field(default=None, repr=False)
    resources_released: bool = False
    abort_target_clearers: list[
        tuple[_AbortTarget, Callable[[_AbortTarget], None]]
    ] = field(default_factory=list, repr=False)


@dataclass(slots=True, repr=False)
class _AnsweredPlaceholder:
    call_control_id: str
    call_leg_id: str | None
    call_session_id: str | None
    first_seen: float
    deadline: float
    precommit_refcount: int = 1
    durable: bool = False
    linked_entry: _CallEntry | None = field(default=None, repr=False)


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
        "_abandon_requested",
        "_entry",
        "_event_type",
        "_generation",
        "_registry",
        "_registry_applied",
        "_settled",
    )

    def __init__(self, registry: CallRegistry, entry: _CallEntry, event_type: str) -> None:
        self._registry = registry
        self._entry = entry
        self._event_type = event_type
        self._generation = entry.generation
        self._settled = False
        self._abandoned = False
        self._abandon_requested = False
        self._abandon_event = asyncio.Event()
        self._registry_applied = False

    def __repr__(self) -> str:
        return "CallReservation()"

    def abandon_before_submit(self) -> None:
        if self._settled or self._abandon_requested:
            return
        self._abandon_requested = True
        self._abandon_event.set()

    async def confirm(self, result: WebhookCommitValue) -> None:
        if self._settled:
            return
        await self._registry._settle_reservation(self, result)
        self._settled = True
        self._abandon_event.set()

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
        self._registry_applied = False

    def __repr__(self) -> str:
        return "AnsweredPlaceholderReservation()"

    def abandon_before_submit(self) -> None:
        if self._settled or self._abandon_requested:
            return
        self._abandon_requested = True
        self._abandon_event.set()

    async def confirm(self, result: WebhookCommitValue) -> None:
        if self._settled:
            return
        await self._registry._settle_placeholder(self, result)
        self._settled = True
        self._abandon_event.set()

    async def confirm_fail_closed(self, result: WebhookCommitValue) -> None:
        if self._settled:
            return
        await self._registry._settle_placeholder(
            self,
            result,
            promote_linked_evidence=False,
        )
        self._settled = True
        self._abandon_event.set()

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


@dataclass(frozen=True, slots=True, repr=False)
class ProcessLeaseClaim:
    """Opaque proof that the exact local pending lease reached durable active."""

    call_control_id: str = field(repr=False)
    call_id: UUID = field(repr=False)
    generation: UUID = field(repr=False)
    token_digest: bytes = field(repr=False)

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


BackgroundTaskFactory = Callable[
    [Coroutine[object, object, None], str], asyncio.Task[None]
]


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
        uuid_factory: Callable[[], UUID] = uuid4,
        candidate_run_id: UUID | None = None,
        candidate_consumed: bool = False,
        admission_expires_at: datetime | None = None,
        background_task_factory: BackgroundTaskFactory = _default_background_task_factory,
    ) -> None:
        if (
            type(capacity) is not int
            or capacity <= 0
            or type(lease_ttl_seconds) is not int
            or lease_ttl_seconds <= 0
            or type(retention_days) is not int
            or retention_days <= 0
            or not all(
                isinstance(value, str) and value
                for value in (tenant_id, agent_id, deployment_id)
            )
            or not isinstance(stream_url, str)
            or not stream_url.startswith("wss://")
            or candidate_run_id is not None
            and not isinstance(candidate_run_id, UUID)
            or type(candidate_consumed) is not bool
            or candidate_consumed
            and candidate_run_id is None
            or not callable(background_task_factory)
            or admission_expires_at is not None
            and (
                not isinstance(admission_expires_at, datetime)
                or admission_expires_at.tzinfo is None
                or admission_expires_at.utcoffset() is None
            )
        ):
            raise ValueError("call_registry_config_invalid")
        self._writer = writer
        self._call_control = call_control
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._deployment_id = deployment_id
        self._capacity = capacity
        self._lease_ttl_seconds = lease_ttl_seconds
        self._stream_url = stream_url
        self._retention_days = retention_days
        self._utcnow = utcnow
        self._monotonic = monotonic
        self._token_factory = token_factory
        self._uuid_factory = uuid_factory
        self._candidate_run_id = candidate_run_id
        self._candidate_consumed = candidate_consumed
        self._admission_expires_at = (
            None
            if admission_expires_at is None
            else admission_expires_at.astimezone(UTC)
        )
        self._lock = asyncio.Lock()
        self._by_control: dict[str, _CallEntry] = {}
        self._by_call_id: dict[UUID, _CallEntry] = {}
        self._answered_placeholders: dict[str, _AnsweredPlaceholder] = {}
        self._permits_used = 0
        self._background_owner = _BackgroundTaskOwner(background_task_factory)
        self._internal_failure_code: str | None = None

    def __repr__(self) -> str:
        return "CallRegistry()"

    @property
    def candidate_run_id(self) -> UUID | None:
        return self._candidate_run_id

    @property
    def internal_failure_code(self) -> str | None:
        return self._internal_failure_code

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
        self, event: VerifiedWebhook, *, minted_monotonic: float
    ) -> _CallEntry:
        if event.call_control_id is None:
            raise CallAdmissionRejected("call_event_invalid")
        created_at = self._require_aware(self._utcnow())
        if not isinstance(minted_monotonic, int | float):
            raise CallAdmissionRejected("call_clock_invalid")
        raw_token = self._token_factory(32)
        if not isinstance(raw_token, str) or _TOKEN_PATTERN.fullmatch(raw_token) is None:
            raise CallAdmissionRejected("stream_token_invalid")
        call_id = self._uuid_factory()
        answer_id = self._uuid_factory()
        streaming_id = self._uuid_factory()
        hangup_id = self._uuid_factory()
        identifiers = (call_id, answer_id, streaming_id, hangup_id)
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
            initiated_at=self._require_aware(event.occurred_at),
            created_at=created_at,
            expires_at=expires_at,
            token_deadline=float(minted_monotonic) + self._lease_ttl_seconds,
            token_digest=digest,
            raw_token=raw_token,
            answer=_ActionSlot(answer_id),
            streaming=_ActionSlot(streaming_id),
            hangup_command_id=hangup_id,
            precommit_refcount=1,
        )

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
        return WebhookDurableEffect(lease=lease, operation=operation)

    def _terminal_effect(
        self, entry: _CallEntry, event: VerifiedWebhook
    ) -> WebhookDurableEffect:
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
        answered = entry.answer_evidence
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=uuid5(entry.call_id, event.event_id),
            deployment_id=self._deployment_id,
            call_id=entry.call_id,
            occurred_at=closed_at,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=entry.call_control_id,
                telnyx_call_leg_id=entry.call_leg_id,
                telnyx_call_session_id=entry.call_session_id,
                status="closed" if answered else "failed",
                disclosure_state="failed",
                started_at=entry.initiated_at if answered else None,
                ended_at=closed_at,
                end_reason="telnyx_hangup",
                retention_until=closed_at + timedelta(days=self._retention_days),
            ),
        )
        return WebhookDurableEffect(lease=lease, operation=operation)

    async def resolve_webhook(self, event: VerifiedWebhook) -> ResolvedWebhook:
        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

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
                    or existing.call_leg_id != event.call_leg_id
                    or existing.call_session_id != event.call_session_id
                    or existing.initiated_at != event.occurred_at.astimezone(UTC)
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
                entry = self._new_entry(event, minted_monotonic=now)
                if placeholder is not None:
                    if placeholder.durable:
                        entry.answer_evidence = True
                        entry.answer.state = "accepted"
                        if placeholder.precommit_refcount == 0:
                            self._answered_placeholders.pop(event.call_control_id, None)
                    else:
                        placeholder.linked_entry = entry
                self._by_control[entry.call_control_id] = entry
                self._by_call_id[entry.call_id] = entry
                self._permits_used += 1
        reservation = await self._register_call_reservation(entry, event.event_type)
        return ResolvedWebhook(self._pending_effect(entry), reservation)

    async def resolve_duplicate_webhook(self, event: VerifiedWebhook) -> ResolvedWebhook:
        """Resolve a classified duplicate without creating process-local identity."""

        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

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
                effect, await self._register_call_reservation(entry, event.event_type)
            )
        if placeholder is not None:
            return ResolvedWebhook(
                None, await self._register_placeholder_reservation(placeholder)
            )
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
                )
                self._answered_placeholders[event.call_control_id] = placeholder
        if call_entry is not None:
            return ResolvedWebhook(
                None, await self._register_call_reservation(call_entry, event.event_type)
            )
        if placeholder is None:
            raise RuntimeError("placeholder_resolution_failed")
        return ResolvedWebhook(
            None, await self._register_placeholder_reservation(placeholder)
        )

    async def _resolve_hangup(self, event: VerifiedWebhook) -> ResolvedWebhook:
        from projetv0_voice.telnyx.webhooks import ResolvedWebhook

        if event.call_control_id is None:
            raise CallAdmissionRejected("call_event_invalid")
        entry: _CallEntry | None
        async with self._lock:
            if self._qualification_expired():
                raise CallAdmissionRejected("qualification_window_expired")
            entry = self._by_control.get(event.call_control_id)
            if entry is None or entry.terminal_event is not None:
                return ResolvedWebhook(None)
            if (
                entry.call_leg_id != event.call_leg_id
                or entry.call_session_id != event.call_session_id
            ):
                raise CallAdmissionRejected("call_identity_conflict")
            entry.precommit_refcount += 1
        reservation = await self._register_call_reservation(entry, event.event_type)
        return ResolvedWebhook(self._terminal_effect(entry, event), reservation)

    async def _register_call_reservation(
        self, entry: _CallEntry, event_type: str
    ) -> CallReservation:
        reservation = CallReservation(self, entry, event_type)
        if self._background_owner.spawn(
            reservation._run_abandonment(), name="voice-reservation-owner"
        ):
            return reservation
        self._internal_failure_code = "background_task_registration_failed"
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
        if self._background_owner.spawn(
            reservation._run_abandonment(), name="voice-placeholder-owner"
        ):
            return reservation
        self._internal_failure_code = "background_task_registration_failed"
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
        async with self._lock:
            if reservation._registry_applied:
                return
            entry = self._by_control.get(reservation._entry.call_control_id)
            if entry is not reservation._entry or entry.generation != reservation._generation:
                reservation._registry_applied = True
                return
            if entry.precommit_refcount > 0:
                entry.precommit_refcount -= 1
            if reservation._event_type == "call.initiated" and isinstance(
                result, WebhookCommitResult
            ) and (
                result.receipt,
                result.effect,
            ) == ("first", "applied"):
                entry.durable = True
                entry.lease_state = "pending"
                if self._candidate_run_id is not None:
                    self._candidate_consumed = True
            elif isinstance(result, QualificationRunConsumed):
                self._candidate_consumed = True
            if entry.precommit_refcount == 0 and not entry.durable:
                entry.raw_token = None
                if not entry.capacity_released:
                    entry.capacity_released = True
                    self._permits_used -= 1
                self._by_control.pop(entry.call_control_id, None)
                self._by_call_id.pop(entry.call_id, None)
            reservation._registry_applied = True

    async def _settle_placeholder(
        self,
        reservation: _PlaceholderReservation,
        result: WebhookCommitValue | None,
        *,
        promote_linked_evidence: bool = True,
    ) -> None:
        completion: asyncio.Future[WebhookDisposition] | None = None
        async with self._lock:
            if reservation._registry_applied:
                return
            placeholder = self._answered_placeholders.get(
                reservation._placeholder.call_control_id
            )
            if placeholder is not reservation._placeholder:
                reservation._registry_applied = True
                return
            if placeholder.precommit_refcount > 0:
                placeholder.precommit_refcount -= 1
            if isinstance(result, WebhookCommitResult) and (
                result.receipt,
                result.effect,
            ) == ("first", "applied"):
                now = float(self._monotonic())
                if now >= placeholder.deadline:
                    if (
                        self._answered_placeholders.get(placeholder.call_control_id)
                        is placeholder
                    ):
                        self._answered_placeholders.pop(placeholder.call_control_id, None)
                    placeholder.linked_entry = None
                else:
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
                        linked.answer.state = "accepted"
                        linked.answer.disposition = 200
                        completion = linked.answer.completion
            if placeholder.precommit_refcount == 0 and not placeholder.durable:
                self._answered_placeholders.pop(placeholder.call_control_id, None)
                placeholder.linked_entry = None
            elif placeholder.precommit_refcount == 0 and placeholder.linked_entry is not None:
                self._answered_placeholders.pop(placeholder.call_control_id, None)
            reservation._registry_applied = True
        if completion is not None and not completion.done():
            from projetv0_voice.telnyx.webhooks import WebhookDisposition

            completion.set_result(WebhookDisposition(200))

    async def reconcile_after_commit(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
        result: WebhookCommitValue,
    ) -> WebhookDisposition:
        from projetv0_voice.telnyx.webhooks import WebhookDisposition

        reservation = resolution.reservation
        if isinstance(reservation, CallReservation):
            await reservation.confirm(result)
        elif isinstance(reservation, _PlaceholderReservation):
            await reservation.confirm_fail_closed(result)
        if isinstance(result, QualificationRunConsumed):
            return WebhookDisposition(503)
        if result.effect == "existing_terminal":
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
            await self._terminalize_after_hangup(event.call_control_id)
            return WebhookDisposition(200)
        if event.event_type != "call.initiated":
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
        linked_placeholder_entry: _CallEntry | None = None
        if isinstance(reservation, CallReservation):
            await reservation.confirm(result)
        elif isinstance(reservation, _PlaceholderReservation):
            async with self._lock:
                linked_placeholder_entry = reservation._placeholder.linked_entry
            await reservation.confirm_fail_closed(result)
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
        elif isinstance(reservation, _PlaceholderReservation):
            linked = linked_placeholder_entry
            if linked is not None:
                generation = CallGenerationHandle(
                    linked.call_control_id, linked.generation
                )
                if isinstance(result, WebhookCommitResult) and (
                    result.receipt,
                    result.effect,
                ) == ("first", "applied"):
                    abort_scheduled = self._schedule_abort_exact(
                        linked,
                        linked.generation,
                    )
            else:
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

        owner = False
        completion: asyncio.Future[WebhookDisposition] | None = None
        command_id: UUID | None = None
        raw_token: str | None = None
        entry: _CallEntry | None = None
        generation: UUID | None = None
        deadline_work: _TerminalWork | None = None
        async with self._lock:
            if self._qualification_expired():
                return WebhookDisposition(503)
            entry = self._by_control.get(call_control_id)
            if entry is None or entry.terminal_event is not None or not entry.durable:
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
            if action == "answer":
                call_result = await self._call_control.answer(
                    call_control_id, command_id=cast(UUID, command_id)
                )
            else:
                call_result = await self._call_control.start_streaming(
                    call_control_id,
                    StreamingStartV1(
                        stream_url=self._stream_url,
                        stream_auth_token=SecretStr(cast(str, raw_token)),
                    ),
                    command_id=cast(UUID, command_id),
                )
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
                    current.answer_evidence
                    if action == "answer"
                    else current.streaming_evidence
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
                    self._internal_failure_code = "background_task_registration_failed"
                    while True:
                        try:
                            await self._run_terminal_cleanup(
                                result_deadline_work,
                                retry_cancelled_io=True,
                            )
                            break
                        except asyncio.CancelledError:
                            continue
                    self._internal_failure_code = "background_task_registration_failed"
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

        if entry.terminal_event is not None:
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
        lifecycle_owners = tuple(
            cast(RegistryLifecycleOwner, owner)
            for owner in (entry.construction_owner, entry.session_owner)
            if owner is not None and self._valid_lifecycle_owner(owner)
        )
        return _TerminalWork(
            entry=entry,
            generation=entry.generation,
            action_owners=tuple(dict.fromkeys(action_owners)),
            lifecycle_owners=lifecycle_owners,
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
                lifecycle_owner.request_drain()
                lifecycle_waits.append(lifecycle_owner.wait())
            except BaseException:
                self._internal_failure_code = "lifecycle_drain_failed"
        if lifecycle_waits:
            results = await asyncio.gather(*lifecycle_waits, return_exceptions=True)
            if any(isinstance(result, BaseException) for result in results):
                self._internal_failure_code = "lifecycle_drain_failed"
        entry = work.entry
        closed_at = self._require_aware(self._utcnow())
        if work.persist_terminal:
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
                        closed_at=closed_at,
                    )
                except asyncio.CancelledError:
                    if retry_cancelled_io:
                        continue
                    self._internal_failure_code = "terminal_persistence_failed"
                except BaseException:
                    self._internal_failure_code = "terminal_persistence_failed"
                break
        if work.cleanup_hangup:
            while True:
                try:
                    await self._call_control.hangup(
                        entry.call_control_id,
                        command_id=entry.hangup_command_id,
                    )
                except asyncio.CancelledError:
                    if retry_cancelled_io:
                        continue
                except BaseException:
                    pass
                break
        generation = CallGenerationHandle(entry.call_control_id, work.generation)
        while True:
            try:
                await self.complete_terminal_cleanup(generation)
                break
            except asyncio.CancelledError:
                continue

    async def complete_terminal_cleanup(self, generation: CallGenerationHandle) -> bool:
        abort_target_clearers: tuple[
            tuple[_AbortTarget, Callable[[_AbortTarget], None]], ...
        ] = ()
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            if (
                entry is None
                or entry.generation != generation.generation
                or entry.terminal_event is None
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

    @staticmethod
    def _valid_lifecycle_owner(owner: object) -> bool:
        return callable(getattr(owner, "request_drain", None)) and callable(
            getattr(owner, "wait", None)
        )

    async def attach_construction_owner(
        self,
        generation: CallGenerationHandle,
        owner: RegistryLifecycleOwner,
    ) -> bool:
        if not isinstance(generation, CallGenerationHandle) or not self._valid_lifecycle_owner(
            owner
        ):
            return False
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            if (
                entry is None
                or entry.generation != generation.generation
                or entry.terminal_event is not None
                or entry.drain_intent
                or entry.construction_owner is not None
            ):
                return False
            entry.construction_owner = owner
            return True

    async def attach_session_owner(
        self,
        generation: CallGenerationHandle,
        session: object,
        owner: RegistryLifecycleOwner,
    ) -> bool:
        if not isinstance(generation, CallGenerationHandle) or not self._valid_lifecycle_owner(
            owner
        ):
            return False
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            if (
                entry is None
                or entry.generation != generation.generation
                or entry.terminal_event is not None
                or entry.drain_intent
                or entry.session_owner is not None
            ):
                return False
            entry.session = session
            entry.session_owner = owner
            return True

    async def request_generation_drain(
        self, generation: CallGenerationHandle
    ) -> bool:
        if not isinstance(generation, CallGenerationHandle):
            return False
        async with self._lock:
            entry = self._by_control.get(generation.call_control_id)
            if entry is None or entry.generation != generation.generation:
                return False
            entry.drain_intent = True
            owners = tuple(
                cast(RegistryLifecycleOwner, owner)
                for owner in (entry.construction_owner, entry.session_owner)
                if owner is not None and self._valid_lifecycle_owner(owner)
            )
        waits: list[Awaitable[None]] = []
        for lifecycle_owner in owners:
            try:
                lifecycle_owner.request_drain()
                waits.append(lifecycle_owner.wait())
            except BaseException:
                self._internal_failure_code = "lifecycle_drain_failed"
        if waits:
            results = await asyncio.gather(*waits, return_exceptions=True)
            if any(isinstance(result, BaseException) for result in results):
                self._internal_failure_code = "lifecycle_drain_failed"
        return True

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
            self._internal_failure_code = "background_task_registration_failed"
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
                    self._answered_placeholders.pop(call_control_id, None)
                    placeholder.linked_entry = None
                    expired_placeholders += 1
            for entry in tuple(self._by_control.values()):
                if (
                    not qualification_expired
                    and entry.attached
                    and entry.lease_state == "active"
                ):
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
