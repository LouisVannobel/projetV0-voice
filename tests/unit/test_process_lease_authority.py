from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from projetv0_voice.persistence.writer import WebhookCommitResult
from projetv0_voice.telnyx.call_control import CallControlResult
from projetv0_voice.telnyx.webhooks import VerifiedWebhook

NOW = datetime(2026, 8, 29, 10, tzinfo=UTC)
DIGEST = bytes.fromhex("0f007385b6f9d4b7eeb2748605afe1a984a0a3bfa3f014d09e2a784ce9e5cd1a")
DIGEST_B = bytes.fromhex("412dc46cc9e3cb26f29f7c1415c556349af62904c5d15b0a2d8cfdc5cfa22b34")


def _event(
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


class Writer:
    def __init__(self) -> None:
        self.commits: list[dict[str, object]] = []
        self.active_error: BaseException | None = None
        self.control_commits: list[object] = []

    async def commit_lease(self, **values: object) -> None:
        if values["state"] == "active" and self.active_error is not None:
            raise self.active_error
        self.commits.append(values)

    async def commit_control(self, command: object) -> None:
        self.control_commits.append(command)


class CallControl:
    def __init__(
        self,
        streaming_outcome: str = "accepted",
        *,
        hangup_outcome: str = "accepted",
    ) -> None:
        self.streaming_outcome = streaming_outcome
        self.hangup_outcome = hangup_outcome
        self.hangups: list[tuple[str, UUID]] = []

    async def answer(self, *_: object, **__: object) -> CallControlResult:
        return CallControlResult("accepted")

    async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
        return CallControlResult(self.streaming_outcome)  # type: ignore[arg-type]

    async def hangup(
        self, call_control_id: str, *, command_id: UUID, client_state: object = None
    ) -> CallControlResult:
        del client_state
        self.hangups.append((call_control_id, command_id))
        return CallControlResult(self.hangup_outcome)  # type: ignore[arg-type]


def _registry(
    writer: Writer,
    control: CallControl,
    clock: Any,
    utcnow: Any = lambda: NOW,
    background_task_factory: Any = None,
    capacity: int = 1,
    token_factory: Any = None,
    prefix_factory: Any = lambda: 0,
    candidate_run_id: UUID | None = None,
    qualification_observer: Any = None,
) -> Any:
    from projetv0_voice.admission import CallRegistry

    ids = iter(
        UUID(value)
        for value in (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            "44444444-4444-4444-8444-444444444444",
            "55555555-5555-4555-8555-555555555555",
            "66666666-6666-4666-8666-666666666666",
            "77777777-7777-4777-8777-777777777777",
            "88888888-8888-4888-8888-888888888888",
        )
    )
    kwargs: dict[str, object] = {}
    if background_task_factory is not None:
        kwargs["background_task_factory"] = background_task_factory
    if qualification_observer is not None:
        kwargs["qualification_observer"] = qualification_observer
    selected_token_factory = (
        (lambda _: "A" * 43) if token_factory is None else token_factory
    )
    return CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="agent-a",
        capacity=capacity,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=utcnow,
        monotonic=clock,
        token_factory=selected_token_factory,
        prefix_factory=prefix_factory,
        uuid_factory=lambda: next(ids),
        candidate_run_id=candidate_run_id,
        **kwargs,
    )


def _encoded_call_id(prefix: int, counter: int) -> UUID:
    payload = (prefix << 32) | counter
    high48 = payload >> 74
    mid12 = (payload >> 62) & ((1 << 12) - 1)
    low62 = payload & ((1 << 62) - 1)
    return UUID(
        int=(high48 << 80) | (4 << 76) | (mid12 << 64) | (0b10 << 62) | low62
    )


def test_process_call_id_allocator_encodes_boundaries_and_exhausts_permanently() -> None:
    from projetv0_voice.admission import _ProcessCallIdAllocator

    prefix = (1 << 90) - 1
    allocator = _ProcessCallIdAllocator(prefix_factory=lambda: prefix)

    first = allocator.next()
    assert first == _encoded_call_id(prefix, 0)
    assert (first.version, first.variant) == (4, "specified in RFC 4122")

    allocator._counter = (1 << 32) - 1  # noqa: SLF001
    last = allocator.next()
    assert last == _encoded_call_id(prefix, (1 << 32) - 1)
    with pytest.raises(RuntimeError, match="^call_identifier_exhausted$") as raised:
        allocator.next()
    assert raised.value.__cause__ is None
    with pytest.raises(RuntimeError, match="^call_identifier_exhausted$"):
        allocator.next()
    assert repr(allocator) == "ProcessCallIdAllocator()"
    assert not hasattr(allocator, "__dict__")
    assert allocator.__slots__ == ("_counter", "_prefix")


def test_process_call_id_allocator_prefixes_are_disjoint() -> None:
    from projetv0_voice.admission import _ProcessCallIdAllocator

    first = _ProcessCallIdAllocator(prefix_factory=lambda: 1)
    second = _ProcessCallIdAllocator(prefix_factory=lambda: 2)

    assert {first.next(), first.next()}.isdisjoint({second.next(), second.next()})


@pytest.mark.asyncio
async def test_failed_entry_construction_burns_allocated_call_identifier() -> None:
    writer = Writer()
    token_results = iter(("invalid", "A" * 43))
    registry = _registry(
        writer,
        CallControl(),
        lambda: 100.0,
        token_factory=lambda _size: next(token_results),
        prefix_factory=lambda: 7,
    )

    with pytest.raises(Exception, match="stream_token_invalid"):
        await registry.resolve_webhook(_event("call.initiated", "failed"))
    resolution = await registry.resolve_webhook(
        _event("call.initiated", "success")
    )

    assert resolution.effect is not None
    assert resolution.effect.lease is not None
    assert resolution.effect.lease["call_id"] == _encoded_call_id(7, 1)


async def _durable_waiting_wss(registry: Any) -> None:
    initiated = _event("call.initiated", "event-a")
    resolution = await registry.resolve_webhook(initiated)
    await registry.reconcile_after_commit(
        initiated, resolution, WebhookCommitResult("first", "applied")
    )
    answered = _event("call.answered", "event-b")
    resolution = await registry.resolve_webhook(answered)
    await registry.reconcile_after_commit(
        answered, resolution, WebhookCommitResult("first", "applied")
    )


async def _durable_waiting_wss_for(
    registry: Any,
    *,
    call_control_id: str,
    call_leg_id: str,
    call_session_id: str,
    event_prefix: str,
) -> None:
    initiated = _event(
        "call.initiated",
        f"{event_prefix}-initiated",
        call_control_id=call_control_id,
        call_leg_id=call_leg_id,
        call_session_id=call_session_id,
    )
    resolution = await registry.resolve_webhook(initiated)
    await registry.reconcile_after_commit(
        initiated, resolution, WebhookCommitResult("first", "applied")
    )
    answered = _event(
        "call.answered",
        f"{event_prefix}-answered",
        call_control_id=call_control_id,
        call_leg_id=call_leg_id,
        call_session_id=call_session_id,
    )
    resolution = await registry.resolve_webhook(answered)
    await registry.reconcile_after_commit(
        answered, resolution, WebhookCommitResult("first", "applied")
    )


@pytest.mark.asyncio
async def test_begin_drain_closes_new_admission_action_and_ordinary_claims() -> None:
    from projetv0_voice.admission import (
        CallAdmissionRejected,
        ProcessLeaseAuthority,
    )

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    initiated = _event("call.initiated", "event-before-drain")
    resolution = await registry.resolve_webhook(initiated)

    await registry.begin_drain()

    with pytest.raises(CallAdmissionRejected, match="^call_draining$") as captured:
        await registry.resolve_webhook(
            _event("call.initiated", "event-after-drain", call_control_id="other")
        )
    assert captured.value.status_code == 503
    disposition = await registry.reconcile_after_commit(
        initiated,
        resolution,
        WebhookCommitResult("first", "applied"),
    )
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a",
        token_digest=snapshot.token_digest,
    )

    assert disposition.status_code == 503
    assert claim is None
    assert await registry.qualification_state_valid() is True


@pytest.mark.asyncio
async def test_candidate_consumption_closes_readiness_but_keeps_accepted_claim() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    transitions: list[str] = []
    registry = _registry(
        writer,
        CallControl(),
        lambda: 100.0,
        candidate_run_id=UUID("99999999-9999-4999-8999-999999999999"),
        qualification_observer=transitions.append,
    )

    await _durable_waiting_wss(registry)
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a",
        token_digest=snapshot.token_digest,
    )

    assert await registry.qualification_state() == "consumed"
    assert claim is not None
    assert transitions == ["consumed"]


class _ConstructionOwner:
    def __init__(self) -> None:
        self.causes: list[str] = []
        self.closed = asyncio.Event()
        self.requested = asyncio.Event()

    def request_drain(self, cause: str) -> None:
        self.causes.append(cause)
        self.requested.set()

    async def wait(self) -> None:
        await self.closed.wait()


class _WaitBarrierEvent(asyncio.Event):
    def __init__(self) -> None:
        super().__init__()
        self.wait_entered = asyncio.Event()

    async def wait(self) -> bool:
        self.wait_entered.set()
        return await super().wait()


class _TwoWaiterEvent(asyncio.Event):
    def __init__(self) -> None:
        super().__init__()
        self.waiters = 0
        self.first_waiter = asyncio.Event()
        self.two_waiters = asyncio.Event()

    async def wait(self) -> bool:
        self.waiters += 1
        self.first_waiter.set()
        if self.waiters >= 2:
            self.two_waiters.set()
        return await super().wait()


class _SecondAcquireBarrierLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.acquires = 0
        self.second_entered = asyncio.Event()
        self.second_release = asyncio.Event()

    async def __aenter__(self) -> _SecondAcquireBarrierLock:
        self.acquires += 1
        if self.acquires == 2:
            self.second_entered.set()
            await self.second_release.wait()
        await self._lock.acquire()
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class _AcquireNoticeLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.entered = asyncio.Event()

    async def __aenter__(self) -> _AcquireNoticeLock:
        self.entered.set()
        await self._lock.acquire()
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class _SequencedCloseWriter(Writer):
    def __init__(
        self,
        outcomes: list[BaseException | None],
        *,
        blocked_attempts: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__()
        self.outcomes = outcomes
        self.blocked_attempts = blocked_attempts
        self.attempts: list[object] = []
        self.started = [asyncio.Event() for _ in outcomes]
        self.releases = [asyncio.Event() for _ in outcomes]

    async def commit_control(self, command: object) -> None:
        index = len(self.attempts)
        self.attempts.append(command)
        self.control_commits.append(command)
        self.started[index].set()
        if index in self.blocked_attempts:
            await self.releases[index].wait()
        outcome = self.outcomes[index]
        if outcome is not None:
            raise outcome


class _BlockingHangupControl(CallControl):
    def __init__(self) -> None:
        super().__init__()
        self.hangup_started = asyncio.Event()
        self.hangup_release = asyncio.Event()

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: object = None,
    ) -> CallControlResult:
        del client_state
        self.hangups.append((call_control_id, command_id))
        self.hangup_started.set()
        await self.hangup_release.wait()
        return CallControlResult("accepted")


async def _blocked_owner_task(release: asyncio.Event) -> None:
    await release.wait()


@pytest.mark.asyncio
async def test_atomic_consume_publishes_exact_immutable_claim_time_grant_once() -> None:
    from projetv0_voice.admission import (
        CallGenerationHandle,
        ProcessLeaseAuthority,
    )

    wall_clock = NOW
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0, utcnow=lambda: wall_clock)
    await _durable_waiting_wss(registry)
    wall_clock = NOW.replace(minute=7)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    wall_clock = NOW.replace(hour=12)
    release = asyncio.Event()
    tasks = [
        asyncio.create_task(_blocked_owner_task(release), name=f"candidate-{index}")
        for index in range(10)
    ]
    owners = [_ConstructionOwner() for _ in tasks]
    try:
        grants = await asyncio.gather(
            *(
                registry.consume_claim_for_construction(
                    claim,
                    "stream-exact",
                    owner,
                    task,
                )
                for owner, task in zip(owners, tasks, strict=True)
            )
        )
        winners = [grant for grant in grants if grant is not None]
        assert len(winners) == 1
        grant = winners[0]
        assert grant.call_id == claim.call_id
        assert isinstance(grant.generation, CallGenerationHandle)
        assert grant.lease_claim is claim
        assert grant.deployment_id == "agent-a"
        assert grant.telnyx_call_control_id == "control-a"
        assert grant.telnyx_call_leg_id == "leg-a"
        assert grant.telnyx_call_session_id == "session-a"
        assert grant.stream_id == "stream-exact"
        assert grant.started_at == NOW.replace(minute=7)
        assert grant.retention_until == NOW + timedelta(days=7)
        assert repr(grant) == "CallConstructionGrant()"
        assert not hasattr(grant, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            grant.stream_id = "changed"
        with pytest.raises(
            ValueError, match="^call_construction_grant_invalid$"
        ):
            dataclasses.replace(grant, deployment_id="")
        with pytest.raises(
            ValueError, match="^call_construction_grant_invalid$"
        ):
            dataclasses.replace(
                grant,
                generation=CallGenerationHandle(
                    "control-a",
                    UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
                ),
            )
    finally:
        release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_consume_requires_open_registration_live_deadline_and_retained_task() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    now = 100.0
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: now)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    release = asyncio.Event()
    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task
    live_task = asyncio.create_task(_blocked_owner_task(release))
    try:
        assert (
            await registry.consume_claim_for_construction(
                claim, "stream-a", owner, done_task
            )
            is None
        )
        now = 130.0
        assert (
            await registry.consume_claim_for_construction(
                claim, "stream-a", owner, live_task
            )
            is None
        )
        now = 129.999
        await registry.close_session_owner_registration()
        assert (
            await registry.consume_claim_for_construction(
                claim, "stream-a", owner, live_task
            )
            is None
        )
    finally:
        release.set()
        await live_task


@pytest.mark.asyncio
async def test_consume_rejects_retained_task_with_pending_cancellation() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    cancellation_seen = asyncio.Event()
    owner_started = asyncio.Event()
    release = asyncio.Event()

    async def cancelling_owner() -> None:
        try:
            owner_started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    owner_task = asyncio.create_task(cancelling_owner())
    await owner_started.wait()
    owner_task.cancel("candidate-cancel")
    await cancellation_seen.wait()
    owner = _ConstructionOwner()
    try:
        assert owner_task.cancelling() > 0
        assert (
            await registry.consume_claim_for_construction(
                claim,
                "stream-a",
                owner,
                owner_task,
            )
            is None
        )
    finally:
        release.set()
        await owner_task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "commit_result",
    [
        WebhookCommitResult("first", "applied"),
        WebhookCommitResult("duplicate", "duplicate"),
    ],
)
async def test_provider_pending_commit_precedes_freeze_and_owns_external_completion(
    commit_result: WebhookCommitResult,
) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority, TerminalProposal

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    start_owner = asyncio.Event()
    authority_ready = asyncio.Event()
    authority_box: list[object] = []

    async def owner_flow() -> None:
        await start_owner.wait()
        task = asyncio.current_task()
        assert task is not None
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="session_construction_failed",
                metric_class="failed",
                cleanup_hangup=True,
            ),
        )
        authority_box.append(authority)
        authority_ready.set()

    owner_task = asyncio.create_task(owner_flow())
    grant = await registry.consume_claim_for_construction(
        claim, "stream-a", owner, owner_task
    )
    assert grant is not None
    hangup = _event("call.hangup", "hangup-a")
    resolution = await registry.resolve_webhook(hangup)
    assert resolution.reservation is not None
    entry = registry._by_control["control-a"]  # noqa: SLF001
    barrier = _WaitBarrierEvent()
    entry.terminal_settled_event = barrier

    start_owner.set()
    await barrier.wait_entered.wait()
    assert authority_ready.is_set() is False
    await resolution.reservation.confirm(commit_result)
    await authority_ready.wait()
    await owner_task

    authority = authority_box[0]
    assert repr(authority) == "TerminalAuthority()"
    assert authority.status == "closed"
    assert authority.reason == "telnyx_hangup"
    assert authority.metric_class == "closed"
    assert authority.persist_call is False
    assert authority.persist_lease is False
    assert authority.cleanup_hangup is False
    assert await registry.complete_reserved_terminal(authority) is True
    assert await registry.snapshot("control-a") is None
    assert control.hangups == []


@pytest.mark.asyncio
async def test_local_terminal_authority_is_first_writer_and_completes_fixed_work_once() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority, TerminalProposal

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    start_owner = asyncio.Event()
    completion_box: list[bool] = []

    async def owner_flow() -> None:
        await start_owner.wait()
        task = asyncio.current_task()
        assert task is not None
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="session_construction_failed",
                metric_class="failed",
                cleanup_hangup=True,
            ),
        )
        assert authority.persist_call is True
        assert authority.persist_lease is True
        assert authority.cleanup_hangup is True
        completion_box.append(await registry.complete_reserved_terminal(authority))

    owner_task = asyncio.create_task(owner_flow())
    grant = await registry.consume_claim_for_construction(
        claim, "stream-a", owner, owner_task
    )
    assert grant is not None
    start_owner.set()
    await owner_task

    assert completion_box == [True]
    assert [commit["state"] for commit in writer.commits] == ["active", "terminal"]
    assert len(control.hangups) == 1
    assert await registry.complete_reserved_terminal(object()) is False
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_provider_authority_completion_is_takeover_safe_after_promoter_cancellation() -> None:
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    hangup = _event("call.hangup", "hangup-takeover")
    resolution = await registry.resolve_webhook(hangup)
    assert resolution.reservation is not None
    promoted = asyncio.Event()
    hold = asyncio.Event()
    authority_box: list[object] = []

    async def promote_then_block() -> None:
        await resolution.reservation.confirm(
            WebhookCommitResult("first", "applied")
        )
        entry = registry._by_control["control-a"]  # noqa: SLF001
        assert entry.terminal_authority is not None
        authority_box.append(entry.terminal_authority)
        promoted.set()
        await hold.wait()

    promoter = asyncio.create_task(promote_then_block())
    await promoted.wait()
    promoter.cancel("finalizer-cancelled")
    await asyncio.gather(promoter, return_exceptions=True)

    assert await registry.complete_reserved_terminal(authority_box[0]) is True
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_provider_finalizer_cancellation_waits_owner_then_completes_and_reraises() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    owner_release = asyncio.Event()
    owner_task = asyncio.create_task(_blocked_owner_task(owner_release))
    owner._task = owner_task
    grant = await registry.consume_claim_for_construction(
        claim, "stream-a", owner, owner_task
    )
    assert grant is not None
    hangup = _event("call.hangup", "hangup-cancelled-finalizer")
    resolution = await registry.resolve_webhook(hangup)
    finalizer = asyncio.create_task(
        registry.reconcile_after_commit(
            hangup,
            resolution,
            WebhookCommitResult("first", "applied"),
        )
    )
    await owner.requested.wait()

    finalizer.cancel("finalizer-cancelled")
    owner.closed.set()
    owner_release.set()
    await owner_task
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await finalizer

    assert cancelled.value.args == ("finalizer-cancelled",)
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_duplicate_provider_finalizer_joins_fixed_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    event = _event("call.hangup", "hangup-join")
    first = await registry.resolve_webhook(event)
    duplicate = await registry.resolve_duplicate_webhook(event)
    assert first.reservation is not None
    assert duplicate.reservation is not None
    first_entered = asyncio.Event()
    first_release = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0
    complete = registry.complete_reserved_terminal

    async def block_first_completion(authority: object) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await first_release.wait()
        else:
            second_entered.set()
        return await complete(authority)

    monkeypatch.setattr(
        registry,
        "complete_reserved_terminal",
        block_first_completion,
    )
    first_finalizer = asyncio.create_task(
        registry.reconcile_after_commit(
            event,
            first,
            WebhookCommitResult("first", "applied"),
        )
    )
    await first_entered.wait()
    duplicate_finalizer = asyncio.create_task(
        registry.reconcile_after_commit(
            event,
            duplicate,
            WebhookCommitResult("duplicate", "duplicate"),
        )
    )
    await second_entered.wait()

    assert duplicate_finalizer.done()
    duplicate_disposition = await duplicate_finalizer
    assert duplicate_disposition.status_code == 200
    assert await registry.snapshot("control-a") is None
    first_release.set()
    first_disposition = await first_finalizer

    assert first_disposition.status_code == 200
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_first_provider_commit_wakes_owner_despite_pending_duplicate() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority, TerminalProposal

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    start_owner = asyncio.Event()
    authority_ready = asyncio.Event()

    async def owner_flow() -> None:
        await start_owner.wait()
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="session_construction_failed",
                metric_class="failed",
                cleanup_hangup=True,
            ),
        )
        assert authority.reason == "telnyx_hangup"
        authority_ready.set()

    owner_task = asyncio.create_task(owner_flow())
    grant = await registry.consume_claim_for_construction(
        claim, "stream-a", owner, owner_task
    )
    assert grant is not None
    first = await registry.resolve_webhook(_event("call.hangup", "hangup-first"))
    duplicate = await registry.resolve_duplicate_webhook(
        _event("call.hangup", "hangup-duplicate")
    )
    assert first.reservation is not None
    assert duplicate.reservation is not None
    entry = registry._by_control["control-a"]  # noqa: SLF001
    barrier = _WaitBarrierEvent()
    entry.terminal_settled_event = barrier
    start_owner.set()
    await barrier.wait_entered.wait()

    await first.reservation.confirm(WebhookCommitResult("first", "applied"))

    assert barrier.is_set()
    await authority_ready.wait()
    await owner_task
    await duplicate.reservation.confirm(
        WebhookCommitResult("duplicate", "duplicate")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hangup_outcome",
    ["rejected", "rate_limited", "retryable_not_sent", "outcome_unknown"],
)
async def test_nonaccepted_terminal_hangup_latches_process_failure(
    hangup_outcome: str,
) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority, TerminalProposal

    writer = Writer()
    registry = _registry(
        writer,
        CallControl(hangup_outcome=hangup_outcome),
        lambda: 100.0,
    )
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    start = asyncio.Event()

    async def owner_flow() -> None:
        await start.wait()
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="call_failed",
                metric_class="failed",
                cleanup_hangup=True,
            ),
        )
        assert await registry.complete_reserved_terminal(authority)

    owner_task = asyncio.create_task(owner_flow())
    grant = await registry.consume_claim_for_construction(
        claim, "stream-a", owner, owner_task
    )
    assert grant is not None
    start.set()
    await owner_task

    assert registry.internal_failure_code == "terminal_hangup_failed"


@pytest.mark.asyncio
async def test_terminal_removal_retries_lock_after_cancellation_then_reraises() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority, TerminalProposal

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    reserve = asyncio.Event()
    complete = asyncio.Event()
    authority_box: list[object] = []

    async def owner_flow() -> None:
        await reserve.wait()
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="call_failed",
                metric_class="failed",
                cleanup_hangup=False,
            ),
        )
        authority_box.append(authority)
        complete.set()
        await completion_start.wait()
        await registry.complete_reserved_terminal(authority)

    completion_start = asyncio.Event()
    owner_task = asyncio.create_task(owner_flow())
    owner._task = owner_task
    grant = await registry.consume_claim_for_construction(
        claim, "stream-a", owner, owner_task
    )
    assert grant is not None
    reserve.set()
    await complete.wait()
    barrier_lock = _SecondAcquireBarrierLock()
    registry._lock = barrier_lock  # type: ignore[assignment]  # noqa: SLF001
    completion_start.set()
    await barrier_lock.second_entered.wait()

    owner_task.cancel("removal-cancelled")
    barrier_lock.second_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await owner_task

    assert cancelled.value.args == ("removal-cancelled",)
    assert await registry.snapshot("control-a") is None
    assert authority_box


@pytest.mark.asyncio
async def test_active_unconsumed_required_drain_waits_for_provider_pending_authority() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    hangup = _event("call.hangup", "hangup-pending-required")
    resolution = await registry.resolve_webhook(hangup)
    assert resolution.reservation is not None
    entry = registry._by_control["control-a"]  # noqa: SLF001
    barrier = _WaitBarrierEvent()
    entry.terminal_settled_event = barrier
    drain_task = asyncio.create_task(
        registry.prepare_required_recording_drain(claim.call_id)
    )
    wait_entered = asyncio.create_task(barrier.wait_entered.wait())

    done, _pending = await asyncio.wait(
        (drain_task, wait_entered),
        return_when=asyncio.FIRST_COMPLETED,
    )
    waited_for_provider = wait_entered in done and drain_task not in done
    if waited_for_provider:
        await resolution.reservation.confirm(
            WebhookCommitResult("first", "applied")
        )
        assert await drain_task is None
    else:
        resolution.reservation.abandon_before_submit()
        await resolution.reservation._run_abandonment()
    wait_entered.cancel()
    await asyncio.gather(wait_entered, return_exceptions=True)

    assert waited_for_provider


@pytest.mark.asyncio
async def test_owner_self_required_drain_latches_without_cancel_or_self_wait() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority, TerminalProposal

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    owner = _ConstructionOwner()
    start = asyncio.Event()
    grant_box: list[object] = []

    async def owner_flow() -> None:
        await start.wait()
        assert await registry.prepare_required_recording_drain(claim.call_id) is None
        authority = await registry.reserve_or_read_terminal(
            grant_box[0],
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="recording_required_error",
                metric_class="failed",
                cleanup_hangup=False,
            ),
        )
        assert await registry.complete_reserved_terminal(authority)

    owner_task = asyncio.create_task(owner_flow())
    owner._task = owner_task
    grant = await registry.consume_claim_for_construction(
        claim,
        "stream-a",
        owner,
        owner_task,
    )
    assert grant is not None
    grant_box.append(grant)
    start.set()
    await owner_task

    assert owner.causes == ["recording_required_error"]
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_close_session_registration_terminalizes_unconsumed_as_process_draining() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None

    await registry.close_session_owner_registration()

    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert [commit["state"] for commit in writer.commits] == ["active", "terminal"]
    assert len(writer.control_commits) == 1
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.status == "closed"
    assert operation.payload.end_reason == "process_draining"
    assert len(control.hangups) == 1


@pytest.mark.asyncio
async def test_process_close_waits_for_active_unconsumed_provider_pending_authority() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    hangup = _event("call.hangup", "hangup-pending-close")
    resolution = await registry.resolve_webhook(hangup)
    assert resolution.reservation is not None
    entry = registry._by_control["control-a"]  # noqa: SLF001
    barrier = _WaitBarrierEvent()
    entry.terminal_settled_event = barrier
    closing = asyncio.create_task(registry.close_session_owner_registration())
    wait_entered = asyncio.create_task(barrier.wait_entered.wait())

    done, _pending = await asyncio.wait(
        (closing, wait_entered),
        return_when=asyncio.FIRST_COMPLETED,
    )
    waited_for_provider = wait_entered in done and closing not in done
    if waited_for_provider:
        await resolution.reservation.confirm(
            WebhookCommitResult("first", "applied")
        )
        authority = entry.terminal_authority
        assert authority is not None
        assert await registry.complete_reserved_terminal(authority)
        await closing
    else:
        resolution.reservation.abandon_before_submit()
        await resolution.reservation._run_abandonment()
    wait_entered.cancel()
    await asyncio.gather(wait_entered, return_exceptions=True)

    assert waited_for_provider
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_close_cancel_then_retry_failure_preserves_first_cancel_and_fixed_work() -> None:
    writer = _SequencedCloseWriter(
        [
            asyncio.CancelledError("writer-attempt-cancel"),
            RuntimeError("writer-ordinary-secret"),
        ],
        blocked_attempts=frozenset({0}),
    )
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    from projetv0_voice.admission import ProcessLeaseAuthority

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    closing = asyncio.create_task(registry.close_session_owner_registration())
    await writer.started[0].wait()
    entry = registry._by_control["control-a"]  # noqa: SLF001
    authority = entry.terminal_authority
    assert authority is not None

    closing.cancel("first-close-cancel")
    writer.releases[0].set()
    await writer.started[1].wait()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await closing

    assert cancelled.value.args == ("first-close-cancel",)
    assert len(writer.attempts) == 2
    operations = [
        command.payload["operation"]  # type: ignore[union-attr]
        for command in writer.attempts
    ]
    assert operations[0].operation_id == operations[1].operation_id
    assert operations[0].occurred_at == operations[1].occurred_at
    assert operations[0].payload == operations[1].payload
    assert operations[0].payload.retention_until == authority._entry.claimed_at + timedelta(days=7)  # noqa: SLF001
    assert registry.internal_failure_code == "terminal_persistence_failed"
    assert await registry.snapshot("control-a") is not None
    assert await registry.live_call_count() == 1
    assert [commit["state"] for commit in writer.commits] == ["active"]
    assert control.hangups == []
    close_task = getattr(registry, "_session_close_task", None)
    assert close_task is not None
    assert authority._completion_owner is close_task  # noqa: SLF001


@pytest.mark.asyncio
async def test_two_concurrent_closes_share_one_successful_fixed_work_barrier() -> None:
    writer = _SequencedCloseWriter([None], blocked_attempts=frozenset({0}))
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    from projetv0_voice.admission import ProcessLeaseAuthority

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    first = asyncio.create_task(registry.close_session_owner_registration())
    await writer.started[0].wait()
    entry = registry._by_control["control-a"]  # noqa: SLF001
    authority = entry.terminal_authority
    assert authority is not None
    notice_lock = _AcquireNoticeLock()
    registry._lock = notice_lock  # type: ignore[assignment]  # noqa: SLF001
    second = asyncio.create_task(registry.close_session_owner_registration())
    await notice_lock.entered.wait()
    second_waited = not second.done()

    writer.releases[0].set()
    assert await asyncio.gather(first, second) == [None, None]

    assert second_waited
    close_task = getattr(registry, "_session_close_task", None)
    assert close_task is not None
    assert close_task.done()
    assert authority._completion_owner is close_task  # noqa: SLF001
    assert len(writer.attempts) == 1
    assert [commit["state"] for commit in writer.commits] == ["active", "terminal"]
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert not any(
        task is not asyncio.current_task()
        and task.get_name() == "voice-session-owner-close"
        and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_cancelled_close_joiner_is_local_and_core_effects_remain_once() -> None:
    writer = _SequencedCloseWriter([None], blocked_attempts=frozenset({0}))
    control = _BlockingHangupControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    from projetv0_voice.admission import ProcessLeaseAuthority

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    first = asyncio.create_task(registry.close_session_owner_registration())
    await writer.started[0].wait()
    notice_lock = _AcquireNoticeLock()
    registry._lock = notice_lock  # type: ignore[assignment]  # noqa: SLF001
    second = asyncio.create_task(registry.close_session_owner_registration())
    await notice_lock.entered.wait()

    second.cancel("joiner-first-cancel")
    writer.releases[0].set()
    await control.hangup_started.wait()
    second.cancel("joiner-second-cancel")
    control.hangup_release.set()
    await first
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await second

    assert cancelled.value.args == ("joiner-first-cancel",)
    assert len(writer.attempts) == 1
    assert [commit["state"] for commit in writer.commits] == ["active", "terminal"]
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_shared_close_failure_is_retained_without_sequential_restart() -> None:
    writer = _SequencedCloseWriter(
        [RuntimeError("close-persistence-secret")],
        blocked_attempts=frozenset({0}),
    )
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    from projetv0_voice.admission import ProcessLeaseAuthority

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    first = asyncio.create_task(registry.close_session_owner_registration())
    await writer.started[0].wait()
    entry = registry._by_control["control-a"]  # noqa: SLF001
    authority = entry.terminal_authority
    assert authority is not None
    token = authority.completion_token
    closed_at = authority._closed_at  # noqa: SLF001
    notice_lock = _AcquireNoticeLock()
    registry._lock = notice_lock  # type: ignore[assignment]  # noqa: SLF001
    second = asyncio.create_task(registry.close_session_owner_registration())
    await notice_lock.entered.wait()
    writer.releases[0].set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert all(str(result) == "terminal_persistence_failed" for result in results)
    assert len(writer.attempts) == 1
    with pytest.raises(RuntimeError, match="^terminal_persistence_failed$"):
        await registry.close_session_owner_registration()
    assert len(writer.attempts) == 1
    retained = registry._by_control["control-a"].terminal_authority  # noqa: SLF001
    assert retained is authority
    assert retained.completion_token == token
    assert retained._closed_at == closed_at  # noqa: SLF001
    close_task = getattr(registry, "_session_close_task", None)
    assert close_task is not None
    assert authority._completion_owner is close_task  # noqa: SLF001
    assert await registry.live_call_count() == 1


@pytest.mark.asyncio
async def test_provider_pending_two_closes_share_unique_provider_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    event = _event("call.hangup", "provider-pending-two-closes")
    resolution = await registry.resolve_webhook(event)
    assert resolution.reservation is not None
    entry = registry._by_control["control-a"]  # noqa: SLF001
    waiters = _TwoWaiterEvent()
    entry.terminal_settled_event = waiters
    settlement_entered = asyncio.Event()
    settlement_release = asyncio.Event()
    settle = registry._settle_reservation  # noqa: SLF001
    complete = registry.complete_reserved_terminal
    provider_completion_entered = asyncio.Event()
    close_completion_entered = asyncio.Event()
    completion_release = asyncio.Event()
    completion_authorities: list[object] = []

    async def block_settlement(*args: object) -> None:
        settlement_entered.set()
        await settlement_release.wait()
        await settle(*args)  # type: ignore[arg-type]

    async def block_completion(authority: object) -> bool:
        completion_authorities.append(authority)
        task = asyncio.current_task()
        if task is not None and task.get_name() == "voice-session-owner-close":
            close_completion_entered.set()
        else:
            provider_completion_entered.set()
        await completion_release.wait()
        return await complete(authority)

    monkeypatch.setattr(registry, "_settle_reservation", block_settlement)
    monkeypatch.setattr(registry, "complete_reserved_terminal", block_completion)
    provider = asyncio.create_task(
        registry.reconcile_after_commit(
            event,
            resolution,
            WebhookCommitResult("first", "applied"),
        )
    )
    await settlement_entered.wait()
    provider.cancel("provider-finalizer-cancel")
    first_close = asyncio.create_task(registry.close_session_owner_registration())
    second_close = asyncio.create_task(registry.close_session_owner_registration())
    close_returned = asyncio.Event()

    def note_close_returned(_task: asyncio.Task[None]) -> None:
        close_returned.set()

    first_close.add_done_callback(note_close_returned)
    second_close.add_done_callback(note_close_returned)
    await waiters.first_waiter.wait()
    close_task = getattr(registry, "_session_close_task", None)
    assert close_task is not None
    second_close.cancel("r8-close-caller-cancel")

    settlement_release.set()
    await provider_completion_entered.wait()
    close_wait = asyncio.create_task(close_completion_entered.wait())
    close_return = asyncio.create_task(close_returned.wait())
    done, _pending = await asyncio.wait(
        (close_wait, close_return),
        return_when=asyncio.FIRST_COMPLETED,
    )
    close_joined_provider = close_wait in done and close_return not in done
    for waiter in (close_wait, close_return):
        if not waiter.done():
            waiter.cancel()
    await asyncio.gather(close_wait, close_return, return_exceptions=True)

    provider_authority = entry.terminal_authority
    assert close_joined_provider
    assert provider_authority is not None
    assert provider_authority.persist_call is False
    assert provider_authority.persist_lease is False
    assert provider_authority.cleanup_hangup is False
    assert provider_authority._completion_owner is not close_task  # noqa: SLF001
    completion_token = provider_authority.completion_token
    closed_at = provider_authority._closed_at  # noqa: SLF001
    assert entry.terminal_event == "call.hangup"
    assert entry.terminal_state == "reserved"
    assert await registry.live_call_count() == 1
    assert not first_close.done()
    assert not second_close.done()

    completion_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await provider
    await first_close
    with pytest.raises(asyncio.CancelledError) as close_cancelled:
        await second_close

    assert cancelled.value.args == ("provider-finalizer-cancel",)
    assert close_cancelled.value.args == ("r8-close-caller-cancel",)
    assert len(completion_authorities) == 2
    assert all(authority is provider_authority for authority in completion_authorities)
    assert provider_authority.completion_token == completion_token
    assert provider_authority._closed_at == closed_at  # noqa: SLF001
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert writer.control_commits == []
    assert [commit["state"] for commit in writer.commits] == ["active"]
    assert control.hangups == []
    assert close_task.done()
    assert not any(
        task is not asyncio.current_task()
        and task.get_name()
        in {"voice-provider-terminal-settlement", "voice-session-owner-close"}
        and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_cancelled_process_close_finishes_all_reserved_work_then_reraises() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0, capacity=2)
    await _durable_waiting_wss_for(
        registry,
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        event_prefix="first",
    )
    await _durable_waiting_wss_for(
        registry,
        call_control_id="control-b",
        call_leg_id="leg-b",
        call_session_id="session-b",
        event_prefix="second",
    )
    authority = ProcessLeaseAuthority(registry)
    first_claim = await authority.claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    second_claim = await authority.claim_once(
        call_control_id="control-b", token_digest=DIGEST
    )
    assert first_claim is not None
    assert second_claim is not None
    owner = _ConstructionOwner()
    owner_release = asyncio.Event()
    owner_task = asyncio.create_task(_blocked_owner_task(owner_release))
    owner._task = owner_task
    grant = await registry.consume_claim_for_construction(
        first_claim,
        "stream-a",
        owner,
        owner_task,
    )
    assert grant is not None
    closing = asyncio.create_task(registry.close_session_owner_registration())
    await owner.requested.wait()
    late_provider = await registry.resolve_webhook(
        _event(
            "call.hangup",
            "late-provider-after-close",
            call_control_id="control-a",
            call_leg_id="leg-a",
            call_session_id="session-a",
        )
    )
    provider_race_blocked = late_provider.reservation is None

    closing.cancel("process-close-cancel")
    owner.closed.set()
    owner_release.set()
    await owner_task
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await closing
    if late_provider.reservation is not None:
        late_provider.reservation.abandon_before_submit()
        await late_provider.reservation._run_abandonment()

    assert cancelled.value.args == ("process-close-cancel",)
    assert provider_race_blocked
    assert await registry.snapshot("control-b") is None
    assert len(writer.control_commits) == 1
    operation = writer.control_commits[0].payload["operation"]  # type: ignore[union-attr]
    assert operation.payload.end_reason == "process_draining"
    assert [commit["state"] for commit in writer.commits].count("terminal") == 1
    assert len(control.hangups) == 1


@pytest.mark.asyncio
async def test_digest_only_claim_race_has_one_post_commit_winner() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    now = 100.0
    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: now)
    await _durable_waiting_wss(registry)
    before = await registry.snapshot("control-a")
    assert before is not None
    assert before.streaming_state == "accepted"
    assert before.raw_token_retained is False

    authority = ProcessLeaseAuthority(registry)
    claims = await asyncio.gather(
        *(
            authority.claim_once(call_control_id="control-a", token_digest=DIGEST)
            for _ in range(10)
        )
    )

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert repr(winners[0]) == "ProcessLeaseClaim()"
    assert [commit["state"] for commit in writer.commits] == ["active"]
    after = await registry.snapshot("control-a")
    assert after is not None
    assert after.lease_state == "active"
    assert (
        await authority.claim_once(call_control_id="control-a", token_digest=DIGEST)
        is None
    )
    assert (
        await authority.claim_once(call_control_id="control-a", token_digest=b"x" * 32)
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("now", "wins"),
    [(129.999, True), (130.0, False), (130.001, False)],
)
async def test_claim_deadline_epsilon_wins_but_equality_and_later_lose(
    now: float, wins: bool
) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    clock = 100.0
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: clock)
    await _durable_waiting_wss(registry)
    clock = now

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )

    assert (claim is not None) is wins
    assert [commit["state"] for commit in writer.commits] == (["active"] if wins else [])


@pytest.mark.asyncio
async def test_wall_clock_jump_cannot_override_monotonic_claim_window() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    monotonic = 100.0
    wall_clock = NOW
    writer = Writer()
    registry = _registry(
        writer,
        CallControl(),
        lambda: monotonic,
        utcnow=lambda: wall_clock,
    )
    await _durable_waiting_wss(registry)
    wall_clock = datetime(2099, 1, 1, tzinfo=UTC)
    monotonic = 129.999

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )

    assert claim is not None
    assert [commit["state"] for commit in writer.commits] == ["active"]


@pytest.mark.asyncio
async def test_reaper_wins_against_claim_blocked_before_active_commit() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BarrierWriter(Writer):
        async def commit_lease(self, **values: object) -> None:
            if values["state"] == "active":
                active_started.set()
                await release_active.wait()
            self.commits.append(values)

    now = 100.0
    writer = BarrierWriter()
    registry = _registry(writer, CallControl(), lambda: now)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)
    claim_task = asyncio.create_task(
        authority.claim_once(call_control_id="control-a", token_digest=DIGEST)
    )
    await active_started.wait()
    now = 130.0

    assert await registry.reap_expired() == 1
    release_active.set()
    assert await claim_task is None
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert [commit["state"] for commit in writer.commits] == ["terminal", "active"]


@pytest.mark.asyncio
async def test_unknown_active_persistence_then_matching_abort_terminalizes_once() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    writer.active_error = asyncio.CancelledError()
    control = CallControl(streaming_outcome="outcome_unknown")
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)

    with pytest.raises(asyncio.CancelledError):
        await authority.claim_once(call_control_id="control-a", token_digest=DIGEST)
    authority.schedule_abort_if_matches(
        call_control_id="control-a", token_digest=DIGEST
    )
    authority.schedule_abort_if_matches(
        call_control_id="control-a", token_digest=DIGEST
    )
    await registry.wait_background()

    assert await registry.snapshot("control-a") is None
    assert [commit["state"] for commit in writer.commits] == ["terminal"]
    assert len(control.hangups) == 1


def test_unauthenticated_gate_is_fixed_size_closeable_and_release_is_idempotent() -> None:
    from projetv0_voice.admission import SynchronousUnauthenticatedGate

    gate = SynchronousUnauthenticatedGate(2)
    first = gate.try_acquire()
    second = gate.try_acquire()

    assert first is not None
    assert second is not None
    assert gate.try_acquire() is None
    first.release()
    first.release()
    replacement = gate.try_acquire()
    assert replacement is not None
    gate.close()
    second.release()
    replacement.release()
    assert gate.try_acquire() is None
    assert gate.in_use == 0


@pytest.mark.asyncio
async def test_attached_active_call_survives_past_token_deadline() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    now = 100.0
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: now)
    await _durable_waiting_wss(registry)
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    assert await registry.mark_attached(claim) is True
    now = 131.0

    assert await registry.reap_expired() == 0
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.lease_state == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late_outcome",
    [
        "accepted",
        "rate_limited",
        "retryable_not_sent",
        "rejected",
        "outcome_unknown",
        "exception",
    ],
)
async def test_authenticated_wss_evidence_wins_every_late_streaming_outcome(
    late_outcome: str,
) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    entered = asyncio.Event()
    release = asyncio.Event()

    class LateStreaming(CallControl):
        async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
            entered.set()
            await release.wait()
            if late_outcome == "exception":
                raise RuntimeError("synthetic late provider failure")
            return CallControlResult(late_outcome)  # type: ignore[arg-type]

    writer = Writer()
    control = LateStreaming()
    registry = _registry(writer, control, lambda: 100.0)
    initiated = _event("call.initiated", "event-a")
    resolution = await registry.resolve_webhook(initiated)
    await registry.reconcile_after_commit(
        initiated, resolution, WebhookCommitResult("first", "applied")
    )
    answered = _event("call.answered", "event-b")
    resolution = await registry.resolve_webhook(answered)
    streaming_owner = asyncio.create_task(
        registry.reconcile_after_commit(
            answered, resolution, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )
    assert claim is not None
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.streaming_state == "accepted"

    release.set()
    late = await streaming_owner
    snapshot = await registry.snapshot("control-a")
    assert late.status_code == 200
    assert snapshot is not None
    assert snapshot.streaming_state == "accepted"
    assert snapshot.lease_state == "active"


@pytest.mark.asyncio
async def test_delayed_public_abort_cannot_hit_replacement_with_reused_digest() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    release_delayed = asyncio.Event()

    def delayed_factory(
        coroutine: Any,
        name: str,
    ) -> asyncio.Task[None]:
        async def delayed() -> None:
            await release_delayed.wait()
            await coroutine

        return asyncio.create_task(delayed(), name=name)

    writer = Writer()
    control = CallControl()
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)
    writer.active_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await authority.claim_once(call_control_id="control-a", token_digest=DIGEST)
    writer.active_error = None
    registry._background_owner._task_factory = delayed_factory  # type: ignore[attr-defined]

    authority.schedule_abort_if_matches(
        call_control_id="control-a",
        token_digest=DIGEST,
    )
    hangup = _event("call.hangup", "event-hangup")
    hangup_resolution = await registry.resolve_webhook(hangup)
    await registry.reconcile_after_commit(
        hangup,
        hangup_resolution,
        WebhookCommitResult("first", "applied"),
    )
    replacement_event = _event("call.initiated", "event-replacement")
    replacement = await registry.resolve_webhook(replacement_event)
    assert replacement.reservation is not None
    await replacement.reservation.confirm(  # type: ignore[attr-defined]
        WebhookCommitResult("first", "applied")
    )
    replacement_handle = await registry.generation_handle("control-a")

    release_delayed.set()
    await registry.join_until_empty()

    assert replacement_handle is not None
    assert await registry.generation_handle("control-a") == replacement_handle
    assert await registry.snapshot("control-a") is not None


@pytest.mark.asyncio
async def test_public_abort_schedules_exact_target_while_registry_lock_is_contended() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    control = CallControl(streaming_outcome="outcome_unknown")
    registry = _registry(writer, control, lambda: 100.0)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)
    writer.active_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await authority.claim_once(call_control_id="control-a", token_digest=DIGEST)

    class GuardedRegistryMap(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            assert registry._lock.locked() is True  # type: ignore[attr-defined]
            return super().get(key, default)

    registry._by_control = GuardedRegistryMap(registry._by_control)  # type: ignore[attr-defined]
    await registry._lock.acquire()  # type: ignore[attr-defined]
    authority.schedule_abort_if_matches(
        call_control_id="control-a",
        token_digest=DIGEST,
    )
    registry._lock.release()  # type: ignore[attr-defined]
    await registry.join_until_empty()

    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert authority.internal_failure_code is None


def test_public_abort_without_published_target_latches_fixed_failure() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    registry = _registry(Writer(), CallControl(), lambda: 100.0)
    authority = ProcessLeaseAuthority(registry)

    authority.schedule_abort_if_matches(
        call_control_id="control-a",
        token_digest=DIGEST,
    )

    assert authority.internal_failure_code == "abort_target_unavailable"


@pytest.mark.asyncio
async def test_public_abort_task_creation_failure_closes_and_latches() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)
    writer.active_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await authority.claim_once(call_control_id="control-a", token_digest=DIGEST)

    def fail_task_creation(*_: object, **__: object) -> asyncio.Task[None]:
        raise RuntimeError("synthetic task creation failure")

    registry._background_owner._task_factory = fail_task_creation  # type: ignore[attr-defined]
    authority.schedule_abort_if_matches(
        call_control_id="control-a",
        token_digest=DIGEST,
    )

    assert authority.internal_failure_code == "abort_target_unavailable"
    assert registry.internal_failure_code == "background_task_registration_failed"
    assert await registry.snapshot("control-a") is not None


@pytest.mark.asyncio
async def test_two_distinct_concurrent_claims_keep_independent_abort_handoffs() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    class BarrierWriter(Writer):
        def __init__(self) -> None:
            super().__init__()
            self.active_calls: set[str] = set()
            self.both_active_started = asyncio.Event()
            self.release_active = asyncio.Event()

        async def commit_lease(self, **values: object) -> None:
            if values["state"] == "active":
                self.active_calls.add(str(values["call_control_id"]))
                if self.active_calls == {"control-a", "control-b"}:
                    self.both_active_started.set()
                await self.release_active.wait()
            await super().commit_lease(**values)

    tokens = iter(("A" * 43, "B" * 43))
    writer = BarrierWriter()
    control = CallControl(streaming_outcome="outcome_unknown")
    registry = _registry(
        writer,
        control,
        lambda: 100.0,
        capacity=2,
        token_factory=lambda _: next(tokens),
    )
    await _durable_waiting_wss_for(
        registry,
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        event_prefix="event-a",
    )
    await _durable_waiting_wss_for(
        registry,
        call_control_id="control-b",
        call_leg_id="leg-b",
        call_session_id="session-b",
        event_prefix="event-b",
    )
    authority = ProcessLeaseAuthority(registry)
    claim_a = asyncio.create_task(
        authority.claim_once(call_control_id="control-a", token_digest=DIGEST)
    )
    claim_b = asyncio.create_task(
        authority.claim_once(call_control_id="control-b", token_digest=DIGEST_B)
    )
    await writer.both_active_started.wait()
    writer.release_active.set()

    assert await claim_a is not None
    assert await claim_b is not None
    authority.schedule_abort_if_matches(
        call_control_id="control-b",
        token_digest=DIGEST_B,
    )
    await registry.join_until_empty()

    snapshot_a = await registry.snapshot("control-a")
    assert snapshot_a is not None
    assert snapshot_a.lease_state == "active"
    assert await registry.snapshot("control-b") is None
    assert [call_control_id for call_control_id, _ in control.hangups] == ["control-b"]
    assert authority.internal_failure_code is None


@pytest.mark.asyncio
async def test_abort_target_publication_failure_rolls_back_before_active_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)

    def refuse_publication(
        _authority: ProcessLeaseAuthority,
        _target: object,
    ) -> bool:
        return False

    monkeypatch.setattr(
        ProcessLeaseAuthority,
        "_publish_abort_target",
        refuse_publication,
    )

    claim = await authority.claim_once(
        call_control_id="control-a",
        token_digest=DIGEST,
    )

    assert claim is None
    assert writer.commits == []
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.lease_state == "pending"


@pytest.mark.asyncio
async def test_repeated_public_abort_schedule_creates_at_most_one_task() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    writer = Writer()
    writer.active_error = asyncio.CancelledError()
    registry = _registry(writer, CallControl(), lambda: 100.0)
    await _durable_waiting_wss(registry)
    authority = ProcessLeaseAuthority(registry)
    with pytest.raises(asyncio.CancelledError):
        await authority.claim_once(call_control_id="control-a", token_digest=DIGEST)

    created_names: list[str] = []

    def counting_factory(
        coroutine: Any,
        name: str,
    ) -> asyncio.Task[None]:
        created_names.append(name)
        return asyncio.create_task(coroutine, name=name)

    registry._background_owner._task_factory = counting_factory  # type: ignore[attr-defined]
    authority.schedule_abort_if_matches(
        call_control_id="control-a",
        token_digest=DIGEST,
    )
    authority.schedule_abort_if_matches(
        call_control_id="control-a",
        token_digest=DIGEST,
    )
    await registry.join_until_empty()

    assert created_names == ["voice-matching-lease-abort"]
