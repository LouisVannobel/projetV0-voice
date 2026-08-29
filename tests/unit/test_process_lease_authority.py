from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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

    async def commit_lease(self, **values: object) -> None:
        if values["state"] == "active" and self.active_error is not None:
            raise self.active_error
        self.commits.append(values)


class CallControl:
    def __init__(self, streaming_outcome: str = "accepted") -> None:
        self.streaming_outcome = streaming_outcome
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
        return CallControlResult("accepted")


def _registry(
    writer: Writer,
    control: CallControl,
    clock: Any,
    utcnow: Any = lambda: NOW,
    background_task_factory: Any = None,
    capacity: int = 1,
    token_factory: Any = None,
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
        uuid_factory=lambda: next(ids),
        **kwargs,
    )


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
