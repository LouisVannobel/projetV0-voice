from __future__ import annotations

import asyncio
import dataclasses
import gc
import weakref
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import SecretStr

from projetv0_voice.telnyx.call_control import CallControlResult, StreamingStartV1
from projetv0_voice.telnyx.webhooks import VerifiedWebhook

NOW = datetime(2026, 8, 29, 10, tzinfo=UTC)


def _initiated(event_id: str = "event-a", control_id: str = "control-a") -> VerifiedWebhook:
    return VerifiedWebhook(
        event_id=event_id,
        event_type="call.initiated",
        occurred_at=NOW,
        call_control_id=control_id,
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=(event_id.encode() + b"_" * 32)[:32],
        direction="incoming",
        call_state="parked",
    )


def _answered(event_id: str = "answer-a", control_id: str = "control-a") -> VerifiedWebhook:
    return VerifiedWebhook(
        event_id=event_id,
        event_type="call.answered",
        occurred_at=NOW,
        call_control_id=control_id,
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=(event_id.encode() + b"_" * 32)[:32],
        call_state="answered",
    )


def _hangup(event_id: str = "hangup-a", control_id: str = "control-a") -> VerifiedWebhook:
    return VerifiedWebhook(
        event_id=event_id,
        event_type="call.hangup",
        occurred_at=NOW,
        call_control_id=control_id,
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=(event_id.encode() + b"_" * 32)[:32],
    )


class Writer:
    async def commit_lease(self, **_: object) -> None:
        return None


class BlockingTerminalWriter:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.commits: list[dict[str, object]] = []

    async def commit_lease(self, **values: object) -> None:
        self.commits.append(values)
        self.entered.set()
        await self.release.wait()


class CallControl:
    def __init__(
        self,
        *,
        answer_outcome: str = "accepted",
        streaming_outcomes: list[str] | None = None,
        streaming_error: bool = False,
    ) -> None:
        self.answers: list[tuple[str, UUID]] = []
        self.streams: list[tuple[str, UUID]] = []
        self.stream_tokens: list[str] = []
        self.hangups: list[tuple[str, UUID]] = []
        self.streaming_outcomes = streaming_outcomes or ["accepted"]
        self.streaming_error = streaming_error
        self.answer_outcome = answer_outcome

    async def answer(self, call_control_id: str, *, command_id: UUID) -> CallControlResult:
        self.answers.append((call_control_id, command_id))
        await asyncio.sleep(0)
        return CallControlResult(self.answer_outcome)  # type: ignore[arg-type]

    async def start_streaming(
        self, call_control_id: str, request: object, *, command_id: UUID
    ) -> CallControlResult:
        if self.streaming_error:
            raise RuntimeError("RAW-PROVIDER-SENTINEL")
        self.streams.append((call_control_id, command_id))
        self.stream_tokens.append(request.stream_auth_token.get_secret_value())  # type: ignore[attr-defined]
        await asyncio.sleep(0)
        return CallControlResult(self.streaming_outcomes.pop(0))  # type: ignore[arg-type]

    async def hangup(
        self, call_control_id: str, *, command_id: UUID, client_state: object = None
    ) -> CallControlResult:
        del client_state
        self.hangups.append((call_control_id, command_id))
        return CallControlResult("accepted")


def _registry(
    capacity: int = 2,
    *,
    monotonic: Any = lambda: 100.0,
    control: CallControl | None = None,
    candidate_run_id: UUID | None = None,
    admission_expires_at: datetime | None = None,
    utcnow: Any = lambda: NOW,
    background_task_factory: Any = None,
    writer: Any = None,
    token_factory: Any = None,
    prefix_factory: Any = lambda: 0,
) -> tuple[Any, CallControl]:
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
    control = control or CallControl()
    kwargs: dict[str, object] = {}
    if background_task_factory is not None:
        kwargs["background_task_factory"] = background_task_factory
    return (
        CallRegistry(
            writer=writer or Writer(),
            call_control=control,
            tenant_id="tenant-a",
            agent_id="agent-a",
            deployment_id="agent-a",
            capacity=capacity,
            lease_ttl_seconds=30,
            stream_url="wss://voice.invalid/telnyx/stream",
            retention_days=7,
            utcnow=utcnow,
            monotonic=monotonic,
            token_factory=token_factory or (lambda _: "A" * 43),
            prefix_factory=prefix_factory,
            uuid_factory=lambda: next(ids),
            candidate_run_id=candidate_run_id,
            admission_expires_at=admission_expires_at,
            **kwargs,
        ),
        control,
    )


def _coroutine_frames(task: asyncio.Task[Any]) -> list[Any]:
    frames: list[Any] = []
    current: Any = task.get_coro()
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frame = getattr(current, "cr_frame", None)
        if frame is not None:
            frames.append(frame)
        current = getattr(current, "cr_await", None)
    return frames


def _run_action_frame(task: asyncio.Task[Any]) -> Any:
    frames = [
        frame
        for frame in _coroutine_frames(task)
        if frame.f_code.co_name == "_run_action"
    ]
    assert len(frames) == 1
    return frames[0]


def _frame_owns_token_identity(frame: Any, token: str) -> bool:
    for value in frame.f_locals.values():
        if value is token:
            return True
        if isinstance(value, SecretStr) and value.get_secret_value() is token:
            return True
        if (
            isinstance(value, StreamingStartV1)
            and value.stream_auth_token.get_secret_value() is token
        ):
            return True
    return False


def _task_owns_token_identity(task: asyncio.Task[Any], token: str) -> bool:
    return any(_frame_owns_token_identity(frame, token) for frame in _coroutine_frames(task))


async def _wait_for_reservation_settlement(reservation: Any) -> None:
    for _ in range(20):
        if reservation._registry_applied:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0)
    assert reservation._registry_applied


@pytest.mark.asyncio
@pytest.mark.parametrize("deliveries", [2, 50])
async def test_simultaneous_initiated_deliveries_own_one_generation_and_answer(
    deliveries: int,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry()
    resolutions = await asyncio.gather(
        *(registry.resolve_webhook(_initiated()) for _ in range(deliveries))
    )

    effects = [resolution.effect for resolution in resolutions]
    assert all(effect is not None for effect in effects)
    leases = [effect.lease for effect in effects if effect is not None]
    assert {lease["call_id"] for lease in leases if lease is not None} == {
        UUID("00000000-0000-4000-8000-000000000000")
    }
    assert {lease["token_hash"] for lease in leases if lease is not None} == {
        bytes.fromhex("0f007385b6f9d4b7eeb2748605afe1a984a0a3bfa3f014d09e2a784ce9e5cd1a")
    }
    results = [WebhookCommitResult("first", "applied")]
    results.extend(
        WebhookCommitResult("duplicate", "duplicate") for _ in range(deliveries - 1)
    )

    dispositions = await asyncio.gather(
        *(
            registry.reconcile_after_commit(_initiated(), resolution, result)
            for resolution, result in zip(resolutions, results, strict=True)
        )
    )
    snapshot = await registry.snapshot("control-a")

    assert [item.status_code for item in dispositions] == [200] * deliveries
    assert snapshot is not None
    assert snapshot.durable is True
    assert snapshot.precommit_refcount == 0
    assert snapshot.answer_state == "accepted"
    assert snapshot.raw_token_retained is True
    assert await registry.live_call_count() == 1
    assert control.answers == [
        ("control-a", UUID("11111111-1111-4111-8111-111111111111"))
    ]
    assert "A" * 43 not in repr(snapshot)
    assert "A" * 43 not in repr(resolutions)


@pytest.mark.asyncio
async def test_strict_capacity_rejects_before_creating_a_reservation() -> None:
    from projetv0_voice.admission import CallAdmissionRejected

    registry, _ = _registry(capacity=2)
    await registry.resolve_webhook(_initiated("event-a", "control-a"))
    await registry.resolve_webhook(_initiated("event-b", "control-b"))

    with pytest.raises(CallAdmissionRejected, match="call_capacity_reached"):
        await registry.resolve_webhook(_initiated("event-c", "control-c"))

    assert await registry.live_call_count() == 2


@pytest.mark.asyncio
async def test_early_answered_placeholder_is_capped_nonextending_and_consumed() -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.persistence.writer import WebhookCommitResult

    now = 100.0
    registry, control = _registry(capacity=1, monotonic=lambda: now)
    first = await registry.resolve_webhook(_answered())
    assert first.effect is None
    assert first.reservation is not None
    assert (
        await registry.reconcile_after_commit(
            _answered(), first, WebhookCommitResult("first", "applied")
        )
    ).status_code == 200
    assert await registry.placeholder_count() == 1
    assert await registry.placeholder_deadline("control-a") == 130.0

    now = 120.0
    duplicate = await registry.resolve_webhook(_answered())
    await registry.reconcile_after_commit(
        _answered(), duplicate, WebhookCommitResult("duplicate", "duplicate")
    )
    assert await registry.placeholder_count() == 1
    assert await registry.placeholder_deadline("control-a") == 130.0
    with pytest.raises(CallAdmissionRejected, match="placeholder_capacity_reached"):
        await registry.resolve_webhook(_answered("answer-b", "control-b"))

    initiated = await registry.resolve_webhook(_initiated())
    disposition = await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )

    assert disposition.status_code == 200
    assert await registry.placeholder_count() == 0
    assert control.answers == []
    assert control.streams == [
        ("control-a", UUID("22222222-2222-4222-8222-222222222222"))
    ]
    assert control.hangups == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("now", "expected_answers", "expected_streams"),
    [(129.999, 0, 1), (130.0, 1, 0), (130.001, 1, 0)],
)
async def test_durable_placeholder_transfer_obeys_own_deadline(
    now: float,
    expected_answers: int,
    expected_streams: int,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(monotonic=lambda: clock[0])
    answered = _answered()
    answer_resolution = await registry.resolve_webhook(answered)
    await registry.reconcile_after_commit(
        answered,
        answer_resolution,
        WebhookCommitResult("first", "applied"),
    )

    clock[0] = now
    initiated = _initiated()
    initiated_resolution = await registry.resolve_webhook(initiated)
    disposition = await registry.reconcile_after_commit(
        initiated,
        initiated_resolution,
        WebhookCommitResult("first", "applied"),
    )

    assert disposition.status_code == 200
    assert len(control.answers) == expected_answers
    assert len(control.streams) == expected_streams
    assert await registry.placeholder_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("now", "expected_streams"),
    [(129.999, 1), (130.0, 0), (130.001, 0)],
)
async def test_linked_placeholder_commit_obeys_placeholder_and_entry_deadlines(
    now: float,
    expected_streams: int,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(monotonic=lambda: clock[0])
    answered = _answered()
    answer_resolution = await registry.resolve_webhook(answered)

    clock[0] = 120.0
    initiated = _initiated()
    initiated_resolution = await registry.resolve_webhook(initiated)
    initiated_disposition = await registry.reconcile_after_commit(
        initiated,
        initiated_resolution,
        WebhookCommitResult("first", "applied"),
    )

    clock[0] = now
    answer_disposition = await registry.reconcile_after_commit(
        answered,
        answer_resolution,
        WebhookCommitResult("first", "applied"),
    )

    assert initiated_disposition.status_code == 200
    assert answer_disposition.status_code == 200
    assert len(control.answers) == 1
    assert len(control.streams) == expected_streams
    assert await registry.placeholder_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("now", [130.0, 130.001])
async def test_unlinked_placeholder_commit_at_or_after_deadline_is_removed_without_future_evidence(
    now: float,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(monotonic=lambda: clock[0])
    answered = _answered()
    answer_resolution = await registry.resolve_webhook(answered)

    clock[0] = now
    answer_disposition = await registry.reconcile_after_commit(
        answered,
        answer_resolution,
        WebhookCommitResult("first", "applied"),
    )

    assert answer_disposition.status_code == 200
    assert await registry.placeholder_count() == 0

    initiated = _initiated()
    initiated_resolution = await registry.resolve_webhook(initiated)
    initiated_disposition = await registry.reconcile_after_commit(
        initiated,
        initiated_resolution,
        WebhookCommitResult("first", "applied"),
    )

    assert initiated_disposition.status_code == 200
    assert len(control.answers) == 1
    assert control.streams == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reap_first", [True, False])
async def test_placeholder_deadline_result_is_independent_of_reaper_order(
    reap_first: bool,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(monotonic=lambda: clock[0])
    answered = _answered()
    answer_resolution = await registry.resolve_webhook(answered)
    await registry.reconcile_after_commit(
        answered,
        answer_resolution,
        WebhookCommitResult("first", "applied"),
    )
    clock[0] = 130.0

    reaped = await registry.reap_expired() if reap_first else 0
    initiated = _initiated()
    initiated_resolution = await registry.resolve_webhook(initiated)
    disposition = await registry.reconcile_after_commit(
        initiated,
        initiated_resolution,
        WebhookCommitResult("first", "applied"),
    )

    assert reaped == (1 if reap_first else 0)
    assert disposition.status_code == 200
    assert len(control.answers) == 1
    assert control.streams == []
    assert await registry.placeholder_count() == 0


@pytest.mark.asyncio
async def test_streaming_retry_reuses_original_uuid_and_token() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    control = CallControl(streaming_outcomes=["rate_limited", "accepted"])
    registry, _ = _registry(capacity=1, control=control)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())
    first = await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    retry = await registry.resolve_webhook(_answered())
    second = await registry.reconcile_after_commit(
        _answered(), retry, WebhookCommitResult("duplicate", "duplicate")
    )

    assert (first.status_code, second.status_code) == (503, 200)
    assert control.streams == [
        ("control-a", UUID("22222222-2222-4222-8222-222222222222")),
        ("control-a", UUID("22222222-2222-4222-8222-222222222222")),
    ]
    assert control.stream_tokens == ["A" * 43, "A" * 43]
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.streaming_state == "accepted"
    assert snapshot.raw_token_retained is False


@pytest.mark.asyncio
async def test_action_start_at_token_deadline_terminalizes_without_provider_call() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    authority_reached = asyncio.Event()
    provider_entered = asyncio.Event()
    writer = BlockingTerminalWriter()

    class ForbiddenAnswer(CallControl):
        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            provider_entered.set()
            authority_reached.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    original_commit = writer.commit_lease

    async def signal_commit(**values: object) -> None:
        authority_reached.set()
        await original_commit(**values)

    writer.commit_lease = signal_commit  # type: ignore[method-assign]
    clock = [100.0]
    registry, control = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=ForbiddenAnswer(),
        writer=writer,
    )
    initiated = await registry.resolve_webhook(_initiated())
    clock[0] = 130.0
    action = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
    )

    await authority_reached.wait()
    took_deadline_path = writer.entered.is_set()
    if not took_deadline_path:
        action.cancel()
        await asyncio.gather(action, return_exceptions=True)
    assert took_deadline_path
    assert provider_entered.is_set() is False
    assert control.answers == []
    assert control.streams == []
    snapshot = await asyncio.wait_for(registry.snapshot("control-a"), timeout=1.0)
    assert snapshot is not None
    assert snapshot.lease_state == "terminal"
    assert snapshot.raw_token_retained is False

    writer.release.set()
    disposition = await action

    assert disposition.status_code == 200
    assert len(writer.commits) == 1
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_streaming_retry_at_token_deadline_does_not_call_provider_again() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    control = CallControl(streaming_outcomes=["rate_limited", "accepted"])
    registry, _ = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=control,
    )
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())
    first = await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    original_stream = control.streams[0]
    original_token = control.stream_tokens[0]

    clock[0] = 130.0
    duplicate = await registry.resolve_webhook(_initiated())
    retry = await registry.reconcile_after_commit(
        _initiated(), duplicate, WebhookCommitResult("duplicate", "duplicate")
    )

    assert (first.status_code, retry.status_code) == (503, 200)
    assert control.streams == [original_stream]
    assert control.stream_tokens == [original_token]
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["answer", "streaming"])
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
async def test_inflight_action_result_at_token_deadline_terminalizes_before_publication(
    action: str,
    late_outcome: str,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    writer = BlockingTerminalWriter()

    class LateControl(CallControl):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def _late(self) -> CallControlResult:
            self.entered.set()
            await self.release.wait()
            if late_outcome == "exception":
                raise RuntimeError("synthetic late provider failure")
            return CallControlResult(late_outcome)  # type: ignore[arg-type]

        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            if action == "answer":
                return await self._late()
            return CallControlResult("accepted")

        async def start_streaming(
            self, call_control_id: str, request: object, *, command_id: UUID
        ) -> CallControlResult:
            self.streams.append((call_control_id, command_id))
            self.stream_tokens.append(  # type: ignore[attr-defined]
                request.stream_auth_token.get_secret_value()
            )
            if action == "streaming":
                return await self._late()
            return CallControlResult("accepted")

    control = LateControl()
    registry, _ = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=control,
        writer=writer,
    )
    initiated = await registry.resolve_webhook(_initiated())
    if action == "answer":
        owner = asyncio.create_task(
            registry.reconcile_after_commit(
                _initiated(), initiated, WebhookCommitResult("first", "applied")
            )
        )
    else:
        await registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
        answered = await registry.resolve_webhook(_answered())
        owner = asyncio.create_task(
            registry.reconcile_after_commit(
                _answered(), answered, WebhookCommitResult("first", "applied")
            )
        )
    await control.entered.wait()
    async with registry._lock:  # type: ignore[attr-defined]
        entry = registry._by_control["control-a"]  # type: ignore[attr-defined]
        slot = entry.answer if action == "answer" else entry.streaming
        assert slot.completion is not None
        joiner = asyncio.ensure_future(asyncio.shield(slot.completion))

    clock[0] = 130.0
    control.release.set()
    writer_wait = asyncio.create_task(writer.entered.wait())
    done, _ = await asyncio.wait(
        {owner, writer_wait}, return_when=asyncio.FIRST_COMPLETED
    )
    took_deadline_path = writer_wait in done
    if not took_deadline_path:
        writer_wait.cancel()
        await asyncio.gather(owner, joiner, writer_wait, return_exceptions=True)
    assert took_deadline_path

    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.lease_state == "terminal"
    assert snapshot.raw_token_retained is False
    assert (
        snapshot.answer_state if action == "answer" else snapshot.streaming_state
    ) == "unknown"
    assert joiner.done() is True
    assert (await joiner).status_code == 200

    writer.release.set()
    owner_result = await owner

    assert owner_result.status_code == 200
    assert len(writer.commits) == 1
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_cancelled_inflight_action_at_token_deadline_terminalizes_then_reraises() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    writer = BlockingTerminalWriter()

    class BlockingAnswer(CallControl):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()

        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    control = BlockingAnswer()
    registry, _ = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=control,
        writer=writer,
    )
    initiated = await registry.resolve_webhook(_initiated())
    owner = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
    )
    await control.entered.wait()
    async with registry._lock:  # type: ignore[attr-defined]
        entry = registry._by_control["control-a"]  # type: ignore[attr-defined]
        assert entry.answer.completion is not None
        joiner = asyncio.ensure_future(asyncio.shield(entry.answer.completion))

    clock[0] = 130.0
    owner.cancel()
    writer_wait = asyncio.create_task(writer.entered.wait())
    done, _ = await asyncio.wait(
        {owner, writer_wait}, return_when=asyncio.FIRST_COMPLETED
    )
    took_deadline_path = writer_wait in done
    if not took_deadline_path:
        writer_wait.cancel()
        await asyncio.gather(owner, joiner, writer_wait, return_exceptions=True)
    assert took_deadline_path

    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.lease_state == "terminal"
    assert snapshot.answer_state == "unknown"
    assert snapshot.raw_token_retained is False
    assert joiner.done() is True
    assert (await joiner).status_code == 200

    writer.release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner

    assert len(writer.commits) == 1
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repeated_cancel_phase",
    ["terminal_persistence", "cleanup_hangup", "exact_removal"],
)
async def test_cancelled_deadline_cleanup_registration_failure_finishes_before_reraise(
    monkeypatch: pytest.MonkeyPatch,
    repeated_cancel_phase: str,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    real_create_task = asyncio.create_task

    class CompletionWriter:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.completed = asyncio.Event()
            self.commits: list[dict[str, object]] = []

        async def commit_lease(self, **values: object) -> None:
            self.commits.append(values)
            self.entered.set()
            await self.release.wait()
            self.completed.set()

    writer = CompletionWriter()

    def fail_cleanup_registration(
        coroutine: Any, name: str
    ) -> asyncio.Task[None]:
        if name == "voice-action-deadline-cleanup":
            raise RuntimeError("synthetic cleanup registration failure")
        return real_create_task(coroutine, name=name)

    def reject_current_cleanup_registration(
        coroutine: Any,
        *,
        name: str | None = None,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        if name == "voice-action-deadline-cleanup":
            raise RuntimeError("synthetic cleanup registration failure")
        return real_create_task(coroutine, name=name, context=context)

    monkeypatch.setattr(asyncio, "create_task", reject_current_cleanup_registration)

    class BlockingAnswer(CallControl):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.hangup_entered = asyncio.Event()
            self.hangup_release = asyncio.Event()
            self.hangup_completed = asyncio.Event()

        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def hangup(
            self,
            call_control_id: str,
            *,
            command_id: UUID,
            client_state: object = None,
        ) -> CallControlResult:
            del client_state
            self.hangups.append((call_control_id, command_id))
            self.hangup_entered.set()
            await self.hangup_release.wait()
            self.hangup_completed.set()
            return CallControlResult("accepted")

    control = BlockingAnswer()
    registry, _ = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=control,
        writer=writer,
        background_task_factory=fail_cleanup_registration,
    )
    cleanup_coroutines: list[Any] = []
    run_terminal_cleanup = registry._run_terminal_cleanup  # type: ignore[attr-defined]

    def track_terminal_cleanup(work: Any, **kwargs: object) -> Any:
        coroutine = run_terminal_cleanup(work, **kwargs)
        cleanup_coroutines.append(coroutine)
        return coroutine

    registry._run_terminal_cleanup = track_terminal_cleanup  # type: ignore[attr-defined,method-assign]

    if repeated_cancel_phase == "cleanup_hangup":
        writer.release.set()
    else:
        control.hangup_release.set()

    registry_lock_held = False
    try:
        initiated = await registry.resolve_webhook(_initiated())
        owner = asyncio.create_task(
            registry.reconcile_after_commit(
                _initiated(), initiated, WebhookCommitResult("first", "applied")
            )
        )
        await control.entered.wait()
        async with registry._lock:  # type: ignore[attr-defined]
            entry = registry._by_control["control-a"]  # type: ignore[attr-defined]
            assert entry.answer.completion is not None
            joiner = asyncio.ensure_future(asyncio.shield(entry.answer.completion))

        clock[0] = 130.0
        owner.cancel()
        writer_wait = asyncio.create_task(writer.entered.wait())
        done, _ = await asyncio.wait(
            {owner, writer_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        cleanup_started = writer_wait in done
        terminal_snapshot = await registry.snapshot("control-a")
        if cleanup_started:
            if repeated_cancel_phase == "terminal_persistence":
                owner.cancel()
                await asyncio.sleep(0)
                writer.release.set()
            elif repeated_cancel_phase == "cleanup_hangup":
                await control.hangup_entered.wait()
                owner.cancel()
                await asyncio.sleep(0)
                control.hangup_release.set()
            else:
                await registry._lock.acquire()  # type: ignore[attr-defined]
                registry_lock_held = True
                writer.release.set()
                await control.hangup_completed.wait()
                await asyncio.sleep(0)
                owner.cancel()
                await asyncio.sleep(0)
                registry._lock.release()  # type: ignore[attr-defined]
                registry_lock_held = False
        else:
            writer_wait.cancel()
        owner_result = (await asyncio.gather(owner, return_exceptions=True))[0]
        await asyncio.gather(writer_wait, return_exceptions=True)
        joiner_status = (await joiner).status_code
        final_snapshot = await registry.snapshot("control-a")
        live_count = await registry.live_call_count()

        observed = {
            "cleanup_started": cleanup_started,
            "internal_failure": registry.internal_failure_code,
            "waiter_status": joiner_status,
            "raw_token_erased_before_cleanup": (
                terminal_snapshot is not None
                and terminal_snapshot.raw_token_retained is False
            ),
            "terminal_persistence_completed": writer.completed.is_set(),
            "terminal_persistence_same_work": bool(writer.commits)
            and len(writer.commits) <= 2
            and all(commit == writer.commits[0] for commit in writer.commits),
            "cleanup_hangup_completed": control.hangup_completed.is_set(),
            "cleanup_hangup_same_command": bool(control.hangups)
            and len(control.hangups) <= 2
            and all(hangup == control.hangups[0] for hangup in control.hangups),
            "generation_absent": final_snapshot is None,
            "live_count": live_count,
            "cleanup_coroutines_closed": bool(cleanup_coroutines)
            and all(coroutine.cr_frame is None for coroutine in cleanup_coroutines),
            "owner_cancelled": isinstance(owner_result, asyncio.CancelledError),
        }
        assert observed == {
            "cleanup_started": True,
            "internal_failure": "background_task_registration_failed",
            "waiter_status": 200,
            "raw_token_erased_before_cleanup": True,
            "terminal_persistence_completed": True,
            "terminal_persistence_same_work": True,
            "cleanup_hangup_completed": True,
            "cleanup_hangup_same_command": True,
            "generation_absent": True,
            "live_count": 0,
            "cleanup_coroutines_closed": True,
            "owner_cancelled": True,
        }
    finally:
        writer.release.set()
        control.hangup_release.set()
        if registry_lock_held:
            registry._lock.release()  # type: ignore[attr-defined]
        for coroutine in cleanup_coroutines:
            coroutine.close()


@pytest.mark.asyncio
async def test_streaming_exception_is_unknown_erases_token_and_returns_500() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    control = CallControl(streaming_error=True)
    registry, _ = _registry(capacity=1, control=control)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())

    result = await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )

    snapshot = await registry.snapshot("control-a")
    assert result.status_code == 500
    assert snapshot is not None
    assert snapshot.streaming_state == "unknown"
    assert snapshot.raw_token_retained is False


@pytest.mark.asyncio
async def test_streaming_cancellation_publishes_unknown_then_remains_cancelled() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    entered = asyncio.Event()

    class BlockingControl(CallControl):
        async def start_streaming(
            self, call_control_id: str, request: object, *, command_id: UUID
        ) -> CallControlResult:
            del call_control_id, request, command_id
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    registry, _ = _registry(capacity=1, control=BlockingControl())
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())
    action = asyncio.create_task(
        registry.reconcile_after_commit(
            _answered(), answered, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()

    action.cancel()
    with pytest.raises(asyncio.CancelledError):
        await action
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.streaming_state == "unknown"
    assert snapshot.raw_token_retained is False


@pytest.mark.asyncio
async def test_rejected_answer_terminalizes_without_positive_evidence() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    control = CallControl(answer_outcome="rejected")
    registry, _ = _registry(capacity=1, control=control)
    initiated = await registry.resolve_webhook(_initiated())

    result = await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )

    assert result.status_code == 200
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert len(control.hangups) == 1


@pytest.mark.asyncio
async def test_hangup_carries_terminal_effect_and_releases_without_orphan_hangup() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    hangup = await registry.resolve_webhook(_hangup())

    assert hangup.effect is not None
    assert hangup.effect.lease is not None
    assert hangup.effect.lease["state"] == "terminal"
    assert hangup.effect.operation is not None
    assert hangup.effect.operation.payload.status == "failed"
    disposition = await registry.reconcile_after_commit(
        _hangup(), hangup, WebhookCommitResult("first", "applied")
    )

    assert disposition.status_code == 200
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert control.hangups == []


def test_capacity_selection_is_exact_for_strict_candidate_and_override() -> None:
    from projetv0_voice.admission import select_call_capacity
    from projetv0_voice.config import AgentManifestV1
    from projetv0_voice.qualified_profile import (
        QualificationCandidateProfileV1,
        QualificationOverrideV1,
        QualifiedDeploymentProfileV1,
    )

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    strict = QualifiedDeploymentProfileV1.model_validate_json(
        (fixtures / "qualified-deployment-profile-v1.json").read_text(encoding="utf-8")
    )
    candidate = QualificationCandidateProfileV1.model_validate_json(
        (fixtures / "qualification-candidate-v1.json").read_text(encoding="utf-8")
    )
    override = QualificationOverrideV1.model_validate_json(
        (fixtures / "qualification-override-v1.json").read_text(encoding="utf-8")
    )
    manifest_data = {
        "schema_version": 1,
        "tenant_id": "tenant-a",
        "agent_id": "agent-a",
        "revision": "r1",
        "dids": ["+33102030405"],
        "language": "fr-FR",
        "prompt_path": "prompt.md",
        "prompt_revision": "p1",
        "greeting": "Bonjour",
        "conversation_mode": "freeform",
        "max_concurrent_calls": 10,
        "direction": "inbound_only",
        "transport_codec": "PCMU",
        "transport_sample_rate_hz": 8000,
        "transcript_retention_days": 7,
        "recording_mode": "off",
        "recording_format": "wav",
        "recording_retention_days": None,
        "recording_required": False,
        "recording_play_beep": False,
    }
    manifest = AgentManifestV1.model_validate(manifest_data)

    assert (
        select_call_capacity(
            profile=strict,
            manifest=manifest,
            deployment_max_calls=10,
        )
        == 10
    )
    assert (
        select_call_capacity(
            profile=candidate,
            manifest=manifest,
            deployment_max_calls=20,
        )
        == 1
    )
    assert (
        select_call_capacity(
            profile=strict,
            manifest=manifest,
            deployment_max_calls=15,
            override=override,
        )
        == 15
    )
    for profile, host, selected_override in (
        (strict, 9, None),
        (candidate, 0, None),
        (strict, 20, override),
    ):
        with pytest.raises(ValueError, match="capacity"):
            select_call_capacity(
                profile=profile,
                manifest=manifest,
                deployment_max_calls=host,
                override=selected_override,
            )


@pytest.mark.asyncio
async def test_candidate_first_commit_closes_distinct_admission_but_keeps_duplicate() -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.persistence.writer import WebhookCommitResult

    run_id = UUID("99999999-9999-4999-8999-999999999999")
    registry, _ = _registry(capacity=2, candidate_run_id=run_id)
    first = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), first, WebhookCommitResult("first", "applied")
    )

    duplicate = await registry.resolve_webhook(_initiated())
    assert duplicate.reservation is not None
    with pytest.raises(CallAdmissionRejected, match="qualification_run_consumed"):
        await registry.resolve_webhook(_initiated("event-b", "control-b"))
    assert registry.candidate_run_id == run_id


@pytest.mark.asyncio
async def test_live_qualification_expiry_closes_new_admission() -> None:
    from projetv0_voice.admission import CallAdmissionRejected

    now = NOW
    registry, _ = _registry(
        capacity=1,
        admission_expires_at=NOW + timedelta(seconds=1),
        utcnow=lambda: now,
    )
    now = NOW + timedelta(seconds=1)

    with pytest.raises(CallAdmissionRejected, match="qualification_window_expired"):
        await registry.resolve_webhook(_initiated())


@pytest.mark.asyncio
async def test_cancelled_confirmation_remains_retryable_without_reference_leak() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, _ = _registry(capacity=1)
    resolution = await registry.resolve_webhook(_initiated())
    assert resolution.reservation is not None
    await registry._lock.acquire()  # type: ignore[attr-defined]
    confirmation = asyncio.create_task(
        resolution.reservation.confirm(  # type: ignore[attr-defined]
            WebhookCommitResult("first", "applied")
        )
    )
    await asyncio.sleep(0)
    confirmation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await confirmation
    registry._lock.release()  # type: ignore[attr-defined]

    await resolution.reservation.confirm(  # type: ignore[attr-defined]
        WebhookCommitResult("first", "applied")
    )
    snapshot = await registry.snapshot("control-a")

    assert snapshot is not None
    assert snapshot.durable is True
    assert snapshot.precommit_refcount == 0


@pytest.mark.asyncio
async def test_reservation_owner_spawn_failure_releases_generation_synchronously() -> None:
    from projetv0_voice.admission import CallAdmissionRejected

    def fail_spawn(*_: object, **__: object) -> asyncio.Task[None]:
        raise RuntimeError("synthetic spawn failure")

    registry, _ = _registry(
        capacity=1,
        background_task_factory=fail_spawn,
    )

    with pytest.raises(CallAdmissionRejected, match="owner_registration_failed"):
        await registry.resolve_webhook(_initiated())

    assert await registry.live_call_count() == 0
    assert await registry.snapshot("control-a") is None
    assert registry.internal_failure_code == "background_task_registration_failed"


@pytest.mark.asyncio
async def test_uncommitted_early_answer_never_starts_evidence_streaming() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    answered = await registry.resolve_webhook(_answered())
    initiated = await registry.resolve_webhook(_initiated())

    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    assert control.streams == []
    assert await registry.placeholder_count() == 1

    await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    assert len(control.streams) == 1
    assert await registry.placeholder_count() == 0


@pytest.mark.asyncio
async def test_early_answer_abandon_after_initiated_never_promotes_evidence() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    answered = await registry.resolve_webhook(_answered())
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )

    assert answered.reservation is not None
    answered.reservation.abandon_before_submit()
    await registry.join_until_empty()

    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.answer_state == "accepted"
    assert snapshot.streaming_state == "idle"
    assert control.streams == []


@pytest.mark.asyncio
async def test_cancelled_early_answer_confirmation_retries_before_streaming() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    answered = await registry.resolve_webhook(_answered())
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    assert answered.reservation is not None
    await registry._lock.acquire()  # type: ignore[attr-defined]
    confirmation = asyncio.create_task(
        answered.reservation.confirm(  # type: ignore[attr-defined]
            WebhookCommitResult("first", "applied")
        )
    )
    await asyncio.sleep(0)
    confirmation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await confirmation
    registry._lock.release()  # type: ignore[attr-defined]
    assert control.streams == []

    disposition = await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )

    assert disposition.status_code == 200
    assert len(control.streams) == 1
    assert await registry.placeholder_count() == 0


@pytest.mark.asyncio
async def test_terminal_while_answer_inflight_completes_duplicate_joiner() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingAnswer(CallControl):
        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            entered.set()
            await release.wait()
            return CallControlResult("accepted")

    registry, _ = _registry(capacity=1, control=BlockingAnswer())
    first = await registry.resolve_webhook(_initiated())
    owner = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), first, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()
    duplicate = await registry.resolve_webhook(_initiated())
    joiner = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), duplicate, WebhookCommitResult("duplicate", "duplicate")
        )
    )
    await asyncio.sleep(0)
    assert joiner.done() is False

    hangup = await registry.resolve_webhook(_hangup())
    await registry.reconcile_after_commit(
        _hangup(), hangup, WebhookCommitResult("first", "applied")
    )

    assert joiner.done() is True
    assert (await joiner).status_code == 200
    release.set()
    await asyncio.gather(owner, return_exceptions=True)
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


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
async def test_signed_answer_evidence_wins_every_late_provider_outcome(
    late_outcome: str,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    entered = asyncio.Event()
    release = asyncio.Event()

    class LateAnswer(CallControl):
        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            entered.set()
            await release.wait()
            if late_outcome == "exception":
                raise RuntimeError("synthetic late provider failure")
            return CallControlResult(late_outcome)  # type: ignore[arg-type]

    clock = [100.0]
    registry, control = _registry(
        capacity=1,
        control=LateAnswer(),
        monotonic=lambda: clock[0],
    )
    initiated = await registry.resolve_webhook(_initiated())
    action_owner = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()
    clock[0] = 129.999
    answered = await registry.resolve_webhook(_answered())
    evidence = await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    clock[0] = 130.0
    release.set()
    late = await action_owner
    snapshot = await registry.snapshot("control-a")

    assert (evidence.status_code, late.status_code) == (200, 200)
    assert snapshot is not None
    assert snapshot.answer_state == "accepted"
    assert await registry.live_call_count() == 1
    assert control.hangups == []


@pytest.mark.asyncio
async def test_stale_generation_cleanup_cannot_remove_replacement() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, _ = _registry(capacity=1)
    first = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), first, WebhookCommitResult("first", "applied")
    )
    stale = await registry.generation_handle("control-a")
    assert stale is not None
    hangup = await registry.resolve_webhook(_hangup())
    await registry.reconcile_after_commit(
        _hangup(), hangup, WebhookCommitResult("first", "applied")
    )
    replacement = await registry.resolve_webhook(
        _initiated("event-new", "control-a")
    )
    replacement_handle = await registry.generation_handle("control-a")

    assert replacement.reservation is not None
    assert replacement_handle is not None
    assert replacement_handle != stale
    assert await registry.complete_terminal_cleanup(stale) is False
    assert await registry.snapshot("control-a") is not None
    replacement.reservation.abandon_before_submit()
    await registry.join_until_empty()


@pytest.mark.asyncio
async def test_fail_closed_late_commit_confirms_then_aborts_without_answer() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    initiated = await registry.resolve_webhook(_initiated())

    confirmation = await registry.confirm_late_after_fail_closed(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    await registry.join_until_empty()

    assert confirmation.abort_scheduled is True
    assert control.answers == []
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_owner_close_registration_joins_existing_and_rejects_new_without_leak() -> None:
    from projetv0_voice.admission import CallAdmissionRejected

    registry, _ = _registry(capacity=2)
    existing = await registry.resolve_webhook(_initiated())
    registry.close_registration()
    assert existing.reservation is not None
    existing.reservation.abandon_before_submit()
    await registry.join_until_empty()

    with pytest.raises(CallAdmissionRejected, match="owner_registration_failed"):
        await registry.resolve_webhook(_initiated("event-b", "control-b"))

    assert await registry.live_call_count() == 0
    assert registry.internal_failure_code == "background_task_registration_failed"


@pytest.mark.asyncio
async def test_eager_task_factory_cannot_run_abandonment_before_reservation_transfer() -> None:
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        registry, _ = _registry(capacity=1)
        resolution = await registry.resolve_webhook(_initiated())
        snapshot = await registry.snapshot("control-a")

        assert resolution.reservation is not None
        assert snapshot is not None
        assert snapshot.precommit_refcount == 1
        resolution.reservation.abandon_before_submit()
        await registry.join_until_empty()
        assert await registry.snapshot("control-a") is None
        assert await registry.live_call_count() == 0
    finally:
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
async def test_reaper_and_answer_evidence_at_deadline_cannot_resurrect_generation() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    entered = asyncio.Event()

    class BlockingAnswer(CallControl):
        async def answer(self, *_: object, **__: object) -> CallControlResult:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    now = 100.0
    registry, control = _registry(
        capacity=1,
        monotonic=lambda: now,
        control=BlockingAnswer(),
    )
    initiated = await registry.resolve_webhook(_initiated())
    answer_owner = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()
    answered = await registry.resolve_webhook(_answered())
    now = 130.0

    evidence, reaped = await asyncio.gather(
        registry.reconcile_after_commit(
            _answered(), answered, WebhookCommitResult("first", "applied")
        ),
        registry.reap_expired(),
    )

    assert evidence.status_code == 200
    assert reaped == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert control.streams == []
    await asyncio.gather(answer_owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_linked_placeholder_late_fail_closed_commit_schedules_exact_abort() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    answered = await registry.resolve_webhook(_answered())
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    linked = await registry.generation_handle("control-a")

    confirmation = await registry.confirm_late_after_fail_closed(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    await registry.join_until_empty()

    assert linked is not None
    assert confirmation.generation == linked
    assert confirmation.abort_scheduled is True
    assert control.streams == []
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_expired_linked_placeholder_late_fail_closed_commit_keeps_exact_abort() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(capacity=1, monotonic=lambda: clock[0])
    answered = await registry.resolve_webhook(_answered())
    clock[0] = 120.0
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    linked = await registry.generation_handle("control-a")

    clock[0] = 130.0
    confirmation = await registry.confirm_late_after_fail_closed(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    await registry.join_until_empty()

    assert linked is not None
    assert confirmation.generation == linked
    assert confirmation.abort_scheduled is True
    assert control.streams == []
    assert len(control.hangups) == 1
    assert await registry.placeholder_count() == 0
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_unlinked_placeholder_late_fail_closed_commit_stays_generationless() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    answered = await registry.resolve_webhook(_answered())

    confirmation = await registry.confirm_late_after_fail_closed(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    await registry.join_until_empty()

    assert confirmation.generation is None
    assert confirmation.abort_scheduled is False
    assert control.hangups == []
    assert await registry.placeholder_count() == 0


@pytest.mark.asyncio
async def test_positive_answer_evidence_survives_provider_task_cancellation() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    entered = asyncio.Event()

    class BlockingAnswer(CallControl):
        async def answer(self, *_: object, **__: object) -> CallControlResult:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    registry, _ = _registry(capacity=1, control=BlockingAnswer())
    initiated = await registry.resolve_webhook(_initiated())
    provider = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()
    answered = await registry.resolve_webhook(_answered())
    await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )

    provider.cancel()
    with pytest.raises(asyncio.CancelledError):
        await provider
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    assert snapshot.answer_state == "accepted"


@pytest.mark.asyncio
async def test_cancelled_abandonment_waiting_registry_lock_still_releases() -> None:
    registry, _ = _registry(capacity=1)
    resolution = await registry.resolve_webhook(_initiated())
    assert resolution.reservation is not None
    await registry._lock.acquire()  # type: ignore[attr-defined]
    resolution.reservation.abandon_before_submit()
    await asyncio.sleep(0)
    owner_tasks = tuple(registry._background_owner._tasks)  # type: ignore[attr-defined]
    assert len(owner_tasks) == 1
    owner_tasks[0].cancel()
    await asyncio.sleep(0)
    registry._lock.release()  # type: ignore[attr-defined]

    await registry.join_until_empty()

    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
async def test_post_submit_settlement_joins_abandonment_before_reraising_cancellation() -> None:
    registry, _control = _registry()
    resolution = await registry.resolve_webhook(_initiated())
    reservation = resolution.reservation
    assert reservation is not None

    entered = asyncio.Event()
    release = asyncio.Event()
    real_settle = registry._settle_reservation  # noqa: SLF001

    async def observed_settle(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await release.wait()
        await real_settle(*args, **kwargs)

    registry._settle_reservation = observed_settle  # type: ignore[method-assign]
    await registry._lock.acquire()  # noqa: SLF001
    try:
        settling = asyncio.create_task(reservation.settle_after_submit_failure())
        await entered.wait()
        assert reservation._abandon_event.is_set()  # noqa: SLF001
        settling.cancel()
        assert settling.cancelling() > 0
        assert settling.done() is False
    finally:
        release.set()
        registry._lock.release()  # noqa: SLF001

    with pytest.raises(asyncio.CancelledError):
        await settling
    await registry.join_until_empty()
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_registry_never_calls_writer_provider_or_task_factory_under_lock() -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry_ref: dict[str, Any] = {}

    def assert_unlocked() -> None:
        assert registry_ref["registry"]._lock.locked() is False

    class InstrumentedWriter:
        async def commit_lease(self, **_: object) -> None:
            assert_unlocked()

    class InstrumentedControl(CallControl):
        async def answer(self, *_: object, **__: object) -> CallControlResult:
            assert_unlocked()
            return CallControlResult("accepted")

        async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
            assert_unlocked()
            return CallControlResult("accepted")

        async def hangup(self, *_: object, **__: object) -> CallControlResult:
            assert_unlocked()
            return CallControlResult("accepted")

    def task_factory(coroutine: Any, name: str) -> asyncio.Task[None]:
        assert_unlocked()
        return asyncio.create_task(coroutine, name=name)

    registry, _ = _registry(
        capacity=1,
        control=InstrumentedControl(),
        writer=InstrumentedWriter(),
        background_task_factory=task_factory,
    )
    registry_ref["registry"] = registry
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())
    await registry.reconcile_after_commit(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    snapshot = await registry.snapshot("control-a")
    assert snapshot is not None
    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a",
        token_digest=snapshot.token_digest,
    )
    assert claim is not None
    hangup = await registry.resolve_webhook(_hangup())
    await registry.reconcile_after_commit(
        _hangup(), hangup, WebhookCommitResult("first", "applied")
    )
    assert await registry.snapshot("control-a") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reaper_first", [False, True])
async def test_linked_placeholder_exact_abort_is_independent_of_reaper_order(
    reaper_first: bool,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(capacity=1, monotonic=lambda: clock[0])
    answered = await registry.resolve_webhook(_answered())
    clock[0] = 120.0
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    generation = await registry.generation_handle("control-a")
    assert generation is not None

    clock[0] = 130.0
    if reaper_first:
        reaped = await registry.reap_expired()
        confirmation = await registry.confirm_late_after_fail_closed(
            _answered(), answered, WebhookCommitResult("first", "applied")
        )
    else:
        confirmation = await registry.confirm_late_after_fail_closed(
            _answered(), answered, WebhookCommitResult("first", "applied")
        )
        reaped = await registry.reap_expired()
    await registry.join_until_empty()

    assert reaped == (1 if reaper_first else 0)
    assert confirmation.generation == generation
    assert confirmation.abort_scheduled is True
    assert control.streams == []
    assert len(control.hangups) == 1
    assert await registry.placeholder_count() == 0
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_late_placeholder_settlement_schedules_abort_before_next_lock_await() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(capacity=1, monotonic=lambda: clock[0])
    answered = await registry.resolve_webhook(_answered())
    clock[0] = 120.0
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    generation = await registry.generation_handle("control-a")
    assert generation is not None
    clock[0] = 130.0
    assert await registry.reap_expired() == 1

    settlement_returned_with_lock_held = asyncio.Event()
    settle_placeholder = registry._settle_placeholder  # type: ignore[attr-defined]

    async def settle_then_hold_lock(*args: Any, **kwargs: Any) -> Any:
        target = await settle_placeholder(*args, **kwargs)
        await registry._lock.acquire()  # type: ignore[attr-defined]
        settlement_returned_with_lock_held.set()
        return target

    registry._settle_placeholder = settle_then_hold_lock  # type: ignore[attr-defined,method-assign]
    confirmation_task = asyncio.create_task(
        registry.confirm_late_after_fail_closed(
            _answered(), answered, WebhookCommitResult("first", "applied")
        )
    )
    await settlement_returned_with_lock_held.wait()
    await asyncio.sleep(0)
    scheduled_without_another_lock_await = confirmation_task.done()
    if not scheduled_without_another_lock_await:
        confirmation_task.cancel()
    registry._lock.release()  # type: ignore[attr-defined]
    confirmation_result = (
        await asyncio.gather(confirmation_task, return_exceptions=True)
    )[0]

    assert scheduled_without_another_lock_await
    assert not isinstance(confirmation_result, BaseException)
    assert confirmation_result.generation == generation
    assert confirmation_result.abort_scheduled is True
    await registry.join_until_empty()
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("first_applied_last", [False, True])
@pytest.mark.parametrize("other_release", ["duplicate", "abandon"])
async def test_reaped_placeholder_identity_lives_until_every_exact_reservation_settles(
    first_applied_last: bool,
    other_release: str,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    token = "".join(chr(65 + index % 26) for index in range(43))
    registry, control = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        token_factory=lambda _: token,
    )
    answered = await registry.resolve_webhook(_answered())
    duplicate = await registry.resolve_duplicate_webhook(_answered())
    assert answered.reservation is not None
    assert duplicate.reservation is not None
    placeholder = answered.reservation._placeholder

    clock[0] = 120.0
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    async with registry._lock:  # type: ignore[attr-defined]
        entry = registry._by_control["control-a"]  # type: ignore[attr-defined]
        identity = placeholder.linked_abort_identity
    assert identity is not None
    assert tuple(field.name for field in dataclasses.fields(identity)) == (
        "call_control_id",
        "generation",
        "token_digest",
    )
    assert all(value is not entry for value in dataclasses.astuple(identity))
    assert all(value is not token for value in dataclasses.astuple(identity))

    clock[0] = 130.0
    assert await registry.reap_expired() == 1
    assert placeholder.linked_entry is None
    assert placeholder.linked_abort_identity is identity
    assert placeholder.precommit_refcount == 2

    async def release_other() -> None:
        if other_release == "duplicate":
            await duplicate.reservation.confirm(
                WebhookCommitResult("duplicate", "duplicate")
            )
        else:
            duplicate.reservation.abandon_before_submit()
            await _wait_for_reservation_settlement(duplicate.reservation)

    if first_applied_last:
        await release_other()
        assert control.hangups == []
        assert placeholder.precommit_refcount == 1
        assert placeholder.linked_abort_identity is identity
        confirmation = await registry.confirm_late_after_fail_closed(
            _answered(), answered, WebhookCommitResult("first", "applied")
        )
    else:
        confirmation = await registry.confirm_late_after_fail_closed(
            _answered(), answered, WebhookCommitResult("first", "applied")
        )
        assert placeholder.precommit_refcount == 1
        assert placeholder.linked_abort_identity is identity
        await release_other()

    await asyncio.wait_for(registry.join_until_empty(), timeout=1)
    assert confirmation.abort_scheduled is True
    assert placeholder.precommit_refcount == 0
    assert placeholder.linked_abort_identity is None
    assert placeholder.linked_entry is None
    assert len(control.hangups) == 1
    assert control.streams == []
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert not registry._background_owner._tasks  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_answer_owner_never_captures_raw_token_local() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    token = "".join(chr(65 + index % 26) for index in range(43))
    entered = asyncio.Event()

    class BlockingAnswer(CallControl):
        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    clock = [100.0]
    registry, _ = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=BlockingAnswer(),
        token_factory=lambda _: token,
    )
    initiated = await registry.resolve_webhook(_initiated())
    owner = asyncio.create_task(
        registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
    )
    await entered.wait()

    frame = _run_action_frame(owner)
    answer_token_local_cleared = frame.f_locals.get("raw_token") is None
    assert answer_token_local_cleared
    assert not _frame_owns_token_identity(frame, token)
    async with registry._lock:  # type: ignore[attr-defined]
        registry_still_owns_token = (  # type: ignore[attr-defined]
            registry._by_control["control-a"].raw_token is token
        )
    assert registry_still_owns_token

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    clock[0] = 130.0
    assert await registry.reap_expired() == 1
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["answer", "streaming"])
@pytest.mark.parametrize("cleanup_registration", ["registered", "fallback"])
@pytest.mark.parametrize(
    "blocked_phase", ["terminal_persistence", "cleanup_hangup", "exact_removal"]
)
async def test_cancelled_deadline_clears_action_secret_frames_before_cleanup(
    action: str,
    cleanup_registration: str,
    blocked_phase: str,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    token = "".join(chr(65 + index % 26) for index in range(43))
    clock = [100.0]
    cleanup_tasks: list[asyncio.Task[None]] = []

    class CompletionWriter(BlockingTerminalWriter):
        def __init__(self) -> None:
            super().__init__()
            self.completed = asyncio.Event()

        async def commit_lease(self, **values: object) -> None:
            await super().commit_lease(**values)
            self.completed.set()

    class BlockingActionControl(CallControl):
        def __init__(self) -> None:
            super().__init__()
            self.provider_entered = asyncio.Event()
            self.request_ref: weakref.ReferenceType[object] | None = None
            self.hangup_entered = asyncio.Event()
            self.hangup_release = asyncio.Event()
            self.hangup_completed = asyncio.Event()

        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            if action == "answer":
                self.provider_entered.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return CallControlResult("accepted")

        async def start_streaming(
            self, call_control_id: str, request: object, *, command_id: UUID
        ) -> CallControlResult:
            self.streams.append((call_control_id, command_id))
            if action == "streaming":
                self.request_ref = weakref.ref(request)
                self.provider_entered.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return CallControlResult("accepted")

        async def hangup(
            self,
            call_control_id: str,
            *,
            command_id: UUID,
            client_state: object = None,
        ) -> CallControlResult:
            del client_state
            self.hangups.append((call_control_id, command_id))
            self.hangup_entered.set()
            await self.hangup_release.wait()
            self.hangup_completed.set()
            return CallControlResult("accepted")

    def task_factory(coroutine: Any, name: str) -> asyncio.Task[None]:
        if (
            cleanup_registration == "fallback"
            and name == "voice-action-deadline-cleanup"
        ):
            raise RuntimeError("synthetic cleanup registration failure")
        task = asyncio.create_task(coroutine, name=name)
        if name == "voice-action-deadline-cleanup":
            cleanup_tasks.append(task)
        return task

    writer = CompletionWriter()
    control = BlockingActionControl()
    registry, _ = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        control=control,
        writer=writer,
        background_task_factory=task_factory,
        token_factory=lambda _: token,
    )
    if blocked_phase == "cleanup_hangup":
        writer.release.set()
    elif blocked_phase in {"terminal_persistence", "exact_removal"}:
        control.hangup_release.set()

    initiated = await registry.resolve_webhook(_initiated())
    if action == "answer":
        owner = asyncio.create_task(
            registry.reconcile_after_commit(
                _initiated(), initiated, WebhookCommitResult("first", "applied")
            )
        )
    else:
        await registry.reconcile_after_commit(
            _initiated(), initiated, WebhookCommitResult("first", "applied")
        )
        answered = await registry.resolve_webhook(_answered())
        owner = asyncio.create_task(
            registry.reconcile_after_commit(
                _answered(), answered, WebhookCommitResult("first", "applied")
            )
        )
    await control.provider_entered.wait()
    async with registry._lock:  # type: ignore[attr-defined]
        entry = registry._by_control["control-a"]  # type: ignore[attr-defined]
        slot = entry.answer if action == "answer" else entry.streaming
        assert slot.completion is not None
        waiter = asyncio.ensure_future(asyncio.shield(slot.completion))

    clock[0] = 130.0
    owner.cancel()
    await asyncio.wait_for(writer.entered.wait(), timeout=1)

    registry_lock_held = False
    try:
        if blocked_phase == "cleanup_hangup":
            await asyncio.wait_for(control.hangup_entered.wait(), timeout=1)
        elif blocked_phase == "exact_removal":
            await registry._lock.acquire()  # type: ignore[attr-defined]
            registry_lock_held = True
            writer.release.set()
            await asyncio.wait_for(control.hangup_completed.wait(), timeout=1)
            await asyncio.sleep(0)

        action_frame = _run_action_frame(owner)
        has_streaming_request_local = "streaming_request" in action_frame.f_locals
        assert has_streaming_request_local
        action_token_local_cleared = action_frame.f_locals["raw_token"] is None
        action_request_local_cleared = (
            action_frame.f_locals["streaming_request"] is None
        )
        assert action_token_local_cleared
        assert action_request_local_cleared
        assert owner.done() is False
        assert not _task_owns_token_identity(owner, token)
        assert all(
            not _task_owns_token_identity(cleanup_task, token)
            for cleanup_task in cleanup_tasks
        )
        if cleanup_registration == "registered":
            assert len(cleanup_tasks) == 1
            assert cleanup_tasks[0].done() is False
        else:
            assert cleanup_tasks == []
        if action == "streaming":
            assert control.request_ref is not None
            gc.collect()
            request_released = control.request_ref() is None
            assert request_released
        if registry_lock_held:
            current = registry._by_control["control-a"]  # type: ignore[attr-defined]
            registry_token_cleared = current.raw_token is None
            assert registry_token_cleared
            assert current.lease_state == "terminal"
        else:
            snapshot = await registry.snapshot("control-a")
            assert snapshot is not None
            assert snapshot.raw_token_retained is False
            assert snapshot.lease_state == "terminal"
        assert waiter.done() is True
        assert (await waiter).status_code == 200
    finally:
        writer.release.set()
        control.hangup_release.set()
        if registry_lock_held:
            registry._lock.release()  # type: ignore[attr-defined]

    with pytest.raises(asyncio.CancelledError):
        await owner
    await registry.join_until_empty()

    assert writer.completed.is_set()
    assert len(writer.commits) == 1
    assert control.hangup_completed.is_set()
    assert len(control.hangups) == 1
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert not registry._background_owner._tasks  # type: ignore[attr-defined]
    if cleanup_registration == "registered":
        assert len(cleanup_tasks) == 1
    else:
        assert cleanup_tasks == []
        assert registry.internal_failure_code == "background_task_registration_failed"


@pytest.mark.asyncio
async def test_reaped_placeholder_abandonment_releases_identity_without_provider_work() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(capacity=1, monotonic=lambda: clock[0])
    answered = await registry.resolve_webhook(_answered())
    assert answered.reservation is not None
    placeholder = answered.reservation._placeholder
    clock[0] = 120.0
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    assert placeholder.linked_abort_identity is not None

    clock[0] = 130.0
    assert await registry.reap_expired() == 1
    answered.reservation.abandon_before_submit()
    await registry.join_until_empty()

    assert placeholder.precommit_refcount == 0
    assert placeholder.linked_abort_identity is None
    assert placeholder.linked_entry is None
    assert control.hangups == []
    assert await registry.snapshot("control-a") is not None
    assert await registry.live_call_count() == 1
    assert not registry._background_owner._tasks  # type: ignore[attr-defined]

    clock[0] = 150.0
    assert await registry.reap_expired() == 1
    assert len(control.hangups) == 1
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_retired_placeholder_settlement_preserves_same_key_replacement_placeholder() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    registry, control = _registry(capacity=1, monotonic=lambda: clock[0])
    answered_a = await registry.resolve_webhook(_answered("answer-a"))
    assert answered_a.reservation is not None
    retired = answered_a.reservation._placeholder
    clock[0] = 120.0
    initiated_a = await registry.resolve_webhook(_initiated("event-a"))
    await registry.reconcile_after_commit(
        _initiated("event-a"), initiated_a, WebhookCommitResult("first", "applied")
    )
    generation_a = await registry.generation_handle("control-a")
    assert generation_a is not None

    clock[0] = 130.0
    assert await registry.reap_expired() == 1
    clock[0] = 150.0
    assert await registry.reap_expired() == 1
    answered_b = await registry.resolve_webhook(_answered("answer-b"))
    assert answered_b.reservation is not None
    replacement = answered_b.reservation._placeholder
    assert replacement is not retired

    confirmation = await registry.confirm_late_after_fail_closed(
        _answered("answer-a"),
        answered_a,
        WebhookCommitResult("first", "applied"),
    )

    assert confirmation.generation == generation_a
    assert confirmation.abort_scheduled is False
    assert retired.precommit_refcount == 0
    assert retired.linked_abort_identity is None
    async with registry._lock:  # type: ignore[attr-defined]
        assert registry._answered_placeholders["control-a"] is replacement  # type: ignore[attr-defined]
    assert len(control.hangups) == 1

    answered_b.reservation.abandon_before_submit()
    await registry.join_until_empty()
    assert await registry.placeholder_count() == 0
    assert not registry._background_owner._tasks  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_delayed_retired_abort_cannot_touch_same_key_replacement_generation() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    clock = [100.0]
    abort_entered = asyncio.Event()
    abort_release = asyncio.Event()
    abort_tasks: list[asyncio.Task[None]] = []

    def task_factory(coroutine: Any, name: str) -> asyncio.Task[None]:
        async def run() -> None:
            if name == "voice-fail-closed-lease-abort":
                abort_entered.set()
                await abort_release.wait()
            await coroutine

        task = asyncio.create_task(run(), name=name)
        if name == "voice-fail-closed-lease-abort":
            abort_tasks.append(task)
        return task

    registry, control = _registry(
        capacity=1,
        monotonic=lambda: clock[0],
        background_task_factory=task_factory,
    )
    answered = await registry.resolve_webhook(_answered())
    clock[0] = 120.0
    initiated_a = await registry.resolve_webhook(_initiated("event-a"))
    await registry.reconcile_after_commit(
        _initiated("event-a"), initiated_a, WebhookCommitResult("first", "applied")
    )
    clock[0] = 130.0
    assert await registry.reap_expired() == 1
    confirmation = await registry.confirm_late_after_fail_closed(
        _answered(), answered, WebhookCommitResult("first", "applied")
    )
    await asyncio.wait_for(abort_entered.wait(), timeout=1)

    clock[0] = 150.0
    assert await registry.reap_expired() == 1
    replacement = await registry.resolve_webhook(_initiated("event-b"))
    replacement_generation = await registry.generation_handle("control-a")
    assert replacement.reservation is not None
    assert replacement_generation is not None
    assert confirmation.abort_scheduled is True

    abort_release.set()
    await asyncio.gather(*abort_tasks)

    assert await registry.generation_handle("control-a") == replacement_generation
    replacement_snapshot = await registry.snapshot("control-a")
    assert replacement_snapshot is not None
    assert replacement_snapshot.lease_state == "provisional"
    assert len(control.hangups) == 1
    assert await registry.live_call_count() == 1

    replacement.reservation.abandon_before_submit()
    await registry.join_until_empty()
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_delayed_old_task9_call_id_hook_cannot_touch_replacement() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, _control = _registry(capacity=1)
    initiated = await registry.resolve_webhook(_initiated("old-initiated"))
    await registry.reconcile_after_commit(
        _initiated("old-initiated"),
        initiated,
        WebhookCommitResult("first", "applied"),
    )
    old = await registry.snapshot("control-a")
    assert old is not None
    hangup = await registry.resolve_webhook(_hangup("old-hangup"))
    await registry.reconcile_after_commit(
        _hangup("old-hangup"),
        hangup,
        WebhookCommitResult("first", "applied"),
    )
    replacement_resolution = await registry.resolve_webhook(
        _initiated("replacement-initiated")
    )
    replacement = await registry.snapshot("control-a")
    assert replacement is not None
    assert replacement.call_id != old.call_id

    assert await registry.prepare_required_recording_drain(old.call_id) is None

    current = await registry.snapshot("control-a")
    assert current is not None
    assert current.call_id == replacement.call_id
    assert replacement_resolution.reservation is not None
    replacement_resolution.reservation.abandon_before_submit()
    await registry.join_until_empty()


@pytest.mark.asyncio
async def test_provider_commit_cancellation_during_action_join_finishes_fixed_authority() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    class CancellationResistantStreaming(CallControl):
        def __init__(self) -> None:
            super().__init__()
            self.streaming_entered = asyncio.Event()
            self.streaming_cancelled = asyncio.Event()
            self.streaming_release = asyncio.Event()

        async def start_streaming(
            self,
            call_control_id: str,
            request: object,
            *,
            command_id: UUID,
        ) -> CallControlResult:
            self.streams.append((call_control_id, command_id))
            self.stream_tokens.append(
                request.stream_auth_token.get_secret_value()  # type: ignore[attr-defined]
            )
            self.streaming_entered.set()
            while not self.streaming_release.is_set():
                try:
                    await self.streaming_release.wait()
                except asyncio.CancelledError:
                    self.streaming_cancelled.set()
            return CallControlResult("accepted")

    control = CancellationResistantStreaming()
    registry, _ = _registry(capacity=1, control=control)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())
    streaming = asyncio.create_task(
        registry.reconcile_after_commit(
            _answered(),
            answered,
            WebhookCommitResult("first", "applied"),
        )
    )
    await control.streaming_entered.wait()
    hangup_event = _hangup("provider-terminal-cancel")
    hangup = await registry.resolve_webhook(hangup_event)
    finalizer = asyncio.create_task(
        registry.reconcile_after_commit(
            hangup_event,
            hangup,
            WebhookCommitResult("first", "applied"),
        )
    )
    await control.streaming_cancelled.wait()

    finalizer.cancel("post-commit-finalizer-cancel")
    control.streaming_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await finalizer
    await asyncio.gather(streaming, return_exceptions=True)

    assert cancelled.value.args == ("post-commit-finalizer-cancel",)
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0
    assert control.streaming_cancelled.is_set()
    retry = await registry.reconcile_after_commit(
        hangup_event,
        hangup,
        WebhookCommitResult("first", "applied"),
    )
    assert retry.status_code == 200
    assert not any(
        task is not asyncio.current_task()
        and task.get_name() == "voice-provider-terminal-settlement"
        and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_concurrent_provider_confirmers_keep_cancellation_attempt_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    class CancellationResistantStreaming(CallControl):
        def __init__(self) -> None:
            super().__init__()
            self.streaming_entered = asyncio.Event()
            self.streaming_cancelled = asyncio.Event()
            self.streaming_release = asyncio.Event()

        async def start_streaming(
            self,
            call_control_id: str,
            request: object,
            *,
            command_id: UUID,
        ) -> CallControlResult:
            self.streams.append((call_control_id, command_id))
            self.stream_tokens.append(
                request.stream_auth_token.get_secret_value()  # type: ignore[attr-defined]
            )
            self.streaming_entered.set()
            while not self.streaming_release.is_set():
                try:
                    await self.streaming_release.wait()
                except asyncio.CancelledError:
                    self.streaming_cancelled.set()
            return CallControlResult("accepted")

    control = CancellationResistantStreaming()
    registry, _ = _registry(capacity=1, control=control)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    answered = await registry.resolve_webhook(_answered())
    streaming = asyncio.create_task(
        registry.reconcile_after_commit(
            _answered(),
            answered,
            WebhookCommitResult("first", "applied"),
        )
    )
    await control.streaming_entered.wait()
    hangup_event = _hangup("provider-concurrent-confirmers")
    hangup = await registry.resolve_webhook(hangup_event)
    assert hangup.reservation is not None
    from projetv0_voice.admission import CallReservation

    confirm = CallReservation.confirm
    confirm_calls = 0
    second_confirmer_entered = asyncio.Event()

    async def observe_confirm(self: object, result: object) -> object:
        nonlocal confirm_calls
        confirm_calls += 1
        if confirm_calls == 2:
            second_confirmer_entered.set()
        return await confirm(self, result)  # type: ignore[arg-type]

    monkeypatch.setattr(CallReservation, "confirm", observe_confirm)
    first = asyncio.create_task(
        registry.reconcile_after_commit(
            hangup_event,
            hangup,
            WebhookCommitResult("first", "applied"),
        )
    )
    await control.streaming_cancelled.wait()
    second = asyncio.create_task(
        registry.reconcile_after_commit(
            hangup_event,
            hangup,
            WebhookCommitResult("first", "applied"),
        )
    )
    await second_confirmer_entered.wait()

    first.cancel("first-confirmer-cancel")
    control.streaming_release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await first
    second_result = await second
    await asyncio.gather(streaming, return_exceptions=True)

    assert cancelled.value.args == ("first-confirmer-cancel",)
    assert second_result.status_code == 200
    assert confirm_calls == 2
    assert hangup.reservation._settlement_task is not None  # type: ignore[attr-defined]  # noqa: SLF001
    assert hangup.reservation._settlement_task.done()  # type: ignore[attr-defined]  # noqa: SLF001
    assert await registry.snapshot("control-a") is None
    assert len(control.hangups) == 0
    assert not any(
        task is not asyncio.current_task()
        and task.get_name() == "voice-provider-terminal-settlement"
        and not task.done()
        for task in asyncio.all_tasks()
    )
