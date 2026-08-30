from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from projetv0_voice.telnyx.call_control import CallControlResult
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
            token_factory=lambda _: "A" * 43,
            uuid_factory=lambda: next(ids),
            candidate_run_id=candidate_run_id,
            admission_expires_at=admission_expires_at,
            **kwargs,
        ),
        control,
    )


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
        UUID("11111111-1111-4111-8111-111111111111")
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
        ("control-a", UUID("22222222-2222-4222-8222-222222222222"))
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
        ("control-a", UUID("33333333-3333-4333-8333-333333333333"))
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
        ("control-a", UUID("33333333-3333-4333-8333-333333333333")),
        ("control-a", UUID("33333333-3333-4333-8333-333333333333")),
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
async def test_terminal_cleanup_drains_attached_owners_before_removal_and_release() -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, _ = _registry(capacity=1)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    generation = await registry.generation_handle("control-a")
    assert generation is not None
    construction_release = asyncio.Event()
    session_release = asyncio.Event()

    class DrainOwner:
        def __init__(self, release: asyncio.Event) -> None:
            self.release = release
            self.requested = asyncio.Event()

        def request_drain(self) -> None:
            self.requested.set()

        async def wait(self) -> None:
            await self.release.wait()

    construction = DrainOwner(construction_release)
    session = DrainOwner(session_release)
    assert await registry.attach_construction_owner(generation, construction)
    assert await registry.attach_session_owner(generation, object(), session)

    hangup = await registry.resolve_webhook(_hangup())
    terminal = asyncio.create_task(
        registry.reconcile_after_commit(
            _hangup(), hangup, WebhookCommitResult("first", "applied")
        )
    )
    await asyncio.gather(construction.requested.wait(), session.requested.wait())

    assert await registry.snapshot("control-a") is not None
    assert await registry.live_call_count() == 1
    construction_release.set()
    await asyncio.sleep(0)
    assert terminal.done() is False
    session_release.set()
    await terminal
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


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
async def test_exact_duplicate_during_terminal_cleanup_is_200_without_action() -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.persistence.writer import WebhookCommitResult

    registry, control = _registry(capacity=1)
    initiated = await registry.resolve_webhook(_initiated())
    await registry.reconcile_after_commit(
        _initiated(), initiated, WebhookCommitResult("first", "applied")
    )
    generation = await registry.generation_handle("control-a")
    assert generation is not None
    drain_requested = asyncio.Event()
    release_drain = asyncio.Event()

    class DrainOwner:
        def request_drain(self) -> None:
            drain_requested.set()

        async def wait(self) -> None:
            await release_drain.wait()

    assert await registry.attach_session_owner(generation, object(), DrainOwner())
    hangup = await registry.resolve_webhook(_hangup())
    terminal = asyncio.create_task(
        registry.reconcile_after_commit(
            _hangup(), hangup, WebhookCommitResult("first", "applied")
        )
    )
    await drain_requested.wait()

    duplicate = await registry.resolve_duplicate_webhook(_initiated())
    duplicate_result = await registry.reconcile_after_commit(
        _initiated(), duplicate, WebhookCommitResult("duplicate", "duplicate")
    )
    mismatched = dataclasses.replace(_initiated(), call_leg_id="leg-other")
    with pytest.raises(CallAdmissionRejected, match="call_identity_conflict"):
        await registry.resolve_duplicate_webhook(mismatched)

    assert duplicate_result.status_code == 200
    assert duplicate.effect is None
    assert duplicate.reservation is None
    assert len(control.answers) == 1
    assert control.streams == []
    release_drain.set()
    await terminal


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
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_registry_never_calls_writer_provider_or_lifecycle_owner_under_lock() -> None:
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
    generation = await registry.generation_handle("control-a")
    assert generation is not None

    class LifecycleOwner:
        def request_drain(self) -> None:
            assert_unlocked()

        async def wait(self) -> None:
            assert_unlocked()

    assert await registry.attach_session_owner(generation, object(), LifecycleOwner())
    hangup = await registry.resolve_webhook(_hangup())
    await registry.reconcile_after_commit(
        _hangup(), hangup, WebhookCommitResult("first", "applied")
    )
    assert await registry.snapshot("control-a") is None
