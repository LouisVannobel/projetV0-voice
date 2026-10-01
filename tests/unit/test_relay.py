from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.postgres_sink import (
    OperationConflictError,
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkTransientError,
)
from projetv0_voice.persistence.relay import (
    OutboxRelay,
    RelayAlreadyRunningError,
    RelayResult,
)
from projetv0_voice.persistence.writer import OutboxItem, RelayClaimResult

NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def operation(number: int) -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(int=number),
        deployment_id="agent-a",
        call_id=UUID(int=100 + number),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id=f"control-{number}",
            telnyx_call_leg_id=None,
            telnyx_call_session_id=None,
            status="pending",
            disclosure_state="pending",
            started_at=None,
            ended_at=None,
            end_reason=None,
            retention_until=NOW + timedelta(days=1),
        ),
    )


def item(number: int, *, attempt: int = 1, expires: datetime | None = None) -> OutboxItem:
    return OutboxItem(
        queue_id=number,
        operation=operation(number),
        created_at=NOW,
        claim_attempt=attempt,
        next_attempt_at=NOW,
        claim_expires_at=expires or NOW + timedelta(seconds=10),
        last_error_code=None,
    )


class FakeWriter:
    def __init__(
        self,
        claims: list[OutboxItem] | None = None,
        *,
        ages: list[datetime | None] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.claims = deque(claims or [])
        self.ages = deque(ages or [NOW] * (len(claims or []) + 1))
        self.events = events if events is not None else []
        self.ack_results: deque[RelayClaimResult] = deque()
        self.retry_results: deque[RelayClaimResult] = deque()
        self.retry_calls: list[dict[str, object]] = []
        self.claim_calls: list[tuple[int, datetime, int]] = []
        self.age_error: BaseException | None = None
        self.claim_error: BaseException | None = None
        self.ack_error: BaseException | None = None
        self.retry_error: BaseException | None = None
        self.ack_gate: asyncio.Event | None = None
        self.ack_entered = asyncio.Event()

    async def oldest_outbox_created_at(self) -> datetime | None:
        self.events.append("age")
        if self.age_error is not None:
            raise self.age_error
        return self.ages.popleft() if self.ages else None

    async def read_relay_batch(
        self, *, batch_size: int, now: datetime, lease_seconds: int
    ) -> tuple[OutboxItem, ...]:
        self.events.append("claim")
        self.claim_calls.append((batch_size, now, lease_seconds))
        if self.claim_error is not None:
            raise self.claim_error
        return (self.claims.popleft(),) if self.claims else ()

    async def ack_outbox(
        self, *, queue_id: int, expected_claim_attempt: int
    ) -> RelayClaimResult:
        self.events.append(f"ack-{queue_id}")
        self.ack_entered.set()
        if self.ack_gate is not None:
            await self.ack_gate.wait()
        if self.ack_error is not None:
            raise self.ack_error
        return self.ack_results.popleft() if self.ack_results else RelayClaimResult(True)

    async def retry_outbox(
        self,
        *,
        queue_id: int,
        expected_claim_attempt: int,
        next_attempt_at: datetime,
        error_code: str,
    ) -> RelayClaimResult:
        self.events.append(f"retry-{queue_id}")
        self.retry_calls.append(
            {
                "queue_id": queue_id,
                "expected_claim_attempt": expected_claim_attempt,
                "next_attempt_at": next_attempt_at,
                "error_code": error_code,
            }
        )
        if self.retry_error is not None:
            raise self.retry_error
        return self.retry_results.popleft() if self.retry_results else RelayClaimResult(True)


class FakeSink:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        outcomes: list[BaseException | None] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.outcomes = deque(outcomes or [])
        self.gate = gate
        self.calls = 0
        self.entered = asyncio.Event()

    async def ingest(self, candidate: VoiceOperationV1) -> None:
        self.calls += 1
        self.events.append(f"sink-{candidate.operation_id.int}")
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        outcome = self.outcomes.popleft() if self.outcomes else None
        if outcome is not None:
            raise outcome


class Clock:
    def __init__(self, values: list[datetime]) -> None:
        self.values = deque(values)
        self.last = values[-1]

    def __call__(self) -> datetime:
        if self.values:
            self.last = self.values.popleft()
        return self.last


def relay(
    writer: FakeWriter,
    sink: FakeSink,
    *,
    clock: Any | None = None,
    random_value: float = 0.0,
    on_degraded: Any | None = None,
    drain: Any | None = None,
    claim_lease_seconds: int = 10,
    drain_timeout_seconds: float = 0.01,
) -> OutboxRelay:
    async def noop() -> None:
        return None

    return OutboxRelay(
        writer,
        sink,
        utcnow=clock or (lambda: NOW),
        random=lambda: random_value,
        on_degraded=on_degraded or noop,
        drain=drain or noop,
        claim_lease_seconds=claim_lease_seconds,
        drain_timeout_seconds=drain_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_two_rows_follow_exact_age_claim_sink_ack_sequence_one_at_a_time() -> None:
    events: list[str] = []
    writer = FakeWriter([item(1), item(2)], ages=[NOW, NOW, None], events=events)
    sink = FakeSink(events=events)
    result = await relay(writer, sink).run_once(batch_size=2)
    assert result == RelayResult(status="delivered", processed=2, acked=2, retried=0)
    assert events == [
        "age",
        "claim",
        "sink-1",
        "ack-1",
        "age",
        "claim",
        "sink-2",
        "ack-2",
    ]
    assert [call[0] for call in writer.claim_calls] == [1, 1]


@pytest.mark.asyncio
async def test_empty_age_returns_empty_without_claim_or_network() -> None:
    writer = FakeWriter([], ages=[None])
    sink = FakeSink()
    result = await relay(writer, sink).run_once(batch_size=10)
    assert result == RelayResult(status="empty", processed=0, acked=0, retried=0)
    assert writer.claim_calls == []
    assert sink.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("age_seconds", "expected"),
    [(900, "delivered"), (901, "degraded")],
)
async def test_durable_age_exact_boundary_and_terminal_overage(
    age_seconds: int, expected: str
) -> None:
    events: list[str] = []

    async def degraded() -> None:
        events.append("degraded")

    async def drain() -> None:
        events.append("drain")

    writer = FakeWriter([item(1)], ages=[NOW - timedelta(seconds=age_seconds)], events=events)
    sink = FakeSink(events=events)
    result = await relay(writer, sink, on_degraded=degraded, drain=drain).run_once()
    assert result.status == expected
    if expected == "degraded":
        assert events == ["age", "degraded", "drain"]
        assert sink.calls == 0


@pytest.mark.asyncio
async def test_age_is_recomputed_between_rows_and_blocks_second_network_call() -> None:
    events: list[str] = []
    writer = FakeWriter(
        [item(1), item(2)],
        ages=[NOW - timedelta(seconds=900), NOW - timedelta(seconds=901)],
        events=events,
    )
    sink = FakeSink(events=events)
    result = await relay(writer, sink).run_once(batch_size=2)
    assert result == RelayResult(status="degraded", processed=1, acked=1, retried=0)
    assert sink.calls == 1
    assert events[:5] == ["age", "claim", "sink-1", "ack-1", "age"]


@pytest.mark.asyncio
async def test_transient_failure_schedules_one_durable_retry_and_stops_before_row_two() -> None:
    writer = FakeWriter([item(1, attempt=2), item(2)], ages=[NOW])
    sink = FakeSink(outcomes=[OperationSinkTransientError("pool_acquire_timeout")])
    result = await relay(writer, sink, random_value=1.0).run_once(batch_size=2)
    assert result == RelayResult(status="retry_scheduled", processed=1, acked=0, retried=1)
    assert writer.retry_calls == [
        {
            "queue_id": 1,
            "expected_claim_attempt": 2,
            "next_attempt_at": NOW + timedelta(seconds=1.25),
            "error_code": "postgres_transient",
        }
    ]
    assert len(writer.claim_calls) == 1


@pytest.mark.asyncio
async def test_retry_schedule_uses_fresh_post_failure_clock() -> None:
    clock = Clock([NOW, NOW, NOW, NOW + timedelta(seconds=5)])
    writer = FakeWriter([item(1)], ages=[NOW])
    sink = FakeSink(outcomes=[OperationSinkTransientError("pool_acquire_timeout")])
    result = await relay(writer, sink, clock=clock).run_once()
    assert result.status == "retry_scheduled"
    assert writer.retry_calls[0]["next_attempt_at"] == NOW + timedelta(seconds=5.5)


@pytest.mark.asyncio
async def test_backoff_is_capped_at_30_seconds_without_sleep() -> None:
    writer = FakeWriter([item(1, attempt=1_000_000)], ages=[NOW])
    sink = FakeSink(outcomes=[OperationSinkTransientError("pool_acquire_timeout")])
    result = await relay(writer, sink, random_value=1.0).run_once()
    assert result.status == "retry_scheduled"
    assert writer.retry_calls[0]["next_attempt_at"] == NOW + timedelta(seconds=30)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [OperationConflictError("operation_hash_conflict"), OperationSinkContractError("bad")],
)
async def test_permanent_failure_latches_before_callbacks_once_and_blocks_later_runs(
    failure: BaseException,
) -> None:
    events: list[str] = []

    async def degraded() -> None:
        events.append("callback")

    async def drain() -> None:
        events.append("drain")

    writer = FakeWriter([item(1)], ages=[NOW], events=events)
    sink = FakeSink(events=events, outcomes=[failure])
    candidate = relay(writer, sink, on_degraded=degraded, drain=drain)
    first = await candidate.run_once()
    second = await candidate.run_once()
    assert first == RelayResult(status="degraded", processed=1, acked=0, retried=0)
    assert second == RelayResult(status="degraded", processed=0, acked=0, retried=0)
    assert events == ["age", "claim", "sink-1", "callback", "drain"]


@pytest.mark.asyncio
async def test_callback_error_and_drain_timeout_do_not_clear_degradation_or_leave_orphan() -> None:
    drain_cancelled = asyncio.Event()

    async def callback() -> None:
        raise RuntimeError("RAW-CALLBACK")

    async def drain() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            drain_cancelled.set()

    writer = FakeWriter([item(1)], ages=[NOW])
    sink = FakeSink(outcomes=[OperationConflictError("operation_hash_conflict")])
    candidate = relay(writer, sink, on_degraded=callback, drain=drain)
    assert (await candidate.run_once()).status == "degraded"
    assert drain_cancelled.is_set()
    assert (await candidate.run_once()).status == "degraded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [asyncio.CancelledError(), OperationSinkCommitAmbiguousError("ambiguous")],
)
async def test_cancellation_or_ambiguous_failure_leaves_claim_untouched(
    failure: BaseException,
) -> None:
    writer = FakeWriter([item(1)], ages=[NOW])
    sink = FakeSink(outcomes=[failure])
    with pytest.raises(type(failure)):
        await relay(writer, sink).run_once()
    assert writer.retry_calls == []
    assert not any(event.startswith("ack") for event in writer.events)


@pytest.mark.asyncio
async def test_cancellation_after_sink_success_during_ack_leaves_claim_replayable() -> None:
    writer = FakeWriter([item(1)], ages=[NOW])
    writer.ack_gate = asyncio.Event()
    sink = FakeSink()
    candidate = relay(writer, sink)
    running = asyncio.create_task(candidate.run_once())
    await writer.ack_entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert writer.events.count("ack-1") == 1
    assert writer.retry_calls == []
    assert writer.ack_results == deque()


@pytest.mark.asyncio
async def test_insufficient_remaining_claim_budget_starts_no_network_or_writer_mutation() -> None:
    expiring = item(1, expires=NOW + timedelta(seconds=9, milliseconds=499))
    writer = FakeWriter([expiring], ages=[NOW])
    sink = FakeSink()
    result = await relay(writer, sink).run_once()
    assert result == RelayResult(
        status="claim_budget_expired", processed=0, acked=0, retried=0
    )
    assert sink.calls == 0
    assert writer.retry_calls == []
    assert not any(event.startswith("ack") for event in writer.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(("mutation", "status"), [("ack", "stale_claim"), ("retry", "stale_claim")])
async def test_stale_ack_and_retry_are_nonfatal_stops(mutation: str, status: str) -> None:
    if mutation == "ack":
        writer = FakeWriter([item(1)], ages=[NOW])
        writer.ack_results.append(RelayClaimResult(False))
        sink = FakeSink()
    else:
        writer = FakeWriter([item(1)], ages=[NOW])
        writer.retry_results.append(RelayClaimResult(False))
        sink = FakeSink(outcomes=[OperationSinkTransientError("pool")])
    result = await relay(writer, sink).run_once()
    assert result.status == status


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["age", "claim", "ack", "retry"])
async def test_writer_failures_propagate_without_becoming_postgres_conflicts(stage: str) -> None:
    raw = RuntimeError("writer-local-failure")
    writer = FakeWriter([item(1)], ages=[NOW])
    sink = FakeSink(
        outcomes=[OperationSinkTransientError("pool")] if stage == "retry" else []
    )
    if stage == "age":
        writer.age_error = raw
    elif stage == "claim":
        writer.claim_error = raw
    elif stage == "ack":
        writer.ack_error = raw
    else:
        writer.retry_error = raw
    with pytest.raises(RuntimeError, match="writer-local-failure"):
        await relay(writer, sink).run_once()


@pytest.mark.asyncio
async def test_same_instance_overlap_rejects_without_extra_writer_or_sink_work() -> None:
    gate = asyncio.Event()
    writer = FakeWriter([item(1)], ages=[NOW])
    sink = FakeSink(gate=gate)
    candidate = relay(writer, sink)
    active = asyncio.create_task(candidate.run_once())
    await sink.entered.wait()
    with pytest.raises(RelayAlreadyRunningError, match="relay_run_already_active"):
        await candidate.run_once()
    assert len(writer.claim_calls) == 1
    assert sink.calls == 1
    gate.set()
    await active


@pytest.mark.asyncio
async def test_claim_lease_configuration_rejects_nine_and_accepts_ten_seconds() -> None:
    writer = FakeWriter([], ages=[None])
    sink = FakeSink()
    with pytest.raises(ValueError, match="claim_lease_seconds"):
        relay(writer, sink, claim_lease_seconds=9)
    assert (await relay(writer, sink, claim_lease_seconds=10).run_once()).status == "empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [True, 0, 101, 1.0, "1"])
async def test_batch_size_is_strictly_bounded_before_work(batch_size: Any) -> None:
    writer = FakeWriter([item(1)], ages=[NOW])
    sink = FakeSink()
    with pytest.raises(ValueError, match="batch_size"):
        await relay(writer, sink).run_once(batch_size=batch_size)
    assert writer.events == []
    assert sink.calls == 0


def test_relay_result_is_frozen_bounded_and_contains_no_sensitive_fields() -> None:
    result = RelayResult(status="delivered", processed=1, acked=1, retried=0)
    assert result == RelayResult(status="delivered", processed=1, acked=1, retried=0)
    assert result.__slots__ == ("status", "processed", "acked", "retried", "discarded")
    with pytest.raises((AttributeError, TypeError)):
        result.processed = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        RelayResult(status="delivered", processed=-1, acked=0, retried=0)
