"""One-row-at-a-time relay from the encrypted SQLite outbox."""

from __future__ import annotations

import asyncio
import math
import random as random_module
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol
from uuid import UUID

from projetv0_voice.persistence.postgres_sink import (
    OperationConflictError,
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkErasedError,
    OperationSinkStaleLeaseError,
    OperationSinkTransientError,
    PostgresOperationSink,
)
from projetv0_voice.persistence.writer import OutboxItem, PersistenceWriter, RelayClaimResult

DURABLE_AGE_LIMIT_SECONDS = 900.0
ACK_BUDGET_SECONDS = 1.5
POOL_ACQUIRE_BUDGET_SECONDS = 2.0
SQL_TRANSACTION_BUDGET_SECONDS = 5.0
CLAIM_SAFETY_SECONDS = 1.0
MINIMUM_CLAIM_BUDGET_SECONDS = (
    POOL_ACQUIRE_BUDGET_SECONDS
    + SQL_TRANSACTION_BUDGET_SECONDS
    + ACK_BUDGET_SECONDS
    + CLAIM_SAFETY_SECONDS
)

RelayStatus = Literal[
    "empty",
    "delivered",
    "retry_scheduled",
    "stale_claim",
    "claim_budget_expired",
    "degraded",
]


class RelayAlreadyRunningError(RuntimeError):
    """The single process-lifetime relay already owns a run."""


@dataclass(frozen=True, slots=True)
class RelayResult:
    status: RelayStatus
    processed: int
    acked: int
    retried: int
    discarded: int = 0

    def __post_init__(self) -> None:
        if self.status not in {
            "empty",
            "delivered",
            "retry_scheduled",
            "stale_claim",
            "claim_budget_expired",
            "degraded",
        }:
            raise ValueError("invalid_relay_status")
        if any(type(value) is not int or value < 0 for value in self._counters()):
            raise ValueError("invalid_relay_counters")

    def _counters(self) -> tuple[int, int, int, int]:
        return self.processed, self.acked, self.retried, self.discarded


class RelayWriter(Protocol):
    async def erase_call_content(
        self, call_id: UUID, *, now: datetime, expected_item: OutboxItem | None = None
    ) -> datetime | None: ...
    async def oldest_outbox_created_at(self) -> datetime | None: ...

    async def read_relay_batch(
        self, *, batch_size: int, now: datetime, lease_seconds: int
    ) -> tuple[OutboxItem, ...]: ...

    async def ack_outbox(
        self, *, queue_id: int, expected_claim_attempt: int
    ) -> RelayClaimResult: ...

    async def retry_outbox(
        self,
        *,
        queue_id: int,
        expected_claim_attempt: int,
        next_attempt_at: datetime,
        error_code: str,
    ) -> RelayClaimResult: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def maintain_call_content(
    writer: PersistenceWriter,
    sink: PostgresOperationSink,
    stop_call: Callable[[UUID], Awaitable[None]],
    *,
    utcnow: Callable[[], datetime],
    timeout_seconds: float,
) -> None:
    """Native local cleanup and exact ACK replay, awaited before FIFO age gating."""

    async def acknowledge(call_id: UUID, token: UUID, cleaned_at: datetime) -> None:
        acknowledged = False
        try:
            await sink.ack_call_erasure(call_id, token, cleaned_at)
            acknowledged = True
        except OperationSinkStaleLeaseError:
            pass  # A replaced lease is settled locally, never called a native success.
        await writer.finish_erasure_ack(call_id, token, acknowledged=acknowledged)

    async with asyncio.timeout(timeout_seconds):
        for call_id, token, cleaned_at in await writer.pending_erasure_acks():
            await acknowledge(call_id, token, cleaned_at)
        leases = await sink.lease_call_erasures("voice-call-erasure", 30, 100)
        for lease in leases:
            await stop_call(lease.call_id)
            local_cleaned_at = await writer.erase_call_content(
                lease.call_id, lease_token=lease.lease_token, now=_aware_utc(utcnow())
            )
            if local_cleaned_at is None:
                raise RuntimeError("local_erasure_not_completed")
            await acknowledge(lease.call_id, lease.lease_token, local_cleaned_at)
        for call_id in await writer.expired_content_calls(now=_aware_utc(utcnow())):
            await stop_call(call_id)
            await writer.erase_call_content(call_id, now=_aware_utc(utcnow()))
        # Only the real FIFO head, with a known provider identity and durable
        # content tombstone, may hand off its unchanged native purge obligation.
        for _ in range(100):
            head = await writer.erased_recording_head()
            if head is None:
                break
            claimed = await writer.read_relay_batch(
                batch_size=1, now=_aware_utc(utcnow()), lease_seconds=30
            )
            if not claimed or claimed[0].queue_id != head:
                raise OperationSinkTransientError("erased_recording_claim_held")
            item = claimed[0]
            await sink.ingest(item.operation)
            acknowledged = await writer.ack_outbox(
                queue_id=item.queue_id, expected_claim_attempt=item.claim_attempt
            )
            if not acknowledged.applied:
                raise OperationSinkTransientError("erased_recording_claim_replaced")
        else:
            if await writer.erased_recording_head() is not None:
                raise OperationSinkTransientError("erased_recording_handoff_pending")
        await writer.cleanup_local_state(now=_aware_utc(utcnow()))


class OutboxRelay:
    """Relay FIFO rows without holding SQLite ownership across network I/O."""

    def __init__(
        self,
        writer: RelayWriter,
        sink: PostgresOperationSink,
        *,
        utcnow: Callable[[], datetime] = _utc_now,
        random: Callable[[], float] = random_module.random,
        on_degraded: Callable[[], Awaitable[None]],
        drain: Callable[[], Awaitable[None]],
        claim_lease_seconds: int = 10,
        drain_timeout_seconds: float = 5.0,
        before_fifo: Callable[[], Awaitable[None]] | None = None,
        stop_erased_call: Callable[[UUID], Awaitable[None]] | None = None,
    ) -> None:
        if (
            type(claim_lease_seconds) is not int
            or claim_lease_seconds < 10
            or claim_lease_seconds > 300
            or claim_lease_seconds < MINIMUM_CLAIM_BUDGET_SECONDS
        ):
            raise ValueError("claim_lease_seconds is outside the supported range")
        if (
            isinstance(drain_timeout_seconds, bool)
            or not isinstance(drain_timeout_seconds, (int, float))
            or not math.isfinite(drain_timeout_seconds)
            or drain_timeout_seconds <= 0
        ):
            raise ValueError("drain_timeout_seconds must be positive")
        self._writer = writer
        self._sink = sink
        self._utcnow = utcnow
        self._random = random
        self._on_degraded = on_degraded
        self._drain = drain
        self._claim_lease_seconds = claim_lease_seconds
        self._drain_timeout_seconds = float(drain_timeout_seconds)
        self._before_fifo = before_fifo
        self._stop_erased_call = stop_erased_call
        self._running = False
        self._degraded = False
        self._callback_started = False
        self._drain_started = False

    @property
    def claim_lease_seconds(self) -> int:
        """Expose the exact ambiguity holdoff owned by lifecycle supervision."""

        return self._claim_lease_seconds

    async def prepare_before_fifo(self) -> None:
        if self._before_fifo is not None:
            await self._before_fifo()

    async def run_once(self, *, batch_size: int = 100) -> RelayResult:
        if type(batch_size) is not int or not 1 <= batch_size <= 100:
            raise ValueError("batch_size is outside the supported range")
        if self._running:
            raise RelayAlreadyRunningError("relay_run_already_active")
        if self._degraded:
            return RelayResult("degraded", 0, 0, 0)
        self._running = True
        processed = 0
        acked = 0
        retried = 0
        discarded = 0
        try:
            if self._before_fifo is not None:
                try:
                    await self.prepare_before_fifo()
                except OperationSinkTransientError:
                    return RelayResult("retry_scheduled", 0, 0, 0)
            for _ in range(batch_size):
                oldest = await self._writer.oldest_outbox_created_at()
                if oldest is None:
                    status: RelayStatus = "delivered" if processed else "empty"
                    return RelayResult(status, processed, acked, retried, discarded)
                age_now = self._fresh_utc()
                if (age_now - _aware_utc(oldest)).total_seconds() > DURABLE_AGE_LIMIT_SECONDS:
                    await self._latch_degradation()
                    return RelayResult("degraded", processed, acked, retried, discarded)

                claim_now = self._fresh_utc()
                claimed = await self._writer.read_relay_batch(
                    batch_size=1,
                    now=claim_now,
                    lease_seconds=self._claim_lease_seconds,
                )
                if not claimed:
                    status = "delivered" if processed else "empty"
                    return RelayResult(status, processed, acked, retried, discarded)
                if len(claimed) != 1:
                    raise RuntimeError("writer_claim_cardinality_invalid")
                item = claimed[0]

                before_network = self._fresh_utc()
                remaining = (
                    _aware_utc(item.claim_expires_at) - before_network
                ).total_seconds()
                if remaining < MINIMUM_CLAIM_BUDGET_SECONDS:
                    return RelayResult(
                        "claim_budget_expired", processed, acked, retried, discarded
                    )

                processed += 1
                try:
                    await self._sink.ingest(item.operation)
                except OperationSinkErasedError:
                    if item.operation.kind == "recording.upsert":
                        # A rejected recording has not handed off its real purge identity.
                        await self._latch_degradation()
                        return RelayResult("degraded", processed, acked, retried, discarded)
                    if self._stop_erased_call is not None:
                        await self._stop_erased_call(item.operation.call_id)
                    cleaned = await self._writer.erase_call_content(
                        item.operation.call_id, now=self._fresh_utc(), expected_item=item
                    )
                    if cleaned is None:
                        return RelayResult("stale_claim", processed, acked, retried, discarded)
                    discarded += 1
                    continue
                except asyncio.CancelledError:
                    raise
                except OperationSinkCommitAmbiguousError:
                    raise
                except OperationSinkTransientError:
                    next_attempt = self._fresh_utc() + timedelta(
                        seconds=self._retry_delay(item.claim_attempt)
                    )
                    retry = await self._writer.retry_outbox(
                        queue_id=item.queue_id,
                        expected_claim_attempt=item.claim_attempt,
                        next_attempt_at=next_attempt,
                        error_code="postgres_transient",
                    )
                    if not retry.applied:
                        return RelayResult("stale_claim", processed, acked, retried, discarded)
                    retried += 1
                    return RelayResult(
                        "retry_scheduled", processed, acked, retried, discarded
                    )
                except (OperationConflictError, OperationSinkContractError):
                    await self._latch_degradation()
                    return RelayResult("degraded", processed, acked, retried, discarded)

                acknowledged = await self._writer.ack_outbox(
                    queue_id=item.queue_id,
                    expected_claim_attempt=item.claim_attempt,
                )
                if not acknowledged.applied:
                    return RelayResult("stale_claim", processed, acked, retried, discarded)
                acked += 1
            return RelayResult("delivered", processed, acked, retried, discarded)
        finally:
            self._running = False

    def _fresh_utc(self) -> datetime:
        return _aware_utc(self._utcnow())

    def _retry_delay(self, claim_attempt: int) -> float:
        if type(claim_attempt) is not int or claim_attempt <= 0:
            raise RuntimeError("writer_claim_attempt_invalid")
        jitter_source = self._random()
        if (
            isinstance(jitter_source, bool)
            or not isinstance(jitter_source, (int, float))
            or not math.isfinite(jitter_source)
            or not 0.0 <= jitter_source <= 1.0
        ):
            raise RuntimeError("relay_random_invalid")
        exponent = min(claim_attempt - 1, 20)
        delay = math.ldexp(0.5, exponent) * (1.0 + float(jitter_source) * 0.25)
        return min(30.0, delay)

    async def _latch_degradation(self) -> None:
        self._degraded = True
        if not self._callback_started:
            self._callback_started = True
            try:
                await self._on_degraded()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        if not self._drain_started:
            self._drain_started = True
            try:
                await asyncio.wait_for(
                    self._drain(), timeout=self._drain_timeout_seconds
                )
            except asyncio.CancelledError:
                raise
            except (TimeoutError, Exception):
                pass


def _aware_utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("relay clock must return an aware datetime")
    return value.astimezone(UTC)
