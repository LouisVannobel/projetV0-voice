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


def _event(event_type: str, event_id: str) -> VerifiedWebhook:
    return VerifiedWebhook(
        event_id=event_id,
        event_type=event_type,
        occurred_at=NOW,
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
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
    writer: Writer, control: CallControl, clock: Any
) -> Any:
    from projetv0_voice.admission import CallRegistry

    ids = iter(
        UUID(value)
        for value in (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            "44444444-4444-4444-8444-444444444444",
        )
    )
    return CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="agent-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=clock,
        token_factory=lambda _: "A" * 43,
        uuid_factory=lambda: next(ids),
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
@pytest.mark.parametrize("now", [130.0, 130.001])
async def test_claim_at_or_after_monotonic_deadline_loses(now: float) -> None:
    from projetv0_voice.admission import ProcessLeaseAuthority

    clock = 100.0
    writer = Writer()
    registry = _registry(writer, CallControl(), lambda: clock)
    await _durable_waiting_wss(registry)
    clock = now

    claim = await ProcessLeaseAuthority(registry).claim_once(
        call_control_id="control-a", token_digest=DIGEST
    )

    assert claim is None
    assert writer.commits == []


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
