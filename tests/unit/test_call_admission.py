from __future__ import annotations

import asyncio
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
    return (
        CallRegistry(
            writer=Writer(),
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
