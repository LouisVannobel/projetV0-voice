from __future__ import annotations

import asyncio
import base64
import dataclasses
import importlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from nacl.signing import SigningKey

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.telnyx.call_control import CallControlResult
from projetv0_voice.telnyx.webhooks import (
    InvalidWebhookPayload,
    TelnyxWebhookProcessor,
    TelnyxWebhookVerifier,
    WebhookDisposition,
)

TIMESTAMP = 1_777_118_400


def _signed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    event_type: str,
    payload: dict[str, object],
) -> tuple[TelnyxWebhookVerifier, bytes, list[tuple[str, str]]]:
    body = json.dumps(
        {
            "data": {
                "id": "event-a",
                "event_type": event_type,
                "occurred_at": "2026-08-29T10:00:00Z",
                "payload": payload,
            }
        }
    ).encode()
    key = SigningKey.generate()
    public_key = base64.b64encode(bytes(key.verify_key)).decode("ascii")
    signature = base64.b64encode(
        key.sign(f"{TIMESTAMP}|".encode() + body).signature
    ).decode("ascii")
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    monkeypatch.setattr(verification.time, "time", lambda: float(TIMESTAMP))
    return (
        TelnyxWebhookVerifier(public_key=public_key),
        body,
        [
            ("Telnyx-Signature-Ed25519", signature),
            ("Telnyx-Timestamp", str(TIMESTAMP)),
        ],
    )


@pytest.mark.parametrize(
    ("event_type", "payload", "direction", "call_state"),
    [
        (
            "call.initiated",
            {"call_control_id": "control-a", "direction": "incoming", "state": "parked"},
            "incoming",
            "parked",
        ),
        (
            "call.answered",
            {"call_control_id": "control-a", "state": "answered"},
            None,
            "answered",
        ),
        ("call.hangup", {"call_control_id": "control-a"}, None, None),
        ("call.recording.saved", {"recording_id": "recording-a"}, None, None),
        ("call.recording.error", {}, None, None),
        ("future.event", {"provider_field": "ignored"}, None, None),
    ],
)
def test_signed_event_allowlist_extracts_only_exact_bounded_action_fields(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    payload: dict[str, object],
    direction: str | None,
    call_state: str | None,
) -> None:
    verifier, body, headers = _signed(
        monkeypatch, event_type=event_type, payload=payload
    )

    event = verifier.verify(body=body, headers=headers)

    assert event.direction == direction
    assert event.call_state == call_state
    assert not hasattr(event, "payload")


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        ("call.initiated", {"call_control_id": "control-a"}),
        (
            "call.initiated",
            {"call_control_id": "control-a", "direction": "outgoing", "state": "parked"},
        ),
        (
            "call.initiated",
            {"call_control_id": "control-a", "direction": "incoming", "state": "answered"},
        ),
        ("call.answered", {"call_control_id": "control-a"}),
        ("call.answered", {"call_control_id": "control-a", "state": "parked"}),
        ("call.hangup", {}),
    ],
)
def test_handled_event_with_invalid_event_specific_fields_is_400_payload(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    payload: dict[str, object],
) -> None:
    verifier, body, headers = _signed(
        monkeypatch, event_type=event_type, payload=payload
    )

    with pytest.raises(InvalidWebhookPayload, match="invalid_payload"):
        verifier.verify(body=body, headers=headers)


def test_direction_and_call_state_change_the_semantic_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: list[bytes] = []
    for event_type, payload in (
        (
            "call.initiated",
            {"call_control_id": "control-a", "direction": "incoming", "state": "parked"},
        ),
        (
            "call.answered",
            {"call_control_id": "control-a", "state": "answered"},
        ),
    ):
        verifier, body, headers = _signed(
            monkeypatch, event_type=event_type, payload=payload
        )
        values.append(verifier.verify(body=body, headers=headers).semantic_fingerprint_sha256)

    assert values[0] != values[1]


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        ("future.event", {"direction": "outgoing", "state": "completed"}),
        (
            "call.hangup",
            {"call_control_id": "control-a", "direction": "outgoing", "state": "hangup"},
        ),
        (
            "call.recording.error",
            {"direction": "outgoing", "state": "failed", "reason": "ignored"},
        ),
    ],
)
def test_unrelated_direction_and_state_are_ignored_for_non_action_events(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    payload: dict[str, object],
) -> None:
    verifier, body, headers = _signed(
        monkeypatch, event_type=event_type, payload=payload
    )

    event = verifier.verify(body=body, headers=headers)

    assert event.direction is None
    assert event.call_state is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("classification", "resolver_error", "expected"),
    [
        ("missing", "call_capacity_reached", 503),
        ("missing", "qualification_window_expired", 503),
        ("missing", "call_identity_conflict", 400),
        ("missing", "call_event_invalid", 400),
        ("missing", "unexpected", 500),
        ("conflict", None, 400),
        ("duplicate", None, 200),
    ],
)
async def test_processor_maps_closed_admission_and_receipt_classifications(
    classification: str,
    resolver_error: str | None,
    expected: int,
) -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.telnyx.webhooks import ResolvedWebhook, VerifiedWebhook

    event = VerifiedWebhook(
        event_id="event-a",
        event_type="call.initiated",
        occurred_at=datetime(2026, 8, 29, 10, tzinfo=UTC),
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
        direction="incoming",
        call_state="parked",
    )
    resolver_calls = 0
    started = 0

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class Handle:
        async def wait(self) -> Any:
            return WebhookDisposition(200)

    class Owner:
        async def classify_webhook_receipt(self, _: VerifiedWebhook) -> str:
            return classification

        def start_webhook_finalization(self, _: Any, resolution: Any) -> Handle:
            nonlocal started
            started += 1
            assert resolution.effect is None
            return Handle()

    def resolver(_: VerifiedWebhook) -> ResolvedWebhook:
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_error == "unexpected":
            raise RuntimeError("RAW-UNEXPECTED-SENTINEL")
        if resolver_error is not None:
            raise CallAdmissionRejected(resolver_error)
        return ResolvedWebhook(None)

    disposition = await TelnyxWebhookProcessor(
        verifier=Verifier(),
        resolver=resolver,
        finalizer_owner=Owner(),
    ).process(body=b"{}", headers=[])

    assert disposition.status_code == expected
    assert resolver_calls == (1 if classification == "missing" else 0)
    assert started == (1 if classification in {"missing", "duplicate"} and expected == 200 else 0)


@pytest.mark.asyncio
async def test_signed_initiated_flows_through_async_registry_and_postcommit_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from projetv0_voice.admission import CallRegistry

    verifier, body, headers = _signed(
        monkeypatch,
        event_type="call.initiated",
        payload={
            "call_control_id": "control-a",
            "call_leg_id": "leg-a",
            "call_session_id": "session-a",
            "direction": "incoming",
            "state": "parked",
        },
    )
    database = tmp_path / "flow.sqlite"
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: bytes(range(32))}, active_version=1),
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()

    class Control:
        def __init__(self) -> None:
            self.answers: list[tuple[str, UUID]] = []

        async def answer(
            self, call_control_id: str, *, command_id: UUID
        ) -> CallControlResult:
            self.answers.append((call_control_id, command_id))
            return CallControlResult("accepted")

        async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

    ids = iter(
        UUID(value)
        for value in (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            "44444444-4444-4444-8444-444444444444",
        )
    )
    control = Control()
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="agent-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: datetime(2026, 8, 29, 10, tzinfo=UTC),
        monotonic=lambda: 100.0,
        token_factory=lambda _: "A" * 43,
        uuid_factory=lambda: next(ids),
    )

    class Handle:
        def __init__(self, task: asyncio.Task[Any]) -> None:
            self.task = task

        async def wait(self) -> Any:
            return await self.task

    class Owner:
        async def classify_webhook_receipt(self, event: Any) -> str:
            return await writer.classify_webhook_receipt(
                event_id=event.event_id,
                semantic_fingerprint_sha256=event.semantic_fingerprint_sha256,
            )

        def start_webhook_finalization(self, event: Any, resolution: Any) -> Handle:
            async def finalize() -> Any:
                effect = resolution.effect
                result = await writer.submit_webhook(
                    receipt={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "call_control_id": event.call_control_id,
                        "occurred_at": event.occurred_at,
                        "received_at": datetime(2026, 8, 29, 10, tzinfo=UTC),
                        "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
                    },
                    lease=effect.lease,
                    operation=effect.operation,
                ).wait()
                confirmation = await registry.confirm_committed(
                    event, resolution, result
                )
                return confirmation.disposition

            return Handle(asyncio.create_task(finalize()))

    processor = TelnyxWebhookProcessor(
        verifier=verifier,
        resolver=registry.resolve_webhook,
        duplicate_resolver=registry.resolve_duplicate_webhook,
        finalizer_owner=Owner(),
    )

    disposition = await processor.process(body=body, headers=headers)
    assert disposition.status_code == 200
    assert control.answers == [
        ("control-a", UUID("22222222-2222-4222-8222-222222222222"))
    ]
    await writer.drain(2)
    await writer_task
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT state FROM call_leases").fetchone() == ("pending",)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
    assert ("A" * 43).encode() not in database.read_bytes()


@pytest.mark.asyncio
async def test_http_cancellation_detaches_from_registered_webhook_finalizer() -> None:
    module = importlib.import_module("projetv0_voice.telnyx.webhooks")
    event = module.VerifiedWebhook(
        event_id="event-a",
        event_type="future.event",
        occurred_at=datetime(2026, 8, 29, 10, tzinfo=UTC),
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
    )
    owner_started = asyncio.Event()
    release_owner = asyncio.Event()
    owner_finished = asyncio.Event()

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            return event

    class Reservation:
        def __init__(self) -> None:
            self.abandoned = 0

        def abandon_before_submit(self) -> None:
            self.abandoned += 1

    class Handle:
        def __init__(self, task: asyncio.Task[Any]) -> None:
            self.task = task

        async def wait(self) -> Any:
            return await self.task

    class Owner:
        def __init__(self) -> None:
            self.task: asyncio.Task[Any] | None = None

        async def classify_webhook_receipt(self, _: Any) -> str:
            return "missing"

        def start_webhook_finalization(self, received: Any, resolution: Any) -> Handle:
            assert received is event
            assert resolution.reservation is reservation

            async def finalize() -> Any:
                owner_started.set()
                await release_owner.wait()
                owner_finished.set()
                return module.WebhookDisposition(200)

            self.task = asyncio.create_task(finalize())
            return Handle(self.task)

    reservation = Reservation()
    owner = Owner()
    processor = module.TelnyxWebhookProcessor(
        verifier=StubVerifier(),
        resolver=lambda _: module.ResolvedWebhook(None, reservation),
        finalizer_owner=owner,
    )
    request = asyncio.create_task(processor.process(body=b"{}", headers=[]))
    await owner_started.wait()

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert owner.task is not None
    assert owner.task.cancelled() is False
    assert reservation.abandoned == 0

    release_owner.set()
    await owner.task
    assert owner_finished.is_set()


@pytest.mark.asyncio
async def test_owner_registration_failure_abandons_before_submit_and_returns_503() -> None:
    module = importlib.import_module("projetv0_voice.telnyx.webhooks")
    event = module.VerifiedWebhook(
        event_id="event-a",
        event_type="future.event",
        occurred_at=datetime(2026, 8, 29, 10, tzinfo=UTC),
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
    )

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            return event

    class Reservation:
        def __init__(self) -> None:
            self.abandoned = 0

        def abandon_before_submit(self) -> None:
            self.abandoned += 1

    class RejectingOwner:
        async def classify_webhook_receipt(self, _: Any) -> str:
            return "missing"

        def start_webhook_finalization(self, *_: object) -> Any:
            raise RuntimeError("RAW-REGISTRATION-SENTINEL")

    reservation = Reservation()
    processor = module.TelnyxWebhookProcessor(
        verifier=StubVerifier(),
        resolver=lambda _: module.ResolvedWebhook(None, reservation),
        finalizer_owner=RejectingOwner(),
    )

    disposition = await processor.process(body=b"RAW-BODY-SENTINEL", headers=[])

    assert disposition == module.WebhookDisposition(503)
    assert reservation.abandoned == 1
    assert "RAW" not in repr(disposition)


@pytest.mark.asyncio
async def test_consumed_candidate_restart_classifies_duplicate_before_any_mint(
    tmp_path: Path,
) -> None:
    from projetv0_voice.admission import CallRegistry
    from projetv0_voice.telnyx.webhooks import ResolvedWebhook, VerifiedWebhook

    run_id = UUID("99999999-9999-4999-8999-999999999999")
    database = tmp_path / "candidate-restart.sqlite"
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: bytes(range(32))}, active_version=1),
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    duplicate_event = VerifiedWebhook(
        event_id="event-a",
        event_type="call.initiated",
        occurred_at=datetime(2026, 8, 29, 10, tzinfo=UTC),
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
        direction="incoming",
        call_state="parked",
    )
    await writer.submit_webhook(
        receipt={
            "event_id": duplicate_event.event_id,
            "event_type": duplicate_event.event_type,
            "call_control_id": duplicate_event.call_control_id,
            "occurred_at": duplicate_event.occurred_at,
            "received_at": duplicate_event.occurred_at,
            "semantic_fingerprint_sha256": duplicate_event.semantic_fingerprint_sha256,
        },
        lease={
            "action": "upsert",
            "call_control_id": "control-a",
            "call_id": UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            "tenant_id": "tenant-a",
            "agent_id": "agent-a",
            "state": "pending",
            "token_hash": b"t" * 32,
            "created_at": duplicate_event.occurred_at,
            "expires_at": duplicate_event.occurred_at.replace(second=30),
            "closed_at": None,
        },
        operation=None,
        qualification_run_id=run_id,
    ).wait()
    assert await writer.qualification_run_consumed(run_id)
    mint_count = 0

    def reject_token_mint(_: int) -> str:
        nonlocal mint_count
        mint_count += 1
        raise AssertionError("duplicate must not mint")

    class Control:
        async def answer(self, *_: object, **__: object) -> CallControlResult:
            raise AssertionError("duplicate must not answer")

        async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
            raise AssertionError("duplicate must not stream")

        async def hangup(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

    registry = CallRegistry(
        writer=writer,
        call_control=Control(),
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="agent-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        candidate_run_id=run_id,
        candidate_consumed=True,
        utcnow=lambda: duplicate_event.occurred_at,
        monotonic=lambda: 100.0,
        token_factory=reject_token_mint,
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("duplicate must not UUID")),
    )

    class Handle:
        def __init__(self, task: asyncio.Task[WebhookDisposition]) -> None:
            self.task = task

        async def wait(self) -> WebhookDisposition:
            return await self.task

    class Owner:
        async def classify_webhook_receipt(self, event: VerifiedWebhook) -> str:
            return await writer.classify_webhook_receipt(
                event_id=event.event_id,
                semantic_fingerprint_sha256=event.semantic_fingerprint_sha256,
            )

        def start_webhook_finalization(
            self, event: VerifiedWebhook, resolution: ResolvedWebhook
        ) -> Handle:
            async def finalize() -> WebhookDisposition:
                result = await writer.submit_webhook(
                    receipt={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "call_control_id": event.call_control_id,
                        "occurred_at": event.occurred_at,
                        "received_at": event.occurred_at,
                        "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
                    },
                    lease=None if resolution.effect is None else resolution.effect.lease,
                    operation=(
                        None if resolution.effect is None else resolution.effect.operation
                    ),
                ).wait()
                return await registry.reconcile_after_commit(event, resolution, result)

            return Handle(asyncio.create_task(finalize()))

    class Verifier:
        def __init__(self, event: VerifiedWebhook) -> None:
            self.event = event

        def verify(self, **_: object) -> VerifiedWebhook:
            return self.event

    owner = Owner()
    duplicate = await TelnyxWebhookProcessor(
        verifier=Verifier(duplicate_event),
        resolver=registry.resolve_webhook,
        duplicate_resolver=registry.resolve_duplicate_webhook,
        finalizer_owner=owner,
    ).process(body=b"{}", headers=[])
    distinct_event = dataclasses.replace(
        duplicate_event,
        event_id="event-b",
        call_control_id="control-b",
        semantic_fingerprint_sha256=b"g" * 32,
    )
    distinct = await TelnyxWebhookProcessor(
        verifier=Verifier(distinct_event),
        resolver=registry.resolve_webhook,
        duplicate_resolver=registry.resolve_duplicate_webhook,
        finalizer_owner=owner,
    ).process(body=b"{}", headers=[])

    assert (duplicate.status_code, distinct.status_code) == (200, 503)
    assert mint_count == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (1,)
    await writer.drain(2)
    await writer_task


@pytest.mark.asyncio
async def test_real_owner_registration_failure_abandons_registry_reservation() -> None:
    from projetv0_voice.admission import CallRegistry
    from projetv0_voice.telnyx.webhooks import VerifiedWebhook

    event = VerifiedWebhook(
        event_id="event-a",
        event_type="call.initiated",
        occurred_at=datetime(2026, 8, 29, 10, tzinfo=UTC),
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
        direction="incoming",
        call_state="parked",
    )

    class Writer:
        async def commit_lease(self, **_: object) -> None:
            return None

    class Control:
        async def answer(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

    ids = iter(
        UUID(value)
        for value in (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            "44444444-4444-4444-8444-444444444444",
        )
    )
    registry = CallRegistry(
        writer=Writer(),
        call_control=Control(),
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="agent-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: event.occurred_at,
        monotonic=lambda: 100.0,
        token_factory=lambda _: "A" * 43,
        uuid_factory=lambda: next(ids),
    )

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class RejectingOwner:
        async def classify_webhook_receipt(self, _: VerifiedWebhook) -> str:
            return "missing"

        def start_webhook_finalization(self, *_: object) -> Any:
            raise RuntimeError("synthetic closed owner")

    disposition = await TelnyxWebhookProcessor(
        verifier=Verifier(),
        resolver=registry.resolve_webhook,
        duplicate_resolver=registry.resolve_duplicate_webhook,
        finalizer_owner=RejectingOwner(),
    ).process(body=b"{}", headers=[])
    await registry.join_until_empty()

    assert disposition.status_code == 503
    assert await registry.snapshot("control-a") is None
    assert await registry.live_call_count() == 0


@pytest.mark.asyncio
async def test_timeout_then_late_commit_uses_fail_closed_confirmation_only(
    tmp_path: Path,
) -> None:
    from projetv0_voice.admission import CallRegistry
    from projetv0_voice.telnyx.webhooks import VerifiedWebhook

    commit_blocked = asyncio.Event()
    release_commit = asyncio.Event()

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            commit_blocked.set()
            await release_commit.wait()

    writer = PersistenceWriter(
        tmp_path / "late-fail-closed.sqlite",
        CryptoKeyring({1: bytes(range(32))}, active_version=1),
        failpoint=failpoint,
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    answers: list[str] = []
    hangups: list[str] = []

    class Control:
        async def answer(self, call_control_id: str, **_: object) -> CallControlResult:
            answers.append(call_control_id)
            return CallControlResult("accepted")

        async def start_streaming(self, *_: object, **__: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, call_control_id: str, **_: object) -> CallControlResult:
            hangups.append(call_control_id)
            return CallControlResult("accepted")

    ids = iter(
        UUID(value)
        for value in (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            "44444444-4444-4444-8444-444444444444",
        )
    )
    event = VerifiedWebhook(
        event_id="event-a",
        event_type="call.initiated",
        occurred_at=datetime(2026, 8, 29, 10, tzinfo=UTC),
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
        direction="incoming",
        call_state="parked",
    )
    registry = CallRegistry(
        writer=writer,
        call_control=Control(),
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="agent-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: event.occurred_at,
        monotonic=lambda: 100.0,
        token_factory=lambda _: "A" * 43,
        uuid_factory=lambda: next(ids),
    )
    resolution = await registry.resolve_webhook(event)
    assert resolution.effect is not None
    ticket = writer.submit_webhook(
        receipt={
            "event_id": event.event_id,
            "event_type": event.event_type,
            "call_control_id": event.call_control_id,
            "occurred_at": event.occurred_at,
            "received_at": event.occurred_at,
            "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
        },
        lease=resolution.effect.lease,
        operation=resolution.effect.operation,
    )
    await commit_blocked.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(ticket.wait()), timeout=0.01)
    writer.latch_control_commit_timeout()
    release_commit.set()
    result = await ticket.wait()

    confirmation = await registry.confirm_late_after_fail_closed(
        event, resolution, result
    )
    await registry.join_until_empty()

    assert confirmation.abort_scheduled is True
    assert answers == []
    assert hangups == ["control-a"]
    assert await registry.snapshot("control-a") is None
    await writer.drain(2)
    await writer_task
