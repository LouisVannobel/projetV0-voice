from __future__ import annotations

import asyncio
import base64
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
                return await registry.reconcile_after_commit(event, resolution, result)

            return Handle(asyncio.create_task(finalize()))

    processor = TelnyxWebhookProcessor(
        verifier=verifier,
        resolver=registry.resolve_webhook,
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
