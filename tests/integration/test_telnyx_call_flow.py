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
from projetv0_voice.persistence.schema import V1_SCHEMA_SQL
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.telnyx.call_control import CallControlResult
from projetv0_voice.telnyx.webhooks import (
    InvalidWebhookPayload,
    TelnyxWebhookProcessor,
    TelnyxWebhookVerifier,
    WebhookDisposition,
)

TIMESTAMP = 1_777_118_400

_TASK9_FINGERPRINT_CASES = (
    (
        "call.recording.saved",
        {
            "call_control_id": "control-a",
            "call_leg_id": "leg-a",
            "call_session_id": "session-a",
            "recording_id": "recording-a",
            "stream_id": "stream-a",
            "client_state": "Y2xpZW50",
            "recording_started_at": "2026-08-29T10:00:01Z",
            "recording_ended_at": "2026-08-29T10:00:02Z",
            "channels": "dual",
        },
        "ee386beef0b2150270123fb7b921b95c38c019141fb7eb06d51b23411774a68f",
        "ee386beef0b2150270123fb7b921b95c38c019141fb7eb06d51b23411774a68f",
        None,
    ),
    (
        "call.recording.error",
        {"call_control_id": "control-a", "recording_id": "recording-a"},
        "4adff04e91ddab242b7da27455c53e47281da8ad4e212a2f7310412482890e91",
        "4adff04e91ddab242b7da27455c53e47281da8ad4e212a2f7310412482890e91",
        None,
    ),
    (
        "call.hangup",
        {"call_control_id": "control-a"},
        "fbb8378f2552106a0ec14b9a7283f4cad06d54e2949aef52607f99dc1c86f6f2",
        "fbb8378f2552106a0ec14b9a7283f4cad06d54e2949aef52607f99dc1c86f6f2",
        None,
    ),
    (
        "future.event",
        {"provider_field": "ignored"},
        "bd2b9c69b44a94fa3ed2bf5214edfcd6054190cc52740b3ba118889596fd8943",
        "bd2b9c69b44a94fa3ed2bf5214edfcd6054190cc52740b3ba118889596fd8943",
        None,
    ),
    (
        "call.initiated",
        {
            "call_control_id": "control-a",
            "call_leg_id": "leg-a",
            "call_session_id": "session-a",
            "direction": "incoming",
            "state": "parked",
        },
        "ed766819519ec6f7ec9c479a643f9d72b232ac0d4722341cbfb4350ee812098c",
        "2759dd9333080e63bec2d57b9bbe5d02354c9cdfe052bd2b0c34a73eb6f5c13a",
        "ed766819519ec6f7ec9c479a643f9d72b232ac0d4722341cbfb4350ee812098c",
    ),
    (
        "call.answered",
        {
            "call_control_id": "control-a",
            "call_leg_id": "leg-a",
            "call_session_id": "session-a",
            "state": "answered",
        },
        "dada33a9b81ebeff7cf7312684749ac363b48dce7502c94ac823ca687c00a085",
        "83e464f83cbd9fd77ba4e5ca0d8cfc263cb2f29131ff6e0866e746842505a8c7",
        "dada33a9b81ebeff7cf7312684749ac363b48dce7502c94ac823ca687c00a085",
    ),
)


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


@pytest.mark.parametrize(
    ("event_type", "payload", "legacy_hex", "current_hex", "alias_hex"),
    _TASK9_FINGERPRINT_CASES,
)
def test_task9_fingerprint_bytes_remain_primary_or_one_verified_action_alias(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    payload: dict[str, object],
    legacy_hex: str,
    current_hex: str,
    alias_hex: str | None,
) -> None:
    verifier, body, headers = _signed(
        monkeypatch, event_type=event_type, payload=payload
    )

    event = verifier.verify(body=body, headers=headers)

    assert event.semantic_fingerprint_sha256 == bytes.fromhex(current_hex)
    assert event.legacy_v1_semantic_fingerprint_sha256 == (
        None if alias_hex is None else bytes.fromhex(alias_hex)
    )
    if event_type in {"call.initiated", "call.answered"}:
        assert event.semantic_fingerprint_sha256 != bytes.fromhex(legacy_hex)
        changed_payload = {**payload, "call_leg_id": "leg-changed"}
        changed_verifier, changed_body, changed_headers = _signed(
            monkeypatch,
            event_type=event_type,
            payload=changed_payload,
        )
        changed = changed_verifier.verify(
            body=changed_body,
            headers=changed_headers,
        )
        assert changed.semantic_fingerprint_sha256 != event.semantic_fingerprint_sha256
        assert (
            changed.legacy_v1_semantic_fingerprint_sha256
            != event.legacy_v1_semantic_fingerprint_sha256
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "payload", "legacy_hex", "_current_hex", "_alias_hex"),
    _TASK9_FINGERPRINT_CASES,
)
async def test_real_v1_migration_redelivery_accepts_only_verified_fingerprint_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    event_type: str,
    payload: dict[str, object],
    legacy_hex: str,
    _current_hex: str,
    _alias_hex: str | None,
) -> None:
    from projetv0_voice.persistence.writer import WebhookCommitResult
    from projetv0_voice.telnyx.webhooks import ResolvedWebhook, VerifiedWebhook

    verifier, body, headers = _signed(
        monkeypatch, event_type=event_type, payload=payload
    )
    verified = verifier.verify(body=body, headers=headers)
    database = tmp_path / f"legacy-{event_type.replace('.', '-')}.sqlite"
    legacy_fingerprint = bytes.fromhex(legacy_hex)
    with sqlite3.connect(database) as connection:
        connection.executescript(V1_SCHEMA_SQL)
        connection.execute(
            """
            INSERT INTO webhook_receipts (
                event_id, event_type, call_control_id, occurred_at, received_at,
                semantic_fingerprint_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                verified.event_id,
                verified.event_type,
                verified.call_control_id,
                "2026-08-29T10:00:00Z",
                "2026-08-29T10:00:00Z",
                legacy_fingerprint,
            ),
        )

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: bytes(range(32))}, active_version=1),
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    missing_resolutions = 0
    duplicate_resolutions = 0

    async def reject_missing(_: VerifiedWebhook) -> ResolvedWebhook:
        nonlocal missing_resolutions
        missing_resolutions += 1
        raise AssertionError("legacy duplicate must not enter missing admission")

    async def reconcile_duplicate(_: VerifiedWebhook) -> ResolvedWebhook:
        nonlocal duplicate_resolutions
        duplicate_resolutions += 1
        return ResolvedWebhook(None)

    class Handle:
        def __init__(self, task: asyncio.Task[WebhookDisposition]) -> None:
            self.task = task

        async def wait(self) -> WebhookDisposition:
            return await self.task

    class Owner:
        def __init__(self) -> None:
            self.results: list[WebhookCommitResult] = []

        async def classify_webhook_receipt(self, event: VerifiedWebhook) -> str:
            return await writer.classify_webhook_receipt(
                event_id=event.event_id,
                semantic_fingerprint_sha256=event.semantic_fingerprint_sha256,
                legacy_v1_semantic_fingerprint_sha256=(
                    event.legacy_v1_semantic_fingerprint_sha256
                ),
            )

        def start_webhook_finalization(
            self,
            event: VerifiedWebhook,
            resolution: ResolvedWebhook,
            _receipt: object = "first",
        ) -> Handle:
            assert resolution.effect is None

            async def finalize() -> WebhookDisposition:
                result = await writer.submit_webhook(
                    receipt={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "call_control_id": event.call_control_id,
                        "occurred_at": event.occurred_at,
                        "received_at": event.occurred_at,
                        "semantic_fingerprint_sha256": (
                            event.semantic_fingerprint_sha256
                        ),
                    },
                    lease=None,
                    operation=None,
                    legacy_v1_semantic_fingerprint_sha256=(
                        event.legacy_v1_semantic_fingerprint_sha256
                    ),
                ).wait()
                assert isinstance(result, WebhookCommitResult)
                self.results.append(result)
                return WebhookDisposition(200)

            return Handle(asyncio.create_task(finalize()))

    owner = Owner()
    disposition = await TelnyxWebhookProcessor(
        verifier=verifier,
        resolver=reject_missing,
        duplicate_resolver=reconcile_duplicate,
        finalizer_owner=owner,
    ).process(body=body, headers=headers)

    assert disposition.status_code == 200
    assert missing_resolutions == 0
    assert duplicate_resolutions == 1
    assert len(owner.results) == 1
    assert owner.results[0].receipt == "duplicate"
    assert owner.results[0].effect == "duplicate"
    assert writer.fatal_fault is None
    await writer.drain(2)
    await writer_task
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute(
            "SELECT semantic_fingerprint_sha256 FROM webhook_receipts"
        ).fetchone() == (legacy_fingerprint,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)


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

        def start_webhook_finalization(
            self,
            _: Any,
            resolution: Any,
            _receipt: object = "first",
        ) -> Handle:
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

        def start_webhook_finalization(
            self,
            event: Any,
            resolution: Any,
            _receipt: object = "first",
        ) -> Handle:
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
        ("control-a", UUID("11111111-1111-4111-8111-111111111111"))
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

        def start_webhook_finalization(
            self,
            received: Any,
            resolution: Any,
            _receipt: object = "first",
        ) -> Handle:
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
            self,
            event: VerifiedWebhook,
            resolution: ResolvedWebhook,
            _receipt: object = "first",
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


@pytest.mark.asyncio
async def test_public_observed_ingress_preserves_both_legacy_alias_phases_unchanged() -> None:
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
        semantic_fingerprint_sha256=b"n" * 32,
        legacy_v1_semantic_fingerprint_sha256=b"l" * 32,
        direction="incoming",
        call_state="parked",
    )
    phases: list[tuple[str, bytes | None]] = []

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class Handle:
        async def wait(self) -> WebhookDisposition:
            return WebhookDisposition(200)

    class Owner:
        async def classify_webhook_receipt(self, received: VerifiedWebhook) -> str:
            phases.append(
                ("classification", received.legacy_v1_semantic_fingerprint_sha256)
            )
            return "missing"

        def start_webhook_finalization(
            self,
            received: VerifiedWebhook,
            resolution: ResolvedWebhook,
            _receipt: object = "first",
        ) -> Handle:
            assert resolution.effect is None
            phases.append(
                ("submission", received.legacy_v1_semantic_fingerprint_sha256)
            )
            return Handle()

    async def resolve(received: VerifiedWebhook) -> ResolvedWebhook:
        assert received is event
        return ResolvedWebhook(None)

    processor = TelnyxWebhookProcessor(
        verifier=Verifier(),  # type: ignore[arg-type]
        resolver=resolve,
        finalizer_owner=Owner(),  # type: ignore[arg-type]
    )

    outcome = await processor.process_observed(body=b"{}", headers=[])
    compatibility = await TelnyxWebhookProcessor(
        verifier=Verifier(),  # type: ignore[arg-type]
        resolver=resolve,
        finalizer_owner=Owner(),  # type: ignore[arg-type]
    ).process(body=b"{}", headers=[])

    assert outcome.disposition == WebhookDisposition(200)
    assert compatibility == outcome.disposition
    assert phases == [
        ("classification", b"l" * 32),
        ("submission", b"l" * 32),
        ("classification", b"l" * 32),
        ("submission", b"l" * 32),
    ]
    assert outcome.webhook_class == "initiated"
    assert outcome.receipt == "first"
    assert outcome.metric_disposition == "ok"
    assert outcome.admission_rejection is None


def test_public_observed_result_rejects_values_outside_closed_domains() -> None:
    from projetv0_voice.telnyx.webhooks import ObservedWebhookResult

    with pytest.raises(ValueError, match="^observed_webhook_result_invalid$"):
        ObservedWebhookResult(
            disposition=WebhookDisposition(200),
            webhook_class="private",  # type: ignore[arg-type]
            receipt="first",
            metric_disposition="ok",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("resolver-reject", 503),
        ("resolver-error", 500),
        ("resolver-invalid", 500),
        ("finalizer-transfer", 503),
        ("finalizer-wait", 500),
    ],
)
async def test_terminal_duplicate_never_becomes_new_admission_rejection(
    failure: str,
    expected_status: int,
) -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.telnyx.webhooks import ResolvedWebhook, VerifiedWebhook

    event = VerifiedWebhook(
        event_id="duplicate-event",
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
        semantic_fingerprint_sha256=b"n" * 32,
        legacy_v1_semantic_fingerprint_sha256=b"l" * 32,
        direction="incoming",
        call_state="parked",
    )

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class Handle:
        async def wait(self) -> WebhookDisposition:
            if failure == "finalizer-wait":
                raise RuntimeError("private-finalizer-wait")
            return WebhookDisposition(200)

    class Owner:
        async def classify_webhook_receipt(self, _: VerifiedWebhook) -> str:
            return "duplicate"

        def start_webhook_finalization(
            self,
            _event: VerifiedWebhook,
            _resolution: ResolvedWebhook,
            _receipt: object = "first",
        ) -> Handle:
            if failure == "finalizer-transfer":
                raise RuntimeError("private-finalizer-transfer")
            return Handle()

    async def duplicate_resolver(_event: VerifiedWebhook) -> object:
        if failure == "resolver-reject":
            raise CallAdmissionRejected("call_draining")
        if failure == "resolver-error":
            raise RuntimeError("private-resolver")
        if failure == "resolver-invalid":
            return object()
        return ResolvedWebhook(None)

    observed = await TelnyxWebhookProcessor(
        verifier=Verifier(),  # type: ignore[arg-type]
        resolver=lambda _: ResolvedWebhook(None),
        duplicate_resolver=duplicate_resolver,  # type: ignore[arg-type]
        finalizer_owner=Owner(),  # type: ignore[arg-type]
    ).process_observed(body=b"{}", headers=[])

    assert observed.disposition == WebhookDisposition(expected_status)
    assert observed.receipt == "duplicate"
    assert observed.admission_rejection is None
