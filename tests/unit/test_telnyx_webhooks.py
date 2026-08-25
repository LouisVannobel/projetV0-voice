from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from nacl.signing import SigningKey

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    FatalPersistenceError,
    PersistenceCommand,
)
from projetv0_voice.persistence.writer import PersistenceWriter

NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)
KEY = bytes(range(32))


def webhooks() -> Any:
    return importlib.import_module("projetv0_voice.telnyx.webhooks")


def event_body(
    *,
    event_id: str = "event-1",
    event_type: str = "call.initiated",
    occurred_at: str = "2026-08-25T14:00:00+02:00",
    payload: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> bytes:
    data: dict[str, object] = {
        "id": event_id,
        "event_type": event_type,
        "occurred_at": occurred_at,
        "payload": payload if payload is not None else {"call_control_id": "control-1"},
    }
    data.update(extra or {})
    return json.dumps({"data": data, "provider_extra": "ignored"}).encode()


def call_operation() -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(int=1),
        deployment_id="agent-a",
        call_id=UUID(int=2),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="control-1",
            telnyx_call_leg_id=None,
            telnyx_call_session_id=None,
            status="pending",
            disclosure_state="pending",
            started_at=None,
            ended_at=None,
            end_reason=None,
            retention_until=NOW + timedelta(days=7),
        ),
    )


def signing_fixture(
    body: bytes, *, timestamp: int = 1_777_118_400
) -> tuple[str, list[tuple[str, str]]]:
    signing_key = SigningKey.generate()
    public_key = base64.b64encode(bytes(signing_key.verify_key)).decode("ascii")
    signed = f"{timestamp}|".encode() + body
    signature = base64.b64encode(signing_key.sign(signed).signature).decode("ascii")
    return public_key, [
        ("Telnyx-Signature-Ed25519", signature),
        ("TELNYX-TIMESTAMP", str(timestamp)),
    ]


def verifier_for(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    *,
    timestamp: int = 1_777_118_400,
    required_types: frozenset[str] = frozenset(),
) -> tuple[Any, list[tuple[str, str]]]:
    module = webhooks()
    public_key, headers = signing_fixture(body, timestamp=timestamp)
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    monkeypatch.setattr(verification.time, "time", lambda: float(timestamp))
    return (
        module.TelnyxWebhookVerifier(
            public_key=public_key,
            call_control_required_types=required_types,
        ),
        headers,
    )


@pytest.mark.parametrize(
    "configured_key",
    ["", "RAW-PUBLIC-KEY-SENTINEL", base64.b64encode(b"x" * 31).decode("ascii")],
)
def test_verifier_preflights_public_key_without_reflecting_configuration(
    configured_key: str,
) -> None:
    module = webhooks()

    with pytest.raises(module.WebhookConfigurationError, match="webhook_config_invalid") as raised:
        module.TelnyxWebhookVerifier(public_key=configured_key)

    assert "RAW-PUBLIC-KEY-SENTINEL" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_missing_verification_dependency_is_constant_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    public_key = base64.b64encode(b"x" * 32).decode("ascii")

    def missing_dependency(_: str) -> Any:
        raise ModuleNotFoundError("RAW-MISSING-DEPENDENCY-SENTINEL")

    monkeypatch.setattr(module.importlib, "import_module", missing_dependency)

    with pytest.raises(module.WebhookConfigurationError, match="webhook_config_invalid") as raised:
        module.TelnyxWebhookVerifier(public_key=public_key)

    assert "RAW-MISSING-DEPENDENCY-SENTINEL" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("missing", ["module", "symbol"])
def test_verifier_preflights_pynacl_when_telnyx_helper_import_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    module = webhooks()
    public_key = base64.b64encode(b"x" * 32).decode("ascii")
    real_import = module.importlib.import_module
    verification = real_import("telnyx.lib.webhook_verification")

    def import_with_missing_signing(name: str) -> Any:
        if name == "telnyx.lib.webhook_verification":
            return verification
        if name == "nacl.signing":
            if missing == "module":
                raise ModuleNotFoundError("RAW-MISSING-PYNACL-SENTINEL")
            return SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", import_with_missing_signing)

    with pytest.raises(module.WebhookConfigurationError, match="webhook_config_invalid") as raised:
        module.TelnyxWebhookVerifier(public_key=public_key)

    assert "RAW-MISSING-PYNACL-SENTINEL" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_verifier_constructs_verify_key_from_strict_decoded_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    decoded_key = b"x" * 32
    public_key = base64.b64encode(decoded_key).decode("ascii")
    real_import = module.importlib.import_module
    verification = real_import("telnyx.lib.webhook_verification")
    preflighted: list[bytes] = []

    def verify_key(value: bytes) -> object:
        preflighted.append(value)
        return object()

    def import_with_preflight(name: str) -> Any:
        if name == "telnyx.lib.webhook_verification":
            return verification
        if name == "nacl.signing":
            return SimpleNamespace(VerifyKey=verify_key)
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", import_with_preflight)

    module.TelnyxWebhookVerifier(public_key=public_key)

    assert preflighted == [decoded_key]


@pytest.mark.parametrize("offset", [-301, -300, 0, 300, 301])
def test_real_ed25519_verification_enforces_exact_timestamp_boundary(
    monkeypatch: pytest.MonkeyPatch, offset: int
) -> None:
    module = webhooks()
    now = 1_777_118_400
    body = event_body()
    verifier, headers = verifier_for(monkeypatch, body, timestamp=now + offset)
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    monkeypatch.setattr(verification.time, "time", lambda: float(now))

    if abs(offset) <= 300:
        verified = verifier.verify(body=body, headers=headers)
        assert verified.event_id == "event-1"
        assert verified.occurred_at == NOW
    else:
        with pytest.raises(module.InvalidWebhookSignature, match="invalid_signature"):
            verifier.verify(body=body, headers=headers)


@pytest.mark.parametrize("mutation", ["tamper", "missing", "duplicate", "empty", "invalid_utf8"])
def test_signature_boundary_rejects_tamper_and_structural_headers_without_leaks(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    module = webhooks()
    body = event_body(event_id="RAW-EVENT-SENTINEL")
    verifier, headers = verifier_for(monkeypatch, body)
    candidate_body = body
    candidate_headers = list(headers)
    if mutation == "tamper":
        candidate_body = body.replace(b"call.initiated", b"call.answered")
    elif mutation == "missing":
        candidate_headers = candidate_headers[1:]
    elif mutation == "duplicate":
        candidate_headers.append(("telnyx-signature-ed25519", candidate_headers[0][1]))
    elif mutation == "empty":
        candidate_headers[0] = (candidate_headers[0][0], "")
    else:
        candidate_body = b"\xff"
        public_key, candidate_headers = signing_fixture(candidate_body)
        verifier = module.TelnyxWebhookVerifier(public_key=public_key)

    with pytest.raises(module.InvalidWebhookSignature, match="invalid_signature") as raised:
        verifier.verify(body=candidate_body, headers=candidate_headers)

    rendered = repr(raised.value)
    assert "RAW-EVENT-SENTINEL" not in rendered
    if candidate_headers[0][1]:
        assert candidate_headers[0][1] not in rendered
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_verification_precedes_json_and_oversize_precedes_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    body = event_body()
    verifier, headers = verifier_for(monkeypatch, body)
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    calls: list[bytes] = []

    def fail_after_recording(payload: bytes, *_: object) -> None:
        calls.append(payload)
        raise verification.WebhookVerificationError("RAW-UPSTREAM-SENTINEL")

    monkeypatch.setattr(verification, "verify_webhook_signature", fail_after_recording)
    with pytest.raises(module.InvalidWebhookSignature):
        verifier.verify(body=b"not-json", headers=headers)
    assert calls == [b"not-json"]

    calls.clear()
    with pytest.raises(module.WebhookBodyTooLarge, match="body_too_large"):
        verifier.verify(body=b"x" * 65_537, headers=headers)
    assert calls == []


def test_exact_64kib_body_is_admitted_and_header_values_are_not_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    base = event_body()
    body = base + b" " * (65_536 - len(base))
    timestamp = " 1777118400 "
    signing_key = SigningKey.generate()
    public_key = base64.b64encode(bytes(signing_key.verify_key)).decode("ascii")
    signature = base64.b64encode(
        signing_key.sign(timestamp.encode() + b"|" + body).signature
    ).decode("ascii")
    headers = [
        ("Telnyx-Signature-Ed25519", signature),
        ("Telnyx-Timestamp", timestamp),
    ]
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    monkeypatch.setattr(verification.time, "time", lambda: 1_777_118_400.0)

    verified = module.TelnyxWebhookVerifier(public_key=public_key).verify(
        body=body,
        headers=headers,
    )

    assert len(body) == 65_536
    assert verified.event_id == "event-1"


def test_fresh_signature_accepts_old_occurred_at_without_using_it_for_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = event_body(occurred_at="2020-01-01T00:00:00Z")
    verifier, headers = verifier_for(monkeypatch, body)

    verified = verifier.verify(body=body, headers=headers)

    assert verified.occurred_at == datetime(2020, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "body",
    [
        event_body(event_id="x" * 257),
        event_body(event_type="x" * 129),
        event_body(payload={"call_control_id": "x" * 257}),
    ],
)
def test_minimal_envelope_identifiers_have_fixed_structural_bounds(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    module = webhooks()
    verifier, headers = verifier_for(monkeypatch, body)

    with pytest.raises(module.InvalidWebhookPayload, match="invalid_payload"):
        verifier.verify(body=body, headers=headers)


@pytest.mark.parametrize(
    "body",
    [
        b'{"data":{"id":"a","id":"b","event_type":"x","occurred_at":"2026-08-25T12:00:00Z","payload":{}}}',
        b'{"data":{"id":"a","event_type":"x","occurred_at":"2026-08-25T12:00:00Z","payload":{"nested":{"x":1,"x":2}}}}',
        b'{"data":{"id":"a","event_type":"x","occurred_at":"2026-08-25T12:00:00Z","payload":{"value":NaN}}}',
        b'{"data":{"id":"a","event_type":"x","occurred_at":"not-a-time","payload":{}}}',
    ],
)
def test_signed_payload_requires_strict_recursive_json_and_envelope(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    module = webhooks()
    verifier, headers = verifier_for(monkeypatch, body)

    with pytest.raises(module.InvalidWebhookPayload, match="invalid_payload") as raised:
        verifier.verify(body=body, headers=headers)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_verified_webhook_is_canonical_minimal_frozen_and_input_redacting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = event_body(
        event_id="RAW-EVENT-SENTINEL",
        event_type="unknown.future",
        occurred_at="2026-08-25T14:00:00+02:00",
        payload={
            "call_leg_id": "leg-1",
            "arbitrary": "RAW-PAYLOAD-SENTINEL",
        },
        extra={"arbitrary_nested": {"secret": "RAW-NESTED-SENTINEL"}},
    )
    verifier, headers = verifier_for(monkeypatch, body)

    verified = verifier.verify(body=body, headers=headers)

    assert verified.event_id == "RAW-EVENT-SENTINEL"
    assert verified.event_type == "unknown.future"
    assert verified.occurred_at == NOW
    assert verified.call_control_id is None
    assert verified.call_leg_id == "leg-1"
    assert not hasattr(verified, "payload")
    rendered = repr(verified)
    for sentinel in ("RAW-EVENT-SENTINEL", "RAW-PAYLOAD-SENTINEL", "RAW-NESTED-SENTINEL"):
        assert sentinel not in rendered
    with pytest.raises((AttributeError, TypeError)):
        verified.event_id = "changed"  # type: ignore[misc]


def test_explicit_handled_type_may_require_call_control_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    body = event_body(event_type="handled.event", payload={})
    verifier, headers = verifier_for(
        monkeypatch,
        body,
        required_types=frozenset({"handled.event"}),
    )

    with pytest.raises(module.InvalidWebhookPayload, match="invalid_payload"):
        verifier.verify(body=body, headers=headers)


def test_durable_effect_rejects_invalid_lease_mapping_without_reflecting_it() -> None:
    module = webhooks()

    with pytest.raises(TypeError, match="invalid_webhook_effect") as raised:
        module.WebhookDurableEffect(lease={"raw": "RAW-LEASE-SENTINEL"})

    assert "RAW-LEASE-SENTINEL" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_unexpected_helper_error_is_constant_internal_failure_not_signature_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    body = event_body()
    verifier, headers = verifier_for(monkeypatch, body)
    verification = importlib.import_module("telnyx.lib.webhook_verification")

    def explode(*_: object) -> None:
        raise RuntimeError("RAW-HELPER-SENTINEL")

    monkeypatch.setattr(verification, "verify_webhook_signature", explode)
    with pytest.raises(module.WebhookInternalError, match="webhook_internal_error") as raised:
        verifier.verify(body=body, headers=headers)

    assert not isinstance(raised.value, module.InvalidWebhookSignature)
    assert "RAW-HELPER-SENTINEL" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


class StubWriter:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.commands: list[PersistenceCommand] = []

    async def commit_control(self, command: PersistenceCommand) -> None:
        self.commands.append(command)
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_processor_commits_one_atomic_receipt_only_effect_before_empty_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = webhooks()
    body = event_body(event_type="unknown.future", payload={})
    verifier, headers = verifier_for(monkeypatch, body)
    writer = StubWriter()
    resolved: list[Any] = []

    def resolver(event: Any) -> None:
        resolved.append(event)
        return None

    processor = module.TelnyxWebhookProcessor(
        verifier=verifier,
        writer=writer,
        resolver=resolver,
        utcnow=lambda: NOW + timedelta(seconds=1),
    )

    response = await processor.process(body=body, headers=headers)

    assert (response.status_code, response.body) == (200, b"")
    assert len(resolved) == 1
    assert len(writer.commands) == 1
    command = writer.commands[0]
    assert command.kind == "webhook_effect"
    assert command.payload == {
        "receipt": {
            "event_id": "event-1",
            "event_type": "unknown.future",
            "call_control_id": None,
            "occurred_at": NOW,
            "received_at": NOW + timedelta(seconds=1),
        },
        "lease": None,
        "operation": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("body", 413),
        ("signature", 403),
        ("payload", 400),
        ("internal", 500),
        ("conflict", 400),
        ("lease_conflict", 500),
        ("unavailable", 503),
        ("timeout", 503),
        ("resolver", 500),
    ],
)
async def test_processor_has_exact_empty_safe_status_matrix(source: str, expected: int) -> None:
    module = webhooks()
    verified = module.VerifiedWebhook(
        event_id="RAW-EVENT-SENTINEL",
        event_type="unknown.future",
        occurred_at=NOW,
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
    )

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            errors = {
                "body": module.WebhookBodyTooLarge("body_too_large"),
                "signature": module.InvalidWebhookSignature("invalid_signature"),
                "payload": module.InvalidWebhookPayload("invalid_payload"),
                "internal": module.WebhookInternalError("webhook_internal_error"),
            }
            if source in errors:
                raise errors[source]
            return verified

    writer_errors = {
        "conflict": CommandConflictError("webhook_identity_conflict"),
        "lease_conflict": CommandConflictError("lease_transition_conflict"),
        "unavailable": FatalPersistenceError("persistence_degraded"),
        "timeout": TimeoutError("RAW-TIMEOUT-SENTINEL"),
    }
    writer = StubWriter(writer_errors.get(source))

    def resolver(_: Any) -> None:
        if source == "resolver":
            raise RuntimeError("RAW-RESOLVER-SENTINEL")
        return None

    response = await module.TelnyxWebhookProcessor(
        verifier=StubVerifier(), writer=writer, resolver=resolver, utcnow=lambda: NOW
    ).process(body=b"RAW-BODY-SENTINEL", headers=[])

    assert (response.status_code, response.body) == (expected, b"")
    rendered = repr(response)
    for sentinel in ("RAW-EVENT-SENTINEL", "RAW-TIMEOUT-SENTINEL", "RAW-RESOLVER-SENTINEL"):
        assert sentinel not in rendered


@pytest.mark.asyncio
async def test_processor_rejects_async_or_invalid_resolver_without_committing() -> None:
    module = webhooks()
    event = module.VerifiedWebhook(
        event_id="event-1",
        event_type="unknown.future",
        occurred_at=NOW,
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
    )

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            return event

    async def async_resolver(_: Any) -> None:
        return None

    for resolver in (async_resolver, lambda _: SimpleNamespace(lease=None, operation=None)):
        writer = StubWriter()
        response = await module.TelnyxWebhookProcessor(
            verifier=StubVerifier(), writer=writer, resolver=resolver, utcnow=lambda: NOW
        ).process(body=b"{}", headers=[])
        assert (response.status_code, response.body) == (500, b"")
        assert writer.commands == []


async def start_writer(path: Path) -> tuple[PersistenceWriter, Any]:
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1))
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is True
    return writer, task


@pytest.mark.asyncio
async def test_identical_duplicate_discards_entire_different_candidate_effect(
    tmp_path: Path,
) -> None:
    writer, task = await start_writer(tmp_path / "different-candidate.sqlite")
    receipt = {
        "event_id": "event-1",
        "event_type": "unknown.future",
        "call_control_id": None,
        "occurred_at": NOW,
        "received_at": NOW,
    }
    first = PersistenceCommand(
        "webhook_effect",
        {"receipt": receipt, "lease": None, "operation": None},
        None,
    )
    deliberately_invalid_if_applied = PersistenceCommand(
        "webhook_effect",
        {
            "receipt": {**receipt, "received_at": NOW + timedelta(seconds=10)},
            "lease": {"deliberately": "different-candidate"},
            "operation": object(),
        },
        None,
    )

    await writer.commit_control(first)
    await writer.commit_control(deliberately_invalid_if_applied)
    await writer.drain(timeout_seconds=2)
    await task

    with sqlite3.connect(tmp_path / "different-candidate.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)


@pytest.mark.asyncio
async def test_concurrent_duplicate_processor_calls_commit_one_durable_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = webhooks()
    database = tmp_path / "concurrent-duplicate.sqlite"
    writer, writer_task = await start_writer(database)
    body = event_body(
        event_type="handled.event",
        payload={
            "call_control_id": "control-1",
            "arbitrary": "RAW-UNPERSISTED-PAYLOAD-SENTINEL",
        },
    )
    verifier, headers = verifier_for(
        monkeypatch,
        body,
        required_types=frozenset({"handled.event"}),
    )
    effect = module.WebhookDurableEffect(
        lease={
            "action": "upsert",
            "call_control_id": "control-1",
            "call_id": UUID(int=2),
            "tenant_id": "tenant-a",
            "agent_id": "agent-a",
            "state": "pending",
            "token_hash": bytes(range(32)),
            "created_at": NOW,
            "expires_at": NOW + timedelta(seconds=30),
            "closed_at": None,
        },
        operation=call_operation(),
    )
    processor = module.TelnyxWebhookProcessor(
        verifier=verifier,
        writer=writer,
        resolver=lambda _: effect,
        utcnow=lambda: NOW,
    )

    first, second = await asyncio.gather(
        processor.process(body=body, headers=headers),
        processor.process(body=body, headers=headers),
    )

    assert (first.status_code, second.status_code) == (200, 200)
    await writer.drain(timeout_seconds=2)
    await writer_task
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
        stored = repr(connection.execute("SELECT * FROM webhook_receipts").fetchall())
    assert "RAW-UNPERSISTED-PAYLOAD-SENTINEL" not in stored


@pytest.mark.asyncio
async def test_real_writer_late_commit_requires_restart_then_redelivery_is_duplicate_200(
    tmp_path: Path,
) -> None:
    module = webhooks()
    database = tmp_path / "late-webhook.sqlite"
    commit_blocked = asyncio.Event()
    release_commit = asyncio.Event()
    event = module.VerifiedWebhook(
        event_id="event-1",
        event_type="handled.event",
        occurred_at=NOW,
        call_control_id="control-1",
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
    )

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            return event

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            commit_blocked.set()
            await release_commit.wait()

    effect = module.WebhookDurableEffect(
        lease={
            "action": "upsert",
            "call_control_id": "control-1",
            "call_id": UUID(int=2),
            "tenant_id": "tenant-a",
            "agent_id": "agent-a",
            "state": "pending",
            "token_hash": bytes(range(32)),
            "created_at": NOW,
            "expires_at": NOW + timedelta(seconds=30),
            "closed_at": None,
        },
        operation=call_operation(),
    )
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        control_commit_timeout_seconds=0.01,
        failpoint=failpoint,
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is True
    processor = module.TelnyxWebhookProcessor(
        verifier=StubVerifier(),
        writer=writer,
        resolver=lambda _: effect,
        utcnow=lambda: NOW,
    )

    first_task = asyncio.create_task(processor.process(body=b"{}", headers=[]))
    await commit_blocked.wait()
    first = await first_task
    assert (first.status_code, first.body) == (503, b"")
    assert writer.is_degraded is True

    release_commit.set()
    await writer.wait_until_idle()
    same_process = await processor.process(body=b"{}", headers=[])
    assert (same_process.status_code, same_process.body) == (503, b"")
    await writer.drain(timeout_seconds=2)
    await writer_task

    restarted, restarted_task = await start_writer(database)
    restarted_processor = module.TelnyxWebhookProcessor(
        verifier=StubVerifier(),
        writer=restarted,
        resolver=lambda _: effect,
        utcnow=lambda: NOW + timedelta(seconds=1),
    )

    redelivery = await restarted_processor.process(body=b"{}", headers=[])
    assert (redelivery.status_code, redelivery.body) == (200, b"")
    await restarted.drain(timeout_seconds=2)
    await restarted_task

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
