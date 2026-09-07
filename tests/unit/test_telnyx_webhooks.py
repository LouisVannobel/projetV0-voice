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
from pydantic import SecretStr

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    FatalPersistenceError,
    PersistenceCommand,
    PersistenceError,
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
    selected_payload = dict(payload) if payload is not None else {"call_control_id": "control-1"}
    if event_type == "call.initiated":
        selected_payload.setdefault("direction", "incoming")
        selected_payload.setdefault("state", "parked")
    elif event_type == "call.answered":
        selected_payload.setdefault("state", "answered")
    data: dict[str, object] = {
        "id": event_id,
        "event_type": event_type,
        "occurred_at": occurred_at,
        "payload": selected_payload,
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
    ],
)
def test_minimal_envelope_identifiers_have_fixed_structural_bounds(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    module = webhooks()
    verifier, headers = verifier_for(monkeypatch, body)

    with pytest.raises(module.InvalidWebhookPayload, match="invalid_payload"):
        verifier.verify(body=body, headers=headers)


@pytest.mark.parametrize("length", [257, 1024])
@pytest.mark.parametrize("required", [False, True])
def test_signed_call_control_id_preserves_1024_bound_and_redaction(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    required: bool,
) -> None:
    call_control_id = "c" * length
    event_type = "call.initiated" if required else "call.recording.saved"
    body = event_body(
        event_type=event_type,
        payload={"call_control_id": call_control_id},
    )
    verifier, headers = verifier_for(
        monkeypatch,
        body,
        required_types=frozenset({event_type}) if required else frozenset(),
    )

    verified = verifier.verify(body=body, headers=headers)

    assert verified.call_control_id == call_control_id
    assert call_control_id not in repr(verified)


@pytest.mark.parametrize("required", [False, True])
def test_signed_call_control_id_rejects_1025_characters(
    monkeypatch: pytest.MonkeyPatch,
    required: bool,
) -> None:
    event_type = "call.initiated" if required else "call.recording.saved"
    body = event_body(
        event_type=event_type,
        payload={"call_control_id": "c" * 1025},
    )
    verifier, headers = verifier_for(
        monkeypatch,
        body,
        required_types=frozenset({event_type}) if required else frozenset(),
    )

    with pytest.raises(webhooks().InvalidWebhookPayload, match="invalid_payload"):
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


def test_current_recording_saved_shape_extracts_only_url_free_semantic_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = event_body(
        event_type="call.recording.saved",
        payload={
            "call_leg_id": "leg-1",
            "call_session_id": "session-1",
            "client_state": "RAW-CLIENT-STATE-SENTINEL",
            "recording_started_at": "2026-08-25T11:59:00Z",
            "recording_ended_at": "2026-08-25T12:00:00Z",
            "channels": "dual",
            "public_recording_urls": {"wav": "https://RAW-URL-SENTINEL"},
        },
    )
    verifier, headers = verifier_for(monkeypatch, body)

    verified = verifier.verify(body=body, headers=headers)

    assert verified.call_control_id is None
    assert verified.recording_id is None
    assert verified.recording_started_at == NOW - timedelta(minutes=1)
    assert verified.recording_ended_at == NOW
    assert verified.recording_channels == "dual"
    assert isinstance(verified.client_state, SecretStr)
    assert verified.client_state.get_secret_value() == "RAW-CLIENT-STATE-SENTINEL"
    assert len(verified.semantic_fingerprint_sha256) == 32
    rendered = repr(verified)
    for sentinel in (
        "RAW-CLIENT-STATE-SENTINEL",
        "RAW-URL-SENTINEL",
        "leg-1",
        "session-1",
    ):
        assert sentinel not in rendered
    assert not hasattr(verified, "public_recording_urls")


def test_recording_error_shape_ignores_reason_and_semantic_fingerprint_ignores_retry_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def body(*, attempt: int, reason: str, url: str) -> bytes:
        return event_body(
            event_type="call.recording.error",
            payload={
                "client_state": "RAW-CLIENT-STATE-SENTINEL",
                "reason": reason,
                "public_recording_urls": {"wav": url},
            },
            extra={"meta": {"attempt": attempt}},
        )

    first_body = body(attempt=1, reason="first", url="https://one.invalid")
    second_body = body(attempt=9, reason="changed", url="https://two.invalid")
    first_verifier, first_headers = verifier_for(monkeypatch, first_body)
    first = first_verifier.verify(body=first_body, headers=first_headers)
    second_verifier, second_headers = verifier_for(monkeypatch, second_body)
    second = second_verifier.verify(body=second_body, headers=second_headers)

    assert first.semantic_fingerprint_sha256 == second.semantic_fingerprint_sha256
    assert first.recording_started_at is None
    assert first.recording_ended_at is None
    assert first.recording_channels is None
    assert not hasattr(first, "reason")


def test_semantic_fingerprint_changes_when_an_effect_driving_field_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprints: list[bytes] = []
    for client_state, ended_at in (
        ("capsule-a", "2026-08-25T12:00:00Z"),
        ("capsule-b", "2026-08-25T12:00:00Z"),
        ("capsule-a", "2026-08-25T12:00:01Z"),
    ):
        body = event_body(
            event_type="call.recording.saved",
            payload={
                "client_state": client_state,
                "recording_started_at": "2026-08-25T11:59:00Z",
                "recording_ended_at": ended_at,
                "channels": "dual",
            },
        )
        verifier, headers = verifier_for(monkeypatch, body)
        fingerprints.append(
            verifier.verify(body=body, headers=headers).semantic_fingerprint_sha256
        )

    assert len(set(fingerprints)) == 3


def test_signed_recording_id_rejects_url_shape_before_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = event_body(
        event_type="call.recording.saved",
        payload={"recording_id": "https://RAW-URL-SENTINEL"},
    )
    verifier, headers = verifier_for(monkeypatch, body)

    with pytest.raises(webhooks().InvalidWebhookPayload, match="invalid_payload"):
        verifier.verify(body=body, headers=headers)


def test_signed_recording_id_accepts_256_url_safe_opaque_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_id = "r." + "A" * 251 + "~_-"
    body = event_body(
        event_type="call.recording.saved",
        payload={"recording_id": provider_id},
    )
    verifier, headers = verifier_for(monkeypatch, body)

    verified = verifier.verify(body=body, headers=headers)

    assert verified.recording_id == provider_id


@pytest.mark.parametrize("provider_id", [".", ".."])
def test_signed_recording_id_rejects_exact_dot_segments(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
) -> None:
    body = event_body(
        event_type="call.recording.saved",
        payload={"recording_id": provider_id},
    )
    verifier, headers = verifier_for(monkeypatch, body)

    with pytest.raises(webhooks().InvalidWebhookPayload, match="invalid_payload"):
        verifier.verify(body=body, headers=headers)


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


class _FinalizationHandle:
    def __init__(self, task: asyncio.Task[Any]) -> None:
        self.task = task

    async def wait(self) -> Any:
        return await self.task


class FakeFinalizerOwner:
    def __init__(
        self,
        *,
        module: Any,
        writer: Any,
        utcnow: Any,
        after_commit: Any = None,
    ) -> None:
        self.module = module
        self.writer = writer
        self.utcnow = utcnow
        self.after_commit = after_commit
        self.tasks: list[asyncio.Task[Any]] = []

    async def classify_webhook_receipt(self, event: Any) -> str:
        if isinstance(self.writer, PersistenceWriter):
            return await self.writer.classify_webhook_receipt(
                event_id=event.event_id,
                semantic_fingerprint_sha256=event.semantic_fingerprint_sha256,
            )
        return "missing"

    def start_webhook_finalization(
        self,
        event: Any,
        resolution: Any,
        _receipt: str,
    ) -> _FinalizationHandle:
        task = asyncio.create_task(self._run(event, resolution))
        self.tasks.append(task)
        return _FinalizationHandle(task)

    async def _run(self, event: Any, resolution: Any) -> Any:
        effect = resolution.effect
        receipt = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "call_control_id": event.call_control_id,
            "occurred_at": event.occurred_at,
            "received_at": self.utcnow(),
            "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
        }
        try:
            if isinstance(self.writer, PersistenceWriter):
                ticket = self.writer.submit_webhook(
                    receipt=receipt,
                    lease=None if effect is None else effect.lease,
                    operation=None if effect is None else effect.operation,
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(ticket.wait()),
                        timeout=self.writer.control_commit_timeout_seconds,
                    )
                except TimeoutError:
                    self.writer._signal_fatal("control_commit_timeout")  # type: ignore[attr-defined]
                    return self.module.WebhookDisposition(503)
            else:
                await self.writer.commit_control(
                    PersistenceCommand(
                        "webhook_effect",
                        {
                            "receipt": receipt,
                            "lease": None if effect is None else effect.lease,
                            "operation": None if effect is None else effect.operation,
                        },
                        None,
                    )
                )
        except CommandConflictError as error:
            return self.module.WebhookDisposition(
                400 if error.args == ("webhook_identity_conflict",) else 500
            )
        except (PersistenceError, TimeoutError):
            return self.module.WebhookDisposition(503)
        except Exception:
            return self.module.WebhookDisposition(500)
        if self.after_commit is not None:
            disposition = await self.after_commit(event, effect)
            if disposition is not None:
                return disposition
        return self.module.WebhookDisposition(200)


def processor_with_fake_owner(
    *,
    module: Any,
    verifier: Any,
    writer: Any,
    resolver: Any,
    utcnow: Any,
    after_commit: Any = None,
) -> Any:
    def wrapped(event: Any) -> Any:
        value = resolver(event)
        if value is None:
            return module.ResolvedWebhook(None)
        if isinstance(value, module.WebhookDurableEffect):
            return module.ResolvedWebhook(value)
        return value

    return module.TelnyxWebhookProcessor(
        verifier=verifier,
        resolver=wrapped,
        duplicate_resolver=wrapped,
        finalizer_owner=FakeFinalizerOwner(
            module=module,
            writer=writer,
            utcnow=utcnow,
            after_commit=after_commit,
        ),
    )


@pytest.mark.asyncio
async def test_processor_awaits_after_commit_and_accepts_only_recording_dispositions() -> None:
    module = webhooks()
    event = module.VerifiedWebhook(
        event_id="event-1",
        event_type="call.recording.saved",
        occurred_at=NOW,
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
        client_state=SecretStr("capsule"),
        recording_started_at=NOW - timedelta(minutes=1),
        recording_ended_at=NOW,
        recording_channels="dual",
        semantic_fingerprint_sha256=b"s" * 32,
    )
    order: list[str] = []

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            return event

    class OrderedWriter(StubWriter):
        async def commit_control(self, command: PersistenceCommand) -> None:
            await super().commit_control(command)
            order.append("committed")

    async def after_commit(received: Any, effect: Any) -> Any:
        assert received is event
        assert effect is None
        order.append("after_commit")
        return module.WebhookDisposition(503)

    response = await processor_with_fake_owner(
        module=module,
        verifier=StubVerifier(),
        writer=OrderedWriter(),
        resolver=lambda _: None,
        utcnow=lambda: NOW,
        after_commit=after_commit,
    ).process(body=b"{}", headers=[])

    assert response == module.WebhookDisposition(503)
    assert order == ["committed", "after_commit"]


@pytest.mark.asyncio
async def test_processor_preserves_the_closed_qualification_rejection_reason() -> None:
    module = webhooks()
    event = module.VerifiedWebhook(
        event_id="qualification-consumed",
        event_type="call.initiated",
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
        semantic_fingerprint_sha256=b"q" * 32,
        direction="incoming",
        call_state="parked",
    )

    class Handle:
        async def wait(self) -> Any:
            return module.WebhookDisposition(
                503,
                admission_rejection="qualification",
            )

    class Owner:
        async def classify_webhook_receipt(self, _event: Any) -> str:
            return "missing"

        def start_webhook_finalization(self, *_args: object) -> Handle:
            return Handle()

    class Verifier:
        def verify(self, **_kwargs: object) -> Any:
            return event

    processor = module.TelnyxWebhookProcessor(
        verifier=Verifier(),
        resolver=lambda _event: module.ResolvedWebhook(None),
        finalizer_owner=Owner(),
    )

    observed = await processor.process_observed(body=b"{}", headers=[])

    assert observed.disposition == module.WebhookDisposition(503)
    assert observed.webhook_class == "initiated"
    assert observed.receipt == "first"
    assert observed.metric_disposition == "unavailable"
    assert observed.admission_rejection == "qualification"


def test_webhook_disposition_rejects_dynamic_admission_rejection() -> None:
    module = webhooks()

    with pytest.raises(ValueError, match="^webhook_disposition_invalid$"):
        module.WebhookDisposition(503, admission_rejection="PRIVATE-REASON")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("classification", "expected_receipt"),
    [("missing", "first"), ("duplicate", "duplicate")],
)
async def test_processor_transfers_closed_receipt_classification_to_finalizer(
    classification: str,
    expected_receipt: str,
) -> None:
    module = webhooks()
    event = module.VerifiedWebhook(
        event_id="receipt-classification",
        event_type="future.event",
        occurred_at=NOW,
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"r" * 32,
    )
    received: list[str] = []

    class Handle:
        async def wait(self) -> Any:
            return module.WebhookDisposition(200)

    class Owner:
        async def classify_webhook_receipt(self, _event: Any) -> str:
            return classification

        def start_webhook_finalization(
            self,
            _event: Any,
            _resolution: Any,
            receipt: str,
        ) -> Handle:
            received.append(receipt)
            return Handle()

    class Verifier:
        def verify(self, **_kwargs: object) -> Any:
            return event

    processor = module.TelnyxWebhookProcessor(
        verifier=Verifier(),
        resolver=lambda _event: module.ResolvedWebhook(None),
        duplicate_resolver=lambda _event: module.ResolvedWebhook(None),
        finalizer_owner=Owner(),
    )

    observed = await processor.process_observed(body=b"{}", headers=[])

    assert observed.disposition == module.WebhookDisposition(200)
    assert observed.receipt == expected_receipt
    assert received == [expected_receipt]


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

    processor = processor_with_fake_owner(
        module=module,
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
            "semantic_fingerprint_sha256": resolved[0].semantic_fingerprint_sha256,
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
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"s" * 32,
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

    response = await processor_with_fake_owner(
        module=module,
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
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"s" * 32,
    )

    class StubVerifier:
        def verify(self, **_: object) -> Any:
            return event

    async def async_resolver(_: Any) -> None:
        return None

    for resolver in (async_resolver, lambda _: SimpleNamespace(lease=None, operation=None)):
        writer = StubWriter()
        response = await processor_with_fake_owner(
            module=module,
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
        "semantic_fingerprint_sha256": b"s" * 32,
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
    processor = processor_with_fake_owner(
        module=module,
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
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"s" * 32,
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
    processor = processor_with_fake_owner(
        module=module,
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
    restarted_processor = processor_with_fake_owner(
        module=module,
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
