from __future__ import annotations

import asyncio
import base64
import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from nacl.signing import SigningKey
from pydantic import SecretStr

import projetv0_voice.telnyx.call_control as call_control_module
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    PersistenceCommand,
    canonical_operation_bytes,
)
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.session import CallIdentity
from projetv0_voice.telnyx.call_control import (
    CallControlClient,
    CallControlResult,
    RecordingCatalogTransientError,
)
from projetv0_voice.telnyx.recordings import (
    ProviderRecordingPageV1,
    ProviderRecordingV1,
    after_recording_webhook_commit,
    build_recording_correlation,
    encode_recording_correlation,
    resolve_recording_webhook,
)
from projetv0_voice.telnyx.webhooks import (
    ResolvedWebhook,
    TelnyxWebhookProcessor,
    TelnyxWebhookVerifier,
    VerifiedWebhook,
    WebhookDisposition,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
CALL_ID = UUID("12345678-1234-4234-8234-1234567890ab")
KEY = bytes(range(32))


def identity(call_control_id: str = "v3:control-1") -> CallIdentity:
    return CallIdentity(
        call_id=CALL_ID,
        durable_generation="generation-ignored",
        lease_identity="lease-ignored",
        lease_claim=object(),
        deployment_id="agent-révision-7",
        registry_handle=object(),
        telnyx_call_control_id=call_control_id,
        telnyx_call_leg_id="leg-1",
        telnyx_call_session_id="session-1",
        stream_id="stream-ignored",
        started_at=NOW,
        retention_until=NOW + timedelta(days=30),
    )


def saved_event() -> VerifiedWebhook:
    correlation = build_recording_correlation(
        identity(), retention_days=30, required=False
    )
    return VerifiedWebhook(
        event_id="event-saved-1",
        event_type="call.recording.saved",
        occurred_at=NOW + timedelta(minutes=2),
        call_control_id=None,
        call_leg_id="leg-1",
        call_session_id="session-1",
        recording_id=None,
        stream_id=None,
        client_state=encode_recording_correlation(correlation),
        recording_started_at=NOW,
        recording_ended_at=NOW + timedelta(minutes=1),
        recording_channels="dual",
        semantic_fingerprint_sha256=b"f" * 32,
    )


def catalog_item(**updates: object) -> ProviderRecordingV1:
    values: dict[str, object] = {
        "recording_id": "recording_Ab-12",
        "call_control_id": "v3:control-1",
        "call_leg_id": "leg-1",
        "call_session_id": "session-1",
        "channels": "dual",
        "status": "completed",
        "source": "call",
        "initiated_by": "StartCallRecordingAPI",
        "recording_started_at": NOW,
        "recording_ended_at": NOW + timedelta(minutes=1),
    }
    values.update(updates)
    return ProviderRecordingV1(**values)  # type: ignore[arg-type]


class CatalogApi:
    def __init__(
        self,
        page: ProviderRecordingPageV1 | BaseException,
        *,
        retrieve: ProviderRecordingV1 | BaseException | None = None,
    ) -> None:
        self.page = page
        self.retrieve = retrieve
        self.list_calls: list[dict[str, object]] = []
        self.retrieve_calls: list[tuple[str, float]] = []

    async def list_recordings_one_page(self, **kwargs: object) -> ProviderRecordingPageV1:
        self.list_calls.append(dict(kwargs))
        if isinstance(self.page, BaseException):
            raise self.page
        return self.page

    async def retrieve_recording(
        self, recording_id: str, *, timeout_seconds: float
    ) -> ProviderRecordingV1:
        self.retrieve_calls.append((recording_id, timeout_seconds))
        if isinstance(self.retrieve, BaseException):
            raise self.retrieve
        assert self.retrieve is not None
        return self.retrieve

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult:
        del call_control_id, command_id, client_state
        return CallControlResult("accepted")


class CapturingWriter:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.commands: list[PersistenceCommand] = []

    async def commit_control(self, command: PersistenceCommand) -> None:
        self.commands.append(command)
        if self.error is not None:
            raise self.error


async def no_drain(_call_id: UUID, _reason: str) -> None:
    return None


@pytest.mark.asyncio
async def test_saved_without_provider_id_reconciles_one_exact_first_page_and_enqueues() -> None:
    event = saved_event()
    effect = resolve_recording_webhook(event)
    api = CatalogApi(ProviderRecordingPageV1(1, 1, (catalog_item(),)))
    writer = CapturingWriter()

    result = await after_recording_webhook_commit(
        event,
        effect,
        telnyx=api,
        writer=writer,
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    )

    assert result == WebhookDisposition(200)
    assert api.list_calls == [
        {
            "call_control_id": "v3:control-1",
            "call_leg_id": "leg-1",
            "call_session_id": "session-1",
            "start_gte_iso": "2026-08-29T11:59:55Z",
            "start_lte_iso": "2026-08-29T12:00:05Z",
            "end_gte_iso": "2026-08-29T12:00:55Z",
            "end_lte_iso": "2026-08-29T12:01:05Z",
            "timeout_seconds": 1.0,
        }
    ]
    assert api.retrieve_calls == []
    assert len(writer.commands) == 1
    command = writer.commands[0]
    assert command.kind == "webhook_enrichment"
    assert command.payload["enrichment_fingerprint_sha256"] == (
        b"4R\n~V&\xd1cZ\xc0\xe0\\\xf3\xfb\xbd`"
        b"B\xdf\xdc\xa7\xc1\x19\xfa\xba\x91\x9fL\xd0\xcd6}\xef"
    )
    operation = command.payload["operation"]
    assert operation.operation_id == UUID("b10b98ee-616c-50d0-81e4-af5d8762e082")
    assert operation.occurred_at == event.occurred_at
    assert operation.payload.telnyx_recording_id == "recording_Ab-12"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (ProviderRecordingPageV1(1, 1, ()), 503),
        (
            ProviderRecordingPageV1(
                1,
                1,
                (catalog_item(), catalog_item(recording_id="recording_Other-2")),
            ),
            500,
        ),
        (ProviderRecordingPageV1(None, 1, (catalog_item(),)), 500),
        (ProviderRecordingPageV1(2, 2, (catalog_item(),)), 500),
        (ProviderRecordingPageV1(1, 1, (catalog_item(channels="single"),)), 500),
        (ProviderRecordingPageV1(1, 1, (catalog_item(status="processing"),)), 500),
        (ProviderRecordingPageV1(1, 1, (catalog_item(source="conference"),)), 500),
        (ProviderRecordingPageV1(1, 1, (catalog_item(initiated_by="DialVerb"),)), 500),
        (ProviderRecordingPageV1(1, 1, (catalog_item(call_leg_id="wrong"),)), 500),
        (
            ProviderRecordingPageV1(
                1,
                1,
                (catalog_item(recording_started_at=NOW - timedelta(seconds=6)),),
            ),
            500,
        ),
        (ProviderRecordingPageV1(1, 1, (catalog_item(recording_id=None),)), 500),
    ],
)
async def test_reconciliation_rejects_ambiguity_and_retries_only_not_indexed(
    page: ProviderRecordingPageV1, expected: int
) -> None:
    event = saved_event()
    api = CatalogApi(page)

    result = await after_recording_webhook_commit(
        event,
        resolve_recording_webhook(event),
        telnyx=api,
        writer=CapturingWriter(),
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    )

    assert result == WebhookDisposition(expected)
    assert api.retrieve_calls == []


@pytest.mark.asyncio
async def test_missing_scalar_with_valid_id_allows_exactly_one_full_retrieve() -> None:
    event = saved_event()
    api = CatalogApi(
        ProviderRecordingPageV1(1, 1, (catalog_item(source=None),)),
        retrieve=catalog_item(),
    )
    writer = CapturingWriter()

    result = await after_recording_webhook_commit(
        event,
        resolve_recording_webhook(event),
        telnyx=api,
        writer=writer,
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    )

    assert result == WebhookDisposition(200)
    assert api.retrieve_calls == [("recording_Ab-12", 1.0)]
    assert len(writer.commands) == 1


@pytest.mark.asyncio
async def test_retrieve_not_yet_indexed_is_503_after_one_fallback() -> None:
    event = saved_event()
    api = CatalogApi(
        ProviderRecordingPageV1(1, 1, (catalog_item(source=None),)),
        retrieve=RecordingCatalogTransientError("recording_catalog_transient"),
    )

    result = await after_recording_webhook_commit(
        event,
        resolve_recording_webhook(event),
        telnyx=api,
        writer=CapturingWriter(),
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    )

    assert result == WebhookDisposition(503)
    assert api.retrieve_calls == [("recording_Ab-12", 1.0)]


@pytest.mark.asyncio
async def test_provider_transient_is_503_and_writer_identity_conflict_is_500() -> None:
    event = saved_event()
    transient = CatalogApi(RecordingCatalogTransientError("recording_catalog_transient"))
    assert await after_recording_webhook_commit(
        event,
        resolve_recording_webhook(event),
        telnyx=transient,
        writer=CapturingWriter(),
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    ) == WebhookDisposition(503)

    valid = CatalogApi(ProviderRecordingPageV1(1, 1, (catalog_item(),)))
    conflict = CapturingWriter(CommandConflictError("webhook_enrichment_conflict"))
    assert await after_recording_webhook_commit(
        event,
        resolve_recording_webhook(event),
        telnyx=valid,
        writer=conflict,
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    ) == WebhookDisposition(500)


async def start_writer(path: Path) -> tuple[PersistenceWriter, asyncio.Task[None]]:
    writer = PersistenceWriter(
        path,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW + timedelta(minutes=3),
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is True
    return writer, task


@pytest.mark.asyncio
async def test_redelivery_after_local_deletion_reconstructs_byte_identical_enrichment(
    tmp_path: Path,
) -> None:
    event = saved_event()
    effect = resolve_recording_webhook(event)
    assert effect is not None
    writer, task = await start_writer(tmp_path / "redelivery.sqlite")
    receipt = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "call_control_id": event.call_control_id,
        "occurred_at": event.occurred_at,
        "received_at": event.occurred_at,
        "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
    }
    await writer.commit_control(
        PersistenceCommand(
            "webhook_effect",
            {"receipt": receipt, "lease": None, "operation": effect.operation},
            None,
        )
    )
    api = CatalogApi(ProviderRecordingPageV1(1, 1, (catalog_item(),)))

    first_response = await after_recording_webhook_commit(
        event,
        effect,
        telnyx=api,
        writer=writer,
        local_drain=no_drain,
        monotonic=lambda: 10.0,
        timeout_seconds=2.0,
    )
    assert first_response == WebhookDisposition(200)
    first_batch = await writer.read_relay_batch(
        batch_size=10,
        now=NOW + timedelta(minutes=4),
        lease_seconds=5,
    )
    first_enrichment = next(
        item
        for item in first_batch
        if item.operation.payload.telnyx_recording_id == "recording_Ab-12"
    )
    first_bytes = canonical_operation_bytes(first_enrichment.operation)
    assert (
        await writer.ack_outbox(
        queue_id=first_enrichment.queue_id,
        expected_claim_attempt=first_enrichment.claim_attempt,
        )
    ).applied is True

    second_response = await after_recording_webhook_commit(
        event,
        effect,
        telnyx=api,
        writer=writer,
        local_drain=no_drain,
        monotonic=lambda: 20.0,
        timeout_seconds=2.0,
    )
    assert second_response == WebhookDisposition(200)
    second_batch = await writer.read_relay_batch(
        batch_size=10,
        now=NOW + timedelta(minutes=5),
        lease_seconds=5,
    )
    second_enrichment = next(
        item
        for item in second_batch
        if item.operation.payload.telnyx_recording_id == "recording_Ab-12"
    )
    assert canonical_operation_bytes(second_enrichment.operation) == first_bytes
    assert second_enrichment.operation.occurred_at == event.occurred_at

    api.page = ProviderRecordingPageV1(
        1,
        1,
        (catalog_item(recording_id="recording_Changed-9"),),
    )
    assert await after_recording_webhook_commit(
        event,
        effect,
        telnyx=api,
        writer=writer,
        local_drain=no_drain,
        monotonic=lambda: 30.0,
        timeout_seconds=2.0,
    ) == WebhookDisposition(500)
    await task


@pytest.mark.asyncio
async def test_signed_saved_reconciliation_preserves_1024_call_control_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_control_id = "c" * 1024
    correlation = build_recording_correlation(
        identity(call_control_id),
        retention_days=30,
        required=False,
    )
    body = json.dumps(
        {
            "data": {
                "id": "event-long-control-1",
                "event_type": "call.recording.saved",
                "occurred_at": "2026-08-29T12:02:00Z",
                "payload": {
                    "call_leg_id": "leg-1",
                    "call_session_id": "session-1",
                    "client_state": encode_recording_correlation(
                        correlation
                    ).get_secret_value(),
                    "recording_started_at": "2026-08-29T12:00:00Z",
                    "recording_ended_at": "2026-08-29T12:01:00Z",
                    "channels": "dual",
                    "public_recording_urls": {
                        "wav": "https://RAW-URL-SENTINEL"
                    },
                },
            }
        }
    ).encode("utf-8")
    timestamp = 1_777_118_400
    signing_key = SigningKey.generate()
    public_key = base64.b64encode(bytes(signing_key.verify_key)).decode("ascii")
    signature = base64.b64encode(
        signing_key.sign(f"{timestamp}|".encode() + body).signature
    ).decode("ascii")
    headers = [
        ("Telnyx-Signature-Ed25519", signature),
        ("Telnyx-Timestamp", str(timestamp)),
    ]
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    monkeypatch.setattr(verification.time, "time", lambda: float(timestamp))

    class PoisonRow:
        id = "recording_Ab-12"
        call_leg_id = "leg-1"
        call_session_id = "session-1"
        channels = "dual"
        status = "completed"
        source = "call"
        initiated_by = "StartCallRecordingAPI"
        recording_started_at = "2026-08-29T12:00:00Z"
        recording_ended_at = "2026-08-29T12:01:00Z"

        @property
        def call_control_id(self) -> str:
            return call_control_id

        @property
        def download_urls(self) -> object:
            raise AssertionError("recording URL must never be read")

    page = SimpleNamespace(
        meta=SimpleNamespace(page_number=1, total_pages=1),
        data=[PoisonRow()],
    )

    class OnePageAwaitable:
        def __await__(self):  # type: ignore[no-untyped-def]
            if False:
                yield None
            return page

    class FakeRecordings:
        def __init__(self) -> None:
            self.list_calls: list[dict[str, object]] = []

        def list(self, **kwargs: object) -> OnePageAwaitable:
            self.list_calls.append(dict(kwargs))
            return OnePageAwaitable()

    class FakeSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=SimpleNamespace())

        async def close(self) -> None:
            return None

    sdk = FakeSDK()
    monkeypatch.setattr(call_control_module.telnyx, "AsyncTelnyx", lambda **_: sdk)
    telnyx_client = CallControlClient(api_key="offline-test-key")
    writer, writer_task = await start_writer(tmp_path / "long-control.sqlite")

    async def after_commit(
        event: VerifiedWebhook,
        effect: object,
    ) -> WebhookDisposition | None:
        return await after_recording_webhook_commit(
            event,
            effect,  # type: ignore[arg-type]
            telnyx=telnyx_client,
            writer=writer,
            local_drain=no_drain,
            monotonic=lambda: 10.0,
            timeout_seconds=2.0,
        )

    class Handle:
        def __init__(self, task: asyncio.Task[WebhookDisposition]) -> None:
            self.task = task

        async def wait(self) -> WebhookDisposition:
            return await self.task

    class Owner:
        def start_webhook_finalization(
            self, event: VerifiedWebhook, resolution: object
        ) -> Handle:
            async def finalize() -> WebhookDisposition:
                effect = resolution.effect  # type: ignore[attr-defined]
                ticket = writer.submit_webhook(
                    receipt={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "call_control_id": event.call_control_id,
                        "occurred_at": event.occurred_at,
                        "received_at": NOW + timedelta(minutes=2),
                        "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
                    },
                    lease=None if effect is None else effect.lease,
                    operation=None if effect is None else effect.operation,
                )
                await ticket.wait()
                disposition = await after_commit(event, effect)
                return disposition or WebhookDisposition(200)

            return Handle(asyncio.create_task(finalize()))

    processor = TelnyxWebhookProcessor(
        verifier=TelnyxWebhookVerifier(public_key=public_key),
        resolver=lambda event: ResolvedWebhook(resolve_recording_webhook(event)),
        finalizer_owner=Owner(),
    )
    try:
        first_response = await processor.process(body=body, headers=headers)
        assert first_response == WebhookDisposition(200)
        first_batch = await writer.read_relay_batch(
            batch_size=10,
            now=NOW + timedelta(minutes=4),
            lease_seconds=5,
        )
        base = next(
            item
            for item in first_batch
            if item.operation.payload.telnyx_recording_id is None
        )
        enrichment = next(
            item
            for item in first_batch
            if item.operation.payload.telnyx_recording_id == "recording_Ab-12"
        )
        assert base.operation.payload.status == "saved"
        assert enrichment.operation.payload.status == "saved"
        assert sdk.recordings.list_calls[0]["filter"]["call_control_id"] == (
            call_control_id
        )
        first_bytes = canonical_operation_bytes(enrichment.operation)
        assert (
            await writer.ack_outbox(
                queue_id=enrichment.queue_id,
                expected_claim_attempt=enrichment.claim_attempt,
            )
        ).applied is True

        second_response = await processor.process(body=body, headers=headers)
        assert second_response == WebhookDisposition(200)
        second_batch = await writer.read_relay_batch(
            batch_size=10,
            now=NOW + timedelta(minutes=5),
            lease_seconds=5,
        )
        rebuilt = next(
            item
            for item in second_batch
            if item.operation.payload.telnyx_recording_id == "recording_Ab-12"
        )
        assert canonical_operation_bytes(rebuilt.operation) == first_bytes
        assert rebuilt.operation.occurred_at == enrichment.operation.occurred_at
        assert len(sdk.recordings.list_calls) == 2
    finally:
        if not writer_task.done():
            await writer.drain(timeout_seconds=2)
        await writer_task
        await telnyx_client.aclose()
