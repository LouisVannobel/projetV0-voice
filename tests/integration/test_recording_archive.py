from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from projetv0_voice.admission import CallGenerationHandle, ProcessLeaseClaim
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.models import (
    BeginCallSnapshotV1,
    CallUpsertPayloadV1,
    RecordingUpsertPayloadV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import PersistenceCommand, canonical_operation_bytes
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts, PersistenceWriter
from projetv0_voice.session import CallIdentity
from projetv0_voice.telnyx.recordings import (
    ProviderRecordingV1,
    after_recording_webhook_commit,
    build_recording_correlation,
    decode_recording_correlation,
    encode_recording_correlation,
    resolve_recording_webhook,
)
from projetv0_voice.telnyx.webhooks import VerifiedWebhook

NOW = datetime(2026, 10, 4, 12, tzinfo=UTC)
CALL_ID = UUID("12345678-1234-4234-8234-1234567890ab")
DEADLINE = datetime(2026, 11, 3, 12, tzinfo=UTC)
PROVIDER_ID = "recording_Ab-12"
URL = "https://recordings.example.invalid/private.wav?transient=sentinel"
KEY = bytes(range(32))


def identity(*, company: bool = True) -> CallIdentity:
    generation = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    snapshot = (
        BeginCallSnapshotV1.model_validate(
            {
                "schema_version": 1,
                "call_id": str(CALL_ID),
                "configuration_revision": 7,
                "knowledge": {
                    "business_name": "Garage",
                    "sector": "garage",
                    "opening_hours": "",
                    "services": "",
                    "prices": "",
                    "faq": "",
                    "instructions": "",
                },
                "transfer_destination": None,
                "retention_until": "2026-11-03T12:00:00.000Z",
                "recording_enabled": True,
            }
        )
        if company
        else None
    )
    return CallIdentity(
        call_id=CALL_ID,
        generation=CallGenerationHandle("v3:control", generation),
        lease_claim=ProcessLeaseClaim("v3:control", CALL_ID, generation, b"d" * 32, NOW),
        deployment_id="deployment-1",
        telnyx_call_control_id="v3:control",
        telnyx_call_leg_id="leg-1",
        telnyx_call_session_id="session-1",
        stream_id="stream",
        started_at=NOW,
        retention_until=DEADLINE,
        begin_snapshot=snapshot,
    )


def saved_event(*, company: bool = True) -> VerifiedWebhook:
    correlation = build_recording_correlation(
        identity(company=company), retention_days=30, required=company
    )
    return VerifiedWebhook(
        event_id="saved-1",
        event_type="call.recording.saved",
        occurred_at=NOW + timedelta(minutes=2),
        call_control_id="v3:control",
        call_leg_id="leg-1",
        call_session_id="session-1",
        recording_id=PROVIDER_ID,
        stream_id=None,
        client_state=encode_recording_correlation(correlation),
        recording_started_at=NOW,
        recording_ended_at=NOW + timedelta(minutes=1),
        recording_channels="dual",
        semantic_fingerprint_sha256=b"f" * 32,
    )


def archive_module():
    try:
        return importlib.import_module("projetv0_voice.recording_archive")
    except ModuleNotFoundError:
        pytest.fail("the saved recording has no encrypted byte archive consumer")


def wav() -> bytes:
    samples = b"\x01\x00\x02\x00" * 32
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 2, 8000, 32000, 4, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


class Provider:
    def __init__(
        self, *, url: str = URL, wrong_leg: bool = False, utcnow=lambda: NOW + timedelta(minutes=2)
    ) -> None:
        self.url = url
        self.wrong_leg = wrong_leg
        self.utcnow = utcnow

    async def retrieve_recording_download(self, recording_id: str, *, timeout_seconds: float):
        assert recording_id == PROVIDER_ID and 0 < timeout_seconds <= 1
        return archive_module().ProviderRecordingDownloadV1(
            recording=ProviderRecordingV1(
                PROVIDER_ID,
                "v3:control",
                "foreign-leg" if self.wrong_leg else "leg-1",
                "session-1",
                "dual",
                "completed",
                "call",
                "StartCallRecordingAPI",
                NOW,
                NOW + timedelta(minutes=1),
            ),
            wav_url=SecretStr(self.url),
            retrieved_at=self.utcnow(),
        )


def test_company_capsule_saved_deadline_uses_admission_and_legacy_omits_extension() -> None:
    event = saved_event()
    capsule = json.loads(base64.b64decode(event.client_state.get_secret_value()))
    assert capsule.get("admission_retention_until") == "2026-11-03T12:00:00.000Z"
    correlation = decode_recording_correlation(event.client_state)
    assert encode_recording_correlation(correlation) == event.client_state
    effect = resolve_recording_webhook(event)
    assert effect.operation.payload.retention_until == DEADLINE
    legacy = saved_event(company=False)
    assert "admission_retention_until" not in json.loads(
        base64.b64decode(legacy.client_state.get_secret_value())
    )
    assert resolve_recording_webhook(legacy).operation.payload.retention_until == (
        NOW + timedelta(days=30, minutes=1)
    )


def test_archive_receipt_roundtrip_is_strict_and_absent_legacy_bytes_are_preserved() -> None:
    base = resolve_recording_webhook(saved_event()).operation
    receipt = {
        "recording_id": str(base.payload.recording_id),
        "ciphertext_sha256": "a" * 64,
        "encrypted_bytes": 188,
        "key_version": 1,
        "retention_until": "2026-11-03T12:00:00.000Z",
    }
    raw = base.model_dump(mode="json")
    raw["payload"]["archive_receipt"] = receipt
    try:
        operation = VoiceOperationV1.model_validate(raw)
    except ValidationError:
        pytest.fail("durable recording operation rejects the bounded archive receipt")
    assert json.loads(canonical_operation_bytes(operation))["payload"]["archive_receipt"] == receipt
    assert "archive_receipt" not in json.loads(canonical_operation_bytes(base))["payload"]
    for field, value in [
        ("encrypted_bytes", True),
        ("encrypted_bytes", 33554449),
        ("ciphertext_sha256", "A" * 64),
        ("key_version", "1"),
        ("recording_id", str(UUID(int=99))),
        ("retention_until", "2026-11-03T12:00:00.001Z"),
    ]:
        malformed = {**receipt, field: value}
        with pytest.raises(ValidationError):
            VoiceOperationV1.model_validate(
                {**raw, "payload": {**raw["payload"], "archive_receipt": malformed}}
            )
    with pytest.raises(ValidationError):
        VoiceOperationV1.model_validate(
            {**raw, "payload": {**raw["payload"], "archive_receipt": {**receipt, "url": URL}}}
        )


async def queue_saved(
    writer: PersistenceWriter,
    event: VerifiedWebhook,
    *,
    observed_at: datetime = NOW + timedelta(minutes=2),
    historic_pin: bool = True,
) -> UUID:
    admitted = VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(int=7),
        deployment_id="deployment-1",
        call_id=CALL_ID,
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="v3:control",
            telnyx_call_leg_id="leg-1",
            telnyx_call_session_id="session-1",
            status="pending",
            disclosure_state="pending",
            started_at=None,
            ended_at=None,
            end_reason=None,
            retention_until=DEADLINE,
        ),
    )
    ticket = writer.submit_webhook(
        receipt={
            "event_id": "admitted-1",
            "event_type": "call.initiated",
            "call_control_id": "v3:control",
            "occurred_at": NOW,
            "received_at": NOW,
            "semantic_fingerprint_sha256": b"a" * 32,
        },
        lease={
            "action": "upsert",
            "call_control_id": "v3:control",
            "call_id": CALL_ID,
            "tenant_id": "tenant-1",
            "agent_id": "agent-1",
            "state": "pending",
            "token_hash": b"d" * 32,
            "created_at": NOW,
            "expires_at": NOW + timedelta(hours=1),
            "closed_at": None,
        },
        operation=admitted,
        admission_facts=LocalCallAdmissionFacts(
            CALL_ID,
            NOW,
            DEADLINE,
            "leg-1",
            "session-1",
            admission_generation=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        ),
    )
    await ticket.wait()
    if (
        historic_pin
        and (await writer.read_call_lifecycle(CALL_ID)).recording_policy_revision is None
    ):
        # Controlled historical admission setup: actual writer/pin/reservation,
        # never a native audio qualification or France-residence flag.
        await writer.bind_recording_policy(
            identity().begin_snapshot,
            generation=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        )
        await writer.reserve_recording_audio(
            CALL_ID,
            generation=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        )
        await commit_historic_input_gate(writer)
    effect = resolve_recording_webhook(event)
    await after_recording_webhook_commit(
        event,
        effect,
        telnyx=Provider(),
        writer=writer,
        local_drain=lambda *_: None,
        monotonic=lambda: 0.0,
        timeout_seconds=30,
        utcnow=lambda: observed_at,
    )
    return effect.operation.payload.recording_id


async def commit_historic_input_gate(writer):
    # Real owned writer consumes controlled historical call/disclosure facts;
    # this does not assert actual native playback or an open capability gate.
    operation = VoiceOperationV1.model_validate(
        {
            "schema_version": 1,
            "operation_id": str(UUID(int=17)),
            "call_id": str(CALL_ID),
            "deployment_id": "deployment-1",
            "occurred_at": NOW + timedelta(seconds=2),
            "kind": "call.upsert",
            "payload": {
                "telnyx_call_control_id": "v3:control",
                "telnyx_call_leg_id": "leg-1",
                "telnyx_call_session_id": "session-1",
                "status": "active",
                "disclosure_state": "completed",
                "started_at": NOW,
                "ended_at": None,
                "end_reason": None,
                "retention_until": DEADLINE,
                "disclosure_evidence": {
                    "schema_version": 1,
                    "started_at": NOW,
                    "completed_at": NOW + timedelta(seconds=1),
                    "failed_at": None,
                    "input_gate_opened_at": NOW + timedelta(seconds=2),
                },
            },
        }
    )
    await writer.commit_control(PersistenceCommand("outbox", {"operation": operation}, None))


async def start_writer(database: Path, keyring: CryptoKeyring):
    writer = PersistenceWriter(database, keyring=keyring, utcnow=lambda: NOW + timedelta(minutes=2))
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    return writer, task


async def stop_writer(writer, task):
    await writer.drain(2)
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_real_wav_ciphertext_receipt_and_pending_job_survive_sqlite_restart(
    tmp_path: Path,
) -> None:
    archive = archive_module()
    directory = tmp_path / "audio"
    directory.mkdir(mode=0o700)
    database = tmp_path / "voice.sqlite"
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    writer, task = await start_writer(database, keyring)
    consumer = archive.RecordingArchive(
        directory=directory,
        keyring=keyring,
        writer=writer,
        telnyx=Provider(),
        allowed_origins=("https://recordings.example.invalid",),
        download_transport=httpx.MockTransport(lambda request: httpx.Response(200, content=wav())),
        directory_sync=lambda: None,
        utcnow=lambda: NOW + timedelta(minutes=2),
    )
    try:
        recording_id = await queue_saved(writer, saved_event())
        pending = await writer.read_recording_archive(recording_id)
        assert pending.state == "pending" and pending.retention_until == DEADLINE
        await stop_writer(writer, task)
        writer, task = await start_writer(database, keyring)
        await consumer.aclose()
        consumer = archive.RecordingArchive(
            directory=directory,
            keyring=keyring,
            writer=writer,
            telnyx=Provider(),
            allowed_origins=("https://recordings.example.invalid",),
            download_transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=wav())
            ),
            directory_sync=lambda: None,
            utcnow=lambda: NOW + timedelta(minutes=2),
        )
        result = await consumer.archive_recording_once(recording_id)
        assert result.outcome == "archived" and result.native_acknowledged is False
        job = await writer.read_recording_archive(recording_id)
        ciphertext = (directory / str(recording_id)).read_bytes()
        assert wav() not in ciphertext
        assert (
            keyring.decrypt(
                EncryptedValue(job.receipt.key_version, job.nonce, ciphertext),
                aad=f"recording:{CALL_ID}:{recording_id}".encode(),
            )
            == wav()
        )
        assert hashlib.sha256(ciphertext).hexdigest() == job.receipt.ciphertext_sha256
        assert len(ciphertext) == job.receipt.encrypted_bytes == 188
        assert job.receipt.retention_until == DEADLINE
        operations = await writer.read_relay_batch(
            batch_size=10, now=NOW + timedelta(minutes=3), lease_seconds=30
        )
        receipts = [
            item.operation
            for item in operations
            if isinstance(item.operation.payload, RecordingUpsertPayloadV1)
            and item.operation.payload.archive_receipt is not None
        ]
        assert len(receipts) == 1 and receipts[0].payload.archive_receipt == job.receipt
        assert len(canonical_operation_bytes(receipts[0])) < 65536
        await stop_writer(writer, task)
        writer, task = await start_writer(database, keyring)
        restarted = await writer.read_recording_archive(recording_id)
        assert restarted.receipt == job.receipt and restarted.nonce == job.nonce
        assert wav() not in database.read_bytes() and URL.encode() not in database.read_bytes()
        assert [path.name for path in directory.iterdir()] == [str(recording_id)]
    finally:
        await consumer.aclose()
        await stop_writer(writer, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["wrong_leg", "foreign_url", "redirect", "malformed", "expired"])
async def test_archive_failure_never_publishes_receipt_or_plaintext(
    tmp_path: Path, fault: str
) -> None:
    archive = archive_module()
    directory = tmp_path / "audio"
    directory.mkdir(mode=0o700)
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    writer, task = await start_writer(tmp_path / "voice.sqlite", keyring)
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(302, headers={"location": URL})
            if fault == "redirect"
            else (httpx.Response(200, content=b"not a wav" if fault == "malformed" else wav()))
        )

    consumer = archive.RecordingArchive(
        directory=directory,
        keyring=keyring,
        writer=writer,
        telnyx=Provider(
            wrong_leg=fault == "wrong_leg",
            url=("https://foreign.example.invalid/audio" if fault == "foreign_url" else URL),
        ),
        allowed_origins=("https://recordings.example.invalid",),
        download_transport=httpx.MockTransport(respond),
        directory_sync=lambda: None,
        utcnow=lambda: DEADLINE if fault == "expired" else NOW + timedelta(minutes=2),
    )
    try:
        recording_id = await queue_saved(writer, saved_event())
        result = await consumer.archive_recording_once(recording_id)
        assert result.outcome == ("expired" if fault == "expired" else "unavailable")
        assert not list(directory.iterdir())
        job = await writer.read_recording_archive(recording_id)
        if fault == "expired":
            assert job is None  # Complete expiry removes private audio context before cleanup ACK.
        else:
            assert job.receipt is None
        if fault in {"wrong_leg", "foreign_url", "expired"}:
            assert requests == []
    finally:
        await consumer.aclose()
        await stop_writer(writer, task)
