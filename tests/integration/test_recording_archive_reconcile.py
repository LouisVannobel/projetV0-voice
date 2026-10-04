from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid5

import httpx
import pytest
from test_recording_archive import (
    KEY,
    NOW,
    PROVIDER_ID,
    Provider,
    archive_module,
    queue_saved,
    saved_event,
    start_writer,
    stop_writer,
    wav,
)

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.telnyx.recordings import ProviderRecordingPageV1, resolve_recording_webhook


@pytest.mark.asyncio
@pytest.mark.parametrize("overdue", [False, True])
async def test_missing_provider_id_real_maintenance_reconciles_without_resetting_observation(
    tmp_path, overdue
):
    clock = [NOW + timedelta(minutes=2)]
    directory = tmp_path / "audio"
    directory.mkdir(mode=0o700)
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    writer, owner = await start_writer(tmp_path / "voice.sqlite", keyring)
    lookups, requests = [], []

    class Catalogue(Provider):
        async def list_recordings_one_page(self, **kwargs):
            lookups.append(kwargs)
            download = await self.retrieve_recording_download(PROVIDER_ID, timeout_seconds=0.75)
            return ProviderRecordingPageV1(1, 1, (download.recording,))

    def response(request):
        requests.append(request)
        return httpx.Response(200, content=wav())

    def consumer():
        return archive_module().RecordingArchive(
            directory=directory,
            writer=writer,
            keyring=keyring,
            telnyx=Catalogue(utcnow=lambda: clock[0]),
            allowed_origins=("https://recordings.example.invalid",),
            utcnow=lambda: clock[0],
            directory_sync=lambda: None,
            download_transport=httpx.MockTransport(response),
        )

    archive = consumer()
    event = replace(saved_event(), recording_id=None)
    try:
        recording_id = await queue_saved(writer, event, observed_at=clock[0])
        pending = await writer.read_recording_archive(recording_id)
        assert pending is not None, "eligible missing-ID callback lost its persistent archive job"
        assert pending.operation.payload.telnyx_recording_id is None
        first_observed = pending.observed_at
        effect = resolve_recording_webhook(event)
        await writer.submit_webhook(
            receipt={
                "event_id": event.event_id,
                "event_type": event.event_type,
                "call_control_id": event.call_control_id,
                "occurred_at": event.occurred_at,
                "received_at": clock[0],
                "semantic_fingerprint_sha256": event.semantic_fingerprint_sha256,
            },
            lease=None,
            operation=effect.operation,
        ).wait()
        await archive.aclose()
        await stop_writer(writer, owner)
        if overdue:
            clock[0] += timedelta(seconds=120)
        writer, owner = await start_writer(tmp_path / "voice.sqlite", keyring)
        archive = consumer()
        await archive.run_once()
        settled = await writer.read_recording_archive(recording_id)
        assert settled.observed_at == first_observed
        if overdue:
            assert settled.state == "unavailable" and lookups == [] and requests == []
        else:
            assert (
                settled.state == "archived"
                and settled.operation.payload.telnyx_recording_id == PROVIDER_ID
            )
            assert len(lookups) == 1 and len(requests) == 1
            assert lookups[0]["call_control_id"] == "v3:control"
            assert lookups[0]["call_leg_id"] == "leg-1"
            assert lookups[0]["call_session_id"] == "session-1"
    finally:
        await archive.aclose()
        if writer.fatal_event.is_set():
            await asyncio.wait_for(owner, 2)
        else:
            await stop_writer(writer, owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ["same_id", "changed_id", "wrong_type"])
async def test_missing_id_fallback_preserves_catalogue_candidate_before_real_binding(
    tmp_path, fallback
):
    directory = tmp_path / "audio"
    directory.mkdir(mode=0o700)
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    writer, owner = await start_writer(tmp_path / "voice.sqlite", keyring)
    lookups, downloads, requests = [], [], []
    full = (
        await Provider().retrieve_recording_download(PROVIDER_ID, timeout_seconds=0.75)
    ).recording
    returned = replace(full, recording_id="recording_Other-B") if fallback == "changed_id" else full

    class Catalogue(Provider):
        async def list_recordings_one_page(self, **_kwargs):
            return ProviderRecordingPageV1(1, 1, (replace(full, call_leg_id=None),))

        async def retrieve_recording(self, recording_id, *, timeout_seconds):
            lookups.append(recording_id)
            assert recording_id == PROVIDER_ID and 0 < timeout_seconds <= 1
            return SimpleNamespace(**asdict(returned)) if fallback == "wrong_type" else returned

        async def retrieve_recording_download(self, recording_id, *, timeout_seconds):
            downloads.append(recording_id)
            base = await super().retrieve_recording_download(
                PROVIDER_ID, timeout_seconds=timeout_seconds
            )
            return replace(base, recording=returned)

    def response(request):
        requests.append(request)
        return httpx.Response(200, content=wav())

    archive = archive_module().RecordingArchive(
        directory=directory,
        writer=writer,
        keyring=keyring,
        telnyx=Catalogue(),
        allowed_origins=("https://recordings.example.invalid",),
        utcnow=lambda: NOW + timedelta(minutes=2),
        directory_sync=lambda: None,
        download_transport=httpx.MockTransport(response),
    )
    try:
        recording_id = await queue_saved(writer, replace(saved_event(), recording_id=None))
        assert (
            await writer.read_recording_archive(recording_id)
        ).operation.payload.telnyx_recording_id is None
        outcome = await archive.archive_recording_once(recording_id)
        settled = await writer.read_recording_archive(recording_id)
        rows = await writer.read_relay_batch(
            batch_size=100, now=NOW + timedelta(minutes=2), lease_seconds=30
        )
        binding_ids = {row.operation.operation_id for row in rows}
        assert lookups == [PROVIDER_ID]
        if fallback == "same_id":
            assert (
                outcome.outcome == "archived"
                and settled.operation.payload.telnyx_recording_id == PROVIDER_ID
            )
            assert uuid5(recording_id, "archive-provider-id-v1") in binding_ids
            assert len(downloads) == 1 and len(requests) == 1
        else:
            assert outcome.outcome == "unavailable"
            assert settled.operation.payload.telnyx_recording_id is None and settled.receipt is None
            assert uuid5(recording_id, "archive-provider-id-v1") not in binding_ids
            assert downloads == [] and requests == [] and list(directory.iterdir()) == []
    finally:
        await archive.aclose()
        if writer.fatal_event.is_set():
            await asyncio.wait_for(owner, 2)
        else:
            await stop_writer(writer, owner)
