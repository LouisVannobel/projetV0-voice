from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from projetv0_voice.models import BeginCallSnapshotV1, CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.writer import LocalCallAdmissionFacts
from tests.integration.test_recording_archive import (
    CALL_ID,
    DEADLINE,
    NOW,
    commit_historic_input_gate,
    identity,
    queue_saved,
    saved_event,
)
from tests.integration.test_recording_archive_ownership import owned as history_owned
from tests.integration.test_recording_archive_ownership import restart


@asynccontextmanager
async def owned(tmp_path, **kwargs):
    async with history_owned(tmp_path, historic_pin=False, **kwargs) as box:
        yield box


def native_pin(*, enabled=True, revision=7):
    return BeginCallSnapshotV1.model_validate(
        {
            **identity().begin_snapshot.model_dump(mode="json"),
            "recording_enabled": enabled,
            "configuration_revision": revision,
        }
    )


async def bind(box, *, enabled=True, revision=7, generation=None):
    function = getattr(box.writer, "bind_recording_policy", None)
    if not callable(function):
        pytest.fail("the real writer has no authenticated native-snapshot policy consumer")
    facts = await box.writer.read_call_lifecycle(CALL_ID)
    return await function(
        native_pin(enabled=enabled, revision=revision),
        generation=generation or facts.admission_generation,
    )


async def reserve(box):
    function = getattr(box.writer, "reserve_recording_audio", None)
    if not callable(function):
        pytest.fail("the owned archive has no durable capacity consumer")
    facts = await box.writer.read_call_lifecycle(CALL_ID)
    return await function(CALL_ID, generation=facts.admission_generation)


@pytest.mark.asyncio
async def test_unbound_sparra_saved_callback_never_downloads_or_installs_audio(tmp_path):
    async with owned(tmp_path) as box:
        result = await box.consumer.archive_recording_once(box.recording_id)
        assert result.outcome == "unknown"
        assert box.requests == [] and not list(box.directory.iterdir())
        assert await box.writer.read_recording_archive(box.recording_id) is None
        assert not box.writer.is_degraded


@pytest.mark.asyncio
async def test_bound_off_refuses_copy_without_poisoning_other_content(tmp_path):
    async with owned(tmp_path) as box:
        await bind(box, enabled=False)
        await queue_saved(box.writer, saved_event(), historic_pin=False)
        assert (await box.consumer.archive_recording_once(box.recording_id)).outcome == "unknown"
        assert box.requests == [] and not list(box.directory.iterdir())
        assert await box.writer.read_recording_archive(box.recording_id) is None
        assert not box.writer.is_degraded


@pytest.mark.asyncio
async def test_pending_native_pin_and_space_cannot_copy_before_completed_session_input_gate(
    tmp_path,
):
    async with owned(tmp_path) as box:
        await bind(box)
        await reserve(box)
        await queue_saved(box.writer, saved_event(), historic_pin=False)
        result = await box.consumer.archive_recording_once(box.recording_id)
        assert result.outcome == "unknown"
        assert box.requests == [] and not list(box.directory.iterdir())
        assert await box.writer.read_recording_archive(box.recording_id) is None


@pytest.mark.asyncio
async def test_native_policy_replay_is_immutable_and_survives_writer_restart(tmp_path):
    async with owned(tmp_path) as box:
        await bind(box)
        await reserve(box)
        first = await box.writer.read_call_lifecycle(CALL_ID)
        assert first.recording_policy_revision == 7 and first.recording_enabled is True
        assert first.audio_reserved_bytes == 33554448
        await bind(box)
        await reserve(box)
        await restart(box)
        assert await box.writer.read_call_lifecycle(CALL_ID) == first
        await bind(box)
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 33554448
        with pytest.raises(PersistenceError, match="recording_policy_conflict"):
            await bind(box, enabled=False, revision=8)
        assert await box.writer.read_call_lifecycle(CALL_ID) == first
        assert not box.writer.is_degraded


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["wrong_generation", "erased", "expired"])
async def test_native_pin_rejects_wrong_generation_erasure_and_late_authority(tmp_path, fence):
    async with owned(tmp_path) as box:
        if fence == "erased":
            await box.writer.erase_call_content(CALL_ID, now=box.clock.utcnow())
        elif fence == "expired":
            box.clock.seconds = (DEADLINE - box.clock.utcnow()).total_seconds()
        with pytest.raises(PersistenceError, match="recording_policy_unavailable"):
            await bind(box, generation=UUID(int=999) if fence == "wrong_generation" else None)
        facts = await box.writer.read_call_lifecycle(CALL_ID)
        assert facts.recording_policy_revision is None and facts.audio_reserved_bytes == 0
        assert not box.writer.is_degraded


@pytest.mark.asyncio
async def test_capacity_checks_actual_consumer_and_persistent_pending_reservation(
    tmp_path, monkeypatch
):
    async with owned(tmp_path) as box:
        await bind(box)
        set_consumer_free_bytes(monkeypatch, box, 33554447)
        with pytest.raises(PersistenceError, match="recording_archive_unavailable"):
            await reserve(box)
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 0
        assert not box.writer.is_degraded
        set_consumer_free_bytes(monkeypatch, box, 33554448)
        await reserve(box)
        await reserve(box)
        await restart(box)
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 33554448
        await reserve(box)


def replace_usage(usage, free):
    return type(usage)(usage.total, usage.used, free)


def set_consumer_free_bytes(monkeypatch, box, free):
    if box.consumer._directory_fd is not None:
        def descriptor_usage(descriptor):
            assert descriptor == box.consumer._directory_fd
            return SimpleNamespace(f_bavail=free, f_frsize=1)

        monkeypatch.setattr(os, "fstatvfs", descriptor_usage)
    else:
        usage = shutil.disk_usage(box.directory)
        monkeypatch.setattr(shutil, "disk_usage", lambda path: replace_usage(usage, free))


@pytest.mark.asyncio
async def test_capacity_counts_committed_file_usage_once_and_releases_after_cleanup(tmp_path):
    async with owned(tmp_path) as box:
        await bind(box)
        await reserve(box)
        await commit_historic_input_gate(box.writer)
        await queue_saved(box.writer, saved_event(), historic_pin=False)
        assert (await box.consumer.archive_recording_once(box.recording_id)).outcome == "archived"
        facts = await box.writer.read_call_lifecycle(CALL_ID)
        assert (
            facts.audio_reserved_bytes == 0
        )  # Filesystem free space already accounts for real bytes.
        await reserve(box)
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 0
        await box.writer.erase_call_content(CALL_ID, now=box.clock.utcnow())
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 0
        assert not list(box.directory.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["closed_consumer", "pending_overdue", "copy_failed", "unacknowledged"]
)
async def test_broken_or_overdue_copy_refuses_new_on_capacity_but_keeps_off_writer(
    tmp_path, failure
):
    async with owned(tmp_path) as box:
        await bind(box)
        await reserve(box)
        await commit_historic_input_gate(box.writer)
        await queue_saved(box.writer, saved_event(), historic_pin=False)
        if failure == "closed_consumer":
            await box.consumer.aclose()
        elif failure == "copy_failed":
            await box.writer.fail_recording_archive(
                box.recording_id, expired=False, now=box.clock.utcnow()
            )
        elif failure == "unacknowledged":
            await box.consumer.archive_recording_once(box.recording_id)
            box.clock.seconds = 121
        else:
            box.clock.seconds = 121
        with pytest.raises(PersistenceError, match="recording_archive_unavailable"):
            await reserve(box)
        assert not box.writer.is_degraded


@pytest.mark.asyncio
async def test_two_real_calls_cannot_double_book_pending_additional_space(tmp_path, monkeypatch):
    async with owned(tmp_path) as box:
        await bind(box)
        set_consumer_free_bytes(monkeypatch, box, 33554448)
        await reserve(box)
        call_id, generation = UUID(int=22), UUID(int=23)
        admitted = VoiceOperationV1(
            schema_version=1,
            operation_id=UUID(int=24),
            call_id=call_id,
            deployment_id="deployment-1",
            occurred_at=NOW,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id="v3:other",
                telnyx_call_leg_id="leg-other",
                telnyx_call_session_id="session-other",
                status="pending",
                disclosure_state="pending",
                started_at=None,
                ended_at=None,
                end_reason=None,
                retention_until=DEADLINE,
            ),
        )
        await box.writer.submit_webhook(
            receipt={
                "event_id": "other-admission",
                "event_type": "call.initiated",
                "call_control_id": "v3:other",
                "occurred_at": NOW,
                "received_at": NOW,
                "semantic_fingerprint_sha256": b"b" * 32,
            },
            lease={
                "action": "upsert",
                "call_control_id": "v3:other",
                "call_id": call_id,
                "tenant_id": "other",
                "agent_id": "other",
                "state": "pending",
                "token_hash": b"t" * 32,
                "created_at": NOW,
                "expires_at": NOW + timedelta(hours=1),
                "closed_at": None,
            },
            operation=admitted,
            admission_facts=LocalCallAdmissionFacts(
                call_id,
                NOW,
                DEADLINE,
                "leg-other",
                "session-other",
                admission_generation=generation,
            ),
        ).wait()
        second = BeginCallSnapshotV1.model_validate(
            {**native_pin().model_dump(mode="json"), "call_id": str(call_id)}
        )
        await box.writer.bind_recording_policy(second, generation=generation)
        with pytest.raises(PersistenceError, match="recording_archive_unavailable"):
            await box.writer.reserve_recording_audio(call_id, generation=generation)
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 33554448
        assert (await box.writer.read_call_lifecycle(call_id)).audio_reserved_bytes == 0
        assert not box.writer.is_degraded


@pytest.mark.asyncio
async def test_missing_saved_callback_after_actual_terminal_observation_blocks_on_across_restart(
    tmp_path,
):
    async with owned(tmp_path) as box:
        await bind(box)
        await reserve(box)
        await box.writer.submit_webhook(
            receipt={
                "event_id": "terminal-without-copy",
                "event_type": "call.hangup",
                "call_control_id": "v3:control",
                "call_leg_id": "leg-1",
                "call_session_id": "session-1",
                "occurred_at": box.clock.utcnow(),
                "received_at": box.clock.utcnow(),
                "semantic_fingerprint_sha256": b"e" * 32,
            },
            lease={
                "action": "upsert",
                "call_control_id": "v3:control",
                "call_id": CALL_ID,
                "tenant_id": "tenant-1",
                "agent_id": "agent-1",
                "state": "terminal",
                "token_hash": b"d" * 32,
                "created_at": NOW,
                "expires_at": NOW + timedelta(hours=1),
                "closed_at": box.clock.utcnow(),
            },
            operation=None,
        ).wait()
        assert await box.writer.read_recording_archive(box.recording_id) is None
        await restart(box)
        box.clock.seconds = 121
        with pytest.raises(PersistenceError, match="recording_archive_unavailable"):
            await reserve(box)
        assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 33554448
        assert not box.writer.is_degraded


@pytest.mark.asyncio
async def test_pending_space_is_released_only_after_actual_fenced_audio_cleanup(tmp_path):
    async with history_owned(tmp_path) as box:
        entered, release = asyncio.Event(), asyncio.Event()
        cleanup = box.writer._audio_cleanup

        async def held(call_id, recordings):
            entered.set()
            await release.wait()
            await cleanup(call_id, recordings)

        box.writer._audio_cleanup = held
        erase = asyncio.create_task(box.writer.erase_call_content(CALL_ID, now=box.clock.utcnow()))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            fenced = await box.writer.read_call_lifecycle(CALL_ID)
            assert fenced.content_erased is True
            assert fenced.audio_reserved_bytes == 33554448
            assert not erase.done()
            release.set()
            await asyncio.wait_for(erase, 2)
            assert (await box.writer.read_call_lifecycle(CALL_ID)).audio_reserved_bytes == 0
        finally:
            release.set()
            await asyncio.gather(erase, return_exceptions=True)
            box.writer._audio_cleanup = cleanup
