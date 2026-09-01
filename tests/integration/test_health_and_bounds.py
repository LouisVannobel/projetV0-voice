from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimePublication, RuntimePublishedSnapshot
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.persistence.writer import PersistenceWriter

NOW = datetime(2026, 9, 1, 9, tzinfo=UTC)
KEY = bytes(range(32))


def _initial_snapshot() -> RuntimePublishedSnapshot:
    return RuntimePublishedSnapshot(
        generation=0,
        ready=False,
        draining=False,
        writer_queue_depth=0,
        writer_queue_oldest_age=0.0,
        writer_quick_check=False,
        outbox_depth=0,
        outbox_oldest_age=0.0,
        outbox_bytes=0,
        storage_bytes=0,
        startup_profile_and_stale_recovery_complete=False,
        admission_open=False,
        qualification_state_valid=False,
        writer_owner_alive_and_ready=False,
        no_writer_fatal_or_degradation=False,
        relay_supervisor_alive=False,
        no_permanent_relay_or_sink_degradation=False,
    )


def _operation() -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID("11111111-1111-4111-8111-111111111111"),
        deployment_id="deployment-a",
        call_id=UUID("22222222-2222-4222-8222-222222222222"),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="control-a",
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


@pytest.mark.asyncio
async def test_writer_runtime_observation_uses_one_owner_queued_outbox_aggregate(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"

    def file_size(path: Path) -> int:
        return 100 if path == database else 20

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        monotonic=lambda: 10.0,
        utcnow=lambda: NOW,
        file_size=file_size,
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    await writer.commit_control(
        PersistenceCommand("outbox", {"operation": _operation()}, None)
    )

    observed = await writer.runtime_observation()

    assert observed.writer_queue_depth == 0
    assert observed.writer_queue_oldest_age == 0.0
    assert observed.writer_quick_check is True
    assert observed.outbox_depth == 1
    assert observed.outbox_oldest_age == 0.0
    assert observed.outbox_bytes > 0
    assert observed.storage_bytes == 120

    await writer.drain(timeout_seconds=2.0)
    await writer_task
    with sqlite3.connect(database) as connection:
        expected_bytes = connection.execute(
            "SELECT COALESCE(SUM(length(nonce) + length(ciphertext)), 0) FROM outbox"
        ).fetchone()[0]
    assert observed.outbox_bytes == expected_bytes


@pytest.mark.parametrize(
    "change",
    [
        {"startup_profile_and_stale_recovery_complete": False},
        {"draining": True},
        {"admission_open": False},
        {"qualification_state_valid": False},
        {"writer_owner_alive_and_ready": False},
        {"writer_quick_check": False},
        {"no_writer_fatal_or_degradation": False},
        {"storage_bytes": 268_435_457},
        {"writer_queue_oldest_age": 1.000_001},
        {"relay_supervisor_alive": False},
        {"no_permanent_relay_or_sink_degradation": False},
        {"outbox_oldest_age": 900.000_001},
    ],
)
def test_readiness_truth_table_closes_each_required_predicate(
    change: dict[str, object],
) -> None:
    from projetv0_voice.lifecycle import publish_runtime_readiness
    from projetv0_voice.persistence.writer import WriterRuntimeObservation

    publication = RuntimePublication(_initial_snapshot())
    writer = WriterRuntimeObservation(
        writer_queue_depth=0,
        writer_queue_oldest_age=0.0,
        writer_quick_check=True,
        outbox_depth=0,
        outbox_oldest_age=0.0,
        outbox_bytes=0,
        storage_bytes=268_435_456,
    )
    values: dict[str, object] = {
        "startup_profile_and_stale_recovery_complete": True,
        "draining": False,
        "admission_open": True,
        "qualification_state_valid": True,
        "writer_owner_alive_and_ready": True,
        "no_writer_fatal_or_degradation": True,
        "relay_supervisor_alive": True,
        "no_permanent_relay_or_sink_degradation": True,
    }
    for name in tuple(change):
        if hasattr(writer, name):
            writer = replace(writer, **{name: change.pop(name)})
    values.update(change)

    published = publish_runtime_readiness(
        publication,
        writer=writer,
        **values,
    )

    assert published.ready is False
    assert publication.snapshot() is published


def test_readiness_healthy_boundaries_publish_one_complete_ready_generation() -> None:
    from projetv0_voice.lifecycle import publish_runtime_readiness
    from projetv0_voice.persistence.writer import WriterRuntimeObservation

    publication = RuntimePublication(_initial_snapshot())
    published = publish_runtime_readiness(
        publication,
        writer=WriterRuntimeObservation(
            writer_queue_depth=256,
            writer_queue_oldest_age=1.0,
            writer_quick_check=True,
            outbox_depth=1,
            outbox_oldest_age=900.0,
            outbox_bytes=12,
            storage_bytes=268_435_456,
        ),
        startup_profile_and_stale_recovery_complete=True,
        draining=False,
        admission_open=True,
        qualification_state_valid=True,
        writer_owner_alive_and_ready=True,
        no_writer_fatal_or_degradation=True,
        relay_supervisor_alive=True,
        no_permanent_relay_or_sink_degradation=True,
    )

    assert published.ready is True
    assert published.generation == 1
    assert publication.snapshot() == published
