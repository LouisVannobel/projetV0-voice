from __future__ import annotations

import asyncio
import base64
import errno
import gc
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import aiosqlite
import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import (
    CallUpsertPayloadV1,
    RecordingUpsertPayloadV1,
    TurnUpsertPayloadV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    CommandSerializationError,
    EncryptedCommandTooLarge,
    FatalPersistenceError,
    PersistenceCommand,
    canonical_operation_bytes,
    encrypt_operation,
    operation_aad,
)
from projetv0_voice.persistence.schema import SCHEMA_SQL
from projetv0_voice.persistence.writer import PersistenceWriter

KEY = bytes(range(32))
NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def operation(
    number: int = 1,
    *,
    operation_id: UUID | None = None,
    call_control_id: str | None = None,
) -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=operation_id or UUID(int=number),
        deployment_id="agent-a",
        call_id=UUID(int=10_000 + number),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id=call_control_id or f"call-{number}",
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


def turn_operation(
    number: int = 1,
    *,
    operation_id: UUID | None = None,
    encoded_content: bytes | None = None,
) -> VoiceOperationV1:
    content = encoded_content or f"encrypted-{number}".encode()
    return VoiceOperationV1(
        schema_version=1,
        operation_id=operation_id or UUID(int=number),
        deployment_id="agent-a",
        call_id=UUID(int=10_000 + number),
        occurred_at=NOW + timedelta(seconds=2),
        kind="turn.upsert",
        payload=TurnUpsertPayloadV1(
            turn_id=UUID(int=20_000 + number),
            turn_no=number,
            role="user",
            source="stt_final",
            crypto_version=1,
            key_version=1,
            nonce_b64=base64.b64encode(bytes(range(12))).decode(),
            ciphertext_b64=base64.b64encode(content).decode(),
            started_at=NOW,
            ended_at=NOW + timedelta(seconds=1),
            interrupted=False,
        ),
    )


def recording_operation() -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID("b10b98ee-616c-50d0-81e4-af5d8762e082"),
        deployment_id="agent-a",
        call_id=UUID(int=123),
        occurred_at=NOW + timedelta(minutes=2),
        kind="recording.upsert",
        payload=RecordingUpsertPayloadV1(
            recording_id=UUID(int=456),
            status="saved",
            telnyx_recording_id="recording_Ab-12",
            channels="dual",
            format="wav",
            started_at=NOW,
            ended_at=NOW + timedelta(minutes=1),
            retention_until=NOW + timedelta(days=30),
        ),
    )


def lease_payload(
    *,
    call_control_id: str = "control-1",
    state: str = "pending",
    expires_at: datetime = NOW + timedelta(seconds=30),
) -> dict[str, object]:
    return {
        "action": "upsert",
        "call_control_id": call_control_id,
        "call_id": UUID(int=123),
        "tenant_id": "tenant-a",
        "agent_id": "agent-a",
        "state": state,
        "token_hash": bytes(range(32)),
        "created_at": NOW,
        "expires_at": expires_at,
        "closed_at": NOW if state == "terminal" else None,
    }


def receipt_payload(event_id: str = "event-1") -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "call.initiated",
        "call_control_id": "control-1",
        "occurred_at": NOW,
        "received_at": NOW,
        "semantic_fingerprint_sha256": bytes(range(32)),
    }


async def start_writer(
    path: Path,
    **kwargs: object,
) -> tuple[PersistenceWriter, asyncio.Task[None]]:
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1), **kwargs)
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is True
    return writer, task


async def stop_writer(writer: PersistenceWriter, task: asyncio.Task[None]) -> None:
    await writer.drain(timeout_seconds=2)
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_turn_queue_is_exactly_256_put_nowait_and_full_is_one_shot_fatal(
    tmp_path: Path,
) -> None:
    faults: list[object] = []
    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        fatal_handler=faults.append,
    )

    for number in range(1, 257):
        assert writer.try_enqueue_turn(turn_operation(number)) is True

    assert writer.queue_size == 256
    assert writer.try_enqueue_turn(turn_operation(257)) is False
    assert writer.try_enqueue_turn(turn_operation(258)) is False
    assert writer.transcript_loss_count == 2
    assert writer.is_degraded is True
    assert len(faults) == 1
    assert "call-257" not in repr(faults[0])


@pytest.mark.asyncio
async def test_fifo_queue_ids_and_raw_plaintext_never_reach_database_or_journal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(database)
    sentinel = "PLAINTEXT-TRANSCRIPT-SENTINEL-9D17"
    plaintext_operation = operation(1, call_control_id=sentinel)
    assert sentinel.encode() in canonical_operation_bytes(plaintext_operation)

    await writer.commit_control(
        PersistenceCommand("outbox", {"operation": plaintext_operation}, None)
    )
    assert writer.try_enqueue_turn(turn_operation(2))
    assert writer.try_enqueue_turn(turn_operation(3))
    await stop_writer(writer, task)

    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT queue_id, op_id FROM outbox ORDER BY queue_id").fetchall()
    assert rows == [(1, str(UUID(int=1))), (2, str(UUID(int=2))), (3, str(UUID(int=3)))]
    database_bytes = await asyncio.to_thread(database.read_bytes)
    assert sentinel.encode() not in database_bytes
    journal = Path(f"{database}-journal")
    if await asyncio.to_thread(journal.exists):
        journal_bytes = await asyncio.to_thread(journal.read_bytes)
        assert sentinel.encode() not in journal_bytes


@pytest.mark.asyncio
async def test_drain_deadline_includes_waiting_to_enqueue_shutdown_on_a_full_queue(
    tmp_path: Path,
) -> None:
    owner_stuck = asyncio.Event()
    release_owner = asyncio.Event()

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            owner_stuck.set()
            await release_owner.wait()

    writer, run_task = await start_writer(
        tmp_path / "full-drain.sqlite",
        failpoint=failpoint,
    )
    first_commit = asyncio.create_task(
        writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    )
    await owner_stuck.wait()
    for number in range(1, 257):
        assert writer.try_enqueue_turn(turn_operation(number))

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(writer.drain(timeout_seconds=0.01), timeout=0.3)

        assert loop.time() - started_at < 0.1
        assert writer.queue_size == 256
    finally:
        release_owner.set()
        await asyncio.gather(first_commit, return_exceptions=True)
        run_task.cancel()
        await asyncio.wait_for(run_task, timeout=1)


@pytest.mark.asyncio
async def test_control_future_resolves_only_after_commit_and_default_budget_is_1_5_seconds(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(database)
    assert writer.control_commit_timeout_seconds == 1.5

    await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state FROM call_leases WHERE call_control_id = 'control-1'"
        ).fetchone() == ("pending",)
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_control_timeout_is_safe_fatal_and_leaves_no_hanging_future(tmp_path: Path) -> None:
    release_commit = asyncio.Event()

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            await release_commit.wait()

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        control_commit_timeout_seconds=0.01,
        failpoint=failpoint,
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    command = PersistenceCommand("lease", lease_payload(), None)

    with pytest.raises(FatalPersistenceError, match="control_commit_timeout"):
        await writer.commit_control(command)

    assert writer.is_degraded
    release_commit.set()
    await writer.wait_until_idle()
    assert not task.done()
    await writer.drain(2)
    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("future_state", ["pending", "done", "cancelled"])
async def test_commit_control_rejects_every_caller_owned_future_before_enqueue(
    tmp_path: Path, future_state: str
) -> None:
    writer = PersistenceWriter(
        tmp_path / f"caller-future-{future_state}.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        control_commit_timeout_seconds=0.01,
    )
    caller_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    if future_state == "done":
        caller_future.set_result(None)
    elif future_state == "cancelled":
        caller_future.cancel()

    command = PersistenceCommand("lease", lease_payload(), caller_future)
    for _ in range(2):
        with pytest.raises(CommandSerializationError, match="command_future_must_be_unset"):
            await writer.commit_control(command)

    assert writer.queue_size == 0
    assert writer.is_degraded is False


@pytest.mark.asyncio
async def test_connection_is_opened_once_inside_run_and_sql_executes_on_owner_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect_calls: list[Path] = []
    owner_tasks: list[asyncio.Task[object] | None] = []
    real_connect = aiosqlite.connect

    def tracking_connect(database: str | Path, *args: object, **kwargs: object) -> Any:
        connect_calls.append(Path(database))
        return real_connect(database, *args, **kwargs)

    def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            owner_tasks.append(asyncio.current_task())

    monkeypatch.setattr(aiosqlite, "connect", tracking_connect)
    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        failpoint=failpoint,
    )
    assert connect_calls == []

    run_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    assert connect_calls == [tmp_path / "voice.sqlite"]
    await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    assert owner_tasks == [run_task]
    await stop_writer(writer, run_task)
    assert connect_calls == [tmp_path / "voice.sqlite"]


@pytest.mark.asyncio
async def test_startup_and_periodic_quick_check_are_owner_executed_without_public_sql(
    tmp_path: Path,
) -> None:
    clock_value = 10.0
    checks: list[float] = []
    second_check = asyncio.Event()

    def clock() -> float:
        return clock_value

    def on_check(when: float) -> None:
        checks.append(when)
        if len(checks) == 2:
            second_check.set()

    writer, task = await start_writer(
        tmp_path / "voice.sqlite",
        monotonic=clock,
        quick_check_interval_seconds=5.0,
        quick_check_observer=on_check,
    )
    assert await writer.quick_check() is True
    assert checks == [10.0]

    clock_value = 16.0
    await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    await asyncio.wait_for(second_check.wait(), timeout=1)
    assert checks == [10.0, 16.0]
    assert await writer.quick_check() is True
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_idle_periodic_quick_check_runs_without_commands(tmp_path: Path) -> None:
    second_check = asyncio.Event()
    count = 0

    def observe(_: float) -> None:
        nonlocal count
        count += 1
        if count == 2:
            second_check.set()

    writer, task = await start_writer(
        tmp_path / "voice.sqlite",
        quick_check_interval_seconds=0.01,
        quick_check_observer=observe,
    )
    await asyncio.wait_for(second_check.wait(), timeout=1)
    assert await writer.quick_check() is True
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_command_just_before_due_check_waits_only_the_remaining_interval(
    tmp_path: Path,
) -> None:
    clock_value = 0.0
    checks: list[float] = []
    second_check = asyncio.Event()

    def clock() -> float:
        return clock_value

    def observe(when: float) -> None:
        checks.append(when)
        if len(checks) == 2:
            second_check.set()

    writer, task = await start_writer(
        tmp_path / "remaining-interval.sqlite",
        monotonic=clock,
        quick_check_interval_seconds=1.0,
        quick_check_observer=observe,
    )
    try:
        clock_value = 0.999
        await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
        clock_value = 1.0

        await asyncio.wait_for(second_check.wait(), timeout=0.2)
        assert checks == [0.0, 1.0]
    finally:
        await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_pending_age_above_one_second_is_fatal_without_processing(tmp_path: Path) -> None:
    now = 1.0
    faults: list[object] = []

    def clock() -> float:
        return now

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        monotonic=clock,
        fatal_handler=faults.append,
    )
    assert writer.try_enqueue_turn(turn_operation())
    now = 2.001

    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    await asyncio.wait_for(writer.fatal_event.wait(), timeout=1)
    await asyncio.wait_for(task, timeout=1)

    assert len(faults) == 1
    assert faults[0].code == "queue_oldest_age_exceeded"  # type: ignore[attr-defined]
    with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)


@pytest.mark.asyncio
async def test_db_plus_journal_cap_rolls_back_before_commit_and_completes_failure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"
    max_storage_bytes = 268_435_456
    mutation_has_run = False

    def file_size(path: Path) -> int:
        if mutation_has_run and path == database:
            return max_storage_bytes + 1
        return 0

    def failpoint(name: str) -> None:
        nonlocal mutation_has_run
        if name == "after_mutation_before_commit":
            mutation_has_run = True

    writer, task = await start_writer(
        database,
        file_size=file_size,
        max_storage_bytes=max_storage_bytes,
        failpoint=failpoint,
    )
    with pytest.raises(FatalPersistenceError, match="storage_limit_exceeded"):
        await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    await asyncio.wait_for(writer.wait_until_idle(), timeout=1)
    await asyncio.wait_for(task, timeout=1)
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "storage_limit_exceeded"
    assert writer.queue_size == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)


@pytest.mark.asyncio
async def test_queue_age_watchdog_fires_while_sqlite_owner_is_stuck_and_cleans_up(
    tmp_path: Path,
) -> None:
    clock_value = 0.0
    owner_stuck = asyncio.Event()
    release_owner = asyncio.Event()

    def clock() -> float:
        return clock_value

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            owner_stuck.set()
            await release_owner.wait()

    writer, run_task = await start_writer(
        tmp_path / "watchdog.sqlite",
        monotonic=clock,
        failpoint=failpoint,
    )
    first_commit = asyncio.create_task(
        writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    )
    await owner_stuck.wait()
    assert writer.try_enqueue_turn(turn_operation())
    clock_value = 1.001

    try:
        await asyncio.wait_for(writer.fatal_event.wait(), timeout=0.2)
        assert writer.fatal_fault is not None
        assert writer.fatal_fault.code == "queue_oldest_age_exceeded"
    finally:
        release_owner.set()
        await asyncio.gather(first_commit, return_exceptions=True)
        await asyncio.wait_for(run_task, timeout=1)

    assert all(
        task.get_name() != "voice-persistence-queue-watchdog"
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_webhook_receipt_and_effect_are_atomic_and_duplicate_is_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(database)
    command = PersistenceCommand(
        "webhook_effect",
        {"receipt": receipt_payload(), "lease": lease_payload(), "operation": operation()},
        None,
    )

    await writer.commit_control(command)
    await writer.commit_control(command)
    await stop_writer(writer, task)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)


@pytest.mark.asyncio
async def test_duplicate_webhook_semantic_fingerprint_conflict_fails_before_effect(
    tmp_path: Path,
) -> None:
    database = tmp_path / "webhook-fingerprint.sqlite"
    writer, task = await start_writer(database)
    await writer.commit_control(
        PersistenceCommand(
            "webhook_effect",
            {"receipt": receipt_payload(), "lease": None, "operation": None},
            None,
        )
    )
    changed = {**receipt_payload(), "semantic_fingerprint_sha256": b"z" * 32}

    with pytest.raises(CommandConflictError, match="webhook_identity_conflict"):
        await writer.commit_control(
            PersistenceCommand(
                "webhook_effect",
                {"receipt": changed, "lease": lease_payload(), "operation": operation()},
                None,
            )
        )
    await task

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)


@pytest.mark.asyncio
async def test_enrichment_fingerprint_must_match_complete_canonical_operation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "enrichment-fingerprint.sqlite"
    writer, task = await start_writer(database)
    receipt = {
        **receipt_payload(),
        "event_type": "call.recording.saved",
        "call_control_id": None,
    }
    await writer.commit_control(
        PersistenceCommand(
            "webhook_effect",
            {"receipt": receipt, "lease": None, "operation": None},
            None,
        )
    )

    with pytest.raises(FatalPersistenceError, match="invalid_webhook_enrichment"):
        await writer.commit_control(
            PersistenceCommand(
                "webhook_enrichment",
                {
                    "receipt": {
                        key: receipt[key]
                        for key in (
                            "event_id",
                            "event_type",
                            "call_control_id",
                            "occurred_at",
                            "semantic_fingerprint_sha256",
                        )
                    },
                    "enrichment_fingerprint_sha256": b"wrong-fingerprint".ljust(32, b"!"),
                    "operation": recording_operation(),
                },
                None,
            )
        )
    await task
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)
        assert connection.execute(
            "SELECT provider_enrichment_fingerprint_sha256 FROM webhook_receipts"
        ).fetchone() == (None,)


@pytest.mark.asyncio
async def test_injected_failure_rolls_back_receipt_and_effect_and_completes_future(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"

    def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            raise OSError("synthetic failpoint")

    writer, task = await start_writer(database, failpoint=failpoint)
    command = PersistenceCommand(
        "webhook_effect",
        {"receipt": receipt_payload(), "lease": lease_payload(), "operation": operation()},
        None,
    )

    with pytest.raises(FatalPersistenceError):
        await writer.commit_control(command)
    await asyncio.wait_for(writer.wait_until_idle(), timeout=1)
    await asyncio.wait_for(task, timeout=1)

    with sqlite3.connect(database) as connection:
        for table in ("webhook_receipts", "call_leases", "outbox"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


@pytest.mark.asyncio
async def test_duplicate_operation_is_idempotent_only_for_identical_canonical_content(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(database)
    operation_id = uuid4()
    first = turn_operation(1, operation_id=operation_id)
    assert writer.try_enqueue_turn(first)
    await writer.wait_until_idle()
    assert writer.try_enqueue_turn(first)
    await writer.wait_until_idle()

    conflicting = turn_operation(2, operation_id=operation_id)
    assert writer.try_enqueue_turn(conflicting)
    await asyncio.wait_for(writer.fatal_event.wait(), timeout=1)
    await asyncio.wait_for(task, timeout=1)

    assert isinstance(writer.fatal_exception, CommandConflictError)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)


def test_canonical_json_aad_and_encrypted_command_64_kib_boundary() -> None:
    candidate = operation(call_control_id="unicode-é")
    dumped = canonical_operation_bytes(candidate)
    decoded = json.loads(dumped)

    assert dumped == json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    aad = operation_aad(candidate)
    assert aad.startswith(b"projetv0-voice/outbox/aes-256-gcm/v1\x00")
    assert b'"operation_id":"00000000-0000-0000-0000-000000000001"' in aad
    assert b"unicode" not in aad

    base = operation(call_control_id="x")
    target_plaintext_size = 65_536 - len(operation_aad(base)) - 12 - 16
    required_identifier_size = 1 + target_plaintext_size - len(canonical_operation_bytes(base))
    exact = operation(call_control_id="x" * required_identifier_size)
    encrypted = encrypt_operation(exact, CryptoKeyring({1: KEY}, active_version=1))
    assert encrypted.envelope_size == 65_536

    oversized = operation(call_control_id="x" * (required_identifier_size + 1))
    with pytest.raises(EncryptedCommandTooLarge, match="encrypted_command_too_large"):
        encrypt_operation(oversized, CryptoKeyring({1: KEY}, active_version=1))


def test_persistence_command_repr_redacts_payload_and_future() -> None:
    sentinel = "COMMAND-CONTENT-SENTINEL"
    command = PersistenceCommand("outbox", {"operation": sentinel}, None)
    rendered = repr(command)
    assert sentinel not in rendered
    assert "payload" not in rendered
    assert "committed" not in rendered


@pytest.mark.asyncio
async def test_just_over_64_kib_is_fatal_and_never_inserted(tmp_path: Path) -> None:
    base = operation(call_control_id="x")
    target_plaintext_size = 65_536 - len(operation_aad(base)) - 12 - 16
    identifier_size = 1 + target_plaintext_size - len(canonical_operation_bytes(base))
    exact = operation(call_control_id="x" * identifier_size)
    oversized = operation(
        operation_id=UUID(int=2), call_control_id="x" * (identifier_size + 1)
    )
    database = tmp_path / "size.sqlite"
    writer, task = await start_writer(database)

    await writer.commit_control(
        PersistenceCommand("outbox", {"operation": exact}, None)
    )
    with pytest.raises(FatalPersistenceError, match="encrypted_command_too_large"):
        await writer.commit_control(
            PersistenceCommand("outbox", {"operation": oversized}, None)
        )
    await asyncio.wait_for(task, timeout=1)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT op_id FROM outbox").fetchall() == [
            (str(exact.operation_id),)
        ]


@pytest.mark.asyncio
async def test_schema_has_exact_v3_tables_required_columns_checks_and_delete_journal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(database)
    assert writer.pragma_state == {
        "foreign_keys": 1,
        "journal_mode": "delete",
        "synchronous": 3,
    }
    await stop_writer(writer, task)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        ddl = "\n".join(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
            )
        ).lower()
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        user_version = connection.execute("PRAGMA user_version").fetchone()

    assert tables == {
        "call_leases",
        "webhook_receipts",
        "outbox",
        "qualification_runs",
    }
    assert "natural_key" not in ddl
    assert "delivered_at" not in ddl
    assert "generic" not in ddl
    for column in ("turn_id", "recording_id", "crypto_version", "key_version", "nonce"):
        assert column in ddl
    assert "check" in ddl
    assert "length(token_hash) = 32" in ddl
    assert "semantic_fingerprint_sha256" in ddl
    assert "provider_enrichment_fingerprint_sha256" in ddl
    assert "length(semantic_fingerprint_sha256) = 32" in ddl
    assert journal_mode == ("delete",)
    assert "lifecycle_json" in ddl
    assert user_version == (3,)


@pytest.mark.parametrize("legacy_version", [1, 2])
@pytest.mark.asyncio
async def test_forward_lifecycle_migration_preserves_every_legacy_outbox_byte(
    tmp_path, legacy_version
):
    from projetv0_voice.persistence.schema import V1_SCHEMA_SQL, V2_SCHEMA_SQL

    legacy = operation()
    prepared = encrypt_operation(legacy, CryptoKeyring({1: KEY}, active_version=1))
    database = tmp_path / f"legacy-{legacy_version}.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(V1_SCHEMA_SQL if legacy_version == 1 else V2_SCHEMA_SQL)
        connection.execute(
            "INSERT INTO outbox (op_id,deployment_id,kind,schema_version,call_id,"
            "turn_id,recording_id,crypto_version,key_version,nonce,ciphertext,"
            "created_at,attempts,next_attempt_at,last_error_code) "
            "VALUES (?,?,?,?,?,NULL,NULL,1,?,?,?, ?,0,?,NULL)",
            (
                str(legacy.operation_id),
                legacy.deployment_id,
                legacy.kind,
                1,
                str(legacy.call_id),
                prepared.encrypted.key_version,
                prepared.encrypted.nonce,
                prepared.encrypted.ciphertext,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        before = connection.execute("SELECT * FROM outbox").fetchall()
    writer, task = await start_writer(database)
    assert await writer.read_call_lifecycle(legacy.call_id) is None
    await stop_writer(writer, task)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM outbox").fetchall() == before
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
        assert "lifecycle_json" in {
            row[1] for row in connection.execute("PRAGMA table_info(call_leases)")
        }


@pytest.mark.asyncio
async def test_original_admission_metadata_receipt_lease_and_outbox_roll_back_together(tmp_path):
    from projetv0_voice.persistence.writer import LocalCallAdmissionFacts

    database = tmp_path / "atomic-admission.sqlite"

    async def failpoint(name):
        if name == "after_mutation_before_commit":
            raise sqlite3.OperationalError("owned fixture failpoint")

    writer, task = await start_writer(database, failpoint=failpoint)
    original = operation(call_control_id="control-1")
    original = original.model_copy(
        update={
            "payload": original.payload.model_copy(
                update={"retention_until": NOW + timedelta(days=30)}
            )
        }
    )
    lease = {**lease_payload(), "call_id": original.call_id}
    facts = LocalCallAdmissionFacts(original.call_id, NOW, NOW + timedelta(days=30), None, None)
    ticket = writer.submit_webhook(
        receipt=receipt_payload(), lease=lease, operation=original, admission_facts=facts
    )
    with pytest.raises(FatalPersistenceError):
        await ticket.wait()
    await task
    with sqlite3.connect(database) as connection:
        for table in ("call_leases", "webhook_receipts", "outbox"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["version_2", "user_version_0_with_tables", "superset", "weakened", "missing_index"],
)
async def test_existing_schema_must_be_exact_v1_and_is_never_repaired(
    tmp_path: Path, mutation: str
) -> None:
    database = tmp_path / f"schema-{mutation}.sqlite"
    with sqlite3.connect(database) as connection:
        if mutation == "version_2":
            connection.execute("PRAGMA user_version = 2")
        else:
            schema = SCHEMA_SQL
            if mutation == "weakened":
                schema = schema.replace(
                    "CHECK (length(token_hash) = 32)",
                    "CHECK (length(token_hash) > 0)",
                )
            elif mutation == "missing_index":
                schema = schema.replace(
                    """CREATE INDEX IF NOT EXISTS outbox_due_fifo_idx
    ON outbox (deployment_id, queue_id, next_attempt_at);
""",
                    "",
                )
            connection.executescript(schema)
            if mutation == "user_version_0_with_tables":
                connection.execute("PRAGMA user_version = 0")
            elif mutation == "superset":
                connection.execute("CREATE TABLE unexpected_table (id INTEGER PRIMARY KEY)")
        original_version = connection.execute("PRAGMA user_version").fetchone()

    writer = PersistenceWriter(database, CryptoKeyring({1: KEY}, active_version=1))
    task = asyncio.create_task(writer.run())
    ready = await writer.wait_ready()
    if ready:
        await stop_writer(writer, task)
    else:
        await task

    assert ready is False
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "sqlite_schema_mismatch"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == original_version


@pytest.mark.asyncio
async def test_non_ok_owner_quick_check_fails_startup_closed(tmp_path: Path) -> None:
    writer = PersistenceWriter(
        tmp_path / "non-ok.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        quick_check_result=lambda: "row 7 is malformed",
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is False
    await task
    assert await writer.quick_check() is False
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "quick_check_failed"


@pytest.mark.asyncio
async def test_non_ok_idle_periodic_quick_check_fails_closed(tmp_path: Path) -> None:
    checks = iter(("ok", "not ok"))
    second_check = asyncio.Event()

    def result() -> str:
        return next(checks)

    def observe(_: float) -> None:
        if not second_check.is_set():
            second_check.set()

    writer = PersistenceWriter(
        tmp_path / "periodic-non-ok.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        quick_check_interval_seconds=0.01,
        quick_check_observer=observe,
        quick_check_result=result,
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    second_check.clear()
    await asyncio.wait_for(writer.fatal_event.wait(), timeout=1)
    await task
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "quick_check_failed"


@pytest.mark.asyncio
async def test_relay_claim_retry_and_ack_stay_fifo_and_ack_deletes(tmp_path: Path) -> None:
    database = tmp_path / "relay.sqlite"
    writer, task = await start_writer(database, utcnow=lambda: NOW)
    assert writer.try_enqueue_turn(turn_operation(1))
    assert writer.try_enqueue_turn(turn_operation(2))
    await writer.wait_until_idle()

    batch = await writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=5)
    assert [item.queue_id for item in batch] == [1, 2]
    assert all(item.created_at == NOW for item in batch)
    assert all(item.claim_attempt == 1 for item in batch)
    assert all(item.claim_expires_at == NOW + timedelta(seconds=5) for item in batch)
    retry_result = await writer.retry_outbox(
        queue_id=1,
        expected_claim_attempt=batch[0].claim_attempt,
        next_attempt_at=NOW + timedelta(seconds=5),
        error_code="postgres_unavailable",
    )
    assert retry_result.applied is True
    batch = await writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=5)
    assert batch == ()
    batch = await writer.read_relay_batch(
        batch_size=10, now=NOW + timedelta(seconds=5), lease_seconds=5
    )
    assert [item.queue_id for item in batch] == [1, 2]
    assert all(item.claim_attempt == 2 for item in batch)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT attempts, next_attempt_at, last_error_code FROM outbox WHERE queue_id=1"
        ).fetchone() == (2, "2026-08-25T12:00:10Z", "postgres_unavailable")
    ack_result = await writer.ack_outbox(
        queue_id=1,
        expected_claim_attempt=batch[0].claim_attempt,
    )
    assert ack_result.applied is True
    batch = await writer.read_relay_batch(
        batch_size=10, now=NOW + timedelta(seconds=10), lease_seconds=5
    )
    assert [item.queue_id for item in batch] == [2]
    assert batch[0].claim_attempt == 3
    ack_result = await writer.ack_outbox(
        queue_id=2,
        expected_claim_attempt=batch[0].claim_attempt,
    )
    assert ack_result.applied is True
    assert await writer.read_relay_batch(
        batch_size=10, now=NOW + timedelta(seconds=15), lease_seconds=5
    ) == ()

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox WHERE queue_id=2").fetchone() == (
            0,
        )
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_stale_relay_worker_cannot_ack_or_retry_a_newer_claim_in_same_process(
    tmp_path: Path,
) -> None:
    database = tmp_path / "claim-fence.sqlite"
    writer, task = await start_writer(database, utcnow=lambda: NOW)
    assert writer.try_enqueue_turn(turn_operation(1))
    await writer.wait_until_idle()
    claim_a = (
        await writer.read_relay_batch(batch_size=1, now=NOW, lease_seconds=5)
    )[0]
    claim_b = (
        await writer.read_relay_batch(
            batch_size=1,
            now=NOW + timedelta(seconds=5),
            lease_seconds=5,
        )
    )[0]
    assert claim_a.claim_expires_at == NOW + timedelta(seconds=5)
    assert claim_b.claim_expires_at == NOW + timedelta(seconds=10)
    assert claim_a.claim_attempt == 1
    assert claim_b.claim_attempt == 2

    current_retry = await writer.retry_outbox(
        queue_id=claim_b.queue_id,
        expected_claim_attempt=claim_b.claim_attempt,
        next_attempt_at=claim_a.claim_expires_at,
        error_code="current_worker_retry",
    )
    assert current_retry.applied is True

    stale_ack = await writer.ack_outbox(
        queue_id=claim_a.queue_id,
        expected_claim_attempt=claim_a.claim_attempt,
    )
    stale_retry = await writer.retry_outbox(
        queue_id=claim_a.queue_id,
        expected_claim_attempt=claim_a.claim_attempt,
        next_attempt_at=NOW + timedelta(seconds=30),
        error_code="stale_worker",
    )

    assert stale_ack.applied is False
    assert stale_retry.applied is False
    assert writer.fatal_fault is None
    assert writer.is_degraded is False
    assert task.done() is False
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT attempts, next_attempt_at, last_error_code FROM outbox WHERE queue_id = ?",
            (claim_b.queue_id,),
        ).fetchone() == (2, "2026-08-25T12:00:05Z", "current_worker_retry")

    current_ack = await writer.ack_outbox(
        queue_id=claim_b.queue_id,
        expected_claim_attempt=claim_b.claim_attempt,
    )
    assert current_ack.applied is True
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_relay_mutations_require_a_positive_exact_claim_attempt(
    tmp_path: Path,
) -> None:
    writer, task = await start_writer(tmp_path / "claim-input.sqlite", utcnow=lambda: NOW)
    assert writer.try_enqueue_turn(turn_operation(1))
    await writer.wait_until_idle()
    claim = (await writer.read_relay_batch(batch_size=1, now=NOW, lease_seconds=30))[0]

    try:
        with pytest.raises(TypeError):
            await writer.ack_outbox(queue_id=claim.queue_id)  # type: ignore[call-arg]
        for invalid_attempt in (True, 0, -1, 1.0, "1"):
            with pytest.raises(ValueError, match="expected_claim_attempt"):
                await writer.retry_outbox(
                    queue_id=claim.queue_id,
                    expected_claim_attempt=invalid_attempt,  # type: ignore[arg-type]
                    next_attempt_at=NOW + timedelta(seconds=60),
                    error_code="invalid_input",
                )
        assert writer.fatal_fault is None
        assert task.done() is False
    finally:
        if not task.done():
            await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_public_relay_read_decrypts_each_selected_row_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_calls = 0
    decrypt_calls = 0
    real_claim = PersistenceWriter._claim_batch
    real_decrypt = CryptoKeyring.decrypt

    async def tracking_claim(
        self: PersistenceWriter, payload: dict[str, object]
    ) -> None:
        nonlocal claim_calls
        claim_calls += 1
        await real_claim(self, payload)

    def tracking_decrypt(self: CryptoKeyring, value: Any, *, aad: bytes) -> bytes:
        nonlocal decrypt_calls
        decrypt_calls += 1
        return real_decrypt(self, value, aad=aad)

    monkeypatch.setattr(PersistenceWriter, "_claim_batch", tracking_claim)
    monkeypatch.setattr(CryptoKeyring, "decrypt", tracking_decrypt)
    writer, task = await start_writer(tmp_path / "single-read.sqlite", utcnow=lambda: NOW)
    assert writer.try_enqueue_turn(turn_operation(1))
    assert writer.try_enqueue_turn(turn_operation(2))
    await writer.wait_until_idle()

    batch = await writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=30)

    assert [item.queue_id for item in batch] == [1, 2]
    assert claim_calls == 1
    assert decrypt_calls == 2
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_concurrent_relay_readers_claim_each_row_once_and_lease_is_bounded(
    tmp_path: Path,
) -> None:
    writer, task = await start_writer(tmp_path / "claims.sqlite", utcnow=lambda: NOW)
    assert writer.try_enqueue_turn(turn_operation(1))
    assert writer.try_enqueue_turn(turn_operation(2))
    await writer.wait_until_idle()

    first, second = await asyncio.gather(
        writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=30),
        writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=30),
    )
    claimed, empty = sorted((first, second), key=len, reverse=True)

    assert [item.queue_id for item in claimed] == [1, 2]
    assert empty == ()
    assert all(item.claim_attempt == 1 for item in claimed)
    assert all(item.created_at == NOW for item in claimed)
    assert all(item.next_attempt_at == NOW for item in claimed)
    assert all(item.claim_expires_at == NOW + timedelta(seconds=30) for item in claimed)
    for invalid_lease in (True, 0, 301):
        with pytest.raises(ValueError, match="lease_seconds"):
            await writer.read_relay_batch(
                batch_size=10,
                now=NOW,
                lease_seconds=invalid_lease,  # type: ignore[arg-type]
            )
    await stop_writer(writer, task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_token",
    [b"raw-token", b"x" * 31, b"x" * 33, "RAW-TOKEN-SENTINEL", bytearray(32)],
)
async def test_lease_token_hash_must_be_exact_sha256_bytes_without_leaking(
    tmp_path: Path, invalid_token: object
) -> None:
    fixture_name = (
        f"invalid-token-{type(invalid_token).__name__}-{len(invalid_token)}.sqlite"  # type: ignore[arg-type]
    )
    database = tmp_path / fixture_name
    writer, task = await start_writer(database)
    payload = lease_payload()
    payload["token_hash"] = invalid_token

    with pytest.raises(FatalPersistenceError, match="invalid_lease_command") as raised:
        await writer.commit_control(PersistenceCommand("lease", payload, None))
    await asyncio.wait_for(task, timeout=1)

    rendered = repr(raised.value) + repr(writer.fatal_fault)
    assert "RAW-TOKEN-SENTINEL" not in rendered
    assert "raw-token" not in rendered
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)


@pytest.mark.asyncio
async def test_cleanup_bounds_receipts_to_7_days_and_closed_leases_to_24_hours(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cleanup.sqlite"
    writer, task = await start_writer(database, utcnow=lambda: NOW)
    old_receipt = receipt_payload("old-event")
    old_receipt["received_at"] = NOW - timedelta(days=8)
    terminal = lease_payload(
        call_control_id="old-terminal",
        state="terminal",
        expires_at=NOW - timedelta(days=2),
    )
    terminal["created_at"] = NOW - timedelta(days=3)
    terminal["closed_at"] = NOW - timedelta(hours=25)
    pending = dict(terminal)
    pending["state"] = "pending"
    pending["closed_at"] = None
    await writer.commit_control(PersistenceCommand("lease", pending, None))
    await writer.commit_control(
        PersistenceCommand(
            "webhook_effect",
            {"receipt": old_receipt, "lease": terminal, "operation": None},
            None,
        )
    )

    deleted = await writer.cleanup_local_state(now=NOW)
    assert deleted.receipts == 1
    assert deleted.leases == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_illegal_or_regressive_lease_transition_is_permanent_safe_conflict(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lease.sqlite"
    writer, task = await start_writer(database)
    await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    await writer.commit_control(PersistenceCommand("lease", lease_payload(state="terminal"), None))

    with pytest.raises(CommandConflictError, match="lease_transition_conflict"):
        await writer.commit_control(
            PersistenceCommand("lease", lease_payload(state="active"), None)
        )
    await asyncio.wait_for(task, timeout=1)
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "lease_transition_conflict"


@pytest.mark.asyncio
async def test_duplicate_webhook_event_with_different_identity_is_fatal_conflict(
    tmp_path: Path,
) -> None:
    database = tmp_path / "webhook.sqlite"
    writer, task = await start_writer(database)
    await writer.commit_control(
        PersistenceCommand(
            "webhook_effect",
            {"receipt": receipt_payload(), "lease": None, "operation": None},
            None,
        )
    )
    conflicting = receipt_payload()
    conflicting["event_type"] = "call.answered"

    with pytest.raises(CommandConflictError, match="webhook_identity_conflict"):
        await writer.commit_control(
            PersistenceCommand(
                "webhook_effect",
                {"receipt": conflicting, "lease": None, "operation": None},
                None,
            )
        )
    await asyncio.wait_for(task, timeout=1)
    assert writer.fatal_fault is not None


@pytest.mark.asyncio
async def test_fatal_sqlite_and_os_errors_are_constant_safe_and_rollback(
    tmp_path: Path,
) -> None:
    cases: list[tuple[BaseException, str]] = []
    for sqlite_code, expected in (
        (sqlite3.SQLITE_FULL, "sqlite_full"),
        (sqlite3.SQLITE_CORRUPT, "sqlite_corrupt"),
        (sqlite3.SQLITE_IOERR, "sqlite_ioerr"),
    ):
        error = sqlite3.OperationalError("RAW-SQLITE-SENTINEL")
        error.sqlite_errorcode = sqlite_code  # type: ignore[attr-defined]
        cases.append((error, expected))
    cases.append((OSError(errno.ENOSPC, "RAW-ENOSPC-SENTINEL"), "storage_enospc"))

    for index, (injected, expected) in enumerate(cases):
        faults: list[object] = []

        def failpoint(name: str, *, error: BaseException = injected) -> None:
            if name == "after_mutation_before_commit":
                raise error

        database = tmp_path / f"fatal-{index}.sqlite"
        writer, task = await start_writer(
            database, failpoint=failpoint, fatal_handler=faults.append
        )
        with pytest.raises(FatalPersistenceError, match=expected):
            await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
        await asyncio.wait_for(task, timeout=1)
        assert len(faults) == 1
        rendered = repr(faults[0]) + repr(writer.fatal_exception)
        assert "RAW-SQLITE-SENTINEL" not in rendered
        assert "RAW-ENOSPC-SENTINEL" not in rendered
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (0,)


@pytest.mark.asyncio
async def test_writer_task_death_fails_all_queued_futures_once(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    faults: list[object] = []

    class SyntheticWriterDeath(BaseException):
        pass

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            entered.set()
            await release.wait()
            raise SyntheticWriterDeath()

    writer, run_task = await start_writer(
        tmp_path / "death.sqlite", failpoint=failpoint, fatal_handler=faults.append
    )
    first = asyncio.create_task(
        writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    )
    await entered.wait()
    second = asyncio.create_task(
        writer.commit_control(
            PersistenceCommand(
                "lease", lease_payload(call_control_id="control-2"), None
            )
        )
    )
    release.set()

    results = await asyncio.wait_for(
        asyncio.gather(first, second, return_exceptions=True), timeout=1
    )
    await asyncio.wait_for(run_task, timeout=1)
    assert all(isinstance(result, FatalPersistenceError) for result in results)
    assert len(faults) == 1
    assert writer.queue_size == 0


@pytest.mark.asyncio
async def test_oldest_outbox_timestamp_empty_and_includes_non_due_claimed_all_deployments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "oldest.sqlite"
    current = NOW
    writer, task = await start_writer(database, utcnow=lambda: current)
    assert await writer.oldest_outbox_created_at() is None
    await writer.commit_control(PersistenceCommand("outbox", {"operation": operation(1)}, None))
    current = NOW + timedelta(seconds=10)
    await writer.commit_control(
        PersistenceCommand(
            "outbox",
            {"operation": operation(2).model_copy(update={"deployment_id": "agent-b"})},
            None,
        )
    )
    claimed = await writer.read_relay_batch(batch_size=1, now=current, lease_seconds=30)
    assert len(claimed) == 1
    second_claim = await writer.read_relay_batch(
        batch_size=1, now=current, lease_seconds=30
    )
    assert len(second_claim) == 1
    retried = await writer.retry_outbox(
        queue_id=2,
        expected_claim_attempt=1,
        next_attempt_at=current + timedelta(hours=1),
        error_code="not_due",
    )
    assert retried.applied is True

    def forbidden_decrypt(*_: object, **__: object) -> bytes:
        raise AssertionError("oldest timestamp must not decrypt payloads")

    monkeypatch.setattr(CryptoKeyring, "decrypt", forbidden_decrypt)
    assert await writer.oldest_outbox_created_at() == NOW
    await stop_writer(writer, task)


@pytest.mark.asyncio
async def test_oldest_outbox_runs_on_owner_and_resolves_only_after_command_commit(
    tmp_path: Path,
) -> None:
    reached_commit = asyncio.Event()
    release_commit = asyncio.Event()
    owner_tasks: list[asyncio.Task[object] | None] = []
    block_oldest = False

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit" and block_oldest:
            owner_tasks.append(asyncio.current_task())
            reached_commit.set()
            await release_commit.wait()

    writer, owner = await start_writer(tmp_path / "oldest-owner.sqlite", failpoint=failpoint)
    block_oldest = True
    pending = asyncio.create_task(writer.oldest_outbox_created_at())
    await reached_commit.wait()
    assert pending.done() is False
    assert owner_tasks == [owner]
    release_commit.set()
    assert await pending is None
    await stop_writer(writer, owner)


@pytest.mark.asyncio
async def test_cancelled_oldest_request_leaves_no_orphan_exceptional_future(
    tmp_path: Path,
) -> None:
    reached_commit = asyncio.Event()
    release_commit = asyncio.Event()

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            reached_commit.set()
            await release_commit.wait()

    writer, owner = await start_writer(tmp_path / "oldest-cancel.sqlite", failpoint=failpoint)
    pending = asyncio.create_task(writer.oldest_outbox_created_at())
    await reached_commit.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release_commit.set()
    await writer.wait_until_idle()
    assert owner.done() is False
    await stop_writer(writer, owner)


@pytest.mark.asyncio
async def test_cancelled_oldest_request_consumes_late_writer_failure(
    tmp_path: Path,
) -> None:
    reached_commit = asyncio.Event()
    release_commit = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            reached_commit.set()
            await release_commit.wait()
            raise RuntimeError("RAW-LATE-WRITER-SENTINEL")

    loop.set_exception_handler(lambda _loop, context: loop_errors.append(dict(context)))
    try:
        writer, owner = await start_writer(
            tmp_path / "oldest-cancel-late-failure.sqlite", failpoint=failpoint
        )
        pending = asyncio.create_task(writer.oldest_outbox_created_at())
        await reached_commit.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release_commit.set()
        await owner
        del pending
        gc.collect()
        await asyncio.sleep(0)
        assert loop_errors == []
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_invalid_stored_oldest_timestamp_is_constant_safe(tmp_path: Path) -> None:
    database = tmp_path / "oldest-invalid.sqlite"
    writer, owner = await start_writer(database)
    await writer.commit_control(PersistenceCommand("outbox", {"operation": operation(1)}, None))
    await stop_writer(writer, owner)
    sentinel = "RAW-TIMESTAMP-SENTINEL"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE outbox SET created_at = ?", (sentinel,))
        connection.commit()

    restarted, task = await start_writer(database)
    with pytest.raises(FatalPersistenceError, match="stored_datetime_invalid") as captured:
        await restarted.oldest_outbox_created_at()
    rendered = repr(captured.value)
    pending_traceback = captured.value.__traceback__
    while pending_traceback is not None:
        filename = pending_traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/projetv0_voice/persistence/writer.py" in filename:
            for value in pending_traceback.tb_frame.f_locals.values():
                rendered += repr(value)
        pending_traceback = pending_traceback.tb_next
    if captured.value.__cause__ is not None:
        rendered += repr(captured.value.__cause__)
    if captured.value.__context__ is not None:
        rendered += repr(captured.value.__context__)
    assert sentinel not in repr(captured.value)
    assert sentinel not in rendered
    assert restarted.fatal_fault is not None
    assert restarted.fatal_fault.code == "stored_datetime_invalid"
    assert task.done() is False
    await restarted.drain(2)
    await task
