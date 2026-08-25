from __future__ import annotations

import asyncio
import base64
import errno
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import aiosqlite
import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, TurnUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    EncryptedCommandTooLarge,
    FatalPersistenceError,
    PersistenceCommand,
    canonical_operation_bytes,
    encrypt_operation,
    operation_aad,
)
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
async def test_fifo_queue_ids_and_plaintext_never_reach_database_or_journal(tmp_path: Path) -> None:
    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(database)
    sentinel = "PLAINTEXT-TRANSCRIPT-SENTINEL-9D17"
    encoded_sentinel = base64.b64encode(sentinel.encode())

    assert writer.try_enqueue_turn(turn_operation(1, encoded_content=encoded_sentinel))
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
        assert encoded_sentinel not in journal_bytes
    assert encoded_sentinel not in database_bytes


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
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    command = PersistenceCommand("lease", lease_payload(), future)

    with pytest.raises(FatalPersistenceError, match="control_commit_timeout"):
        await writer.commit_control(command)

    assert not future.done()
    assert writer.is_degraded
    release_commit.set()
    await asyncio.wait_for(future, timeout=1)
    await writer.wait_until_idle()
    assert not task.done()
    await writer.drain(2)
    await task


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

    def clock() -> float:
        return clock_value

    def on_check(when: float) -> None:
        checks.append(when)

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
async def test_db_plus_journal_cap_is_owner_checked_with_injected_sizes(tmp_path: Path) -> None:
    sizes: dict[str, int] = {}

    def file_size(path: Path) -> int:
        return sizes.get(str(path), 0)

    database = tmp_path / "voice.sqlite"
    writer, task = await start_writer(
        database,
        file_size=file_size,
        max_storage_bytes=268_435_456,
    )
    assert writer.max_storage_bytes == 268_435_456
    sizes[str(database)] = 200_000_000
    sizes[f"{database}-journal"] = 68_435_457

    with pytest.raises(FatalPersistenceError, match="storage_limit_exceeded"):
        await writer.commit_control(PersistenceCommand("lease", lease_payload(), None))
    await asyncio.wait_for(writer.fatal_event.wait(), timeout=1)
    await asyncio.wait_for(task, timeout=1)
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "storage_limit_exceeded"


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
async def test_injected_failure_rolls_back_receipt_and_effect_and_completes_future(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"

    def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            raise OSError("synthetic failpoint")

    writer, task = await start_writer(database, failpoint=failpoint)
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    command = PersistenceCommand(
        "webhook_effect",
        {"receipt": receipt_payload(), "lease": lease_payload(), "operation": operation()},
        future,
    )

    with pytest.raises(FatalPersistenceError):
        await writer.commit_control(command)
    assert future.done()
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
async def test_schema_has_exact_three_tables_required_columns_checks_and_delete_journal(
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

    assert tables == {"call_leases", "webhook_receipts", "outbox"}
    assert "natural_key" not in ddl
    assert "delivered_at" not in ddl
    assert "generic" not in ddl
    for column in ("turn_id", "recording_id", "crypto_version", "key_version", "nonce"):
        assert column in ddl
    assert "check" in ddl
    assert journal_mode == ("delete",)
    assert user_version == (1,)


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
async def test_relay_read_retry_and_ack_stay_fifo_and_ack_deletes(tmp_path: Path) -> None:
    database = tmp_path / "relay.sqlite"
    writer, task = await start_writer(database, utcnow=lambda: NOW)
    assert writer.try_enqueue_turn(turn_operation(1))
    assert writer.try_enqueue_turn(turn_operation(2))
    await writer.wait_until_idle()

    batch = await writer.read_relay_batch(batch_size=10, now=NOW)
    assert [item.queue_id for item in batch] == [1, 2]
    await writer.retry_outbox(
        queue_id=1,
        next_attempt_at=NOW + timedelta(seconds=5),
        error_code="postgres_unavailable",
    )
    batch = await writer.read_relay_batch(batch_size=10, now=NOW)
    assert batch == ()
    batch = await writer.read_relay_batch(batch_size=10, now=NOW + timedelta(seconds=5))
    assert [item.queue_id for item in batch] == [1, 2]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT attempts, next_attempt_at, last_error_code FROM outbox WHERE queue_id=1"
        ).fetchone() == (1, "2026-08-25T12:00:05Z", "postgres_unavailable")
    await writer.ack_outbox(queue_id=1)
    batch = await writer.read_relay_batch(batch_size=10, now=NOW + timedelta(seconds=5))
    assert [item.queue_id for item in batch] == [2]
    await writer.ack_outbox(queue_id=2)
    assert await writer.read_relay_batch(
        batch_size=10, now=NOW + timedelta(seconds=5)
    ) == ()

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox WHERE queue_id=2").fetchone() == (
            0,
        )
    await stop_writer(writer, task)


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
