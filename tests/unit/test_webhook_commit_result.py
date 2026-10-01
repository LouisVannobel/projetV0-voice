from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    FatalPersistenceError,
    PersistenceCommand,
)
from projetv0_voice.persistence.schema import V1_SCHEMA_SQL
from projetv0_voice.persistence.writer import PersistenceWriter

KEY = bytes(range(32))
NOW = datetime(2026, 8, 29, 10, tzinfo=UTC)
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


async def _start_writer(path: Path) -> tuple[PersistenceWriter, asyncio.Task[None]]:
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1))
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is True
    return writer, task


async def _stop_writer(writer: PersistenceWriter, task: asyncio.Task[None]) -> None:
    await writer.drain(2)
    await task


def _receipt(event_id: str, fingerprint: bytes = b"f" * 32) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "call.initiated",
        "call_control_id": "call-control-a",
        "occurred_at": NOW,
        "received_at": NOW,
        "semantic_fingerprint_sha256": fingerprint,
    }


def _lease(state: str = "pending") -> dict[str, object]:
    return {
        "action": "upsert",
        "call_control_id": "call-control-a",
        "call_id": UUID("22222222-2222-4222-8222-222222222222"),
        "tenant_id": "tenant-a",
        "agent_id": "agent-a",
        "state": state,
        "token_hash": b"t" * 32,
        "created_at": NOW,
        "expires_at": NOW + timedelta(seconds=30),
        "closed_at": NOW + timedelta(seconds=1) if state == "terminal" else None,
    }


def _operation(operation_id: int = 1) -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(int=operation_id),
        deployment_id="agent-a",
        call_id=UUID("22222222-2222-4222-8222-222222222222"),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="call-control-a",
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
@pytest.mark.parametrize("legacy", [False, True])
async def test_writer_establishes_exact_sqlite_v4_before_readiness(
    tmp_path: Path, legacy: bool
) -> None:
    database = tmp_path / "voice.sqlite"
    if legacy:
        with sqlite3.connect(database) as connection:
            connection.executescript(V1_SCHEMA_SQL)

    writer, task = await _start_writer(database)
    await _stop_writer(writer, task)

    with sqlite3.connect(database) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        objects = {
            (row[0], row[1])
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        }

    assert version == (4,)
    assert objects == {
        ("index", "outbox_due_fifo_idx"),
        ("table", "call_leases"),
        ("table", "outbox"),
        ("table", "qualification_runs"),
        ("table", "webhook_receipts"),
        ("table", "sparra_turn_decisions"),
        ("table", "sparra_content_fences"),
        ("table", "sparra_publications"),
    }


@pytest.mark.asyncio
async def test_webhook_ticket_resolves_only_to_post_commit_closed_results(
    tmp_path: Path,
) -> None:
    database = tmp_path / "results.sqlite"
    writer, task = await _start_writer(database)

    first = writer.submit_webhook(
        receipt=_receipt("event-1"),
        lease=_lease(),
        operation=_operation(),
    )
    assert first.done() is False
    first_result = await first.wait()
    duplicate_result = await writer.submit_webhook(
        receipt=_receipt("event-1"),
        lease=_lease(),
        operation=_operation(),
    ).wait()

    await writer.commit_control(
        PersistenceCommand("lease", _lease("terminal"), None)
    )
    terminal_result = await writer.submit_webhook(
        receipt=_receipt("event-2", b"g" * 32),
        lease=_lease(),
        operation=_operation(2),
    ).wait()
    await _stop_writer(writer, task)

    assert (first_result.receipt, first_result.effect) == ("first", "applied")
    assert (duplicate_result.receipt, duplicate_result.effect) == (
        "duplicate",
        "duplicate",
    )
    assert (terminal_result.receipt, terminal_result.effect) == (
        "first",
        "existing_terminal",
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)


@pytest.mark.asyncio
async def test_candidate_latch_checks_duplicate_before_consumed_without_new_mutation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidate.sqlite"
    writer, task = await _start_writer(database)

    accepted = await writer.submit_webhook(
        receipt=_receipt("event-1"),
        lease=_lease(),
        operation=_operation(),
        qualification_run_id=RUN_ID,
    ).wait()
    assert (accepted.receipt, accepted.effect) == ("first", "applied")
    assert await writer.qualification_run_consumed(RUN_ID) is True

    duplicate = await writer.submit_webhook(
        receipt=_receipt("event-1"),
        lease=_lease(),
        operation=_operation(),
        qualification_run_id=RUN_ID,
    ).wait()
    consumed = await writer.submit_webhook(
        receipt=_receipt("event-2", b"g" * 32),
        lease={**_lease(), "call_control_id": "call-control-b"},
        operation=None,
        qualification_run_id=RUN_ID,
    ).wait()
    await _stop_writer(writer, task)

    assert (duplicate.receipt, duplicate.effect) == ("duplicate", "duplicate")
    assert type(consumed).__name__ == "QualificationRunConsumed"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM qualification_runs").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)


@pytest.mark.asyncio
async def test_v1_migration_failure_rolls_back_without_publishing_readiness(
    tmp_path: Path,
) -> None:
    database = tmp_path / "migration-rollback.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(V1_SCHEMA_SQL)

    def failpoint(name: str) -> None:
        if name == "after_v1_migration_before_commit":
            raise OSError("RAW-MIGRATION-SENTINEL")

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        failpoint=failpoint,
    )
    task = asyncio.create_task(writer.run())

    assert await writer.wait_ready() is False
    await task
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "sqlite_schema_mismatch"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'qualification_runs'"
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_webhook_ticket_survives_waiter_cancellation_and_never_resolves_precommit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "detached-ticket.sqlite"
    mutation_complete = asyncio.Event()
    release_commit = asyncio.Event()

    async def failpoint(name: str) -> None:
        if name == "after_mutation_before_commit":
            mutation_complete.set()
            await release_commit.wait()

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        failpoint=failpoint,
    )
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    ticket = writer.submit_webhook(
        receipt=_receipt("event-1"),
        lease=_lease(),
        operation=_operation(),
    )
    waiter = asyncio.create_task(ticket.wait())
    await mutation_complete.wait()
    assert ticket.done() is False

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert ticket.done() is False
    release_commit.set()
    result = await ticket.wait()
    assert (result.receipt, result.effect) == ("first", "applied")
    await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM call_leases").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)


@pytest.mark.asyncio
async def test_receipt_classifier_is_owner_queued_and_uses_only_exact_identity(
    tmp_path: Path,
) -> None:
    writer, task = await _start_writer(tmp_path / "classification.sqlite")

    assert (
        await writer.classify_webhook_receipt(
            event_id="event-1", semantic_fingerprint_sha256=b"f" * 32
        )
        == "missing"
    )
    await writer.submit_webhook(
        receipt=_receipt("event-1"), lease=None, operation=None
    ).wait()

    assert (
        await writer.classify_webhook_receipt(
            event_id="event-1", semantic_fingerprint_sha256=b"f" * 32
        )
        == "duplicate"
    )
    assert (
        await writer.classify_webhook_receipt(
            event_id="event-1", semantic_fingerprint_sha256=b"g" * 32
        )
        == "conflict"
    )
    assert (
        await writer.classify_webhook_receipt(
            event_id="event-2", semantic_fingerprint_sha256=b"f" * 32
        )
        == "missing"
    )
    await _stop_writer(writer, task)


@pytest.mark.asyncio
async def test_legacy_v1_fingerprint_alias_is_authoritative_without_rewriting(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-fingerprint.sqlite"
    legacy_fingerprint = bytes.fromhex(
        "ed766819519ec6f7ec9c479a643f9d72b232ac0d4722341cbfb4350ee812098c"
    )
    current_fingerprint = bytes.fromhex(
        "2759dd9333080e63bec2d57b9bbe5d02354c9cdfe052bd2b0c34a73eb6f5c13a"
    )
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
                "event-a",
                "call.initiated",
                "control-a",
                "2026-08-29T10:00:00Z",
                "2026-08-29T10:00:00Z",
                legacy_fingerprint,
            ),
        )

    writer, task = await _start_writer(database)

    classification = await writer.classify_webhook_receipt(
        event_id="event-a",
        semantic_fingerprint_sha256=current_fingerprint,
        legacy_v1_semantic_fingerprint_sha256=legacy_fingerprint,
    )
    duplicate = await writer.submit_webhook(
        receipt={
            "event_id": "event-a",
            "event_type": "call.initiated",
            "call_control_id": "control-a",
            "occurred_at": NOW,
            "received_at": NOW,
            "semantic_fingerprint_sha256": current_fingerprint,
        },
        lease=None,
        operation=None,
        legacy_v1_semantic_fingerprint_sha256=legacy_fingerprint,
    ).wait()
    fresh = await writer.submit_webhook(
        receipt=_receipt("event-b", b"n" * 32),
        lease=None,
        operation=None,
        legacy_v1_semantic_fingerprint_sha256=b"o" * 32,
    ).wait()
    changed_classification = await writer.classify_webhook_receipt(
        event_id="event-a",
        semantic_fingerprint_sha256=b"x" * 32,
        legacy_v1_semantic_fingerprint_sha256=b"y" * 32,
    )

    assert classification == "duplicate"
    assert duplicate.receipt == "duplicate"
    assert duplicate.effect == "duplicate"
    assert fresh.receipt == "first"
    assert fresh.effect == "applied"
    assert changed_classification == "conflict"
    with pytest.raises(CommandConflictError, match="webhook_identity_conflict"):
        await writer.submit_webhook(
            receipt={
                "event_id": "event-a",
                "event_type": "call.initiated",
                "call_control_id": "control-a",
                "occurred_at": NOW,
                "received_at": NOW,
                "semantic_fingerprint_sha256": b"x" * 32,
            },
            lease=None,
            operation=None,
            legacy_v1_semantic_fingerprint_sha256=b"y" * 32,
        ).wait()
    await task
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT event_id, semantic_fingerprint_sha256
            FROM webhook_receipts ORDER BY event_id
            """
        ).fetchall()
    assert rows == [("event-a", legacy_fingerprint), ("event-b", b"n" * 32)]


@pytest.mark.asyncio
async def test_postcommit_quick_check_failure_cannot_replace_webhook_result(
    tmp_path: Path,
) -> None:
    clock = 0.0
    checks = iter(("ok", "not ok"))

    def advance_after_mutation(name: str) -> None:
        nonlocal clock
        if name == "after_mutation_before_commit":
            clock = 31.0

    writer = PersistenceWriter(
        tmp_path / "postcommit-health.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        monotonic=lambda: clock,
        quick_check_result=lambda: next(checks),
        failpoint=advance_after_mutation,
    )
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready()

    ticket = writer.submit_webhook(
        receipt=_receipt("event-1"), lease=_lease(), operation=_operation()
    )
    result = await ticket.wait()

    assert (result.receipt, result.effect) == ("first", "applied")
    await owner
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "quick_check_failed"


@pytest.mark.asyncio
async def test_webhook_submit_queue_full_is_synchronous_and_constant_safe(
    tmp_path: Path,
) -> None:
    writer = PersistenceWriter(
        tmp_path / "queue-full.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
    )
    for number in range(256):
        writer.submit_webhook(
            receipt=_receipt(f"event-{number}"),
            lease=None,
            operation=None,
        )

    with pytest.raises(FatalPersistenceError, match="queue_full") as raised:
        writer.submit_webhook(
            receipt=_receipt("event-overflow"),
            lease=None,
            operation=None,
        )

    assert writer.queue_size == 256
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "queue_full"
    assert "event-overflow" not in repr(raised.value)


def test_public_control_timeout_latch_is_constant_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    faults: list[object] = []
    writer = PersistenceWriter(
        tmp_path / "timeout-latch.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        fatal_handler=faults.append,
    )

    first = writer.latch_control_commit_timeout()
    second = writer.latch_control_commit_timeout()

    assert isinstance(first, FatalPersistenceError)
    assert isinstance(second, FatalPersistenceError)
    assert first.args == ("control_commit_timeout",)
    assert second.args == ("control_commit_timeout",)
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code == "control_commit_timeout"
    assert writer.is_degraded is True
    assert len(faults) == 1
