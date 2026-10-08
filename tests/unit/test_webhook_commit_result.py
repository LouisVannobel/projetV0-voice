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


def _candidate_admission(writer: PersistenceWriter, number: int, **extra: object):
    control = f"candidate-{number}"
    return writer.submit_webhook(
        receipt={**_receipt(f"candidate-event-{number}"), "call_control_id": control},
        lease={**_lease(), "call_control_id": control, "call_id": UUID(int=number)},
        operation=None,
        qualification_run_id=RUN_ID,
        qualification_total_calls=3,
        qualification_profile_sha256=b"p" * 32,
        **extra,
    )


@pytest.mark.asyncio
async def test_finite_candidate_counts_commits_durably_and_refuses_fourth_before_any_effect(
    tmp_path,
):
    database = tmp_path / "finite.sqlite"
    writer, owner = await _start_writer(database)
    first_consumed_at = None
    try:
        for number in (1, 2):
            result = await _candidate_admission(writer, number).wait()
            assert result.qualification_exhausted is False
            assert (
                await writer.qualification_run_consumed(
                    RUN_ID, total_calls=3, profile_sha256=b"p" * 32
                )
                is False
            )
            with sqlite3.connect(database) as connection:
                stamp = connection.execute("SELECT consumed_at FROM qualification_runs").fetchone()
                if first_consumed_at is None:
                    first_consumed_at = stamp
                assert stamp == first_consumed_at
    finally:
        await _stop_writer(writer, owner)
    writer, owner = await _start_writer(database)
    try:
        final = await _candidate_admission(writer, 3).wait()
        assert final.qualification_exhausted is True
        rejected = await _candidate_admission(writer, 4).wait()
        assert type(rejected).__name__ == "QualificationRunConsumed"
    finally:
        await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT profile_sha256,total_calls,used_calls FROM qualification_runs"
        ).fetchall() == [(b"p" * 32, 3, 3)]
        assert (
            connection.execute("SELECT consumed_at FROM qualification_runs").fetchone()
            == first_consumed_at
        )
        assert connection.execute("SELECT count(*) FROM call_leases").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "active", "terminal"])
async def test_candidate_new_event_same_admitted_lease_never_uses_another_unit(tmp_path, state):
    database = tmp_path / "replay.sqlite"
    writer, owner = await _start_writer(database)
    try:
        await _candidate_admission(writer, 1).wait()
        await writer.commit_control(
            PersistenceCommand(
                "lease",
                {**_lease(state), "call_control_id": "candidate-1", "call_id": UUID(int=1)},
                None,
            )
        )
        replay = await writer.submit_webhook(
            receipt={**_receipt("new-id"), "call_control_id": "candidate-1"},
            lease={**_lease(), "call_control_id": "candidate-1", "call_id": UUID(int=1)},
            operation=None,
            qualification_run_id=RUN_ID,
            qualification_total_calls=3,
            qualification_profile_sha256=b"p" * 32,
        ).wait()
        assert replay.effect == ("existing_terminal" if state == "terminal" else "duplicate")
    finally:
        await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (1,)
        assert connection.execute("SELECT state FROM call_leases").fetchone() == (state,)


@pytest.mark.asyncio
async def test_candidate_limit_and_profile_are_immutable_after_first_use(tmp_path):
    database = tmp_path / "immutable.sqlite"
    writer, owner = await _start_writer(database)
    await _candidate_admission(writer, 1).wait()
    with pytest.raises(CommandConflictError, match="qualification_run_conflict"):
        await writer.qualification_run_consumed(RUN_ID, total_calls=4, profile_sha256=b"p" * 32)
    await owner
    writer, owner = await _start_writer(database)
    with pytest.raises(CommandConflictError, match="qualification_run_conflict"):
        await writer.qualification_run_consumed(RUN_ID, total_calls=3, profile_sha256=b"q" * 32)
    await owner
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT total_calls,used_calls FROM qualification_runs"
        ).fetchone() == (3, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [2, 3, 4, 5])
async def test_candidate_migration_preserves_consumed_history_without_inventing_fingerprint(
    tmp_path, version
):
    from projetv0_voice.persistence import schema

    database = tmp_path / "legacy-candidate.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(getattr(schema, f"V{version}_SCHEMA_SQL"))
        connection.execute(
            "INSERT INTO qualification_runs VALUES (?,?)", (str(RUN_ID), "old-first-use")
        )
    writer, owner = await _start_writer(database)
    try:
        assert (
            await writer.qualification_run_consumed(RUN_ID, total_calls=3, profile_sha256=b"p" * 32)
            is True
        )
        assert (
            type(await _candidate_admission(writer, 1).wait()).__name__
            == "QualificationRunConsumed"
        )
    finally:
        await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM qualification_runs").fetchall() == [
            (str(RUN_ID), "old-first-use", None, 1, 1)
        ]
        assert connection.execute("SELECT count(*) FROM call_leases").fetchone() == (0,)


@pytest.mark.asyncio
async def test_candidate_rollback_keeps_all_units_and_receipts_absent(tmp_path):
    database = tmp_path / "rollback-allowance.sqlite"

    def failpoint(name):
        if name == "after_mutation_before_commit":
            raise OSError("offline-rollback")

    writer = PersistenceWriter(
        database, CryptoKeyring({1: KEY}, active_version=1), failpoint=failpoint
    )
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    with pytest.raises(FatalPersistenceError):
        await _candidate_admission(writer, 1).wait()
    await owner
    with sqlite3.connect(database) as connection:
        for table in ("qualification_runs", "call_leases", "webhook_receipts", "outbox"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)


@pytest.mark.asyncio
async def test_candidate_concurrent_submissions_admit_exactly_the_remaining_units(tmp_path):
    database = tmp_path / "concurrent-allowance.sqlite"
    writer, owner = await _start_writer(database)
    try:
        results = await asyncio.gather(*[
            _candidate_admission(writer, number).wait() for number in range(1, 5)
        ])
        assert [type(result).__name__ for result in results] == [
            "WebhookCommitResult", "WebhookCommitResult", "WebhookCommitResult",
            "QualificationRunConsumed",
        ]
        assert [result.qualification_exhausted for result in results[:3]] == [False, False, True]
    finally:
        await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (3,)


@pytest.mark.asyncio
async def test_candidate_late_commit_keeps_ownership_and_spends_once_after_waiter_cancel(tmp_path):
    database = tmp_path / "late-allowance.sqlite"
    entered, release = asyncio.Event(), asyncio.Event()

    async def failpoint(name):
        if name == "after_mutation_before_commit":
            entered.set()
            await release.wait()

    writer = PersistenceWriter(
        database, CryptoKeyring({1: KEY}, active_version=1), failpoint=failpoint
    )
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    ticket = _candidate_admission(writer, 1)
    waiter = asyncio.create_task(ticket.wait())
    await entered.wait()
    assert not ticket.done()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM qualification_runs").fetchone() == (0,)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert (await ticket.wait()).qualification_exhausted is False
    await _stop_writer(writer, owner)
    writer, owner = await _start_writer(database)
    try:
        replay = await _candidate_admission(writer, 1).wait()
        assert replay.receipt == "duplicate"
    finally:
        await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (1,)


@pytest.mark.asyncio
async def test_candidate_lost_commit_response_never_refunds_committed_admission(
    tmp_path, monkeypatch
):
    import aiosqlite

    database = tmp_path / "lost-commit.sqlite"
    writer, owner = await _start_writer(database)
    native_commit = aiosqlite.Connection.commit

    async def commit_then_lose_response(connection):
        await native_commit(connection)
        raise OSError("offline-lost-commit-response")

    with monkeypatch.context() as patch:
        patch.setattr(aiosqlite.Connection, "commit", commit_then_lose_response)
        with pytest.raises(FatalPersistenceError):
            await _candidate_admission(writer, 1).wait()
        await owner
    writer, owner = await _start_writer(database)
    try:
        assert await writer.qualification_run_consumed(
            RUN_ID, total_calls=3, profile_sha256=b"p" * 32
        ) is False
        assert (await _candidate_admission(writer, 1).wait()).receipt == "duplicate"
        assert (await _candidate_admission(writer, 2).wait()).qualification_exhausted is False
        assert (await _candidate_admission(writer, 3).wait()).qualification_exhausted is True
    finally:
        await _stop_writer(writer, owner)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (3,)


@pytest.mark.asyncio
async def test_candidate_new_event_replay_rejects_divergent_lease_without_spending(tmp_path):
    database = tmp_path / "divergent.sqlite"
    writer, owner = await _start_writer(database)
    await _candidate_admission(writer, 1).wait()
    with pytest.raises(CommandConflictError, match="lease_identity_conflict"):
        await writer.submit_webhook(
            receipt={**_receipt("divergent"), "call_control_id": "candidate-1"},
            lease={**_lease(), "call_control_id": "candidate-1", "call_id": UUID(int=2)},
            operation=None, qualification_run_id=RUN_ID,
            qualification_total_calls=3, qualification_profile_sha256=b"p" * 32,
        ).wait()
    await owner
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (1,)


@pytest.mark.asyncio
async def test_candidate_new_event_replay_retains_native_admission_fact_identity_checks(tmp_path):
    from dataclasses import replace

    from projetv0_voice.persistence.writer import LocalCallAdmissionFacts

    database = tmp_path / "divergent-facts.sqlite"
    writer, owner = await _start_writer(database)
    operation = _operation().model_copy(update={
        "payload": _operation().payload.model_copy(
            update={"retention_until": NOW + timedelta(days=30)}
        )
    })
    facts = LocalCallAdmissionFacts(operation.call_id, NOW, NOW + timedelta(days=30), None, None)
    await writer.submit_webhook(
        receipt=_receipt("initial"), lease=_lease(), operation=operation,
        admission_facts=facts, qualification_run_id=RUN_ID,
        qualification_total_calls=3, qualification_profile_sha256=b"p" * 32,
    ).wait()
    with pytest.raises(CommandConflictError, match="local_admission_identity_conflict"):
        await writer.submit_webhook(
            receipt=_receipt("different-facts"), lease=_lease(),
            operation=operation.model_copy(update={
                "payload": operation.payload.model_copy(update={"telnyx_call_leg_id": "different"})
            }),
            admission_facts=replace(facts, telnyx_call_leg_id="different"),
            qualification_run_id=RUN_ID,
            qualification_total_calls=3, qualification_profile_sha256=b"p" * 32,
        ).wait()
    await owner
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (1,)


@pytest.mark.asyncio
async def test_candidate_pending_replay_preserves_exact_lease_deadline_guard(tmp_path):
    database = tmp_path / "changed-deadline.sqlite"
    writer, owner = await _start_writer(database)
    await _candidate_admission(writer, 1).wait()
    with pytest.raises(CommandConflictError, match="lease_transition_conflict"):
        await writer.submit_webhook(
            receipt={**_receipt("changed-deadline"), "call_control_id": "candidate-1"},
            lease={**_lease(), "call_control_id": "candidate-1", "call_id": UUID(int=1),
                   "expires_at": NOW + timedelta(seconds=40)},
            operation=None, qualification_run_id=RUN_ID,
            qualification_total_calls=3, qualification_profile_sha256=b"p" * 32,
        ).wait()
    await owner
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT used_calls FROM qualification_runs").fetchone() == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize("migration_fails", [False, True])
async def test_v5_six_consumed_runs_migrate_or_roll_back_as_one_authority(
    tmp_path, migration_fails
):
    from projetv0_voice.persistence.schema import V5_SCHEMA_SQL

    database = tmp_path / "six-historical-runs.sqlite"
    history = [(str(UUID(int=number)), f"historical-first-use-{number}") for number in range(1, 7)]
    with sqlite3.connect(database) as connection:
        connection.executescript(V5_SCHEMA_SQL)
        connection.executemany("INSERT INTO qualification_runs VALUES (?,?)", history)

    def failpoint(name):
        if migration_fails and name == "after_v1_migration_before_commit":
            raise OSError("offline-migration-rollback")

    writer = PersistenceWriter(
        database, CryptoKeyring({1: KEY}, active_version=1), failpoint=failpoint
    )
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready() is not migration_fails
    if not migration_fails:
        await _stop_writer(writer, owner)
    else:
        await owner
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT run_id,consumed_at FROM qualification_runs"
        ).fetchall() == history
        assert connection.execute("PRAGMA user_version").fetchone() == (
            (5,) if migration_fails else (6,)
        )
        if not migration_fails:
            assert connection.execute(
                "SELECT profile_sha256,total_calls,used_calls FROM qualification_runs"
            ).fetchall() == [(None, 1, 1)] * 6


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
async def test_writer_establishes_exact_sqlite_v6_before_readiness(
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

    assert version == (6,)
    assert objects == {
        ("index", "outbox_due_fifo_idx"),
        ("table", "call_leases"),
        ("table", "outbox"),
        ("table", "qualification_runs"),
        ("table", "webhook_receipts"),
        ("table", "sparra_turn_decisions"),
        ("table", "sparra_content_fences"),
        ("table", "sparra_publications"),
        ("table", "recording_archives"),
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
