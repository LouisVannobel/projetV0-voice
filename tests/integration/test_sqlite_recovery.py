from __future__ import annotations

import base64
import os
import sqlite3
import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.models import TurnUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import decode_operation, operation_aad_from_metadata
from projetv0_voice.persistence.writer import PersistenceWriter

KEY_V1 = bytes(range(32))
KEY_V2 = bytes(range(32, 64))
NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


CHILD = textwrap.dedent(
    """
    import asyncio
    import base64
    import os
    import sys
    from pathlib import Path

    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.models import VoiceOperationV1
    from projetv0_voice.persistence.writer import PersistenceWriter

    async def main():
        database = Path(sys.argv[1])
        mode = sys.argv[2]
        operation = VoiceOperationV1.model_validate_json(base64.b64decode(sys.argv[3]))
        key = base64.b64decode(sys.argv[4])

        def failpoint(name):
            if mode == "before" and name == "after_mutation_before_commit":
                os._exit(91)

        writer = PersistenceWriter(
            database,
            CryptoKeyring({1: key}, active_version=1),
            failpoint=failpoint,
        )
        run_task = asyncio.create_task(writer.run())
        if not await writer.wait_ready():
            raise RuntimeError("writer did not become ready")
        if not writer.try_enqueue_turn(operation):
            raise RuntimeError("enqueue failed")
        await writer.wait_until_idle()
        if mode == "after":
            os._exit(92)
        await writer.drain(2)
        await run_task

    asyncio.run(main())
    """
)


def synthetic_turn(
    keyring: CryptoKeyring, *, transcript: bytes = b"restored synthetic turn"
) -> VoiceOperationV1:
    inner = keyring.encrypt(transcript, aad=b"turn:00000000-0000-0000-0000-000000000123")
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(int=501),
        deployment_id="agent-a",
        call_id=UUID(int=502),
        occurred_at=NOW + timedelta(seconds=2),
        kind="turn.upsert",
        payload=TurnUpsertPayloadV1(
            turn_id=UUID(int=123),
            turn_no=1,
            role="user",
            source="stt_final",
            crypto_version=1,
            key_version=inner.key_version,
            nonce_b64=base64.b64encode(inner.nonce).decode("ascii"),
            ciphertext_b64=base64.b64encode(inner.ciphertext).decode("ascii"),
            started_at=NOW,
            ended_at=NOW + timedelta(seconds=1),
            interrupted=False,
        ),
    )


def run_child(
    database: Path, mode: str, candidate: VoiceOperationV1
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(database),
            mode,
            base64.b64encode(candidate.model_dump_json().encode()).decode(),
            base64.b64encode(KEY_V1).decode(),
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_hard_kill_before_commit_loses_row_but_after_commit_survives(tmp_path: Path) -> None:
    candidate = synthetic_turn(CryptoKeyring({1: KEY_V1}, active_version=1))

    before_database = tmp_path / "before.sqlite"
    before = run_child(before_database, "before", candidate)
    assert before.returncode == 91, before.stderr
    with sqlite3.connect(before_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (0,)
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)

    after_database = tmp_path / "after.sqlite"
    after = run_child(after_database, "after", candidate)
    assert after.returncode == 92, after.stderr
    with sqlite3.connect(after_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)


@pytest.mark.asyncio
async def test_restart_reemits_expired_leases_until_terminal_ack_and_replays_fifo_outbox(
    tmp_path: Path,
) -> None:
    database = tmp_path / "restart.sqlite"
    keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    writer = PersistenceWriter(database, keyring, utcnow=lambda: NOW)
    first_task = __import__("asyncio").create_task(writer.run())
    assert await writer.wait_ready()
    await writer.commit_lease(
        call_control_id="expired-control",
        call_id=UUID(int=800),
        tenant_id="tenant-a",
        agent_id="agent-a",
        state="pending",
        token_hash=bytes(range(32)),
        created_at=NOW - timedelta(minutes=2),
        expires_at=NOW - timedelta(minutes=1),
        closed_at=None,
    )
    await writer.commit_lease(
        call_control_id="expired-active",
        call_id=UUID(int=801),
        tenant_id="tenant-a",
        agent_id="agent-a",
        state="pending",
        token_hash=bytes(range(32, 64)),
        created_at=NOW - timedelta(minutes=3),
        expires_at=NOW - timedelta(minutes=1),
        closed_at=None,
    )
    await writer.commit_lease(
        call_control_id="future-pending",
        call_id=UUID(int=802),
        tenant_id="tenant-a",
        agent_id="agent-a",
        state="pending",
        token_hash=bytes(range(64, 96)),
        created_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=1),
        closed_at=None,
    )
    await writer.commit_lease(
        call_control_id="expired-active",
        call_id=UUID(int=801),
        tenant_id="tenant-a",
        agent_id="agent-a",
        state="active",
        token_hash=bytes(range(32, 64)),
        created_at=NOW - timedelta(minutes=3),
        expires_at=NOW - timedelta(minutes=1),
        closed_at=None,
    )
    assert writer.try_enqueue_turn(synthetic_turn(keyring))
    second_turn = synthetic_turn(keyring).model_copy(
        update={"operation_id": UUID(int=503), "call_id": UUID(int=504)}
    )
    assert writer.try_enqueue_turn(second_turn)
    await writer.wait_until_idle()
    await writer.drain(2)
    await first_task

    restarted = PersistenceWriter(database, keyring, utcnow=lambda: NOW)
    second_task = __import__("asyncio").create_task(restarted.run())
    assert await restarted.wait_ready()
    stale = restarted.take_stale_leases()
    batch = await restarted.read_relay_batch(batch_size=100, now=NOW, lease_seconds=30)

    assert [item.call_control_id for item in stale] == [
        "expired-active",
        "expired-control",
        "future-pending",
    ]
    assert [item.previous_state for item in stale] == ["active", "pending", "pending"]
    assert [item.queue_id for item in batch] == [1, 2]
    assert [item.operation.operation_id for item in batch] == [UUID(int=501), UUID(int=503)]
    assert all(item.created_at == NOW for item in batch)
    assert all(item.claim_expires_at == NOW + timedelta(seconds=30) for item in batch)
    await restarted.drain(2)
    await second_task

    again = PersistenceWriter(database, keyring, utcnow=lambda: NOW)
    third_task = __import__("asyncio").create_task(again.run())
    assert await again.wait_ready()
    stale_again = again.take_stale_leases()
    assert [item.call_control_id for item in stale_again] == [
        "expired-active",
        "expired-control",
        "future-pending",
    ]
    for item in stale_again:
        await again.commit_lease(
            call_control_id=item.call_control_id,
            call_id=item.call_id,
            tenant_id=item.tenant_id,
            agent_id=item.agent_id,
            state="terminal",
            token_hash=item.token_hash,
            created_at=item.created_at,
            expires_at=item.expires_at,
            closed_at=NOW,
        )
    await again.drain(2)
    await third_task

    acknowledged = PersistenceWriter(database, keyring, utcnow=lambda: NOW)
    fourth_task = __import__("asyncio").create_task(acknowledged.run())
    assert await acknowledged.wait_ready()
    assert acknowledged.take_stale_leases() == ()
    await acknowledged.drain(2)
    await fourth_task


@pytest.mark.asyncio
async def test_relay_claim_survives_restart_and_reappears_only_after_lease_expiry(
    tmp_path: Path,
) -> None:
    database = tmp_path / "claim-restart.sqlite"
    keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    writer = PersistenceWriter(database, keyring, utcnow=lambda: NOW)
    first_task = __import__("asyncio").create_task(writer.run())
    assert await writer.wait_ready()
    assert writer.try_enqueue_turn(synthetic_turn(keyring))
    await writer.wait_until_idle()
    claimed = await writer.read_relay_batch(batch_size=10, now=NOW, lease_seconds=30)
    assert [item.queue_id for item in claimed] == [1]
    assert claimed[0].created_at == NOW
    assert claimed[0].claim_expires_at == NOW + timedelta(seconds=30)
    await writer.drain(2)
    await first_task

    restarted = PersistenceWriter(database, keyring, utcnow=lambda: NOW)
    second_task = __import__("asyncio").create_task(restarted.run())
    assert await restarted.wait_ready()
    assert await restarted.read_relay_batch(
        batch_size=10,
        now=NOW + timedelta(seconds=29),
        lease_seconds=30,
    ) == ()
    available = await restarted.read_relay_batch(
        batch_size=10,
        now=NOW + timedelta(seconds=30),
        lease_seconds=30,
    )
    assert [item.queue_id for item in available] == [1]
    assert available[0].created_at == NOW
    assert available[0].claim_expires_at == NOW + timedelta(seconds=60)
    await restarted.drain(2)
    await second_task


@pytest.mark.asyncio
async def test_restored_turn_decrypts_with_retained_old_key(tmp_path: Path) -> None:
    database = tmp_path / "restore.sqlite"
    old_keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    candidate = synthetic_turn(old_keyring)
    writer = PersistenceWriter(database, old_keyring, utcnow=lambda: NOW)
    task = __import__("asyncio").create_task(writer.run())
    assert await writer.wait_ready()
    assert writer.try_enqueue_turn(candidate)
    await writer.wait_until_idle()
    await writer.drain(2)
    await task

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT schema_version, op_id, deployment_id, call_id, kind,
                   key_version, nonce, ciphertext
            FROM outbox WHERE queue_id = 1
            """
        ).fetchone()
    assert row is not None
    metadata = {
        "schema_version": row[0],
        "operation_id": row[1],
        "deployment_id": row[2],
        "call_id": row[3],
        "kind": row[4],
    }
    outer = EncryptedValue(key_version=row[5], nonce=row[6], ciphertext=row[7])
    retained = CryptoKeyring({1: KEY_V1, 2: KEY_V2}, active_version=2)
    restored = decode_operation(
        retained.decrypt(outer, aad=operation_aad_from_metadata(metadata))
    )
    inner = EncryptedValue(
        key_version=restored.payload.key_version,
        nonce=base64.b64decode(restored.payload.nonce_b64),
        ciphertext=base64.b64decode(restored.payload.ciphertext_b64),
    )
    assert retained.decrypt(
        inner, aad=b"turn:00000000-0000-0000-0000-000000000123"
    ) == b"restored synthetic turn"


@pytest.mark.asyncio
async def test_corrupt_database_fails_startup_quick_check_closed(tmp_path: Path) -> None:
    database = tmp_path / "corrupt.sqlite"
    database.write_bytes(b"not-a-sqlite-database")
    writer = PersistenceWriter(database, CryptoKeyring({1: KEY_V1}, active_version=1))
    task = __import__("asyncio").create_task(writer.run())

    assert await writer.wait_ready() is False
    await task
    assert writer.is_degraded
    assert writer.fatal_fault is not None
    assert writer.fatal_fault.code in {"sqlite_corrupt", "quick_check_failed"}


@pytest.mark.asyncio
async def test_missing_historical_outer_key_fails_relay_read_closed(tmp_path: Path) -> None:
    database = tmp_path / "missing-key.sqlite"
    old_keyring = CryptoKeyring({1: KEY_V1}, active_version=1)
    writer = PersistenceWriter(database, old_keyring, utcnow=lambda: NOW)
    first_task = __import__("asyncio").create_task(writer.run())
    assert await writer.wait_ready()
    assert writer.try_enqueue_turn(synthetic_turn(old_keyring))
    await writer.wait_until_idle()
    await writer.drain(2)
    await first_task

    missing_old_key = CryptoKeyring({2: KEY_V2}, active_version=2)
    restarted = PersistenceWriter(database, missing_old_key, utcnow=lambda: NOW)
    second_task = __import__("asyncio").create_task(restarted.run())
    assert await restarted.wait_ready()
    with pytest.raises(Exception, match="unknown_key_version"):
        await restarted.read_relay_batch(batch_size=10, now=NOW, lease_seconds=30)
    await second_task
    assert restarted.fatal_fault is not None
    assert restarted.fatal_fault.code == "unknown_key_version"
