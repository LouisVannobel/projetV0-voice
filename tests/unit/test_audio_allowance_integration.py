"""Audio schema generations retain the established finite admission authority."""

from __future__ import annotations

import asyncio
import sqlite3
from uuid import UUID

import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence import schema
from projetv0_voice.persistence.writer import PersistenceWriter
from tests.unit.test_webhook_commit_result import RUN_ID, _candidate_admission


def test_audio_versions_follow_v6_allocation_without_reusing_its_number():
    assert schema.SCHEMA_VERSION == 6
    assert getattr(schema, "LOCAL_AUDIO_SCHEMA_VERSION", None) == 7
    assert schema.LOCAL_AUDIO_CHOICE_SCHEMA_VERSION == 8
    assert schema.LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [7, 8, 9])
async def test_audio_upgrade_preserves_finite_and_legacy_v6_authority(tmp_path, version):
    assert getattr(schema, "LOCAL_AUDIO_SCHEMA_VERSION", None) == 7
    ddl = {
        7: schema.LOCAL_AUDIO_SCHEMA_SQL,
        8: schema.LOCAL_AUDIO_CHOICE_SCHEMA_SQL,
        9: schema.LOCAL_AUDIO_TERMINAL_SCHEMA_SQL,
    }[version]
    path = tmp_path / "authority.sqlite"
    rows = [(str(RUN_ID), "original-first-use", b"p" * 32, 3, 2),
            (str(UUID(int=99)), "spent-legacy", None, 1, 1)]
    with sqlite3.connect(path) as db:
        db.executescript(ddl)
        db.executemany("INSERT INTO qualification_runs VALUES(?,?,?,?,?)", rows)
    writer = PersistenceWriter(path, CryptoKeyring({1: bytes(range(32))}, active_version=1),
                               contract_version=2, process_agent_id="agent-a",
                               process_deployment_id="agent-a")
    task = asyncio.create_task(writer.run())
    try:
        assert await writer.wait_ready()
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (9,)
            assert db.execute("SELECT * FROM qualification_runs").fetchall() == rows
        assert not await writer.qualification_run_consumed(
            RUN_ID, total_calls=3, profile_sha256=b"p" * 32)
        final = await _candidate_admission(writer, 3).wait()
        assert final.qualification_exhausted
        rejected = await _candidate_admission(writer, 4).wait()
        assert type(rejected).__name__ == "QualificationRunConsumed"
        assert await writer.qualification_run_consumed(
            UUID(int=99), total_calls=3, profile_sha256=b"p" * 32)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT * FROM qualification_runs").fetchall() == [
                (*rows[0][:-1], 3), rows[1]]
            assert db.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (1,)
            assert db.execute("SELECT count(*) FROM call_leases").fetchone() == (1,)
    finally:
        if writer.is_degraded:
            await task
        else:
            await writer.drain(2)
            await task


@pytest.mark.asyncio
async def test_v6_finite_spool_cannot_be_converted_to_audio_or_refund_authority(tmp_path):
    assert getattr(schema, "LOCAL_AUDIO_SCHEMA_VERSION", None) == 7
    path = tmp_path / "v6.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(schema.SCHEMA_SQL)
        db.execute("INSERT INTO qualification_runs VALUES(?,?,?,?,?)",
                   (str(RUN_ID), "original-first-use", b"p" * 32, 3, 2))
    before = path.read_bytes()
    writer = PersistenceWriter(path, CryptoKeyring({1: bytes(range(32))}, active_version=1),
                               contract_version=2, process_agent_id="agent-a",
                               process_deployment_id="agent-a")
    task = asyncio.create_task(writer.run())
    assert not await writer.wait_ready()
    await task
    assert path.read_bytes() == before
    assert writer.fatal_fault.code == "writer_contract_mismatch"
