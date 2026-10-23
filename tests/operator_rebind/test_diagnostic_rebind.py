"""Synthetic SQLite behavior tests; never accepts a database path from the caller."""

from __future__ import annotations

import asyncio
import ctypes
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

import diagnostic_rebind as component
from projetv0_voice.budget_observation import BudgetObservation, BudgetObservationIdentity
from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPermit, OwnedPilotBudgetPolicy
from projetv0_voice.persistence import schema


NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
TARGET_IMAGE = "synthetic.local/voice@sha256:" + "a" * 64
TARGET_PROFILE = "b" * 64
SQL = {
    10: schema.OWNED_BUDGET_SCHEMA_SQL,
    11: schema.OWNED_AUDIO_BUDGET_SCHEMA_SQL,
    12: schema.SIXTH_BUDGET_SCHEMA_SQL,
    13: schema.SIXTH_AUDIO_BUDGET_SCHEMA_SQL,
}
MATRIX = [(10, 5, 2), (10, 5, 5), (11, 5, 2), (11, 5, 5),
          (12, 6, 5), (12, 6, 6), (13, 6, 5), (13, 6, 6)]


def fresh_memory_guard():
    if sys.platform == "linux":
        # Independent physical and kernel commit headroom; no infinity exemption.
        import re
        from pathlib import Path

        with Path("/proc/meminfo").open(encoding="ascii") as handle:
            raw = handle.read(65_537)
        assert len(raw) <= 65_536, "fresh2GiB_metrics_oversize"
        required = {"MemAvailable", "CommitLimit", "Committed_AS"}
        memory = {}
        for line in raw.splitlines():
            key, separator, value = line.partition(":")
            if key in required:
                assert separator and key not in memory, "fresh2GiB_metrics_ambiguous"
                matched = re.fullmatch(r"\s*([0-9]+)\s+kB\s*", value)
                assert matched is not None, "fresh2GiB_metrics_invalid"
                memory[key] = int(matched[1]) * 1024
        assert set(memory) == required, "fresh2GiB_metrics_missing"
        available_bytes = (memory["MemAvailable"], memory["CommitLimit"] - memory["Committed_AS"])
        available = tuple(round(value / 1024**3, 3) for value in available_bytes)
        assert min(available_bytes) >= 2 * 1024**3, (
            "fresh2GiB_refused physical/commit GiB", available)
        return available
    assert sys.platform == "win32", "fresh2GiB_platform_unsupported"
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual",
             "avail_virtual", "extended")
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)), "memory_guard_unavailable"
    available = tuple(round(value / 1024**3, 3) for value in
                      (status.avail_phys, status.avail_page, status.avail_virtual))
    assert min(status.avail_phys, status.avail_virtual, status.avail_page) >= 2 * 1024**3, (
        "fresh2GiB_refused physical/page-available/virtual GiB", available)
    return available


@pytest.fixture(autouse=True)
def before_every_test():
    fresh_memory_guard()


def policy(total=5, **changes):
    values = dict(
        policy_id=UUID(int=1),
        identity=BudgetObservationIdentity(
            "1" * 64, "2" * 64, "3" * 64, "4" * 64, "+33102030405",
            "synthetic-connection", NOW - timedelta(days=1), NOW + timedelta(days=1),
        ),
        total_calls=total,
        openrouter_anchor_usd=Decimal("0.30"), telnyx_anchor_usd=Decimal("0.10"),
        paid_until=NOW + timedelta(days=1), qualified_profile_sha256="5" * 64,
        runtime_contract_sha256="6" * 64, agent_bundle_sha256="7" * 64,
        inference_profile_sha256="8" * 64, image_digest="fixture@sha256:" + "9" * 64,
        deployment_id="fixture",
    )
    return OwnedPilotBudgetPolicy(**(values | changes))


def permit(p, ordinal=5):
    return OwnedPilotBudgetPermit(
        policy_sha256=p.sha256, ordinal=ordinal, call_id=UUID(int=2),
        call_control_id="historical", call_leg_id="leg", call_session_id="session",
        generation=UUID(int=3), paid_until=p.paid_until,
    )


def seed(db, version=13, used=5):
    db.executescript(SQL[version])
    p = policy(5 if version in (10, 11) else 6)
    db.execute("INSERT INTO owned_pilot_budget VALUES(?,?,?,?,?,?,?,?)", (
        "sparra-original-pilot", p.model_dump_json(), p.sha256, used, "1.10", "0.20", 0, None,
    ))
    db.execute("INSERT INTO call_leases VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
        "historical", str(UUID(int=2)), "tenant", "fixture", "terminal", bytes(32),
        NOW.isoformat(), (NOW + timedelta(minutes=1)).isoformat(), NOW.isoformat(),
        None, permit(p, used).model_dump_json(),
    ))
    db.execute("INSERT INTO webhook_receipts VALUES(?,?,?,?,?,?,?)", (
        "event-history", "call.hangup", "historical", NOW.isoformat(), NOW.isoformat(), bytes(32), None,
    ))
    db.execute("INSERT INTO qualification_runs(run_id,consumed_at) VALUES(?,?)", (
        str(UUID(int=99)), NOW.isoformat(),
    ))
    db.execute("INSERT INTO qualification_runs VALUES(?,?,?,?,?)", (
        str(UUID(int=100)), NOW.isoformat(), bytes.fromhex("5" * 64), 2, 2,
    ))
    db.execute("INSERT INTO sparra_turn_decisions VALUES(?,?,?,?,?,?,?,?,?)", (
        str(UUID(int=2)), str(UUID(int=4)), str(UUID(int=5)), "fixture", bytes(32), None, None, None, 1,
    ))
    db.execute("INSERT INTO sparra_publications VALUES(?,?,?,?,?,?)", (
        str(UUID(int=2)), str(UUID(int=6)), "fixture", 1, bytes(12), bytes(16),
    ))
    db.execute("INSERT INTO sparra_content_fences VALUES(?,?,?,?,?,?)", (
        str(UUID(int=7)), NOW.isoformat(), str(UUID(int=8)), NOW.isoformat(), 1, 1,
    ))
    db.execute("INSERT INTO recording_archives VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        "recording-history", str(UUID(int=2)), "fixture", bytes(32), "acknowledged",
        NOW.isoformat(), (NOW + timedelta(days=30)).isoformat(), 1, bytes(12), bytes(16),
        "{}", bytes(12), str(UUID(int=6)),
    ))
    if version in (11, 13):
        db.execute("INSERT INTO local_audio_terminal VALUES(?,?,?,?,?,?,?,?,?)", (
            str(UUID(int=2)), "audio.finish", str(UUID(int=9)), "fixture", bytes(32), 1, bytes(12), bytes(16), 1,
        ))
    # Exercise nonempty sqlite_sequence even though outbox is quiescent.
    add_outbox(db)
    db.execute("DELETE FROM outbox")
    db.commit()
    return p


def add_outbox(db):
    version = db.execute("PRAGMA user_version").fetchone()[0]
    db.execute("INSERT INTO outbox VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        None, str(UUID(int=10)), "fixture", "call.upsert", 2 if version in (11, 13) else 1,
        str(UUID(int=2)), None, None, 1, 1, bytes(12), bytes(16), NOW.isoformat(), 0,
        NOW.isoformat(), None,
    ))


def full_state(db):
    # Independent pre/post comparison, including hidden rowids, sequence and schema roots.
    objects = db.execute("SELECT * FROM sqlite_master ORDER BY type,name").fetchall()
    tables = {
        row[1]: db.execute('SELECT rowid,* FROM "' + row[1].replace('"', '""') + '" ORDER BY rowid').fetchall()
        for row in objects if row[0] == "table"
    }
    return (db.execute("PRAGMA user_version").fetchone(),
            db.execute("PRAGMA schema_version").fetchone(), objects, tables)


def apply(db, preimage, **changes):
    return component.rebind(db, preimage, **(
        dict(image_digest=TARGET_IMAGE, qualified_profile_sha256=TARGET_PROFILE) | changes
    ))


@pytest.mark.parametrize("version,total,used", MATRIX)
def test_exact_conservation_changes_only_two_fields(tmp_path, version, total, used):
    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        old = seed(db, version, used)
        before = full_state(db)
        preimage = component.capture_preimage(db)
        trace = []
        db.set_trace_callback(trace.append)
        result = apply(db, preimage)
        db.set_trace_callback(None)
        assert result.status == "committed"
        after = full_state(db)
        row = after[3]["owned_pilot_budget"][0]
        new = OwnedPilotBudgetPolicy.model_validate_json(row[2])
        assert new.total_calls == total
        assert new.sha256 == row[3] and new.sha256 != old.sha256
        assert new.image_digest == TARGET_IMAGE and new.qualified_profile_sha256 == TARGET_PROFILE
        assert new.model_dump(exclude={"image_digest", "qualified_profile_sha256"}) == old.model_dump(
            exclude={"image_digest", "qualified_profile_sha256"})
        expected_row = list(before[3]["owned_pilot_budget"][0])
        expected_row[2:4] = [new.model_dump_json(), new.sha256]
        before[3]["owned_pilot_budget"] = [tuple(expected_row)]
        assert after == before
        assert sum(sql.lstrip().upper().startswith("UPDATE ") for sql in trace) == 1
        assert not db.in_transaction
        assert component.classify(db, result) == "exact_new"
        assert apply(db, preimage).status == "refused"
        assert apply(db, component.capture_preimage(db)).status == "refused"


@pytest.mark.parametrize("kind", [
    "suspended", "paid_intent", "active", "pending", "outbox", "fence_ack", "fence_settled",
    "audio_ack", "archive_pending", "archive_archived", "zero_budget", "two_budget",
    "unknown_version", "n6_on_n5", "n5_on_n6", "wrong_policy_hash", "noncanonical_policy",
    "extra_table", "trigger", "wrong_slot", "bad_floor", "marker", "temp_table",
])
def test_refuses_incompatible_authority_schema_or_quiescence(tmp_path, kind):
    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        p = seed(db)
        if kind == "suspended":
            db.execute("UPDATE owned_pilot_budget SET suspended=1")
        elif kind == "paid_intent":
            db.execute("UPDATE owned_pilot_budget SET paid_call_control_id='unresolved'")
        elif kind in ("active", "pending"):
            db.execute("UPDATE call_leases SET state=?,closed_at=NULL", (kind,))
        elif kind == "outbox":
            add_outbox(db)
        elif kind.startswith("fence_"):
            db.execute("UPDATE sparra_content_fences SET " +
                       ("lease_acked=0" if kind == "fence_ack" else "lease_settled=0"))
        elif kind == "audio_ack":
            db.execute("UPDATE local_audio_terminal SET acked=0")
        elif kind.startswith("archive_"):
            db.execute("UPDATE recording_archives SET state=?", (kind.removeprefix("archive_"),))
        elif kind == "zero_budget":
            db.execute("DELETE FROM owned_pilot_budget")
        elif kind in ("two_budget", "wrong_slot"):
            db.execute("PRAGMA ignore_check_constraints=ON")
            if kind == "two_budget":
                db.execute("INSERT INTO owned_pilot_budget SELECT 'foreign',policy_json,policy_sha256,"
                           "used_calls,openrouter_floor,telnyx_floor,suspended,paid_call_control_id FROM owned_pilot_budget")
            else:
                db.execute("UPDATE owned_pilot_budget SET slot='foreign'")
            db.execute("PRAGMA ignore_check_constraints=OFF")
        elif kind == "unknown_version":
            db.execute("PRAGMA user_version=99")
        elif kind in ("n6_on_n5", "n5_on_n6"):
            if kind == "n6_on_n5":
                db.execute("PRAGMA user_version=11")
            else:
                changed = policy(5)
                db.execute("UPDATE owned_pilot_budget SET policy_json=?,policy_sha256=?", (
                    changed.model_dump_json(), changed.sha256))
        elif kind == "wrong_policy_hash":
            db.execute("UPDATE owned_pilot_budget SET policy_sha256=?", ("f" * 64,))
        elif kind == "noncanonical_policy":
            db.execute("UPDATE owned_pilot_budget SET policy_json=?", (p.model_dump_json() + " ",))
        elif kind == "extra_table":
            db.execute("CREATE TABLE unexpected(value TEXT)")
        elif kind == "trigger":
            db.execute("CREATE TRIGGER unexpected AFTER UPDATE ON owned_pilot_budget "
                       "BEGIN UPDATE qualification_runs SET used_calls=used_calls; END")
        elif kind == "bad_floor":
            db.execute("UPDATE owned_pilot_budget SET openrouter_floor='NaN'")
        elif kind == "marker":
            db.execute("DELETE FROM local_audio_contract")
        elif kind == "temp_table":
            db.execute("CREATE TEMP TABLE unexpected(value TEXT)")
        db.commit()
        before = full_state(db)
        result = apply(db, component.capture_preimage(db))
        assert result.status == "refused"
        assert full_state(db) == before and not db.in_transaction


@pytest.mark.parametrize("field", list(policy().model_dump(exclude={"image_digest", "qualified_profile_sha256"})))
def test_stale_preimage_rejects_each_altered_policy_authority(tmp_path, field):
    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        p = seed(db)
        preimage = component.capture_preimage(db)
        values = p.model_dump(mode="json")
        replacements = dict(
            policy_id=str(UUID(int=42)), identity=values["identity"] | {"telnyx_account_sha256": "c" * 64},
            cost_scope="unrecognized", total_calls=5, openrouter_anchor_usd="0.31", telnyx_anchor_usd="0.11",
            paid_until=(NOW + timedelta(hours=23)).isoformat(), runtime_contract_sha256="c" * 64,
            agent_bundle_sha256="c" * 64, inference_profile_sha256="c" * 64, deployment_id="foreign",
        )
        values[field] = replacements[field]
        import json
        raw = json.dumps(values)
        db.execute("UPDATE owned_pilot_budget SET policy_json=?", (raw,))
        db.commit()
        before = full_state(db)
        assert apply(db, preimage).status == "refused"
        assert full_state(db) == before


@pytest.mark.parametrize("mutation", ["counter", "floor", "history", "schema"])
def test_exact_preimage_rejects_any_database_drift(tmp_path, mutation):
    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        seed(db)
        preimage = component.capture_preimage(db)
        db.execute({
            "counter": "UPDATE owned_pilot_budget SET used_calls=6",
            "floor": "UPDATE owned_pilot_budget SET telnyx_floor='0.21'",
            "history": "UPDATE qualification_runs SET consumed_at='changed'",
            "schema": "CREATE INDEX unexpected ON qualification_runs(consumed_at)",
        }[mutation])
        db.commit()
        before = full_state(db)
        assert apply(db, preimage).status == "refused"
        assert full_state(db) == before


@pytest.mark.parametrize("target", [
    {"image_digest": "mutable:latest"}, {"image_digest": "voice@sha256:" + "A" * 64},
    {"image_digest": "https://host/voice@sha256:" + "a" * 64},
    {"qualified_profile_sha256": "b" * 63}, {"qualified_profile_sha256": "B" * 64},
    {"image_digest": None}, {"qualified_profile_sha256": None},
])
def test_invalid_target_cannot_write(tmp_path, target):
    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        seed(db)
        before = full_state(db)
        assert apply(db, component.capture_preimage(db), **target).status == "refused"
        assert full_state(db) == before


class FaultConnection(sqlite3.Connection):
    fault = None
    commits_attempted = 0
    updates_attempted = 0

    def execute(self, sql, parameters=()):
        if sql.startswith("UPDATE owned_pilot_budget SET policy_json="):
            self.updates_attempted += 1
            if self.fault == "sql":
                super().execute(sql, parameters)
                raise sqlite3.OperationalError("synthetic confidential SQL detail")
            if self.fault == "trigger":
                super().execute("CREATE TRIGGER injected AFTER UPDATE ON owned_pilot_budget "
                                "BEGIN UPDATE qualification_runs SET consumed_at='mutated'; END")
        if sql == "COMMIT":
            self.commits_attempted += 1
            if self.fault == "before_commit":
                raise sqlite3.OperationalError("synthetic confidential detail")
            cursor = super().execute(sql, parameters)
            if self.fault == "lost_ack":
                raise sqlite3.OperationalError("synthetic confidential acknowledgement")
            return cursor
        return super().execute(sql, parameters)


@pytest.mark.parametrize("fault", ["sql", "trigger"])
def test_sql_failure_or_unexpected_trigger_mutation_rolls_back(tmp_path, fault):
    with sqlite3.connect(tmp_path / "synthetic.sqlite", factory=FaultConnection) as db:
        seed(db)
        before = full_state(db)
        preimage = component.capture_preimage(db)
        db.fault = fault
        result = apply(db, preimage)
        assert result.status == "rolled_back"
        assert db.updates_attempted == 1 and db.commits_attempted == 0
        assert full_state(db) == before and not db.in_transaction
        assert "confidential" not in repr(result)


@pytest.mark.parametrize("fault,want", [("lost_ack", "exact_new"), ("before_commit", "exact_old")])
def test_commit_uncertainty_never_retries_and_observation_only_reads(tmp_path, fault, want):
    path = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(path, factory=FaultConnection) as db:
        seed(db)
        preimage = component.capture_preimage(db)
        db.fault = fault
        result = apply(db, preimage)
        assert result.status == "commit_uncertain"
        assert db.commits_attempted == 1 and db.updates_attempted == 1
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as observation:
        trace = []
        observation.set_trace_callback(trace.append)
        before = full_state(observation)
        assert component.classify(observation, result) == want
        assert full_state(observation) == before
        assert not any(sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE", "CREATE")) for sql in trace)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE qualification_runs SET consumed_at='later'")
        db.commit()
        assert component.classify(db, result) == "other"


def test_existing_transaction_is_not_committed_or_rolled_back(tmp_path):
    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        seed(db)
        preimage = component.capture_preimage(db)
        db.execute("BEGIN")
        db.execute("UPDATE qualification_runs SET consumed_at='caller-transaction'")
        before = full_state(db)
        assert apply(db, preimage).status == "refused"
        assert db.in_transaction and full_state(db) == before
        db.rollback()


def test_invalid_preimage_never_echoes_unvalidated_metadata(tmp_path):
    from dataclasses import replace

    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        seed(db)
        before = full_state(db)
        invalid = replace(component.capture_preimage(db), state_sha256="synthetic-private-value")
        result = apply(db, invalid)
        assert result.status == "refused"
        assert "synthetic-private-value" not in repr(result)
        assert full_state(db) == before


@pytest.mark.parametrize("version,total,used", MATRIX)
def test_native_writer_gate_reopened_twice_preserves_authority_and_rejects_old_permit(tmp_path, version, total, used):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate
    from projetv0_voice.persistence.commands import CommandConflictError
    from projetv0_voice.persistence.writer import PersistenceWriter

    path = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(path) as db:
        old = seed(db, version, used)
        old_permit = permit(old, used)
        assert apply(db, component.capture_preimage(db)).status == "committed"
        successor = OwnedPilotBudgetPolicy.model_validate_json(
            db.execute("SELECT policy_json FROM owned_pilot_budget").fetchone()[0])
        after_cas = full_state(db)

    async def reopen_twice():
        for restart in range(2):
            writer = PersistenceWriter(
                path, CryptoKeyring({1: bytes(range(32))}, active_version=1),
                contract_version=1 if version in (10, 12) else 2,
                utcnow=lambda: NOW, process_agent_id="fixture", process_deployment_id="fixture",
            )

            async def collect():
                return BudgetObservation(
                    NOW, NOW, successor.identity, Decimal("1.10"), Decimal("10"), Decimal("5"),
                    Decimal("10"), None, successor.identity.openrouter_expires_at_utc,
                    Decimal("0.20"), Decimal("10"), frozenset(), "c" * 64,
                )

            gate = OwnedPilotBudgetGate(writer, successor, collect, utcnow=lambda: NOW)
            owner = asyncio.create_task(writer.run())
            try:
                assert await writer.wait_ready()
                assert bool(await gate.refresh()) is (used < total)
                assert not gate.permits(old_permit)
                assert not await writer.owned_budget_request("dispatch", permit=old_permit)
                # Native consumed-ID reader, including attempts to change historical allowance.
                assert await writer.qualification_run_consumed(UUID(int=99))
                assert await writer.qualification_run_consumed(
                    UUID(int=99), total_calls=6, profile_sha256=bytes.fromhex(TARGET_PROFILE))
                assert await writer.qualification_run_consumed(
                    UUID(int=100), total_calls=2, profile_sha256=bytes.fromhex("5" * 64))
                changes = (
                    dict(total_calls=2, profile_sha256=bytes.fromhex(TARGET_PROFILE)),
                    dict(total_calls=6, profile_sha256=bytes.fromhex("5" * 64)),
                )[restart]
                assert writer.fatal_fault is None
                # Each independent owner rejects one conflict and then closes natively.
                with pytest.raises(CommandConflictError, match="^qualification_run_conflict$"):
                    await writer.qualification_run_consumed(UUID(int=100), **changes)
                assert writer.fatal_fault.code == "qualification_run_conflict"
            finally:
                if writer.fatal_fault is None:
                    await writer.drain(2)
                await asyncio.wait_for(owner, timeout=2)
            with sqlite3.connect(path) as observed:
                # No bookkeeping exemption: the native owner must preserve the complete fixture.
                assert full_state(observed) == after_cas
                assert observed.execute("SELECT budget_permit_json FROM call_leases").fetchone() == (
                    old_permit.model_dump_json(),)

    asyncio.run(reopen_twice())


if __name__ == "__main__":
    print("fresh2GiB GiB (Windows:physical/page/virtual; Linux:physical/commit):",
          fresh_memory_guard(), flush=True)
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", "--tb=short", *sys.argv[1:]]))
