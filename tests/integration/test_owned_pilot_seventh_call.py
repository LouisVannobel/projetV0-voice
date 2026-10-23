"""One explicit native V2 seventh permit; no provider traffic or financial renewal."""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from tests.integration.test_fixed_v2_process import NOW, registry_case
from tests.integration.test_owned_pilot_budget_admission import (
    attach,
    budget_rows,
    committed,
    saved_permit,
)
from tests.integration.test_owned_pilot_sixth_call import all_rows, bind, extend, extended
from tests.unit.test_owned_pilot_budget import budget_policy, observation
from tests.unit.test_sparra_admission import event


def seventh(**changes):
    return budget_policy(total_calls=7, qualified_profile_sha256="c" * 64,
                         image_digest="fixture@sha256:" + "d" * 64, **changes)


async def extend_seventh(writer, **changes):
    return await writer.owned_budget_request("extend-seventh-call", **{
        "observation": observation(), "predecessor_retired": True,
        "expected_policy_sha256": extended().sha256, **changes,
    })


async def native_call(case, ordinal):
    control = f"native-{ordinal}"
    assert (await committed(case.registry, case.writer,
        event(occurred_at=NOW, call_control_id=control))).status_code == 200
    assert saved_permit(case.path, control).ordinal == ordinal
    await committed(case.registry, case.writer,
        event("call.hangup", occurred_at=NOW, call_control_id=control))


async def exhausted_six(tmp_path):
    async with registry_case(tmp_path) as case:
        await attach(case)
        for ordinal in range(1, 6):
            await native_call(case, ordinal)
    async with registry_case(tmp_path) as case:
        gate = await bind(case)
        assert await extend(case.writer)
        assert await gate.refresh()
        await native_call(case, 6)
        assert budget_rows(case.path)[0][0] == 6


@pytest.mark.asyncio
async def test_explicit_seventh_preserves_six_native_admissions_and_refuses_eighth(tmp_path):
    await exhausted_six(tmp_path)
    before = all_rows(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        gate = await bind(case, seventh())
        assert not await gate.refresh()
        assert not await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True)
        outcomes = await asyncio.gather(extend_seventh(case.writer), extend_seventh(case.writer))
        assert outcomes.count(True) == 1, "native seventh extension must succeed exactly once"
        after = all_rows(case.path)
        for table, rows in before.items():
            if table != "owned_pilot_budget":
                assert after[table] == rows
        assert after["owned_pilot_budget"][0][3:] == before["owned_pilot_budget"][0][3:]
        assert not gate.ready
        assert await gate.refresh()
        observed = event(occurred_at=NOW, call_control_id="seventh")
        assert (await committed(case.registry, case.writer, observed)).status_code == 200
        duplicate = await committed(case.registry, case.writer, observed, duplicate=True)
        assert duplicate.status_code == 200
        permit = saved_permit(case.path, "seventh")
        assert permit.ordinal == 7 and gate.permits(permit)
        await committed(case.registry, case.writer,
            event("call.hangup", occurred_at=NOW, call_control_id="seventh"))
    async with registry_case(tmp_path) as case:
        gate = await bind(case, seventh())
        assert not await gate.refresh() and not await extend_seventh(case.writer)
        assert (await committed(case.registry, case.writer,
            event(occurred_at=NOW, call_control_id="eighth"))).status_code == 503
        assert not case.provider.actions and budget_rows(case.path)[0][0] == 7
        with sqlite3.connect(case.path) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (14,)


def test_raw_policy_seven_and_historical_five_six_keep_finite_reserve_math():
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPolicy, affordable
    for count in (5, 6, 7):
        policy = OwnedPilotBudgetPolicy.model_validate_json(
            budget_policy().model_copy(update={"total_calls": count}).model_dump_json())
        assert policy.total_calls == count
        assert affordable(policy, observation(), count - 1,
                          reported_floors=(Decimal("0"), Decimal("0")))
        assert not affordable(policy, observation(), count,
                              reported_floors=(Decimal("0"), Decimal("0")))
    with pytest.raises(ValidationError):
        budget_policy(total_calls=8)


def seed_six(path, *, version=13, **changes):
    from projetv0_voice.persistence.schema import (
        OWNED_AUDIO_BUDGET_SCHEMA_SQL,
        SIXTH_AUDIO_BUDGET_SCHEMA_SQL,
        SIXTH_BUDGET_SCHEMA_SQL,
    )
    schema = {11: OWNED_AUDIO_BUDGET_SCHEMA_SQL, 12: SIXTH_BUDGET_SCHEMA_SQL,
              13: SIXTH_AUDIO_BUDGET_SCHEMA_SQL}[version]
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        values = dict(policy_json=extended().model_dump_json(), policy_sha256=extended().sha256,
                      used_calls=5 if version == 11 else 6, openrouter_floor="1.10",
                      telnyx_floor="0.20", suspended=0, paid_call_control_id=None) | changes
        db.execute("INSERT INTO owned_pilot_budget VALUES(?,?,?,?,?,?,?,?)",
                   ("sparra-original-pilot", *values.values()))
        db.execute("INSERT INTO qualification_runs(run_id,consumed_at) VALUES(?,?)",
                   (str(UUID(int=99)), NOW.isoformat()))


def authority(path):
    with sqlite3.connect(path) as db:
        return (all_rows(path), db.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall(),
                db.execute("PRAGMA user_version").fetchone())


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "expected_hash", "missing_hash", "retired", "count", "suspended", "unsettled",
    "active_local", "old_policy", "skip_five", "active_carrier", "expired", "envelope",
    "stale", "foreign_observation", "cash_or", "cash_tel", "key_remaining", "cap_or",
    "cap_tel", "identity", "uuid", "anchor", "paid_until", "runtime", "deployment",
    "bundle", "inference", "malformed_predecessor", "noncanonical", "wrong_predecessor_hash",
    "unknown_schema", "old_schema", "floor_nan", "floor_negative",
])
async def test_rejected_seventh_preserves_every_row_and_schema(tmp_path, invalid):
    path = tmp_path / "voice.sqlite"
    row_changes = {
        "count": {"used_calls": 5}, "suspended": {"suspended": 1},
        "unsettled": {"paid_call_control_id": "unknown"},
        "malformed_predecessor": {"policy_json": "{}"},
        "noncanonical": {"policy_json": extended().model_dump_json(indent=2)},
        "wrong_predecessor_hash": {"policy_sha256": "f" * 64},
        "skip_five": {"policy_json": budget_policy().model_dump_json(),
                      "policy_sha256": budget_policy().sha256, "used_calls": 5},
        "floor_nan": {"openrouter_floor": "NaN"},
        "floor_negative": {"telnyx_floor": "-1"},
    }
    seed_six(path, version=11 if invalid == "old_schema" else 13,
             **row_changes.get(invalid, {}))
    policy_changes = {
        "identity": {"identity": replace(extended().identity, telnyx_account_sha256="e" * 64)},
        "uuid": {"policy_id": UUID(int=2)}, "anchor": {"openrouter_anchor_usd": Decimal("0.1")},
        "paid_until": {"paid_until": NOW + timedelta(hours=23)},
        "runtime": {"runtime_contract_sha256": "e" * 64},
        "deployment": {"deployment_id": "other"}, "bundle": {"agent_bundle_sha256": "e" * 64},
        "inference": {"inference_profile_sha256": "e" * 64},
    }
    fact_changes = {
        "identity": {"identity": policy_changes["identity"]["identity"]},
        "anchor": {"openrouter_reported_spend_usd": Decimal("0.1")},
        "active_carrier": {"active_call_control_ids": frozenset({"active"})},
        "stale": {"collection_started_at_utc": NOW - timedelta(seconds=61)},
        "foreign_observation": {"identity": replace(observation().identity, did="+33102030406")},
        "cash_or": {"openrouter_available_cash_usd": Decimal("3.780407")},
        "cash_tel": {"telnyx_available_cash_usd": Decimal("1.219999")},
        "key_remaining": {"openrouter_limit_remaining_usd": Decimal("3.780407")},
        "cap_or": {"openrouter_reported_spend_usd": Decimal("4.250001")},
        "cap_tel": {"telnyx_reported_spend_usd": Decimal("4.500001")},
    }
    clock = NOW + timedelta(days=2) if invalid == "expired" else NOW
    if invalid == "envelope":
        clock = extended().paid_until - timedelta(seconds=334)
    if invalid in {"expired", "envelope"}:
        fact_changes[invalid] = {"collection_started_at_utc": clock,
                                 "collection_finished_at_utc": clock}
    async with registry_case(tmp_path, utcnow=lambda: clock) as case:
        await bind(case, extended() if invalid == "old_policy"
                   else seventh(**policy_changes.get(invalid, {})))
        if invalid == "active_local":
            with sqlite3.connect(path) as db:
                db.execute("INSERT INTO call_leases(call_control_id,call_id,tenant_id,agent_id,"
                           "state,token_hash,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)",
                           ("active", str(UUID(int=7)), "fixture", "fixture", "pending", bytes(32),
                            NOW.isoformat(), (NOW + timedelta(minutes=1)).isoformat()))
        if invalid == "unknown_schema":
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE foreign_schema(value TEXT)")
        before = authority(path)
        assert not await extend_seventh(case.writer, **{
            "observation": observation(**fact_changes.get(invalid, {})),
            **({"expected_policy_sha256": "f" * 64} if invalid in {
                "expected_hash", "wrong_predecessor_hash"} else {}),
            **({"expected_policy_sha256": None} if invalid == "missing_hash" else {}),
            **({"predecessor_retired": False} if invalid == "retired" else {}),
            **({"expected_policy_sha256": budget_policy().sha256}
               if invalid == "skip_five" else {}),
        })
        assert authority(path) == before


@pytest.mark.asyncio
async def test_empty_seventh_policy_cannot_enroll_or_extend_and_startup_does_not_migrate(tmp_path):
    async with registry_case(tmp_path) as case:
        await bind(case, seventh())
        before = authority(case.path)
        assert not await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True)
        assert not await extend_seventh(case.writer)
        assert authority(case.path) == before
        assert before[2] == (11,)


@pytest.mark.asyncio
async def test_highwaters_survive_extension_and_cold_schema14_startup(tmp_path):
    path = tmp_path / "voice.sqlite"
    seed_six(path, openrouter_floor="1.50", telnyx_floor="0.20")
    async with registry_case(tmp_path) as case:
        await bind(case, seventh())
        assert await extend_seventh(case.writer, observation=observation(
            openrouter_reported_spend_usd=Decimal("1.40"),
            telnyx_reported_spend_usd=Decimal("0.30")))
        assert budget_rows(path) == [(6, "1.50", "0.30", None, 0)]
    before = authority(path)
    async with registry_case(tmp_path) as case:
        gate = await bind(case, seventh())
        assert authority(path) == before
        assert await gate.refresh()
        assert budget_rows(path) == [(6, "1.50", "0.30", None, 0)]
        with sqlite3.connect(path) as db:
            for invalid in (-1, 8, 1.5):
                with pytest.raises(sqlite3.IntegrityError):
                    db.execute("UPDATE owned_pilot_budget SET used_calls=?", (invalid,))


@pytest.mark.asyncio
async def test_v1_cannot_extend_to_seventh(tmp_path):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.integration.test_fixed_v2_process import KEY
    path = tmp_path / "voice.sqlite"
    seed_six(path, version=12)
    before = authority(path)
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
                               contract_version=1, utcnow=lambda: NOW)
    writer.set_owned_budget(seventh())
    owner = asyncio.create_task(writer.run())
    try:
        assert await writer.wait_ready()
        assert not await extend_seventh(writer)
        assert authority(path) == before
    finally:
        await writer.drain(2)
        await owner


@pytest.mark.asyncio
async def test_two_native_owners_cannot_extend_same_sixth_slot_twice(tmp_path):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.integration.test_fixed_v2_process import KEY
    path = tmp_path / "voice.sqlite"
    seed_six(path)
    writers = [PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
               contract_version=2, utcnow=lambda: NOW, process_agent_id="fixture",
               process_deployment_id="fixture") for _ in range(2)]
    for writer in writers:
        writer.set_owned_budget(seventh())
    owners = [asyncio.create_task(writer.run()) for writer in writers]
    try:
        assert all(await asyncio.gather(*(writer.wait_ready() for writer in writers)))
        outcomes = await asyncio.gather(*(extend_seventh(writer) for writer in writers))
        assert outcomes.count(True) == 1
        assert budget_rows(path) == [(6, "1.10", "0.20", None, 0)]
        assert not any(await asyncio.gather(*(extend_seventh(writer) for writer in writers)))
    finally:
        for writer in writers:
            await writer.drain(2)
        await asyncio.gather(*owners)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unknown_version", "malformed", "v1"])
async def test_schema14_invalid_startup_is_inert(tmp_path, invalid):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.schema import SEVENTH_AUDIO_BUDGET_SCHEMA_SQL
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.integration.test_fixed_v2_process import KEY
    path = tmp_path / "voice.sqlite"
    schema = SEVENTH_AUDIO_BUDGET_SCHEMA_SQL
    if invalid == "unknown_version":
        schema = schema.replace("user_version = 14", "user_version = 15")
    if invalid == "malformed":
        schema = schema.replace("used_calls BETWEEN 0 AND 7", "used_calls BETWEEN 0 AND 8")
    with sqlite3.connect(path) as db:
        db.executescript(schema)
    frozen = path.read_bytes()
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
                               contract_version=1 if invalid == "v1" else 2,
                               utcnow=lambda: NOW, process_agent_id="fixture",
                               process_deployment_id="fixture")
    writer.set_owned_budget(seventh())
    owner = asyncio.create_task(writer.run())
    assert not await writer.wait_ready()
    await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 2)
    assert path.read_bytes() == frozen
