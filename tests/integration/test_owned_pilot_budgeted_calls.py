"""Budget-only native admission, using synthetic funding and real SQLite only."""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate
from projetv0_voice.persistence.commands import FatalPersistenceError
from projetv0_voice.persistence.schema import (
    BUDGETED_AUDIO_BUDGET_SCHEMA_SQL,
    SEVENTH_AUDIO_BUDGET_SCHEMA_SQL,
)
from tests.integration.test_fixed_v2_process import NOW, registry_case
from tests.integration.test_owned_pilot_budget_admission import budget_rows, committed, saved_permit
from tests.integration.test_owned_pilot_seventh_call import authority, native_call
from tests.integration.test_owned_pilot_sixth_call import all_rows
from tests.unit.test_owned_pilot_budget import budgeted_observation, budgeted_policy
from tests.unit.test_sparra_admission import event


def predecessor():
    return budgeted_policy(total_calls=7)


def candidate(**changes):
    return budgeted_policy(**{
        "qualified_profile_sha256": "c" * 64,
        "image_digest": "fixture@sha256:" + "d" * 64, **changes,
    })


def seed_seven(path, **changes):
    with sqlite3.connect(path) as db:
        db.executescript(SEVENTH_AUDIO_BUDGET_SCHEMA_SQL)
        values = dict(policy_json=predecessor().model_dump_json(),
                      policy_sha256=predecessor().sha256, used_calls=7,
                      openrouter_floor="0.33", telnyx_floor="3.30", suspended=0,
                      paid_call_control_id=None) | changes
        db.execute("INSERT INTO owned_pilot_budget VALUES(?,?,?,?,?,?,?,?)",
                   ("sparra-original-pilot", *values.values()))
        db.execute("INSERT INTO qualification_runs(run_id,consumed_at) VALUES(?,?)",
                   (str(UUID(int=99)), NOW.isoformat()))


async def bind_budgeted(case, *, policy=None, fact=None, utcnow=lambda: NOW):
    async def collect():
        return fact or budgeted_observation()

    gate = OwnedPilotBudgetGate(case.writer, policy or candidate(), collect, utcnow=utcnow)
    case.registry.set_owned_budget(gate)
    return gate


async def enable(writer, **changes):
    return await writer.owned_budget_request("enable-budgeted-calls", **{
        "observation": budgeted_observation(), "predecessor_retired": True,
        "expected_policy_sha256": predecessor().sha256, **changes,
    })


async def exhausted_seven(tmp_path):
    for limit in (5, 6, 7):
        async with registry_case(tmp_path) as case:
            gate = await bind_budgeted(case, policy=budgeted_policy(total_calls=limit))
            action = {5: "enroll", 6: "extend-sixth-call", 7: "extend-seventh-call"}[limit]
            assert await case.writer.owned_budget_request(
                action, observation=budgeted_observation(), predecessor_retired=True,
                expected_policy_sha256=budgeted_policy(total_calls=limit - 1).sha256
                if limit > 5 else None)
            assert await gate.refresh()
            for ordinal in (range(1, 6) if limit == 5 else (limit,)):
                await native_call(case, ordinal)


@pytest.mark.asyncio
@pytest.mark.parametrize("answered_first", [False, True])
async def test_native_eighth_permit_outlives_closed_new_admissions(tmp_path, answered_first):
    await exhausted_seven(tmp_path)
    before = all_rows(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        assert not await gate.refresh()
        assert await enable(case.writer), "explicit schema14/N7 -> schema15/null must succeed"
        after = all_rows(case.path)
        for table, rows in before.items():
            if table != "owned_pilot_budget":
                assert after[table] == rows
        assert budget_rows(case.path) == [(7, "0.33", "3.30", None, 0)]
        assert not gate.ready
        assert await gate.refresh()
        old_permit = saved_permit(case.path, "native-7")
        assert not gate.permits(old_permit)
        assert not await gate.dispatch(old_permit)
        initiated = event(occurred_at=NOW, call_control_id="eighth")
        answered = event("call.answered", occurred_at=NOW, call_control_id="eighth")
        if answered_first:
            await committed(case.registry, case.writer, answered)
        assert (await committed(case.registry, case.writer, initiated)).status_code == 200
        duplicate = await committed(case.registry, case.writer, initiated, duplicate=True)
        assert duplicate.status_code == 200
        if not answered_first:
            await committed(case.registry, case.writer, answered)
        permit = saved_permit(case.path, "eighth")
        assert permit.ordinal == 8 and gate.permits(permit)
        assert not await gate.dispatch(permit.model_copy(update={"generation": UUID(int=100)}))
        assert not gate.ready and not await gate.refresh()
        assert gate.permits(permit)
        assert await gate.dispatch(permit)
        await committed(case.registry, case.writer,
                        event("call.hangup", occurred_at=NOW, call_control_id="eighth"))
        assert not await gate.refresh()
        assert (await committed(case.registry, case.writer,
                    event(occurred_at=NOW, call_control_id="ninth"))).status_code == 503
        assert budget_rows(case.path)[0][0] == 8
        assert [a[0] for a in case.provider.actions] == (
            ["streaming"] if answered_first else ["answer", "streaming"])
        with sqlite3.connect(case.path) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (15,)
    before_restart = authority(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        assert authority(case.path) == before_restart
        assert not await gate.refresh() and not await enable(case.writer)
        assert not case.provider.actions and not case.begins


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "expected_hash", "missing_hash", "retired", "count", "suspended", "unsettled",
    "active_local", "finite_policy", "active_carrier", "expired", "envelope", "stale",
    "foreign_observation", "cash_or", "cash_tel", "key_floor", "cap_or", "cap_tel",
    "identity", "uuid", "anchor", "paid_until", "runtime", "deployment", "bundle",
    "inference", "malformed_predecessor", "noncanonical", "wrong_predecessor_hash",
    "skip_six", "unknown_schema", "floor_nan", "floor_negative",
])
async def test_refused_enablement_preserves_all_rows_and_schema(tmp_path, invalid):
    path = tmp_path / "voice.sqlite"
    row_changes = {
        "count": {"used_calls": 6}, "suspended": {"suspended": 1},
        "unsettled": {"paid_call_control_id": "unknown-paid-intent"},
        "malformed_predecessor": {"policy_json": "{}"},
        "noncanonical": {"policy_json": predecessor().model_dump_json(indent=2)},
        "wrong_predecessor_hash": {"policy_sha256": "f" * 64},
        "skip_six": {"policy_json": budgeted_policy(total_calls=6).model_dump_json(),
                     "policy_sha256": budgeted_policy(total_calls=6).sha256},
        "floor_nan": {"openrouter_floor": "NaN"},
        "floor_negative": {"telnyx_floor": "-1"},
    }
    seed_seven(path, **row_changes.get(invalid, {}))
    policy_changes = {
        "identity": {"identity": replace(predecessor().identity, telnyx_account_sha256="e" * 64)},
        "uuid": {"policy_id": UUID(int=2)}, "anchor": {"openrouter_anchor_usd": Decimal("0.31")},
        "paid_until": {"paid_until": NOW + timedelta(hours=23)},
        "runtime": {"runtime_contract_sha256": "e" * 64}, "deployment": {"deployment_id": "other"},
        "bundle": {"agent_bundle_sha256": "e" * 64},
        "inference": {"inference_profile_sha256": "e" * 64},
        "finite_policy": {"total_calls": 7},
    }
    fact_changes = {
        "identity": {"identity": policy_changes["identity"]["identity"]},
        "active_carrier": {"active_call_control_ids": frozenset({"active"})},
        "stale": {"collection_started_at_utc": NOW - timedelta(seconds=61)},
        "foreign_observation": {"identity": replace(predecessor().identity, did="+33102030406")},
        "cash_or": {"openrouter_available_cash_usd": Decimal("4.255475")},
        "cash_tel": {"telnyx_available_cash_usd": Decimal("1.319999")},
        "key_floor": {"openrouter_limit_remaining_usd": Decimal("4.005476")},
        "cap_or": {"openrouter_reported_spend_usd": Decimal("4.250001")},
        "cap_tel": {"telnyx_reported_spend_usd": Decimal("4.500001")},
    }
    clock = NOW + timedelta(days=2) if invalid == "expired" else NOW
    if invalid == "envelope":
        clock = predecessor().paid_until - timedelta(seconds=334)
    if invalid in {"expired", "envelope"}:
        fact_changes[invalid] = {"collection_started_at_utc": clock,
                                 "collection_finished_at_utc": clock}
    async with registry_case(tmp_path, utcnow=lambda: clock) as case:
        await bind_budgeted(case, policy=candidate(**policy_changes.get(invalid, {})))
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
        assert not await enable(case.writer, **{
            "observation": budgeted_observation(**fact_changes.get(invalid, {})),
            **({"expected_policy_sha256": "f" * 64} if invalid in {
                "expected_hash", "wrong_predecessor_hash"} else {}),
            **({"expected_policy_sha256": None} if invalid == "missing_hash" else {}),
            **({"predecessor_retired": False} if invalid == "retired" else {}),
        })
        assert authority(path) == before


@pytest.mark.asyncio
async def test_competing_enablement_is_one_cas_and_does_not_replay(tmp_path):
    seed_seven(tmp_path / "voice.sqlite", openrouter_floor="0.40", telnyx_floor="3.32")
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        outcomes = await asyncio.gather(enable(case.writer), enable(case.writer))
        assert outcomes.count(True) == 1
        assert budget_rows(case.path) == [(7, "0.40", "3.32", None, 0)]
        assert not gate.ready
        assert await gate.refresh()
        assert not await enable(case.writer)


@pytest.mark.asyncio
async def test_missing_budgeted_store_never_enrolls_or_creates_allowance(tmp_path):
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        before = authority(case.path)
        assert not await enable(case.writer)
        assert not await case.writer.owned_budget_request(
            "enroll", observation=budgeted_observation(), predecessor_retired=True)
        assert not await gate.refresh() and not gate.ready
        assert authority(case.path) == before


@pytest.mark.asyncio
async def test_enablement_rollback_keeps_exact_schema14_and_history(tmp_path):
    path = tmp_path / "voice.sqlite"
    seed_seven(path)
    fail = False

    async def rollback(name):
        if fail and name == "after_mutation_before_commit":
            raise RuntimeError("offline transition rollback")

    async with registry_case(tmp_path, failpoint=rollback) as case:
        gate = await bind_budgeted(case)
        before = authority(path)
        fail = True
        with pytest.raises(FatalPersistenceError):
            await enable(case.writer)
        assert authority(path) == before
        assert not gate.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("reported", [False, True])
async def test_pending_eighth_admission_restricts_committed_without_publishing_permit(
    tmp_path, monkeypatch, stage, reported,
):
    policy = candidate(admission_accounting="reported" if reported else None)
    clock = [NOW]
    if reported:
        seed_release(tmp_path / "voice.sqlite", policy=policy, used_calls=8)
    else:
        seed_seven(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path, utcnow=lambda: clock[0]) as case:
        gate = await bind_budgeted(case, policy=policy, utcnow=lambda: clock[0])
        if reported:
            clock[0] += timedelta(microseconds=1)

            async def collect():
                return budgeted_observation(collection_started_at_utc=clock[0],
                                            collection_finished_at_utc=clock[0])

            gate.collect = collect
        else:
            assert await enable(case.writer)
        assert await gate.refresh()
        entered, release = asyncio.Event(), asyncio.Event()
        native_commit = case.writer._connection.commit

        async def commit():
            if stage == "after":
                await native_commit()
            entered.set()
            await release.wait()
            if stage == "before":
                await native_commit()

        monkeypatch.setattr(case.writer._connection, "commit", commit)
        admission = asyncio.create_task(committed(case.registry, case.writer,
                                    event(occurred_at=NOW, call_control_id="eighth")))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            previous, pending = case.writer.owned_budget_snapshots()
            assert previous.admission_available and not pending.admission_available
            assert not gate.ready and not case.provider.actions and not case.begins
            assert not admission.done()
            assert budget_rows(case.path)[0][0] == (8 if stage == "after" else 7) + reported
        finally:
            release.set()
            assert (await admission).status_code == 200
        monkeypatch.setattr(case.writer._connection, "commit", native_commit)
        permit = saved_permit(case.path, "eighth")
        assert permit.ordinal == 8 + reported and gate.permits(permit) and not gate.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("improving", [False, True])
async def test_budgeted_permit_financial_views_intersect_in_both_orders(
    tmp_path, monkeypatch, stage, improving,
):
    seed_seven(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        assert await enable(case.writer) and await gate.refresh()
        assert (await committed(case.registry, case.writer,
                    event(occurred_at=NOW, call_control_id="eighth"))).status_code == 200
        permit = saved_permit(case.path, "eighth")
        bad = budgeted_observation(openrouter_limit_remaining_usd=Decimal("4.005476"))
        good = budgeted_observation()
        if improving:
            assert not await case.writer.owned_budget_request("observe", observation=bad)
            assert not gate.permits(permit)
        next_fact = good if improving else bad

        async def collect():
            return next_fact

        gate.collect = collect
        entered, release = asyncio.Event(), asyncio.Event()
        native_commit = case.writer._connection.commit

        async def commit():
            if stage == "after":
                await native_commit()
            entered.set()
            await release.wait()
            if stage == "before":
                await native_commit()

        monkeypatch.setattr(case.writer._connection, "commit", commit)
        refresh = asyncio.create_task(gate.refresh())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            views = case.writer.owned_budget_snapshots()
            assert len(views) == 2 and not gate.permits(permit) and not gate.ready
        finally:
            release.set()
            await refresh
        monkeypatch.setattr(case.writer._connection, "commit", native_commit)
        assert gate.permits(permit) is improving
        assert await gate.dispatch(permit) is improving


@pytest.mark.asyncio
async def test_sqlite_count_maximum_is_technical_and_cannot_overflow(tmp_path):
    seed_seven(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        assert await enable(case.writer)
        with sqlite3.connect(case.path) as db:
            for invalid in (-1, 1.5, str(2**63)):
                with pytest.raises(sqlite3.IntegrityError):
                    db.execute("UPDATE owned_pilot_budget SET used_calls=?", (invalid,))
            db.execute("UPDATE owned_pilot_budget SET used_calls=?", (2**63 - 1,))
        assert not await gate.refresh()
        assert (await committed(case.registry, case.writer,
                    event(occurred_at=NOW, call_control_id="overflow"))).status_code == 503
        assert budget_rows(case.path)[0][0] == 2**63 - 1
        assert not case.provider.actions and not case.begins


@pytest.mark.asyncio
async def test_two_native_owners_enable_exactly_one_transition(tmp_path):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.integration.test_fixed_v2_process import KEY

    path = tmp_path / "voice.sqlite"
    seed_seven(path)
    writers = [PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
               contract_version=2, utcnow=lambda: NOW, process_agent_id="fixture",
               process_deployment_id="fixture") for _ in range(2)]
    for writer in writers:
        writer.set_owned_budget(candidate())
    owners = [asyncio.create_task(writer.run()) for writer in writers]
    try:
        assert all(await asyncio.gather(*(writer.wait_ready() for writer in writers)))
        outcomes = await asyncio.gather(*(enable(writer) for writer in writers))
        assert outcomes.count(True) == 1
        assert budget_rows(path) == [(7, "0.33", "3.30", None, 0)]
        assert not any(await asyncio.gather(*(enable(writer) for writer in writers)))
    finally:
        for writer in writers:
            await writer.drain(2)
        await asyncio.gather(*owners)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unknown_version", "malformed", "v1"])
async def test_schema15_requires_exact_v2_shape_before_any_startup_mutation(tmp_path, invalid):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.schema import BUDGETED_AUDIO_BUDGET_SCHEMA_SQL
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.integration.test_fixed_v2_process import KEY

    path = tmp_path / "voice.sqlite"
    schema = BUDGETED_AUDIO_BUDGET_SCHEMA_SQL
    if invalid == "unknown_version":
        schema = schema.replace("user_version = 15", "user_version = 16")
    if invalid == "malformed":
        schema = schema.replace("used_calls BETWEEN 0 AND 9223372036854775807",
                                "used_calls BETWEEN 0 AND 7")
    with sqlite3.connect(path) as db:
        db.executescript(schema)
    frozen = path.read_bytes()
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
                               contract_version=1 if invalid == "v1" else 2,
                               utcnow=lambda: NOW, process_agent_id="fixture",
                               process_deployment_id="fixture")
    writer.set_owned_budget(candidate())
    owner = asyncio.create_task(writer.run())
    assert not await writer.wait_ready()
    await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 2)
    assert path.read_bytes() == frozen


@pytest.mark.asyncio
async def test_suspended_restored_budgeted_store_has_no_regrant_path(tmp_path):
    path = tmp_path / "voice.sqlite"
    seed_seven(path)
    async with registry_case(tmp_path) as case:
        await bind_budgeted(case)
        assert await enable(case.writer)
        assert await case.writer.owned_budget_request("suspend")
    before = authority(path)
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        assert not await gate.refresh() and not await enable(case.writer)
        assert not await case.writer.owned_budget_request(
            "enroll", observation=budgeted_observation(), predecessor_retired=True)
        assert not gate.ready and authority(path) == before


def seed_release(path, *, policy=None, **changes):
    previous = policy or candidate()
    with sqlite3.connect(path) as db:
        db.executescript(BUDGETED_AUDIO_BUDGET_SCHEMA_SQL)
        values = dict(policy_json=previous.model_dump_json(), policy_sha256=previous.sha256,
                      used_calls=7, openrouter_floor="0.33", telnyx_floor="3.30", suspended=0,
                      paid_call_control_id=None) | changes
        db.execute("INSERT INTO owned_pilot_budget VALUES(?,?,?,?,?,?,?,?)",
                   ("sparra-original-pilot", *values.values()))
        db.execute("INSERT INTO qualification_runs(run_id,consumed_at) VALUES(?,?)",
                   (str(UUID(int=99)), NOW.isoformat()))


def release_candidate(**changes):
    return candidate(qualified_profile_sha256="e" * 64,
                     image_digest="fixture@sha256:" + "f" * 64, **changes)


async def rebind_release(writer, **changes):
    return await writer.owned_budget_request("rebind-release", **{
        "observation": budgeted_observation(), "predecessor_retired": True,
        "expected_policy_sha256": candidate().sha256,
        "predecessor_qualified_profile_sha256": candidate().qualified_profile_sha256,
        "predecessor_image_digest": candidate().image_digest, **changes,
    })


async def enable_reported(writer, **changes):
    return await writer.owned_budget_request("enable-reported-spend", **{
        "observation": budgeted_observation(), "predecessor_retired": True,
        "expected_policy_sha256": candidate().sha256, "expected_used_calls": 8, **changes,
    })


@pytest.mark.asyncio
async def test_reported_opt_in_conserves_eight_native_calls_and_requires_new_collection(tmp_path):
    await exhausted_seven(tmp_path)
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case)
        assert await enable(case.writer) and await gate.refresh()
        await native_call(case, 8)
    before_rows, schema, version = authority(tmp_path / "voice.sqlite")
    policy = candidate(admission_accounting="reported")
    clock = [NOW]

    async def collect():
        return budgeted_observation(collection_started_at_utc=clock[0],
                                    collection_finished_at_utc=clock[0])

    async with registry_case(tmp_path, utcnow=lambda: clock[0]) as case:
        gate = await bind_budgeted(case, policy=policy, utcnow=lambda: clock[0])
        gate.collect = collect
        assert not gate.ready
        assert await enable_reported(case.writer)
        expected = dict(before_rows)
        row = list(before_rows["owned_pilot_budget"][0])
        row[1:3] = [policy.model_dump_json(), policy.sha256]
        expected["owned_pilot_budget"] = [tuple(row)]
        assert authority(case.path) == (expected, schema, version)
        assert version == (15,) and budget_rows(case.path)[0][0] == 8
        assert not await enable_reported(case.writer)
        assert not gate.ready and not await gate.refresh()
        clock[0] += timedelta(microseconds=1)
        assert await gate.refresh()
        await native_call(case, 9)
        assert budget_rows(case.path)[0][0] == 9 and not gate.ready
        clock[0] += timedelta(microseconds=1)
        assert await gate.refresh()
    before_restart = authority(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path, utcnow=lambda: clock[0]) as case:
        gate = await bind_budgeted(case, policy=policy, utcnow=lambda: clock[0])
        gate.collect = collect
        assert authority(case.path) == before_restart
        assert not gate.ready and not await gate.refresh()
        clock[0] += timedelta(microseconds=1)
        assert await gate.refresh()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "hash", "count", "retired", "paid_intent", "suspended", "carrier", "stale",
    "anchor", "schema", "rebind_only",
])
async def test_reported_opt_in_refuses_wrong_predecessor_or_exposure(tmp_path, invalid):
    path = tmp_path / "voice.sqlite"
    seed_release(path, used_calls=8, **{
        "paid_intent": {"paid_call_control_id": "unknown"},
        "suspended": {"suspended": 1},
    }.get(invalid, {}))
    policy = candidate(admission_accounting="reported", **{
        "anchor": {"openrouter_anchor_usd": Decimal("0.31")},
    }.get(invalid, {}))
    async with registry_case(tmp_path) as case:
        await bind_budgeted(case, policy=policy)
        if invalid == "schema":
            with sqlite3.connect(path) as db:
                db.execute("PRAGMA user_version=14")
        before = authority(path)
        changes = {
            "hash": {"expected_policy_sha256": "a" * 64},
            "count": {"expected_used_calls": 7},
            "retired": {"predecessor_retired": False},
            "carrier": {"observation": budgeted_observation(
                active_call_control_ids=frozenset({"active"}))},
            "stale": {"observation": budgeted_observation(
                collection_started_at_utc=NOW - timedelta(seconds=61))},
        }.get(invalid, {})
        if invalid == "rebind_only":
            assert not await rebind_release(case.writer)
        else:
            assert not await enable_reported(case.writer, **changes)
        assert authority(path) == before


@pytest.mark.asyncio
async def test_reported_late_collection_cannot_survive_later_paid_intent_resolution(
    tmp_path, monkeypatch,
):
    policy = candidate(admission_accounting="reported")
    seed_release(tmp_path / "voice.sqlite", policy=policy, used_calls=8)
    clock = [NOW]

    async def collect():
        return budgeted_observation(collection_started_at_utc=clock[0],
                                    collection_finished_at_utc=clock[0])

    async with registry_case(tmp_path, utcnow=lambda: clock[0]) as case:
        gate = await bind_budgeted(case, policy=policy, utcnow=lambda: clock[0])
        gate.collect = collect
        clock[0] += timedelta(microseconds=1)
        assert await gate.refresh()
        assert (await committed(case.registry, case.writer,
                    event(occurred_at=NOW, call_control_id="ninth"))).status_code == 200
        permit = saved_permit(case.path, "ninth")
        assert permit.ordinal == 9 and gate.permits(permit)
        assert await gate.dispatch(permit)
        assert not gate.ready
        assert await case.writer.owned_budget_request("settled", call_control_id="foreign")
        assert gate.permits(permit), "foreign settlement must preserve an active observation"
        native_command = case.writer._owned_budget_command

        async def defer_settlement(payload):
            if payload.get("action") == "settled":
                return True
            return await native_command(payload)

        monkeypatch.setattr(case.writer, "_owned_budget_command", defer_settlement)
        clock[0] += timedelta(seconds=1)
        await committed(case.registry, case.writer,
            event("call.hangup", occurred_at=NOW, call_control_id="ninth"))
        assert budget_rows(case.path)[0][3] == "ninth"
        clock[0] += timedelta(microseconds=1)
        late = await collect()  # Started after terminalization, before paid-intent resolution.
        monkeypatch.setattr(case.writer, "_owned_budget_command", native_command)
        clock[0] += timedelta(seconds=1)
        assert await case.writer.owned_budget_request("settled", call_control_id="ninth")
        assert budget_rows(case.path)[0][3] is None
        assert not await case.writer.owned_budget_request("observe", observation=late)
        assert not gate.ready
        clock[0] += timedelta(microseconds=1)
        assert await gate.refresh(), "unchanged spending may reopen only after a new collection"
        assert await case.writer.owned_budget_request("settled", call_control_id="ninth")
        assert gate.ready, "a repeated settlement must not invalidate the new observation"
        assert budget_rows(case.path)[0][0] == 9


@pytest.mark.asyncio
async def test_reported_unknown_paid_intent_survives_restart_and_cheap_observations(tmp_path):
    policy = candidate(admission_accounting="reported")
    path = tmp_path / "voice.sqlite"
    seed_release(path, policy=policy, used_calls=8, paid_call_control_id="unknown")
    before = authority(path)
    clock = [NOW]
    for _ in range(2):
        async with registry_case(tmp_path, utcnow=lambda: clock[0]) as case:
            gate = await bind_budgeted(case, policy=policy, utcnow=lambda: clock[0])
            clock[0] += timedelta(seconds=1)
            fact = budgeted_observation(collection_started_at_utc=clock[0],
                                        collection_finished_at_utc=clock[0])
            assert not await case.writer.owned_budget_request("observe", observation=fact)
            assert not gate.ready and authority(path) == before
            assert not case.provider.actions and not case.begins


@pytest.mark.asyncio
async def test_release_rebind_preserves_native_history_schema_and_every_budget_column(tmp_path):
    await exhausted_seven(tmp_path)
    async with registry_case(tmp_path) as case:
        await bind_budgeted(case)
        assert await enable(case.writer)
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case, policy=release_candidate())
        before_rows, schema, version = authority(case.path)
        fact = budgeted_observation(openrouter_reported_spend_usd=Decimal("0.99"),
                                   telnyx_reported_spend_usd=Decimal("3.90"))
        assert await rebind_release(case.writer, observation=fact)
        after_rows, after_schema, after_version = authority(case.path)
        expected = dict(before_rows)
        row = list(before_rows["owned_pilot_budget"][0])
        row[1:3] = [release_candidate().model_dump_json(), release_candidate().sha256]
        expected["owned_pilot_budget"] = [tuple(row)]
        assert after_rows == expected and after_schema == schema
        assert after_version == version == (15,)
        assert not gate.ready and not case.provider.actions and not case.begins
        assert not await rebind_release(case.writer)
        assert authority(case.path) == (after_rows, after_schema, after_version)
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case, policy=release_candidate())
        before = authority(case.path)
        assert not gate.ready and not await rebind_release(case.writer)
        assert authority(case.path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", ["funds", "start_window"])
async def test_release_rebind_does_not_finance_or_reopen_a_closed_budget(tmp_path, closed):
    changes = {"paid_until": NOW + timedelta(seconds=334)} if closed == "start_window" else {}
    previous, policy = candidate(**changes), release_candidate(**changes)
    seed_release(tmp_path / "voice.sqlite", policy=previous,
                 openrouter_floor="4.26" if closed == "funds" else "0.33")
    async with registry_case(tmp_path) as case:
        gate = await bind_budgeted(case, policy=policy)
        before = all_rows(case.path)["owned_pilot_budget"][0]
        assert await rebind_release(case.writer, expected_policy_sha256=previous.sha256)
        assert not gate.ready and not await gate.refresh()
        assert (await committed(case.registry, case.writer,
                    event(occurred_at=NOW, call_control_id="unfunded-release"))).status_code == 503
        after = all_rows(case.path)["owned_pilot_budget"][0]
        assert after[3:] == before[3:]
        assert not case.provider.actions and not case.begins


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "cas", "retired", "profile", "image", "finite", "identity", "anchor", "paid_until",
    "runtime", "deployment", "bundle", "inference", "equivalent_anchor",
    "active_local", "active_carrier",
    "unsettled", "suspended", "stale", "fact_identity", "cap", "schema", "noncanonical",
])
async def test_release_rebind_refuses_ambiguity_without_any_persistent_change(tmp_path, invalid):
    path = tmp_path / "voice.sqlite"
    seed_release(path, **{
        "unsettled": {"paid_call_control_id": "unsettled"}, "suspended": {"suspended": 1},
        "noncanonical": {"policy_json": candidate().model_dump_json(indent=2)},
    }.get(invalid, {}))
    changes = {
        "finite": {"total_calls": 7},
        "identity": {"identity": replace(candidate().identity, telnyx_account_sha256="a" * 64)},
        "anchor": {"openrouter_anchor_usd": Decimal("0.31")},
        "equivalent_anchor": {"openrouter_anchor_usd": Decimal("0.300")},
        "paid_until": {"paid_until": NOW + timedelta(hours=23)},
        "runtime": {"runtime_contract_sha256": "a" * 64},
        "deployment": {"deployment_id": "other"}, "bundle": {"agent_bundle_sha256": "a" * 64},
        "inference": {"inference_profile_sha256": "a" * 64},
    }
    async with registry_case(tmp_path) as case:
        await bind_budgeted(case, policy=release_candidate(**changes.get(invalid, {})))
        if invalid == "active_local":
            with sqlite3.connect(path) as db:
                db.execute("INSERT INTO call_leases(call_control_id,call_id,tenant_id,agent_id,"
                           "state,token_hash,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)",
                           ("active", str(UUID(int=7)), "fixture", "fixture", "pending", bytes(32),
                            NOW.isoformat(), (NOW + timedelta(minutes=1)).isoformat()))
        if invalid == "schema":
            with sqlite3.connect(path) as db:
                db.execute("PRAGMA user_version=14")
        before = authority(path)
        facts = {
            "active_carrier": {"active_call_control_ids": frozenset({"active"})},
            "stale": {"collection_started_at_utc": NOW - timedelta(seconds=61)},
            "fact_identity": {"identity": replace(candidate().identity, did="+33102030406")},
            "cap": {"openrouter_limit_usd": Decimal("6")},
        }
        assert not await rebind_release(case.writer, **{
            "observation": budgeted_observation(**facts.get(invalid, {})),
            **{"cas": {"expected_policy_sha256": "a" * 64},
               "retired": {"predecessor_retired": False},
               "profile": {"predecessor_qualified_profile_sha256": "a" * 64},
               "image": {"predecessor_image_digest": "fixture@sha256:" + "a" * 64},
               }.get(invalid, {}),
        })
        assert authority(path) == before


@pytest.mark.asyncio
async def test_release_rebind_rolls_back_both_policy_columns_on_native_commit_failure(tmp_path):
    path = tmp_path / "voice.sqlite"
    seed_release(path)
    fail = False

    async def rollback(name):
        if fail and name == "after_mutation_before_commit":
            cursor = await case.writer._connection.execute(
                "SELECT policy_sha256 FROM owned_pilot_budget")
            updated = await cursor.fetchone()
            await cursor.close()
            if updated == (release_candidate().sha256,):
                raise RuntimeError("offline release rollback")

    async with registry_case(tmp_path, failpoint=rollback) as case:
        gate = await bind_budgeted(case, policy=release_candidate())
        before = authority(path)
        fail = True
        with pytest.raises(FatalPersistenceError):
            await rebind_release(case.writer)
        assert authority(path) == before and not gate.ready


@pytest.mark.asyncio
async def test_release_rebind_two_native_owners_commit_only_one_cas(tmp_path):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter
    from tests.integration.test_fixed_v2_process import KEY

    path = tmp_path / "voice.sqlite"
    seed_release(path)
    writers = [PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
               contract_version=2, utcnow=lambda: NOW, process_agent_id="fixture",
               process_deployment_id="fixture") for _ in range(2)]
    for writer in writers:
        writer.set_owned_budget(release_candidate())
    owners = [asyncio.create_task(writer.run()) for writer in writers]
    try:
        assert all(await asyncio.gather(*(writer.wait_ready() for writer in writers)))
        before = authority(path)
        outcomes = await asyncio.gather(*(rebind_release(writer) for writer in writers))
        assert outcomes.count(True) == 1
        after = authority(path)
        assert after[0]["owned_pilot_budget"][0][3:] == before[0]["owned_pilot_budget"][0][3:]
        assert after[1:] == before[1:]
        assert not any(await asyncio.gather(*(rebind_release(writer) for writer in writers)))
        assert authority(path) == after
    finally:
        for writer in writers:
            await writer.drain(2)
        await asyncio.gather(*owners)
