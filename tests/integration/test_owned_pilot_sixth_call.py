"""Single finite same-budget extension; all provider and call effects are offline."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate, affordable
from projetv0_voice.persistence.schema import (
    OWNED_AUDIO_BUDGET_SCHEMA_SQL,
    OWNED_BUDGET_SCHEMA_SQL,
)
from tests.integration.test_fixed_v2_process import KEY, NOW, registry_case
from tests.integration.test_owned_pilot_budget_admission import (
    attach,
    budget_rows,
    committed,
    saved_permit,
)
from tests.unit.test_owned_pilot_budget import budget_policy, observation
from tests.unit.test_sparra_admission import event


def extended(**changes):
    return budget_policy(
        total_calls=6,
        qualified_profile_sha256="a" * 64,
        image_digest="fixture@sha256:" + "b" * 64,
        **changes,
    )


async def bind(case, policy=None):
    async def collect():
        return observation()

    gate = OwnedPilotBudgetGate(case.writer, policy or extended(), collect, utcnow=lambda: NOW)
    case.registry.set_owned_budget(gate)
    return gate


async def extend(writer, **changes):
    return await writer.owned_budget_request(
        "extend-sixth-call",
        **{
            "observation": observation(),
            "predecessor_retired": True,
            "expected_policy_sha256": budget_policy().sha256,
            **changes,
        },
    )


def seed(path, *, version=11, **changes):
    policy = budget_policy()
    with sqlite3.connect(path) as db:
        db.executescript(
            OWNED_AUDIO_BUDGET_SCHEMA_SQL if version == 11 else OWNED_BUDGET_SCHEMA_SQL
        )
        values = (
            dict(
                policy_json=policy.model_dump_json(),
                policy_sha256=policy.sha256,
                used_calls=5,
                openrouter_floor="1.10",
                telnyx_floor="0.20",
                suspended=0,
                paid_call_control_id=None,
            )
            | changes
        )
        db.execute(
            "INSERT INTO owned_pilot_budget VALUES(?,?,?,?,?,?,?,?)",
            ("sparra-original-pilot", *values.values()),
        )
        db.execute(
            "INSERT INTO qualification_runs(run_id,consumed_at) VALUES(?,?)",
            (str(UUID(int=99)), NOW.isoformat()),
        )


def all_rows(path):
    with sqlite3.connect(path) as db:
        return {
            name: db.execute(f'SELECT * FROM "{name}"').fetchall()
            for (name,) in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }


@pytest.mark.asyncio
async def test_extension_preserves_five_native_permits_and_grants_only_ordinal_six(tmp_path):
    async with registry_case(tmp_path) as case:
        gate = await attach(case, fact=observation(openrouter_reported_spend_usd=Decimal("1.10")))
        for i in range(5):
            control = f"historical-{i}"
            assert (
                await committed(
                    case.registry, case.writer, event(occurred_at=NOW, call_control_id=control)
                )
            ).status_code == 200
            await committed(
                case.registry,
                case.writer,
                event("call.hangup", occurred_at=NOW, call_control_id=control),
            )
        assert not await gate.refresh()
        old_permit = saved_permit(case.path, "historical-4")
        assert (
            await committed(
                case.registry, case.writer, event(occurred_at=NOW, call_control_id="old-sixth")
            )
        ).status_code == 503
    before = all_rows(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        gate = await bind(case)
        assert not await gate.refresh()  # Startup/observe alone cannot extend.
        assert not await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True
        )
        outcomes = await asyncio.gather(extend(case.writer), extend(case.writer))
        assert outcomes.count(True) == 1
        after = all_rows(case.path)
        for table, rows in before.items():
            if table != "owned_pilot_budget":
                assert after[table] == rows
        assert budget_rows(case.path) == [(5, "1.10", "0", None, 0)]
        assert not gate.permits(old_permit)
        assert not await case.writer.owned_budget_request("dispatch", permit=old_permit)
        assert await gate.refresh()
        assert (
            await committed(
                case.registry, case.writer, event(occurred_at=NOW, call_control_id="sixth")
            )
        ).status_code == 200
        permit = saved_permit(case.path, "sixth")
        assert permit.ordinal == 6 and gate.permits(permit)
        await committed(
            case.registry,
            case.writer,
            event("call.hangup", occurred_at=NOW, call_control_id="sixth"),
        )
        assert budget_rows(case.path)[0][0] == 6
    async with registry_case(tmp_path) as case:
        gate = await bind(case)
        assert not await gate.refresh() and not await extend(case.writer)
        assert (
            await committed(
                case.registry, case.writer, event(occurred_at=NOW, call_control_id="seventh")
            )
        ).status_code == 503
        assert not case.provider.actions and budget_rows(case.path)[0][0] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "expected_hash",
        "missing_hash",
        "retired",
        "count",
        "suspended",
        "unsettled",
        "active_local",
        "old_policy",
        "active_carrier",
        "expired",
        "envelope",
        "stale",
        "foreign_observation",
        "cash_or",
        "cash_tel",
        "key_remaining",
        "cap_or",
        "cap_tel",
        "identity",
        "uuid",
        "anchor",
        "paid_until",
        "bundle",
        "inference",
        "malformed_predecessor",
    ],
)
async def test_invalid_extension_is_inert_and_does_not_migrate(tmp_path, invalid):
    row_changes = {
        "count": {"used_calls": 4},
        "suspended": {"suspended": 1},
        "unsettled": {"paid_call_control_id": "unknown"},
        "malformed_predecessor": {"policy_json": "{}"},
    }
    seed(tmp_path / "voice.sqlite", **row_changes.get(invalid, {}))
    policy_changes = {
        "identity": {"identity": replace(budget_policy().identity, telnyx_account_sha256="e" * 64)},
        "uuid": {"policy_id": UUID(int=2)},
        "anchor": {"openrouter_anchor_usd": Decimal("0.1")},
        "paid_until": {"paid_until": NOW + timedelta(hours=23)},
        "bundle": {"agent_bundle_sha256": "e" * 64},
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
        "cap_or": {"openrouter_reported_spend_usd": Decimal("3.744933")},
        "cap_tel": {"telnyx_reported_spend_usd": Decimal("4.380001")},
    }
    clock = NOW + timedelta(days=2) if invalid == "expired" else NOW
    if invalid == "envelope":
        clock = budget_policy().paid_until - timedelta(seconds=334)
    if invalid in {"expired", "envelope"}:
        fact_changes[invalid] = {
            "collection_started_at_utc": clock,
            "collection_finished_at_utc": clock,
        }
    async with registry_case(tmp_path, utcnow=lambda: clock) as case:
        await bind(
            case,
            budget_policy()
            if invalid == "old_policy"
            else extended(**policy_changes.get(invalid, {})),
        )
        if invalid == "active_local":
            with sqlite3.connect(case.path) as db:
                db.execute(
                    "INSERT INTO call_leases(call_control_id,call_id,tenant_id,agent_id,"
                    "state,token_hash,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        "active",
                        str(UUID(int=7)),
                        "fixture",
                        "fixture",
                        "pending",
                        bytes(32),
                        NOW.isoformat(),
                        (NOW + timedelta(minutes=1)).isoformat(),
                    ),
                )
        before = all_rows(case.path)
        assert not await extend(
            case.writer,
            **{
                "observation": observation(**fact_changes.get(invalid, {})),
                **({"expected_policy_sha256": "f" * 64} if invalid == "expected_hash" else {}),
                **({"expected_policy_sha256": None} if invalid == "missing_hash" else {}),
                **({"predecessor_retired": False} if invalid == "retired" else {}),
            },
        )
        assert all_rows(case.path) == before
        with sqlite3.connect(case.path) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (11,)


@pytest.mark.asyncio
async def test_empty_store_cannot_enroll_or_extend_sixth_policy(tmp_path):
    async with registry_case(tmp_path) as case:
        await bind(case)
        assert not await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True
        )
        assert not await extend(case.writer)
        assert budget_rows(case.path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version,target", [(10, 12), (11, 13)])
async def test_explicit_forward_schema_preserves_rows_and_restarts(tmp_path, version, target):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter

    path = tmp_path / "voice.sqlite"
    seed(path, version=version)
    before = all_rows(path)
    for iteration in range(2):
        writer = PersistenceWriter(
            path,
            CryptoKeyring({1: KEY}, active_version=1),
            contract_version=1 if version == 10 else 2,
            utcnow=lambda: NOW,
            process_agent_id="fixture",
            process_deployment_id="fixture",
        )
        writer.set_owned_budget(extended())
        owner = asyncio.create_task(writer.run())
        try:
            assert await writer.wait_ready()
            assert bool(await extend(writer)) is (iteration == 0)
        finally:
            await writer.drain(2)
            await owner
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (target,)
        after = all_rows(path)
        for table, rows in before.items():
            if table != "owned_pilot_budget":
                assert after[table] == rows
        assert after["owned_pilot_budget"][0][3:] == before["owned_pilot_budget"][0][3:]


def test_sixth_policy_preserves_original_reserve_math_and_refuses_seventh():
    assert budget_policy().total_calls == 5
    assert not affordable(
        budget_policy(), observation(), 5, reported_floors=(Decimal("0"), Decimal("0"))
    )
    assert affordable(extended(), observation(), 5, reported_floors=(Decimal("0"), Decimal("0")))
    assert not affordable(
        extended(), observation(), 6, reported_floors=(Decimal("0"), Decimal("0"))
    )


@pytest.mark.asyncio
async def test_extension_persists_new_high_water_before_any_refresh_and_restart(tmp_path):
    seed(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        await bind(case)
        assert await extend(
            case.writer,
            observation=observation(
                openrouter_reported_spend_usd=Decimal("1.40"),
                telnyx_reported_spend_usd=Decimal("0.30"),
            ),
        )
        assert budget_rows(case.path) == [(5, "1.40", "0.30", None, 0)]
    async with registry_case(tmp_path) as case:
        gate = await bind(case)
        assert await gate.refresh()  # A downward restatement cannot lower H.
        assert budget_rows(case.path) == [(5, "1.40", "0.30", None, 0)]


@pytest.mark.asyncio
async def test_extension_rejects_unknown_schema_without_mutating_authority(tmp_path):
    seed(tmp_path / "voice.sqlite")
    async with registry_case(tmp_path) as case:
        await bind(case)
        with sqlite3.connect(case.path) as db:
            db.execute("CREATE TABLE foreign_schema(value TEXT)")
        before = all_rows(case.path)
        assert not await extend(case.writer)
        assert all_rows(case.path) == before
        with sqlite3.connect(case.path) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (11,)

@pytest.mark.asyncio
async def test_two_sqlite_owners_cannot_extend_the_same_original_slot_twice(tmp_path):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter

    path = tmp_path / "voice.sqlite"
    seed(path)
    writers = [PersistenceWriter(
        path, CryptoKeyring({1: KEY}, active_version=1), contract_version=2,
        utcnow=lambda: NOW, process_agent_id="fixture", process_deployment_id="fixture",
    ) for _ in range(2)]
    for writer in writers:
        writer.set_owned_budget(extended())
    owners = [asyncio.create_task(writer.run()) for writer in writers]
    try:
        assert all(await asyncio.gather(*(writer.wait_ready() for writer in writers)))
        outcomes = await asyncio.gather(*(extend(writer) for writer in writers))
        assert outcomes.count(True) == 1
        assert budget_rows(path) == [(5, "1.10", "0.20", None, 0)]
        assert not any(await asyncio.gather(*(extend(writer) for writer in writers)))
    finally:
        for writer in writers:
            await writer.drain(2)
        await asyncio.gather(*owners)
