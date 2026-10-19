"""Original budget at real native receipt/lease boundaries; no provider traffic."""

from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from tests.integration.test_fixed_v2_process import NOW, registry_case
from tests.unit.test_owned_pilot_budget import budget_policy, observation
from tests.unit.test_sparra_admission import event


async def attach(case, *, fact=None, enrolled=True, utcnow=lambda: NOW):
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate

    async def collect():
        return fact or observation()

    gate = OwnedPilotBudgetGate(case.writer, budget_policy(), collect, utcnow=utcnow)
    case.registry.set_owned_budget(gate)
    if enrolled:
        assert await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True
        )
    await gate.refresh()
    return gate


def budget_rows(path):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT used_calls,openrouter_floor,telnyx_floor,"
            "paid_call_control_id,suspended FROM owned_pilot_budget"
        ).fetchall()


async def committed(registry, writer, observed, *, duplicate=False):
    resolved = await (
        registry.resolve_duplicate_webhook(observed)
        if duplicate
        else registry.resolve_webhook(observed)
    )
    effect = resolved.effect
    result = await writer.submit_webhook(
        receipt=dict(
            event_id=observed.event_id,
            event_type=observed.event_type,
            call_control_id=observed.call_control_id,
            occurred_at=observed.occurred_at,
            received_at=NOW,
            semantic_fingerprint_sha256=observed.semantic_fingerprint_sha256,
            call_leg_id=observed.call_leg_id,
            call_session_id=observed.call_session_id,
        ),
        lease=None if effect is None else effect.lease,
        operation=None if effect is None else effect.operation,
        admission_facts=None if effect is None else effect.admission_facts,
        operation_generation=None if effect is None else effect.operation_generation,
    ).wait()
    return await registry.reconcile_after_commit(observed, resolved, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("answered_first", [False, True])
async def test_missing_original_authority_refuses_both_native_orders(tmp_path, answered_first):
    async with registry_case(tmp_path) as case:
        if importlib.util.find_spec("projetv0_voice.owned_pilot_budget") is not None:
            from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate
            from tests.unit.test_owned_pilot_budget import budget_policy, observation

            async def collect():
                return observation()

            gate = OwnedPilotBudgetGate(case.writer, budget_policy(), collect, utcnow=lambda: NOW)
            case.registry.set_owned_budget(gate)
            await gate.refresh()
        if answered_first:
            assert (
                await committed(case.registry, case.writer, event("call.answered", occurred_at=NOW))
            ).status_code == 200
        result = await committed(case.registry, case.writer, event(occurred_at=NOW))
        assert result.status_code == 503
        assert not case.provider.actions
        assert not case.begins
        assert await case.registry.snapshot("original") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("answered_first", [False, True])
async def test_one_native_permit_for_both_orders_duplicate_and_wss_race(tmp_path, answered_first):
    async with registry_case(tmp_path) as case:
        await attach(case)
        initiated = event(occurred_at=NOW)
        answered = event("call.answered", occurred_at=NOW)
        if answered_first:
            await committed(case.registry, case.writer, answered)
        assert (await committed(case.registry, case.writer, initiated)).status_code == 200
        assert (
            await committed(case.registry, case.writer, initiated, duplicate=True)
        ).status_code == 200
        if not answered_first:
            await committed(case.registry, case.writer, answered)
        snapshot = await case.registry.snapshot("original")
        claims = await asyncio.gather(
            *(
                case.registry.claim_once(
                    call_control_id="original",
                    token_digest=snapshot.token_digest,
                    abort_target_publisher=lambda _: True,
                    abort_target_clearer=lambda _: None,
                )
                for _ in range(2)
            )
        )
        assert sum(c is not None for c in claims) == 1
        assert budget_rows(case.path)[0][:3] == (1, "0", "0")
        actions = [a[0] for a in case.provider.actions]
        assert actions == (["streaming"] if answered_first else ["answer", "streaming"])


@pytest.mark.asyncio
async def test_last_affordable_permit_survives_advanced_counter(tmp_path):
    async with registry_case(tmp_path) as case:
        gate = await attach(case)
        for number in range(3):
            control = f"call-{number}"
            assert (
                await committed(
                    case.registry, case.writer, event(occurred_at=NOW, call_control_id=control)
                )
            ).status_code == 200
            await gate.refresh()  # The third admission closes readiness, not its permit.
            assert (
                await committed(
                    case.registry,
                    case.writer,
                    event("call.answered", occurred_at=NOW, call_control_id=control),
                )
            ).status_code == 200
            await committed(
                case.registry,
                case.writer,
                event("call.hangup", occurred_at=NOW, call_control_id=control),
            )
        assert not await gate.refresh()
        before = list(case.provider.actions)
        assert (
            await committed(
                case.registry, case.writer, event(occurred_at=NOW, call_control_id="fourth")
            )
        ).status_code == 503
        assert case.provider.actions == before
        assert budget_rows(case.path)[0][0] == 3


@pytest.mark.asyncio
async def test_restart_preserves_unknown_dispatch_and_no_reenrollment(tmp_path):
    async with registry_case(tmp_path) as case:
        await attach(case)
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        before = budget_rows(case.path)
    async with registry_case(tmp_path) as case:
        await attach(case, enrolled=False)
        assert budget_rows(case.path) == before
        assert not await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True
        )
        assert (
            await committed(
                case.registry, case.writer, event(occurred_at=NOW, call_control_id="new")
            )
        ).status_code == 503
        assert not case.provider.actions


@pytest.mark.asyncio
async def test_report_lag_is_valid_regression_and_known_restore_are_closed(tmp_path):
    async with registry_case(tmp_path) as case:
        gate = await attach(case)
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        # Report0 remains valid although pessimistic debit is now .505068.
        assert await case.writer.owned_budget_request("observe", observation=observation()) is False
        high = observation(openrouter_reported_spend_usd=Decimal("0.1"))
        assert await case.writer.owned_budget_request("observe", observation=high) is False
        assert await case.writer.owned_budget_request("observe", observation=observation()) is None
        assert budget_rows(case.path)[0][1] == "0.1"
        await case.writer.owned_budget_request("suspend")
        assert not await gate.refresh()
        assert budget_rows(case.path)[0][4] == 1


@pytest.mark.asyncio
async def test_stale_after_begin_await_never_enters_transport(tmp_path):
    clock = [NOW]
    async with registry_case(
        tmp_path,
        utcnow=lambda: clock[0],
        reply=lambda value, *_: clock.__setitem__(0, NOW + timedelta(seconds=61)) or value,
    ) as case:
        await attach(case, utcnow=lambda: clock[0])
        assert (
            await committed(case.registry, case.writer, event(occurred_at=NOW))
        ).status_code == 503
        assert not case.provider.actions
        assert budget_rows(case.path)[0][0] == 1  # No refund.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["policy_id", "openrouter_anchor_usd", "qualified_profile_sha256", "paid_until"]
)
async def test_changed_policy_cannot_mint_new_original_slot(tmp_path, field):
    from uuid import uuid4

    async with registry_case(tmp_path) as case:
        await attach(case)
    async with registry_case(tmp_path) as case:
        values = {
            "policy_id": uuid4(),
            "openrouter_anchor_usd": Decimal("0.1"),
            "qualified_profile_sha256": "f" * 64,
            "paid_until": NOW + timedelta(hours=23),
        }
        changed = budget_policy(**{field: values[field]})
        case.writer.set_owned_budget(changed)
        assert not await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True
        )
        assert budget_rows(case.path) == [(0, "0", "0", None, 0)]


@pytest.mark.asyncio
async def test_new_start_envelope_cannot_cross_paid_boundary(tmp_path):
    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate

    p = budget_policy(paid_until=NOW + timedelta(seconds=334))
    async with registry_case(tmp_path) as case:

        async def collect():
            return observation()

        gate = OwnedPilotBudgetGate(case.writer, p, collect, utcnow=lambda: NOW)
        case.registry.set_owned_budget(gate)
        assert await case.writer.owned_budget_request(
            "enroll", observation=observation(), predecessor_retired=True
        )
        await gate.refresh()
        assert (
            await committed(case.registry, case.writer, event(occurred_at=NOW))
        ).status_code == 503
        assert not case.provider.actions and budget_rows(case.path)[0][0] == 0


@pytest.mark.asyncio
async def test_concurrent_same_identity_consumes_once(tmp_path):
    async with registry_case(tmp_path) as case:
        await attach(case)
        results = await asyncio.gather(
            *(committed(case.registry, case.writer, event(occurred_at=NOW)) for _ in range(2))
        )
        assert [r.status_code for r in results] == [200, 200]
        assert budget_rows(case.path)[0][0] == 1
        assert [a[0] for a in case.provider.actions] == ["answer"]


@pytest.mark.asyncio
async def test_late_native_commit_burns_permit_without_dispatch(tmp_path):
    from projetv0_voice.lifecycle import RuntimeSupervisor

    armed = False
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause(name):
        nonlocal armed
        if armed and name == "after_mutation_before_commit":
            armed = False
            entered.set()
            await release.wait()

    async with registry_case(tmp_path, failpoint=pause) as case:
        await attach(case)
        supervisor = RuntimeSupervisor(
            writer=case.writer, registry=case.registry, sparra_enabled=True, utcnow=lambda: NOW
        )
        case.writer.control_commit_timeout_seconds = 0.05
        observed = event(occurred_at=NOW)
        resolution = await case.registry.resolve_webhook(observed)
        armed = True
        finalizer = supervisor.start_webhook_finalization(observed, resolution)
        await asyncio.wait_for(entered.wait(), 2)
        assert (await finalizer.wait()).status_code == 503
        release.set()
        await supervisor.webhook_finalizers.join_until_empty(asyncio.get_running_loop().time() + 2)
        assert budget_rows(case.path)[0][0] == 1
        assert not any(a[0] in {"answer", "streaming"} for a in case.provider.actions)
        await case.writer.drain(2)


@pytest.mark.asyncio
async def test_budget_migration_preserves_thirteen_historical_rows(tmp_path):
    from projetv0_voice.persistence.schema import LOCAL_AUDIO_TERMINAL_SCHEMA_SQL

    path = tmp_path / "voice.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(LOCAL_AUDIO_TERMINAL_SCHEMA_SQL)
        for i in range(13):
            db.execute(
                "INSERT INTO qualification_runs(run_id,consumed_at) VALUES(?,?)",
                (str(i), NOW.isoformat()),
            )
        tables = [
            r[0]
            for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        before = {
            t: (
                tuple(r[1] for r in db.execute(f"PRAGMA table_info({t})")),
                db.execute(f"SELECT * FROM {t}").fetchall(),
            )
            for t in tables
        }
    async with registry_case(tmp_path) as case:
        with sqlite3.connect(case.path) as db:
            for table, (columns, rows) in before.items():
                assert db.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall() == rows
            assert db.execute("SELECT count(*) FROM owned_pilot_budget").fetchone() == (0,)
            assert db.execute("PRAGMA user_version").fetchone() == (11,)


@pytest.mark.asyncio
async def test_stale_after_durable_wss_claim_does_not_publish_claim(tmp_path):
    clock = [NOW]
    armed = False

    def advance(name):
        if armed and name == "after_mutation_before_commit":
            clock[0] = NOW + timedelta(seconds=61)

    async with registry_case(tmp_path, utcnow=lambda: clock[0], failpoint=advance) as case:
        await attach(case, utcnow=lambda: clock[0])
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        await committed(case.registry, case.writer, event("call.answered", occurred_at=NOW))
        saved = await case.registry.snapshot("original")
        armed = True
        assert (
            await case.registry.claim_once(
                call_control_id="original",
                token_digest=saved.token_digest,
                abort_target_publisher=lambda _: True,
                abort_target_clearer=lambda _: None,
            )
            is None
        )
        assert budget_rows(case.path)[0][0] == 1


@pytest.mark.asyncio
async def test_unknown_effect_survives_local_expiry_and_empty_active_observation(tmp_path):
    from projetv0_voice.telnyx.call_control import CallControlResult

    tick = [100.0]
    async with registry_case(tmp_path, monotonic=lambda: tick[0]) as case:
        gate = await attach(case)

        async def unknown(*args, **kwargs):
            return CallControlResult("outcome_unknown")

        case.provider.answer = unknown
        case.provider.hangup = unknown
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        tick[0] += 31
        await case.registry.reap_expired()
        assert not await gate.refresh()
        assert budget_rows(case.path)[0][3] == "original"
    async with registry_case(tmp_path) as case:
        await attach(case, enrolled=False)
        assert (
            await committed(
                case.registry, case.writer, event(occurred_at=NOW, call_control_id="replacement")
            )
        ).status_code == 503
        assert not case.provider.actions


@pytest.mark.asyncio
async def test_paid_permit_cannot_change_generation_or_ordinal(tmp_path):
    from uuid import uuid4

    from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetPermit

    async with registry_case(tmp_path) as case:
        await attach(case)
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        with sqlite3.connect(case.path) as db:
            encoded = db.execute("SELECT budget_permit_json FROM call_leases").fetchone()[0]
        permit = OwnedPilotBudgetPermit.model_validate_json(encoded)
        for changes in ({"generation": uuid4()}, {"ordinal": 2}, {"call_id": uuid4()}):
            assert not await case.writer.owned_budget_request(
                "dispatch", permit=permit.model_copy(update=changes)
            )
        assert budget_rows(case.path)[0][0] == 1


@pytest.mark.asyncio
async def test_same_policy_restart_and_midnight_do_not_reset_count(tmp_path):
    async with registry_case(tmp_path) as case:
        await attach(case)
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        await committed(case.registry, case.writer, event("call.hangup", occurred_at=NOW))
        assert budget_rows(case.path)[0][0] == 1
    tomorrow = NOW + timedelta(hours=15)
    fresh = replace(
        observation(),
        collection_started_at_utc=tomorrow,
        collection_finished_at_utc=tomorrow,
        openrouter_available_cash_usd=Decimal("100"),
        telnyx_available_cash_usd=Decimal("100"),
    )
    async with registry_case(tmp_path, utcnow=lambda: tomorrow) as case:
        gate = await attach(case, enrolled=False, fact=fresh, utcnow=lambda: tomorrow)
        assert gate.ready
        assert budget_rows(case.path)[0][0] == 1


@pytest.mark.asyncio
async def test_consumed_construction_grant_keeps_original_deadline_and_rechecks_freshness(tmp_path):
    from types import SimpleNamespace

    clock = [NOW]
    async with registry_case(tmp_path, utcnow=lambda: clock[0]) as case:
        await attach(case, utcnow=lambda: clock[0])
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        await committed(case.registry, case.writer, event("call.answered", occurred_at=NOW))
        snapshot = await case.registry.snapshot("original")
        clock[0] += timedelta(seconds=10)
        claim = await case.registry.claim_once(
            call_control_id="original",
            token_digest=snapshot.token_digest,
            abort_target_publisher=lambda _: True,
            abort_target_clearer=lambda _: None,
        )

        async def wait():
            return None

        task = asyncio.current_task()
        owner = SimpleNamespace(
            _task=task,
            _phase="constructing",
            _session=None,
            _terminal_capability=None,
            request_drain=lambda _: None,
            wait=wait,
        )
        grant = await case.registry.consume_claim_for_construction(claim, "stream", owner, task)
        assert grant is not None
        assert grant.original_deadline == NOW + timedelta(seconds=300)
        assert grant.started_at == NOW + timedelta(seconds=10)
        assert (
            await case.registry.consume_claim_for_construction(claim, "stream", owner, task) is None
        )
        clock[0] += timedelta(seconds=51)
        assert not await case.registry.construction_is_live(grant, owner, task)


@pytest.mark.asyncio
async def test_native_transaction_failure_rolls_back_budget_receipt_and_lease(tmp_path):
    from projetv0_voice.lifecycle import RuntimeSupervisor

    armed = False

    def fail(name):
        if armed and name == "after_mutation_before_commit":
            raise OSError("synthetic transaction failure")

    async with registry_case(tmp_path, failpoint=fail) as case:
        await attach(case)
        supervisor = RuntimeSupervisor(
            writer=case.writer, registry=case.registry, sparra_enabled=True, utcnow=lambda: NOW
        )
        observed = event(occurred_at=NOW)
        resolution = await case.registry.resolve_webhook(observed)
        armed = True
        assert (
            await supervisor.start_webhook_finalization(observed, resolution).wait()
        ).status_code == 503
        assert budget_rows(case.path) == [(0, "0", "0", None, 0)]
        with sqlite3.connect(case.path) as db:
            assert db.execute("SELECT count(*) FROM call_leases").fetchone() == (0,)
            assert db.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (0,)
        assert not case.provider.actions


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["openrouter_available_cash_usd", "telnyx_available_cash_usd"])
async def test_each_provider_funding_denial_has_zero_native_paid_entry(tmp_path, field):
    async with registry_case(tmp_path) as case:
        await attach(case, fact=observation(**{field: Decimal("0")}))
        result = await committed(case.registry, case.writer, event(occurred_at=NOW))
        assert result.status_code == 503
        assert not case.provider.actions and not case.begins
        assert budget_rows(case.path)[0][0] == 0


@pytest.mark.asyncio
async def test_cancelled_native_answer_keeps_consumption_and_unknown_owner(tmp_path):
    entered = asyncio.Event()
    async with registry_case(tmp_path) as case:
        await attach(case)

        async def pending(control, *, command_id):
            case.provider.actions.append(("answer", control, command_id))
            entered.set()
            await asyncio.Event().wait()

        case.provider.answer = pending
        observed = event(occurred_at=NOW)
        action = asyncio.create_task(committed(case.registry, case.writer, observed))
        await asyncio.wait_for(entered.wait(), 2)
        action.cancel()
        with pytest.raises(asyncio.CancelledError):
            await action
        assert budget_rows(case.path)[0][0] == 1
        assert budget_rows(case.path)[0][3] == "original"
        assert (
            await committed(case.registry, case.writer, observed, duplicate=True)
        ).status_code == 200
        assert len(case.provider.actions) == 1


@pytest.mark.asyncio
async def test_foreign_active_call_closes_readiness_and_new_admission(tmp_path):
    async with registry_case(tmp_path) as case:
        gate = await attach(case, fact=observation(active_call_control_ids=frozenset({"foreign"})))
        assert not gate.ready
        result = await committed(case.registry, case.writer, event(occurred_at=NOW))
        assert result.status_code == 503
        assert not case.provider.actions


@pytest.mark.asyncio
async def test_pending_native_begin_keeps_readiness_busy_despite_reporting_lag(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    async with registry_case(tmp_path) as case:
        gate = await attach(case)
        native_begin = case.registry._begin_call

        async def pending_begin(*args):
            entered.set()
            await release.wait()
            return await native_begin(*args)

        case.registry._begin_call = pending_begin
        action = asyncio.create_task(committed(case.registry, case.writer, event(occurred_at=NOW)))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert not await gate.refresh()
        finally:
            release.set()
            await action
