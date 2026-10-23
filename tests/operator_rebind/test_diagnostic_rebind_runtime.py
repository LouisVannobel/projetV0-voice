"""Complete native Supervisor characterization with offline V2 peers.

No strict production-profile/image/ASGI/host qualification is claimed here.
The accepted CAS implementation and its70 tests remain separate and unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import sqlite3
import sys
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import diagnostic_rebind as component
import pytest
from test_diagnostic_rebind import TARGET_IMAGE, TARGET_PROFILE, fresh_memory_guard, full_state

if __name__ == "__main__":
    print("fresh2GiB physical/page/virtual GiB:", fresh_memory_guard(), flush=True)

from projetv0_voice.admission import CallRegistry, SynchronousUnauthenticatedGate
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.lifecycle import RuntimeSupervisor
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate, OwnedPilotBudgetPolicy
from projetv0_voice.persistence.commands import canonical_operation_bytes
from projetv0_voice.persistence.relay import OutboxRelay, maintain_call_content
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.telnyx.recordings import after_recording_webhook_commit, purge_recordings_once
from tests.contract.test_postgres_sink import sink_with_rows
from tests.integration.test_fixed_v2_process import KEY, NOW, WORKSPACE, fixed_policy, registry_case
from tests.integration.test_owned_pilot_budget_admission import saved_permit
from tests.integration.test_owned_pilot_sixth_call import extend, extended
from tests.unit.test_owned_pilot_budget import budget_policy, observation
from tests.unit.test_sparra_admission import ControlledProvider, event

MATRIX = [(11, 5, 5), (13, 6, 6), (13, 6, 5)]
LEGACY_RUN = UUID(int=99)
FINGERPRINTED_RUN = UUID(int=100)
QUALIFICATION_FINGERPRINT = b"q" * 32


@pytest.fixture(autouse=True)
def fresh_resources_before_each_case():
    try:
        fresh_memory_guard()
    except AssertionError as error:
        pytest.exit(str(error), returncode=2)


def _fact(openrouter="1.10"):
    return observation(
        openrouter_reported_spend_usd=Decimal(openrouter),
        telnyx_reported_spend_usd=Decimal("0.20"),
    )


def _snapshot(path):
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        try:
            return full_state(db)
        finally:
            db.rollback()


def _known_ack_peer(sink, connection, monkeypatch):
    native_ingest = sink.ingest_v2

    async def ingest(operation):
        connection.rows = [
            (
                {
                    "schema_version": 2,
                    "status": "applied",
                    "operation_id": str(operation.operation_id),
                    "payload_sha256": hashlib.sha256(
                        canonical_operation_bytes(operation)
                    ).hexdigest(),
                },
            )
        ]
        try:
            return await native_ingest(operation)
        finally:
            connection.rows = []

    monkeypatch.setattr(sink, "ingest_v2", ingest)


async def _commit_history(case, observed, *, qualification=None):
    resolved = await case.registry.resolve_webhook(observed)
    effect = resolved.effect
    result = await case.writer.submit_webhook(
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
        **(qualification or {}),
    ).wait()
    disposition = await case.registry.reconcile_after_commit(observed, resolved, result)
    assert disposition.status_code == 200


async def _history_call(case, number, saved_events, *, qualification=None):
    control = f"native-history-{number}"
    for kind in ("call.initiated", "call.hangup"):
        observed = event(kind, occurred_at=NOW, call_control_id=control)
        await _commit_history(
            case, observed, qualification=qualification if kind == "call.initiated" else None
        )
        saved_events.append(observed)


async def _settle_outbox(case, monkeypatch):
    _known_ack_peer(case.sink, case.connection, monkeypatch)

    async def unexpected_degradation():
        pytest.fail("native history relay degraded")

    relay = OutboxRelay(
        case.writer,
        case.sink,
        utcnow=lambda: NOW,
        on_degraded=unexpected_degradation,
        drain=unexpected_degradation,
        contract_version=2,
    )
    result = await relay.run_once()
    assert result.status in {"empty", "delivered"}
    assert await case.writer.oldest_outbox_created_at() is None


async def _populated_history(tmp_path, monkeypatch, schema_version, total, used):
    """Initial disposable provisioning only; every row is produced by native APIs."""
    saved_events = []
    initial = budget_policy()
    async with registry_case(tmp_path) as case:
        gate = OwnedPilotBudgetGate(
            case.writer, initial, partial(_collect, _fact()), utcnow=lambda: NOW
        )
        case.registry.set_owned_budget(gate)
        assert await case.writer.owned_budget_request(
            "enroll", observation=_fact(), predecessor_retired=True
        )
        assert await gate.refresh()
        for number in range(5):
            qualification = (
                (
                    dict(qualification_run_id=LEGACY_RUN),
                    dict(
                        qualification_run_id=FINGERPRINTED_RUN,
                        qualification_profile_sha256=QUALIFICATION_FINGERPRINT,
                    ),
                )[number]
                if number < 2
                else None
            )
            await _history_call(case, number, saved_events, qualification=qualification)
        await _settle_outbox(case, monkeypatch)
    predecessor = initial
    if total == 6:
        predecessor = extended()
        async with registry_case(tmp_path) as case:
            gate = OwnedPilotBudgetGate(
                case.writer, predecessor, partial(_collect, _fact()), utcnow=lambda: NOW
            )
            case.registry.set_owned_budget(gate)
            assert await extend(case.writer, observation=_fact())
            assert await gate.refresh()
            if used == 6:
                await _history_call(case, 5, saved_events)
            await _settle_outbox(case, monkeypatch)
    path = tmp_path / "voice.sqlite"
    before = _snapshot(path)
    budget_row = before[3]["owned_pilot_budget"][0]
    assert before[0] == (schema_version,)
    assert budget_row[4:] == (used, "1.10", "0.20", 0, None)
    assert budget_row[2:4] == (predecessor.model_dump_json(), predecessor.sha256)
    assert len(before[3]["webhook_receipts"]) == len(saved_events)
    assert len(before[3]["call_leases"]) == used
    permits = tuple(saved_permit(path, f"native-history-{number}") for number in range(used))
    assert all(permit is not None for permit in permits)
    return SimpleNamespace(
        path=path, before=before, policy=predecessor, events=tuple(saved_events), permits=permits
    )


async def _collect(fact):
    return fact


def _transition(history):
    successor = OwnedPilotBudgetPolicy.model_validate(
        history.policy.model_dump()
        | dict(image_digest=TARGET_IMAGE, qualified_profile_sha256=TARGET_PROFILE)
    )
    with sqlite3.connect(history.path) as db:
        preimage = component.capture_preimage(db)
        trace = []
        db.set_trace_callback(trace.append)
        outcome = component.rebind(
            db, preimage, image_digest=TARGET_IMAGE, qualified_profile_sha256=TARGET_PROFILE
        )
        db.set_trace_callback(None)
    assert outcome.status == "committed"
    assert sum(sql.startswith("UPDATE owned_pilot_budget SET policy_json=") for sql in trace) == 1
    after = _snapshot(history.path)
    expected = (*history.before[:3], history.before[3].copy())
    row = history.before[3]["owned_pilot_budget"][0]
    expected[3]["owned_pilot_budget"] = [
        (row[0], row[1], successor.model_dump_json(), successor.sha256, *row[4:])
    ]
    assert after == expected
    assert successor.model_dump(exclude={"image_digest", "qualified_profile_sha256"}) == (
        history.policy.model_dump(exclude={"image_digest", "qualified_profile_sha256"})
    )
    return SimpleNamespace(policy=successor, postimage=after, outcome=outcome)


class _ControlPeer(ControlledProvider):
    closed = False

    async def aclose(self):
        self.closed = True


def _runtime(history, transition, monkeypatch, *, fact=None, fail_sink_open=False):
    """The real finite Supervisor graph; only external peers/instrumentation vary."""
    sink, pool, connection, _factory = sink_with_rows([])
    _known_ack_peer(sink, connection, monkeypatch)
    native_open, native_close = sink.open, sink.close
    sink_events = []

    async def open_sink():
        await native_open()
        sink_events.append("opened")
        if fail_sink_open:
            raise RuntimeError("synthetic-sink-open-failure")

    async def close_sink():
        await native_close()
        sink_events.append("closed")

    monkeypatch.setattr(sink, "open", open_sink)
    monkeypatch.setattr(sink, "close", close_sink)
    writer = PersistenceWriter(
        history.path,
        CryptoKeyring({1: KEY}, active_version=1),
        contract_version=2,
        utcnow=lambda: NOW,
        process_agent_id="fixture",
        process_deployment_id="fixture",
    )
    observed, requests = asyncio.Event(), []
    native_request = writer.owned_budget_request

    async def observe_native_completion(action, **values):
        result = await native_request(action, **values)
        requests.append(action)
        if action == "observe":
            observed.set()
        return result

    monkeypatch.setattr(writer, "owned_budget_request", observe_native_completion)
    budget = OwnedPilotBudgetGate(
        writer, transition.policy, partial(_collect, fact or _fact()), utcnow=lambda: NOW
    )
    control = _ControlPeer()
    metrics = RuntimeMetrics.in_memory()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=control,
        metrics=metrics,
        owned_budget=budget,
        deployment_id="fixture",
        retention_days=30,
        sparra_enabled=True,
        utcnow=lambda: NOW,
        startup_phase_timeout_seconds=2.0,
        shutdown_timeout_seconds=3.0,
    )

    async def no_live_call(*_args):
        pytest.fail("no-work fixture attempted call content cleanup or admission")

    registry = CallRegistry(
        writer=writer,
        call_control=supervisor.call_control_facade,
        tenant_id=str(WORKSPACE),
        agent_id="fixture",
        deployment_id="fixture",
        capacity=1,
        lease_ttl_seconds=30,
        original_call_limit_seconds=300,
        stream_url="wss://fixture.invalid/media",
        retention_days=30,
        utcnow=lambda: NOW,
        monotonic=lambda: 100.0,
        sparra=fixed_policy(),
        called_did=transition.policy.identity.did,
        begin_call=no_live_call,
    )
    registry.set_owned_budget(budget)
    prepared = asyncio.Event()

    async def prepare_native_content():
        await maintain_call_content(
            writer, sink, no_live_call, utcnow=lambda: NOW, timeout_seconds=2
        )
        prepared.set()

    relay = OutboxRelay(
        writer,
        sink,
        utcnow=lambda: NOW,
        contract_version=2,
        before_fifo=prepare_native_content,
        on_degraded=supervisor.begin_drain,
        drain=supervisor.begin_drain,
        stop_erased_call=no_live_call,
    )
    supervisor.bind_runtime_graph(
        relay=relay,
        registry=registry,
        sink=sink,
        unauthenticated_gate=SynchronousUnauthenticatedGate(1),
        purge_once=partial(
            purge_recordings_once,
            worker_id="rebind-fixture",
            lease_seconds=30,
            batch_size=100,
            telnyx=supervisor.call_control_facade,
            sink=sink,
            utcnow=lambda: NOW,
        ),
        recording_after_commit=partial(
            after_recording_webhook_commit,
            telnyx=supervisor.call_control_facade,
            writer=writer,
            local_drain=no_live_call,
            monotonic=asyncio.get_running_loop().time,
            timeout_seconds=2,
            utcnow=lambda: NOW,
        ),
    )
    return SimpleNamespace(
        supervisor=supervisor,
        writer=writer,
        registry=registry,
        budget=budget,
        control=control,
        metrics=metrics,
        sink=sink,
        pool=pool,
        sink_events=sink_events,
        requests=requests,
        observed=observed,
        prepared=prepared,
    )


async def _start(runtime, *, ready):
    await runtime.supervisor.startup()
    await asyncio.wait_for(runtime.observed.wait(), timeout=2)
    assert runtime.prepared.is_set()
    assert runtime.budget.can_start() is ready
    assert runtime.supervisor.readiness_snapshot().ready is ready
    assert set(runtime.supervisor._fixed_tasks) == {
        "relay",
        "purge",
        "reaper",
        "qualification-expiry",
        "writer-fatal",
        "readiness",
        "event-loop-lag",
    }
    assert all(not task.done() for task in runtime.supervisor._fixed_tasks.values())
    assert runtime.control.actions == []


async def _close(runtime):
    await runtime.supervisor.aclose()
    assert runtime.supervisor._writer_task.done()
    assert all(task.done() for task in runtime.supervisor._fixed_tasks.values())
    assert runtime.control.closed and runtime.sink_events[-1] == "closed"
    assert runtime.pool.active == 0
    assert runtime.control.actions == []
    assert not set(runtime.requests) & {"enroll", "extend-sixth-call", "suspend"}


@pytest.mark.parametrize("schema_version,total,used", MATRIX)
def test_runtime_supervisor_v2_rebind_preserves_history(
    tmp_path, monkeypatch, schema_version, total, used
):
    async def characterize():
        history = await _populated_history(tmp_path, monkeypatch, schema_version, total, used)
        transition = _transition(history)
        old_binding = _runtime(history, SimpleNamespace(policy=history.policy), monkeypatch)
        try:
            await _start(old_binding, ready=False)
            assert not await old_binding.budget.refresh()
            assert await old_binding.writer.owned_budget_request("observe", observation=_fact()) is None
            attempted = event(
                occurred_at=NOW, call_control_id="refused-old-binding-admission"
            )
            resolution = await old_binding.registry.resolve_webhook(attempted)
            refusal = await old_binding.supervisor.start_webhook_finalization(
                attempted, resolution
            ).wait()
            assert refusal.status_code == 503 and refusal.admission_rejection == "qualification"
            assert await old_binding.supervisor.classify_webhook_receipt(attempted) == "missing"
            assert _snapshot(history.path) == transition.postimage
        finally:
            await _close(old_binding)
        assert _snapshot(history.path) == transition.postimage
        for _ in range(2):
            runtime = _runtime(history, transition, monkeypatch)
            try:
                await _start(runtime, ready=used < total)
                assert _snapshot(history.path) == transition.postimage
                for permit in history.permits:
                    assert not runtime.budget.permits(permit)
                    assert not await runtime.writer.owned_budget_request("dispatch", permit=permit)
                for observed in history.events:
                    assert (
                        await runtime.supervisor.classify_webhook_receipt(observed) == "duplicate"
                    )
                    resolution = await runtime.registry.resolve_duplicate_webhook(observed)
                    disposition = await runtime.supervisor.start_webhook_finalization(
                        observed, resolution
                    ).wait()
                    assert disposition.status_code == 200
                assert await runtime.writer.qualification_run_consumed(LEGACY_RUN)
                assert await runtime.writer.qualification_run_consumed(
                    FINGERPRINTED_RUN, profile_sha256=QUALIFICATION_FINGERPRINT
                )
                assert _snapshot(history.path) == transition.postimage
            finally:
                await _close(runtime)
            assert _snapshot(history.path) == transition.postimage

    asyncio.run(characterize())


@pytest.mark.parametrize("schema_version,total,used", MATRIX)
def test_committed_rebind_startup_failure_recovers_same_successor(
    tmp_path, monkeypatch, schema_version, total, used
):
    async def characterize():
        history = await _populated_history(tmp_path, monkeypatch, schema_version, total, used)
        transition = _transition(history)
        failed = _runtime(history, transition, monkeypatch, fail_sink_open=True)
        with pytest.raises(RuntimeError, match="^runtime_startup_failed$"):
            await failed.supervisor.startup()
        await _close(failed)
        assert failed.sink_events == ["opened", "closed"]
        assert not failed.observed.is_set() and not failed.prepared.is_set()
        assert _snapshot(history.path) == transition.postimage
        with sqlite3.connect(f"file:{history.path.as_posix()}?mode=ro", uri=True) as db:
            assert component.classify(db, transition.outcome) == "exact_new"
        recovered = _runtime(history, transition, monkeypatch)
        try:
            await _start(recovered, ready=used < total)
            assert recovered.budget.policy.sha256 == transition.policy.sha256
            assert _snapshot(history.path) == transition.postimage
        finally:
            await _close(recovered)
        assert _snapshot(history.path) == transition.postimage

    asyncio.run(characterize())


def test_runtime_supervisor_preserves_authentic_floor_increase(tmp_path, monkeypatch):
    async def characterize():
        history = await _populated_history(tmp_path, monkeypatch, 13, 6, 5)
        transition = _transition(history)
        expected = (*transition.postimage[:3], transition.postimage[3].copy())
        row = list(expected[3]["owned_pilot_budget"][0])
        row[5] = "1.20"
        expected[3]["owned_pilot_budget"] = [tuple(row)]
        for fact in (_fact("1.20"), _fact("1.00")):
            runtime = _runtime(history, transition, monkeypatch, fact=fact)
            try:
                await _start(runtime, ready=True)
                assert _snapshot(history.path) == expected
            finally:
                await _close(runtime)
            assert _snapshot(history.path) == expected

    asyncio.run(characterize())


if __name__ == "__main__":
    checkout = Path(__file__).resolve().parents[2]
    for native_type, relative in (
        (RuntimeSupervisor, "src/projetv0_voice/lifecycle.py"),
        (PersistenceWriter, "src/projetv0_voice/persistence/writer.py"),
    ):
        origin = Path(inspect.getfile(native_type)).resolve()
        assert origin == (checkout / relative).resolve(), "unexpected_native_source_origin"
        print(
            "native-source:",
            str(origin),
            hashlib.sha256(origin.read_bytes()).hexdigest(),
            flush=True,
        )
    raise SystemExit(
        pytest.main([__file__, "-q", "-p", "no:cacheprovider", "--tb=short", *sys.argv[1:]])
    )
