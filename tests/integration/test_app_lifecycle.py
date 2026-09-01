from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Coroutine
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from projetv0_voice.admission import CallRegistry
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.persistence.relay import OutboxRelay
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.telnyx.call_control import CallControlResult
from projetv0_voice.telnyx.webhooks import ResolvedWebhook, VerifiedWebhook

NOW = datetime(2026, 9, 1, 10, tzinfo=UTC)
KEY = bytes(range(32))


def _close(coroutine: Coroutine[Any, Any, None]) -> None:
    coroutine.close()


@pytest.mark.asyncio
async def test_owned_task_set_rejects_and_closes_unstarted_child() -> None:
    from projetv0_voice.lifecycle import _OwnedTaskSet

    owner = _OwnedTaskSet()
    owner.close_registration()

    async def child() -> None:
        raise AssertionError("rejected child ran")

    coroutine = child()
    assert owner.try_start(coroutine, name="rejected") is None
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_owned_task_set_handles_eager_synchronous_finish_and_failure() -> None:
    from projetv0_voice.lifecycle import _OwnedTaskSet

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    owner = _OwnedTaskSet()
    finished = asyncio.Event()

    async def finish() -> None:
        finished.set()

    async def fail() -> None:
        raise RuntimeError("secret-child-error")

    try:
        assert owner.try_start(finish(), name="finish") is not None
        assert owner.try_start(fail(), name="fail") is not None
        owner.close_registration()
        with pytest.raises(RuntimeError, match="^owned_task_failed$") as captured:
            await owner.join_until_empty(loop.time() + 1.0)
    finally:
        loop.set_task_factory(previous_factory)

    assert finished.is_set()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_owned_task_set_close_race_either_owns_or_closes_child_once() -> None:
    from projetv0_voice.lifecycle import _OwnedTaskSet

    owner = _OwnedTaskSet()
    barrier = threading.Barrier(2)
    ran = 0

    async def child() -> None:
        nonlocal ran
        ran += 1

    coroutine = child()

    def close_registration() -> None:
        barrier.wait()
        owner.close_registration()

    closer = threading.Thread(target=close_registration)
    closer.start()
    barrier.wait()
    task = owner.try_start(coroutine, name="racing-child")
    await asyncio.to_thread(closer.join)
    owner.close_registration()
    await owner.join_until_empty(asyncio.get_running_loop().time() + 1.0)

    assert ran == int(task is not None)
    assert task is not None or inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_owned_task_set_joins_late_child_to_a_true_fixed_point() -> None:
    from projetv0_voice.lifecycle import _OwnedTaskSet

    owner = _OwnedTaskSet()
    parent_started = asyncio.Event()
    allow_late_child = asyncio.Event()
    child_finished = asyncio.Event()

    async def child() -> None:
        child_finished.set()

    async def parent() -> None:
        parent_started.set()
        await allow_late_child.wait()
        assert owner.try_start(child(), name="late-child") is not None

    assert owner.try_start(parent(), name="parent") is not None
    await parent_started.wait()
    joiner = asyncio.create_task(
        owner.join_until_empty(asyncio.get_running_loop().time() + 1.0)
    )
    allow_late_child.set()
    await joiner

    assert child_finished.is_set()


@pytest.mark.asyncio
async def test_runtime_supervisor_constructs_exact_three_owned_task_sets() -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor, _OwnedTaskSet

    supervisor = RuntimeSupervisor()

    owned = tuple(
        value
        for value in vars(supervisor).values()
        if isinstance(value, _OwnedTaskSet)
    )
    assert owned == (
        supervisor.webhook_finalizers,
        supervisor.call_lifecycle_owners,
        supervisor.fixed_supervisors,
    )
    assert len({id(value) for value in owned}) == 3
    assert callable(supervisor.call_lifecycle_owners.try_start)

    await supervisor.aclose()


async def _seed_stale_lease(database: Path) -> None:
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    await writer.commit_lease(
        call_control_id="control-stale",
        call_id=UUID("11111111-1111-4111-8111-111111111111"),
        tenant_id="tenant-a",
        agent_id="agent-a",
        state="pending",
        token_hash=b"d" * 32,
        created_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=1),
        closed_at=None,
    )
    await writer.drain(timeout_seconds=2.0)
    await task


async def _seed_old_outbox(database: Path) -> None:
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW - timedelta(seconds=901),
    )
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    occurred_at = NOW - timedelta(seconds=901)
    await writer.commit_control(
        PersistenceCommand(
            "outbox",
            {
                "operation": VoiceOperationV1(
                    schema_version=1,
                    operation_id=UUID("77777777-7777-4777-8777-777777777777"),
                    deployment_id="deployment-a",
                    call_id=UUID("88888888-8888-4888-8888-888888888888"),
                    occurred_at=occurred_at,
                    kind="call.upsert",
                    payload=CallUpsertPayloadV1(
                        telnyx_call_control_id="old-control",
                        telnyx_call_leg_id=None,
                        telnyx_call_session_id=None,
                        status="failed",
                        disclosure_state="failed",
                        started_at=None,
                        ended_at=occurred_at,
                        end_reason="test_old_outbox",
                        retention_until=occurred_at + timedelta(days=7),
                    ),
                )
            },
            None,
        )
    )
    await writer.drain(timeout_seconds=2.0)
    await task


@pytest.mark.asyncio
async def test_startup_recovers_stale_lease_hangup_before_atomic_terminal(
    tmp_path: Path,
) -> None:
    import sqlite3

    from projetv0_voice.lifecycle import RuntimeSupervisor

    database = tmp_path / "voice.sqlite"
    await _seed_stale_lease(database)
    order: list[str] = []

    class Control:
        async def hangup(
            self, call_control_id: str, *, command_id: UUID, client_state: object = None
        ) -> CallControlResult:
            del client_state
            with sqlite3.connect(database) as connection:
                state = connection.execute(
                    "SELECT state FROM call_leases WHERE call_control_id = ?",
                    (call_control_id,),
                ).fetchone()[0]
            order.append(f"hangup:{state}:{command_id}")
            return CallControlResult("accepted")

        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            raise AssertionError("stale recovery must not answer")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            raise AssertionError("stale recovery must not stream")

        async def aclose(self) -> None:
            order.append("control-closed")

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    metrics = RuntimeMetrics.in_memory()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=Control(),
        metrics=metrics,
        utcnow=lambda: NOW,
        retention_days=7,
        loop_interval_seconds=0.05,
    )

    await supervisor.startup()
    with sqlite3.connect(database) as connection:
        state = connection.execute(
            "SELECT state FROM call_leases WHERE call_control_id = 'control-stale'"
        ).fetchone()[0]
        outbox = connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    await supervisor.aclose()

    assert order[0].startswith("hangup:pending:")
    assert state == "terminal"
    assert outbox == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["rejected", "rate_limited", "retryable_not_sent", "outcome_unknown"],
)
async def test_startup_leaves_stale_lease_reopenable_after_nonaccepted_hangup(
    tmp_path: Path,
    outcome: str,
) -> None:
    import sqlite3

    from projetv0_voice.lifecycle import RuntimeSupervisor

    database = tmp_path / f"{outcome}.sqlite"
    await _seed_stale_lease(database)
    command_ids: list[UUID] = []

    class Control:
        async def hangup(
            self, _call_control_id: str, *, command_id: UUID, client_state: object = None
        ) -> CallControlResult:
            del client_state
            command_ids.append(command_id)
            return CallControlResult(outcome)  # type: ignore[arg-type]

        async def aclose(self) -> None:
            return None

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=Control(),
        metrics=RuntimeMetrics.in_memory(),
        utcnow=lambda: NOW,
        retention_days=7,
        loop_interval_seconds=0.05,
    )

    with pytest.raises(RuntimeError, match="^stale_recovery_failed$"):
        await supervisor.startup()
    with sqlite3.connect(database) as connection:
        state = connection.execute(
            "SELECT state FROM call_leases WHERE call_control_id = 'control-stale'"
        ).fetchone()[0]

    assert state == "pending"
    assert len(command_ids) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "cancellation"])
async def test_startup_leaves_stale_lease_after_hangup_failure(
    tmp_path: Path,
    failure: str,
) -> None:
    import sqlite3

    from projetv0_voice.lifecycle import RuntimeSupervisor

    database = tmp_path / f"{failure}.sqlite"
    await _seed_stale_lease(database)

    class Control:
        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            if failure == "cancellation":
                raise asyncio.CancelledError
            raise RuntimeError("provider-secret")

        async def aclose(self) -> None:
            return None

    supervisor = RuntimeSupervisor(
        writer=PersistenceWriter(
            database,
            CryptoKeyring({1: KEY}, active_version=1),
            utcnow=lambda: NOW,
        ),
        call_control=Control(),  # type: ignore[arg-type]
        metrics=RuntimeMetrics.in_memory(),
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )

    expected = asyncio.CancelledError if failure == "cancellation" else RuntimeError
    with pytest.raises(expected):
        await supervisor.startup()
    with sqlite3.connect(database) as connection:
        state = connection.execute(
            "SELECT state FROM call_leases WHERE call_control_id = 'control-stale'"
        ).fetchone()[0]
    assert state == "pending"


@pytest.mark.asyncio
async def test_startup_accepted_before_terminal_commit_rolls_back_and_reopens(
    tmp_path: Path,
) -> None:
    import sqlite3

    from projetv0_voice.lifecycle import RuntimeSupervisor

    database = tmp_path / "accepted-before-commit.sqlite"
    await _seed_stale_lease(database)

    class Control:
        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            return None

    async def fail_before_commit(name: str) -> None:
        if name == "after_mutation_before_commit":
            raise RuntimeError("local-failpoint")

    supervisor = RuntimeSupervisor(
        writer=PersistenceWriter(
            database,
            CryptoKeyring({1: KEY}, active_version=1),
            utcnow=lambda: NOW,
            failpoint=fail_before_commit,
        ),
        call_control=Control(),  # type: ignore[arg-type]
        metrics=RuntimeMetrics.in_memory(),
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
        shutdown_timeout_seconds=5.0,
    )

    with pytest.raises(RuntimeError, match="^stale_recovery_failed$"):
        await supervisor.startup()
    with sqlite3.connect(database) as connection:
        state = connection.execute(
            "SELECT state FROM call_leases WHERE call_control_id = 'control-stale'"
        ).fetchone()[0]
        outbox = connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    assert (state, outbox) == ("pending", 0)


@pytest.mark.asyncio
async def test_runtime_supervisor_is_real_webhook_finalizer_owner_and_forwards_alias(
    tmp_path: Path,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    database = tmp_path / "voice.sqlite"
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        metrics=RuntimeMetrics.in_memory(),
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )
    event = VerifiedWebhook(
        event_id="event-unsupported",
        event_type="future.event",
        occurred_at=NOW,
        call_control_id=None,
        call_leg_id=None,
        call_session_id=None,
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"c" * 32,
        legacy_v1_semantic_fingerprint_sha256=b"l" * 32,
    )

    legacy = replace(
        event,
        semantic_fingerprint_sha256=b"l" * 32,
        legacy_v1_semantic_fingerprint_sha256=None,
    )
    await supervisor.startup()
    assert await supervisor.classify_webhook_receipt(legacy) == "missing"
    handle = supervisor.start_webhook_finalization(legacy, ResolvedWebhook(None))
    disposition = await handle.wait()
    assert disposition.status_code == 200
    assert await supervisor.classify_webhook_receipt(event) == "duplicate"
    await supervisor.aclose()


def test_relay_exposes_exact_claim_lease_for_ambiguity_supervision(
    tmp_path: Path,
) -> None:
    class Sink:
        async def ingest(self, _operation: object) -> None:
            return None

    async def no_op() -> None:
        return None

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
    )
    relay = OutboxRelay(
        writer,
        Sink(),  # type: ignore[arg-type]
        claim_lease_seconds=17,
        on_degraded=no_op,
        drain=no_op,
    )

    assert relay.claim_lease_seconds == 17


@pytest.mark.asyncio
async def test_shutdown_joins_late_finalizer_lifecycle_child_before_writer(
    tmp_path: Path,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        metrics=RuntimeMetrics.in_memory(),
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )
    finalizer_started = asyncio.Event()
    allow_descendant = asyncio.Event()
    lifecycle_finished = asyncio.Event()

    async def lifecycle_child() -> None:
        await writer.runtime_observation()
        lifecycle_finished.set()

    async def finalizer() -> None:
        finalizer_started.set()
        await allow_descendant.wait()
        assert (
            supervisor.call_lifecycle_owners.try_start(
                lifecycle_child(), name="late-lifecycle-child"
            )
            is not None
        )

    await supervisor.startup()
    assert (
        supervisor.webhook_finalizers.try_start(finalizer(), name="held-finalizer")
        is not None
    )
    await finalizer_started.wait()
    await supervisor.begin_drain()
    closing = asyncio.create_task(supervisor.aclose())
    allow_descendant.set()
    await closing

    assert lifecycle_finished.is_set()
    assert supervisor._writer_task is not None  # noqa: SLF001
    assert supervisor._writer_task.done()  # noqa: SLF001


@pytest.mark.asyncio
async def test_shutdown_deadline_refuses_to_cross_live_finalizer_dependency() -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    closed = False

    class Control:
        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    supervisor = RuntimeSupervisor(
        call_control=Control(),  # type: ignore[arg-type]
        shutdown_timeout_seconds=0.05,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def stuck_finalizer() -> None:
        started.set()
        await release.wait()

    assert (
        supervisor.webhook_finalizers.try_start(
            stuck_finalizer(), name="stuck-finalizer"
        )
        is not None
    )
    await started.wait()
    with pytest.raises(RuntimeError, match="^owned_task_deadline_exceeded$"):
        await supervisor.aclose()

    assert closed is False
    release.set()
    await supervisor.webhook_finalizers.join_until_empty(
        asyncio.get_running_loop().time() + 1.0
    )


@pytest.mark.asyncio
async def test_registry_internal_failure_event_fail_closes_runtime(
    tmp_path: Path,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    class Control:
        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            return None

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    control = Control()
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/media",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=lambda: 10.0,
        token_factory=lambda _size: "A" * 43,
        prefix_factory=lambda: 1,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=control,
        metrics=RuntimeMetrics.in_memory(),
        registry=registry,
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )

    await supervisor.startup()
    assert supervisor.readiness_snapshot().admission_open is True
    registry._note_terminal_failure("terminal_persistence_failed")  # noqa: SLF001
    await asyncio.wait_for(supervisor._drain_event.wait(), timeout=1.0)  # noqa: SLF001

    assert supervisor.readiness_snapshot().draining is True
    assert supervisor.readiness_snapshot().ready is False
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_shutdown_does_not_close_dependencies_while_real_writer_is_stuck(
    tmp_path: Path,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    blocking = False
    entered = asyncio.Event()
    release = asyncio.Event()
    control_closed = False

    async def failpoint(name: str) -> None:
        if blocking and name == "after_mutation_before_commit":
            entered.set()
            await release.wait()

    class Control:
        async def aclose(self) -> None:
            nonlocal control_closed
            control_closed = True

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
        failpoint=failpoint,
    )
    metrics = RuntimeMetrics.in_memory()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=Control(),  # type: ignore[arg-type]
        metrics=metrics,
        utcnow=lambda: NOW,
        loop_interval_seconds=1.0,
        startup_phase_timeout_seconds=2.0,
        shutdown_timeout_seconds=0.01,
    )
    await supervisor.startup()
    blocking = True
    blocked_observation = asyncio.create_task(writer.runtime_observation())
    await entered.wait()

    with pytest.raises(RuntimeError, match="^runtime_shutdown_deadline_exceeded$"):
        await supervisor.aclose()
    assert control_closed is False

    release.set()
    await blocked_observation
    assert supervisor._writer_task is not None  # noqa: SLF001
    await asyncio.wait_for(supervisor._writer_task, timeout=2.0)  # noqa: SLF001
    await metrics.aclose()


@pytest.mark.asyncio
async def test_outbox_age_breach_closes_admission_during_startup(
    tmp_path: Path,
) -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.lifecycle import RuntimeSupervisor

    class Control:
        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            return None

    database = tmp_path / "old-outbox.sqlite"
    await _seed_old_outbox(database)
    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    control = Control()
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/media",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=lambda: 10.0,
        token_factory=lambda _size: "A" * 43,
        prefix_factory=lambda: 2,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=control,
        metrics=RuntimeMetrics.in_memory(),
        registry=registry,
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )

    await supervisor.startup()
    snapshot = supervisor.readiness_snapshot()
    with pytest.raises(CallAdmissionRejected, match="^call_draining$"):
        await registry.resolve_webhook(
            VerifiedWebhook(
                event_id="new-event",
                event_type="call.initiated",
                occurred_at=NOW,
                call_control_id="new-control",
                call_leg_id="leg-a",
                call_session_id="session-a",
                recording_id=None,
                stream_id=None,
                client_state=None,
                recording_started_at=None,
                recording_ended_at=None,
                recording_channels=None,
                semantic_fingerprint_sha256=b"n" * 32,
                direction="incoming",
                call_state="parked",
            )
        )

    assert snapshot.ready is False
    assert snapshot.admission_open is False
    assert snapshot.draining is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_startup_unwind_joins_retained_registry_background_before_dependencies(
    tmp_path: Path,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    release = asyncio.Event()
    child_started = asyncio.Event()
    control_closed = asyncio.Event()

    async def fail_initial_publication(name: str) -> None:
        if name == "after_mutation_before_commit":
            raise RuntimeError("startup-publication-failure")

    class Control:
        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            control_closed.set()

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
        failpoint=fail_initial_publication,
    )
    control = Control()
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/media",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=lambda: 10.0,
        token_factory=lambda _size: "A" * 43,
        prefix_factory=lambda: 3,
    )

    async def retained_child() -> None:
        child_started.set()
        await release.wait()

    assert (
        registry._background_owner.start(  # noqa: SLF001
            retained_child(), name="held-registry-child"
        )
        is not None
    )
    await child_started.wait()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=control,
        metrics=RuntimeMetrics.in_memory(),
        registry=registry,
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
        shutdown_timeout_seconds=1.0,
    )
    startup = asyncio.create_task(supervisor.startup())
    await writer.fatal_event.wait()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(startup), timeout=0.1)
    assert control_closed.is_set() is False

    release.set()
    with pytest.raises(RuntimeError, match="^runtime_startup_failed$"):
        await startup
    assert control_closed.is_set()


@pytest.mark.asyncio
async def test_preclosed_fixed_inventory_is_all_or_clean(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    class BrokenControl:
        async def aclose(self) -> None:
            raise RuntimeError("external-close-failed")

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    metrics = RuntimeMetrics.in_memory()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=BrokenControl(),  # type: ignore[arg-type]
        metrics=metrics,
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )
    supervisor.fixed_supervisors.close_registration()

    with pytest.raises(RuntimeError, match="^owned_task_registration_failed$"):
        await supervisor.startup()

    assert supervisor._writer_task is not None  # noqa: SLF001
    assert supervisor._writer_task.done()  # noqa: SLF001
    assert not [
        warning
        for warning in recwarn
        if "was never awaited" in str(warning.message)
    ]
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name().startswith("voice-")
        and not task.done()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_dependency", ["call_control", "sink", "metrics"])
async def test_shutdown_deadline_bounds_each_dependency_close(
    blocked_dependency: str,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    entered = asyncio.Event()
    release = asyncio.Event()
    closed: list[str] = []

    async def close(name: str) -> None:
        if blocked_dependency == name:
            entered.set()
            await release.wait()
        closed.append(name)

    class Control:
        async def aclose(self) -> None:
            await close("call_control")

    class Sink:
        async def open(self) -> None:
            return None

        async def close(self) -> None:
            await close("sink")

    metrics = RuntimeMetrics.in_memory()
    original_metrics_close = metrics.aclose

    async def close_metrics() -> None:
        await close("metrics")

    metrics.aclose = close_metrics  # type: ignore[method-assign]
    supervisor = RuntimeSupervisor(
        call_control=Control(),  # type: ignore[arg-type]
        sink=Sink(),
        metrics=metrics,
        shutdown_timeout_seconds=0.05,
    )

    with pytest.raises(RuntimeError, match="^runtime_shutdown_deadline_exceeded$"):
        await asyncio.wait_for(supervisor.aclose(), timeout=0.5)
    assert entered.is_set()
    if blocked_dependency == "call_control":
        assert closed == []
    elif blocked_dependency == "sink":
        assert closed == ["call_control"]
    else:
        assert closed == ["call_control", "sink"]

    release.set()
    metrics.aclose = original_metrics_close  # type: ignore[method-assign]
    await original_metrics_close()


@pytest.mark.asyncio
async def test_production_composition_builds_ordered_graph_with_one_measured_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64
    import hashlib

    from nacl.signing import SigningKey
    from pydantic import SecretStr

    from projetv0_voice.config import AgentManifestV1
    from projetv0_voice.lifecycle import (
        RuntimeInferenceFactories,
        RuntimeProductionFactories,
        RuntimeProfileSelection,
        build_production_runtime,
    )
    from projetv0_voice.qualified_profile import (
        InferenceProfileV1,
        QualifiedDeploymentProfileV1,
        canonical_inference_profile_sha256,
    )
    from projetv0_voice.runtime_config import (
        capture_runtime_environment,
        parse_runtime_settings,
    )
    from projetv0_voice.telnyx.handshake import (
        AuthenticatedTelnyxHandshakeService,
    )
    from projetv0_voice.telnyx.webhooks import TelnyxWebhookProcessor

    api_key = "telnyx-api-key-value"
    public_key = base64.b64encode(
        bytes(SigningKey.generate().verify_key)
    ).decode("ascii")
    inference = InferenceProfileV1.model_validate(
        {
            "schema_version": 1,
            "stt_model": "test/stt",
            "llm_model": "test/llm",
            "tts_model": "test/tts",
            "tts_voice": "fr-test",
            "tts_pcm_sample_rate": 24000,
            "tts_pcm_channels": 1,
            "llm_provider_policy": {"allow_fallbacks": True, "sort": "latency"},
            "tts_provider_options": {},
        }
    )
    inference_hash = canonical_inference_profile_sha256(inference)
    settings = parse_runtime_settings(
        capture_runtime_environment(
            {
                "VOICE_RUNTIME_MODE": "strict",
                "VOICE_DEPLOYMENT_ID": "voice-agent-a",
                "VOICE_RUNTIME_CONTRACT_PATH": "/srv/projetv0/runtime-contract.json",
                "VOICE_AGENT_BUNDLE_PATH": "/srv/projetv0/agent-bundle",
                "VOICE_QUALIFIED_PROFILE_PATH": "/srv/projetv0/qualified.json",
                "VOICE_KEYRING_PATH": "/srv/projetv0/keyring.json",
                "VOICE_SQLITE_PATH": "/var/lib/projetv0/voice.sqlite3",
                "VOICE_RUNTIME_CONTRACT_SHA256": "a" * 64,
                "VOICE_IMAGE_DIGEST": (
                    f"ghcr.io/louisvannobel/projetv0-voice@sha256:{'d' * 64}"
                ),
                "VOICE_AGENT_BUNDLE_SHA256": "b" * 64,
                "VOICE_INFERENCE_PROFILE_SHA256": inference_hash,
                "VOICE_DEPLOYMENT_MAX_CALLS": "10",
                "VOICE_HANDSHAKE_TIMEOUT_SECONDS": "5",
                "VOICE_CALL_IDLE_TIMEOUT_SECONDS": "300",
                "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS": "10",
                "VOICE_PRE_DRAIN_GRACE_SECONDS": "15",
                "VOICE_UVICORN_GRACE_SECONDS": "20",
                "VOICE_SHUTDOWN_GRACE_SECONDS": "30",
                "VOICE_TELNYX_API_KEY_FILE": "/run/secrets/telnyx-api-key",
                "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE": "/run/secrets/telnyx-webhook-key",
                "VOICE_OPENROUTER_API_KEY_FILE": "/run/secrets/openrouter-api-key",
                "VOICE_POSTGRES_DSN_FILE": "/run/secrets/postgres-dsn",
                "VOICE_TELNYX_MEDIA_WSS_URL": "wss://voice.invalid/telnyx/media",
                "VOICE_OTLP_HTTP_ENDPOINT": "https://collector.invalid/v1/metrics",
                "VOICE_BIND_HOST": "127.0.0.1",
                "VOICE_BIND_PORT": "8080",
            }
        ),
        geteuid=lambda: 10001,
        getegid=lambda: 10001,
    )
    manifest = AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "tenant_id": "tenant-a",
            "agent_id": "agent-a",
            "revision": "r1",
            "dids": ["+33123456789"],
            "language": "fr",
            "prompt_path": tmp_path / "prompt.md",
            "prompt_revision": "p1",
            "greeting": "Bonjour",
            "conversation_mode": "freeform",
            "max_concurrent_calls": 10,
            "direction": "inbound_only",
            "transport_codec": "PCMU",
            "transport_sample_rate_hz": 8000,
            "transcript_retention_days": 7,
            "recording_mode": "telnyx_dual",
            "recording_format": "wav",
            "recording_retention_days": 30,
            "recording_required": False,
            "recording_play_beep": False,
        }
    )
    profile = QualifiedDeploymentProfileV1(
        schema_version=1,
        deployment_id=settings.deployment_id,
        runtime_contract_sha256=settings.runtime_contract_sha256,
        image_digest=settings.image_digest,
        agent_bundle_sha256=settings.agent_bundle_sha256,
        inference_profile_sha256=settings.inference_profile_sha256,
        inference=inference,
        token_locator_id="telnyx-header-connected-v1",
        telnyx_api_key_sha256=hashlib.sha256(api_key.encode()).hexdigest(),
        telnyx_data_locality="EU",
        telnyx_handshake_fixture_sha256="c" * 64,
        disclosure_mark_timeout_ms=5000,
        call_lease_ttl_seconds=30,
        qualified_at=NOW - timedelta(days=1),
    )
    order: list[str] = []

    class Sink:
        async def open(self) -> None:
            order.append("sink-open")

        async def close(self) -> None:
            order.append("sink-close")

        async def ingest(self, _operation: object) -> None:
            return None

        async def lease_recording_purges(
            self, _worker_id: str, _lease_seconds: int, _batch_size: int
        ) -> tuple[()]:
            return ()

    class Control:
        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            order.append("call-control-close")

    secrets = {
        str(settings.telnyx_api_key_file): SecretStr(api_key),
        str(settings.telnyx_webhook_public_key_file): SecretStr(public_key),
        str(settings.openrouter_api_key_file): SecretStr("openrouter-key"),
        str(settings.postgres_dsn_file): SecretStr("postgres-dsn"),
    }

    def read_secret(path: object) -> SecretStr:
        order.append(f"secret:{path}")
        return secrets[str(path)]

    def build_metrics(
        _cls: type[RuntimeMetrics],
        token: object,
        *,
        endpoint: str,
    ) -> RuntimeMetrics:
        assert token is settings.observability_token()
        assert endpoint == settings.otlp_http_endpoint
        order.append("metrics")
        return RuntimeMetrics.in_memory()

    monkeypatch.setattr(RuntimeMetrics, "production", classmethod(build_metrics))

    factories = RuntimeProductionFactories(
        validate_artifacts=lambda received: order.append(
            "artifacts" if received is settings else "wrong-settings"
        ),
        load_manifest=lambda received: (
            order.append("manifest") or manifest
            if received is settings
            else manifest
        ),
        load_profile=lambda received, selected_manifest, _now: (
            order.append("profile")
            or RuntimeProfileSelection(profile=profile, override=None)
            if received is settings and selected_manifest is manifest
            else RuntimeProfileSelection(profile=profile, override=None)
        ),
        read_secret=read_secret,
        load_keyring=lambda received: (
            order.append("keyring")
            or CryptoKeyring({1: KEY}, active_version=1)
        ),
        sink_factory=lambda _dsn: order.append("sink") or Sink(),
        call_control_factory=lambda _key: order.append("call-control") or Control(),
        inference_factory=lambda _key, _profile: (
            order.append("inference")
            or RuntimeInferenceFactories(
                stt_http_client_factory=lambda: object(),  # type: ignore[arg-type]
                stt_factory=lambda _client: object(),  # type: ignore[arg-type]
                llm_factory=lambda: object(),  # type: ignore[arg-type]
                tts_factory=lambda: object(),  # type: ignore[arg-type]
            )
        ),
    )

    graph = await build_production_runtime(
        settings,
        factories=factories,
        utcnow=lambda: NOW,
        monotonic=lambda: 10.0,
        startup_phase_timeout_seconds=2.0,
    )

    assert order[:12] == [
        "artifacts",
        "manifest",
        "profile",
        f"secret:{settings.telnyx_api_key_file}",
        f"secret:{settings.telnyx_webhook_public_key_file}",
        f"secret:{settings.openrouter_api_key_file}",
        f"secret:{settings.postgres_dsn_file}",
        "keyring",
        "metrics",
        "sink",
        "call-control",
        "inference",
    ]
    assert graph.raw_call_control is not graph.measured_call_control
    assert graph.supervisor.call_control_facade is graph.measured_call_control
    assert graph.registry.call_control_identity is graph.measured_call_control
    assert (
        graph.session_factory.recording_call_control_identity
        is graph.measured_call_control
    )
    assert graph.recording_call_control_identity is graph.measured_call_control
    assert isinstance(graph.handshake, AuthenticatedTelnyxHandshakeService)
    assert isinstance(graph.webhook_processor, TelnyxWebhookProcessor)
    await graph.supervisor.aclose()

    from projetv0_voice.persistence.postgres_sink import OperationSinkTransientError

    sentinel = "PRIVATE-SINK-SENTINEL"

    def fail_sink(_dsn: SecretStr) -> object:
        raise OperationSinkTransientError(sentinel)

    failing_factories = replace(factories, sink_factory=fail_sink)  # type: ignore[arg-type]
    with pytest.raises(
        RuntimeError, match="^runtime_production_composition_failed$"
    ) as captured:
        await build_production_runtime(
            settings,
            factories=failing_factories,
            utcnow=lambda: NOW,
            monotonic=lambda: 10.0,
            startup_phase_timeout_seconds=2.0,
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)


@pytest.mark.asyncio
async def test_startup_phase_timeout_is_independent_and_unwinds_blocked_sink(
    tmp_path: Path,
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    open_entered = asyncio.Event()
    never_open = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_after_cancel = asyncio.Event()
    sink_closed = asyncio.Event()

    class Sink:
        async def open(self) -> None:
            open_entered.set()
            try:
                await never_open.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_after_cancel.wait()

        async def close(self) -> None:
            sink_closed.set()

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        sink=Sink(),
        metrics=RuntimeMetrics.in_memory(),
        startup_phase_timeout_seconds=2.0,
        startup_phase_timeouts={"operation_sink_open_failed": 0.05},
        shutdown_timeout_seconds=30.0,
    )

    startup = asyncio.create_task(supervisor.startup())
    await open_entered.wait()
    done, _pending = await asyncio.wait((startup,), timeout=0.2)
    returned_at_phase_deadline = startup in done
    assert cancellation_seen.is_set()
    release_after_cancel.set()
    with pytest.raises(RuntimeError, match="^runtime_startup_failed$"):
        await startup
    cleanup = supervisor._startup_cleanup_task  # noqa: SLF001
    assert cleanup is not None
    await asyncio.wait_for(asyncio.shield(cleanup), timeout=2.0)

    assert returned_at_phase_deadline
    assert open_entered.is_set()
    assert sink_closed.is_set()
    assert supervisor._writer_task is not None  # noqa: SLF001
    assert supervisor._writer_task.done()  # noqa: SLF001


@pytest.mark.asyncio
async def test_pre_supervisor_composition_unwind_uses_one_deadline() -> None:
    from typing import NoReturn

    from projetv0_voice.lifecycle import _close_failed_composition

    class HardExitSentinel(BaseException):
        pass

    never = asyncio.Event()
    entered: list[str] = []
    hard_exit_calls: list[int] = []

    async def block(name: str) -> None:
        entered.append(name)
        try:
            await never.wait()
        except asyncio.CancelledError:
            return

    class Control:
        async def aclose(self) -> None:
            await block("call_control")

    class Sink:
        async def close(self) -> None:
            await block("sink")

    class Metrics:
        async def aclose(self) -> None:
            await block("metrics")

    def hard_exit(code: int) -> NoReturn:
        hard_exit_calls.append(code)
        raise HardExitSentinel

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(HardExitSentinel):
        await _close_failed_composition(
            Control(),  # type: ignore[arg-type]
            Sink(),  # type: ignore[arg-type]
            Metrics(),  # type: ignore[arg-type]
            timeout_seconds=0.05,
            hard_exit=hard_exit,
        )
    elapsed = loop.time() - started

    assert elapsed < 0.1
    assert entered == ["call_control"]
    assert hard_exit_calls == [72]
    live_closes = tuple(
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("voice-composition-close-")
    )
    await asyncio.gather(*live_closes, return_exceptions=True)


@pytest.mark.asyncio
async def test_startup_cleanup_hard_exits_when_phase_child_remains_live(
    tmp_path: Path,
) -> None:
    from typing import NoReturn

    from projetv0_voice.lifecycle import RuntimeSupervisor

    class HardExitSentinel(BaseException):
        pass

    entered = asyncio.Event()
    release = asyncio.Event()
    hard_exit_calls: list[int] = []

    class Sink:
        async def open(self) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        async def close(self) -> None:
            return None

    def hard_exit(code: int) -> NoReturn:
        hard_exit_calls.append(code)
        raise HardExitSentinel

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    metrics = RuntimeMetrics.in_memory()
    sink = Sink()
    supervisor = RuntimeSupervisor(
        writer=writer,
        sink=sink,
        metrics=metrics,
        startup_phase_timeout_seconds=1.0,
        startup_phase_timeouts={"operation_sink_open_failed": 0.01},
        shutdown_timeout_seconds=0.05,
        hard_exit=hard_exit,
    )

    with pytest.raises(RuntimeError, match="^runtime_startup_failed$"):
        await supervisor.startup()
    assert entered.is_set()
    cleanup = supervisor._startup_cleanup_task  # noqa: SLF001
    assert cleanup is not None
    with pytest.raises(HardExitSentinel):
        await cleanup
    assert hard_exit_calls == [72]
    assert supervisor.readiness_snapshot().ready is False
    assert supervisor.readiness_snapshot().admission_open is False

    release.set()
    await asyncio.gather(
        *tuple(supervisor._startup_phase_tasks),  # noqa: SLF001
        return_exceptions=True,
    )
    await writer.drain(timeout_seconds=1.0)
    assert supervisor._writer_task is not None  # noqa: SLF001
    await supervisor._writer_task  # noqa: SLF001
    await sink.close()
    await metrics.aclose()


@pytest.mark.asyncio
async def test_composition_cleanup_hard_exits_instead_of_abandoning_live_close() -> None:
    from typing import NoReturn

    from projetv0_voice.lifecycle import _close_failed_composition

    class HardExitSentinel(BaseException):
        pass

    entered = asyncio.Event()
    release = asyncio.Event()
    hard_exit_calls: list[int] = []

    class Control:
        async def aclose(self) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

    def hard_exit(code: int) -> NoReturn:
        hard_exit_calls.append(code)
        raise HardExitSentinel

    with pytest.raises(HardExitSentinel):
        await _close_failed_composition(
            Control(),  # type: ignore[arg-type]
            None,
            None,
            timeout_seconds=0.01,
            hard_exit=hard_exit,
        )
    assert entered.is_set()
    assert hard_exit_calls == [72]

    release.set()
    live_closes = tuple(
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("voice-composition-close-")
    )
    await asyncio.gather(*live_closes, return_exceptions=True)


def test_composition_cleanup_production_hard_exit_is_nonreturning_subprocess() -> None:
    import subprocess
    import sys

    script = r"""
import asyncio
from projetv0_voice.lifecycle import _close_failed_composition

class Control:
    async def aclose(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()

asyncio.run(_close_failed_composition(Control(), None, None, timeout_seconds=0.01))
print("POST-HARD-EXIT-MARKER")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 72
    assert "POST-HARD-EXIT-MARKER" not in result.stdout
    assert "POST-HARD-EXIT-MARKER" not in result.stderr


@pytest.mark.asyncio
async def test_composition_cleanup_cancellation_cannot_abandon_live_close() -> None:
    from typing import NoReturn

    from projetv0_voice.lifecycle import _close_failed_composition

    class HardExitSentinel(BaseException):
        pass

    entered = asyncio.Event()
    release = asyncio.Event()
    hard_exit_calls: list[int] = []

    class Control:
        async def aclose(self) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

    def hard_exit(code: int) -> NoReturn:
        hard_exit_calls.append(code)
        raise HardExitSentinel

    cleanup = asyncio.create_task(
        _close_failed_composition(
            Control(),  # type: ignore[arg-type]
            None,
            None,
            timeout_seconds=0.05,
            hard_exit=hard_exit,
        )
    )
    await entered.wait()
    cleanup.cancel()
    with pytest.raises(HardExitSentinel):
        await cleanup
    assert hard_exit_calls == [72]

    release.set()
    live_closes = tuple(
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("voice-composition-close-")
    )
    await asyncio.gather(*live_closes, return_exceptions=True)


@pytest.mark.asyncio
async def test_public_startup_failure_clears_private_sink_exception_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor
    from projetv0_voice.persistence.postgres_sink import OperationSinkTransientError

    sentinel = "PRIVATE-SINK-SENTINEL"

    class Sink:
        async def open(self) -> None:
            raise OperationSinkTransientError(sentinel)

        async def close(self) -> None:
            return None

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        sink=Sink(),
        metrics=RuntimeMetrics.in_memory(),
        startup_phase_timeout_seconds=2.0,
        shutdown_timeout_seconds=30.0,
    )

    with pytest.raises(RuntimeError, match="^runtime_startup_failed$") as captured:
        await supervisor.startup()

    output = capsys.readouterr()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in str(captured.value)
    assert sentinel not in repr(captured.value)
    assert sentinel not in output.out
    assert sentinel not in output.err
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_public_shutdown_failure_clears_private_sink_exception_context() -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor
    from projetv0_voice.persistence.postgres_sink import OperationSinkTransientError

    sentinel = "PRIVATE-SINK-SENTINEL"

    class Sink:
        async def open(self) -> None:
            return None

        async def close(self) -> None:
            raise OperationSinkTransientError(sentinel)

    supervisor = RuntimeSupervisor(sink=Sink(), shutdown_timeout_seconds=1.0)

    with pytest.raises(RuntimeError, match="^runtime_shutdown_failed$") as captured:
        await supervisor.aclose()

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)


@pytest.mark.asyncio
async def test_public_shutdown_maps_private_fixed_supervisor_failure() -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor

    sentinel = "PRIVATE-FIXED-SENTINEL"
    supervisor = RuntimeSupervisor(shutdown_timeout_seconds=1.0)

    async def fail() -> None:
        raise RuntimeError(sentinel)

    task = supervisor.fixed_supervisors.try_start(
        fail(), name="voice-purge-supervisor"
    )
    assert task is not None
    supervisor._fixed_tasks["purge"] = task  # noqa: SLF001
    supervisor.fixed_supervisors.close_registration()
    await asyncio.gather(task, return_exceptions=True)

    with pytest.raises(RuntimeError, match="^owned_task_failed$") as captured:
        await supervisor.aclose()

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in repr(captured.value)


@pytest.mark.asyncio
async def test_parent_startup_cancellation_cancels_phase_once_and_preserves_first(
    tmp_path: Path,
) -> None:
    from typing import NoReturn

    from projetv0_voice.lifecycle import RuntimeSupervisor

    entered = asyncio.Event()
    release_phase = asyncio.Event()
    child_saw_cancel = asyncio.Event()
    sink_close_entered = asyncio.Event()
    allow_sink_close = asyncio.Event()
    child_cancel_count = 0
    hard_exit_calls: list[int] = []

    class Sink:
        async def open(self) -> None:
            nonlocal child_cancel_count
            entered.set()
            try:
                await release_phase.wait()
            except asyncio.CancelledError:
                child_cancel_count += 1
                child_saw_cancel.set()
                raise

        async def close(self) -> None:
            sink_close_entered.set()
            await allow_sink_close.wait()

    def hard_exit(code: int) -> NoReturn:
        hard_exit_calls.append(code)
        raise AssertionError("recoverable parent cancellation must not hard exit")

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    supervisor = RuntimeSupervisor(
        writer=writer,
        sink=Sink(),
        metrics=RuntimeMetrics.in_memory(),
        startup_phase_timeout_seconds=5.0,
        shutdown_timeout_seconds=30.0,
        hard_exit=hard_exit,
    )
    startup = asyncio.create_task(supervisor.startup())
    await entered.wait()

    startup.cancel("first-parent-cancel")
    cancel_waiter = asyncio.create_task(child_saw_cancel.wait())
    cancel_seen, _pending = await asyncio.wait(
        (cancel_waiter,), timeout=0.2
    )
    child_was_cancelled = bool(cancel_seen)
    if not child_was_cancelled:
        cancel_waiter.cancel()
        await asyncio.gather(cancel_waiter, return_exceptions=True)
        release_phase.set()
    await sink_close_entered.wait()
    startup.cancel("second-parent-cancel")
    allow_sink_close.set()

    with pytest.raises(asyncio.CancelledError) as captured:
        await startup

    assert str(captured.value) == "first-parent-cancel"
    assert child_was_cancelled
    assert child_cancel_count == 1
    assert hard_exit_calls == []
    assert supervisor.readiness_snapshot().ready is False
    assert supervisor.readiness_snapshot().admission_open is False
    assert supervisor._startup_phase_tasks == set()  # noqa: SLF001
    assert supervisor._writer_task is not None  # noqa: SLF001
    assert supervisor._writer_task.done()  # noqa: SLF001


@pytest.mark.asyncio
async def test_parent_cancel_swallowing_phase_hard_exits_only_at_cleanup_deadline(
    tmp_path: Path,
) -> None:
    from typing import NoReturn

    from projetv0_voice.lifecycle import RuntimeSupervisor

    class HardExitSentinel(BaseException):
        pass

    entered = asyncio.Event()
    child_saw_cancel = asyncio.Event()
    release = asyncio.Event()
    child_cancel_count = 0
    hard_exit_calls: list[int] = []

    class Sink:
        async def open(self) -> None:
            nonlocal child_cancel_count
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                child_cancel_count += 1
                child_saw_cancel.set()
                await release.wait()

        async def close(self) -> None:
            return None

    def hard_exit(code: int) -> NoReturn:
        hard_exit_calls.append(code)
        raise HardExitSentinel

    writer = PersistenceWriter(
        tmp_path / "voice.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
    )
    metrics = RuntimeMetrics.in_memory()
    supervisor = RuntimeSupervisor(
        writer=writer,
        sink=Sink(),
        metrics=metrics,
        startup_phase_timeout_seconds=5.0,
        shutdown_timeout_seconds=0.05,
        hard_exit=hard_exit,
    )
    startup = asyncio.create_task(supervisor.startup())
    await entered.wait()
    startup.cancel("parent-cancel")

    with pytest.raises(HardExitSentinel):
        await startup
    assert child_saw_cancel.is_set()
    assert child_cancel_count == 1
    assert hard_exit_calls == [72]

    release.set()
    await asyncio.gather(
        *tuple(supervisor._startup_phase_tasks),  # noqa: SLF001
        return_exceptions=True,
    )
    await writer.drain(timeout_seconds=1.0)
    assert supervisor._writer_task is not None  # noqa: SLF001
    await supervisor._writer_task  # noqa: SLF001
    await metrics.aclose()
