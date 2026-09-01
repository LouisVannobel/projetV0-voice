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
    )

    with pytest.raises(RuntimeError, match="^runtime_startup_failed$"):
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
        shutdown_timeout_seconds=0.05,
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
