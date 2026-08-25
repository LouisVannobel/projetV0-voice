from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest
from psycopg.types.json import Jsonb
from psycopg_pool import PoolTimeout

from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.postgres_sink import (
    OperationConflictError,
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkPermanentError,
    OperationSinkStaleLeaseError,
    OperationSinkTransientError,
    PsycopgOperationSink,
)

NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)
INGEST_SQL = "SELECT voice.ingest_operation_v1(%s::jsonb)"
LEASE_SQL = "SELECT * FROM voice.lease_recording_purge_v1(%s,%s,%s)"
ACK_SQL = "SELECT voice.ack_recording_purge_v1(%s,%s,%s,%s)"


def operation() -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID(int=1),
        deployment_id="agent-a",
        call_id=UUID(int=2),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="control-1",
            telnyx_call_leg_id=None,
            telnyx_call_session_id=None,
            status="pending",
            disclosure_state="pending",
            started_at=None,
            ended_at=None,
            end_reason=None,
            retention_until=NOW.replace(day=26),
        ),
    )


def ingest_result(status: str = "applied") -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "operation_id": str(UUID(int=1)),
        "payload_sha256": "a" * 64,
    }


class SqlstateError(Exception):
    def __init__(self, sqlstate: str, raw: str = "RAW-DB-SENTINEL") -> None:
        super().__init__(raw)
        self.sqlstate = sqlstate


class FakeCursor:
    def __init__(
        self,
        rows: Sequence[tuple[object, ...]],
        *,
        fetch_error: BaseException | None = None,
        fetch_gate: asyncio.Event | None = None,
    ) -> None:
        self.rows = list(rows)
        self.fetch_error = fetch_error
        self.fetch_gate = fetch_gate

    async def fetchall(self) -> list[tuple[object, ...]]:
        if self.fetch_gate is not None:
            await self.fetch_gate.wait()
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.rows


class FakeTransaction:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeTransaction:
        self.connection.events.append("transaction-enter")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        self.connection.transaction_exit_types.append(exc_type)
        self.connection.events.append("transaction-exit")
        if self.connection.exit_gate is not None:
            await self.connection.exit_gate.wait()
        if exc_type is None and self.connection.commit_error is not None:
            raise self.connection.commit_error
        return False


class FakeConnection:
    def __init__(
        self,
        rows: Sequence[tuple[object, ...]],
        *,
        execute_error: BaseException | None = None,
        fetch_error: BaseException | None = None,
        execute_gate: asyncio.Event | None = None,
        fetch_gate: asyncio.Event | None = None,
        exit_gate: asyncio.Event | None = None,
        commit_error: BaseException | None = None,
    ) -> None:
        self.rows = rows
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.execute_gate = execute_gate
        self.fetch_gate = fetch_gate
        self.exit_gate = exit_gate
        self.commit_error = commit_error
        self.calls: list[tuple[str, tuple[object, ...], bool]] = []
        self.events: list[str] = []
        self.transaction_exit_types: list[type[BaseException] | None] = []

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def execute(
        self, sql: str, params: tuple[object, ...], *, prepare: bool
    ) -> FakeCursor:
        self.calls.append((sql, params, prepare))
        self.events.append("execute")
        if self.execute_gate is not None:
            await self.execute_gate.wait()
        if self.execute_error is not None:
            raise self.execute_error
        return FakeCursor(self.rows, fetch_error=self.fetch_error, fetch_gate=self.fetch_gate)


class ConnectionContext:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> FakeConnection:
        if self.pool.acquire_gate is not None:
            await self.pool.acquire_gate.wait()
        if self.pool.acquire_error is not None:
            raise self.pool.acquire_error
        self.pool.active += 1
        return self.pool.connection_value

    async def __aexit__(self, *_: object) -> None:
        self.pool.active -= 1
        self.pool.returned.set()


class FakePool:
    def __init__(
        self,
        connection: FakeConnection,
        *,
        acquire_error: BaseException | None = None,
        acquire_gate: asyncio.Event | None = None,
        open_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.connection_value = connection
        self.acquire_error = acquire_error
        self.acquire_gate = acquire_gate
        self.open_error = open_error
        self.close_error = close_error
        self.connection_timeouts: list[float | None] = []
        self.open_calls: list[tuple[bool, float]] = []
        self.close_calls: list[float] = []
        self.active = 0
        self.returned = asyncio.Event()

    def connection(self, timeout: float | None = None) -> ConnectionContext:
        self.connection_timeouts.append(timeout)
        return ConnectionContext(self)

    async def open(self, wait: bool = False, timeout: float = 30.0) -> None:  # noqa: ASYNC109
        self.open_calls.append((wait, timeout))
        if self.open_error is not None:
            raise self.open_error

    async def close(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109
        self.close_calls.append(timeout)
        if self.close_error is not None:
            raise self.close_error


class PoolFactory:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> FakePool:
        self.calls.append((args, kwargs))
        return self.pool


def sink_with_rows(
    rows: Sequence[tuple[object, ...]],
    **connection_options: object,
) -> tuple[PsycopgOperationSink, FakePool, FakeConnection, PoolFactory]:
    connection = FakeConnection(rows, **connection_options)
    pool = FakePool(connection)
    factory = PoolFactory(pool)
    sink = PsycopgOperationSink("postgresql://user:SECRET@host/db", pool_factory=factory)
    return sink, pool, connection, factory


def exception_graph(error: BaseException) -> str:
    parts: list[str] = []
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        parts.append(repr(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return " ".join(parts)


@pytest.mark.asyncio
async def test_pool_factory_is_called_once_with_pgbouncer_safe_settings_and_nonblocking_open(
) -> None:
    sink, pool, _, factory = sink_with_rows([(ingest_result(),)])
    assert len(factory.calls) == 1
    args, kwargs = factory.calls[0]
    assert args == ("postgresql://user:SECRET@host/db",)
    assert kwargs == {
        "kwargs": {"prepare_threshold": None},
        "min_size": 1,
        "max_size": 3,
        "open": False,
        "timeout": 2.0,
    }

    await sink.open()
    assert pool.open_calls == [(False, 30.0)]
    assert "SECRET" not in repr(sink)


@pytest.mark.asyncio
async def test_pool_open_failure_is_typed_safe() -> None:
    connection = FakeConnection([(ingest_result(),)])
    pool = FakePool(connection, open_error=RuntimeError("RAW-OPEN-SECRET"))
    sink = PsycopgOperationSink("postgresql://SECRET", pool_factory=PoolFactory(pool))
    with pytest.raises(OperationSinkTransientError, match="operation_pool_open_failed") as error:
        await sink.open()
    assert "RAW-OPEN-SECRET" not in exception_graph(error.value)


def test_pool_factory_failure_is_typed_and_does_not_retain_dsn_or_raw_error() -> None:
    def fail_factory(*_: object, **__: object) -> FakePool:
        raise RuntimeError("RAW-FACTORY-postgresql://SECRET")

    with pytest.raises(
        OperationSinkPermanentError, match="operation_pool_factory_failed"
    ) as error:
        PsycopgOperationSink("postgresql://SECRET", pool_factory=fail_factory)  # type: ignore[arg-type]
    rendered = exception_graph(error.value)
    assert "RAW-FACTORY" not in rendered
    assert "postgresql://SECRET" not in rendered


@pytest.mark.asyncio
async def test_ingest_uses_one_exact_statement_jsonb_and_validates_before_commit() -> None:
    candidate = operation()
    sink, pool, connection, _ = sink_with_rows([(ingest_result(),)])

    await sink.ingest(candidate)

    assert pool.connection_timeouts == [2.0]
    assert len(connection.calls) == 1
    sql, params, prepare = connection.calls[0]
    assert sql == INGEST_SQL
    assert prepare is False
    assert len(params) == 1 and isinstance(params[0], Jsonb)
    assert params[0].obj == candidate.model_dump(mode="json")
    assert connection.transaction_exit_types == [None]
    assert connection.events == ["transaction-enter", "execute", "transaction-exit"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["applied", "duplicate"])
async def test_ingest_accepts_applied_and_duplicate(status: str) -> None:
    sink, _, _, _ = sink_with_rows([(ingest_result(status),)])
    await sink.ingest(operation())


@pytest.mark.asyncio
async def test_ingest_conflict_is_typed_and_rolls_back_before_commit() -> None:
    sink, _, connection, _ = sink_with_rows([(ingest_result("conflict"),)])
    with pytest.raises(OperationConflictError, match="operation_hash_conflict"):
        await sink.ingest(operation())
    assert connection.transaction_exit_types == [OperationConflictError]


INVALID_INGEST_ROWS: list[Sequence[tuple[object, ...]]] = [
    [],
    [(ingest_result(),), (ingest_result(),)],
    [()],
    [(42,)],
    [({key: value for key, value in ingest_result().items() if key != "status"},)],
    [({**ingest_result(), "extra": 1},)],
    [({**ingest_result(), "schema_version": 2},)],
    [({**ingest_result(), "status": "unknown"},)],
    [({**ingest_result(), "operation_id": str(UUID(int=99))},)],
    [({**ingest_result(), "payload_sha256": "A" * 64},)],
    [({**ingest_result(), "schema_version": True},)],
]


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", INVALID_INGEST_ROWS)
async def test_ingest_rejects_malformed_result_inside_transaction(
    rows: Sequence[tuple[object, ...]],
) -> None:
    sink, _, connection, _ = sink_with_rows(rows)
    with pytest.raises(OperationSinkContractError, match="ingest_result_invalid"):
        await sink.ingest(operation())
    assert connection.transaction_exit_types == [OperationSinkContractError]


@pytest.mark.asyncio
async def test_pool_timeout_before_dispatch_is_safe_transient() -> None:
    connection = FakeConnection([(ingest_result(),)])
    pool = FakePool(connection, acquire_error=PoolTimeout("RAW-POOL-SENTINEL"))
    sink = PsycopgOperationSink("postgresql://SECRET", pool_factory=PoolFactory(pool))
    with pytest.raises(OperationSinkTransientError, match="pool_acquire_timeout") as captured:
        await sink.ingest(operation())
    assert connection.calls == []
    assert "RAW-POOL-SENTINEL" not in exception_graph(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["execute", "fetch", "commit"])
async def test_failure_after_dispatch_is_commit_ambiguous_and_safe(stage: str) -> None:
    raw = RuntimeError("RAW-DB-SENTINEL")
    options = {
        "execute_error": raw if stage == "execute" else None,
        "fetch_error": raw if stage == "fetch" else None,
        "commit_error": raw if stage == "commit" else None,
    }
    sink, _, _, _ = sink_with_rows([(ingest_result(),)], **options)
    with pytest.raises(
        OperationSinkCommitAmbiguousError, match="operation_commit_ambiguous"
    ) as captured:
        await sink.ingest(operation())
    assert "RAW-DB-SENTINEL" not in exception_graph(captured.value)


@pytest.mark.asyncio
async def test_transaction_timeout_is_exactly_five_seconds_and_is_commit_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float | None] = []
    real_timeout = asyncio.timeout

    def observe_timeout(delay: float | None) -> asyncio.Timeout:
        observed.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(
        "projetv0_voice.persistence.postgres_sink.asyncio.timeout", observe_timeout
    )
    sink, _, _, _ = sink_with_rows([(ingest_result(),)])
    await sink.ingest(operation())
    assert observed == [5.0]


@pytest.mark.asyncio
async def test_cancellation_from_validation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cancel_validation(*_: object) -> dict[str, object]:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "projetv0_voice.persistence.postgres_sink._one_json_object", cancel_validation
    )
    sink, _, _, _ = sink_with_rows([(ingest_result(),)])
    with pytest.raises(asyncio.CancelledError):
        await sink.ingest(operation())


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["acquire", "execute", "fetch", "commit"])
async def test_cancellation_always_propagates_unmapped(stage: str) -> None:
    gate = asyncio.Event()
    connection = FakeConnection(
        [(ingest_result(),)],
        execute_gate=gate if stage == "execute" else None,
        fetch_gate=gate if stage == "fetch" else None,
        exit_gate=gate if stage == "commit" else None,
    )
    pool = FakePool(connection, acquire_gate=gate if stage == "acquire" else None)
    sink = PsycopgOperationSink("postgresql://SECRET", pool_factory=PoolFactory(pool))
    task = asyncio.create_task(sink.ingest(operation()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def purge_row(**updates: object) -> tuple[dict[str, object]]:
    value: dict[str, object] = {
        "schema_version": 1,
        "recording_id": str(UUID(int=10)),
        "lease_token": str(UUID(int=11)),
        "telnyx_recording_id": "recording_Ab-12",
        "purge_attempt": 1,
        "lease_expires_at": "2026-08-25T12:00:30+00:00",
    }
    value.update(updates)
    return (value,)


@pytest.mark.asyncio
async def test_purge_lease_uses_exact_statement_and_returns_strict_utc_value() -> None:
    sink, _, connection, _ = sink_with_rows([purge_row()])
    leases = await sink.lease_recording_purges("worker-1", 30, 10)
    assert connection.calls == [(LEASE_SQL, ("worker-1", 30, 10), False)]
    assert len(leases) == 1
    assert leases[0].recording_id == UUID(int=10)
    assert leases[0].lease_token == UUID(int=11)
    assert leases[0].telnyx_recording_id == "recording_Ab-12"
    assert leases[0].lease_expires_at == datetime(2026, 8, 25, 12, 0, 30, tzinfo=UTC)
    assert "recording_Ab-12" not in repr(leases[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([purge_row(extra="signed-url")], "purge_lease_result_invalid"),
        ([purge_row(telnyx_recording_id="")], "purge_lease_result_invalid"),
        ([purge_row(telnyx_recording_id="https://signed.example")], "purge_lease_result_invalid"),
        ([purge_row(purge_attempt=True)], "purge_lease_result_invalid"),
        ([purge_row(lease_expires_at="2026-08-25T12:00:30")], "purge_lease_result_invalid"),
        ([purge_row(), purge_row()], "purge_lease_result_invalid"),
    ],
)
async def test_purge_lease_rejects_extra_unsafe_invalid_and_duplicate_rows(
    rows: Sequence[tuple[object, ...]], code: str
) -> None:
    sink, _, _, _ = sink_with_rows(rows)
    with pytest.raises(OperationSinkContractError, match=code):
        await sink.lease_recording_purges("worker-1", 30, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        [purge_row(), purge_row(recording_id=str(UUID(int=12)))],
        [purge_row(), purge_row(lease_token=str(UUID(int=12)))],
        [purge_row(purge_attempt=0)],
        [purge_row(purge_attempt=1_000_001)],
        [purge_row(telnyx_recording_id="x" * 129)],
        [purge_row(recording_id="NOT-A-UUID")],
        [purge_row(lease_token="NOT-A-UUID")],
    ],
)
async def test_purge_lease_rejects_each_identity_and_bound_violation(
    rows: Sequence[tuple[object, ...]],
) -> None:
    sink, _, _, _ = sink_with_rows(rows)
    with pytest.raises(OperationSinkContractError, match="purge_lease_result_invalid"):
        await sink.lease_recording_purges("worker-1", 30, 10)


@pytest.mark.asyncio
async def test_purge_lease_rejects_more_rows_than_requested_batch() -> None:
    sink, _, _, _ = sink_with_rows([purge_row(), purge_row(recording_id=str(UUID(int=12)))])
    with pytest.raises(OperationSinkContractError, match="purge_lease_result_invalid"):
        await sink.lease_recording_purges("worker-1", 30, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_id", "lease_seconds", "batch_size"),
    [
        ("", 30, 1),
        ("bad worker", 30, 1),
        ("x" * 65, 30, 1),
        ("worker", True, 1),
        ("worker", 0, 1),
        ("worker", 301, 1),
        ("worker", 30, 0),
        ("worker", 30, 101),
        ("worker", 30, 1.0),
    ],
)
async def test_purge_lease_rejects_coercion_and_bounds_before_sql(
    worker_id: str, lease_seconds: Any, batch_size: Any
) -> None:
    sink, _, connection, _ = sink_with_rows([])
    with pytest.raises(ValueError):
        await sink.lease_recording_purges(worker_id, lease_seconds, batch_size)
    assert connection.calls == []


@pytest.mark.asyncio
async def test_purge_ack_uses_exact_statement_and_strict_void_result() -> None:
    sink, _, connection, _ = sink_with_rows([(None,)])
    await sink.ack_recording_purge(UUID(int=10), UUID(int=11), "deleted", NOW)
    assert connection.calls == [
        (ACK_SQL, (UUID(int=10), UUID(int=11), "deleted", NOW), False)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [(None,), (None,)], [()], [("not-null",)]])
async def test_purge_ack_rejects_invalid_void_cardinality(
    rows: Sequence[tuple[object, ...]],
) -> None:
    sink, _, _, _ = sink_with_rows(rows)
    with pytest.raises(OperationSinkContractError, match="purge_ack_result_invalid"):
        await sink.ack_recording_purge(UUID(int=10), UUID(int=11), "retry", NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sqlstate", "error_type", "code"),
    [
        ("PV201", OperationSinkStaleLeaseError, "purge_stale_lease"),
        ("PV202", OperationSinkPermanentError, "purge_contract_failure"),
    ],
)
async def test_purge_ack_maps_frozen_sqlstates_without_raw_error(
    sqlstate: str, error_type: type[Exception], code: str
) -> None:
    sink, _, _, _ = sink_with_rows([(None,)], execute_error=SqlstateError(sqlstate))
    with pytest.raises(error_type, match=code) as captured:
        await sink.ack_recording_purge(UUID(int=10), UUID(int=11), "failed", NOW)
    assert "RAW-DB-SENTINEL" not in exception_graph(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recording_id", "lease_token", "outcome", "occurred_at"),
    [
        ("not-uuid", UUID(int=11), "deleted", NOW),
        (UUID(int=10), "not-uuid", "deleted", NOW),
        (UUID(int=10), UUID(int=11), "other", NOW),
        (UUID(int=10), UUID(int=11), "deleted", datetime(2026, 8, 25, 12)),
        (
            UUID(int=10),
            UUID(int=11),
            "deleted",
            datetime(2026, 8, 25, 14, tzinfo=timezone(timedelta(hours=2))),
        ),
    ],
)
async def test_purge_ack_rejects_invalid_or_non_utc_inputs_before_sql(
    recording_id: object, lease_token: object, outcome: object, occurred_at: object
) -> None:
    sink, _, connection, _ = sink_with_rows([(None,)])
    with pytest.raises(ValueError):
        await sink.ack_recording_purge(  # type: ignore[arg-type]
            recording_id, lease_token, outcome, occurred_at
        )
    assert connection.calls == []


@pytest.mark.asyncio
async def test_sink_emits_no_sensitive_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "recording_SECRET_SENTINEL"
    sink, _, _, _ = sink_with_rows([purge_row(telnyx_recording_id=sentinel)])
    await sink.lease_recording_purges("worker-1", 30, 1)
    assert sentinel not in caplog.text
    assert "postgresql://" not in caplog.text


@pytest.mark.asyncio
async def test_close_refuses_new_calls_waits_for_active_call_then_uses_bounded_pool_close(
) -> None:
    gate = asyncio.Event()
    sink, pool, _, _ = sink_with_rows([(ingest_result(),)], execute_gate=gate)
    active = asyncio.create_task(sink.ingest(operation()))
    await asyncio.sleep(0)
    closing = asyncio.create_task(sink.close())
    await asyncio.sleep(0)
    with pytest.raises(OperationSinkPermanentError, match="operation_sink_closing"):
        await sink.ingest(operation())
    assert not closing.done()
    gate.set()
    await active
    await closing
    assert pool.close_calls == [5.0]


@pytest.mark.asyncio
async def test_pool_close_failure_is_typed_safe_and_leaves_sink_fail_closed() -> None:
    connection = FakeConnection([(ingest_result(),)])
    pool = FakePool(connection, close_error=RuntimeError("RAW-CLOSE-SECRET"))
    sink = PsycopgOperationSink("postgresql://SECRET", pool_factory=PoolFactory(pool))
    with pytest.raises(OperationSinkPermanentError, match="operation_sink_close_failed") as error:
        await sink.close()
    assert "RAW-CLOSE-SECRET" not in exception_graph(error.value)
    with pytest.raises(OperationSinkPermanentError, match="operation_sink_closing"):
        await sink.ingest(operation())
