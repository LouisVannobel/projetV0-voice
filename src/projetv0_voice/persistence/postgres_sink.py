"""Execute-only PostgreSQL boundary for versioned voice operations."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import ValidationError

from projetv0_voice.audio_contract import BeginCallSnapshotV2, VoiceOperationV2
from projetv0_voice.models import (
    BeginCallSnapshotV1,
    RoutingV1,
    VoiceOperationV1,
    is_valid_provider_recording_id,
    validate_deployment_id,
)
from projetv0_voice.persistence.commands import canonical_operation_bytes

INGEST_SQL = "SELECT voice.ingest_operation_v1(%s::jsonb)"
INGEST_V2_SQL = "SELECT voice.ingest_operation_v2(%s::jsonb)"
LEASE_PURGES_SQL = "SELECT * FROM voice.lease_recording_purge_v1(%s,%s,%s)"
ACK_PURGE_SQL = "SELECT voice.ack_recording_purge_v1(%s,%s,%s,%s)"
BEGIN_CALL_SQL = "SELECT voice.begin_call_v1(%s,%s,%s::jsonb)"
BEGIN_CALL_V2_SQL = "SELECT voice.begin_call_v2(%s,%s,%s::jsonb)"
LEASE_CALL_ERASURES_SQL = "SELECT * FROM voice.lease_call_erasure_v1(%s,%s,%s)"
ACK_CALL_ERASURE_SQL = "SELECT voice.ack_call_erasure_v1(%s,%s,%s)"

POOL_ACQUIRE_TIMEOUT_SECONDS = 2.0
SQL_TRANSACTION_TIMEOUT_SECONDS = 5.0
POOL_CLOSE_TIMEOUT_SECONDS = 5.0

PurgeOutcome = Literal["deleted", "not_found", "retry", "failed"]
_PURGE_OUTCOMES: frozenset[str] = frozenset(
    {"deleted", "not_found", "retry", "failed"}
)
_WORKER_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_PAYLOAD_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OperationSinkError(RuntimeError):
    """A constant-code error that never retains the underlying database error."""


class OperationSinkTransientError(OperationSinkError):
    """No application SQL was dispatched and a later retry is safe."""


class OperationSinkCommitAmbiguousError(OperationSinkError):
    """SQL may have committed; the durable claim must expire before replay."""


class OperationSinkPermanentError(OperationSinkError):
    """The operation cannot safely progress without intervention."""


class OperationSinkContractError(OperationSinkPermanentError):
    """The versioned PostgreSQL result violated its frozen contract."""


class OperationConflictError(OperationSinkPermanentError):
    """An operation ID was reused with a different canonical payload hash."""


class OperationSinkStaleLeaseError(OperationSinkError):
    """The purge lease token no longer owns the recording transition."""


class OperationSinkErasedError(OperationSinkError):
    """PV301 confirms terminal discard of erased or expired call content."""


@dataclass(frozen=True, slots=True)
class RecordingPurgeLease:
    schema_version: int
    recording_id: UUID = field(repr=False)
    lease_token: UUID = field(repr=False)
    telnyx_recording_id: str = field(repr=False)
    purge_attempt: int
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class CallErasureLease:
    schema_version: int
    call_id: UUID = field(repr=False)
    lease_token: UUID = field(repr=False)
    deployment_id: str = field(repr=False)
    original_retention_until: datetime
    lease_expires_at: datetime


class PostgresOperationSink(Protocol):
    async def begin_call(
        self, deployment_id: str, call_id: UUID, routing: RoutingV1
    ) -> BeginCallSnapshotV1: ...

    async def ingest(self, operation: VoiceOperationV1) -> None: ...

    async def lease_call_erasures(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> Sequence[CallErasureLease]: ...

    async def ack_call_erasure(
        self, call_id: UUID, lease_token: UUID, occurred_at: datetime
    ) -> None: ...

    async def lease_recording_purges(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> Sequence[RecordingPurgeLease]: ...

    async def ack_recording_purge(
        self,
        recording_id: UUID,
        lease_token: UUID,
        outcome: PurgeOutcome,
        occurred_at: datetime,
    ) -> None: ...


PoolFactory = Callable[..., AsyncConnectionPool[Any]]

_FailureKind = Literal[
    "transient",
    "ambiguous",
    "permanent",
    "contract",
    "conflict",
    "stale",
    "erased",
]


@dataclass(frozen=True, slots=True)
class _SafeFailure:
    kind: _FailureKind
    code: str


class _RollbackBoundary(Exception):
    """Force pool exception-exit using only a constant safe kind and code."""

    __slots__ = ("kind", "code")

    def __init__(self, kind: _FailureKind, code: str) -> None:
        super().__init__(code)
        self.kind = kind
        self.code = code


@dataclass(frozen=True, slots=True)
class _PoolBuildResult:
    pool: AsyncConnectionPool[Any] | None = field(repr=False)
    failure: _SafeFailure | None


@dataclass(frozen=True, slots=True)
class _LeaseCallResult:
    leases: tuple[RecordingPurgeLease, ...] = field(repr=False)
    failure: _SafeFailure | None


@dataclass(frozen=True, slots=True)
class _BeginCallResult:
    snapshot: BeginCallSnapshotV1 | None = field(repr=False)
    failure: _SafeFailure | None


@dataclass(frozen=True, slots=True)
class _BeginCallV2Result:
    snapshot: BeginCallSnapshotV2 | None = field(repr=False)
    failure: _SafeFailure | None


@dataclass(frozen=True, slots=True)
class _ErasureLeaseResult:
    leases: tuple[CallErasureLease, ...] = field(repr=False)
    failure: _SafeFailure | None


def _raise_safe_failure(failure: _SafeFailure) -> None:
    exception_type: type[OperationSinkError]
    if failure.kind == "transient":
        exception_type = OperationSinkTransientError
    elif failure.kind == "ambiguous":
        exception_type = OperationSinkCommitAmbiguousError
    elif failure.kind == "contract":
        exception_type = OperationSinkContractError
    elif failure.kind == "conflict":
        exception_type = OperationConflictError
    elif failure.kind == "stale":
        exception_type = OperationSinkStaleLeaseError
    elif failure.kind == "erased":
        exception_type = OperationSinkErasedError
    else:
        exception_type = OperationSinkPermanentError
    raise exception_type(failure.code)


def _build_pool(conninfo: str, pool_factory: PoolFactory) -> _PoolBuildResult:
    try:
        pool = pool_factory(
            conninfo,
            kwargs={"prepare_threshold": None},
            min_size=1,
            max_size=3,
            open=False,
            timeout=POOL_ACQUIRE_TIMEOUT_SECONDS,
        )
    except Exception:
        return _PoolBuildResult(
            pool=None,
            failure=_SafeFailure("permanent", "operation_pool_factory_failed"),
        )
    return _PoolBuildResult(pool=pool, failure=None)


def _safe_failure_from_exception(error: OperationSinkError) -> _SafeFailure:
    if isinstance(error, OperationConflictError):
        kind: _FailureKind = "conflict"
    elif isinstance(error, OperationSinkContractError):
        kind = "contract"
    elif isinstance(error, OperationSinkStaleLeaseError):
        kind = "stale"
    elif isinstance(error, OperationSinkErasedError):
        kind = "erased"
    elif isinstance(error, OperationSinkCommitAmbiguousError):
        kind = "ambiguous"
    elif isinstance(error, OperationSinkTransientError):
        kind = "transient"
    else:
        kind = "permanent"
    return _SafeFailure(kind, str(error))


def _failure_from_raw_exception(
    error: Exception,
    *,
    dispatched: bool,
    map_purge_ack_sqlstates: bool,
    map_erased_sqlstate: bool = False,
    map_contract_sqlstate: bool = False,
) -> _SafeFailure:
    sqlstate = getattr(error, "sqlstate", None)
    if map_erased_sqlstate and sqlstate == "PV301":
        return _SafeFailure("erased", "operation_call_erased")
    if map_purge_ack_sqlstates:
        if sqlstate == "PV201":
            return _SafeFailure("stale", "purge_stale_lease")
        if sqlstate == "PV202":
            return _SafeFailure("permanent", "purge_contract_failure")
    if map_contract_sqlstate and sqlstate == "PV202":
        return _SafeFailure("permanent", "operation_contract_failure")
    if dispatched:
        return _SafeFailure("ambiguous", "operation_commit_ambiguous")
    return _SafeFailure("transient", "operation_pre_dispatch_failed")


class PsycopgOperationSink:
    """Own one small PgBouncer-safe pool and only versioned function calls."""

    def __init__(
        self,
        conninfo: str,
        *,
        pool_factory: PoolFactory = AsyncConnectionPool,
    ) -> None:
        built = _build_pool(conninfo, pool_factory)
        if built.failure is not None:
            failure = built.failure
            del built, conninfo, pool_factory, self
            _raise_safe_failure(failure)
        if built.pool is None:
            raise RuntimeError("operation_pool_factory_result_invalid")
        self._pool = built.pool
        self._condition = asyncio.Condition()
        self._active_calls = 0
        self._closing = False
        self._closed = False

    def __repr__(self) -> str:
        return "PsycopgOperationSink()"

    async def open(self) -> None:
        failure = await self._open_result()
        if failure is not None:
            del self
            _raise_safe_failure(failure)

    async def _open_result(self) -> _SafeFailure | None:
        async with self._condition:
            if self._closing or self._closed:
                return _SafeFailure("permanent", "operation_sink_closing")
        try:
            await self._pool.open(wait=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _SafeFailure("transient", "operation_pool_open_failed")
        return None

    async def close(self) -> None:
        failure = await self._close_result()
        if failure is not None:
            del self
            _raise_safe_failure(failure)

    async def _close_result(self) -> _SafeFailure | None:
        async with self._condition:
            if self._closed:
                return None
            self._closing = True
            try:
                async with asyncio.timeout(POOL_CLOSE_TIMEOUT_SECONDS):
                    await self._condition.wait_for(lambda: self._active_calls == 0)
            except TimeoutError:
                return _SafeFailure("permanent", "operation_sink_quiesce_timeout")
        try:
            await self._pool.close(timeout=POOL_CLOSE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _SafeFailure("permanent", "operation_sink_close_failed")
        async with self._condition:
            self._closed = True
        return None

    async def ingest(self, operation: VoiceOperationV1) -> None:
        if not isinstance(operation, VoiceOperationV1):
            raise ValueError("operation must be VoiceOperationV1")
        failure = await self._ingest_result(operation)
        if failure is not None:
            del self, operation
            _raise_safe_failure(failure)

    async def ingest_v2(self, operation: VoiceOperationV2) -> None:
        if not isinstance(operation, VoiceOperationV2):
            raise ValueError("operation must be VoiceOperationV2")
        failure = await self._ingest_v2_result(operation)
        if failure is not None:
            del self, operation
            _raise_safe_failure(failure)

    async def begin_call(
        self, deployment_id: str, call_id: UUID, routing: RoutingV1
    ) -> BeginCallSnapshotV1:
        validate_deployment_id(deployment_id)
        if not isinstance(call_id, UUID) or not isinstance(routing, RoutingV1):
            raise ValueError("call identity and routing must be typed")
        result = await self._begin_call_result(deployment_id, call_id, routing)
        if result.failure is not None:
            failure = result.failure
            del result, self, deployment_id, call_id, routing
            _raise_safe_failure(failure)
        if result.snapshot is None:
            raise OperationSinkContractError("begin_call_result_invalid")
        return result.snapshot

    async def _begin_call_result(
        self, deployment_id: str, call_id: UUID, routing: RoutingV1
    ) -> _BeginCallResult:
        snapshots: list[BeginCallSnapshotV1] = []

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            raw = _one_json_object(rows, "begin_call_result_invalid")
            try:
                snapshot = BeginCallSnapshotV1.model_validate(raw)
            except (ValidationError, ValueError):
                raise OperationSinkContractError("begin_call_result_invalid") from None
            if snapshot.call_id != call_id or snapshot.retention_until != (
                routing.admitted_at + timedelta(seconds=2_592_000)
            ):
                raise OperationSinkContractError("begin_call_result_invalid")
            snapshots.append(snapshot)

        failure = await self._execute_result(
            BEGIN_CALL_SQL,
            (deployment_id, call_id, Jsonb(routing.model_dump(mode="json"))),
            validate,
            map_purge_ack_sqlstates=False,
        )
        return _BeginCallResult(
            snapshot=None if failure is not None else snapshots[0], failure=failure
        )

    async def begin_call_v2(
        self, deployment_id: str, call_id: UUID, routing: RoutingV1
    ) -> BeginCallSnapshotV2:
        validate_deployment_id(deployment_id)
        if not isinstance(call_id, UUID) or not isinstance(routing, RoutingV1):
            raise ValueError("call identity and routing must be typed")
        result = await self._begin_call_v2_result(deployment_id, call_id, routing)
        if result.failure is not None:
            failure = result.failure
            del result, self, deployment_id, call_id, routing
            _raise_safe_failure(failure)
        if result.snapshot is None:
            raise OperationSinkContractError("begin_call_v2_result_invalid")
        return result.snapshot

    async def _begin_call_v2_result(
        self, deployment_id: str, call_id: UUID, routing: RoutingV1
    ) -> _BeginCallV2Result:
        snapshots: list[BeginCallSnapshotV2] = []

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            raw = _one_json_object(rows, "begin_call_v2_result_invalid")
            snapshot: BeginCallSnapshotV2 | None = None
            with suppress(ValidationError, ValueError):
                snapshot = BeginCallSnapshotV2.model_validate(raw)
            # Raise outside the parser's suppression scope; its input-bearing
            # ValidationError must not become the safe RPC error's context.
            if (
                snapshot is None
                or snapshot.call_id != call_id
                or snapshot.retention_until
                != routing.admitted_at + timedelta(seconds=2_592_000)
            ):
                del raw, snapshot
                raise OperationSinkContractError("begin_call_v2_result_invalid")
            snapshots.append(snapshot)

        failure = await self._execute_result(
            BEGIN_CALL_V2_SQL,
            (deployment_id, call_id, Jsonb(routing.model_dump(mode="json"))),
            validate,
            map_purge_ack_sqlstates=False,
        )
        return _BeginCallV2Result(
            snapshot=None if failure is not None else snapshots[0], failure=failure
        )

    async def lease_call_erasures(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> Sequence[CallErasureLease]:
        if not isinstance(worker_id, str) or _WORKER_ID.fullmatch(worker_id) is None:
            raise ValueError("worker_id is outside the supported range")
        _exact_bounded_int(lease_seconds, 1, 300, "lease_seconds")
        _exact_bounded_int(batch_size, 1, 100, "batch_size")
        result = await self._lease_call_erasures_result(worker_id, lease_seconds, batch_size)
        if result.failure is not None:
            failure = result.failure
            del result, self, worker_id, lease_seconds, batch_size
            _raise_safe_failure(failure)
        return result.leases

    async def _lease_call_erasures_result(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> _ErasureLeaseResult:
        leases: list[CallErasureLease] = []

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            code = "call_erasure_lease_result_invalid"
            if len(rows) > batch_size:
                raise OperationSinkContractError(code)
            call_ids: set[UUID] = set()
            tokens: set[UUID] = set()
            for row in rows:
                value = _row_json_object(row, code)
                if set(value) != {
                    "schema_version",
                    "call_id",
                    "lease_token",
                    "deployment_id",
                    "original_retention_until",
                    "lease_expires_at",
                }:
                    raise OperationSinkContractError(code)
                if type(value["schema_version"]) is not int or value["schema_version"] != 1:
                    raise OperationSinkContractError(code)
                call_id = _strict_uuid(value["call_id"], code)
                token = _strict_uuid(value["lease_token"], code)
                try:
                    deployment_id = validate_deployment_id(value["deployment_id"])
                except ValueError:
                    raise OperationSinkContractError(code) from None
                retention = _strict_aware_datetime(value["original_retention_until"], code)
                expires = _strict_aware_datetime(value["lease_expires_at"], code)
                if call_id in call_ids or token in tokens:
                    raise OperationSinkContractError(code)
                call_ids.add(call_id)
                tokens.add(token)
                leases.append(
                    CallErasureLease(1, call_id, token, deployment_id, retention, expires)
                )

        failure = await self._execute_result(
            LEASE_CALL_ERASURES_SQL,
            (worker_id, lease_seconds, batch_size),
            validate,
            map_purge_ack_sqlstates=False,
        )
        return _ErasureLeaseResult(
            leases=() if failure is not None else tuple(leases), failure=failure
        )

    async def ack_call_erasure(
        self, call_id: UUID, lease_token: UUID, occurred_at: datetime
    ) -> None:
        if not isinstance(call_id, UUID) or not isinstance(lease_token, UUID):
            raise ValueError("call and lease IDs must be UUIDs")
        normalized_time = _input_aware_datetime(occurred_at)
        failure = await self._ack_call_erasure_result(call_id, lease_token, normalized_time)
        if failure is not None:
            del self, call_id, lease_token, occurred_at, normalized_time
            _raise_safe_failure(failure)

    async def _ack_call_erasure_result(
        self, call_id: UUID, lease_token: UUID, occurred_at: datetime
    ) -> _SafeFailure | None:
        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is not None:
                raise OperationSinkContractError("call_erasure_ack_result_invalid")

        return await self._execute_result(
            ACK_CALL_ERASURE_SQL,
            (call_id, lease_token, occurred_at),
            validate,
            map_purge_ack_sqlstates=True,
        )

    async def _ingest_result(self, operation: VoiceOperationV1) -> _SafeFailure | None:
        expected_id = operation.operation_id

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            result = _one_json_object(rows, "ingest_result_invalid")
            if set(result) != {
                "schema_version",
                "status",
                "operation_id",
                "payload_sha256",
            }:
                raise OperationSinkContractError("ingest_result_invalid")
            if type(result["schema_version"]) is not int or result["schema_version"] != 1:
                raise OperationSinkContractError("ingest_result_invalid")
            status = result["status"]
            if not isinstance(status, str) or status not in {
                "applied",
                "duplicate",
                "conflict",
            }:
                raise OperationSinkContractError("ingest_result_invalid")
            if _strict_uuid(result["operation_id"], "ingest_result_invalid") != expected_id:
                raise OperationSinkContractError("ingest_result_invalid")
            payload_hash = result["payload_sha256"]
            if not isinstance(payload_hash, str) or _PAYLOAD_SHA256.fullmatch(payload_hash) is None:
                raise OperationSinkContractError("ingest_result_invalid")
            if status == "conflict":
                raise OperationConflictError("operation_hash_conflict")

        return await self._execute_result(
            INGEST_SQL,
            (Jsonb(operation.model_dump(mode="json")),),
            validate,
            map_purge_ack_sqlstates=False,
        )

    async def _ingest_v2_result(self, operation: VoiceOperationV2) -> _SafeFailure | None:
        expected_id = operation.operation_id
        wire = canonical_operation_bytes(operation).decode("utf-8")

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            result = _one_json_object(rows, "ingest_v2_result_invalid")
            if set(result) != {
                "schema_version", "status", "operation_id", "payload_sha256"
            }:
                raise OperationSinkContractError("ingest_v2_result_invalid")
            if type(result["schema_version"]) is not int or result["schema_version"] != 2:
                raise OperationSinkContractError("ingest_v2_result_invalid")
            status = result["status"]
            if not isinstance(status, str) or status not in {
                "applied", "duplicate", "conflict"
            }:
                raise OperationSinkContractError("ingest_v2_result_invalid")
            if _strict_uuid(result["operation_id"], "ingest_v2_result_invalid") != expected_id:
                raise OperationSinkContractError("ingest_v2_result_invalid")
            payload_hash = result["payload_sha256"]
            if not isinstance(payload_hash, str) or _PAYLOAD_SHA256.fullmatch(payload_hash) is None:
                raise OperationSinkContractError("ingest_v2_result_invalid")
            if status == "conflict":
                raise OperationConflictError("operation_hash_conflict")

        return await self._execute_result(
            INGEST_V2_SQL,
            (Jsonb(operation.model_dump(mode="json"), dumps=lambda _: wire),),
            validate,
            map_purge_ack_sqlstates=False,
        )

    async def lease_recording_purges(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> Sequence[RecordingPurgeLease]:
        if not isinstance(worker_id, str) or _WORKER_ID.fullmatch(worker_id) is None:
            raise ValueError("worker_id is outside the supported range")
        _exact_bounded_int(lease_seconds, 1, 300, "lease_seconds")
        _exact_bounded_int(batch_size, 1, 100, "batch_size")
        result = await self._lease_recording_purges_result(
            worker_id, lease_seconds, batch_size
        )
        if result.failure is not None:
            failure = result.failure
            del result, self, worker_id, lease_seconds, batch_size
            _raise_safe_failure(failure)
        return result.leases

    async def _lease_recording_purges_result(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> _LeaseCallResult:
        leases: list[RecordingPurgeLease] = []

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            if len(rows) > batch_size:
                raise OperationSinkContractError("purge_lease_result_invalid")
            recording_ids: set[UUID] = set()
            lease_tokens: set[UUID] = set()
            for row in rows:
                result = _row_json_object(row, "purge_lease_result_invalid")
                if set(result) != {
                    "schema_version",
                    "recording_id",
                    "lease_token",
                    "telnyx_recording_id",
                    "purge_attempt",
                    "lease_expires_at",
                }:
                    raise OperationSinkContractError("purge_lease_result_invalid")
                if type(result["schema_version"]) is not int or result["schema_version"] != 1:
                    raise OperationSinkContractError("purge_lease_result_invalid")
                recording_id = _strict_uuid(
                    result["recording_id"], "purge_lease_result_invalid"
                )
                lease_token = _strict_uuid(
                    result["lease_token"], "purge_lease_result_invalid"
                )
                provider_id = result["telnyx_recording_id"]
                if not is_valid_provider_recording_id(provider_id):
                    raise OperationSinkContractError("purge_lease_result_invalid")
                attempt = result["purge_attempt"]
                if type(attempt) is not int or not 1 <= attempt <= 1_000_000:
                    raise OperationSinkContractError("purge_lease_result_invalid")
                expires_at = _strict_aware_datetime(
                    result["lease_expires_at"], "purge_lease_result_invalid"
                )
                if recording_id in recording_ids or lease_token in lease_tokens:
                    raise OperationSinkContractError("purge_lease_result_invalid")
                recording_ids.add(recording_id)
                lease_tokens.add(lease_token)
                leases.append(
                    RecordingPurgeLease(
                        schema_version=1,
                        recording_id=recording_id,
                        lease_token=lease_token,
                        telnyx_recording_id=provider_id,
                        purge_attempt=attempt,
                        lease_expires_at=expires_at,
                    )
                )

        failure = await self._execute_result(
            LEASE_PURGES_SQL,
            (worker_id, lease_seconds, batch_size),
            validate,
            map_purge_ack_sqlstates=False,
        )
        if failure is not None:
            return _LeaseCallResult(leases=(), failure=failure)
        return _LeaseCallResult(leases=tuple(leases), failure=None)

    async def ack_recording_purge(
        self,
        recording_id: UUID,
        lease_token: UUID,
        outcome: PurgeOutcome,
        occurred_at: datetime,
    ) -> None:
        if not isinstance(recording_id, UUID) or not isinstance(lease_token, UUID):
            raise ValueError("recording and lease IDs must be UUIDs")
        if not isinstance(outcome, str) or outcome not in _PURGE_OUTCOMES:
            raise ValueError("outcome is outside the supported values")
        normalized_time = _input_aware_datetime(occurred_at)
        failure = await self._ack_recording_purge_result(
            recording_id,
            lease_token,
            outcome,
            normalized_time,
        )
        if failure is not None:
            del self, recording_id, lease_token, outcome, occurred_at, normalized_time
            _raise_safe_failure(failure)

    async def _ack_recording_purge_result(
        self,
        recording_id: UUID,
        lease_token: UUID,
        outcome: PurgeOutcome,
        occurred_at: datetime,
    ) -> _SafeFailure | None:

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is not None:
                raise OperationSinkContractError("purge_ack_result_invalid")

        return await self._execute_result(
            ACK_PURGE_SQL,
            (recording_id, lease_token, outcome, occurred_at),
            validate,
            map_purge_ack_sqlstates=True,
        )

    async def _execute_result(
        self,
        sql: str,
        params: tuple[object, ...],
        validate: Callable[[Sequence[tuple[object, ...]]], None],
        *,
        map_purge_ack_sqlstates: bool,
    ) -> _SafeFailure | None:
        if not await self._enter_call():
            return _SafeFailure("permanent", "operation_sink_closing")
        dispatched = False
        failure: _SafeFailure | None = None
        try:
            try:
                async with self._pool.connection(
                    timeout=POOL_ACQUIRE_TIMEOUT_SECONDS
                ) as connection:
                    rollback: _SafeFailure | None = None
                    try:
                        async with asyncio.timeout(SQL_TRANSACTION_TIMEOUT_SECONDS):
                            async with connection.transaction():
                                dispatched = True
                                cursor = await connection.execute(sql, params, prepare=False)
                                rows = await cursor.fetchall()
                                validate(cast(Sequence[tuple[object, ...]], rows))
                    except asyncio.CancelledError:
                        raise
                    except OperationSinkError as safe_error:
                        rollback = _safe_failure_from_exception(safe_error)
                    except Exception as raw_error:
                        rollback = _failure_from_raw_exception(
                            raw_error,
                            dispatched=dispatched,
                            map_purge_ack_sqlstates=map_purge_ack_sqlstates,
                            map_erased_sqlstate=sql in {INGEST_SQL, INGEST_V2_SQL},
                            map_contract_sqlstate=sql in {
                                BEGIN_CALL_SQL, BEGIN_CALL_V2_SQL, LEASE_CALL_ERASURES_SQL
                            },
                        )
                    if rollback is not None:
                        raise _RollbackBoundary(rollback.kind, rollback.code)
            except asyncio.CancelledError:
                raise
            except _RollbackBoundary as rollback_error:
                failure = _SafeFailure(rollback_error.kind, rollback_error.code)
            except PoolTimeout:
                failure = _SafeFailure(
                    "ambiguous" if dispatched else "transient",
                    "operation_commit_ambiguous" if dispatched else "pool_acquire_timeout",
                )
            except OperationSinkError as safe_error:
                failure = _safe_failure_from_exception(safe_error)
            except Exception as raw_error:
                failure = _failure_from_raw_exception(
                    raw_error,
                    dispatched=dispatched,
                    map_purge_ack_sqlstates=map_purge_ack_sqlstates,
                    map_erased_sqlstate=sql in {INGEST_SQL, INGEST_V2_SQL},
                    map_contract_sqlstate=sql in {
                        BEGIN_CALL_SQL, BEGIN_CALL_V2_SQL, LEASE_CALL_ERASURES_SQL
                    },
                )
        finally:
            await self._leave_call()
        return failure

    async def _enter_call(self) -> bool:
        async with self._condition:
            if self._closing or self._closed:
                return False
            self._active_calls += 1
            return True

    async def _leave_call(self) -> None:
        async with self._condition:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._condition.notify_all()


def _one_json_object(
    rows: Sequence[tuple[object, ...]], code: str
) -> dict[str, object]:
    if len(rows) != 1:
        raise OperationSinkContractError(code)
    return _row_json_object(rows[0], code)


def _row_json_object(row: tuple[object, ...], code: str) -> dict[str, object]:
    if len(row) != 1 or not isinstance(row[0], dict):
        raise OperationSinkContractError(code)
    if not all(isinstance(key, str) for key in row[0]):
        raise OperationSinkContractError(code)
    return cast(dict[str, object], row[0])


def _strict_uuid(value: object, code: str) -> UUID:
    if not isinstance(value, str):
        raise OperationSinkContractError(code)
    try:
        parsed = UUID(value)
    except ValueError:
        raise OperationSinkContractError(code) from None
    if str(parsed) != value:
        raise OperationSinkContractError(code)
    return parsed


def _strict_aware_datetime(value: object, code: str) -> datetime:
    if not isinstance(value, str):
        raise OperationSinkContractError(code)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise OperationSinkContractError(code) from None
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset() != timedelta(0)
    ):
        raise OperationSinkContractError(code)
    return parsed.astimezone(UTC)


def _input_aware_datetime(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be an aware datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError("occurred_at must be UTC")
    return value.astimezone(UTC)


def _exact_bounded_int(value: object, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the supported range")
    return value
