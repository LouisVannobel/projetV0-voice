"""Execute-only PostgreSQL boundary for versioned voice operations."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from projetv0_voice.models import VoiceOperationV1

INGEST_SQL = "SELECT voice.ingest_operation_v1(%s::jsonb)"
LEASE_PURGES_SQL = "SELECT * FROM voice.lease_recording_purge_v1(%s,%s,%s)"
ACK_PURGE_SQL = "SELECT voice.ack_recording_purge_v1(%s,%s,%s,%s)"

POOL_ACQUIRE_TIMEOUT_SECONDS = 2.0
SQL_TRANSACTION_TIMEOUT_SECONDS = 5.0
POOL_CLOSE_TIMEOUT_SECONDS = 5.0

PurgeOutcome = Literal["deleted", "not_found", "retry", "failed"]
_PURGE_OUTCOMES: frozenset[str] = frozenset(
    {"deleted", "not_found", "retry", "failed"}
)
_WORKER_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_TELNYX_RECORDING_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
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


@dataclass(frozen=True, slots=True)
class RecordingPurgeLease:
    schema_version: int
    recording_id: UUID = field(repr=False)
    lease_token: UUID = field(repr=False)
    telnyx_recording_id: str = field(repr=False)
    purge_attempt: int
    lease_expires_at: datetime


class PostgresOperationSink(Protocol):
    async def ingest(self, operation: VoiceOperationV1) -> None: ...

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


class PsycopgOperationSink:
    """Own one small PgBouncer-safe pool and only versioned function calls."""

    def __init__(
        self,
        conninfo: str,
        *,
        pool_factory: PoolFactory = AsyncConnectionPool,
    ) -> None:
        factory_failure: OperationSinkPermanentError | None = None
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
            factory_failure = OperationSinkPermanentError("operation_pool_factory_failed")
        if factory_failure is not None:
            raise factory_failure
        self._pool = pool
        self._condition = asyncio.Condition()
        self._active_calls = 0
        self._closing = False
        self._closed = False

    def __repr__(self) -> str:
        return "PsycopgOperationSink()"

    async def open(self) -> None:
        async with self._condition:
            if self._closing or self._closed:
                raise OperationSinkPermanentError("operation_sink_closing")
        open_failure: OperationSinkTransientError | None = None
        try:
            await self._pool.open(wait=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            open_failure = OperationSinkTransientError("operation_pool_open_failed")
        if open_failure is not None:
            raise open_failure

    async def close(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closing = True
            try:
                async with asyncio.timeout(POOL_CLOSE_TIMEOUT_SECONDS):
                    await self._condition.wait_for(lambda: self._active_calls == 0)
            except TimeoutError:
                raise OperationSinkPermanentError("operation_sink_quiesce_timeout") from None
        close_failure: OperationSinkPermanentError | None = None
        try:
            await self._pool.close(timeout=POOL_CLOSE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            close_failure = OperationSinkPermanentError("operation_sink_close_failed")
        if close_failure is not None:
            raise close_failure
        async with self._condition:
            self._closed = True

    async def ingest(self, operation: VoiceOperationV1) -> None:
        if not isinstance(operation, VoiceOperationV1):
            raise ValueError("operation must be VoiceOperationV1")
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

        await self._execute(
            INGEST_SQL,
            (Jsonb(operation.model_dump(mode="json")),),
            validate,
        )

    async def lease_recording_purges(
        self, worker_id: str, lease_seconds: int, batch_size: int
    ) -> Sequence[RecordingPurgeLease]:
        if not isinstance(worker_id, str) or _WORKER_ID.fullmatch(worker_id) is None:
            raise ValueError("worker_id is outside the supported range")
        _exact_bounded_int(lease_seconds, 1, 300, "lease_seconds")
        _exact_bounded_int(batch_size, 1, 100, "batch_size")
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
                if (
                    not isinstance(provider_id, str)
                    or _TELNYX_RECORDING_ID.fullmatch(provider_id) is None
                ):
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

        await self._execute(
            LEASE_PURGES_SQL,
            (worker_id, lease_seconds, batch_size),
            validate,
        )
        return tuple(leases)

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

        def validate(rows: Sequence[tuple[object, ...]]) -> None:
            if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is not None:
                raise OperationSinkContractError("purge_ack_result_invalid")

        await self._execute(
            ACK_PURGE_SQL,
            (recording_id, lease_token, outcome, normalized_time),
            validate,
        )

    async def _execute(
        self,
        sql: str,
        params: tuple[object, ...],
        validate: Callable[[Sequence[tuple[object, ...]]], None],
    ) -> None:
        safe_failure: OperationSinkError | None = None
        try:
            await self._execute_with_transaction(sql, params, validate)
        except asyncio.CancelledError:
            raise
        except OperationSinkError as error:
            error.__cause__ = None
            error.__context__ = None
            safe_failure = error
        if safe_failure is not None:
            raise safe_failure

    async def _execute_with_transaction(
        self,
        sql: str,
        params: tuple[object, ...],
        validate: Callable[[Sequence[tuple[object, ...]]], None],
    ) -> None:
        await self._enter_call()
        dispatched = False
        try:
            async with self._pool.connection(
                timeout=POOL_ACQUIRE_TIMEOUT_SECONDS
            ) as connection:
                try:
                    async with asyncio.timeout(SQL_TRANSACTION_TIMEOUT_SECONDS):
                        async with connection.transaction():
                            dispatched = True
                            cursor = await connection.execute(sql, params, prepare=False)
                            rows = await cursor.fetchall()
                            validate(cast(Sequence[tuple[object, ...]], rows))
                except asyncio.CancelledError:
                    raise
                except (
                    OperationConflictError,
                    OperationSinkContractError,
                    OperationSinkStaleLeaseError,
                    OperationSinkPermanentError,
                ):
                    raise
                except Exception as error:
                    mapped = self._map_sqlstate(error)
                    if mapped is not None:
                        raise mapped from None
                    if dispatched:
                        raise OperationSinkCommitAmbiguousError(
                            "operation_commit_ambiguous"
                        ) from None
                    raise OperationSinkTransientError("operation_pre_dispatch_failed") from None
        except asyncio.CancelledError:
            raise
        except PoolTimeout:
            if dispatched:
                raise OperationSinkCommitAmbiguousError("operation_commit_ambiguous") from None
            raise OperationSinkTransientError("pool_acquire_timeout") from None
        except (
            OperationConflictError,
            OperationSinkContractError,
            OperationSinkStaleLeaseError,
            OperationSinkPermanentError,
            OperationSinkCommitAmbiguousError,
            OperationSinkTransientError,
        ):
            raise
        except Exception:
            if dispatched:
                raise OperationSinkCommitAmbiguousError("operation_commit_ambiguous") from None
            raise OperationSinkTransientError("operation_pre_dispatch_failed") from None
        finally:
            await self._leave_call()

    @staticmethod
    def _map_sqlstate(error: Exception) -> OperationSinkError | None:
        sqlstate = getattr(error, "sqlstate", None)
        if sqlstate == "PV201":
            return OperationSinkStaleLeaseError("purge_stale_lease")
        if sqlstate == "PV202":
            return OperationSinkPermanentError("purge_contract_failure")
        return None

    async def _enter_call(self) -> None:
        async with self._condition:
            if self._closing or self._closed:
                raise OperationSinkPermanentError("operation_sink_closing")
            self._active_calls += 1

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
    if parsed.tzinfo is None or parsed.utcoffset() is None:
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
