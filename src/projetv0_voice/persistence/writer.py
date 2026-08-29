"""Single-owner bounded SQLite command writer."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import inspect
import re
import sqlite3
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import aiosqlite

from projetv0_voice.crypto import (
    CRYPTO_VERSION,
    CryptoError,
    CryptoKeyring,
    EncryptedValue,
    UnknownKeyVersionError,
)
from projetv0_voice.models import (
    RecordingUpsertPayloadV1,
    TurnUpsertPayloadV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    CommandSerializationError,
    FatalPersistenceError,
    PersistenceCommand,
    PersistenceError,
    canonical_operation_bytes,
    decode_operation,
    encrypt_operation,
    operation_aad_from_metadata,
    require_operation,
)
from projetv0_voice.persistence.schema import SCHEMA_SQL, SCHEMA_VERSION

PERSISTENCE_QUEUE_MAX_ITEMS = 256
CONTROL_COMMIT_TIMEOUT_SECONDS = 1.5
QUEUE_OLDEST_LIMIT_SECONDS = 1.0
MAX_STORAGE_BYTES = 268_435_456
RELAY_CLAIM_MAX_SECONDS = 300
RECEIPT_RETENTION = timedelta(days=7)
CLOSED_LEASE_RETENTION = timedelta(hours=24)
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,64}$")

Failpoint = Callable[[str], Awaitable[None] | None]


def _normalize_schema_sql(sql: str) -> str:
    normalized: list[str] = []
    in_literal = False
    index = 0
    while index < len(sql):
        character = sql[index]
        if character == "'":
            normalized.append(character)
            if in_literal and index + 1 < len(sql) and sql[index + 1] == "'":
                normalized.append("'")
                index += 2
                continue
            in_literal = not in_literal
        elif in_literal:
            normalized.append(character)
        elif not character.isspace():
            normalized.append(character.casefold())
        index += 1
    compact = "".join(normalized)
    return compact.replace("createtableifnotexists", "createtable").replace(
        "createindexifnotexists", "createindex"
    )


def _expected_schema_objects() -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}
    for statement in SCHEMA_SQL.split(";"):
        compact = " ".join(statement.split())
        matched = re.match(
            r"^CREATE (TABLE|INDEX) IF NOT EXISTS ([A-Za-z_][A-Za-z0-9_]*)\b",
            compact,
            flags=re.IGNORECASE,
        )
        if matched is not None:
            expected[(matched.group(1).casefold(), matched.group(2))] = _normalize_schema_sql(
                statement
            )
    if set(expected) != {
        ("table", "call_leases"),
        ("table", "webhook_receipts"),
        ("table", "outbox"),
        ("index", "outbox_due_fifo_idx"),
    }:
        raise RuntimeError("invalid_expected_sqlite_schema")
    return expected


_EXPECTED_SCHEMA_OBJECTS = _expected_schema_objects()


@dataclass(frozen=True, slots=True)
class FatalPersistenceFault:
    code: str


@dataclass(frozen=True, slots=True)
class StaleLease:
    call_control_id: str
    call_id: UUID
    tenant_id: str
    agent_id: str
    token_hash: bytes = field(repr=False)
    previous_state: Literal["pending", "active"]
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class OutboxItem:
    queue_id: int
    operation: VoiceOperationV1 = field(repr=False)
    created_at: datetime
    claim_attempt: int
    next_attempt_at: datetime
    claim_expires_at: datetime
    last_error_code: str | None


@dataclass(frozen=True, slots=True)
class RelayClaimResult:
    applied: bool


@dataclass(frozen=True, slots=True)
class CleanupResult:
    receipts: int
    leases: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CommandSerializationError("datetime_must_be_aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CommandSerializationError("stored_datetime_invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CommandSerializationError("stored_datetime_invalid")
    return parsed.astimezone(UTC)


class PersistenceWriter:
    """Own one queue, one coroutine, and one aiosqlite connection."""

    def __init__(
        self,
        database_path: Path,
        keyring: CryptoKeyring,
        *,
        fatal_handler: Callable[[FatalPersistenceFault], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = _utc_now,
        control_commit_timeout_seconds: float = CONTROL_COMMIT_TIMEOUT_SECONDS,
        quick_check_interval_seconds: float = 30.0,
        quick_check_observer: Callable[[float], None] | None = None,
        quick_check_result: Callable[[], str | None] | None = None,
        file_size: Callable[[Path], int] = _default_file_size,
        max_storage_bytes: int = MAX_STORAGE_BYTES,
        failpoint: Failpoint | None = None,
    ) -> None:
        if control_commit_timeout_seconds <= 0 or quick_check_interval_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if max_storage_bytes <= 0:
            raise ValueError("max_storage_bytes must be positive")
        self._database_path = Path(database_path)
        self._keyring = keyring
        self._fatal_handler = fatal_handler
        self._monotonic = monotonic
        self._utcnow = utcnow
        self.control_commit_timeout_seconds = control_commit_timeout_seconds
        self._quick_check_interval_seconds = quick_check_interval_seconds
        self._quick_check_observer = quick_check_observer
        self._quick_check_result = quick_check_result
        self._file_size = file_size
        self.max_storage_bytes = max_storage_bytes
        self._failpoint = failpoint

        self._queue: asyncio.Queue[PersistenceCommand] = asyncio.Queue(
            maxsize=PERSISTENCE_QUEUE_MAX_ITEMS
        )
        self._pending_commands: deque[PersistenceCommand] = deque()
        self._queue_watchdog_wakeup = asyncio.Event()
        self._queue_watchdog_task: asyncio.Task[None] | None = None
        self._accepting = True
        self._degraded = False
        self._run_started = False
        self._stop_requested = False
        self._connection: aiosqlite.Connection | None = None
        self._owner_task: asyncio.Task[object] | None = None
        self._ready_event = asyncio.Event()
        self._ready_ok = False
        self._closed_event = asyncio.Event()
        self.fatal_event = asyncio.Event()
        self.fatal_fault: FatalPersistenceFault | None = None
        self.fatal_exception: FatalPersistenceError | None = None
        self.transcript_loss_count = 0
        self._last_quick_check = False
        self._last_check_at = float("-inf")
        self._stale_leases: list[StaleLease] = []
        self.pragma_state: dict[str, int | str] = {}

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    def _new_safe_error(
        self, code: str, source: BaseException | None = None
    ) -> FatalPersistenceError:
        if isinstance(source, CommandConflictError):
            return CommandConflictError(code)
        return FatalPersistenceError(code)

    def _signal_fatal(
        self,
        code: str,
        *,
        source: BaseException | None = None,
    ) -> FatalPersistenceError:
        safe_error = self._new_safe_error(code, source)
        self._accepting = False
        self._degraded = True
        if self.fatal_fault is None:
            self.fatal_fault = FatalPersistenceFault(code)
            self.fatal_exception = safe_error
            self.fatal_event.set()
            if self._fatal_handler is not None:
                with contextlib.suppress(Exception):
                    self._fatal_handler(self.fatal_fault)
        return safe_error

    def try_enqueue_turn(self, operation: VoiceOperationV1) -> bool:
        if operation.kind != "turn.upsert":
            raise ValueError("try_enqueue_turn requires turn.upsert")
        if not self._accepting or self._degraded:
            self.transcript_loss_count += 1
            self._signal_fatal("persistence_degraded")
            return False
        command = PersistenceCommand(
            "outbox",
            {"operation": operation},
            None,
            enqueued_at=self._monotonic(),
        )
        try:
            self._queue.put_nowait(command)
        except asyncio.QueueFull:
            self.transcript_loss_count += 1
            self._signal_fatal("queue_full")
            return False
        self._track_pending(command)
        return True

    async def commit_control(self, command: PersistenceCommand) -> None:
        if command.kind == "shutdown":
            raise ValueError("shutdown is internal")
        if command.committed is not None:
            raise CommandSerializationError("command_future_must_be_unset")
        if not self._accepting or self._degraded:
            raise self._new_safe_error(
                self.fatal_fault.code if self.fatal_fault else "persistence_degraded"
            )
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        queued = replace(command, committed=future, enqueued_at=self._monotonic())
        try:
            self._queue.put_nowait(queued)
        except asyncio.QueueFull as error:
            safe_error = self._signal_fatal("queue_full", source=error)
            if not future.done():
                future.set_exception(safe_error)
            raise safe_error from None
        self._track_pending(queued)
        try:
            await asyncio.wait_for(
                asyncio.shield(future), timeout=self.control_commit_timeout_seconds
            )
        except asyncio.CancelledError:
            future.add_done_callback(self._consume_control_commit_exception)
            raise
        except TimeoutError:
            raise self._signal_fatal("control_commit_timeout") from None

    @staticmethod
    def _consume_control_commit_exception(result: asyncio.Future[None]) -> None:
        if not result.cancelled():
            result.exception()

    async def quick_check(self) -> bool:
        return self._last_quick_check

    async def wait_ready(self) -> bool:
        await self._ready_event.wait()
        return self._ready_ok

    async def wait_until_idle(self) -> None:
        await self._queue.join()

    def take_stale_leases(self) -> tuple[StaleLease, ...]:
        stale = tuple(self._stale_leases)
        self._stale_leases.clear()
        return stale

    async def commit_lease(
        self,
        *,
        call_control_id: str,
        call_id: UUID,
        tenant_id: str,
        agent_id: str,
        state: Literal["pending", "active", "terminal"],
        token_hash: bytes,
        created_at: datetime,
        expires_at: datetime,
        closed_at: datetime | None,
    ) -> None:
        await self.commit_control(
            PersistenceCommand(
                "lease",
                {
                    "action": "upsert",
                    "call_control_id": call_control_id,
                    "call_id": call_id,
                    "tenant_id": tenant_id,
                    "agent_id": agent_id,
                    "state": state,
                    "token_hash": token_hash,
                    "created_at": created_at,
                    "expires_at": expires_at,
                    "closed_at": closed_at,
                },
                None,
            )
        )

    async def read_relay_batch(
        self, *, batch_size: int, now: datetime, lease_seconds: int
    ) -> tuple[OutboxItem, ...]:
        if type(batch_size) is not int or batch_size <= 0 or batch_size > 1000:
            raise ValueError("batch_size is outside the supported range")
        if (
            type(lease_seconds) is not int
            or lease_seconds <= 0
            or lease_seconds > RELAY_CLAIM_MAX_SECONDS
        ):
            raise ValueError("lease_seconds is outside the supported range")
        result: asyncio.Future[tuple[OutboxItem, ...]] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "relay_batch",
                    {
                        "action": "read",
                        "batch_size": batch_size,
                        "now": now,
                        "lease_seconds": lease_seconds,
                        "result": result,
                    },
                    None,
                )
            )
        except BaseException:
            if result.done() and not result.cancelled():
                result.exception()
            raise
        return await result

    async def oldest_outbox_created_at(self) -> datetime | None:
        result: asyncio.Future[datetime | None] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "relay_batch",
                    {"action": "oldest_created_at", "result": result},
                    None,
                )
            )
        except asyncio.CancelledError:
            result.add_done_callback(self._consume_relay_result_exception)
            raise
        except BaseException:
            if result.done() and not result.cancelled():
                result.exception()
            raise
        return await result

    @staticmethod
    def _consume_relay_result_exception(
        result: asyncio.Future[datetime | None],
    ) -> None:
        if not result.cancelled():
            result.exception()

    async def ack_outbox(
        self,
        *,
        queue_id: int,
        expected_claim_attempt: int,
    ) -> RelayClaimResult:
        expected_claim = self._validate_expected_claim_attempt(expected_claim_attempt)
        return await self._commit_relay_claim_mutation(
            {
                "action": "ack",
                "queue_id": queue_id,
                "expected_claim_attempt": expected_claim,
            }
        )

    async def retry_outbox(
        self,
        *,
        queue_id: int,
        expected_claim_attempt: int,
        next_attempt_at: datetime,
        error_code: str,
    ) -> RelayClaimResult:
        if _SAFE_ERROR_CODE.fullmatch(error_code) is None:
            raise ValueError("error_code must be a bounded safe code")
        expected_claim = self._validate_expected_claim_attempt(expected_claim_attempt)
        return await self._commit_relay_claim_mutation(
            {
                "action": "retry",
                "queue_id": queue_id,
                "expected_claim_attempt": expected_claim,
                "next_attempt_at": next_attempt_at,
                "error_code": error_code,
            }
        )

    @staticmethod
    def _validate_expected_claim_attempt(value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("expected_claim_attempt must be a positive exact integer")
        return value

    async def _commit_relay_claim_mutation(
        self, payload: dict[str, object]
    ) -> RelayClaimResult:
        result: asyncio.Future[RelayClaimResult] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand("relay_batch", {**payload, "result": result}, None)
            )
        except BaseException:
            if result.done() and not result.cancelled():
                result.exception()
            raise
        return await result

    async def cleanup_local_state(self, *, now: datetime) -> CleanupResult:
        result: asyncio.Future[CleanupResult] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "relay_batch", {"action": "cleanup", "now": now, "result": result}, None
                )
            )
        except BaseException:
            if result.done() and not result.cancelled():
                result.exception()
            raise
        return await result

    async def drain(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._accepting = False
        if not self._run_started or self._closed_event.is_set():
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        command = PersistenceCommand(
            "shutdown", {}, future, enqueued_at=self._monotonic()
        )
        async with asyncio.timeout(timeout_seconds):
            await self._queue.put(command)
            self._track_pending(command)
            await asyncio.shield(future)
            await self._queue.join()
            await self._closed_event.wait()

    async def run(self) -> None:
        if self._run_started:
            raise RuntimeError("writer_run_already_started")
        self._run_started = True
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("writer_requires_asyncio_task")
        self._owner_task = cast(asyncio.Task[object], current)
        current_command: PersistenceCommand | None = None
        current_owned = False
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = await aiosqlite.connect(self._database_path)
            await self._initialize_owner_connection()
            await self._discover_stale_leases()
            if not await self._perform_quick_check():
                raise FatalPersistenceError("quick_check_failed")
            self._check_storage_limit()
            self._queue_watchdog_task = asyncio.create_task(
                self._watch_queue_age(),
                name="voice-persistence-queue-watchdog",
            )
            self._ready_ok = True
            self._ready_event.set()

            while True:
                wait_timeout = max(
                    0.0,
                    self._quick_check_interval_seconds
                    - (self._monotonic() - self._last_check_at),
                )
                if wait_timeout == 0.0:
                    await self._run_periodic_check_if_due()
                    continue
                try:
                    current_command = await asyncio.wait_for(
                        self._queue.get(), timeout=wait_timeout
                    )
                    current_owned = True
                    self._untrack_pending(current_command)
                except TimeoutError:
                    await self._run_periodic_check_if_due()
                    continue

                if (
                    current_command.kind != "shutdown"
                    and self._monotonic() - current_command.enqueued_at
                    > QUEUE_OLDEST_LIMIT_SECONDS
                ):
                    raise FatalPersistenceError("queue_oldest_age_exceeded")

                should_stop = await self._process_command(current_command)
                if should_stop:
                    self._resolve_success(current_command)
                    self._queue.task_done()
                    current_owned = False
                    current_command = None
                    break
                await self._run_periodic_check_if_due()
                self._resolve_success(current_command)
                self._queue.task_done()
                current_owned = False
                current_command = None
        except BaseException as error:
            safe_error = self._classify_error(error)
            self._signal_fatal(str(safe_error), source=error)
            if current_owned and current_command is not None:
                self._resolve_failure(current_command, safe_error)
                self._count_lost_command(current_command)
                self._queue.task_done()
            self._fail_pending(safe_error)
        finally:
            self._ready_event.set()
            if self._queue_watchdog_task is not None:
                self._queue_watchdog_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._queue_watchdog_task
                self._queue_watchdog_task = None
            if self._connection is not None:
                try:
                    await self._connection.close()
                except Exception:
                    self._signal_fatal("sqlite_close_failed")
                self._connection = None
            self._closed_event.set()

    def _track_pending(self, command: PersistenceCommand) -> None:
        self._pending_commands.append(command)
        self._queue_watchdog_wakeup.set()

    def _untrack_pending(self, command: PersistenceCommand) -> None:
        if not self._pending_commands or self._pending_commands[0] is not command:
            raise FatalPersistenceError("queue_tracking_mismatch")
        self._pending_commands.popleft()
        self._queue_watchdog_wakeup.set()

    async def _watch_queue_age(self) -> None:
        try:
            while True:
                oldest = next(
                    (
                        command
                        for command in self._pending_commands
                        if command.kind != "shutdown"
                    ),
                    None,
                )
                if oldest is None:
                    self._queue_watchdog_wakeup.clear()
                    await self._queue_watchdog_wakeup.wait()
                    continue

                remaining = QUEUE_OLDEST_LIMIT_SECONDS - (
                    self._monotonic() - oldest.enqueued_at
                )
                if remaining < 0.0:
                    self._signal_fatal("queue_oldest_age_exceeded")
                    return

                self._queue_watchdog_wakeup.clear()
                try:
                    await asyncio.wait_for(
                        self._queue_watchdog_wakeup.wait(), timeout=remaining
                    )
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._signal_fatal("queue_watchdog_failed", source=error)

    def _require_owner_connection(self) -> aiosqlite.Connection:
        if asyncio.current_task() is not self._owner_task or self._connection is None:
            raise FatalPersistenceError("writer_owner_violation")
        return self._connection

    async def _initialize_owner_connection(self) -> None:
        connection = self._require_owner_connection()
        cursor = await connection.execute("PRAGMA journal_mode=DELETE")
        journal_row = await cursor.fetchone()
        await cursor.close()
        await connection.execute("PRAGMA synchronous=EXTRA")
        await connection.execute("PRAGMA foreign_keys=ON")
        existing_version = self._pragma_int(await self._pragma_scalar("user_version"))
        existing_schema = await self._application_schema_objects()
        if existing_schema:
            if (
                existing_version != SCHEMA_VERSION
                or existing_schema != _EXPECTED_SCHEMA_OBJECTS
            ):
                raise FatalPersistenceError("sqlite_schema_mismatch")
        else:
            if existing_version != 0:
                raise FatalPersistenceError("sqlite_schema_mismatch")
            await connection.executescript(SCHEMA_SQL)
            await connection.commit()
            if (
                self._pragma_int(await self._pragma_scalar("user_version")) != SCHEMA_VERSION
                or await self._application_schema_objects() != _EXPECTED_SCHEMA_OBJECTS
            ):
                raise FatalPersistenceError("sqlite_schema_mismatch")
        synchronous = self._pragma_int(await self._pragma_scalar("synchronous"))
        foreign_keys = self._pragma_int(await self._pragma_scalar("foreign_keys"))
        journal_mode = str(journal_row[0]).lower() if journal_row else ""
        self.pragma_state = {
            "journal_mode": journal_mode,
            "synchronous": synchronous,
            "foreign_keys": foreign_keys,
        }
        if self.pragma_state != {
            "journal_mode": "delete",
            "synchronous": 3,
            "foreign_keys": 1,
        }:
            raise FatalPersistenceError("sqlite_pragma_mismatch")

    async def _application_schema_objects(self) -> dict[tuple[str, str], str]:
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
              AND type IN ('table', 'index', 'view', 'trigger')
            ORDER BY type, name
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        objects: dict[tuple[str, str], str] = {}
        for object_type, name, sql in rows:
            if not isinstance(object_type, str) or not isinstance(name, str) or not isinstance(
                sql, str
            ):
                raise FatalPersistenceError("sqlite_schema_mismatch")
            objects[(object_type, name)] = _normalize_schema_sql(sql)
        return objects

    async def _pragma_scalar(self, name: str) -> object:
        connection = self._require_owner_connection()
        cursor = await connection.execute(f"PRAGMA {name}")
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise FatalPersistenceError("sqlite_pragma_missing")
        return row[0]

    @staticmethod
    def _pragma_int(value: object) -> int:
        if type(value) is not int:
            raise FatalPersistenceError("sqlite_pragma_mismatch")
        return value

    async def _perform_quick_check(self) -> bool:
        self._require_owner_connection()
        override = self._quick_check_result() if self._quick_check_result else None
        if override is None:
            connection = self._require_owner_connection()
            cursor = await connection.execute("PRAGMA quick_check")
            rows = await cursor.fetchall()
            await cursor.close()
            result = tuple(str(row[0]).lower() for row in rows)
        else:
            result = (override.lower(),)
        self._last_quick_check = result == ("ok",)
        self._last_check_at = self._monotonic()
        if self._quick_check_observer is not None:
            self._quick_check_observer(self._last_check_at)
        return self._last_quick_check

    async def _run_periodic_check_if_due(self) -> None:
        if self._monotonic() - self._last_check_at < self._quick_check_interval_seconds:
            return
        if not await self._perform_quick_check():
            raise FatalPersistenceError("quick_check_failed")
        self._check_storage_limit()

    def _check_storage_limit(self) -> None:
        journal_path = Path(f"{self._database_path}-journal")
        try:
            total = self._file_size(self._database_path) + self._file_size(journal_path)
        except OSError as error:
            raise error
        if total > self.max_storage_bytes:
            raise FatalPersistenceError("storage_limit_exceeded")

    async def _call_failpoint(self, name: str) -> None:
        if self._failpoint is None:
            return
        result = self._failpoint(name)
        if inspect.isawaitable(result):
            await result

    async def _process_command(self, command: PersistenceCommand) -> bool:
        if command.kind == "shutdown":
            return True

        connection = self._require_owner_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            if command.kind == "outbox":
                await self._insert_outbox(require_operation(command.payload))
            elif command.kind == "lease":
                await self._apply_lease(command.payload)
            elif command.kind == "webhook_effect":
                await self._apply_webhook_effect(command.payload)
            elif command.kind == "webhook_enrichment":
                await self._apply_webhook_enrichment(command.payload)
            elif command.kind == "relay_batch":
                await self._apply_relay_command(command.payload)
            else:
                raise CommandSerializationError("unknown_persistence_command")
            await self._call_failpoint("after_mutation_before_commit")
            self._check_storage_limit()
            await connection.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                await connection.rollback()
            raise
        return False

    async def _insert_outbox(self, operation: VoiceOperationV1) -> None:
        connection = self._require_owner_connection()
        prepared = encrypt_operation(operation, self._keyring)
        created_at = self._utcnow()
        turn_id: str | None = None
        recording_id: str | None = None
        if isinstance(operation.payload, TurnUpsertPayloadV1):
            turn_id = str(operation.payload.turn_id)
        elif isinstance(operation.payload, RecordingUpsertPayloadV1):
            recording_id = str(operation.payload.recording_id)
        try:
            await connection.execute(
                """
                INSERT INTO outbox (
                    op_id, deployment_id, kind, schema_version, call_id, turn_id,
                    recording_id, crypto_version, key_version, nonce, ciphertext,
                    created_at, attempts, next_attempt_at, last_error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL)
                """,
                (
                    str(operation.operation_id),
                    operation.deployment_id,
                    operation.kind,
                    operation.schema_version,
                    str(operation.call_id),
                    turn_id,
                    recording_id,
                    CRYPTO_VERSION,
                    prepared.encrypted.key_version,
                    prepared.encrypted.nonce,
                    prepared.encrypted.ciphertext,
                    _iso(created_at),
                    _iso(created_at),
                ),
            )
        except sqlite3.IntegrityError:
            existing = await self._load_operation_by_id(str(operation.operation_id))
            if existing is None or canonical_operation_bytes(existing) != prepared.plaintext:
                raise CommandConflictError("operation_identity_conflict") from None

    async def _load_operation_by_id(self, operation_id: str) -> VoiceOperationV1 | None:
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            """
            SELECT schema_version, op_id, deployment_id, call_id, kind,
                   key_version, nonce, ciphertext
            FROM outbox WHERE op_id = ?
            """,
            (operation_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        metadata = {
            "schema_version": row[0],
            "operation_id": row[1],
            "deployment_id": row[2],
            "call_id": row[3],
            "kind": row[4],
        }
        plaintext = self._keyring.decrypt(
            EncryptedValue(key_version=row[5], nonce=row[6], ciphertext=row[7]),
            aad=operation_aad_from_metadata(metadata),
        )
        return decode_operation(plaintext)

    async def _apply_webhook_effect(self, payload: Mapping[str, object]) -> None:
        receipt = payload.get("receipt")
        if not isinstance(receipt, Mapping):
            raise CommandSerializationError("invalid_webhook_receipt")
        is_new = await self._insert_receipt(receipt)
        if not is_new:
            return
        lease = payload.get("lease")
        operation = payload.get("operation")
        if lease is not None:
            if not isinstance(lease, Mapping):
                raise CommandSerializationError("invalid_lease_command")
            await self._apply_lease(lease)
        if operation is not None:
            if not isinstance(operation, VoiceOperationV1):
                raise CommandSerializationError("invalid_outbox_command")
            await self._insert_outbox(operation)

    async def _insert_receipt(self, receipt: Mapping[str, object]) -> bool:
        connection = self._require_owner_connection()
        try:
            event_id = self._required_str(receipt, "event_id")
            event_type = self._required_str(receipt, "event_type")
            call_control = receipt.get("call_control_id")
            if call_control is not None and not isinstance(call_control, str):
                raise CommandSerializationError("invalid_webhook_receipt")
            occurred_at = _iso(self._required_datetime(receipt, "occurred_at"))
            received_at = _iso(self._required_datetime(receipt, "received_at"))
            semantic_fingerprint = receipt.get("semantic_fingerprint_sha256")
            if type(semantic_fingerprint) is not bytes or len(semantic_fingerprint) != 32:
                raise CommandSerializationError("invalid_webhook_receipt")
        except (KeyError, TypeError) as error:
            raise CommandSerializationError("invalid_webhook_receipt") from error
        cursor = await connection.execute(
            """
            INSERT OR IGNORE INTO webhook_receipts (
                event_id, event_type, call_control_id, occurred_at, received_at,
                semantic_fingerprint_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                call_control,
                occurred_at,
                received_at,
                semantic_fingerprint,
            ),
        )
        inserted = cursor.rowcount == 1
        await cursor.close()
        if inserted:
            return True
        cursor = await connection.execute(
            """
            SELECT event_type, call_control_id, occurred_at, semantic_fingerprint_sha256
            FROM webhook_receipts WHERE event_id = ?
            """,
            (event_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row != (event_type, call_control, occurred_at, semantic_fingerprint):
            raise CommandConflictError("webhook_identity_conflict")
        return False

    async def _apply_webhook_enrichment(self, payload: Mapping[str, object]) -> None:
        receipt = payload.get("receipt")
        operation = payload.get("operation")
        enrichment_fingerprint = payload.get("enrichment_fingerprint_sha256")
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(operation, VoiceOperationV1)
            or not isinstance(operation.payload, RecordingUpsertPayloadV1)
            or operation.payload.status != "saved"
            or operation.payload.telnyx_recording_id is None
            or type(enrichment_fingerprint) is not bytes
            or len(enrichment_fingerprint) != 32
            or enrichment_fingerprint
            != hashlib.sha256(canonical_operation_bytes(operation)).digest()
        ):
            raise CommandSerializationError("invalid_webhook_enrichment")
        event_id = self._required_str(receipt, "event_id")
        event_type = self._required_str(receipt, "event_type")
        call_control = receipt.get("call_control_id")
        if call_control is not None and not isinstance(call_control, str):
            raise CommandSerializationError("invalid_webhook_enrichment")
        occurred_at = _iso(self._required_datetime(receipt, "occurred_at"))
        semantic_fingerprint = receipt.get("semantic_fingerprint_sha256")
        if type(semantic_fingerprint) is not bytes or len(semantic_fingerprint) != 32:
            raise CommandSerializationError("invalid_webhook_enrichment")
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            """
            SELECT event_type, call_control_id, occurred_at,
                   semantic_fingerprint_sha256,
                   provider_enrichment_fingerprint_sha256
            FROM webhook_receipts WHERE event_id = ?
            """,
            (event_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None or row[:4] != (
            event_type,
            call_control,
            occurred_at,
            semantic_fingerprint,
        ):
            raise CommandConflictError("webhook_identity_conflict")
        bound_fingerprint = row[4]
        if bound_fingerprint is None:
            cursor = await connection.execute(
                """
                UPDATE webhook_receipts
                SET provider_enrichment_fingerprint_sha256 = ?
                WHERE event_id = ?
                  AND provider_enrichment_fingerprint_sha256 IS NULL
                """,
                (enrichment_fingerprint, event_id),
            )
            if cursor.rowcount != 1:
                await cursor.close()
                raise CommandConflictError("webhook_enrichment_conflict")
            await cursor.close()
        elif bound_fingerprint != enrichment_fingerprint:
            raise CommandConflictError("webhook_enrichment_conflict")
        await self._insert_outbox(operation)

    async def _apply_lease(self, payload: Mapping[str, object]) -> None:
        if payload.get("action") != "upsert":
            raise CommandSerializationError("invalid_lease_command")
        connection = self._require_owner_connection()
        call_control_id = self._required_str(payload, "call_control_id")
        call_id = self._required_uuid(payload, "call_id")
        tenant_id = self._required_str(payload, "tenant_id")
        agent_id = self._required_str(payload, "agent_id")
        state = payload.get("state")
        token_hash = payload.get("token_hash")
        created_at = self._required_datetime(payload, "created_at")
        expires_at = self._required_datetime(payload, "expires_at")
        closed_value = payload.get("closed_at")
        if (
            state not in {"pending", "active", "terminal"}
            or type(token_hash) is not bytes
            or len(token_hash) != 32
        ):
            raise CommandSerializationError("invalid_lease_command")
        if expires_at <= created_at:
            raise CommandSerializationError("invalid_lease_command")
        if closed_value is not None and not isinstance(closed_value, datetime):
            raise CommandSerializationError("invalid_lease_command")
        closed_at = closed_value
        if (state == "terminal") != (closed_at is not None):
            raise CommandSerializationError("invalid_lease_command")

        cursor = await connection.execute(
            """
            SELECT call_id, tenant_id, agent_id, state, token_hash,
                   created_at, expires_at, closed_at
            FROM call_leases WHERE call_control_id = ?
            """,
            (call_control_id,),
        )
        existing = await cursor.fetchone()
        await cursor.close()
        values = (
            str(call_id),
            tenant_id,
            agent_id,
            state,
            token_hash,
            _iso(created_at),
            _iso(expires_at),
            _iso(closed_at) if closed_at else None,
        )
        if existing is None:
            if state != "pending":
                raise CommandConflictError("lease_transition_conflict")
            await connection.execute(
                """
                INSERT INTO call_leases (
                    call_control_id, call_id, tenant_id, agent_id, state, token_hash,
                    created_at, expires_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (call_control_id, *values),
            )
            return

        old_state = existing[3]
        identities_match = existing[:3] == values[:3] and existing[4] == token_hash
        allowed = old_state == state or (old_state, state) in {
            ("pending", "active"),
            ("pending", "terminal"),
            ("active", "terminal"),
        }
        if not identities_match or not allowed:
            raise CommandConflictError("lease_transition_conflict")
        if old_state == state:
            if tuple(existing) != values:
                raise CommandConflictError("lease_transition_conflict")
            return
        await connection.execute(
            """
            UPDATE call_leases
            SET state = ?, expires_at = ?, closed_at = ?
            WHERE call_control_id = ?
            """,
            (state, _iso(expires_at), _iso(closed_at) if closed_at else None, call_control_id),
        )

    async def _discover_stale_leases(self) -> None:
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            """
            SELECT call_control_id, call_id, tenant_id, agent_id, token_hash, state,
                   created_at, expires_at
            FROM call_leases
            WHERE state IN ('pending', 'active')
            ORDER BY created_at, call_control_id
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        self._stale_leases.extend(
            StaleLease(
                call_control_id=row[0],
                call_id=UUID(row[1]),
                tenant_id=row[2],
                agent_id=row[3],
                token_hash=row[4],
                previous_state=cast(Literal["pending", "active"], row[5]),
                created_at=_parse_datetime(row[6]),
                expires_at=_parse_datetime(row[7]),
            )
            for row in rows
        )

    async def _claim_batch(self, payload: Mapping[str, object]) -> None:
        connection = self._require_owner_connection()
        batch_size = payload.get("batch_size")
        lease_seconds = payload.get("lease_seconds")
        result = payload.get("result")
        if (
            type(batch_size) is not int
            or type(lease_seconds) is not int
            or lease_seconds <= 0
            or lease_seconds > RELAY_CLAIM_MAX_SECONDS
            or not isinstance(result, asyncio.Future)
        ):
            raise CommandSerializationError("invalid_relay_read")
        now = self._required_datetime(payload, "now")
        claim_expires_at = now + timedelta(seconds=lease_seconds)
        cursor = await connection.execute(
            """
            SELECT queue_id, op_id, deployment_id, kind, schema_version, call_id,
                   key_version, nonce, ciphertext, created_at, attempts,
                   next_attempt_at, last_error_code
            FROM outbox ORDER BY deployment_id, queue_id
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()

        selected: list[OutboxItem] = []
        blocked_deployments: set[str] = set()
        now_iso = _iso(now)
        for row in rows:
            deployment_id = row[2]
            if deployment_id in blocked_deployments:
                continue
            if row[11] > now_iso:
                blocked_deployments.add(deployment_id)
                continue
            metadata = {
                "schema_version": row[4],
                "operation_id": row[1],
                "deployment_id": deployment_id,
                "call_id": row[5],
                "kind": row[3],
            }
            plaintext = self._keyring.decrypt(
                EncryptedValue(key_version=row[6], nonce=row[7], ciphertext=row[8]),
                aad=operation_aad_from_metadata(metadata),
            )
            selected.append(
                OutboxItem(
                    queue_id=row[0],
                    operation=decode_operation(plaintext),
                    created_at=_parse_datetime(row[9]),
                    claim_attempt=row[10] + 1,
                    next_attempt_at=_parse_datetime(row[11]),
                    claim_expires_at=claim_expires_at,
                    last_error_code=row[12],
                )
            )
            if len(selected) >= batch_size:
                break
        for item in selected:
            update_cursor = await connection.execute(
                """
                UPDATE outbox
                SET attempts = attempts + 1, next_attempt_at = ?
                WHERE queue_id = ? AND attempts = ?
                """,
                (_iso(claim_expires_at), item.queue_id, item.claim_attempt - 1),
            )
            if update_cursor.rowcount != 1:
                await update_cursor.close()
                raise CommandConflictError("outbox_claim_conflict")
            await update_cursor.close()
        if not result.done():
            result.set_result(tuple(selected))

    async def _apply_relay_command(self, payload: Mapping[str, object]) -> None:
        connection = self._require_owner_connection()
        action = payload.get("action")
        if action == "read":
            await self._claim_batch(payload)
        elif action == "oldest_created_at":
            result = payload.get("result")
            if not isinstance(result, asyncio.Future):
                raise CommandSerializationError("invalid_oldest_created_at_command")
            cursor = await connection.execute("SELECT MIN(created_at) FROM outbox")
            row = await cursor.fetchone()
            await cursor.close()
            if row is None or len(row) != 1:
                raise CommandSerializationError("stored_datetime_invalid")
            stored = row[0]
            if stored is not None and not isinstance(stored, str):
                safe_error = self._signal_fatal("stored_datetime_invalid")
                if not result.done():
                    result.set_exception(safe_error)
                return
            try:
                oldest = None if stored is None else _parse_datetime(stored)
            except CommandSerializationError:
                safe_error = self._signal_fatal("stored_datetime_invalid")
                if not result.done():
                    result.set_exception(safe_error)
                return
            if not result.done():
                result.set_result(oldest)
        elif action == "ack":
            queue_id = self._required_positive_int(payload, "queue_id")
            expected_claim = self._required_positive_int(payload, "expected_claim_attempt")
            result = payload.get("result")
            if not isinstance(result, asyncio.Future):
                raise CommandSerializationError("invalid_relay_ack")
            cursor = await connection.execute(
                "DELETE FROM outbox WHERE queue_id = ? AND attempts = ?",
                (queue_id, expected_claim),
            )
            applied = cursor.rowcount == 1
            await cursor.close()
            if not result.done():
                result.set_result(RelayClaimResult(applied=applied))
        elif action == "retry":
            queue_id = self._required_positive_int(payload, "queue_id")
            expected_claim = self._required_positive_int(payload, "expected_claim_attempt")
            next_attempt = self._required_datetime(payload, "next_attempt_at")
            error_code = self._required_str(payload, "error_code")
            result = payload.get("result")
            if _SAFE_ERROR_CODE.fullmatch(error_code) is None or not isinstance(
                result, asyncio.Future
            ):
                raise CommandSerializationError("invalid_relay_retry")
            cursor = await connection.execute(
                """
                UPDATE outbox
                SET next_attempt_at = ?, last_error_code = ?
                WHERE queue_id = ? AND attempts = ?
                """,
                (
                    _iso(next_attempt),
                    error_code,
                    queue_id,
                    expected_claim,
                ),
            )
            applied = cursor.rowcount == 1
            await cursor.close()
            if not result.done():
                result.set_result(RelayClaimResult(applied=applied))
        elif action == "cleanup":
            now = self._required_datetime(payload, "now")
            result = payload.get("result")
            if not isinstance(result, asyncio.Future):
                raise CommandSerializationError("invalid_cleanup_command")
            receipts_cursor = await connection.execute(
                "DELETE FROM webhook_receipts WHERE received_at < ?",
                (_iso(now - RECEIPT_RETENTION),),
            )
            leases_cursor = await connection.execute(
                "DELETE FROM call_leases WHERE state = 'terminal' AND closed_at < ?",
                (_iso(now - CLOSED_LEASE_RETENTION),),
            )
            cleanup = CleanupResult(receipts_cursor.rowcount, leases_cursor.rowcount)
            await receipts_cursor.close()
            await leases_cursor.close()
            if not result.done():
                result.set_result(cleanup)
        else:
            raise CommandSerializationError("invalid_relay_command")

    @staticmethod
    def _required_str(payload: Mapping[str, object], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise CommandSerializationError("invalid_command_payload")
        return value

    @staticmethod
    def _required_uuid(payload: Mapping[str, object], name: str) -> UUID:
        value = payload.get(name)
        if not isinstance(value, UUID):
            raise CommandSerializationError("invalid_command_payload")
        return value

    @staticmethod
    def _required_datetime(payload: Mapping[str, object], name: str) -> datetime:
        value = payload.get(name)
        if not isinstance(value, datetime):
            raise CommandSerializationError("invalid_command_payload")
        if value.tzinfo is None or value.utcoffset() is None:
            raise CommandSerializationError("invalid_command_payload")
        return value.astimezone(UTC)

    @staticmethod
    def _required_positive_int(payload: Mapping[str, object], name: str) -> int:
        value = payload.get(name)
        if type(value) is not int or value <= 0:
            raise CommandSerializationError("invalid_command_payload")
        return value

    def _resolve_success(self, command: PersistenceCommand) -> None:
        if command.committed is not None and not command.committed.done():
            command.committed.set_result(None)

    def _resolve_failure(
        self, command: PersistenceCommand, error: FatalPersistenceError
    ) -> None:
        result = command.payload.get("result")
        if isinstance(result, asyncio.Future) and not result.done():
            result.set_exception(error)
        if command.committed is not None and not command.committed.done():
            command.committed.set_exception(error)

    def _count_lost_command(self, command: PersistenceCommand) -> None:
        if command.kind == "outbox":
            self.transcript_loss_count += 1

    def _fail_pending(self, error: FatalPersistenceError) -> None:
        while True:
            try:
                command = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._untrack_pending(command)
            self._resolve_failure(command, error)
            self._count_lost_command(command)
            self._queue.task_done()

    def _classify_error(self, error: BaseException) -> FatalPersistenceError:
        if isinstance(error, FatalPersistenceError):
            return error
        if isinstance(error, UnknownKeyVersionError):
            return FatalPersistenceError("unknown_key_version")
        if isinstance(error, CryptoError):
            return FatalPersistenceError("crypto_failed")
        if isinstance(error, sqlite3.Error):
            code = getattr(error, "sqlite_errorcode", None)
            primary = code & 0xFF if isinstance(code, int) else None
            if primary == sqlite3.SQLITE_FULL:
                return FatalPersistenceError("sqlite_full")
            if primary == sqlite3.SQLITE_CORRUPT:
                return FatalPersistenceError("sqlite_corrupt")
            if primary == sqlite3.SQLITE_IOERR:
                return FatalPersistenceError("sqlite_ioerr")
            message = str(error).lower()
            if "not a database" in message or "malformed" in message:
                return FatalPersistenceError("sqlite_corrupt")
            return FatalPersistenceError("sqlite_failed")
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            return FatalPersistenceError("storage_enospc")
        if isinstance(error, PersistenceError):
            return FatalPersistenceError(str(error))
        if isinstance(error, Exception):
            return FatalPersistenceError("persistence_command_failed")
        return FatalPersistenceError("writer_task_died")
