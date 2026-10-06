"""Single-owner bounded SQLite command writer."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import hashlib
import inspect
import json
import math
import re
import sqlite3
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid5

import aiosqlite

from projetv0_voice.audio_contract import (
    AudioChunkPayloadV2,
    AudioFinishPayloadV2,
    AudioRevokePayloadV2,
    BeginCallSnapshotV2,
    VoiceOperationV2,
)

if TYPE_CHECKING:
    from projetv0_voice.telnyx.recordings import RecordingCorrelationV1

from projetv0_voice.crypto import (
    CRYPTO_VERSION,
    CryptoError,
    CryptoKeyring,
    EncryptedValue,
    UnknownKeyVersionError,
)
from projetv0_voice.models import (
    BeginCallSnapshotV1,
    CallUpsertPayloadV1,
    DisclosureEvidenceV1,
    MessageResultV1,
    RecordingArchiveReceiptV1,
    RecordingUpsertPayloadV1,
    TurnUpsertPayloadV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.business_contract import encrypt_message_result, validate_turn_text
from projetv0_voice.persistence.business_result import (
    RetainedCall,
    RetainedTurn,
    validate_result_provenance,
)
from projetv0_voice.persistence.commands import (
    CommandConflictError,
    CommandSerializationError,
    FatalPersistenceError,
    PersistenceCommand,
    PersistenceError,
    PreparedAudioOperation,
    PreparedControlOperationV2,
    canonical_operation_bytes,
    decode_operation,
    decode_operation_v2,
    encrypt_audio_operation,
    encrypt_control_operation_v2,
    encrypt_operation,
    operation_aad,
    operation_aad_from_metadata,
    require_operation,
)
from projetv0_voice.persistence.schema import (
    CALL_LIFECYCLE_MIGRATION_SQL,
    LOCAL_AUDIO_CHOICE_MIGRATION_SQL,
    LOCAL_AUDIO_CHOICE_SCHEMA_SQL,
    LOCAL_AUDIO_CHOICE_SCHEMA_VERSION,
    LOCAL_AUDIO_SCHEMA_SQL,
    LOCAL_AUDIO_SCHEMA_VERSION,
    LOCAL_AUDIO_TERMINAL_MIGRATION_SQL,
    LOCAL_AUDIO_TERMINAL_SCHEMA_SQL,
    LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION,
    QUALIFICATION_RUNS_SQL,
    RECORDING_ARCHIVE_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    SPARRA_CONTENT_SQL,
    V1_SCHEMA_SQL,
    V2_SCHEMA_SQL,
    V3_SCHEMA_SQL,
    V4_SCHEMA_SQL,
)

PERSISTENCE_QUEUE_MAX_ITEMS = 256
CONTROL_COMMIT_TIMEOUT_SECONDS = 1.5
QUEUE_OLDEST_LIMIT_SECONDS = 1.0
MAX_STORAGE_BYTES = 268_435_456
RELAY_CLAIM_MAX_SECONDS = 300
RECEIPT_RETENTION = timedelta(days=7)
CLOSED_LEASE_RETENTION = timedelta(hours=24)
SPARRA_REPLAY_RETENTION = timedelta(seconds=900)
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


def _expected_schema_objects(schema_sql: str) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}
    for statement in schema_sql.split(";"):
        compact = " ".join(statement.split())
        matched = re.match(
            r"^CREATE (TABLE|INDEX)(?: IF NOT EXISTS)? ([A-Za-z_][A-Za-z0-9_]*)\b",
            compact,
            flags=re.IGNORECASE,
        )
        if matched is not None:
            expected[(matched.group(1).casefold(), matched.group(2))] = _normalize_schema_sql(
                statement
            )
    if not {
        ("table", "call_leases"),
        ("table", "webhook_receipts"),
        ("table", "outbox"),
        ("index", "outbox_due_fifo_idx"),
    }.issubset(expected):
        raise RuntimeError("invalid_expected_sqlite_schema")
    return expected


_EXPECTED_V1_SCHEMA_OBJECTS = _expected_schema_objects(V1_SCHEMA_SQL)
_EXPECTED_V2_SCHEMA_OBJECTS = _expected_schema_objects(V2_SCHEMA_SQL)
_EXPECTED_V3_SCHEMA_OBJECTS = _expected_schema_objects(V3_SCHEMA_SQL)
_EXPECTED_V4_SCHEMA_OBJECTS = _expected_schema_objects(V4_SCHEMA_SQL)
_EXPECTED_SCHEMA_OBJECTS = _expected_schema_objects(SCHEMA_SQL)
_EXPECTED_AUDIO_SCHEMA_OBJECTS = _expected_schema_objects(LOCAL_AUDIO_SCHEMA_SQL)
_EXPECTED_AUDIO_CHOICE_SCHEMA_OBJECTS = _expected_schema_objects(LOCAL_AUDIO_CHOICE_SCHEMA_SQL)
_EXPECTED_AUDIO_TERMINAL_SCHEMA_OBJECTS = _expected_schema_objects(LOCAL_AUDIO_TERMINAL_SCHEMA_SQL)
if set(_EXPECTED_V1_SCHEMA_OBJECTS) != {
    ("table", "call_leases"),
    ("table", "webhook_receipts"),
    ("table", "outbox"),
    ("index", "outbox_due_fifo_idx"),
} or set(_EXPECTED_SCHEMA_OBJECTS) != {
    *_EXPECTED_V1_SCHEMA_OBJECTS,
    ("table", "qualification_runs"),
    ("table", "sparra_turn_decisions"),
    ("table", "sparra_publications"),
    ("table", "sparra_content_fences"),
    ("table", "recording_archives"),
}:
    raise RuntimeError("invalid_expected_sqlite_schema")


@dataclass(frozen=True, slots=True)
class FatalPersistenceFault:
    code: str


@dataclass(frozen=True, slots=True)
class LocalCallAdmissionFacts:
    call_id: UUID
    admitted_at: datetime
    retention_until: datetime
    telnyx_call_leg_id: str | None
    telnyx_call_session_id: str | None
    admission_generation: UUID | None = None

    def __post_init__(self) -> None:
        if (
            self.admitted_at.tzinfo is None
            or self.admitted_at.utcoffset() is None
            or self.admitted_at.microsecond % 1000
            or self.retention_until != self.admitted_at + timedelta(days=30)
            or self.admission_generation is not None
            and not isinstance(self.admission_generation, UUID)
        ):
            raise ValueError("local_admission_facts_invalid")


@dataclass(frozen=True, slots=True, repr=False)
class LocalCallLifecycleFacts(LocalCallAdmissionFacts):
    started_at: datetime | None = None
    disclosure_evidence: DisclosureEvidenceV1 | None = None
    transfer_command_id: UUID | None = None
    transfer_correlation: str | None = None
    transfer_generation: UUID | None = None
    transfer_connection_sha256: str | None = None
    transfer_destination_sha256: str | None = None
    target_call_control_id: str | None = None
    target_call_leg_id: str | None = None
    qualified_line_bridged_at: datetime | None = None
    bridge_operation_id: UUID | None = None
    transfer_failed_at: datetime | None = None
    transfer_failure_cause: str | None = None
    local_closing_at: datetime | None = None
    content_erased: bool = False
    content_departed_generation: UUID | None = None
    original_ended_at: datetime | None = None
    recording_policy_revision: int | None = None
    recording_enabled: bool = False
    audio_reserved_bytes: int = 0

    def __post_init__(self) -> None:
        LocalCallAdmissionFacts.__post_init__(self)
        if (
            type(self.recording_enabled) is not bool
            or self.recording_policy_revision is not None
            and (
                type(self.recording_policy_revision) is not int
                or not 1 <= self.recording_policy_revision <= 2_147_483_647
            )
            or type(self.audio_reserved_bytes) is not int
            or not 0 <= self.audio_reserved_bytes <= 33_554_448
            or self.recording_policy_revision is None
            and (self.recording_enabled or self.audio_reserved_bytes)
            or not self.recording_enabled
            and self.audio_reserved_bytes
        ):
            raise ValueError("recording_policy_facts_invalid")

    @property
    def transfer_fenced(self) -> bool:
        return (
            self.content_erased
            or self.local_closing_at is not None
            or self.transfer_command_id is not None
            and (self.transfer_failed_at is None or self.local_closing_at is not None)
        )


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
    lifecycle: LocalCallLifecycleFacts | None = None


@dataclass(frozen=True, slots=True)
class OutboxItem:
    queue_id: int
    operation: VoiceOperationV1 | VoiceOperationV2 = field(repr=False)
    created_at: datetime
    claim_attempt: int
    next_attempt_at: datetime
    claim_expires_at: datetime
    last_error_code: str | None


@dataclass(frozen=True, slots=True, repr=False)
class _AudioPin:
    call_id: UUID
    generation: UUID
    workspace_id: UUID
    deployment_id: str
    recording_id: UUID | None
    revision: int
    policy: Literal["off", "local_30d"]
    available: bool
    admitted_at: datetime
    retention_until: datetime
    denied_at: datetime | None = None
    choice_state: Literal["undecided", "accepted", "off"] = "undecided"
    choice_occurred_at: datetime | None = None
    input_gate_opened: bool = False
    committed_last_sequence: int | None = None
    committed_total_samples: int | None = 0
    terminal_finished: bool = False

    @property
    def denied(self) -> bool:
        return self.denied_at is not None


@dataclass(frozen=True, slots=True, repr=False)
class AudioChoiceFacts:
    call_id: UUID
    generation: UUID
    choice_state: Literal["undecided", "accepted", "off"]
    choice_occurred_at: datetime | None
    denied_at: datetime | None


@dataclass(frozen=True, slots=True)
class RelayClaimResult:
    applied: bool


@dataclass(frozen=True, slots=True)
class CleanupResult:
    receipts: int
    leases: int


@dataclass(frozen=True, slots=True)
class WriterRuntimeObservation:
    """One owner-sampled immutable writer/outbox/storage observation."""

    writer_queue_depth: int
    writer_queue_oldest_age: float
    writer_quick_check: bool
    outbox_depth: int
    outbox_oldest_age: float
    outbox_bytes: int
    storage_bytes: int

    def __post_init__(self) -> None:
        integers = (
            self.writer_queue_depth,
            self.outbox_depth,
            self.outbox_bytes,
            self.storage_bytes,
        )
        ages = (self.writer_queue_oldest_age, self.outbox_oldest_age)
        if (
            any(type(value) is not int or value < 0 for value in integers)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or value < 0
                for value in ages
            )
            or type(self.writer_quick_check) is not bool
        ):
            raise ValueError("writer_runtime_observation_invalid") from None


WebhookReceiptKind = Literal["first", "duplicate"]
WebhookEffectKind = Literal["applied", "duplicate", "existing_terminal"]


@dataclass(frozen=True, slots=True)
class WebhookCommitResult:
    receipt: WebhookReceiptKind
    effect: WebhookEffectKind


@dataclass(frozen=True, slots=True)
class QualificationRunConsumed:
    pass


WebhookCommitValue = WebhookCommitResult | QualificationRunConsumed
WebhookReceiptClassification = Literal["missing", "duplicate", "conflict"]


class WebhookCommitTicket:
    """Opaque independently-awaitable completion for one webhook transaction."""

    __slots__ = ("_result",)

    def __init__(self, result: asyncio.Future[WebhookCommitValue]) -> None:
        self._result = result

    def __repr__(self) -> str:
        return "WebhookCommitTicket()"

    def done(self) -> bool:
        return self._result.done()

    async def wait(self) -> WebhookCommitValue:
        return await asyncio.shield(self._result)


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


@dataclass(frozen=True, slots=True, repr=False)
class AudioErasurePlan:
    recording_ids: tuple[UUID, ...]
    cleaned_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class RecordingArchiveJob:
    operation: VoiceOperationV1
    correlation: RecordingCorrelationV1
    state: Literal["pending", "archived", "acknowledged", "unavailable", "expired", "erased"]
    observed_at: datetime
    retention_until: datetime
    receipt: RecordingArchiveReceiptV1 | None
    nonce: bytes | None
    fenced: bool = False
    policy_bound: bool = False
    recording_enabled: bool = False
    reserved_bytes: int = 0
    input_gate_opened: bool = False


def _valid_contract_version(value: object) -> bool:
    return type(value) is int and value in (1, 2)


class _AudioChunkRefused(PersistenceError):
    """A known optional refusal, never a storage or integrity failure."""


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
        contract_version: Literal[1, 2] = 1,
    ) -> None:
        if not _valid_contract_version(contract_version):
            raise ValueError("invalid_writer_contract")
        self._contract_version: Literal[1, 2] = contract_version
        self._audio_pins: dict[UUID, _AudioPin] = {}
        self._audio_receipt: tuple[UUID, asyncio.Future[None]] | None = None
        self._audio_inflight = False
        self._audio_page_size = 0
        if control_commit_timeout_seconds <= 0 or quick_check_interval_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if max_storage_bytes <= 0:
            raise ValueError("max_storage_bytes must be positive")
        self._database_path = Path(database_path)
        self._keyring = keyring
        self._fatal_handler = fatal_handler
        self._monotonic = monotonic
        self._utcnow = utcnow
        self._sparra_active = False
        self._audio_cleanup: Callable[[UUID, tuple[UUID, ...]], Awaitable[None]] | None = None
        self._audio_free_bytes: Callable[[], int] | None = None
        self._audio_erase_lock = asyncio.Lock()
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
    def contract_version(self) -> Literal[1, 2]:
        return self._contract_version

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

    def try_enqueue_turn(self, operation: VoiceOperationV1, *, truncated: bool = False) -> bool:
        if operation.kind != "turn.upsert":
            raise ValueError("try_enqueue_turn requires turn.upsert")
        if not self._accepting or self._degraded:
            self.transcript_loss_count += 1
            self._signal_fatal("persistence_degraded")
            return False
        command = PersistenceCommand(
            "outbox",
            {"operation": operation, "truncated": truncated},
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
            raise self.latch_control_commit_timeout() from None

    def latch_control_commit_timeout(self) -> FatalPersistenceError:
        """Latch the public constant-safe fatal state for a control COMMIT timeout."""

        return self._signal_fatal("control_commit_timeout")

    def submit_webhook(
        self,
        *,
        receipt: Mapping[str, object],
        lease: Mapping[str, object] | None,
        operation: VoiceOperationV1 | None,
        legacy_v1_semantic_fingerprint_sha256: bytes | None = None,
        qualification_run_id: UUID | None = None,
        admission_facts: LocalCallAdmissionFacts | None = None,
    ) -> WebhookCommitTicket:
        """Synchronously transfer one webhook transaction to the writer owner."""

        if not self._accepting or self._degraded:
            raise self._new_safe_error(
                self.fatal_fault.code if self.fatal_fault else "persistence_degraded"
            )
        if not isinstance(receipt, Mapping):
            raise CommandSerializationError("invalid_webhook_receipt")
        if lease is not None and not isinstance(lease, Mapping):
            raise CommandSerializationError("invalid_lease_command")
        if operation is not None and not isinstance(operation, VoiceOperationV1):
            raise CommandSerializationError("invalid_outbox_command")
        if self._contract_version == 2 and operation is not None:
            raise CommandSerializationError("writer_contract_mismatch")
        if legacy_v1_semantic_fingerprint_sha256 is not None and (
            type(legacy_v1_semantic_fingerprint_sha256) is not bytes
            or len(legacy_v1_semantic_fingerprint_sha256) != 32
        ):
            raise CommandSerializationError("invalid_webhook_receipt")
        if qualification_run_id is not None and not isinstance(qualification_run_id, UUID):
            raise CommandSerializationError("invalid_qualification_run")
        result: asyncio.Future[WebhookCommitValue] = asyncio.get_running_loop().create_future()
        command = PersistenceCommand(
            "webhook_effect",
            {
                "receipt": receipt,
                "lease": lease,
                "operation": operation,
                "legacy_v1_semantic_fingerprint_sha256": (legacy_v1_semantic_fingerprint_sha256),
                "qualification_run_id": qualification_run_id,
                "admission_facts": admission_facts,
                "result": result,
            },
            None,
            enqueued_at=self._monotonic(),
        )
        try:
            self._queue.put_nowait(command)
        except asyncio.QueueFull as error:
            safe_error = self._signal_fatal("queue_full", source=error)
            result.set_exception(safe_error)
            result.exception()
            raise safe_error from None
        self._track_pending(command)
        return WebhookCommitTicket(result)

    async def qualification_run_consumed(self, run_id: UUID) -> bool:
        if not isinstance(run_id, UUID):
            raise CommandSerializationError("invalid_qualification_run")
        result: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "qualification_run_status",
                    {"run_id": run_id, "result": result},
                    None,
                )
            )
        except BaseException:
            if result.done() and not result.cancelled():
                result.exception()
            raise
        return await result

    async def classify_webhook_receipt(
        self,
        *,
        event_id: str,
        semantic_fingerprint_sha256: bytes,
        legacy_v1_semantic_fingerprint_sha256: bytes | None = None,
    ) -> WebhookReceiptClassification:
        if (
            not isinstance(event_id, str)
            or not 0 < len(event_id) <= 256
            or type(semantic_fingerprint_sha256) is not bytes
            or len(semantic_fingerprint_sha256) != 32
            or legacy_v1_semantic_fingerprint_sha256 is not None
            and (
                type(legacy_v1_semantic_fingerprint_sha256) is not bytes
                or len(legacy_v1_semantic_fingerprint_sha256) != 32
            )
        ):
            raise CommandSerializationError("invalid_webhook_receipt")
        result: asyncio.Future[WebhookReceiptClassification] = (
            asyncio.get_running_loop().create_future()
        )
        try:
            await self.commit_control(
                PersistenceCommand(
                    "webhook_receipt_status",
                    {
                        "event_id": event_id,
                        "semantic_fingerprint_sha256": semantic_fingerprint_sha256,
                        "legacy_v1_semantic_fingerprint_sha256": (
                            legacy_v1_semantic_fingerprint_sha256
                        ),
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
        operation: VoiceOperationV1 | None = None,
    ) -> None:
        if self._contract_version != 1 and operation is not None:
            raise PersistenceError("writer_contract_mismatch")
        if operation is not None and (
            not isinstance(operation, VoiceOperationV1)
            or operation.kind != "call.upsert"
            or operation.call_id != call_id
            or state != "terminal"
            or not isinstance(operation.payload, CallUpsertPayloadV1)
            or operation.payload.status not in {"failed", "closed"}
            or operation.payload.telnyx_call_control_id != call_control_id
            or operation.payload.ended_at != closed_at
            or operation.occurred_at != closed_at
        ):
            raise ValueError("terminal_lease_operation_invalid") from None
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
                    **({"operation": operation} if operation is not None else {}),
                },
                None,
            )
        )

    async def terminalize_stale_lease(
        self,
        stale: StaleLease,
        *,
        closed_at: datetime,
        operation: VoiceOperationV1,
    ) -> None:
        """Atomically terminalize one accepted stale cleanup and its call snapshot."""

        if self._contract_version != 1:
            raise PersistenceError("writer_contract_mismatch")
        if (
            not isinstance(stale, StaleLease)
            or not isinstance(operation, VoiceOperationV1)
            or operation.kind != "call.upsert"
            or operation.call_id != stale.call_id
        ):
            raise ValueError("stale_terminal_invalid") from None
        await self.commit_control(
            PersistenceCommand(
                "lease",
                {
                    "action": "stale_terminal",
                    "stale": stale,
                    "closed_at": closed_at,
                    "operation": operation,
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

    async def runtime_observation(self) -> WriterRuntimeObservation:
        """Read one typed aggregate through the sole writer owner/connection."""

        result: asyncio.Future[WriterRuntimeObservation] = (
            asyncio.get_running_loop().create_future()
        )
        try:
            await self.commit_control(
                PersistenceCommand(
                    "relay_batch",
                    {"action": "runtime_observation", "result": result},
                    None,
                )
            )
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

    async def _commit_relay_claim_mutation(self, payload: dict[str, object]) -> RelayClaimResult:
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

    async def read_call_lifecycle(self, call_id: UUID) -> LocalCallLifecycleFacts | None:
        result: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "call_lifecycle_read",
                    {"call_id": call_id, "result": result},
                    None,
                )
            )
        except BaseException:
            # The commit failure is propagated to the caller; retain ownership
            # of the separate result future if its writer completes later.
            result.add_done_callback(
                lambda future: None if future.cancelled() else future.exception()
            )
            raise
        return cast(LocalCallLifecycleFacts | None, await result)

    async def bind_audio_snapshot(self, snapshot: BeginCallSnapshotV2, *, generation: UUID) -> None:
        if self._contract_version != 2 or not isinstance(snapshot, BeginCallSnapshotV2):
            raise PersistenceError("audio_pin_unavailable")
        value = await self._content_request("audio_bind", snapshot=snapshot, generation=generation)
        if not isinstance(value, _AudioPin):
            raise PersistenceError("audio_pin_unavailable")
        self._audio_pins[value.call_id] = value

    async def publish_control_v2(self, operation: VoiceOperationV2, *, generation: UUID) -> None:
        if (
            self._contract_version != 2 or not isinstance(operation, VoiceOperationV2)
            or operation.kind != "call.upsert"
            or not isinstance(operation.payload, CallUpsertPayloadV1)
            or not isinstance(generation, UUID)
        ):
            raise PersistenceError("control_v2_unavailable")
        prepared = encrypt_control_operation_v2(operation, self._keyring)
        published = await self._content_request(
            "control_v2", prepared=prepared, generation=generation
        )
        if published is not True:
            raise PersistenceError("control_v2_refused")

    async def commit_audio_choice(
        self, call_id: UUID, *, generation: UUID, choice: Literal["accept", "off"],
        occurred_at: datetime | None,
    ) -> AudioChoiceFacts:
        if (
            self._contract_version != 2 or not isinstance(call_id, UUID)
            or not isinstance(generation, UUID) or choice not in {"accept", "off"}
        ):
            raise PersistenceError("audio_choice_refused")
        facts = await self._content_request(
            "audio_choice", call_id=call_id, generation=generation, choice=choice,
            occurred_at=occurred_at,
        )
        if not isinstance(facts, AudioChoiceFacts):
            raise PersistenceError("audio_choice_refused")
        return facts

    async def publish_audio_terminal_v2(
        self, operation: VoiceOperationV2, *, generation: UUID
    ) -> None:
        if (
            self._contract_version != 2 or not isinstance(operation, VoiceOperationV2)
            or operation.kind not in {"audio.finish", "audio.revoke"}
            or not isinstance(generation, UUID)
            or operation.kind == "audio.finish" and self._audio_inflight
        ):
            raise PersistenceError("audio_terminal_refused")
        result = await self._content_request(
            "audio_terminal", operation=operation, generation=generation
        )
        if result is not True:
            raise PersistenceError("audio_terminal_refused")

    def offer_audio_chunk(self, operation: VoiceOperationV2) -> bool:
        if (
            self._contract_version != 2 or not self._ready_ok or not self._accepting
            or self._degraded or self._audio_inflight or self._queue.qsize() > 239
            or not isinstance(operation, VoiceOperationV2) or operation.kind != "audio.chunk"
            or not isinstance(operation.payload, AudioChunkPayloadV2)
        ):
            return False
        pin = self._audio_pins.get(operation.call_id)
        if pin is None or not self._audio_operation_matches(operation, pin):
            return False
        if operation.payload.sequence != (
            0 if pin.committed_last_sequence is None else pin.committed_last_sequence + 1
        ):
            return False
        try:
            prepared = encrypt_audio_operation(operation, self._keyring)
            primary = self._file_size(self._database_path)
            measured = self._measure_storage_bytes()
            page = self._audio_page_size
            if type(primary) is not int or primary < 0 or page <= 0:
                return False
            # Conservative allowance for overflow/table/index/sequence page growth.
            growth = ((prepared.envelope_size + page - 1) // page + 32) * page
            if (
                measured > self.max_storage_bytes
                or 2 * (primary + growth) + 33_554_432 > MAX_STORAGE_BYTES
            ):
                return False
        except Exception:
            return False
        receipt: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        command = PersistenceCommand("sparra_content", {
            "action": "audio_chunk", "prepared": prepared, "result": receipt,
        }, None, enqueued_at=self._monotonic())
        try:
            self._queue.put_nowait(command)
        except asyncio.QueueFull:
            return False
        self._audio_inflight = True
        self._audio_receipt = (operation.operation_id, receipt)
        receipt.add_done_callback(self._finish_audio_commit)
        self._track_pending(command)
        return True

    def _finish_audio_commit(self, receipt: asyncio.Future[None]) -> None:
        if not receipt.cancelled():
            receipt.exception()

    async def wait_for_audio_commit(self, operation_id: UUID) -> None:
        receipt = self._audio_receipt
        if receipt is None or receipt[0] != operation_id:
            raise PersistenceError("audio_commit_unknown")
        try:
            await asyncio.shield(receipt[1])
        except asyncio.CancelledError:
            raise
        except PersistenceError:
            if self._audio_receipt is receipt:
                self._audio_receipt = None
                self._audio_inflight = False
            raise
        else:
            if self._audio_receipt is receipt:
                self._audio_receipt = None
                self._audio_inflight = False

    def _audio_operation_matches(self, operation: VoiceOperationV2, pin: _AudioPin) -> bool:
        payload = operation.payload
        return (
            isinstance(payload, AudioChunkPayloadV2) and pin.available and not pin.denied
            and pin.choice_state == "accepted" and pin.input_gate_opened
            and pin.committed_total_samples is not None and not pin.terminal_finished
            and pin.policy == "local_30d" and pin.retention_until > self._utcnow()
            and operation.call_id == pin.call_id and operation.deployment_id == pin.deployment_id
            and payload.workspace_id == pin.workspace_id
            and payload.recording_id == pin.recording_id
            and payload.configuration_revision == pin.revision
            and payload.retention_until == pin.retention_until
        )

    async def _content_request(self, action: str, **values: object) -> object:
        result: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "sparra_content", {"action": action, **values, "result": result}, None
                )
            )
        except BaseException:
            result.add_done_callback(
                lambda future: None if future.cancelled() else future.exception()
            )
            raise
        return await result

    def bind_recording_audio_cleanup(
        self,
        cleanup: Callable[[UUID, tuple[UUID, ...]], Awaitable[None]],
        *,
        free_bytes: Callable[[], int],
    ) -> None:
        if self._audio_cleanup is not None:
            raise RuntimeError("recording_archive_already_bound")
        self._audio_cleanup = cleanup
        self._audio_free_bytes = free_bytes

    def unbind_recording_audio_cleanup(
        self, cleanup: Callable[[UUID, tuple[UUID, ...]], Awaitable[None]]
    ) -> None:
        if self._audio_cleanup == cleanup:
            self._audio_cleanup = None
            self._audio_free_bytes = None

    async def read_recording_archive(self, recording_id: UUID) -> RecordingArchiveJob | None:
        return cast(
            RecordingArchiveJob | None,
            await self._archive_request("read", recording_id=recording_id),
        )

    async def fail_recording_archive(
        self, recording_id: UUID, *, expired: bool, now: datetime
    ) -> None:
        await self._archive_request("fail", recording_id=recording_id, expired=expired, now=now)

    async def bind_recording_archive_provider(
        self, recording_id: UUID, provider_recording_id: str, *, now: datetime
    ) -> RecordingArchiveJob | None:
        return cast(RecordingArchiveJob | None, await self._archive_request(
            "provider", recording_id=recording_id, provider_recording_id=provider_recording_id,
            now=now,
        ))

    async def pending_recording_archives(self) -> tuple[UUID, ...]:
        return cast(tuple[UUID, ...], await self._archive_request("pending"))

    async def recording_audio_cleanup_calls(self, *, now: datetime) -> tuple[UUID, ...]:
        return cast(tuple[UUID, ...], await self._archive_request("cleanup_calls", now=now))

    async def bind_recording_policy(
        self, snapshot: BeginCallSnapshotV1, *, generation: UUID
    ) -> LocalCallLifecycleFacts:
        if not isinstance(snapshot, BeginCallSnapshotV1) or not isinstance(generation, UUID):
            raise PersistenceError("recording_policy_unavailable")
        result = await self._archive_request(
            "policy_bind", snapshot=snapshot, generation=generation
        )
        if isinstance(result, str):
            raise PersistenceError(result)
        if not isinstance(result, LocalCallLifecycleFacts):
            raise PersistenceError("recording_policy_unavailable")
        return result

    async def reserve_recording_audio(self, call_id: UUID, *, generation: UUID) -> None:
        if (
            await self._archive_request("reserve", call_id=call_id, generation=generation)
            is not True
        ):
            raise PersistenceError("recording_archive_unavailable")

    async def release_recording_audio(self, call_id: UUID, *, generation: UUID) -> None:
        await self._archive_request("release", call_id=call_id, generation=generation)

    def submit_recording_archive_finish(
        self,
        recording_id: UUID,
        receipt: RecordingArchiveReceiptV1,
        nonce: bytes,
        *,
        now: datetime,
        publication_deadline: float,
    ) -> asyncio.Future[object]:
        """Transfer receipt ownership synchronously; the result is the real COMMIT outcome."""
        if not self._accepting or self._degraded:
            raise self._new_safe_error(
                self.fatal_fault.code if self.fatal_fault else "persistence_degraded"
            )
        result: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        command = PersistenceCommand(
            "recording_archive",
            {
                "action": "finish",
                "recording_id": recording_id,
                "receipt": receipt,
                "nonce": nonce,
                "now": now,
                "publication_deadline": publication_deadline,
                "result": result,
            },
            None,
            enqueued_at=self._monotonic(),
        )
        try:
            self._queue.put_nowait(command)
        except asyncio.QueueFull as error:
            raise self._signal_fatal("queue_full", source=error) from None
        self._track_pending(command)
        return result

    async def _archive_request(self, action: str, **values: object) -> object:
        result: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        try:
            await self.commit_control(
                PersistenceCommand(
                    "recording_archive", {"action": action, **values, "result": result}, None
                )
            )
        except BaseException:
            result.add_done_callback(
                lambda future: None if future.cancelled() else future.exception()
            )
            raise
        return await result

    async def _read_recording_archive(self, recording_id: UUID) -> RecordingArchiveJob | None:
        cursor = await self._require_owner_connection().execute(
            "SELECT call_id,deployment_id,state,observed_at,retention_until,context_key_version,"
            "context_nonce,context_ciphertext,receipt_json,audio_nonce FROM recording_archives "
            "WHERE recording_id=?",
            (str(recording_id),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        from pydantic import SecretStr

        from projetv0_voice.telnyx.recordings import decode_recording_correlation

        call_id = UUID(row[0])
        context = json.loads(
            self._keyring.decrypt(
                EncryptedValue(row[5], row[6], row[7]),
                aad=f"recording-job:{call_id}:{recording_id}".encode("ascii"),
            )
        )
        operation = VoiceOperationV1.model_validate(context["operation"])
        correlation = decode_recording_correlation(SecretStr(context["correlation"]))
        if (
            operation.call_id != call_id
            or operation.deployment_id != row[1]
            or not isinstance(operation.payload, RecordingUpsertPayloadV1)
            or operation.payload.recording_id != recording_id
            or correlation.call_id != call_id
            or correlation.recording_id != recording_id
            or correlation.deployment_id != row[1]
            or correlation.admission_retention_until != _parse_datetime(row[4])
        ):
            raise CommandConflictError("recording_archive_identity_conflict")
        receipt = None if row[8] is None else RecordingArchiveReceiptV1.model_validate_json(row[8])
        facts = await self._read_call_lifecycle(call_id)
        return RecordingArchiveJob(
            operation,
            correlation,
            row[2],
            _parse_datetime(row[3]),
            _parse_datetime(row[4]),
            receipt,
            row[9],
            await self._content_fenced(call_id),
            facts is not None and facts.recording_policy_revision is not None,
            facts.recording_enabled if facts is not None else False,
            facts.audio_reserved_bytes if facts is not None else 0,
            facts is not None
            and facts.started_at is not None
            and facts.disclosure_evidence is not None
            and facts.disclosure_evidence.input_gate_opened_at is not None,
        )

    async def _apply_archive_command(self, values: Mapping[str, object]) -> object:
        connection = self._require_owner_connection()
        action = values.get("action")
        if action in {"policy_bind", "reserve", "release"}:
            snapshot = values.get("snapshot")
            call_id = (
                snapshot.call_id
                if isinstance(snapshot, BeginCallSnapshotV1)
                else (self._required_uuid(values, "call_id"))
            )
            generation = self._required_uuid(values, "generation")
            facts = await self._read_call_lifecycle(call_id)
            if (
                facts is None
                or facts.admission_generation != generation
                or await self._content_fenced(call_id)
                or facts.content_erased
                or facts.retention_until <= self._utcnow()
            ):
                return "recording_policy_unavailable" if action == "policy_bind" else False
            if action == "policy_bind":
                if (
                    not isinstance(snapshot, BeginCallSnapshotV1)
                    or snapshot.retention_until != facts.retention_until
                ):
                    return "recording_policy_unavailable"
                if facts.recording_policy_revision is not None:
                    return (
                        facts
                        if (
                            facts.recording_policy_revision == snapshot.configuration_revision
                            and facts.recording_enabled is snapshot.recording_enabled
                        )
                        else "recording_policy_conflict"
                    )
                facts = replace(
                    facts,
                    recording_policy_revision=snapshot.configuration_revision,
                    recording_enabled=snapshot.recording_enabled,
                )
                await self._store_lifecycle(facts)
                return facts
            if action == "release":
                await self._store_lifecycle(replace(facts, audio_reserved_bytes=0))
                return True
            if (
                not facts.recording_enabled
                or facts.recording_policy_revision is None
                or facts.local_closing_at is not None
                or not await self._recording_resources_ready()
            ):
                return False
            cursor = await connection.execute(
                "SELECT 1 FROM recording_archives WHERE call_id=? AND receipt_json IS NOT NULL "
                "AND state IN ('archived','acknowledged')",
                (str(call_id),),
            )
            committed_file = await cursor.fetchone()
            await cursor.close()
            if committed_file is not None:
                return True  # The real file is already charged to filesystem free space.
            cursor = await connection.execute(
                "SELECT COALESCE(sum(json_extract(lifecycle_json,'$.audio_reserved_bytes')),0) "
                "FROM call_leases WHERE call_id!=?",
                (str(call_id),),
            )
            reserved = await cursor.fetchone()
            await cursor.close()
            try:
                free = self._audio_free_bytes() if self._audio_free_bytes is not None else 0
            except Exception:
                return False
            if type(free) is not int or free < (reserved[0] if reserved else 0) + 33_554_448:
                return False
            if facts.audio_reserved_bytes == 0:
                await self._store_lifecycle(replace(facts, audio_reserved_bytes=33_554_448))
            return True
        if action == "cleanup_calls":
            now = self._required_datetime(values, "now")
            cursor = await connection.execute(
                "SELECT DISTINCT a.call_id FROM recording_archives a "
                "LEFT JOIN sparra_content_fences f ON f.call_id=a.call_id "
                "WHERE f.call_id IS NOT NULL OR julianday(a.retention_until)<=julianday(?) "
                "ORDER BY a.call_id LIMIT 100",
                (_iso(now),),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return tuple(UUID(row[0]) for row in rows)
        if action == "pending":
            cursor = await connection.execute(
                "SELECT recording_id FROM recording_archives "
                "WHERE state='pending' ORDER BY observed_at,recording_id LIMIT 1"
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return tuple(UUID(row[0]) for row in rows)
        if action == "queue":
            from projetv0_voice.telnyx.recordings import (
                RecordingCorrelationV1,
                encode_recording_correlation,
            )

            operation = values.get("operation")
            correlation = values.get("correlation")
            now = self._required_datetime(values, "now")
            if (
                not isinstance(operation, VoiceOperationV1)
                or not isinstance(operation.payload, RecordingUpsertPayloadV1)
                or not isinstance(correlation, RecordingCorrelationV1)
                or operation.call_id != correlation.call_id
                or operation.deployment_id != correlation.deployment_id
                or operation.payload.recording_id != correlation.recording_id
                or correlation.admission_retention_until is None
            ):
                raise CommandSerializationError("recording_archive_queue_invalid")
            if operation.payload.status != "saved":
                return None
            facts = await self._read_call_lifecycle(operation.call_id)
            if (
                facts is None
                or facts.retention_until != correlation.admission_retention_until
                or facts.telnyx_call_leg_id != correlation.call_leg_id
                or facts.telnyx_call_session_id != correlation.call_session_id
                or operation.payload.retention_until != facts.retention_until
            ):
                raise CommandConflictError("recording_archive_admission_conflict")
            cursor = await connection.execute(
                "SELECT call_control_id FROM call_leases WHERE call_id=?", (str(operation.call_id),)
            )
            bound = await cursor.fetchone()
            await cursor.close()
            if bound != (correlation.call_control_id,):
                raise CommandConflictError("recording_archive_admission_conflict")
            # Retain the genuine provider purge identity even when an already
            # fenced/expired callback must not create private archive metadata.
            await self._insert_outbox(operation)
            if await self._content_fenced(operation.call_id) or facts.retention_until <= now:
                return None
            if (
                facts.recording_policy_revision is None
                or not facts.recording_enabled
                or facts.audio_reserved_bytes != 33_554_448
                or facts.started_at is None
                or facts.disclosure_evidence is None
                or facts.disclosure_evidence.input_gate_opened_at is None
            ):
                return None
            recording_id = correlation.recording_id
            context = json.dumps(
                {
                    "operation": operation.model_dump(mode="json"),
                    "correlation": encode_recording_correlation(correlation).get_secret_value(),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            if len(context) > 65536:
                raise CommandSerializationError("recording_archive_context_too_large")
            fingerprint = hashlib.sha256(context).digest()
            existing = await self._read_recording_archive(recording_id)
            if existing is not None:
                cursor = await connection.execute(
                    "SELECT fingerprint FROM recording_archives WHERE recording_id=?",
                    (str(recording_id),),
                )
                original = await cursor.fetchone()
                await cursor.close()
                if original != (fingerprint,):
                    raise CommandConflictError("recording_archive_identity_conflict")
                return existing
            state = (
                "erased"
                if await self._content_fenced(operation.call_id)
                else ("expired" if facts.retention_until <= now else "pending")
            )
            observed_at = min(now, facts.original_ended_at) if facts.original_ended_at else now
            encrypted = self._keyring.encrypt(
                context, aad=f"recording-job:{operation.call_id}:{recording_id}".encode("ascii")
            )
            await connection.execute(
                "INSERT INTO recording_archives "
                "(recording_id,call_id,deployment_id,fingerprint,state,observed_at,retention_until,"
                "context_key_version,context_nonce,context_ciphertext) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    str(recording_id),
                    str(operation.call_id),
                    operation.deployment_id,
                    fingerprint,
                    state,
                    _iso(observed_at),
                    _iso(facts.retention_until),
                    encrypted.key_version,
                    encrypted.nonce,
                    encrypted.ciphertext,
                ),
            )
            return await self._read_recording_archive(recording_id)
        recording_id = self._required_uuid(values, "recording_id")
        job = await self._read_recording_archive(recording_id)
        if action == "read" or job is None:
            return job
        now = self._required_datetime(values, "now")
        if action == "provider":
            return await self._bind_archive_provider(job, values.get("provider_recording_id"), now)
        if action == "fail":
            if job.state not in {"erased", "expired"}:
                await connection.execute(
                    "UPDATE recording_archives SET state=? WHERE recording_id=?",
                    (
                        "expired" if values.get("expired") is True else "unavailable",
                        str(recording_id),
                    ),
                )
                facts = await self._read_call_lifecycle(job.operation.call_id)
                if facts is not None:
                    await self._store_lifecycle(replace(facts, audio_reserved_bytes=0))
            return None
        if action != "finish":
            raise CommandSerializationError("recording_archive_action_invalid")
        if (
            await self._content_fenced(job.operation.call_id)
            or job.retention_until <= self._utcnow()
            or job.state != "pending"
            or not job.policy_bound
            or not job.recording_enabled
            or job.reserved_bytes != 33_554_448
            or not job.input_gate_opened
            or not self._archive_publication_budget(values, job)
        ):
            return None
        receipt, nonce = values.get("receipt"), values.get("nonce")
        if (
            not isinstance(receipt, RecordingArchiveReceiptV1)
            or receipt.recording_id != recording_id
            or receipt.retention_until != job.retention_until
            or type(nonce) is not bytes
            or len(nonce) != 12
        ):
            raise CommandSerializationError("recording_archive_receipt_invalid")
        receipt_operation = job.operation.model_copy(
            update={
                "operation_id": uuid5(recording_id, "archive-receipt-v1"),
                "occurred_at": max(now, job.operation.occurred_at),
                "payload": job.operation.payload.model_copy(update={"archive_receipt": receipt}),
            }
        )
        await self._insert_outbox(receipt_operation)
        await connection.execute(
            "UPDATE recording_archives SET state='archived',receipt_json=?,"
            "audio_nonce=?,receipt_op_id=? WHERE recording_id=?",
            (
                receipt.model_dump_json(),
                nonce,
                str(receipt_operation.operation_id),
                str(recording_id),
            ),
        )
        facts = await self._read_call_lifecycle(job.operation.call_id)
        if facts is not None:
            await self._store_lifecycle(replace(facts, audio_reserved_bytes=0))
        return await self._read_recording_archive(recording_id)

    async def _bind_archive_provider(
        self,
        job: RecordingArchiveJob,
        provider_id: object,
        now: datetime,
    ) -> RecordingArchiveJob | None:
        from projetv0_voice.models import is_valid_provider_recording_id

        payload = job.operation.payload
        if (
            not is_valid_provider_recording_id(provider_id)
            or not isinstance(payload, RecordingUpsertPayloadV1)
            or job.fenced
            or job.state != "pending"
            or job.retention_until <= self._utcnow()
            or now >= job.observed_at + timedelta(seconds=120)
            or self._utcnow() >= job.observed_at + timedelta(seconds=120)
            or not job.policy_bound
            or not job.recording_enabled
            or not job.input_gate_opened
            or job.reserved_bytes != 33_554_448
        ):
            return None
        if payload.telnyx_recording_id is not None:
            return job if payload.telnyx_recording_id == provider_id else None
        operation = job.operation.model_copy(
            update={
                "operation_id": uuid5(payload.recording_id, "archive-provider-id-v1"),
                "payload": payload.model_copy(update={"telnyx_recording_id": provider_id}),
            }
        )
        from projetv0_voice.telnyx.recordings import encode_recording_correlation

        context = json.dumps(
            {
                "operation": operation.model_dump(mode="json"),
                "correlation": encode_recording_correlation(job.correlation).get_secret_value(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(context) > 65536:
            raise CommandSerializationError("recording_archive_context_too_large")
        encrypted = self._keyring.encrypt(
            context, aad=f"recording-job:{operation.call_id}:{payload.recording_id}".encode("ascii")
        )
        await self._insert_outbox(operation)
        # Keep the original callback fingerprint and observation for immutable replay/budget.
        await self._require_owner_connection().execute(
            "UPDATE recording_archives SET context_key_version=?,context_nonce=?,"
            "context_ciphertext=? WHERE recording_id=?",
            (
                encrypted.key_version,
                encrypted.nonce,
                encrypted.ciphertext,
                str(payload.recording_id),
            ),
        )
        return await self._read_recording_archive(payload.recording_id)

    async def _recording_resources_ready(self) -> bool:
        if self._audio_cleanup is None or self._audio_free_bytes is None:
            return False
        cursor = await self._require_owner_connection().execute(
            "SELECT state,observed_at FROM recording_archives "
            "WHERE state NOT IN ('acknowledged','erased','expired')"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        now = self._utcnow()
        if any(
            state == "unavailable" or _parse_datetime(observed) + timedelta(seconds=120) <= now
            for state, observed in rows
        ):
            return False
        cursor = await self._require_owner_connection().execute(
            "SELECT lifecycle_json FROM call_leases WHERE lifecycle_json IS NOT NULL "
            "AND json_extract(lifecycle_json,'$.audio_reserved_bytes')>0"
        )
        held = await cursor.fetchall()
        await cursor.close()
        return not any(
            (facts := self._decode_lifecycle(row[0])).original_ended_at is not None
            and facts.original_ended_at + timedelta(seconds=120) <= now
            for row in held
        )

    def _archive_publication_budget(
        self, values: Mapping[str, object], job: RecordingArchiveJob
    ) -> bool:
        deadline = values.get("publication_deadline")
        if (
            not isinstance(deadline, int | float)
            or type(deadline) not in {float, int}
            or not math.isfinite(deadline)
        ):
            raise CommandSerializationError("recording_archive_deadline_invalid")
        current = self._utcnow()
        return (
            self._monotonic() < float(deadline)
            and current < job.retention_until
            and current < job.observed_at + timedelta(seconds=120)
        )

    def try_enqueue_capture_loss(self, call_id: UUID, capture_id: UUID) -> bool:
        if not self._accepting or self._degraded:
            return False
        command = PersistenceCommand(
            "sparra_content",
            {
                "action": "capture_loss",
                "call_id": call_id,
                "capture_id": capture_id,
            },
            None,
            enqueued_at=self._monotonic(),
        )
        try:
            self._queue.put_nowait(command)
        except asyncio.QueueFull:
            self._signal_fatal("queue_full")
            return False
        self._track_pending(command)
        return True

    async def read_retained_call(self, call_id: UUID) -> RetainedCall:
        """Queue behind captures; return only committed, authenticated same-call facts."""
        return cast(RetainedCall, await self._content_request("read", call_id=call_id))

    async def freeze_call_publication(
        self,
        operation: VoiceOperationV1,
        result: MessageResultV1 | None,
        *,
        provider_callback: str | None,
        result_permitted: Callable[[], bool] | None = None,
    ) -> VoiceOperationV1 | None:
        return cast(
            VoiceOperationV1 | None,
            await self._content_request(
                "freeze",
                operation=operation,
                call_id=operation.call_id,
                message_result=result,
                provider_callback=provider_callback,
                result_permitted=result_permitted,
            ),
        )

    async def read_frozen_call_publication(self, call_id: UUID) -> VoiceOperationV1 | None:
        return cast(
            VoiceOperationV1 | None, await self._content_request("frozen_read", call_id=call_id)
        )

    async def erase_call_content(
        self,
        call_id: UUID,
        *,
        now: datetime,
        lease_token: UUID | None = None,
        expected_item: OutboxItem | None = None,
        generation: UUID | None = None,
    ) -> datetime | None:
        async def erase_owned() -> datetime | None:
            async with self._audio_erase_lock:
                plan = await self._content_request(
                    "erase",
                    call_id=call_id,
                    now=now,
                    lease_token=lease_token,
                    expected_item=expected_item,
                    generation=generation,
                )
                if not isinstance(plan, AudioErasurePlan):
                    return cast(datetime | None, plan)
                if self._audio_cleanup is None:
                    raise PersistenceError("recording_archive_cleanup_unavailable")
                # The durable fence precedes this join; the SQLite owner remains
                # free to settle any receipt that the audio task already queued.
                await self._audio_cleanup(call_id, plan.recording_ids)
                return cast(
                    datetime,
                    await self._content_request(
                        "audio_erase_complete",
                        call_id=call_id,
                        now=now,
                        lease_token=lease_token,
                        cleaned_at=max(plan.cleaned_at, self._utcnow()),
                    ),
                )

        owner = asyncio.create_task(erase_owned(), name="voice-audio-erasure")
        cancellation: asyncio.CancelledError | None = None
        while not owner.done():
            try:
                await asyncio.shield(owner)
            except asyncio.CancelledError as error:
                cancellation = error
        result = owner.result()
        if cancellation is not None:
            raise cancellation
        return result

    async def expired_content_calls(self, *, now: datetime) -> tuple[UUID, ...]:
        return cast(tuple[UUID, ...], await self._content_request("expired", now=now))

    async def pending_erasure_acks(self) -> tuple[tuple[UUID, UUID, datetime], ...]:
        return cast(
            tuple[tuple[UUID, UUID, datetime], ...], await self._content_request("pending_acks")
        )

    async def erased_recording_head(self) -> int | None:
        return cast(int | None, await self._content_request("recording_head"))

    async def mark_call_departed(self, call_id: UUID, *, now: datetime) -> LocalCallLifecycleFacts:
        return cast(
            LocalCallLifecycleFacts, await self._content_request("depart", call_id=call_id, now=now)
        )

    async def finish_erasure_ack(
        self, call_id: UUID, lease_token: UUID, *, acknowledged: bool = True
    ) -> None:
        await self._content_request(
            "ack_done", call_id=call_id, lease_token=lease_token, acknowledged=acknowledged
        )

    async def _content_fenced(self, call_id: UUID) -> bool:
        cursor = await self._require_owner_connection().execute(
            "SELECT 1 FROM sparra_content_fences WHERE call_id=?", (str(call_id),)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None

    async def _content_denied(self, call_id: UUID) -> bool:
        if await self._content_fenced(call_id):
            return True
        if not self._sparra_active:
            return False
        facts = await self._read_call_lifecycle(call_id)
        # Trusted Sparra activation precedes all producers, including restart.
        # Collected/unknown admission is never legacy content authority.
        return facts is None or facts.retention_until <= self._utcnow()

    async def _collect_terminal_sparra_calls(self, now: datetime) -> int:
        if not self._sparra_active:
            return 0
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            "SELECT c.call_id,c.lifecycle_json FROM call_leases c "
            "JOIN sparra_content_fences f ON f.call_id=c.call_id "
            "WHERE c.state='terminal' AND f.lease_acked=1 AND f.lease_settled=1 "
            "AND f.lease_token IS NOT NULL "
            "AND json_extract(c.lifecycle_json,'$.original_ended_at') IS NOT NULL "
            "AND julianday(json_extract(c.lifecycle_json,'$.retention_until')) <= julianday(?) "
            "AND NOT EXISTS(SELECT 1 FROM outbox o WHERE o.call_id=c.call_id) "
            "AND NOT EXISTS(SELECT 1 FROM sparra_turn_decisions d WHERE d.call_id=c.call_id) "
            "AND NOT EXISTS(SELECT 1 FROM sparra_publications p WHERE p.call_id=c.call_id) "
            "ORDER BY c.created_at LIMIT 100",
            (_iso(now - SPARRA_REPLAY_RETENTION),),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        collected = 0
        for call_id, raw in rows:
            facts = self._decode_lifecycle(raw)
            if (
                not facts.content_erased
                or facts.original_ended_at is None
                or now <= facts.retention_until + SPARRA_REPLAY_RETENTION
            ):
                continue
            # The same owner/transaction has checked real terminal, ACK,
            # retained-content and all FIFO/claimed recording obligations.
            await connection.execute(
                "DELETE FROM sparra_content_fences WHERE call_id=?", (call_id,)
            )
            await connection.execute("DELETE FROM call_leases WHERE call_id=?", (call_id,))
            if self._contract_version == 2:
                await connection.execute(
                    "DELETE FROM local_audio_terminal WHERE call_id=?", (call_id,)
                )
                await connection.execute("DELETE FROM local_audio_pin WHERE call_id=?", (call_id,))
                self._audio_pins.pop(UUID(call_id), None)
            collected += 1
        return collected

    async def _retained_payloads(
        self, call_id: UUID
    ) -> tuple[tuple[TurnUpsertPayloadV1, ...], int]:
        cursor = await self._require_owner_connection().execute(
            "SELECT turn_id,op_id,deployment_id,key_version,nonce,ciphertext,lost "
            "FROM sparra_turn_decisions WHERE call_id=?",
            (str(call_id),),
        )
        rows = list(await cursor.fetchall())
        await cursor.close()
        payloads = []
        for turn_id, op_id, deployment_id, key_version, nonce, ciphertext, _lost in rows:
            if ciphertext is None:
                continue
            aad = operation_aad_from_metadata(
                dict(
                    schema_version=1,
                    kind="turn.upsert",
                    call_id=str(call_id),
                    operation_id=op_id,
                    deployment_id=deployment_id,
                )
            )
            operation = decode_operation(
                self._keyring.decrypt(EncryptedValue(key_version, nonce, ciphertext), aad=aad)
            )
            payload = operation.payload
            if (
                operation.call_id != call_id
                or not isinstance(payload, TurnUpsertPayloadV1)
                or (str(payload.turn_id) != turn_id)
            ):
                raise CommandConflictError("retained_operation_identity_conflict")
            payloads.append(payload)
        return tuple(payloads), sum(row[6] for row in rows)

    async def _retained_call(self, call_id: UUID) -> RetainedCall:
        if await self._content_denied(call_id):
            return RetainedCall((), 0, True)
        payloads, loss_count = await self._retained_payloads(call_id)
        turns = []
        for payload in payloads:
            text = self._keyring.decrypt(
                EncryptedValue(
                    payload.key_version,
                    base64.b64decode(payload.nonce_b64, validate=True),
                    base64.b64decode(payload.ciphertext_b64, validate=True),
                ),
                aad=f"turn:{payload.turn_id}".encode("ascii"),
            ).decode("utf-8")
            validate_turn_text(text)
            turns.append(
                RetainedTurn(
                    payload.turn_id, payload.turn_no, payload.role, text, payload.interrupted
                )
            )
        return RetainedCall(
            tuple(sorted(turns, key=lambda t: (t.turn_no, str(t.turn_id)))),
            loss_count,
        )

    async def _retain_turn(self, operation: VoiceOperationV1, *, truncated: bool) -> bool:
        if await self._read_call_lifecycle(operation.call_id) is None:
            return not self._sparra_active  # Preserve absent-Sparra legacy queue bytes.
        connection = self._require_owner_connection()
        payload = cast(TurnUpsertPayloadV1, operation.payload)
        fingerprint = hashlib.sha256(canonical_operation_bytes(operation)).digest()
        cursor = await connection.execute(
            "SELECT fingerprint FROM sparra_turn_decisions WHERE call_id=? AND turn_id=?",
            (str(operation.call_id), str(payload.turn_id)),
        )
        existing = await cursor.fetchone()
        await cursor.close()
        if existing is not None:
            if existing[0] != fingerprint:
                raise CommandConflictError("retained_turn_identity_conflict")
            return False
        # Authenticate before either retaining or admitting result evidence.
        text = self._keyring.decrypt(
            EncryptedValue(
                payload.key_version,
                base64.b64decode(payload.nonce_b64, validate=True),
                base64.b64decode(payload.ciphertext_b64, validate=True),
            ),
            aad=f"turn:{payload.turn_id}".encode("ascii"),
        ).decode("utf-8")
        validate_turn_text(text)
        previous, _loss = await self._retained_payloads(operation.call_id)
        retained_map = {str(p.turn_id): p.model_dump(mode="json") for p in previous}
        retained_map[str(payload.turn_id)] = payload.model_dump(mode="json")
        # PostgreSQL JSONB ::text uses comma/colon spaces; keys need no escaping.
        fits = (
            len(retained_map) <= 200
            and len(json.dumps(retained_map, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            <= 524288
        )
        encrypted = encrypt_operation(operation, self._keyring).encrypted if fits else None
        await connection.execute(
            "INSERT INTO sparra_turn_decisions VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(operation.call_id),
                str(payload.turn_id),
                str(operation.operation_id),
                operation.deployment_id,
                fingerprint,
                encrypted.key_version if encrypted is not None else None,
                encrypted.nonce if encrypted is not None else None,
                encrypted.ciphertext if encrypted is not None else None,
                int(truncated or not fits),
            ),
        )
        return fits

    async def _frozen_publication(self, call_id: UUID) -> VoiceOperationV1 | None:
        cursor = await self._require_owner_connection().execute(
            "SELECT op_id,deployment_id,key_version,nonce,ciphertext "
            "FROM sparra_publications WHERE call_id=?",
            (str(call_id),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        aad = operation_aad_from_metadata(
            dict(
                schema_version=1,
                operation_id=row[0],
                deployment_id=row[1],
                call_id=str(call_id),
                kind="call.upsert",
            )
        )
        return decode_operation(
            self._keyring.decrypt(EncryptedValue(row[2], row[3], row[4]), aad=aad)
        )

    async def _read_audio_pin(self, call_id: UUID) -> _AudioPin | None:
        cursor = await self._require_owner_connection().execute(
            "SELECT call_id,generation,workspace_id,deployment_id,recording_id,"
            "configuration_revision,recording_policy,audio_available,admitted_at,"
            "retention_until,denied_at,choice_state,choice_occurred_at,"
            "committed_last_sequence,committed_total_samples "
            "FROM local_audio_pin WHERE call_id=?", (str(call_id),)
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        if (
            row[6] not in {"off", "local_30d"} or row[7] not in (0, 1)
            or row[11] not in {"undecided", "accepted", "off"}
        ):
            raise FatalPersistenceError("audio_pin_invalid")
        facts = await self._read_call_lifecycle(call_id)
        evidence = None if facts is None else facts.disclosure_evidence
        gate_opened = (
            facts is not None and not facts.transfer_fenced
            and facts.admission_generation == UUID(row[1]) and evidence is not None
            and evidence.completed_at is not None and evidence.failed_at is None
            and evidence.input_gate_opened_at is not None
        )
        cursor = await self._require_owner_connection().execute(
            "SELECT 1 FROM local_audio_terminal WHERE call_id=? AND kind='audio.finish'",
            (str(call_id),),
        )
        terminal_finished = await cursor.fetchone() is not None
        await cursor.close()
        return _AudioPin(
            UUID(row[0]), UUID(row[1]), UUID(row[2]), row[3],
            None if row[4] is None else UUID(row[4]), row[5],
            cast(Literal["off", "local_30d"], row[6]), row[7] == 1,
            _parse_datetime(row[8]), _parse_datetime(row[9]),
            None if row[10] is None else _parse_datetime(row[10]),
            cast(Literal["undecided", "accepted", "off"], row[11]),
            None if row[12] is None else _parse_datetime(row[12]), gate_opened, row[13], row[14],
            terminal_finished,
        )

    async def _apply_audio_command(self, values: Mapping[str, object]) -> object:
        if self._contract_version != 2:
            raise CommandSerializationError("writer_contract_mismatch")
        connection = self._require_owner_connection()
        if values.get("action") == "audio_bind":
            snapshot = values.get("snapshot")
            generation = values.get("generation")
            if not isinstance(snapshot, BeginCallSnapshotV2) or not isinstance(generation, UUID):
                return None
            facts = await self._read_call_lifecycle(snapshot.call_id)
            cursor = await connection.execute(
                "SELECT tenant_id,agent_id,state,created_at FROM call_leases WHERE call_id=?",
                (str(snapshot.call_id),),
            )
            leases = list(await cursor.fetchall())
            await cursor.close()
            if (
                facts is None or facts.admission_generation != generation or facts.content_erased
                or facts.transfer_fenced or snapshot.retention_until != facts.retention_until
                or facts.retention_until <= self._utcnow()
                or await self._content_fenced(snapshot.call_id)
                or len(leases) != 1 or leases[0][0] != str(snapshot.workspace_id)
                or leases[0][2] not in {"pending", "active"}
                or _parse_datetime(leases[0][3]) != facts.admitted_at
            ):
                return None
            bound_pin = _AudioPin(
                snapshot.call_id, generation, snapshot.workspace_id, leases[0][1],
                snapshot.recording_id, snapshot.configuration_revision, snapshot.recording_policy,
                snapshot.audio_available, facts.admitted_at, facts.retention_until,
            )
            existing_pin = await self._read_audio_pin(snapshot.call_id)
            if existing_pin is not None:
                context = replace(existing_pin, denied_at=None, choice_state="undecided",
                                  choice_occurred_at=None, input_gate_opened=False,
                                  committed_last_sequence=None, committed_total_samples=0,
                                  terminal_finished=False)
                return existing_pin if context == bound_pin else None
            await connection.execute(
                "INSERT INTO local_audio_pin(call_id,generation,workspace_id,deployment_id,"
                "recording_id,configuration_revision,recording_policy,audio_available,"
                "admitted_at,retention_until) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (str(bound_pin.call_id), str(bound_pin.generation), str(bound_pin.workspace_id),
                 bound_pin.deployment_id,
                 None if bound_pin.recording_id is None else str(bound_pin.recording_id),
                 bound_pin.revision, bound_pin.policy, int(bound_pin.available),
                 _iso(bound_pin.admitted_at), _iso(bound_pin.retention_until)),
            )
            return bound_pin
        prepared = values.get("prepared")
        if not isinstance(prepared, PreparedAudioOperation):
            raise CommandSerializationError("invalid_audio_command")
        operation = prepared.operation
        audio_payload = operation.payload
        if not isinstance(audio_payload, AudioChunkPayloadV2):
            raise CommandSerializationError("invalid_audio_command")
        pin = await self._read_audio_pin(operation.call_id)
        facts = await self._read_call_lifecycle(operation.call_id)
        if (
            pin is None or not self._audio_operation_matches(operation, pin) or facts is None
            or facts.admission_generation != pin.generation or facts.transfer_fenced
            or await self._content_denied(operation.call_id)
        ):
            raise _AudioChunkRefused("audio_chunk_refused")
        encrypted = prepared.encrypted
        if self._keyring.decrypt(encrypted, aad=prepared.aad) != canonical_operation_bytes(
            operation
        ):
            raise CommandSerializationError("invalid_audio_command")
        existing_operation = await self._load_operation_by_id(str(operation.operation_id))
        if existing_operation is not None:
            if canonical_operation_bytes(existing_operation) != prepared.plaintext:
                raise CommandConflictError("audio_operation_conflict")
            return None
        expected_sequence = (
            0 if pin.committed_last_sequence is None else pin.committed_last_sequence + 1
        )
        if pin.committed_total_samples is None:
            raise _AudioChunkRefused("audio_chunk_refused")
        total_samples = pin.committed_total_samples + audio_payload.sample_count
        if audio_payload.sequence != expected_sequence or total_samples > 4_800_000:
            raise _AudioChunkRefused("audio_chunk_refused")
        try:
            await connection.execute(
                "INSERT INTO outbox(op_id,deployment_id,kind,schema_version,call_id,recording_id,"
                "crypto_version,key_version,nonce,ciphertext,created_at,next_attempt_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(operation.operation_id), operation.deployment_id, operation.kind, 2,
                 str(operation.call_id), str(audio_payload.recording_id), CRYPTO_VERSION,
                 encrypted.key_version, encrypted.nonce, encrypted.ciphertext,
                 _iso(self._utcnow()), _iso(self._utcnow())),
            )
        except sqlite3.IntegrityError:
            existing_operation = await self._load_operation_by_id(str(operation.operation_id))
            if (
                existing_operation is None
                or canonical_operation_bytes(existing_operation) != prepared.plaintext
            ):
                raise CommandConflictError("audio_operation_conflict") from None
            return None
        await connection.execute(
            "UPDATE local_audio_pin SET committed_last_sequence=?,committed_total_samples=? "
            "WHERE call_id=?", (audio_payload.sequence, total_samples, str(operation.call_id)),
        )
        return None

    async def _apply_content_command(self, values: Mapping[str, object]) -> object:
        connection = self._require_owner_connection()
        action = values.get("action")
        if action == "audio_terminal":
            return await self._apply_audio_terminal(values)
        if action == "audio_choice":
            return await self._apply_audio_choice(values)
        if action == "control_v2":
            return await self._apply_control_v2_command(values)
        if action in {"audio_bind", "audio_chunk"}:
            return await self._apply_audio_command(values)
        if action == "recording_head":
            cursor = await connection.execute(
                "SELECT o.queue_id,o.op_id FROM outbox o "
                "JOIN sparra_content_fences f ON f.call_id=o.call_id "
                "WHERE o.kind='recording.upsert' AND o.queue_id=(SELECT min(queue_id) FROM outbox)"
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            operation = await self._load_operation_by_id(row[1])
            if (
                operation is not None
                and isinstance(operation.payload, RecordingUpsertPayloadV1)
                and operation.payload.telnyx_recording_id is not None
            ):
                return row[0]
            return None
        if action == "pending_acks":
            cursor = await connection.execute(
                "SELECT call_id,lease_token,lease_cleaned_at FROM sparra_content_fences "
                "WHERE lease_token IS NOT NULL AND lease_cleaned_at IS NOT NULL AND lease_settled=0"
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return tuple((UUID(c), UUID(t), _parse_datetime(at)) for c, t, at in rows)
        if action == "expired":
            now = self._required_datetime(values, "now")
            # ISO fractions vary in persisted legacy facts. SQLite's temporal
            # prefilter is inclusive; the decoded datetime below remains exact.
            cursor = await connection.execute(
                "SELECT c.call_id,c.lifecycle_json FROM call_leases c "
                "WHERE c.lifecycle_json IS NOT NULL "
                "AND julianday(json_extract(c.lifecycle_json,'$.retention_until')) <= julianday(?) "
                "AND NOT EXISTS(SELECT 1 FROM sparra_content_fences f WHERE f.call_id=c.call_id) "
                "ORDER BY c.created_at LIMIT 100",
                (_iso(now),),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            expired = []
            for call, raw in rows:
                if self._decode_lifecycle(raw).retention_until <= now:
                    expired.append(UUID(call))
            return tuple(expired)
        call_id = self._required_uuid(values, "call_id")
        if action == "audio_erase_complete":
            if not await self._content_fenced(call_id):
                raise CommandConflictError("recording_audio_erasure_fence_missing")
            await connection.execute(
                "DELETE FROM recording_archives WHERE call_id=?", (str(call_id),)
            )
            facts = await self._read_call_lifecycle(call_id)
            if facts is not None:
                await self._store_lifecycle(replace(facts, audio_reserved_bytes=0))
            cleaned = self._required_datetime(values, "cleaned_at")
            token = values.get("lease_token")
            if isinstance(token, UUID):
                await connection.execute(
                    "UPDATE sparra_content_fences SET lease_cleaned_at=? "
                    "WHERE call_id=? AND lease_token=?",
                    (_iso(cleaned), str(call_id), str(token)),
                )
            else:
                await connection.execute(
                    "UPDATE sparra_content_fences SET lease_cleaned_at=? "
                    "WHERE call_id=? AND lease_token IS NOT NULL AND lease_settled=0",
                    (_iso(cleaned), str(call_id)),
                )
            return cleaned
        if action == "depart":
            facts = await self._read_call_lifecycle(call_id)
            if facts is None or not (
                facts.admission_generation
                or facts.transfer_generation
                or facts.content_departed_generation
            ):
                raise FatalPersistenceError("call_departure_generation_unavailable")
            facts = replace(
                facts,
                local_closing_at=facts.local_closing_at or self._required_datetime(values, "now"),
            )
            await self._store_lifecycle(facts)
            return facts
        if action == "capture_loss":
            capture_id = self._required_uuid(values, "capture_id")
            if not await self._content_denied(call_id):
                await connection.execute(
                    "INSERT OR IGNORE INTO sparra_turn_decisions "
                    "VALUES (?,?,?,?,?,NULL,NULL,NULL,1)",
                    (
                        str(call_id),
                        str(capture_id),
                        str(capture_id),
                        "capture-loss",
                        hashlib.sha256(capture_id.bytes).digest(),
                    ),
                )
            return True
        if action == "ack_done":
            token = self._required_uuid(values, "lease_token")
            await connection.execute(
                "UPDATE sparra_content_fences SET lease_acked=?,lease_settled=1 "
                "WHERE call_id=? AND lease_token=?",
                (int(values.get("acknowledged") is True), str(call_id), str(token)),
            )
            return True
        if action == "read":
            return await self._retained_call(call_id)
        if action == "frozen_read":
            return (
                None
                if await self._content_denied(call_id)
                else await self._frozen_publication(call_id)
            )
        if action == "freeze":
            if await self._content_denied(call_id):
                return None
            existing = await self._frozen_publication(call_id)
            if existing is not None:
                await self._insert_outbox(existing)
                return existing
            operation = require_operation(values)
            if not isinstance(operation.payload, CallUpsertPayloadV1):
                raise CommandSerializationError("result_publication_invalid")
            retained = await self._retained_call(call_id)
            changes: dict[str, object] = {"transcript_loss_count": retained.loss_count}
            result = values.get("message_result")
            facts = await self._read_call_lifecycle(call_id)
            permitted = values.get("result_permitted")
            if (
                isinstance(result, MessageResultV1)
                and (permitted is None or (callable(permitted) and permitted()))
                and not (facts is not None and facts.transfer_fenced)
            ):
                callback = values.get("provider_callback")
                validate_result_provenance(
                    result, retained, callback if isinstance(callback, str) else None
                )
                changes["message_result"] = encrypt_message_result(
                    result,
                    call_id=call_id,
                    keyring=self._keyring,
                    authenticated_turns={t.turn_id: t.role for t in retained.turns},
                )
            operation = operation.model_copy(
                update={"payload": operation.payload.model_copy(update=changes)}
            )
            prepared = encrypt_operation(operation, self._keyring)
            await connection.execute(
                "INSERT INTO sparra_publications VALUES (?,?,?,?,?,?)",
                (
                    str(call_id),
                    str(operation.operation_id),
                    operation.deployment_id,
                    prepared.encrypted.key_version,
                    prepared.encrypted.nonce,
                    prepared.encrypted.ciphertext,
                ),
            )
            await self._insert_outbox(operation)
            return operation
        if action == "erase":
            facts = await self._read_call_lifecycle(call_id)
            generation_value = values.get("generation")
            durable_generation = (
                None
                if facts is None
                else (
                    facts.admission_generation
                    or facts.transfer_generation
                    or facts.content_departed_generation
                )
            )
            if (
                isinstance(generation_value, UUID)
                and durable_generation is not None
                and (generation_value != durable_generation)
            ):
                raise CommandConflictError("content_generation_conflict")
            generation_value = durable_generation or generation_value
            cursor = await connection.execute(
                "SELECT state FROM call_leases WHERE call_id=?", (str(call_id),)
            )
            lease_state = await cursor.fetchone()
            await cursor.close()
            if (
                lease_state is not None
                and lease_state[0] != "terminal"
                and (facts is None or not isinstance(generation_value, UUID))
            ):
                raise FatalPersistenceError("content_recovery_generation_unavailable")
            expected = values.get("expected_item")
            if isinstance(expected, OutboxItem):
                cursor = await connection.execute(
                    "SELECT call_id,attempts FROM outbox WHERE queue_id=?", (expected.queue_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is not None and row != (str(call_id), expected.claim_attempt):
                    return None
            now = self._required_datetime(values, "now")
            lease_token_value = values.get("lease_token")
            cursor = await connection.execute(
                "SELECT recording_id FROM recording_archives WHERE call_id=? ORDER BY recording_id",
                (str(call_id),),
            )
            audio_rows = await cursor.fetchall()
            await cursor.close()
            recording_ids = tuple(UUID(row[0]) for row in audio_rows)
            cursor = await connection.execute(
                "SELECT cleaned_at,lease_token,lease_cleaned_at FROM sparra_content_fences "
                "WHERE call_id=?",
                (str(call_id),),
            )
            old = await cursor.fetchone()
            await cursor.close()
            cleaned = now if old is None else _parse_datetime(old[0])
            lease_cleaned = (
                _parse_datetime(old[2])
                if old is not None and old[1] == str(lease_token_value) and old[2] is not None
                else now
            )
            cleanup_update = (
                "lease_cleaned_at=NULL,"
                if recording_ids
                else "lease_cleaned_at=COALESCE(excluded.lease_cleaned_at,lease_cleaned_at),"
            )
            await connection.execute(
                "INSERT INTO sparra_content_fences"
                "(call_id,cleaned_at,lease_token,lease_cleaned_at) "
                "VALUES (?,?,?,?) ON CONFLICT(call_id) DO UPDATE "
                "SET lease_token=COALESCE(excluded.lease_token,lease_token),"
                + cleanup_update
                + "lease_acked=0,lease_settled=0",
                (
                    str(call_id),
                    _iso(cleaned),
                    str(lease_token_value) if isinstance(lease_token_value, UUID) else None,
                    _iso(lease_cleaned)
                    if isinstance(lease_token_value, UUID) and not recording_ids
                    else None,
                ),
            )
            for table in ("sparra_turn_decisions", "sparra_publications"):
                await connection.execute(f"DELETE FROM {table} WHERE call_id=?", (str(call_id),))
            # Genuine recording identity remains deliverable into native purge storage.
            await connection.execute(
                "DELETE FROM outbox WHERE call_id=? AND kind!='recording.upsert'", (str(call_id),)
            )
            if self._contract_version == 2:
                await connection.execute(
                    "UPDATE local_audio_pin SET denied_at=COALESCE(denied_at,?) WHERE call_id=?",
                    (_iso(now), str(call_id)),
                )
            if facts is not None:
                await self._store_lifecycle(
                    replace(
                        facts,
                        disclosure_evidence=None,
                        content_erased=True,
                        audio_reserved_bytes=facts.audio_reserved_bytes if recording_ids else 0,
                        content_departed_generation=facts.content_departed_generation
                        or (generation_value if isinstance(generation_value, UUID) else None),
                        local_closing_at=facts.local_closing_at or now,
                    )
                )
            if recording_ids:
                return AudioErasurePlan(recording_ids, lease_cleaned)
            return lease_cleaned if isinstance(lease_token_value, UUID) else cleaned
        raise CommandSerializationError("content_action_invalid")

    async def assert_sparra_compatible(self) -> None:
        await self.commit_control(PersistenceCommand("sparra_activation", {}, None))

    async def commit_transfer_intent(self, facts: LocalCallLifecycleFacts) -> None:
        await self.commit_control(PersistenceCommand("transfer_intent", {"facts": facts}, None))

    async def commit_transfer_observation(
        self,
        facts: LocalCallLifecycleFacts,
        operation: VoiceOperationV1 | None,
    ) -> None:
        await self.commit_control(
            PersistenceCommand(
                "transfer_observation",
                {"facts": facts, "operation": operation},
                None,
            )
        )

    async def _read_call_lifecycle(self, call_id: object) -> LocalCallLifecycleFacts | None:
        if not isinstance(call_id, UUID):
            raise CommandSerializationError("local_call_id_invalid")
        cursor = await self._require_owner_connection().execute(
            "SELECT lifecycle_json FROM call_leases WHERE call_id = ?",
            (str(call_id),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None or row[0] is None:
            return None
        return self._decode_lifecycle(row[0])

    @staticmethod
    def _decode_lifecycle(raw: str) -> LocalCallLifecycleFacts:
        values = json.loads(raw)
        for name in (
            "call_id",
            "transfer_command_id",
            "transfer_generation",
            "bridge_operation_id",
            "content_departed_generation",
            "admission_generation",
        ):
            if values.get(name) is not None:
                values[name] = UUID(values[name])
        for name in (
            "admitted_at",
            "retention_until",
            "started_at",
            "qualified_line_bridged_at",
            "transfer_failed_at",
            "local_closing_at",
            "original_ended_at",
        ):
            if values.get(name) is not None:
                values[name] = _parse_datetime(values[name])
        if values.get("disclosure_evidence") is not None:
            values["disclosure_evidence"] = DisclosureEvidenceV1.model_validate(
                values["disclosure_evidence"]
            )
        return LocalCallLifecycleFacts(**values)

    async def _store_lifecycle(self, facts: LocalCallLifecycleFacts) -> None:
        if await self._content_fenced(facts.call_id):
            facts = replace(facts, disclosure_evidence=None, content_erased=True)
        values: dict[str, object] = {}
        for attribute in fields(facts):
            value = getattr(facts, attribute.name)
            values[attribute.name] = (
                value.model_dump(mode="json")
                if isinstance(value, DisclosureEvidenceV1)
                else _iso(value)
                if isinstance(value, datetime)
                else str(value)
                if isinstance(value, UUID)
                else value
            )
        await self._require_owner_connection().execute(
            "UPDATE call_leases SET lifecycle_json = ? WHERE call_id = ?",
            (json.dumps(values, sort_keys=True, separators=(",", ":")), str(facts.call_id)),
        )

    async def _apply_audio_terminal(self, values: Mapping[str, object]) -> bool:
        operation = values.get("operation")
        generation = values.get("generation")
        if (
            self._contract_version != 2 or not isinstance(operation, VoiceOperationV2)
            or not isinstance(generation, UUID)
            or not isinstance(operation.payload, (AudioFinishPayloadV2, AudioRevokePayloadV2))
        ):
            return False
        payload = operation.payload
        pin = await self._read_audio_pin(operation.call_id)
        facts = await self._read_call_lifecycle(operation.call_id)
        if (
            pin is None or facts is None or pin.generation != generation
            or facts.admission_generation != generation
            or operation.deployment_id != pin.deployment_id
            or payload.workspace_id != pin.workspace_id or payload.recording_id != pin.recording_id
            or payload.configuration_revision != pin.revision
            or payload.retention_until != pin.retention_until
            or facts.retention_until != pin.retention_until
        ):
            return False
        connection = self._require_owner_connection()
        plaintext = canonical_operation_bytes(operation)
        fingerprint = hashlib.sha256(plaintext).digest()
        cursor = await connection.execute(
            "SELECT op_id,fingerprint,key_version,nonce,ciphertext,acked FROM local_audio_terminal "
            "WHERE call_id=? AND kind=?", (str(operation.call_id), operation.kind),
        )
        existing = await cursor.fetchone()
        await cursor.close()
        if existing is not None:
            if existing[0] != str(operation.operation_id) or existing[1] != fingerprint:
                return False
            encrypted = EncryptedValue(existing[2], existing[3], existing[4])
            if self._keyring.decrypt(encrypted, aad=operation_aad(operation)) != plaintext:
                raise FatalPersistenceError("audio_terminal_invalid")
            if existing[5] == 1:
                return True
        else:
            if isinstance(payload, AudioFinishPayloadV2) and (
                pin.committed_total_samples is None or pin.denied
                or pin.retention_until <= self._utcnow()
                or facts.content_erased
                or payload.last_sequence != pin.committed_last_sequence
                or payload.total_samples != pin.committed_total_samples
            ):
                return False
            if await self._load_operation_by_id(str(operation.operation_id)) is not None:
                return False
            prepared = encrypt_audio_operation(operation, self._keyring)
            encrypted = prepared.encrypted
            await connection.execute(
                "INSERT INTO local_audio_terminal(call_id,kind,op_id,deployment_id,fingerprint,"
                "key_version,nonce,ciphertext) VALUES(?,?,?,?,?,?,?,?)",
                (str(operation.call_id), operation.kind, str(operation.operation_id),
                 operation.deployment_id, fingerprint, encrypted.key_version, encrypted.nonce,
                 encrypted.ciphertext),
            )
        if isinstance(payload, AudioRevokePayloadV2):
            await connection.execute(
                "UPDATE local_audio_pin SET choice_state='off',denied_at=COALESCE(denied_at,?) "
                "WHERE call_id=?", (_iso(self._utcnow()), str(operation.call_id)),
            )
            await connection.execute(
                "DELETE FROM outbox WHERE call_id=? AND kind IN ('audio.chunk','audio.finish')",
                (str(operation.call_id),),
            )
        existing_outbox = await self._load_operation_by_id(str(operation.operation_id))
        if existing_outbox is not None:
            if canonical_operation_bytes(existing_outbox) != plaintext:
                raise CommandConflictError("audio_terminal_conflict")
            return True
        now = _iso(self._utcnow())
        await connection.execute(
            "INSERT INTO outbox(op_id,deployment_id,kind,schema_version,call_id,recording_id,"
            "crypto_version,key_version,nonce,ciphertext,created_at,next_attempt_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(operation.operation_id), operation.deployment_id, operation.kind, 2,
             str(operation.call_id), str(payload.recording_id), CRYPTO_VERSION,
             encrypted.key_version,
             encrypted.nonce, encrypted.ciphertext, now, now),
        )
        return True

    async def _apply_audio_choice(self, values: Mapping[str, object]) -> AudioChoiceFacts | None:
        call_id = self._required_uuid(values, "call_id")
        generation = values.get("generation")
        choice = values.get("choice")
        if self._contract_version != 2 or not isinstance(generation, UUID):
            return None
        pin = await self._read_audio_pin(call_id)
        facts = await self._read_call_lifecycle(call_id)
        if (
            pin is None or facts is None or pin.generation != generation
            or facts.admission_generation != generation
            or pin.admitted_at != facts.admitted_at or pin.retention_until != facts.retention_until
            or await self._content_fenced(call_id)
        ):
            return None
        now = self._utcnow()
        connection = self._require_owner_connection()
        if choice == "off":
            await connection.execute(
                "UPDATE local_audio_pin SET choice_state='off',denied_at=COALESCE(denied_at,?) "
                "WHERE call_id=?", (_iso(now), str(call_id)),
            )
        elif choice == "accept":
            occurred_at = values.get("occurred_at")
            if (
                pin.denied or pin.choice_state == "off" or not pin.available
                or pin.policy != "local_30d" or pin.retention_until <= now or facts.transfer_fenced
                or not isinstance(occurred_at, datetime) or occurred_at.tzinfo is None
                or occurred_at.utcoffset() is None
            ):
                return None
            occurred_at = occurred_at.astimezone(UTC)
            if pin.choice_state == "accepted":
                if occurred_at != pin.choice_occurred_at:
                    return None
            else:
                evidence = facts.disclosure_evidence
                if (
                    evidence is None or evidence.completed_at is None
                    or evidence.failed_at is not None
                    or evidence.input_gate_opened_at is not None
                    or not evidence.completed_at <= occurred_at <= now
                    or now >= evidence.completed_at + timedelta(seconds=5)
                ):
                    return None
                await connection.execute(
                    "UPDATE local_audio_pin SET choice_state='accepted',choice_occurred_at=? "
                    "WHERE call_id=?", (_iso(occurred_at), str(call_id)),
                )
        else:
            return None
        committed_pin = await self._read_audio_pin(call_id)
        if committed_pin is None:
            raise FatalPersistenceError("audio_pin_invalid")
        return AudioChoiceFacts(call_id, generation, committed_pin.choice_state,
                                committed_pin.choice_occurred_at, committed_pin.denied_at)

    async def _apply_control_v2_command(self, values: Mapping[str, object]) -> bool:
        prepared = values.get("prepared")
        generation = values.get("generation")
        if (
            self._contract_version != 2 or not isinstance(prepared, PreparedControlOperationV2)
            or not isinstance(generation, UUID)
        ):
            raise CommandSerializationError("invalid_control_v2_command")
        operation = prepared.operation
        payload = operation.payload
        if operation.kind != "call.upsert" or not isinstance(payload, CallUpsertPayloadV1):
            raise CommandSerializationError("invalid_control_v2_command")
        facts = await self._read_call_lifecycle(operation.call_id)
        pin = await self._read_audio_pin(operation.call_id)
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            "SELECT call_control_id,tenant_id,agent_id,created_at FROM call_leases WHERE call_id=?",
            (str(operation.call_id),),
        )
        leases = list(await cursor.fetchall())
        await cursor.close()
        if (
            facts is None or pin is None or facts.admission_generation != generation
            or pin.generation != generation or pin.admitted_at != facts.admitted_at
            or pin.retention_until != facts.retention_until
            or payload.retention_until != facts.retention_until
            or facts.retention_until <= self._utcnow() or facts.transfer_fenced
            or await self._content_denied(operation.call_id)
            or operation.deployment_id != pin.deployment_id or len(leases) != 1
            or leases[0][0] != payload.telnyx_call_control_id
            or leases[0][1] != str(pin.workspace_id) or leases[0][2] != pin.deployment_id
            or _parse_datetime(leases[0][3]) != facts.admitted_at
            or payload.telnyx_call_leg_id != facts.telnyx_call_leg_id
            or payload.telnyx_call_session_id != facts.telnyx_call_session_id
        ):
            return False
        if (
            prepared.aad != operation_aad(operation)
            or prepared.plaintext != canonical_operation_bytes(operation)
            or self._keyring.decrypt(prepared.encrypted, aad=prepared.aad) != prepared.plaintext
        ):
            raise CommandSerializationError("invalid_control_v2_command")
        existing = await self._load_operation_by_id(str(operation.operation_id))
        if existing is not None:
            if canonical_operation_bytes(existing) != prepared.plaintext:
                raise CommandConflictError("operation_identity_conflict")
            return True
        await self._merge_lifecycle_operation(operation)
        encrypted = prepared.encrypted
        created = _iso(self._utcnow())
        await connection.execute(
            "INSERT INTO outbox(op_id,deployment_id,kind,schema_version,call_id,crypto_version,"
            "key_version,nonce,ciphertext,created_at,next_attempt_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (str(operation.operation_id), operation.deployment_id, operation.kind, 2,
             str(operation.call_id), CRYPTO_VERSION, encrypted.key_version, encrypted.nonce,
             encrypted.ciphertext, created, created),
        )
        return True

    async def _merge_lifecycle_operation(
        self, operation: VoiceOperationV1 | VoiceOperationV2
    ) -> None:
        facts = await self._read_call_lifecycle(operation.call_id)
        if facts is None or not isinstance(operation.payload, CallUpsertPayloadV1):
            return
        payload = operation.payload
        if payload.retention_until != facts.retention_until:
            raise CommandConflictError("local_retention_conflict")
        evidence = facts.disclosure_evidence
        if payload.disclosure_evidence is not None:
            incoming = payload.disclosure_evidence
            if evidence is None:
                evidence = incoming
            else:
                evidence = DisclosureEvidenceV1(
                    schema_version=1,
                    **{
                        name: getattr(evidence, name) or getattr(incoming, name)
                        for name in (
                            "started_at",
                            "completed_at",
                            "failed_at",
                            "input_gate_opened_at",
                        )
                    },
                )
        started = facts.started_at
        if payload.started_at is not None:
            started = payload.started_at if started is None else min(started, payload.started_at)
        await self._store_lifecycle(
            replace(
                facts,
                started_at=started,
                disclosure_evidence=evidence,
                local_closing_at=facts.local_closing_at
                or (
                    operation.occurred_at
                    if payload.status == "closing" and facts.transfer_command_id is not None
                    else None
                ),
            )
        )

    async def _apply_transfer_facts(self, payload: Mapping[str, object]) -> None:
        facts = payload.get("facts")
        if not isinstance(facts, LocalCallLifecycleFacts):
            raise CommandSerializationError("transfer_facts_invalid")
        current = await self._read_call_lifecycle(facts.call_id)
        if current is None or current.admitted_at != facts.admitted_at:
            raise CommandConflictError("transfer_admission_missing")
        if current.admission_generation is not None and (
            facts.admission_generation not in {None, current.admission_generation}
            or facts.transfer_generation not in {None, current.admission_generation}
        ):
            raise CommandConflictError("admission_generation_conflict")
        if current.transfer_command_id is not None and (
            current.transfer_command_id != facts.transfer_command_id
            or current.transfer_correlation != facts.transfer_correlation
            or current.transfer_generation != facts.transfer_generation
            or current.transfer_connection_sha256 != facts.transfer_connection_sha256
            or current.transfer_destination_sha256 != facts.transfer_destination_sha256
        ):
            raise CommandConflictError("transfer_identity_conflict")
        for name in (
            "target_call_control_id",
            "target_call_leg_id",
            "qualified_line_bridged_at",
            "bridge_operation_id",
        ):
            old, new = getattr(current, name), getattr(facts, name)
            if old is not None:
                if new is not None and new != old:
                    raise CommandConflictError("transfer_observation_conflict")
                # Older in-flight observations may omit newer durable facts.
                facts = replace(facts, **{name: old})
        await self._store_lifecycle(
            replace(
                facts,
                started_at=current.started_at,
                disclosure_evidence=current.disclosure_evidence,
                local_closing_at=current.local_closing_at or facts.local_closing_at,
                content_erased=current.content_erased or facts.content_erased,
                admission_generation=current.admission_generation,
                original_ended_at=current.original_ended_at,
                recording_policy_revision=current.recording_policy_revision,
                recording_enabled=current.recording_enabled,
                audio_reserved_bytes=current.audio_reserved_bytes,
                content_departed_generation=current.content_departed_generation
                or facts.content_departed_generation,
            )
        )
        operation = payload.get("operation")
        if operation is not None:
            if not isinstance(operation, VoiceOperationV1) or operation.call_id != facts.call_id:
                raise CommandSerializationError("transfer_operation_invalid")
            await self._insert_outbox(operation)

    async def drain(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._accepting = False
        if not self._run_started or self._closed_event.is_set():
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        command = PersistenceCommand("shutdown", {}, future, enqueued_at=self._monotonic())
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
                    self._quick_check_interval_seconds - (self._monotonic() - self._last_check_at),
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
                    and self._monotonic() - current_command.enqueued_at > QUEUE_OLDEST_LIMIT_SECONDS
                ):
                    raise FatalPersistenceError("queue_oldest_age_exceeded")

                try:
                    should_stop, command_result = await self._process_command(current_command)
                except _AudioChunkRefused:
                    # The owner confirmed rollback; no content was admitted.
                    result = current_command.payload.get("result")
                    if not isinstance(result, asyncio.Future):
                        raise CommandSerializationError("invalid_audio_command") from None
                    if not result.done():
                        result.set_exception(PersistenceError("audio_chunk_refused"))
                    should_stop = False
                else:
                    self._resolve_success(current_command, command_result)
                if should_stop:
                    self._queue.task_done()
                    current_owned = False
                    current_command = None
                    break
                await self._run_periodic_check_if_due()
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
                    (command for command in self._pending_commands if command.kind != "shutdown"),
                    None,
                )
                if oldest is None:
                    self._queue_watchdog_wakeup.clear()
                    await self._queue_watchdog_wakeup.wait()
                    continue

                remaining = QUEUE_OLDEST_LIMIT_SECONDS - (self._monotonic() - oldest.enqueued_at)
                if remaining < 0.0:
                    self._signal_fatal("queue_oldest_age_exceeded")
                    return

                self._queue_watchdog_wakeup.clear()
                try:
                    await asyncio.wait_for(self._queue_watchdog_wakeup.wait(), timeout=remaining)
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
        if self._contract_version == 2:
            await self._initialize_audio_owner_connection()
            return
        if (
            self._pragma_int(await self._pragma_scalar("user_version"))
            in {LOCAL_AUDIO_SCHEMA_VERSION, LOCAL_AUDIO_CHOICE_SCHEMA_VERSION,
                LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION}
        ):
            raise FatalPersistenceError("writer_contract_mismatch")
        connection = self._require_owner_connection()
        cursor = await connection.execute("PRAGMA journal_mode=DELETE")
        journal_row = await cursor.fetchone()
        await cursor.close()
        await connection.execute("PRAGMA synchronous=EXTRA")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute("PRAGMA secure_delete=ON")
        existing_version = self._pragma_int(await self._pragma_scalar("user_version"))
        existing_schema = await self._application_schema_objects()
        if existing_schema:
            if (
                (existing_version == 1 and existing_schema == _EXPECTED_V1_SCHEMA_OBJECTS)
                or (existing_version == 2 and existing_schema == _EXPECTED_V2_SCHEMA_OBJECTS)
                or (existing_version == 3 and existing_schema == _EXPECTED_V3_SCHEMA_OBJECTS)
                or (existing_version == 4 and existing_schema == _EXPECTED_V4_SCHEMA_OBJECTS)
            ):
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    if existing_version == 1:
                        await connection.execute(QUALIFICATION_RUNS_SQL)
                    if existing_version < 3:
                        await connection.execute(CALL_LIFECYCLE_MIGRATION_SQL)
                    if existing_version < 4:
                        for statement in SPARRA_CONTENT_SQL.split(";"):
                            if statement.strip():
                                await connection.execute(statement)
                    for statement in RECORDING_ARCHIVE_SQL.split(";"):
                        if statement.strip():
                            await connection.execute(statement)
                    await connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    if (
                        self._pragma_int(await self._pragma_scalar("user_version"))
                        != SCHEMA_VERSION
                        or await self._application_schema_objects() != _EXPECTED_SCHEMA_OBJECTS
                    ):
                        raise FatalPersistenceError("sqlite_schema_mismatch")
                    await self._call_failpoint("after_v1_migration_before_commit")
                    await connection.commit()
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await connection.rollback()
                    raise
                except BaseException:
                    with contextlib.suppress(Exception):
                        await connection.rollback()
                    raise FatalPersistenceError("sqlite_schema_mismatch") from None
            elif existing_version != SCHEMA_VERSION or existing_schema != _EXPECTED_SCHEMA_OBJECTS:
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
        if await self._pragma_scalar("secure_delete") != 1:
            raise FatalPersistenceError("sqlite_secure_delete_required")
        journal_mode = str(journal_row[0]).lower() if journal_row else ""
        self.pragma_state = {
            "journal_mode": journal_mode,
            "synchronous": synchronous,
            "foreign_keys": foreign_keys,
            "secure_delete": 1,
        }
        if self.pragma_state != {
            "journal_mode": "delete",
            "synchronous": 3,
            "foreign_keys": 1,
            "secure_delete": 1,
        }:
            raise FatalPersistenceError("sqlite_pragma_mismatch")

    async def _initialize_audio_owner_connection(self) -> None:
        connection = self._require_owner_connection()
        version = self._pragma_int(await self._pragma_scalar("user_version"))
        schema = await self._application_schema_objects()
        if schema:
            expected = {
                LOCAL_AUDIO_SCHEMA_VERSION: _EXPECTED_AUDIO_SCHEMA_OBJECTS,
                LOCAL_AUDIO_CHOICE_SCHEMA_VERSION: _EXPECTED_AUDIO_CHOICE_SCHEMA_OBJECTS,
                LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION: _EXPECTED_AUDIO_TERMINAL_SCHEMA_OBJECTS,
            }.get(version)
            if (
                expected is None or schema != expected
            ):
                raise FatalPersistenceError("writer_contract_mismatch")
            cursor = await connection.execute(
                "SELECT singleton,contract_version FROM local_audio_contract"
            )
            marker = list(await cursor.fetchall())
            await cursor.close()
            cursor = await connection.execute("SELECT count(*) FROM outbox WHERE schema_version!=2")
            legacy = await cursor.fetchone()
            await cursor.close()
            if marker != [(1, 2)] or legacy != (0,):
                raise FatalPersistenceError("writer_contract_mismatch")
            if version != LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    if version == LOCAL_AUDIO_SCHEMA_VERSION:
                        for statement in LOCAL_AUDIO_CHOICE_MIGRATION_SQL.split(";"):
                            if statement.strip():
                                await connection.execute(statement)
                        await self._call_failpoint("after_audio_choice_migration_before_commit")
                    for statement in LOCAL_AUDIO_TERMINAL_MIGRATION_SQL.split(";"):
                        if statement.strip():
                            await connection.execute(statement)
                    await self._call_failpoint("after_audio_terminal_migration_before_commit")
                    self._check_storage_limit()
                    await connection.commit()
                except BaseException:
                    with contextlib.suppress(Exception):
                        await connection.rollback()
                    raise
        elif version != 0:
            raise FatalPersistenceError("sqlite_schema_mismatch")
        # Contract and provenance are checked before any persistent PRAGMA or recovery write.
        cursor = await connection.execute("PRAGMA journal_mode=DELETE")
        journal = await cursor.fetchone()
        await cursor.close()
        await connection.execute("PRAGMA synchronous=EXTRA")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute("PRAGMA secure_delete=ON")
        if not schema:
            try:
                await connection.executescript(
                    "BEGIN IMMEDIATE;\n" + LOCAL_AUDIO_TERMINAL_SCHEMA_SQL + "\nCOMMIT;"
                )
            except BaseException:
                with contextlib.suppress(Exception):
                    await connection.rollback()
                raise
        if (
            self._pragma_int(await self._pragma_scalar("user_version"))
            != LOCAL_AUDIO_TERMINAL_SCHEMA_VERSION
            or await self._application_schema_objects() != _EXPECTED_AUDIO_TERMINAL_SCHEMA_OBJECTS
        ):
            raise FatalPersistenceError("sqlite_schema_mismatch")
        self.pragma_state = {
            "journal_mode": str(journal[0]).lower() if journal else "",
            "synchronous": self._pragma_int(await self._pragma_scalar("synchronous")),
            "foreign_keys": self._pragma_int(await self._pragma_scalar("foreign_keys")),
            "secure_delete": self._pragma_int(await self._pragma_scalar("secure_delete")),
        }
        if self.pragma_state != {"journal_mode": "delete", "synchronous": 3,
                                 "foreign_keys": 1, "secure_delete": 1}:
            raise FatalPersistenceError("sqlite_pragma_mismatch")
        self._audio_page_size = self._pragma_int(await self._pragma_scalar("page_size"))

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
            if (
                not isinstance(object_type, str)
                or not isinstance(name, str)
                or not isinstance(sql, str)
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
        total = self._measure_storage_bytes()
        if total > self.max_storage_bytes:
            raise FatalPersistenceError("storage_limit_exceeded")

    def _measure_storage_bytes(self) -> int:
        journal_path = Path(f"{self._database_path}-journal")
        try:
            total = self._file_size(self._database_path) + self._file_size(journal_path)
        except OSError as error:
            raise error
        if type(total) is not int or total < 0:
            raise FatalPersistenceError("storage_measurement_invalid")
        return total

    async def _call_failpoint(self, name: str) -> None:
        if self._failpoint is None:
            return
        result = self._failpoint(name)
        if inspect.isawaitable(result):
            await result

    async def _process_command(self, command: PersistenceCommand) -> tuple[bool, object | None]:
        if command.kind == "shutdown":
            return True, None

        connection = self._require_owner_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            command_result: object | None = None
            if command.kind == "outbox":
                if self._contract_version != 1:
                    raise CommandSerializationError("writer_contract_mismatch")
                await self._insert_outbox(
                    require_operation(command.payload),
                    truncated=command.payload.get("truncated") is True,
                )
            elif command.kind == "sparra_content":
                command_result = await self._apply_content_command(command.payload)
            elif command.kind == "recording_archive":
                command_result = await self._apply_archive_command(command.payload)
            elif command.kind == "lease":
                await self._apply_lease(command.payload)
                operation = command.payload.get("operation")
                if operation is not None and command.payload.get("action") == "upsert":
                    await self._insert_outbox(require_operation(command.payload))
            elif command.kind == "webhook_effect":
                command_result = await self._apply_webhook_effect(command.payload)
            elif command.kind == "webhook_receipt_status":
                command_result = await self._classify_webhook_receipt(command.payload)
            elif command.kind == "webhook_enrichment":
                await self._apply_webhook_enrichment(command.payload)
            elif command.kind == "relay_batch":
                await self._apply_relay_command(command.payload)
            elif command.kind == "qualification_run_status":
                command_result = await self._qualification_run_consumed(command.payload)
            elif command.kind == "call_lifecycle_read":
                command_result = await self._read_call_lifecycle(command.payload["call_id"])
            elif command.kind == "sparra_activation":
                cursor = await connection.execute(
                    "SELECT count(*) FROM call_leases "
                    "WHERE state != 'terminal' AND lifecycle_json IS NULL",
                )
                missing = await cursor.fetchone()
                await cursor.close()
                cursor = await connection.execute(
                    "SELECT count(*) FROM outbox o LEFT JOIN call_leases c "
                    "ON c.call_id=o.call_id WHERE c.lifecycle_json IS NULL",
                )
                incompatible = await cursor.fetchone()
                await cursor.close()
                if missing != (0,) or incompatible != (0,):
                    raise FatalPersistenceError("sparra_legacy_state_incompatible")
                cursor = await connection.execute(
                    "SELECT count(*) FROM call_leases WHERE state!='terminal' "
                    "AND json_extract(lifecycle_json,'$.admission_generation') IS NULL "
                    "AND json_extract(lifecycle_json,'$.transfer_generation') IS NULL "
                    "AND json_extract(lifecycle_json,'$.content_departed_generation') IS NULL"
                )
                missing_generation = await cursor.fetchone()
                await cursor.close()
                if missing_generation != (0,):
                    raise FatalPersistenceError("sparra_legacy_generation_incompatible")
                cursor = await connection.execute(
                    "SELECT count(*) FROM outbox o JOIN call_leases c ON c.call_id=o.call_id "
                    "LEFT JOIN sparra_turn_decisions d ON d.call_id=o.call_id "
                    "AND d.turn_id=o.turn_id WHERE o.kind='turn.upsert' "
                    "AND c.lifecycle_json IS NOT NULL AND d.turn_id IS NULL"
                )
                unqualified_turns = await cursor.fetchone()
                await cursor.close()
                if unqualified_turns != (0,):
                    raise FatalPersistenceError("sparra_legacy_state_incompatible")
                self._sparra_active = True
            elif command.kind in {"transfer_intent", "transfer_observation"}:
                await self._apply_transfer_facts(command.payload)
            else:
                raise CommandSerializationError("unknown_persistence_command")
            await self._call_failpoint("after_mutation_before_commit")
            if (
                command.kind == "recording_archive"
                and command.payload.get("action") == "finish"
                and isinstance(command_result, RecordingArchiveJob)
                and not self._archive_publication_budget(command.payload, command_result)
            ):
                await connection.rollback()
                return False, None
            self._check_storage_limit()
            await connection.commit()
            if self._contract_version == 2 and command.kind == "sparra_content":
                action = command.payload.get("action")
                if action == "audio_choice" and isinstance(command_result, AudioChoiceFacts):
                    refreshed = await self._read_audio_pin(command_result.call_id)
                    if refreshed is not None:
                        self._audio_pins[refreshed.call_id] = refreshed
                elif action == "control_v2" and command_result is True:
                    prepared = command.payload.get("prepared")
                    if isinstance(prepared, PreparedControlOperationV2):
                        refreshed = await self._read_audio_pin(prepared.operation.call_id)
                        if refreshed is not None:
                            self._audio_pins[refreshed.call_id] = refreshed
                elif action == "audio_chunk":
                    audio_prepared = command.payload.get("prepared")
                    if isinstance(audio_prepared, PreparedAudioOperation):
                        refreshed = await self._read_audio_pin(audio_prepared.operation.call_id)
                        if refreshed is not None:
                            self._audio_pins[refreshed.call_id] = refreshed
                elif action == "audio_terminal" and command_result is True:
                    audio_terminal = command.payload.get("operation")
                    if isinstance(audio_terminal, VoiceOperationV2):
                        refreshed = await self._read_audio_pin(audio_terminal.call_id)
                        if refreshed is not None:
                            self._audio_pins[refreshed.call_id] = refreshed
            if (
                self._contract_version == 2 and command.kind == "sparra_content"
                and command.payload.get("action") == "erase"
            ):
                self._audio_pins.pop(self._required_uuid(command.payload, "call_id"), None)
        except _AudioChunkRefused:
            # Failed rollback remains a genuine fatal owner failure.
            await connection.rollback()
            raise
        except BaseException:
            rolled_back = False
            try:
                await connection.rollback()
                rolled_back = True
            except BaseException:
                pass
            if command.kind == "recording_archive" and command.payload.get("action") == "finish":
                # An exception from COMMIT can follow a successful native commit.
                # Only a successful rollback plus an authoritative receipt-less
                # same-owner read proves the receipt transaction did not commit.
                definitely_uncommitted = False
                if rolled_back:
                    try:
                        recording_id = self._required_uuid(command.payload, "recording_id")
                        stored = await self._read_recording_archive(recording_id)
                        definitely_uncommitted = stored is not None and stored.receipt is None
                    except BaseException:
                        pass
                raise FatalPersistenceError(
                    "recording_archive_rolled_back"
                    if definitely_uncommitted
                    else "recording_archive_commit_unknown"
                ) from None
            raise
        return False, command_result

    async def _apply_stale_terminal(self, payload: Mapping[str, object]) -> None:
        stale = payload.get("stale")
        closed_at = payload.get("closed_at")
        operation = payload.get("operation")
        if (
            not isinstance(stale, StaleLease)
            or not isinstance(closed_at, datetime)
            or closed_at.tzinfo is None
            or closed_at.utcoffset() is None
            or not isinstance(operation, VoiceOperationV1)
            or operation.kind != "call.upsert"
            or operation.call_id != stale.call_id
        ):
            raise CommandSerializationError("stale_terminal_invalid")
        await self._apply_lease(
            {
                "action": "upsert",
                "call_control_id": stale.call_control_id,
                "call_id": stale.call_id,
                "tenant_id": stale.tenant_id,
                "agent_id": stale.agent_id,
                "state": "terminal",
                "token_hash": stale.token_hash,
                "created_at": stale.created_at,
                "expires_at": stale.expires_at,
                "closed_at": closed_at,
            }
        )
        await self._insert_outbox(operation)

    async def _classify_webhook_receipt(
        self, payload: Mapping[str, object]
    ) -> WebhookReceiptClassification:
        event_id = self._required_str(payload, "event_id")
        semantic_fingerprint = payload.get("semantic_fingerprint_sha256")
        if type(semantic_fingerprint) is not bytes or len(semantic_fingerprint) != 32:
            raise CommandSerializationError("invalid_webhook_receipt")
        legacy_fingerprint = payload.get("legacy_v1_semantic_fingerprint_sha256")
        if legacy_fingerprint is not None and (
            type(legacy_fingerprint) is not bytes or len(legacy_fingerprint) != 32
        ):
            raise CommandSerializationError("invalid_webhook_receipt")
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            "SELECT semantic_fingerprint_sha256 FROM webhook_receipts WHERE event_id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return "missing"
        if row == (semantic_fingerprint,) or row == (legacy_fingerprint,):
            return "duplicate"
        return "conflict"

    async def _insert_outbox(self, operation: VoiceOperationV1, *, truncated: bool = False) -> None:
        if self._contract_version != 1:
            raise CommandSerializationError("writer_contract_mismatch")
        connection = self._require_owner_connection()
        if operation.kind != "recording.upsert" and await self._content_denied(operation.call_id):
            return
        if isinstance(operation.payload, CallUpsertPayloadV1):
            await self._merge_lifecycle_operation(operation)
        if isinstance(operation.payload, TurnUpsertPayloadV1) and not await self._retain_turn(
            operation, truncated=truncated
        ):
            return
        cursor = await connection.execute(
            "SELECT key_version,nonce,ciphertext FROM sparra_publications "
            "WHERE call_id=? AND op_id=? UNION ALL "
            "SELECT key_version,nonce,ciphertext FROM sparra_turn_decisions "
            "WHERE call_id=? AND op_id=? AND ciphertext IS NOT NULL LIMIT 1",
            (
                str(operation.call_id),
                str(operation.operation_id),
                str(operation.call_id),
                str(operation.operation_id),
            ),
        )
        frozen = await cursor.fetchone()
        await cursor.close()
        encrypted = (
            EncryptedValue(frozen[0], frozen[1], frozen[2])
            if frozen is not None
            else encrypt_operation(operation, self._keyring).encrypted
        )
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
                    encrypted.key_version,
                    encrypted.nonce,
                    encrypted.ciphertext,
                    _iso(created_at),
                    _iso(created_at),
                ),
            )
        except sqlite3.IntegrityError:
            existing = await self._load_operation_by_id(str(operation.operation_id))
            if existing is None or canonical_operation_bytes(existing) != canonical_operation_bytes(
                operation
            ):
                raise CommandConflictError("operation_identity_conflict") from None

    async def _load_operation_by_id(
        self, operation_id: str
    ) -> VoiceOperationV1 | VoiceOperationV2 | None:
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
        if row[0] != self._contract_version:
            raise FatalPersistenceError("writer_contract_mismatch")
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
        return self._decode_fixed_operation(plaintext)

    def _decode_fixed_operation(self, plaintext: bytes) -> VoiceOperationV1 | VoiceOperationV2:
        if self._contract_version == 2:
            return decode_operation_v2(plaintext)
        return decode_operation(plaintext)

    async def _apply_webhook_effect(self, payload: Mapping[str, object]) -> WebhookCommitValue:
        receipt = payload.get("receipt")
        if not isinstance(receipt, Mapping):
            raise CommandSerializationError("invalid_webhook_receipt")
        normalized_receipt = self._normalized_receipt(receipt)
        legacy_fingerprint = payload.get("legacy_v1_semantic_fingerprint_sha256")
        if legacy_fingerprint is not None and (
            type(legacy_fingerprint) is not bytes or len(legacy_fingerprint) != 32
        ):
            raise CommandSerializationError("invalid_webhook_receipt")
        if await self._receipt_exists(normalized_receipt, legacy_fingerprint):
            await self._record_original_end(normalized_receipt, payload.get("lease"), receipt)
            return WebhookCommitResult("duplicate", "duplicate")
        qualification_run_id = payload.get("qualification_run_id")
        if qualification_run_id is not None:
            if not isinstance(qualification_run_id, UUID):
                raise CommandSerializationError("invalid_qualification_run")
            if await self._qualification_run_is_consumed(qualification_run_id):
                return QualificationRunConsumed()
        await self._insert_new_receipt(normalized_receipt)
        if qualification_run_id is not None:
            await self._consume_qualification_run(qualification_run_id)
        lease = payload.get("lease")
        operation = payload.get("operation")
        if lease is not None:
            if not isinstance(lease, Mapping):
                raise CommandSerializationError("invalid_lease_command")
            if await self._same_identity_is_terminal(lease):
                await self._record_original_end(normalized_receipt, payload.get("lease"), receipt)
                return WebhookCommitResult("first", "existing_terminal")
            await self._apply_lease(lease)
            admission = payload.get("admission_facts")
            if admission is not None:
                if not isinstance(
                    admission, LocalCallAdmissionFacts
                ) or admission.call_id != lease.get("call_id"):
                    raise CommandConflictError("local_admission_identity_conflict")
                if self._contract_version == 2 and operation is None:
                    if (
                        admission.admission_generation is None
                        or admission.admitted_at != lease.get("created_at")
                        or admission.retention_until != admission.admitted_at + timedelta(days=30)
                    ):
                        raise CommandConflictError("local_admission_identity_conflict")
                elif (
                    not isinstance(operation, VoiceOperationV1)
                    or not isinstance(operation.payload, CallUpsertPayloadV1)
                    or operation.call_id != admission.call_id
                    or operation.payload.retention_until != admission.retention_until
                    or operation.payload.telnyx_call_leg_id != admission.telnyx_call_leg_id
                    or operation.payload.telnyx_call_session_id != admission.telnyx_call_session_id
                ):
                    raise CommandConflictError("local_admission_identity_conflict")
                existing = await self._read_call_lifecycle(admission.call_id)
                facts = LocalCallLifecycleFacts(
                    admission.call_id,
                    admission.admitted_at,
                    admission.retention_until,
                    admission.telnyx_call_leg_id,
                    admission.telnyx_call_session_id,
                    admission_generation=admission.admission_generation,
                )
                if existing is not None and (
                    existing.admitted_at != facts.admitted_at
                    or existing.retention_until != facts.retention_until
                    or existing.telnyx_call_leg_id != facts.telnyx_call_leg_id
                    or existing.telnyx_call_session_id != facts.telnyx_call_session_id
                    or existing.admission_generation != facts.admission_generation
                ):
                    raise CommandConflictError("local_admission_identity_conflict")
                if existing is None:
                    await self._store_lifecycle(facts)
        await self._record_original_end(normalized_receipt, payload.get("lease"), receipt)
        if operation is not None:
            if not isinstance(operation, VoiceOperationV1):
                raise CommandSerializationError("invalid_outbox_command")
            await self._insert_outbox(operation)
        return WebhookCommitResult("first", "applied")

    async def _record_original_end(
        self,
        receipt: tuple[str, str, str | None, str, str, bytes],
        lease: object,
        verified_identity: Mapping[str, object],
    ) -> None:
        if receipt[1] != "call.hangup" or receipt[2] is None:
            return
        cursor = await self._require_owner_connection().execute(
            "SELECT lifecycle_json FROM call_leases WHERE call_control_id=? AND state='terminal'",
            (receipt[2],),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None or row[0] is None:
            return
        facts = self._decode_lifecycle(row[0])
        if facts.telnyx_call_leg_id is None or facts.telnyx_call_session_id is None:
            return  # Unknown original identity is not telephone-end authority.
        if "call_leg_id" in verified_identity and "call_session_id" in verified_identity:
            if (
                verified_identity["call_leg_id"] != facts.telnyx_call_leg_id
                or verified_identity["call_session_id"] != facts.telnyx_call_session_id
            ):
                return
        elif not (
            isinstance(lease, Mapping)
            and lease.get("state") == "terminal"
            and lease.get("call_control_id") == receipt[2]
            and lease.get("call_id") == facts.call_id
        ):
            return
        # A matching verified event can observe a previously local/historical
        # terminal row even when the registry has no remaining entry/effect.
        if facts.original_ended_at is None:
            await self._store_lifecycle(
                replace(facts, original_ended_at=_parse_datetime(receipt[3]))
            )

    def _normalized_receipt(
        self, receipt: Mapping[str, object]
    ) -> tuple[str, str, str | None, str, str, bytes]:
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
        return (
            event_id,
            event_type,
            call_control,
            occurred_at,
            received_at,
            semantic_fingerprint,
        )

    async def _receipt_exists(
        self,
        receipt: tuple[str, str, str | None, str, str, bytes],
        legacy_v1_semantic_fingerprint_sha256: bytes | None = None,
    ) -> bool:
        connection = self._require_owner_connection()
        event_id, event_type, call_control, occurred_at, _, semantic_fingerprint = receipt
        cursor = await connection.execute(
            """
            SELECT event_type, call_control_id, occurred_at, semantic_fingerprint_sha256
            FROM webhook_receipts WHERE event_id = ?
            """,
            (event_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return False
        if row[:3] != (event_type, call_control, occurred_at) or row[3] not in {
            semantic_fingerprint,
            legacy_v1_semantic_fingerprint_sha256,
        }:
            raise CommandConflictError("webhook_identity_conflict")
        return True

    async def _insert_new_receipt(
        self, receipt: tuple[str, str, str | None, str, str, bytes]
    ) -> None:
        connection = self._require_owner_connection()
        event_id, event_type, call_control, occurred_at, received_at, semantic_fingerprint = receipt
        cursor = await connection.execute(
            """
            INSERT OR IGNORE INTO webhook_receipts (
                event_id, event_type, call_control_id, occurred_at, received_at,
                semantic_fingerprint_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            receipt,
        )
        inserted = cursor.rowcount == 1
        await cursor.close()
        if not inserted:
            raise CommandConflictError("webhook_identity_conflict")

    async def _qualification_run_is_consumed(self, run_id: UUID) -> bool:
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            "SELECT 1 FROM qualification_runs WHERE run_id = ?",
            (str(run_id),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row == (1,)

    async def _consume_qualification_run(self, run_id: UUID) -> None:
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            "INSERT INTO qualification_runs (run_id, consumed_at) VALUES (?, ?)",
            (str(run_id), _iso(self._utcnow())),
        )
        if cursor.rowcount != 1:
            await cursor.close()
            raise CommandConflictError("qualification_run_conflict")
        await cursor.close()

    async def _qualification_run_consumed(self, payload: Mapping[str, object]) -> bool:
        run_id = payload.get("run_id")
        if not isinstance(run_id, UUID):
            raise CommandSerializationError("invalid_qualification_run")
        return await self._qualification_run_is_consumed(run_id)

    async def _same_identity_is_terminal(self, lease: Mapping[str, object]) -> bool:
        call_control_id = self._required_str(lease, "call_control_id")
        call_id = self._required_uuid(lease, "call_id")
        tenant_id = self._required_str(lease, "tenant_id")
        agent_id = self._required_str(lease, "agent_id")
        token_hash = lease.get("token_hash")
        if type(token_hash) is not bytes or len(token_hash) != 32:
            raise CommandSerializationError("invalid_lease_command")
        connection = self._require_owner_connection()
        cursor = await connection.execute(
            """
            SELECT call_id, tenant_id, agent_id, state, token_hash
            FROM call_leases WHERE call_control_id = ?
            """,
            (call_control_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row == (str(call_id), tenant_id, agent_id, "terminal", token_hash)

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
        if isinstance(operation.payload, RecordingUpsertPayloadV1):
            job = await self._read_recording_archive(operation.payload.recording_id)
            if job is not None and isinstance(job.operation.payload, RecordingUpsertPayloadV1):
                expected = job.operation.payload.model_copy(
                    update={"telnyx_recording_id": operation.payload.telnyx_recording_id}
                )
                if (
                    operation.call_id == job.operation.call_id
                    and operation.deployment_id == job.operation.deployment_id
                    and expected == operation.payload
                ):
                    await self._bind_archive_provider(
                        job, operation.payload.telnyx_recording_id, self._utcnow()
                    )

    async def _apply_lease(self, payload: Mapping[str, object]) -> None:
        if payload.get("action") == "stale_terminal":
            await self._apply_stale_terminal(payload)
            return
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
                   created_at, expires_at, lifecycle_json
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
                lifecycle=None if row[8] is None else self._decode_lifecycle(row[8]),
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
            if row[4] != self._contract_version:
                raise FatalPersistenceError("writer_contract_mismatch")
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
                    operation=self._decode_fixed_operation(plaintext),
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
        elif action == "runtime_observation":
            result = payload.get("result")
            if not isinstance(result, asyncio.Future):
                raise CommandSerializationError("invalid_runtime_observation_command")
            monotonic_now = self._monotonic()
            utc_now = self._utcnow()
            if (
                isinstance(monotonic_now, bool)
                or not isinstance(monotonic_now, int | float)
                or not math.isfinite(float(monotonic_now))
                or not isinstance(utc_now, datetime)
                or utc_now.tzinfo is None
                or utc_now.utcoffset() is None
            ):
                raise FatalPersistenceError("runtime_observation_clock_invalid")
            waiting = tuple(
                command for command in self._pending_commands if command.kind != "shutdown"
            )
            oldest_queue_age = (
                0.0 if not waiting else max(0.0, float(monotonic_now) - waiting[0].enqueued_at)
            )
            cursor = await connection.execute(
                """
                SELECT COUNT(*), MIN(created_at),
                       COALESCE(SUM(length(nonce) + length(ciphertext)), 0)
                FROM outbox
                """
            )
            row = await cursor.fetchone()
            await cursor.close()
            if (
                row is None
                or len(row) != 3
                or type(row[0]) is not int
                or row[0] < 0
                or type(row[2]) is not int
                or row[2] < 0
                or row[1] is not None
                and not isinstance(row[1], str)
            ):
                raise FatalPersistenceError("runtime_observation_invalid")
            oldest_outbox_age = (
                0.0
                if row[1] is None
                else max(
                    0.0,
                    (utc_now.astimezone(UTC) - _parse_datetime(row[1])).total_seconds(),
                )
            )
            observation = WriterRuntimeObservation(
                writer_queue_depth=len(waiting),
                writer_queue_oldest_age=oldest_queue_age,
                writer_quick_check=self._last_quick_check,
                outbox_depth=row[0],
                outbox_oldest_age=oldest_outbox_age,
                outbox_bytes=row[2],
                storage_bytes=self._measure_storage_bytes(),
            )
            if not result.done():
                result.set_result(observation)
        elif action == "ack":
            queue_id = self._required_positive_int(payload, "queue_id")
            expected_claim = self._required_positive_int(payload, "expected_claim_attempt")
            result = payload.get("result")
            if not isinstance(result, asyncio.Future):
                raise CommandSerializationError("invalid_relay_ack")
            cursor = await connection.execute(
                "SELECT op_id FROM outbox WHERE queue_id=? AND attempts=?",
                (queue_id, expected_claim),
            )
            delivered = await cursor.fetchone()
            await cursor.close()
            cursor = await connection.execute(
                "DELETE FROM outbox WHERE queue_id = ? AND attempts = ?",
                (queue_id, expected_claim),
            )
            applied = cursor.rowcount == 1
            await cursor.close()
            if applied and delivered is not None:
                if self._contract_version == 2:
                    await connection.execute(
                        "UPDATE local_audio_terminal SET acked=1 WHERE op_id=?", (delivered[0],)
                    )
                # This exact claim is ACKed only after the relay's native sink
                # ingest returned a known success. Local archive COMMIT alone
                # never reaches this path, and ambiguous native COMMIT does not ACK.
                await connection.execute(
                    "UPDATE recording_archives SET state='acknowledged' "
                    "WHERE receipt_op_id=? AND state='archived' "
                    "AND julianday(retention_until)>julianday(?) "
                    "AND NOT EXISTS(SELECT 1 FROM sparra_content_fences f "
                    "WHERE f.call_id=recording_archives.call_id)",
                    (delivered[0], _iso(self._utcnow())),
                )
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
                "DELETE FROM call_leases WHERE state = 'terminal' AND closed_at < ? "
                "AND lifecycle_json IS NULL",
                (_iso(now - CLOSED_LEASE_RETENTION),),
            )
            sparra_collected = await self._collect_terminal_sparra_calls(now)
            cleanup = CleanupResult(
                receipts_cursor.rowcount, leases_cursor.rowcount + sparra_collected
            )
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

    def _resolve_success(self, command: PersistenceCommand, value: object | None) -> None:
        result = command.payload.get("result")
        if (
            isinstance(result, asyncio.Future)
            and not result.done()
            and (
                value is not None
                or command.kind in {"call_lifecycle_read", "sparra_content", "recording_archive"}
            )
        ):
            result.set_result(value)
        if command.committed is not None and not command.committed.done():
            command.committed.set_result(None)

    def _resolve_failure(self, command: PersistenceCommand, error: FatalPersistenceError) -> None:
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
