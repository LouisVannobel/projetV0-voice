"""One bounded Telnyx WAV transfer into the process-owned encrypted audio archive."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import stat
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr

from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.models import RecordingArchiveReceiptV1, RecordingUpsertPayloadV1
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.writer import PersistenceWriter, RecordingArchiveJob
from projetv0_voice.telnyx.recordings import (
    ProviderRecordingV1,
    RecordingCatalogApi,
    reconcile_archive_provider_recording,
)

MAX_RECORDING_BYTES = 33_554_432
TRANSFER_SECONDS = 120


class RecordingArchiveError(RuntimeError):
    """A recording archive boundary carrying only a constant code."""


@dataclass(frozen=True, slots=True, repr=False)
class ProviderRecordingDownloadV1:
    recording: ProviderRecordingV1 = field(repr=False)
    wav_url: SecretStr = field(repr=False)
    retrieved_at: datetime = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.recording, ProviderRecordingV1) or not isinstance(
            self.wav_url, SecretStr
        ):
            raise ValueError("recording_download_invalid")
        if (
            not isinstance(self.retrieved_at, datetime)
            or self.retrieved_at.tzinfo is None
            or self.retrieved_at.utcoffset() is None
        ):
            raise ValueError("recording_download_invalid")


class RecordingDownloadApi(RecordingCatalogApi, Protocol):
    async def retrieve_recording_download(
        self, recording_id: str, *, timeout_seconds: float
    ) -> ProviderRecordingDownloadV1: ...


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    outcome: Literal["archived", "acknowledged", "unavailable", "expired", "erased", "unknown"]
    native_acknowledged: bool = False


def _origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "\\" in value
            or any(ord(character) < 33 or ord(character) == 127 for character in value)
            or parsed.port not in {None, 443}
        ):
            raise ValueError
        return "https://" + parsed.hostname.lower()
    except ValueError:
        raise RecordingArchiveError("recording_archive_origin_invalid") from None


def _valid_dual_wav(value: bytes) -> bool:
    if (
        len(value) < 48
        or value[:4] != b"RIFF"
        or value[8:12] != b"WAVE"
        or struct.unpack_from("<I", value, 4)[0] + 8 != len(value)
    ):
        return False
    offset, block_align, data_size = 12, None, None
    while offset + 8 <= len(value):
        chunk, size = value[offset : offset + 4], struct.unpack_from("<I", value, offset + 4)[0]
        start, end = offset + 8, offset + 8 + size
        if end > len(value):
            return False
        if chunk == b"fmt ":
            if block_align is not None or size < 16:
                return False
            encoding, channels, rate, byte_rate, align, bits = struct.unpack_from(
                "<HHIIHH", value, start
            )
            if (
                encoding != 1
                or channels != 2
                or not 1 <= rate <= 192000
                or bits not in {8, 16, 24, 32}
                or align != channels * bits // 8
                or byte_rate != rate * align
            ):
                return False
            block_align = align
        elif chunk == b"data":
            if data_size is not None:
                return False
            data_size = size
        offset = end + (size % 2)
    return (
        offset == len(value)
        and block_align is not None
        and data_size is not None
        and data_size > 0
        and data_size % block_align == 0
    )


class RecordingArchive:
    """Own a private directory and one download; local durability is distinct from native ACK."""

    def __init__(
        self,
        *,
        directory: Path,
        keyring: CryptoKeyring,
        writer: PersistenceWriter,
        telnyx: RecordingDownloadApi,
        allowed_origins: tuple[str, ...],
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        download_transport: httpx.AsyncBaseTransport | None = None,
        directory_sync: Callable[[], None] | None = None,
        directory_descriptor: int | None = None,
    ) -> None:
        self._directory = Path(directory)
        info = self._directory.lstat()
        geteuid = getattr(os, "geteuid", None)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or os.name == "posix"
            and (geteuid is None or info.st_uid != geteuid() or stat.S_IMODE(info.st_mode) != 0o700)
        ):
            raise RecordingArchiveError("recording_archive_directory_invalid")
        self._directory_identity = (info.st_dev, info.st_ino)
        self._directory_fd: int | None = None
        if os.name == "posix":
            directory_flag = getattr(os, "O_DIRECTORY", None)
            nofollow_flag = getattr(os, "O_NOFOLLOW", None)
            if type(directory_flag) is not int or type(nofollow_flag) is not int:
                raise RecordingArchiveError("recording_archive_platform_unsupported")
            self._directory_fd = (
                os.dup(directory_descriptor)
                if directory_descriptor is not None
                else os.open(self._directory, os.O_RDONLY | directory_flag | nofollow_flag)
            )
            opened = os.fstat(self._directory_fd)
            if (opened.st_dev, opened.st_ino) != self._directory_identity:
                os.close(self._directory_fd)
                raise RecordingArchiveError("recording_archive_directory_invalid")
        elif directory_sync is None:
            raise RecordingArchiveError("recording_archive_platform_unsupported")
        if not allowed_origins or any(value != _origin(value) for value in allowed_origins):
            if self._directory_fd is not None:
                os.close(self._directory_fd)
            raise RecordingArchiveError("recording_archive_origin_invalid")
        self._origins = frozenset(allowed_origins)
        self._directory_sync = directory_sync
        self._keyring, self._writer, self._telnyx = keyring, writer, telnyx
        self._utcnow, self._monotonic = utcnow, monotonic
        self._lock = asyncio.Lock()
        self._active_task: asyncio.Task[ArchiveResult] | None = None
        self._active_recording_id: UUID | None = None
        self._active_call_id: UUID | None = None
        self._publications: dict[UUID, asyncio.Future[object]] = {}
        self._erased_calls: set[UUID] = set()
        self._closed = False
        self._failed = False
        self._http = httpx.AsyncClient(
            transport=download_transport,
            trust_env=False,
            follow_redirects=False,
            timeout=TRANSFER_SECONDS,
        )
        writer.bind_recording_audio_cleanup(self._erase_owned_files, free_bytes=self._free_bytes)

    def _free_bytes(self) -> int:
        if self._closed or self._failed:
            raise RecordingArchiveError("recording_archive_consumer_closed")
        self._check_directory()
        if self._directory_fd is not None:
            statvfs = getattr(os, "fstatvfs", None)
            if not callable(statvfs):
                raise RecordingArchiveError("recording_archive_platform_unsupported")
            usage = statvfs(self._directory_fd)
            return int(usage.f_bavail * usage.f_frsize)
        return shutil.disk_usage(self._directory).free

    def _now(self) -> datetime:
        now = self._utcnow()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise RecordingArchiveError("recording_archive_clock_invalid")
        return now.astimezone(UTC)

    def _check_directory(self) -> None:
        current = self._directory.lstat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != self._directory_identity
            or os.name == "posix"
            and (
                stat.S_IMODE(current.st_mode) != 0o700
                or current.st_uid != getattr(os, "geteuid", lambda: -1)()
            )
        ):
            raise RecordingArchiveError("recording_archive_directory_changed")

    def _sync_directory(self) -> None:
        self._check_directory()
        if self._directory_sync is not None:
            self._directory_sync()
        elif self._directory_fd is not None:
            os.fsync(self._directory_fd)

    def _unlink(self, recording_id: UUID) -> None:
        self._check_directory()
        try:
            if self._directory_fd is not None:
                os.unlink(str(recording_id), dir_fd=self._directory_fd)
            else:
                (self._directory / str(recording_id)).unlink()
        except FileNotFoundError:
            pass

    async def _erase_owned_files(self, call_id: UUID, recording_ids: tuple[UUID, ...]) -> None:
        self._erased_calls.add(call_id)
        active = self._active_task
        if active is not None and self._active_recording_id in recording_ids:
            active.cancel()
            await self._join_cancelled_archive(active)
        async with self._lock:
            for recording_id in recording_ids:
                future = self._publications.get(recording_id)
                if future is not None:
                    await self._settle_publication(future)
                self._unlink(recording_id)
            await self._recover_owned_files()
            self._sync_directory()

    async def _join_cancelled_archive(self, active: asyncio.Task[ArchiveResult]) -> None:
        while not active.done():
            try:
                await asyncio.shield(active)
            except asyncio.CancelledError:
                continue
        with contextlib.suppress(asyncio.CancelledError):
            active.result()

    async def _recover_owned_files(self) -> None:
        self._check_directory()
        # This directory is exclusively owned by this audio consumer. Refuse an
        # unbounded or foreign inventory rather than extending deletion scope.
        entries: list[UUID] = []
        with os.scandir(
            self._directory_fd if self._directory_fd is not None else self._directory
        ) as inventory:
            for entry in inventory:
                if len(entries) >= 256:
                    raise RecordingArchiveError("recording_archive_inventory_bound")
                try:
                    recording_id = UUID(entry.name)
                except ValueError:
                    raise RecordingArchiveError("recording_archive_inventory_invalid") from None
                if str(recording_id) != entry.name or not entry.is_file(follow_symlinks=False):
                    raise RecordingArchiveError("recording_archive_inventory_invalid")
                entries.append(recording_id)
        removed = False
        for recording_id in entries:
            future = self._publications.get(recording_id)
            if future is not None and not future.done():
                continue
            job = await self._writer.read_recording_archive(recording_id)
            if job is None or job.receipt is None:
                self._unlink(recording_id)
                removed = True
        if removed:
            self._sync_directory()

    def _verify_committed_file(self, recording_id: UUID, job: RecordingArchiveJob) -> None:
        if job.receipt is None or job.nonce is None:
            raise RecordingArchiveError("recording_archive_receipt_missing")
        self._check_directory()
        path = self._directory / str(recording_id)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise RecordingArchiveError("recording_archive_file_invalid")
        descriptor = os.open(
            str(recording_id) if self._directory_fd is not None else path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self._directory_fd,
        )
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_size != job.receipt.encrypted_bytes
                or not 17 <= opened.st_size <= MAX_RECORDING_BYTES + 16
                or os.name == "posix"
                and (
                    stat.S_IMODE(opened.st_mode) != 0o600
                    or opened.st_uid != getattr(os, "geteuid", lambda: -1)()
                )
            ):
                raise RecordingArchiveError("recording_archive_file_invalid")
            ciphertext = stream.read(MAX_RECORDING_BYTES + 17)
            after = os.fstat(stream.fileno())
            if (
                len(ciphertext) != opened.st_size
                or after.st_size != opened.st_size
                or after.st_mtime_ns != opened.st_mtime_ns
                or hashlib.sha256(ciphertext).hexdigest() != job.receipt.ciphertext_sha256
            ):
                raise RecordingArchiveError("recording_archive_file_invalid")
        plaintext = self._keyring.decrypt(
            EncryptedValue(job.receipt.key_version, job.nonce, ciphertext),
            aad=f"recording:{job.operation.call_id}:{recording_id}".encode("ascii"),
        )
        if not _valid_dual_wav(plaintext):
            raise RecordingArchiveError("recording_archive_wav_invalid")

    def _check_budget(self, job: RecordingArchiveJob, deadline: float) -> None:
        if self._monotonic() >= deadline or self._now() >= job.observed_at + timedelta(
            seconds=TRANSFER_SECONDS
        ):
            raise RecordingArchiveError("recording_archive_budget_exhausted")

    async def _settle_publication(
        self, future: asyncio.Future[object]
    ) -> tuple[object | BaseException, asyncio.CancelledError | None]:
        cancellation = None
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError as error:
                cancellation = error
            except BaseException:
                break
        try:
            return future.result(), cancellation
        except BaseException as error:
            return error, cancellation

    def _store(
        self, recording_id: UUID, ciphertext: bytes, job: RecordingArchiveJob, deadline: float
    ) -> None:
        self._check_directory()
        temporary = uuid4()
        path = self._directory / str(temporary)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            str(temporary) if self._directory_fd is not None else path,
            flags,
            0o600,
            dir_fd=self._directory_fd,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(ciphertext)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_budget(job, deadline)
            if self._directory_fd is not None:
                os.replace(
                    str(temporary),
                    str(recording_id),
                    src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd,
                )
            else:
                os.replace(path, self._directory / str(recording_id))
            self._sync_directory()
            self._check_budget(job, deadline)
        finally:
            self._unlink(temporary)

    def _matches(self, download: ProviderRecordingDownloadV1, job: RecordingArchiveJob) -> bool:
        record, correlation = download.recording, job.correlation
        payload = job.operation.payload
        if not isinstance(payload, RecordingUpsertPayloadV1):
            return False
        return (
            record.recording_id == payload.telnyx_recording_id
            and record.call_control_id == correlation.call_control_id
            and record.call_leg_id == correlation.call_leg_id
            and record.call_session_id == correlation.call_session_id
            and record.channels == "dual"
            and record.status == "completed"
            and record.source == "call"
            and record.initiated_by == "StartCallRecordingAPI"
            and record.recording_started_at == payload.started_at
            and record.recording_ended_at == payload.ended_at
        )

    async def archive_recording_once(self, recording_id: UUID) -> ArchiveResult:
        if not isinstance(recording_id, UUID) or self._closed or self._failed:
            raise RecordingArchiveError("recording_archive_input_invalid")
        cancellation = None
        call_id = None
        async with self._lock:
            if self._closed:
                raise RecordingArchiveError("recording_archive_input_invalid")
            await self._recover_owned_files()
            owner = asyncio.create_task(
                self._archive_owned(recording_id), name="voice-recording-archive"
            )
            self._active_task, self._active_recording_id = owner, recording_id
            try:
                while not owner.done():
                    try:
                        await asyncio.shield(owner)
                    except asyncio.CancelledError as error:
                        cancellation = error
                        owner.cancel()
                result = owner.result()
                call_id = self._active_call_id
            finally:
                self._active_task = None
                self._active_recording_id = None
                self._active_call_id = None
        if result.outcome in {"expired", "erased"} and call_id is not None:
            await self._writer.erase_call_content(call_id, now=self._now())
        if cancellation is not None:
            raise cancellation
        return result

    async def _archive_owned(self, recording_id: UUID) -> ArchiveResult:
        job = await self._writer.read_recording_archive(recording_id)
        if job is None:
            return ArchiveResult("unknown")
        self._active_call_id = job.operation.call_id
        payload = job.operation.payload
        if not isinstance(payload, RecordingUpsertPayloadV1):
            raise RecordingArchiveError("recording_archive_operation_invalid")
        now = self._now()
        if job.fenced or job.operation.call_id in self._erased_calls:
            return ArchiveResult("erased")
        if job.retention_until <= now:
            return ArchiveResult("expired")
        if not job.policy_bound or not job.recording_enabled or not job.input_gate_opened:
            await self._writer.fail_recording_archive(recording_id, expired=False, now=now)
            return ArchiveResult("unavailable")
        if job.state in {"unavailable", "expired", "erased"}:
            return ArchiveResult(job.state)
        if job.state in {"archived", "acknowledged"}:
            try:
                self._verify_committed_file(recording_id, job)
            except Exception:
                if job.retention_until <= self._now():
                    return ArchiveResult("expired")
                await self._writer.fail_recording_archive(
                    recording_id, expired=False, now=self._now()
                )
                return ArchiveResult("unavailable")
            if job.retention_until <= self._now():
                return ArchiveResult("expired")
            return ArchiveResult(job.state, job.state == "acknowledged")
        if job.reserved_bytes != MAX_RECORDING_BYTES + 16:
            await self._writer.fail_recording_archive(recording_id, expired=False, now=now)
            return ArchiveResult("unavailable")
        budget = min(
            float(TRANSFER_SECONDS), TRANSFER_SECONDS - (now - job.observed_at).total_seconds()
        )
        if budget <= 0:
            await self._writer.fail_recording_archive(recording_id, expired=False, now=now)
            return ArchiveResult("unavailable")
        deadline = self._monotonic() + budget
        future = None
        try:
            async with asyncio.timeout(budget):
                if payload.telnyx_recording_id is None:
                    self._check_budget(job, deadline)
                    provider_id = await reconcile_archive_provider_recording(
                        job.operation,
                        correlation=job.correlation,
                        telnyx=self._telnyx,
                        monotonic=self._monotonic,
                        deadline=deadline,
                    )
                    self._check_budget(job, deadline)
                    bound = await self._writer.bind_recording_archive_provider(
                        recording_id, provider_id, now=self._now()
                    )
                    if bound is None:
                        return ArchiveResult("unavailable")
                    job = bound
                    self._check_budget(job, deadline)
                    if job.retention_until <= self._now():
                        return ArchiveResult("expired")
                    payload = job.operation.payload
                    if not isinstance(payload, RecordingUpsertPayloadV1):
                        raise RecordingArchiveError("recording_archive_operation_invalid")
                if payload.telnyx_recording_id is None:
                    raise RecordingArchiveError("recording_archive_provider_identity_invalid")
                retrieve_started = self._now()
                download = await self._telnyx.retrieve_recording_download(
                    payload.telnyx_recording_id,
                    timeout_seconds=min(1.0, deadline - self._monotonic()),
                )
                if not isinstance(download, ProviderRecordingDownloadV1) or not self._matches(
                    download, job
                ):
                    raise RecordingArchiveError("recording_archive_provider_identity_invalid")
                if not retrieve_started <= download.retrieved_at <= self._now():
                    raise RecordingArchiveError("recording_archive_retrieve_time_invalid")
                url = download.wav_url.get_secret_value()
                if _origin(url) not in self._origins:
                    raise RecordingArchiveError("recording_archive_origin_invalid")
                self._check_budget(job, deadline)
                buffer = bytearray()
                async with self._http.stream(
                    "GET",
                    url,
                    headers={"Accept-Encoding": "identity"},
                    timeout=max(0.001, deadline - self._monotonic()),
                ) as response:
                    if (
                        response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                    ):
                        raise RecordingArchiveError("recording_archive_download_invalid")
                    length = response.headers.get("content-length")
                    if length is not None and (
                        not length.isascii()
                        or not length.isdecimal()
                        or int(length) > MAX_RECORDING_BYTES
                    ):
                        raise RecordingArchiveError("recording_archive_download_oversize")
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        self._check_budget(job, deadline)
                        if len(buffer) + len(chunk) > MAX_RECORDING_BYTES:
                            raise RecordingArchiveError("recording_archive_download_oversize")
                        buffer.extend(chunk)
                self._check_budget(job, deadline)
                plaintext = bytes(buffer)
                buffer.clear()
                if not _valid_dual_wav(plaintext):
                    raise RecordingArchiveError("recording_archive_wav_invalid")
                self._check_budget(job, deadline)
                encrypted = self._keyring.encrypt(
                    plaintext,
                    aad=f"recording:{job.operation.call_id}:{recording_id}".encode("ascii"),
                )
                del plaintext
                self._check_budget(job, deadline)
                if job.operation.call_id in self._erased_calls:
                    return ArchiveResult("erased")
                if self._now() >= job.retention_until:
                    return ArchiveResult("expired")
                self._store(recording_id, encrypted.ciphertext, job, deadline)
                self._check_budget(job, deadline)
                receipt = RecordingArchiveReceiptV1(
                    recording_id=recording_id,
                    ciphertext_sha256=hashlib.sha256(encrypted.ciphertext).hexdigest(),
                    encrypted_bytes=len(encrypted.ciphertext),
                    key_version=encrypted.key_version,
                    retention_until=job.retention_until,
                )
                self._check_budget(job, deadline)
                future = self._writer.submit_recording_archive_finish(
                    recording_id,
                    receipt,
                    encrypted.nonce,
                    now=self._now(),
                    publication_deadline=deadline,
                )
                self._publications[recording_id] = future
            # The queued writer owns this outcome. A cancelled caller or expired
            # transfer budget cannot turn a committed or uncertain receipt into rollback.
            outcome, cancellation = await self._settle_publication(future)
            if job.retention_until <= self._now():
                # COMMIT may already be durable or unknown. Retain its bytes and
                # identity until the existing cleanup joins outside the archive lock.
                result = ArchiveResult("expired")
            elif isinstance(outcome, BaseException):
                definitely_uncommitted = (
                    isinstance(outcome, PersistenceError)
                    and str(outcome) == "recording_archive_rolled_back"
                )
                if definitely_uncommitted:
                    self._unlink(recording_id)
                    self._sync_directory()
                result = ArchiveResult("unavailable" if definitely_uncommitted else "unknown")
            elif outcome is None:
                self._unlink(recording_id)
                self._sync_directory()
                if not self._writer.fatal_event.is_set():
                    await self._writer.fail_recording_archive(
                        recording_id, expired=False, now=self._now()
                    )
                result = ArchiveResult(
                    "erased" if job.operation.call_id in self._erased_calls else "unavailable"
                )
            elif isinstance(outcome, RecordingArchiveJob):
                try:
                    self._check_budget(job, deadline)
                except RecordingArchiveError:
                    await self._writer.fail_recording_archive(
                        recording_id, expired=False, now=self._now()
                    )
                    result = ArchiveResult("unavailable")
                else:
                    result = ArchiveResult("archived")
            else:
                result = ArchiveResult("unknown")
            if cancellation is not None and result.outcome != "expired":
                raise cancellation
            return result
        except asyncio.CancelledError:
            if future is None:
                self._unlink(recording_id)
                self._sync_directory()
            raise
        except Exception:
            if future is not None:
                return ArchiveResult("unknown")
            self._unlink(recording_id)
            self._sync_directory()
            if not self._writer.fatal_event.is_set():
                await self._writer.fail_recording_archive(
                    recording_id, expired=False, now=self._now()
                )
            return ArchiveResult("unavailable")
        finally:
            if future is not None and future.done():
                self._publications.pop(recording_id, None)

    async def erase_call_audio(self, call_id: UUID) -> None:
        await self._writer.erase_call_content(call_id, now=self._now())

    async def prepare(self) -> None:
        try:
            for call_id in await self._writer.recording_audio_cleanup_calls(now=self._now()):
                await self.erase_call_audio(call_id)
            async with self._lock:
                await self._recover_owned_files()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._failed = True
            raise RecordingArchiveError("recording_archive_prepare_failed") from None

    async def run_once(self) -> None:
        await self.prepare()
        if self._closed or self._failed:
            return
        for recording_id in await self._writer.pending_recording_archives():
            await self.archive_recording_once(recording_id)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        active = self._active_task
        if active is not None:
            active.cancel()
            await self._join_cancelled_archive(active)
        async with self._lock:
            await self._http.aclose()
            if self._directory_fd is not None:
                os.close(self._directory_fd)
            self._writer.unbind_recording_audio_cleanup(self._erase_owned_files)
