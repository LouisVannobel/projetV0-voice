from __future__ import annotations

import asyncio
import os
import sqlite3
import struct
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import aiosqlite
import httpx
import pytest
from test_recording_archive import (
    CALL_ID,
    DEADLINE,
    KEY,
    NOW,
    Provider,
    archive_module,
    queue_saved,
    saved_event,
    start_writer,
    stop_writer,
    wav,
)

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.commands import PersistenceError, canonical_operation_bytes
from projetv0_voice.persistence.writer import PersistenceWriter


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def utcnow(self):
        return NOW + timedelta(minutes=2, seconds=self.seconds)

    def monotonic(self):
        return self.seconds


@asynccontextmanager
async def owned(
    tmp_path,
    *,
    respond=None,
    sync=None,
    nonce_factory=None,
    initial_seconds=0.0,
    historic_pin=True,
):
    directory = tmp_path / "audio"
    directory.mkdir(mode=0o700)
    clock = Clock()
    clock.seconds = initial_seconds
    keyring = CryptoKeyring(
        {1: KEY},
        active_version=1,
        **({} if nonce_factory is None else {"nonce_factory": nonce_factory}),
    )
    database = tmp_path / "voice.sqlite"
    writer = PersistenceWriter(database, keyring, utcnow=clock.utcnow, monotonic=clock.monotonic)
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(200, content=wav()) if respond is None else respond(request)

    box = SimpleNamespace(
        directory=directory,
        clock=clock,
        keyring=keyring,
        database=database,
        writer=writer,
        task=task,
        requests=requests,
    )

    def consumer():
        return archive_module().RecordingArchive(
            directory=directory,
            keyring=keyring,
            writer=box.writer,
            telnyx=Provider(utcnow=clock.utcnow),
            allowed_origins=("https://recordings.example.invalid",),
            download_transport=httpx.MockTransport(response),
            directory_sync=sync or (lambda: None),
            utcnow=clock.utcnow,
            monotonic=clock.monotonic,
        )

    box.new_consumer = consumer
    box.consumer = consumer()
    try:
        box.recording_id = await queue_saved(
            writer,
            saved_event(),
            observed_at=clock.utcnow(),
            historic_pin=historic_pin,
        )
        yield box
    finally:
        await box.consumer.aclose()
        if box.writer.fatal_event.is_set():
            await asyncio.wait_for(box.task, timeout=2)
        else:
            await stop_writer(box.writer, box.task)


async def restart(box, *, consumer=True):
    await box.consumer.aclose()
    if box.writer.fatal_event.is_set():
        await asyncio.wait_for(box.task, timeout=2)
    else:
        await stop_writer(box.writer, box.task)
    box.writer = PersistenceWriter(
        box.database, box.keyring, utcnow=box.clock.utcnow, monotonic=box.clock.monotonic
    )
    box.task = asyncio.create_task(box.writer.run())
    assert await box.writer.wait_ready()
    if consumer:
        box.consumer = box.new_consumer()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "truncated", "corrupt"])
async def test_committed_ciphertext_damage_after_restart_refuses_availability_without_rewriting(
    tmp_path,
    damage,
):
    async with owned(tmp_path) as box:
        assert (await box.consumer.archive_recording_once(box.recording_id)).outcome == "archived"
        before = await box.writer.read_recording_archive(box.recording_id)
        old_cipher = (box.directory / str(box.recording_id)).read_bytes()
        await restart(box)
        path = box.directory / str(box.recording_id)
        if damage == "missing":
            path.unlink()
        else:
            path.write_bytes(old_cipher[:-1] if damage == "truncated" else b"x" * len(old_cipher))
        result = await box.consumer.archive_recording_once(box.recording_id)
        assert result.outcome == "unavailable" and result.native_acknowledged is False
        after = await box.writer.read_recording_archive(box.recording_id)
        assert after.receipt == before.receipt and after.nonce == before.nonce
        assert len(box.requests) == 1
        assert not path.exists() if damage == "missing" else path.read_bytes() != old_cipher


@pytest.mark.asyncio
async def test_healthy_committed_restart_reuses_exact_cipher_nonce_receipt_and_operation(tmp_path):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        before = await box.writer.read_recording_archive(box.recording_id)
        ciphertext = (box.directory / str(box.recording_id)).read_bytes()
        operations = await box.writer.read_relay_batch(
            batch_size=10, now=box.clock.utcnow(), lease_seconds=30
        )
        raw = [canonical_operation_bytes(item.operation) for item in operations]
        await restart(box)
        assert (await box.consumer.archive_recording_once(box.recording_id)).outcome == "archived"
        after = await box.writer.read_recording_archive(box.recording_id)
        assert (box.directory / str(box.recording_id)).read_bytes() == ciphertext
        assert (after.receipt, after.nonce) == (before.receipt, before.nonce)
        box.clock.seconds = 31
        replay = await box.writer.read_relay_batch(
            batch_size=10, now=box.clock.utcnow(), lease_seconds=30
        )
        assert [canonical_operation_bytes(item.operation) for item in replay] == raw
        assert len(box.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["rollback", "committed_ack_lost"])
async def test_actual_sqlite_commit_cut_distinguishes_rollback_from_unknown_and_reconciles_restart(
    tmp_path,
    monkeypatch,
    cut,
):
    async with owned(tmp_path) as box:
        commit = aiosqlite.Connection.commit
        observed = []

        async def interrupted(connection):
            if (box.directory / str(box.recording_id)).exists() and not observed:
                observed.append(cut)
                if cut == "committed_ack_lost":
                    await commit(connection)
                raise sqlite3.OperationalError("owned SQLite acknowledgement cut")
            await commit(connection)

        monkeypatch.setattr(aiosqlite.Connection, "commit", interrupted)
        try:
            result = await box.consumer.archive_recording_once(box.recording_id)
        except PersistenceError as error:
            result = error
        assert observed == [cut]
        assert getattr(result, "outcome", None) == (
            "unknown" if cut == "committed_ack_lost" else "unavailable"
        )
        path = box.directory / str(box.recording_id)
        assert path.exists() is (cut == "committed_ack_lost")
        original = path.read_bytes() if path.exists() else None
        await restart(box)
        restored = await box.writer.read_recording_archive(box.recording_id)
        if cut == "committed_ack_lost":
            assert restored.receipt is not None
            assert (
                await box.consumer.archive_recording_once(box.recording_id)
            ).outcome == "archived"
            assert path.read_bytes() == original
            assert len(box.requests) == 1
        else:
            assert restored.receipt is None


@pytest.mark.asyncio
async def test_cancellation_during_real_receipt_commit_joins_and_preserves_committed_bytes(
    tmp_path,
):
    async with owned(tmp_path) as box:
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(name):
            if (
                name == "after_mutation_before_commit"
                and (box.directory / str(box.recording_id)).exists()
            ):
                entered.set()
                await release.wait()

        box.writer._failpoint = hold
        operation = asyncio.create_task(box.consumer.archive_recording_once(box.recording_id))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            operation.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not operation.done(), (
                "publication waiter escaped before its owned COMMIT settled"
            )
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            job = await box.writer.read_recording_archive(box.recording_id)
            assert job.receipt is not None and (box.directory / str(box.recording_id)).exists()
        finally:
            release.set()
            if not operation.done():
                operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_erasure_unlinks_and_clears_private_archive_metadata_before_ack(tmp_path):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        cleaned = await box.writer.erase_call_content(
            CALL_ID, now=box.clock.utcnow(), lease_token=UUID(int=44)
        )
        assert cleaned is not None and not list(box.directory.iterdir())
        assert await box.writer.read_recording_archive(box.recording_id) is None
        assert await box.writer.pending_erasure_acks() == ((CALL_ID, UUID(int=44), cleaned),)
        with sqlite3.connect(box.database) as connection:
            assert connection.execute("SELECT count(*) FROM recording_archives").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM sparra_content_fences").fetchone() == (
                1,
            )
        replay = await box.writer.read_relay_batch(
            batch_size=10, now=box.clock.utcnow(), lease_seconds=30
        )
        assert any(item.operation.kind == "recording.upsert" for item in replay)


@pytest.mark.asyncio
async def test_restart_without_archive_consumer_fences_but_withholds_cleanup_ack(tmp_path):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        await restart(box, consumer=False)
        with pytest.raises(PersistenceError, match="recording_archive_cleanup_unavailable"):
            await box.writer.erase_call_content(
                CALL_ID, now=box.clock.utcnow(), lease_token=UUID(int=45)
            )
        assert await box.writer.pending_erasure_acks() == ()
        assert (box.directory / str(box.recording_id)).exists()
        with sqlite3.connect(box.database) as connection:
            assert connection.execute("SELECT count(*) FROM sparra_content_fences").fetchone() == (
                1,
            )
        box.consumer = box.new_consumer()
        await box.consumer.run_once()
        assert not list(box.directory.iterdir())
        assert await box.writer.read_recording_archive(box.recording_id) is None


class WaitingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.entered.set()
        await self.release.wait()
        yield wav()

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_erasure_during_download_cancels_joins_and_prevents_late_publication(tmp_path):
    stream = WaitingStream()
    async with owned(tmp_path, respond=lambda request: httpx.Response(200, stream=stream)) as box:
        transfer = asyncio.create_task(box.consumer.archive_recording_once(box.recording_id))
        try:
            await asyncio.wait_for(stream.entered.wait(), 2)
            await asyncio.wait_for(
                box.writer.erase_call_content(
                    CALL_ID, now=box.clock.utcnow(), lease_token=UUID(int=46)
                ),
                2,
            )
            assert transfer.done() and stream.closed
            assert await box.writer.read_recording_archive(box.recording_id) is None
            assert not list(box.directory.iterdir())
        finally:
            stream.release.set()
            if not transfer.done():
                transfer.cancel()
            await asyncio.gather(transfer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["archived", "acknowledged"])
async def test_exact_expiry_sweeps_committed_audio_and_private_metadata(state, tmp_path):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        await box.consumer.aclose()
        await stop_writer(box.writer, box.task)
        with sqlite3.connect(box.database) as connection:
            connection.execute("UPDATE recording_archives SET state=?", (state,))
        box.clock.seconds = (DEADLINE - box.clock.utcnow()).total_seconds()
        box.writer, box.task = await start_writer(box.database, box.keyring)
        box.consumer = box.new_consumer()
        await box.consumer.run_once()
        assert not list(box.directory.iterdir())
        assert await box.writer.read_recording_archive(box.recording_id) is None


@pytest.mark.asyncio
async def test_sync_directory_budget_exhaustion_never_commits_receipt(tmp_path):
    box_ref = []

    def sync():
        box_ref[0].clock.seconds = 121

    async with owned(tmp_path, sync=sync) as box:
        box_ref.append(box)
        result = await box.consumer.archive_recording_once(box.recording_id)
        assert result.outcome == "unavailable"
        assert not list(box.directory.iterdir())
        assert (await box.writer.read_recording_archive(box.recording_id)).receipt is None


@pytest.mark.asyncio
async def test_restart_removes_only_owned_uncommitted_uuid_ciphertext(tmp_path):
    async with owned(tmp_path) as box:
        orphan = box.directory / str(UUID(int=999))
        orphan.write_bytes(b"owned uncommitted ciphertext")
        (box.directory / str(box.recording_id)).write_bytes(b"uncommitted recording ciphertext")
        await restart(box)
        await box.consumer.run_once()
        assert not orphan.exists()
        job = await box.writer.read_recording_archive(box.recording_id)
        assert job.receipt is not None
        assert [path.name for path in box.directory.iterdir()] == [str(box.recording_id)]


@pytest.mark.asyncio
async def test_erasure_queued_behind_receipt_commit_settles_without_owner_deadlock(tmp_path):
    async with owned(tmp_path) as box:
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(name):
            if (
                name == "after_mutation_before_commit"
                and (box.directory / str(box.recording_id)).exists()
            ):
                entered.set()
                await release.wait()

        box.writer._failpoint = hold
        transfer = asyncio.create_task(box.consumer.archive_recording_once(box.recording_id))
        erase = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            erase = asyncio.create_task(
                box.writer.erase_call_content(
                    CALL_ID, now=box.clock.utcnow(), lease_token=UUID(int=48)
                )
            )
            await asyncio.sleep(0)
            assert not erase.done()
            release.set()
            cleaned = await asyncio.wait_for(erase, 2)
            assert cleaned is not None and transfer.done()
            assert not list(box.directory.iterdir())
            assert await box.writer.read_recording_archive(box.recording_id) is None
            assert await box.writer.pending_erasure_acks() == ((CALL_ID, UUID(int=48), cleaned),)
        finally:
            release.set()
            await asyncio.gather(
                transfer, *([] if erase is None else [erase]), return_exceptions=True
            )


@pytest.mark.asyncio
async def test_late_saved_after_erase_restart_keeps_purge_identity_without_private_recreation(
    tmp_path,
):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        await box.writer.erase_call_content(CALL_ID, now=box.clock.utcnow())
        await restart(box)
        await queue_saved(box.writer, saved_event())
        assert await box.writer.read_recording_archive(box.recording_id) is None
        assert (await box.consumer.archive_recording_once(box.recording_id)).outcome == "unknown"
        assert not list(box.directory.iterdir()) and len(box.requests) == 1
        replay = await box.writer.read_relay_batch(
            batch_size=10, now=box.clock.utcnow(), lease_seconds=30
        )
        assert any(
            item.operation.kind == "recording.upsert"
            and item.operation.payload.telnyx_recording_id == "recording_Ab-12"
            for item in replay
        )


@pytest.mark.asyncio
async def test_caller_cancel_during_fenced_cleanup_joins_before_local_ack(tmp_path):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        entered, release = asyncio.Event(), asyncio.Event()
        cleanup = box.writer._audio_cleanup

        async def slow(call_id, ids):
            entered.set()
            await release.wait()
            await cleanup(call_id, ids)

        box.writer._audio_cleanup = slow
        erase = asyncio.create_task(
            box.writer.erase_call_content(CALL_ID, now=box.clock.utcnow(), lease_token=UUID(int=49))
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert await box.writer.pending_erasure_acks() == ()
            erase.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not erase.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await erase
            assert not list(box.directory.iterdir())
            assert await box.writer.read_recording_archive(box.recording_id) is None
            assert len(await box.writer.pending_erasure_acks()) == 1
        finally:
            release.set()
            await asyncio.gather(erase, return_exceptions=True)
            box.writer._audio_cleanup = cleanup


@pytest.mark.asyncio
async def test_first_observation_budget_is_not_renewed_by_duplicate_or_restart(tmp_path):
    async with owned(tmp_path) as box:
        first = await box.writer.read_recording_archive(box.recording_id)
        await restart(box)
        box.clock.seconds = 121
        await queue_saved(box.writer, saved_event())
        repeated = await box.writer.read_recording_archive(box.recording_id)
        assert repeated.observed_at == first.observed_at
        assert (
            await box.consumer.archive_recording_once(box.recording_id)
        ).outcome == "unavailable"
        assert box.requests == [] and not list(box.directory.iterdir())


@pytest.mark.asyncio
async def test_expiry_sweep_one_microsecond_before_deadline_keeps_committed_ciphertext(tmp_path):
    async with owned(tmp_path) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        receipt = (await box.writer.read_recording_archive(box.recording_id)).receipt
        ciphertext = (box.directory / str(box.recording_id)).read_bytes()
        box.clock.seconds = (DEADLINE - box.clock.utcnow()).total_seconds() - 0.000001
        assert box.clock.utcnow() == DEADLINE - timedelta(microseconds=1)
        await box.consumer.run_once()
        job = await box.writer.read_recording_archive(box.recording_id)
        assert job is not None and job.receipt == receipt
        assert (box.directory / str(box.recording_id)).read_bytes() == ciphertext


@pytest.mark.asyncio
@pytest.mark.parametrize("crossing", ["verify", "queued_finish", "actual_commit"])
async def test_retention_crossing_never_reports_late_availability_and_joins_fenced_cleanup(
    tmp_path,
    monkeypatch,
    crossing,
):
    initial = (DEADLINE - NOW - timedelta(minutes=2, milliseconds=100)).total_seconds()
    async with owned(tmp_path, initial_seconds=initial if crossing != "verify" else 0.0) as box:
        observations = []
        release, entered = asyncio.Event(), asyncio.Event()
        queued_read = None
        transfer = None
        if crossing == "verify":
            await box.consumer.archive_recording_once(box.recording_id)
            decrypt = CryptoKeyring.decrypt

            def cross_during_real_decrypt(keyring, encrypted, *, aad):
                plaintext = decrypt(keyring, encrypted, aad=aad)
                if aad.startswith(b"recording:"):
                    observations.append(crossing)
                    box.clock.seconds = initial + 0.1
                return plaintext

            monkeypatch.setattr(CryptoKeyring, "decrypt", cross_during_real_decrypt)
            box.clock.seconds = initial
        elif crossing == "actual_commit":
            commit = aiosqlite.Connection.commit

            async def cross_after_real_commit(connection):
                await commit(connection)
                if (box.directory / str(box.recording_id)).exists() and not observations:
                    observations.append(crossing)
                    box.clock.seconds = initial + 0.1

            monkeypatch.setattr(aiosqlite.Connection, "commit", cross_after_real_commit)
        else:
            armed = []

            async def hold_read(name):
                if name == "after_mutation_before_commit" and armed and not observations:
                    entered.set()
                    await release.wait()

            box.writer._failpoint = hold_read

            class StartQueuedRead(httpx.AsyncByteStream):
                async def __aiter__(self):
                    nonlocal queued_read
                    armed.append(True)
                    queued_read = asyncio.create_task(
                        box.writer.read_recording_archive(box.recording_id)
                    )
                    await entered.wait()
                    yield wav()

            await box.consumer._http.aclose()
            box.consumer._http = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, stream=StartQueuedRead())
                ),
                trust_env=False,
                follow_redirects=False,
            )
        try:
            transfer = asyncio.create_task(box.consumer.archive_recording_once(box.recording_id))
            if crossing == "queued_finish":
                await asyncio.wait_for(entered.wait(), 2)
                for _ in range(20):
                    if (box.directory / str(box.recording_id)).exists():
                        break
                    await asyncio.sleep(0)
                assert (box.directory / str(box.recording_id)).exists()
                observations.append(crossing)
                box.clock.seconds = initial + 0.1
                release.set()
            result = await asyncio.wait_for(transfer, 2)
            assert observations == [crossing]
            assert box.clock.utcnow() == DEADLINE
            assert result.outcome in {"expired", "unavailable"}
            assert result.native_acknowledged is False
            assert not list(box.directory.iterdir())
            assert await box.writer.read_recording_archive(box.recording_id) is None
            with sqlite3.connect(box.database) as connection:
                assert connection.execute(
                    "SELECT count(*) FROM sparra_content_fences"
                ).fetchone() == (1,)
        finally:
            release.set()
            if transfer is not None and not transfer.done():
                transfer.cancel()
            await asyncio.gather(
                *[task for task in (transfer, queued_read) if task is not None],
                return_exceptions=True,
            )


class SizedWav(httpx.AsyncByteStream):
    def __init__(self, total):
        self.total = total
        self.delivered = 0

    async def __aiter__(self):
        size = self.total - 44
        header = (
            b"RIFF"
            + struct.pack("<I", self.total - 8)
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 2, 8000, 32000, 4, 16)
            + b"data"
            + struct.pack("<I", size)
        )
        self.delivered += len(header)
        yield header
        remaining = size
        block = b"\x01\x00\x02\x00" * 16384
        while remaining:
            chunk = block[: min(65536, remaining)]
            self.delivered += len(chunk)
            yield chunk
            remaining -= len(chunk)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [33554432, 33554433])
async def test_bounds_exact_32mib_and_limit_plus_one_stream_without_truncation(tmp_path, size):
    stream = SizedWav(size)
    async with owned(tmp_path, respond=lambda request: httpx.Response(200, stream=stream)) as box:
        result = await box.consumer.archive_recording_once(box.recording_id)
        job = await box.writer.read_recording_archive(box.recording_id)
        if size == 33554432:
            assert result.outcome == "archived"
            assert job.receipt.encrypted_bytes == 33554448
            assert (box.directory / str(box.recording_id)).stat().st_size == 33554448
            assert stream.delivered == 33554432
        else:
            assert result.outcome == "unavailable" and job.receipt is None
            assert not list(box.directory.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["trickle", "aes", "before_commit"])
async def test_bounds_total_budget_survives_sync_and_receipt_boundaries(tmp_path, phase):
    references, advance = [], []

    def nonce(size):
        if advance and phase == "aes":
            references[0].clock.seconds = 121
        return os.urandom(size)

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            if phase == "trickle":
                references[0].clock.seconds = 121
            yield wav()

    async with owned(
        tmp_path, nonce_factory=nonce, respond=lambda request: httpx.Response(200, stream=Trickle())
    ) as box:
        references.append(box)
        advance.append(True)

        async def exhaust(name):
            if (
                name == "after_mutation_before_commit"
                and (box.directory / str(box.recording_id)).exists()
            ):
                box.clock.seconds = 121

        if phase == "before_commit":
            box.writer._failpoint = exhaust
        result = await box.consumer.archive_recording_once(box.recording_id)
        assert result.outcome == "unavailable"
        assert not list(box.directory.iterdir())
        assert (await box.writer.read_recording_archive(box.recording_id)).receipt is None


@pytest.mark.asyncio
async def test_bounds_file_fsync_failure_has_no_receipt_or_installed_file(tmp_path, monkeypatch):
    async with owned(tmp_path) as box:

        def fail(descriptor):
            raise OSError("owned fsync boundary")

        monkeypatch.setattr(os, "fsync", fail)
        result = await box.consumer.archive_recording_once(box.recording_id)
        assert result.outcome == "unavailable"
        assert not list(box.directory.iterdir())
        assert (await box.writer.read_recording_archive(box.recording_id)).receipt is None


@pytest.mark.asyncio
async def test_bounds_cleanup_sync_failure_withholds_ack_and_restart_retry_clears_metadata(
    tmp_path,
):
    fail = []

    def sync():
        if fail:
            raise OSError("owned cleanup directory sync boundary")

    async with owned(tmp_path, sync=sync) as box:
        await box.consumer.archive_recording_once(box.recording_id)
        fail.append(True)
        with pytest.raises((PersistenceError, OSError, archive_module().RecordingArchiveError)):
            await box.writer.erase_call_content(
                CALL_ID, now=box.clock.utcnow(), lease_token=UUID(int=47)
            )
        assert await box.writer.pending_erasure_acks() == ()
        assert await box.writer.read_recording_archive(box.recording_id) is not None
        fail.clear()
        await restart(box)
        await box.consumer.run_once()
        assert await box.writer.read_recording_archive(box.recording_id) is None
        assert not list(box.directory.iterdir())
