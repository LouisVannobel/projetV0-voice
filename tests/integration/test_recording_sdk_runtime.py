from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta

import httpx
import pytest

from projetv0_voice.persistence.postgres_sink import OperationSinkCommitAmbiguousError
from projetv0_voice.persistence.relay import OutboxRelay
from tests.integration.test_recording_archive import NOW, URL
from tests.integration.test_recording_archive_ownership import owned


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "missing_wav", "wrong_id", "rate_limit"])
async def test_actual_installed_sdk_retrieves_only_exact_fresh_wav_secret(failure, monkeypatch):
    module = importlib.import_module("projetv0_voice.telnyx.call_control")
    requests = []

    def api(request):
        requests.append(request)
        if failure == "rate_limit":
            return httpx.Response(429, json={"errors": [{"code": "rate_limited"}]})
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "foreign" if failure == "wrong_id" else "recording_Ab-12",
                    "call_control_id": "v3:control",
                    "call_leg_id": "leg-1",
                    "call_session_id": "session-1",
                    "channels": "dual",
                    "status": "completed",
                    "source": "call",
                    "initiated_by": "StartCallRecordingAPI",
                    "recording_started_at": NOW.isoformat(),
                    "recording_ended_at": (NOW + timedelta(minutes=1)).isoformat(),
                    "download_urls": {
                        "mp3": "https://foreign.invalid/mp3",
                        "wav": None if failure == "missing_wav" else URL,
                    },
                }
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(api), trust_env=False)
    monkeypatch.setattr(module.telnyx, "DefaultAsyncHttpxClient", lambda **kwargs: transport)
    client = module.CallControlClient(api_key="SYNTHETIC-API-KEY")
    function = getattr(client, "retrieve_recording_download", None)
    try:
        if not callable(function):
            pytest.fail("the real installed Telnyx SDK has no exact WAV archive consumer")
        if failure is not None:
            with pytest.raises(
                module.RecordingCatalogTransientError
                if failure == "rate_limit"
                else module.RecordingCatalogInvalidError
            ):
                await function("recording_Ab-12", timeout_seconds=0.75)
        else:
            result = await function("recording_Ab-12", timeout_seconds=0.75)
            assert result.recording.recording_id == "recording_Ab-12"
            assert result.wav_url.get_secret_value() == URL
            assert result.retrieved_at is not None
            assert URL not in repr(result)
        assert len(requests) == 1
        assert (
            requests[0].method == "GET" and requests[0].url.path == "/v2/recordings/recording_Ab-12"
        )
        assert requests[0].url.query == b""
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_native_sink_commit_ack_alone_advances_archive_readiness(unknown, tmp_path):
    async with owned(tmp_path) as box:
        assert (await box.consumer.archive_recording_once(box.recording_id)).outcome == "archived"
        operations = []

        class Sink:
            async def ingest(self, operation):
                operations.append(operation)
                if unknown and getattr(operation.payload, "archive_receipt", None) is not None:
                    raise OperationSinkCommitAmbiguousError("postgres_commit_unknown")

        async def no_op():
            pass

        relay = OutboxRelay(
            box.writer, Sink(), on_degraded=no_op, drain=no_op, utcnow=box.clock.utcnow
        )
        if unknown:
            with pytest.raises(OperationSinkCommitAmbiguousError):
                await relay.run_once(batch_size=10)
        else:
            await relay.run_once(batch_size=10)
        job = await box.writer.read_recording_archive(box.recording_id)
        assert job.state == ("archived" if unknown else "acknowledged")
        assert (
            await box.consumer.archive_recording_once(box.recording_id)
        ).native_acknowledged is (not unknown)
        assert any(getattr(op.payload, "archive_receipt", None) is not None for op in operations)


@pytest.mark.asyncio
async def test_runtime_supervisor_runs_archive_even_off_and_closes_before_writer(tmp_path):
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.lifecycle import RuntimeSupervisor
    from projetv0_voice.persistence.writer import PersistenceWriter

    events = []

    class Archive:
        async def prepare(self):
            pass

        async def run_once(self):
            events.append("archive-maintenance")

        async def aclose(self):
            events.append("archive-close")
            await writer.read_call_lifecycle(__import__("uuid").UUID(int=1))

    writer = PersistenceWriter(
        tmp_path / "life.sqlite", CryptoKeyring({1: bytes(range(32))}, active_version=1)
    )
    supervisor = RuntimeSupervisor(writer=writer, loop_interval_seconds=0.01)
    if "archive" not in __import__("inspect").signature(RuntimeSupervisor).parameters:
        pytest.fail("the existing runtime maintenance/close lifecycle has no archive consumer")
    supervisor = RuntimeSupervisor(writer=writer, archive=Archive(), loop_interval_seconds=0.01)
    await supervisor.startup()
    await asyncio.sleep(0.05)
    await supervisor.aclose()
    assert "archive-maintenance" in events and events[-1] == "archive-close"
