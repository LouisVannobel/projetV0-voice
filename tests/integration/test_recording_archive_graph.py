from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path, PurePosixPath
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr
from test_recording_archive import KEY, NOW, Provider, queue_saved, saved_event, wav

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.lifecycle import (
    RuntimeInferenceFactories,
    RuntimeProductionFactories,
    RuntimeProfileSelection,
    build_production_runtime,
)
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.runtime_config import (
    RuntimeSettingsV1,
    capture_runtime_environment,
    parse_runtime_settings,
)


def graph_settings(tmp_path, profile, *, configured=True):
    paths = RuntimeSettingsV1(
        runtime_mode="strict",
        deployment_id=profile.deployment_id,
        runtime_contract_path=PurePosixPath("/srv/projetv0/runtime.json"),
        agent_bundle_path=PurePosixPath("/srv/projetv0/bundle"),
        qualified_profile_path=PurePosixPath("/srv/projetv0/profile.json"),
        qualification_candidate_path=None,
        qualification_override_path=None,
        keyring_path=PurePosixPath("/run/secrets/aead_keyring_v1.json"),
        sqlite_path=tmp_path / "voice.sqlite",
        runtime_contract_sha256=profile.runtime_contract_sha256,
        image_digest=profile.image_digest,
        agent_bundle_sha256=profile.agent_bundle_sha256,
        inference_profile_sha256=profile.inference_profile_sha256,
        qualification_run_id=None,
        benchmark_did_sha256=None,
        deployment_max_calls=1,
        handshake_timeout_seconds=5,
        call_idle_timeout_seconds=300,
        call_cleanup_phase_timeout_seconds=10,
        pre_drain_grace_seconds=15,
        uvicorn_grace_seconds=20,
        shutdown_grace_seconds=30,
        telnyx_api_key_file=PurePosixPath("/run/secrets/telnyx"),
        telnyx_webhook_public_key_file=PurePosixPath("/run/secrets/webhook"),
        openrouter_api_key_file=PurePosixPath("/run/secrets/openrouter"),
        postgres_dsn_file=PurePosixPath("/run/secrets/postgres"),
        telnyx_media_wss_url="wss://voice.invalid/telnyx/media",
        otlp_http_endpoint="https://collector.invalid/v1/metrics",
        bind_host="127.0.0.1",
        bind_port=8080,
        recording_archive_directory=tmp_path / "audio" if configured else None,
        recording_download_origins=("https://recordings.example.invalid",) if configured else (),
    )
    values = {
        "VOICE_" + field.name.upper(): (",".join(value) if isinstance(value, tuple) else str(value))
        for field in fields(paths)
        if not field.name.startswith("_")
        and (value := getattr(paths, field.name)) is not None
        and value != ()
    }
    values["VOICE_SQLITE_PATH"] = "/var/lib/projetv0/voice.sqlite"
    if configured:
        values["VOICE_RECORDING_ARCHIVE_DIRECTORY"] = "/var/lib/projetv0/audio"
    parsed = parse_runtime_settings(
        capture_runtime_environment(values), geteuid=lambda: 10001, getegid=lambda: 10001
    )
    # Only controlled OS paths differ; readiness token comes from the real parser.
    result = replace(
        parsed,
        sqlite_path=paths.sqlite_path,
        recording_archive_directory=paths.recording_archive_directory,
    )
    object.__setattr__(result, "_observability_token", parsed.observability_token())
    return result


async def build_archive_graph(
    tmp_path,
    monkeypatch,
    *,
    configured=True,
    failed_start=False,
    active_copy=False,
    failed_composition=False,
    audit=None,
    broken_archive=False,
    purge_state=None,
):
    if "archive_factory" not in inspect.signature(RuntimeProductionFactories).parameters:
        pytest.fail("the actual production composition has no archive factory consumer")
    from projetv0_voice.recording_archive import RecordingArchive

    profile_data = json.loads(
        await asyncio.to_thread(
            Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text
        )
    )
    profile_data["telnyx_api_key_sha256"] = hashlib.sha256(b"synthetic-key").hexdigest()
    profile = QualifiedDeploymentProfileV1.model_validate(profile_data)
    settings = graph_settings(tmp_path, profile, configured=configured)
    manifest = AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "tenant_id": "tenant-a",
            "agent_id": "agent-a",
            "revision": "r",
            "dids": ["+33123456789"],
            "language": "fr",
            "prompt_path": tmp_path / "prompt.md",
            "prompt_revision": "p",
            "greeting": "Bonjour",
            "conversation_mode": "freeform",
            "max_concurrent_calls": 1,
            "direction": "inbound_only",
            "transport_codec": "PCMU",
            "transport_sample_rate_hz": 8000,
            "transcript_retention_days": 7,
            "recording_mode": "off",
            "recording_format": "wav",
            "recording_retention_days": None,
            "recording_required": False,
            "recording_play_beep": False,
        }
    )
    events, requests, consumers = [], [], []
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Control(Provider):
        def __init__(self):
            super().__init__()
            self.downloads = []

        async def retrieve_recording_download(self, recording_id, *, timeout_seconds):
            self.downloads.append(recording_id)
            return await super().retrieve_recording_download(
                recording_id, timeout_seconds=timeout_seconds
            )

        async def aclose(self):
            events.append("control-close")

        async def delete_recording(self, recording_id, *, timeout_seconds):
            from projetv0_voice.telnyx.recordings import ProviderDeleteResultV1

            assert purge_state is not None and recording_id == "recording_Purge-A"
            assert 0 < timeout_seconds <= 1
            purge_state["steps"].append("delete")
            return ProviderDeleteResultV1("deleted", recording_id)

    class Sink:
        async def open(self):
            if failed_start:
                raise RuntimeError("synthetic-startup-failure")

        async def close(self):
            events.append("sink-close")

        async def ingest(self, _operation):
            pass

        async def lease_recording_purges(self, *_args):
            if purge_state is not None and purge_state["due"]:
                from projetv0_voice.persistence.postgres_sink import RecordingPurgeLease

                purge_state["steps"].append("lease")
                purge_state["due"] = False
                return (
                    RecordingPurgeLease(
                        1,
                        UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
                        UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
                        "recording_Purge-A",
                        1,
                        NOW + timedelta(minutes=2, seconds=30),
                    ),
                )
            return ()

        async def ack_recording_purge(self, recording_id, lease_token, outcome, occurred_at):
            assert recording_id == UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
            assert lease_token == UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
            assert outcome == "deleted" and occurred_at == NOW + timedelta(minutes=2)
            purge_state["steps"].append("ack")
            purge_state["acked"].set()

    async def response(request):
        requests.append(request)
        if active_copy:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return httpx.Response(200, content=wav())

    def archive_factory(received, writer, keyring, control, utcnow):
        assert received is settings
        if not configured:
            return None

        class Archive(RecordingArchive):
            async def aclose(self):
                if not self._closed:
                    # Existing queue owner must still be running during archive retirement.
                    if writer._ready_ok:
                        await writer.read_call_lifecycle(UUID(int=1))
                    events.append("archive-close")
                await super().aclose()

        directory = Path(str(settings.recording_archive_directory))
        directory.mkdir(mode=0o700)
        (directory / str(UUID(int=29))).write_bytes(b"owned-uncommitted-orphan")
        if broken_archive:
            (directory / "foreign-inventory").write_bytes(b"unowned-name")
        archive = Archive(
            directory=directory,
            writer=writer,
            keyring=keyring,
            telnyx=control,
            allowed_origins=settings.recording_download_origins,
            utcnow=utcnow,
            download_transport=httpx.MockTransport(response),
            directory_sync=lambda: None,
        )
        consumers.append(archive)
        return archive

    secrets = {
        settings.telnyx_api_key_file: SecretStr("synthetic-key"),
        settings.telnyx_webhook_public_key_file: SecretStr(
            base64.b64encode(bytes(range(32))).decode()
        ),
        settings.openrouter_api_key_file: SecretStr("synthetic-inference"),
        settings.postgres_dsn_file: SecretStr("synthetic-dsn"),
    }
    monkeypatch.setattr(
        RuntimeMetrics,
        "production",
        classmethod(lambda *_args, **_kwargs: RuntimeMetrics.in_memory()),
    )

    def inference_factory(*_args):
        if failed_composition:
            raise RuntimeError("synthetic-composition-failure")
        return RuntimeInferenceFactories(
            lambda: object(), lambda _: object(), lambda: object(), lambda: object()
        )

    factories = RuntimeProductionFactories(
        validate_artifacts=lambda _: None,
        load_manifest=lambda _: manifest,
        load_profile=lambda *_: RuntimeProfileSelection(profile=profile, override=None),
        read_secret=lambda path: secrets[path],
        load_keyring=lambda _: CryptoKeyring({1: KEY}, active_version=1),
        sink_factory=lambda _: Sink(),
        call_control_factory=lambda *_: Control(),
        inference_factory=inference_factory,
        archive_factory=archive_factory,
    )
    if audit is not None:
        audit.update(events=events, consumers=consumers)
    graph = await build_production_runtime(
        settings,
        factories=factories,
        utcnow=lambda: NOW.replace(minute=2),
        shutdown_timeout_seconds=0.6,
    )
    return graph, events, requests, consumers, entered, cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_archive_graph_off_independent_and_restart_prepared_before_admission(
    tmp_path, monkeypatch, configured
):
    graph, events, requests, consumers, _, _ = await build_archive_graph(
        tmp_path, monkeypatch, configured=configured
    )
    assert getattr(graph, "archive", None) is (consumers[0] if configured else None)
    await graph.supervisor.startup()
    try:
        assert graph.supervisor.readiness_snapshot().admission_open
        if configured:
            assert list(consumers[0]._directory.iterdir()) == []
        assert requests == []
    finally:
        await graph.supervisor.aclose()
    if configured:
        assert events.index("archive-close") < events.index("control-close")


@pytest.mark.asyncio
async def test_real_graph_due_provider_purge_progresses_while_other_media_is_held(
    tmp_path, monkeypatch
):
    state = {"due": False, "steps": [], "acked": asyncio.Event()}
    graph, events, requests, consumers, entered, cancelled = await build_archive_graph(
        tmp_path, monkeypatch, active_copy=True, purge_state=state
    )
    await graph.supervisor.startup()
    try:
        await queue_saved(graph.writer, saved_event())
        await asyncio.wait_for(entered.wait(), 1)
        # Become due only after B is held; moving purge before the copy does not satisfy this.
        state["due"] = True
        await asyncio.wait_for(state["acked"].wait(), 1)
        assert state["steps"] == ["lease", "delete", "ack"]
        assert not cancelled.is_set() and consumers[0]._active_task is not None
        assert len(requests) == 1
    finally:
        await graph.supervisor.aclose()
    assert cancelled.is_set() and consumers[0]._http.is_closed
    assert consumers[0]._active_task is None
    assert events.index("archive-close") < events.index("control-close")
    assert graph.supervisor._writer_task.done()
    assert all(task.done() for task in graph.supervisor._fixed_tasks.values())


@pytest.mark.asyncio
async def test_archive_graph_failed_startup_retires_consumer_before_writer(tmp_path, monkeypatch):
    graph, events, requests, consumers, _, _ = await build_archive_graph(
        tmp_path, monkeypatch, failed_start=True
    )
    with pytest.raises(RuntimeError, match="runtime_startup_failed"):
        await graph.supervisor.startup()
    assert consumers[0]._closed and requests == []
    assert events.index("archive-close") < events.index("control-close")
    await graph.supervisor.aclose()


@pytest.mark.asyncio
async def test_archive_graph_shutdown_joins_real_active_copy_before_purge_owner(
    tmp_path, monkeypatch
):
    graph, _, requests, consumers, entered, cancelled = await build_archive_graph(
        tmp_path, monkeypatch, active_copy=True
    )
    await graph.supervisor.startup()
    await queue_saved(graph.writer, saved_event())
    await asyncio.wait_for(entered.wait(), 1)
    await graph.supervisor.aclose()
    assert cancelled.is_set() and len(requests) == 1 and consumers[0]._closed
    assert list(consumers[0]._directory.iterdir()) == []


@pytest.mark.asyncio
async def test_archive_graph_failed_composition_retires_owned_http_without_started_writer(
    tmp_path, monkeypatch
):
    # Composition can fail after archive construction but before writer startup.
    audit = {}
    with pytest.raises(RuntimeError, match="runtime_production_composition_failed"):
        await build_archive_graph(tmp_path, monkeypatch, failed_composition=True, audit=audit)
    assert len(audit["consumers"]) == 1 and audit["consumers"][0]._http.is_closed
    assert audit["events"].index("archive-close") < audit["events"].index("control-close")


@pytest.mark.asyncio
async def test_archive_graph_broken_inventory_refuses_capacity_but_keeps_off_admissible(
    tmp_path, monkeypatch
):
    from test_recording_archive import CALL_ID, identity

    from projetv0_voice.persistence.commands import PersistenceError

    graph, _, requests, consumers, _, _ = await build_archive_graph(
        tmp_path, monkeypatch, broken_archive=True
    )
    try:
        await graph.supervisor.startup()
        assert graph.supervisor.readiness_snapshot().admission_open
        assert consumers[0]._failed and requests == []
        assert graph.raw_call_control.downloads == []
        # Controlled historical pin metadata tests real reserve failure, not qualification.
        await queue_saved(graph.writer, saved_event(), historic_pin=False)
        await graph.writer.bind_recording_policy(
            identity().begin_snapshot,
            generation=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        )
        with pytest.raises(PersistenceError, match="recording_archive_unavailable"):
            await graph.writer.reserve_recording_audio(
                CALL_ID, generation=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
            )
        assert not graph.writer.fatal_event.is_set()
        assert requests == [] and graph.raw_call_control.downloads == []
    finally:
        await graph.supervisor.aclose()
