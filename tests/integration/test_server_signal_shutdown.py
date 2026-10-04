from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import signal
import socket
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from projetv0_voice.runtime_config import RuntimeSettingsV1


def _qualified_profile_json() -> str:
    return Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(
        encoding="utf-8"
    )


def _signal_process_script(mode: str) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import os
        import signal
        import socket
        import sys
        import tempfile
        from datetime import UTC, datetime, timedelta
        from pathlib import Path
        from uuid import UUID

        MODE = {mode!r}
        from projetv0_voice.runtime_config import (
            capture_runtime_environment,
            parse_runtime_settings,
        )
        runtime_environment = {{
            "VOICE_RUNTIME_MODE": "strict",
            "VOICE_DEPLOYMENT_ID": "voice-agent-a",
            "VOICE_RUNTIME_CONTRACT_PATH": "/srv/runtime.json",
            "VOICE_AGENT_BUNDLE_PATH": "/srv/bundle",
            "VOICE_QUALIFIED_PROFILE_PATH": "/srv/profile.json",
            "VOICE_KEYRING_PATH": "/run/secrets/aead_keyring_v1.json",
            "VOICE_SQLITE_PATH": "/var/lib/voice.sqlite3",
            "VOICE_RUNTIME_CONTRACT_SHA256": "a" * 64,
            "VOICE_IMAGE_DIGEST": "ghcr.io/example/voice@sha256:" + "d" * 64,
            "VOICE_AGENT_BUNDLE_SHA256": "b" * 64,
            "VOICE_INFERENCE_PROFILE_SHA256": "c" * 64,
            "VOICE_DEPLOYMENT_MAX_CALLS": "1",
            "VOICE_HANDSHAKE_TIMEOUT_SECONDS": "5",
            "VOICE_CALL_IDLE_TIMEOUT_SECONDS": "300",
            "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS": "10",
            "VOICE_PRE_DRAIN_GRACE_SECONDS": "1",
            "VOICE_UVICORN_GRACE_SECONDS": "1",
            "VOICE_SHUTDOWN_GRACE_SECONDS": "5",
            "VOICE_TELNYX_API_KEY_FILE": "/run/secrets/telnyx",
            "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE": "/run/secrets/webhook",
            "VOICE_OPENROUTER_API_KEY_FILE": "/run/secrets/openrouter",
            "VOICE_POSTGRES_DSN_FILE": "/run/secrets/postgres",
            "VOICE_TELNYX_MEDIA_WSS_URL": "wss://voice.invalid/telnyx/media",
            "VOICE_OTLP_HTTP_ENDPOINT": "https://collector.invalid/v1/metrics",
            "VOICE_BIND_HOST": "127.0.0.1",
            "VOICE_BIND_PORT": "18080",
        }}
        capture = capture_runtime_environment(runtime_environment)
        settings = parse_runtime_settings(
            capture,
            geteuid=lambda: 10001,
            getegid=lambda: 10001,
        )

        from projetv0_voice.dependency_logging import configure_dependency_logging

        configure_dependency_logging(settings.observability_token())

        from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

        input_used = False

        async def input_line():
            global input_used
            if input_used:
                return
            input_used = True
            await asyncio.to_thread(sys.stdin.buffer.readline)

        def observe_server_start(server):
            started = asyncio.Event()
            original_startup = server._server.startup

            async def observed_startup(*args, **kwargs):
                await original_startup(*args, **kwargs)
                started.set()

            server._server.startup = observed_startup
            return started

        async def barrier(name):
            if MODE == name:
                print("PHASE:" + name, flush=True)
                await input_line()

        def global_hard_exit(code):
            print("GLOBAL_HARD:" + str(code), flush=True)
            os._exit(code)

        def runtime_hard_exit(code):
            print("RUNTIME_HARD:" + str(code), flush=True)
            os._exit(73)

        shutdown_seen = False
        gate_opened = False

        class App:
            def __init__(self, coordinator, supervisor):
                self.coordinator = coordinator
                self.supervisor = supervisor

            async def __call__(self, scope, receive, send):
                global shutdown_seen, gate_opened
                assert scope["type"] == "lifespan"
                assert (await receive())["type"] == "lifespan.startup"
                failure = None
                try:
                    if MODE == "before_runtime_publication":
                        await barrier("before_runtime_publication")
                    self.coordinator.publish_runtime(self.supervisor)
                    await self.supervisor.startup()
                    gate_opened = True
                    self.coordinator.publish_startup_complete()
                    await send({{"type": "lifespan.startup.complete"}})
                    assert (await receive())["type"] == "lifespan.shutdown"
                    shutdown_seen = True
                except BaseException as error:
                    failure = error
                    self.coordinator.publish_startup_failure()
                try:
                    await self.supervisor.aclose()
                except BaseException:
                    self.coordinator.publish_startup_failure()
                    raise
                finally:
                    self.coordinator.unpublish_runtime(self.supervisor)
                    print("UNPUBLISHED", flush=True)
                    print("GATE:" + str(int(gate_opened)), flush=True)
                if isinstance(failure, asyncio.CancelledError):
                    print("CANCELLED", flush=True)
                    raise failure
                if failure is not None:
                    await send({{
                        "type": "lifespan.startup.failed",
                        "message": "runtime_startup_failed",
                    }})
                    return
                print("SHUTDOWN_COMPLETE", flush=True)
                await send({{"type": "lifespan.shutdown.complete"}})

        if MODE.startswith("programmatic_"):
            class ProgrammaticSupervisor:
                def __init__(self):
                    self.raw_begin_drain_calls = 0
                    self.effective_drain_transitions = 0
                    self.aclose_calls = 0
                    self._draining = False

                async def startup(self):
                    return None

                async def begin_drain(self):
                    self.raw_begin_drain_calls += 1
                    if not self._draining:
                        self._draining = True
                        self.effective_drain_transitions += 1
                        print("DRAIN", flush=True)

                async def aclose(self):
                    self.aclose_calls += 1
                    print("ACLOSE_STARTED", flush=True)
                    await self.begin_drain()
                    if MODE == "programmatic_hard_deadline":
                        await asyncio.Event().wait()
                    await input_line()
                    print("ACLOSE_FINISHED", flush=True)

            async def run_programmatic():
                supervisor = ProgrammaticSupervisor()
                coordinator = FirstSignalDrainCoordinator(
                    settings=settings,
                    hard_exit=global_hard_exit,
                )
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                server = VoiceUvicornServer(
                    App(coordinator, supervisor),
                    settings=settings,
                    coordinator=coordinator,
                )
                started = observe_server_start(server)
                task = asyncio.create_task(server.serve(sockets=[listener]))
                await asyncio.wait_for(started.wait(), timeout=5.0)
                print("READY", flush=True)
                server.handle_exit(signal.SIGTERM, None)
                try:
                    await task
                finally:
                    listener.close()
                print(
                    "SUMMARY:"
                    + ":".join(
                        (
                            str(supervisor.raw_begin_drain_calls),
                            str(supervisor.effective_drain_transitions),
                            str(int(supervisor._draining)),
                            str(int(server.force_exit)),
                            str(int(coordinator.urgent)),
                            str(int(shutdown_seen)),
                        )
                    ),
                    flush=True,
                )

            asyncio.run(run_programmatic())
            raise SystemExit(0)

        from projetv0_voice.crypto import CryptoKeyring
        from projetv0_voice.lifecycle import RuntimeSupervisor
        from projetv0_voice.metrics import RuntimeMetrics
        from projetv0_voice.persistence.writer import PersistenceWriter
        from projetv0_voice.telnyx.call_control import CallControlResult

        class BarrierWriter(PersistenceWriter):
            async def wait_ready(self):
                result = await super().wait_ready()
                await barrier("writer_ready")
                return result

            async def quick_check(self):
                result = await super().quick_check()
                await barrier("writer_quick_check")
                return result

            async def qualification_run_consumed(self, run_id):
                result = await super().qualification_run_consumed(run_id)
                await barrier("qualification_status")
                return result

            async def runtime_observation(self):
                result = await super().runtime_observation()
                await barrier("runtime_publication")
                return result

        class Sink:
            async def open(self):
                await barrier("sink_open")
                if MODE == "startup_failure":
                    print("STARTUP_FAILURE", flush=True)
                    raise RuntimeError("synthetic startup failure")

            async def close(self):
                if MODE == "hard_deadline":
                    await asyncio.Event().wait()

        class Control:
            async def answer(self, *_args, **_kwargs):
                return CallControlResult("accepted")

            async def start_streaming(self, *_args, **_kwargs):
                return CallControlResult("accepted")

            async def hangup(self, *_args, **_kwargs):
                await barrier("stale_recovery")
                return CallControlResult("accepted")

            async def aclose(self):
                return None

        class ObservableSupervisor(RuntimeSupervisor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.raw_begin_drain_calls = 0
                self.effective_drain_transitions = 0
                self.aclose_calls = 0

            async def begin_drain(self):
                self.raw_begin_drain_calls += 1
                was_draining = self._draining
                await super().begin_drain()
                if not was_draining and self._draining:
                    self.effective_drain_transitions += 1
                    print("DRAIN", flush=True)

            async def aclose(self):
                self.aclose_calls += 1
                print("ACLOSE_STARTED", flush=True)
                await super().aclose()
                print("ACLOSE_FINISHED", flush=True)

        async def seed_stale(path, keyring):
            seed = PersistenceWriter(path, keyring)
            owner = asyncio.create_task(seed.run())
            assert await seed.wait_ready()
            now = datetime.now(UTC)
            await seed.commit_lease(
                call_control_id="stale-control",
                call_id=UUID("11111111-1111-4111-8111-111111111111"),
                tenant_id="tenant-a",
                agent_id="agent-a",
                state="pending",
                token_hash=b"d" * 32,
                created_at=now - timedelta(minutes=2),
                expires_at=now - timedelta(minutes=1),
                closed_at=None,
            )
            await seed.drain(2.0)
            await owner

        async def run():
            temporary = tempfile.TemporaryDirectory(prefix="voice-signal-")
            database = Path(temporary.name) / "voice.sqlite3"
            keyring = CryptoKeyring({{1: b"k" * 32}}, active_version=1)
            if MODE == "stale_recovery":
                await seed_stale(database, keyring)
            writer = BarrierWriter(database, keyring)
            metrics = RuntimeMetrics.in_memory()
            candidate = (
                UUID("22222222-2222-4222-8222-222222222222")
                if MODE == "qualification_status"
                else None
            )
            supervisor = ObservableSupervisor(
                writer=writer,
                call_control=Control(),
                metrics=metrics,
                sink=Sink(),
                candidate_run_id=candidate,
                startup_phase_timeout_seconds=30.0,
                shutdown_timeout_seconds=(
                    30.0 if MODE == "hard_deadline" else 5.0
                ),
                hard_exit=runtime_hard_exit,
            )
            coordinator = FirstSignalDrainCoordinator(
                settings=settings,
                hard_exit=global_hard_exit,
            )
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            server = VoiceUvicornServer(
                App(coordinator, supervisor),
                settings=settings,
                coordinator=coordinator,
            )
            started = observe_server_start(server)
            task = asyncio.create_task(server.serve(sockets=[listener]))
            if MODE in {{"postgates", "hard_deadline"}}:
                await asyncio.wait_for(started.wait(), timeout=5.0)
                print("READY", flush=True)
            try:
                try:
                    await task
                except SystemExit as error:
                    print("SERVER_EXIT:" + str(error.code), flush=True)
                    if MODE != "startup_failure":
                        raise
            finally:
                listener.close()
                temporary.cleanup()
            print(
                "SUMMARY:"
                + ":".join(
                    (
                        str(supervisor.raw_begin_drain_calls),
                        str(supervisor.effective_drain_transitions),
                        str(int(supervisor._draining)),
                        str(int(server.force_exit)),
                        str(int(coordinator.urgent)),
                        str(int(shutdown_seen)),
                    )
                ),
                flush=True,
            )

        try:
            asyncio.run(run())
        except SystemExit as error:
            print("PROCESS_EXIT:" + str(error.code), flush=True)
            if MODE != "startup_failure":
                raise
        """
    )


def _settings(*, port: int = 8080, uvicorn_grace_seconds: int = 2) -> RuntimeSettingsV1:
    return RuntimeSettingsV1(
        runtime_mode="strict",
        deployment_id="voice-agent-a",
        runtime_contract_path=PurePosixPath("/srv/projetv0/runtime-contract.json"),
        agent_bundle_path=PurePosixPath("/srv/projetv0/agent-bundle"),
        qualified_profile_path=PurePosixPath("/srv/projetv0/qualified.json"),
        qualification_candidate_path=None,
        qualification_override_path=None,
        keyring_path=PurePosixPath("/srv/projetv0/keyring.json"),
        sqlite_path=PurePosixPath("/var/lib/projetv0/voice.sqlite3"),
        runtime_contract_sha256="a" * 64,
        image_digest=f"ghcr.io/example/voice@sha256:{'d' * 64}",
        agent_bundle_sha256="b" * 64,
        inference_profile_sha256="c" * 64,
        qualification_run_id=None,
        benchmark_did_sha256=None,
        deployment_max_calls=1,
        handshake_timeout_seconds=5,
        call_idle_timeout_seconds=300,
        call_cleanup_phase_timeout_seconds=10,
        pre_drain_grace_seconds=1,
        uvicorn_grace_seconds=uvicorn_grace_seconds,
        shutdown_grace_seconds=5,
        telnyx_api_key_file=PurePosixPath("/run/secrets/telnyx-api-key"),
        telnyx_webhook_public_key_file=PurePosixPath(
            "/run/secrets/telnyx-webhook-key"
        ),
        openrouter_api_key_file=PurePosixPath("/run/secrets/openrouter-api-key"),
        postgres_dsn_file=PurePosixPath("/run/secrets/postgres-dsn"),
        telnyx_media_wss_url="wss://voice.invalid/telnyx/media",
        otlp_http_endpoint="https://collector.invalid/v1/metrics",
        bind_host="127.0.0.1",
        bind_port=port,
    )


def _observe_server_start(
    server: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Event:
    """Return a test-owned event set at Uvicorn's actual startup boundary."""

    started = asyncio.Event()
    original_startup = server._server.startup  # noqa: SLF001

    async def observed_startup(*args: Any, **kwargs: Any) -> None:
        await original_startup(*args, **kwargs)
        started.set()

    monkeypatch.setattr(server._server, "startup", observed_startup)  # noqa: SLF001
    return started


class _LoopbackApp:
    def __init__(self) -> None:
        self.startup = asyncio.Event()
        self.shutdown = asyncio.Event()

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] == "lifespan":
            assert (await receive())["type"] == "lifespan.startup"
            self.startup.set()
            await send({"type": "lifespan.startup.complete"})
            assert (await receive())["type"] == "lifespan.shutdown"
            self.shutdown.set()
            await send({"type": "lifespan.shutdown.complete"})
            return
        assert scope["type"] == "http"
        if scope["path"].startswith("/raise"):
            raise RuntimeError("EXCEPTION-SENTINEL")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"0")],
            }
        )
        await send({"type": "http.response.body", "body": b""})


def test_uvicorn_config_loads_every_literal_surface_value() -> None:
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

    settings = _settings(port=43210)
    server = VoiceUvicornServer(
        _LoopbackApp(),
        settings=settings,
        coordinator=FirstSignalDrainCoordinator(settings=settings),
    )
    server.config.load()

    assert server.config.host == "127.0.0.1"
    assert server.config.port == 43210
    assert server.config.workers == 1
    assert server.config.reload is False
    assert getattr(server.config, "env_file", None) is None
    assert server.config.lifespan == "on"
    assert server.config.loop == "asyncio"
    assert server.config.ws == "websockets-sansio"
    assert server.config.ws_max_size == 524_288
    assert server.config.ws_per_message_deflate is False
    assert server.config.timeout_graceful_shutdown == 2
    assert server.config.log_config is None
    assert server.config.log_level is None
    assert server.config.access_log is False
    assert server.config.proxy_headers is False
    assert server.config.forwarded_allow_ips == []


@pytest.mark.parametrize(
    "mode",
    ["postgates", "before_runtime_publication", "writer_ready"],
)
def test_signal_process_harness_compiles_on_every_host(mode: str) -> None:
    compile(_signal_process_script(mode), "<signal-process>", "exec")


@pytest.mark.asyncio
async def test_default_runtime_builder_uses_lazy_concrete_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.lifecycle as lifecycle
    import projetv0_voice.production_wiring as wiring
    import projetv0_voice.server as server_module

    settings = _settings()
    factories = object()
    graph = object()
    calls: list[tuple[str, object]] = []

    def build_factories(received: RuntimeSettingsV1) -> object:
        calls.append(("factories", received))
        return factories

    async def build_runtime(
        received: RuntimeSettingsV1,
        *,
        factories: object,
    ) -> object:
        calls.append(("runtime", (received, factories)))
        return graph

    monkeypatch.setattr(wiring, "build_production_factories", build_factories)
    monkeypatch.setattr(lifecycle, "build_production_runtime", build_runtime)

    assert await server_module._default_runtime_builder(settings) is graph
    assert calls == [
        ("factories", settings),
        ("runtime", (settings, factories)),
    ]


@pytest.mark.asyncio
async def test_real_loopback_server_preserves_lifespan_and_redacts_protocol_errors(
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    app = _LoopbackApp()
    settings = _settings(port=port)
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    server_started = _observe_server_start(server, monkeypatch)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    await app.startup.wait()
    await asyncio.wait_for(server_started.wait(), timeout=5.0)
    assert server.started is True

    async with httpx.AsyncClient(trust_env=False, timeout=2.0) as client:
        response = await client.get(
            f"http://127.0.0.1:{port}/raise/PATH-SENTINEL?value=QUERY-SENTINEL"
        )
    assert response.status_code == 500

    server.should_exit = True
    await asyncio.wait_for(task, timeout=5.0)
    assert app.shutdown.is_set()
    assert server.force_exit is False
    listener.close()

    captured = capfd.readouterr()
    rendered = captured.out + captured.err + caplog.text
    assert "PATH-SENTINEL" not in rendered
    assert "QUERY-SENTINEL" not in rendered
    assert "EXCEPTION-SENTINEL" not in rendered
    assert not any(
        record.name.startswith("uvicorn") and record.levelno <= logging.INFO
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_real_sansio_loopback_rejects_websocket_byte_524289(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

    class OversizeApp:
        def __init__(self) -> None:
            self.startup = asyncio.Event()
            self.shutdown = asyncio.Event()
            self.received: list[dict[str, Any]] = []

        async def __call__(
            self,
            scope: dict[str, Any],
            receive: Any,
            send: Any,
        ) -> None:
            if scope["type"] == "lifespan":
                assert (await receive())["type"] == "lifespan.startup"
                self.startup.set()
                await send({"type": "lifespan.startup.complete"})
                assert (await receive())["type"] == "lifespan.shutdown"
                self.shutdown.set()
                await send({"type": "lifespan.shutdown.complete"})
                return
            assert scope["type"] == "websocket"
            assert (await receive())["type"] == "websocket.connect"
            await send({"type": "websocket.accept"})
            self.received.append(await receive())

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    app = OversizeApp()
    settings = _settings(port=port)
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    server_started = _observe_server_start(server, monkeypatch)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    await app.startup.wait()
    await asyncio.wait_for(server_started.wait(), timeout=5.0)
    assert server.started is True

    try:
        async with connect(
            f"ws://127.0.0.1:{port}/telnyx/media",
            max_size=None,
        ) as websocket:
            await websocket.send(b"x" * 524_289)
            with pytest.raises(ConnectionClosed) as closed:
                await websocket.recv()
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == 1009
        assert len(app.received) == 1
        assert app.received[0]["type"] == "websocket.disconnect"
        assert int(app.received[0]["code"]) == 1009
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        listener.close()
    assert app.shutdown.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    ["denial", "cancel-accept", "drain-success", "cancel-transfer"],
)
async def test_real_production_wss_route_owns_one_task6_permit_under_eager_factory(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import asynccontextmanager

    from starlette.websockets import WebSocket
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    from projetv0_voice.admission import SynchronousUnauthenticatedGate
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer
    from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshakeService

    token = "synthetic-auth-token"
    call_control_id = "sensitive-call-control-id"
    token_digest = hashlib.sha256(token.encode()).digest()
    fixture_sha256 = "6ef76b42d6f0db4a60fdeda1e6c4a202363674e4eea3d04a82977cba9ef6da80"
    connected = json.dumps(
        {
            "protocol": "Call",
            "version": "1.0.0",
            "event": "connected",
            "connected": {"x-telnyx-streaming-auth-token": token},
        }
    )
    start = json.dumps(
        {
            "stream_id": "sensitive-stream-id",
            "event": "start",
            "sequence_number": "1",
            "start": {
                "call_control_id": call_control_id,
                "from": "+33111111111",
                "to": "+33222222222",
                "media_format": {
                    "encoding": "PCMU",
                    "sample_rate": 8000,
                    "channels": 1,
                },
            },
        }
    )

    class Gate(SynchronousUnauthenticatedGate):
        def __init__(self) -> None:
            super().__init__(1)
            self.acquire_count = 0
            self.release_count = 0

        def try_acquire(self) -> Any:
            self.acquire_count += 1
            return super().try_acquire()

        def _release(self) -> None:
            self.release_count += 1
            super()._release()

    claim_started = asyncio.Event()
    claim_release = asyncio.Event()

    class Authority:
        def __init__(self) -> None:
            self.claims: list[tuple[str, bytes]] = []
            self.aborts: list[tuple[str, bytes]] = []

        async def claim_once(
            self, *, call_control_id: str, token_digest: bytes
        ) -> object:
            self.claims.append((call_control_id, token_digest))
            claim_started.set()
            await claim_release.wait()
            return object()

        def schedule_abort_if_matches(
            self, *, call_control_id: str, token_digest: bytes
        ) -> None:
            self.aborts.append((call_control_id, token_digest))

    session_started = asyncio.Event()
    session_release = asyncio.Event()

    class SessionFactory:
        async def run(self, _handshake: object) -> None:
            session_started.set()
            await session_release.wait()

    profile = QualifiedDeploymentProfileV1.model_validate_json(
        _qualified_profile_json()
    ).model_copy(update={"telnyx_handshake_fixture_sha256": fixture_sha256})
    gate = Gate()
    authority = Authority()
    service = AuthenticatedTelnyxHandshakeService(
        profile=profile,
        lease_authority=authority,
        unauthenticated_gate=gate,
        timeout_seconds=5,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    settings = _settings(port=port, uvicorn_grace_seconds=1)
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    app = create_app(settings, coordinator)
    app.state.runtime_graph = SimpleNamespace(
        unauthenticated_gate=gate,
        handshake=service,
        session_factory=SessionFactory(),
        metrics=RuntimeMetrics.in_memory(),
    )

    @asynccontextmanager
    async def lifespan(_app: object) -> Any:
        yield

    app.router.lifespan_context = lifespan
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    accept_started = asyncio.Event()
    accept_release = asyncio.Event()
    if phase == "cancel-accept":
        original_accept = WebSocket.accept

        async def blocked_accept(
            websocket: WebSocket,
            subprotocol: str | None = None,
            headers: Any | None = None,
        ) -> None:
            accept_started.set()
            await accept_release.wait()
            await original_accept(websocket, subprotocol=subprotocol, headers=headers)

        monkeypatch.setattr(WebSocket, "accept", blocked_accept)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    held = gate.try_acquire() if phase == "denial" else None
    server_started = _observe_server_start(server, monkeypatch)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    await asyncio.wait_for(server_started.wait(), timeout=5.0)
    assert server.started is True
    url = f"ws://127.0.0.1:{port}/telnyx/media"
    headers = {"x-telnyx-streaming-auth-token": token}

    try:
        if phase == "denial":
            with pytest.raises(InvalidStatus) as denied:
                async with connect(url, additional_headers=headers):
                    raise AssertionError("capacity denial must not upgrade")
            assert denied.value.response.status_code == 503
            assert authority.claims == []
            assert gate.in_use == 1
            assert held is not None
            held.release()
            assert (gate.acquire_count, gate.release_count) == (2, 1)
        elif phase == "cancel-accept":
            connection = connect(url, additional_headers=headers)
            connection_task = asyncio.create_task(connection.__aenter__())
            await accept_started.wait()
            assert gate.in_use == 1
            assert authority.claims == []
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5.0)
            with contextlib.suppress(BaseException):
                await connection_task
            assert (gate.acquire_count, gate.release_count, gate.in_use) == (1, 1, 0)
        else:
            async with connect(url, additional_headers=headers) as websocket:
                await websocket.send(connected)
                await websocket.send(start)
                await claim_started.wait()
                assert gate.in_use == 1
                gate.close()
                if phase == "drain-success":
                    claim_release.set()
                    await session_started.wait()
                    assert gate.in_use == 0
                    session_release.set()
                else:
                    server.should_exit = True
                    await asyncio.wait_for(task, timeout=5.0)
            assert authority.claims == [(call_control_id, token_digest)]
            assert (gate.acquire_count, gate.release_count, gate.in_use) == (1, 1, 0)
            if phase == "cancel-transfer":
                assert authority.aborts == [(call_control_id, token_digest)]
        if not task.done():
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5.0)
    finally:
        accept_release.set()
        claim_release.set()
        session_release.set()
        if not task.done():
            server.should_exit = True
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, timeout=5.0)
        if held is not None:
            held.release()
        loop.set_task_factory(previous_factory)
        listener.close()


class _AsgiInertState:
    def __init__(self, boundary: str, side: str) -> None:
        self.boundary = boundary
        self.side = side
        self.boundary_reached = asyncio.Event()
        self.request_cancelled = asyncio.Event()
        self.target_cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.lifespan_shutdown_entered = asyncio.Event()
        self.lifespan_closed = asyncio.Event()
        self.server_started = asyncio.Event()
        self.timeout_logged = asyncio.Event()
        self.session_ready = asyncio.Event()
        self.activated = asyncio.Event()
        self.request_task: asyncio.Task[Any] | None = None
        self.target_task: asyncio.Task[Any] | None = None
        self.owner_task: asyncio.Task[None] | None = None
        self.cleanup_task: asyncio.Task[None] | None = None
        self.finalizer_task: asyncio.Task[None] | None = None
        self.target_result: object | None = None
        self.target_argument: object | None = None
        self.resolution: object | None = None
        self.resolver_result: object | None = None
        self.resolver_real_calls = 0
        self.claim_snapshot: object | None = None
        self.request_cancellation: asyncio.CancelledError | None = None
        self.target_cancellation: asyncio.CancelledError | None = None
        self.reraised_cancellation: asyncio.CancelledError | None = None
        self.target_real_calls = 0
        self.target_wrapper_calls = 0
        self.cleanup_real_calls = 0
        self.builder_calls = 0
        self.supervisor_aclose_calls = 0
        self.writer_close_calls = 0
        self.writer_drain_calls = 0
        self.writer_submit_calls = 0
        self.writer_method_calls: dict[str, int] = {}
        self.registry_close_calls = 0
        self.registry_join_calls = 0
        self.registry_terminal_calls = 0
        self.commit_lease_states: list[str] = []
        self.terminal_operations: list[object] = []
        self.timeout_logs: list[str] = []
        self.close_order: list[str] = []
        self.late_calls: list[str] = []
        self.writer_closed = False
        self.registry_closed = False
        self.meter_closed = False
        self.baseline_tasks = set(asyncio.all_tasks())


class _AsgiInertInstrumentGuard:
    def __init__(self, instrument: object, state: _AsgiInertState) -> None:
        self._instrument = instrument
        self._state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._instrument, name)

    def add(self, *args: object, **kwargs: object) -> Any:
        if self._state.meter_closed:
            self._state.late_calls.append("meter:add")
            raise AssertionError("meter_used_after_close")
        return self._instrument.add(*args, **kwargs)

    def record(self, *args: object, **kwargs: object) -> Any:
        if self._state.meter_closed:
            self._state.late_calls.append("meter:record")
            raise AssertionError("meter_used_after_close")
        return self._instrument.record(*args, **kwargs)


class _AsgiInertMetricReaderGuard:
    def __init__(self, reader: object, state: _AsgiInertState) -> None:
        self._reader = reader
        self._state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    def get_metrics_data(self, *args: object, **kwargs: object) -> Any:
        if self._state.meter_closed:
            self._state.late_calls.append("meter:collect")
            raise AssertionError("meter_collected_after_close")
        return self._reader.get_metrics_data(*args, **kwargs)


class _AsgiInertMeterProviderGuard:
    def __init__(self, provider: object, state: _AsgiInertState) -> None:
        self._provider = provider
        self._state = state
        self.shutdown_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def shutdown(self, *args: object, **kwargs: object) -> Any:
        if self.shutdown_calls != 0:
            self._state.late_calls.append("meter:shutdown")
            raise AssertionError("meter_shutdown_repeated")
        self.shutdown_calls += 1
        result = self._provider.shutdown(*args, **kwargs)
        self._state.meter_closed = True
        self._state.close_order.append("meter")
        return result


@dataclass(slots=True, repr=False)
class _AsgiInertRuntime:
    state: _AsgiInertState
    settings: RuntimeSettingsV1
    graph: Any
    app: Any
    coordinator: Any
    server: Any
    listener: socket.socket
    signing_key: Any
    token: str
    call_control_id: str
    stream_id: str
    raw_control: Any
    sink: Any
    meter_provider: _AsgiInertMeterProviderGuard
    server_task: asyncio.Task[None] | None = None

    def __repr__(self) -> str:
        return "AsgiInertRuntime()"


def _asgi_inert_capture_cancellation(
    state: _AsgiInertState,
    error: asyncio.CancelledError,
) -> None:
    if state.request_cancellation is None:
        state.request_cancellation = error
    state.request_cancelled.set()


async def _asgi_inert_reraise_after_release(
    state: _AsgiInertState,
    error: asyncio.CancelledError,
) -> None:
    while not state.release.is_set():
        try:
            await state.release.wait()
        except asyncio.CancelledError:
            continue
    state.reraised_cancellation = error
    raise error


def _install_asgi_inert_common_wrappers(
    runtime: _AsgiInertRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    import uvicorn.server as uvicorn_server

    from projetv0_voice.admission import CallRegistry
    from projetv0_voice.lifecycle import RuntimeSupervisor, _MeasuredCallControl
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.persistence.writer import PersistenceWriter

    state = runtime.state
    graph = runtime.graph
    writer = graph.writer
    registry = graph.registry
    supervisor = graph.supervisor
    metrics = graph.metrics

    provider_guard = _AsgiInertMeterProviderGuard(metrics._provider, state)
    runtime.meter_provider = provider_guard
    metrics._provider = provider_guard
    metrics._metric_reader = _AsgiInertMetricReaderGuard(
        metrics._metric_reader,
        state,
    )
    for name, instrument in tuple(vars(metrics).items()):
        if name in {"_provider", "_metric_reader"}:
            continue
        if hasattr(instrument, "add") or hasattr(instrument, "record"):
            setattr(metrics, name, _AsgiInertInstrumentGuard(instrument, state))

    writer_methods = (
        "queue_size",
        "try_enqueue_turn",
        "commit_control",
        "latch_control_commit_timeout",
        "submit_webhook",
        "qualification_run_consumed",
        "classify_webhook_receipt",
        "quick_check",
        "wait_ready",
        "wait_until_idle",
        "take_stale_leases",
        "commit_lease",
        "terminalize_stale_lease",
        "read_relay_batch",
        "oldest_outbox_created_at",
        "runtime_observation",
        "ack_outbox",
        "retry_outbox",
        "cleanup_local_state",
        "drain",
        "run",
    )

    def note_writer_call(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        state.writer_method_calls[name] = state.writer_method_calls.get(name, 0) + 1
        if name == "drain":
            state.writer_drain_calls += 1
        elif name == "submit_webhook":
            state.writer_submit_calls += 1
        elif name == "commit_lease":
            lease_state = kwargs.get("state")
            if isinstance(lease_state, str):
                state.commit_lease_states.append(lease_state)
        elif name == "commit_control":
            command = args[0] if args else kwargs.get("command")
            payload = getattr(command, "payload", None)
            operation = None if payload is None else payload.get("operation")
            if getattr(operation, "kind", None) == "call.upsert":
                state.terminal_operations.append(operation)

    def sync_writer_guard(name: str, original: Any) -> Any:
        def guarded(current: Any, *args: Any, **kwargs: Any) -> Any:
            if current is writer:
                if state.writer_closed:
                    state.late_calls.append(f"writer:{name}")
                    raise AssertionError("writer_used_after_close")
                note_writer_call(name, args, kwargs)
            return original(current, *args, **kwargs)

        return guarded

    def async_writer_guard(name: str, original: Any) -> Any:
        async def guarded(current: Any, *args: Any, **kwargs: Any) -> Any:
            if current is writer:
                if state.writer_closed:
                    state.late_calls.append(f"writer:{name}")
                    raise AssertionError("writer_used_after_close")
                note_writer_call(name, args, kwargs)
            return await original(current, *args, **kwargs)

        return guarded

    for name in writer_methods:
        original = getattr(PersistenceWriter, name)
        wrapper = (
            async_writer_guard(name, original)
            if inspect.iscoroutinefunction(original)
            else sync_writer_guard(name, original)
        )
        monkeypatch.setattr(PersistenceWriter, name, wrapper)

    registry_methods = (
        "resolve_webhook",
        "resolve_duplicate_webhook",
        "reconcile_after_commit",
        "confirm_late_after_fail_closed",
        "claim_once",
        "consume_claim_for_construction",
        "activate_session",
        "construction_is_live",
        "reserve_or_read_terminal",
        "complete_reserved_terminal",
        "begin_drain",
        "close_session_owner_registration",
        "close_registration",
        "snapshot",
        "live_call_count",
        "_schedule_abort_target",
        "_note_terminal_failure",
    )

    def note_registry_call(
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del kwargs
        if name == "close_registration":
            state.registry_close_calls += 1
        elif name == "complete_reserved_terminal":
            authority = args[0] if args else None
            entry = getattr(authority, "_entry", None)
            if getattr(entry, "call_control_id", None) == runtime.call_control_id:
                state.registry_terminal_calls += 1

    def sync_registry_guard(name: str, original: Any) -> Any:
        def guarded(current: Any, *args: Any, **kwargs: Any) -> Any:
            if current is registry:
                if state.registry_closed:
                    state.late_calls.append(f"registry:{name}")
                    raise AssertionError("registry_used_after_close")
                note_registry_call(name, args, kwargs)
            return original(current, *args, **kwargs)

        return guarded

    def async_registry_guard(name: str, original: Any) -> Any:
        async def guarded(current: Any, *args: Any, **kwargs: Any) -> Any:
            if current is registry:
                if state.registry_closed:
                    state.late_calls.append(f"registry:{name}")
                    raise AssertionError("registry_used_after_close")
                note_registry_call(name, args, kwargs)
            return await original(current, *args, **kwargs)

        return guarded

    for name in registry_methods:
        original = getattr(CallRegistry, name)
        wrapper = (
            async_registry_guard(name, original)
            if inspect.iscoroutinefunction(original)
            else sync_registry_guard(name, original)
        )
        monkeypatch.setattr(CallRegistry, name, wrapper)

    original_registry_join = CallRegistry.join_until_empty

    async def observed_registry_join(current: Any) -> None:
        if current is registry and state.registry_closed:
            state.late_calls.append("registry:join_until_empty")
            raise AssertionError("registry_joined_after_close")
        await original_registry_join(current)
        if current is registry:
            state.registry_join_calls += 1
            assert current._background_owner._closed is True
            assert not current._background_owner._tasks
            state.registry_closed = True
            state.close_order.append("registry")

    monkeypatch.setattr(CallRegistry, "join_until_empty", observed_registry_join)

    original_close_writer = RuntimeSupervisor._close_writer

    async def observed_close_writer(current: Any, deadline: float) -> None:
        await original_close_writer(current, deadline)
        if current is supervisor:
            state.writer_close_calls += 1
            assert current._writer_task is not None
            assert current._writer_task.done()
            state.writer_closed = True
            state.close_order.append("writer")

    monkeypatch.setattr(RuntimeSupervisor, "_close_writer", observed_close_writer)

    original_aclose = RuntimeSupervisor.aclose

    async def observed_aclose(current: Any) -> None:
        if current is supervisor:
            state.supervisor_aclose_calls += 1
            state.lifespan_shutdown_entered.set()
        await original_aclose(current)
        if current is supervisor:
            state.lifespan_closed.set()

    monkeypatch.setattr(RuntimeSupervisor, "aclose", observed_aclose)

    for method_name in ("answer", "start_streaming", "hangup"):
        original = getattr(_MeasuredCallControl, method_name)

        async def guarded_measured(
            current: Any,
            *args: Any,
            _name: str = method_name,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if current is graph.measured_call_control and runtime.raw_control.closed:
                state.late_calls.append(f"measured_control:{_name}")
                raise AssertionError("measured_control_used_after_close")
            return await _original(current, *args, **kwargs)

        monkeypatch.setattr(_MeasuredCallControl, method_name, guarded_measured)

    original_metrics_aclose = RuntimeMetrics.aclose

    async def observed_metrics_aclose(current: Any) -> None:
        await original_metrics_aclose(current)
        if current is metrics:
            assert provider_guard.shutdown_calls == 1

    monkeypatch.setattr(RuntimeMetrics, "aclose", observed_metrics_aclose)

    implementation = runtime.server._server
    original_startup = type(implementation).startup

    async def observed_server_startup(
        current: Any,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        await original_startup(current, sockets=sockets)
        if current is implementation:
            state.server_started.set()

    monkeypatch.setattr(type(implementation), "startup", observed_server_startup)

    original_log_error = uvicorn_server.logger.error

    def observed_log_error(message: object, *args: object, **kwargs: object) -> None:
        if message == "Cancel %s running task(s), timeout graceful shutdown exceeded":
            rendered = str(message) % args
            state.timeout_logs.append(rendered)
            state.timeout_logged.set()
        original_log_error(message, *args, **kwargs)

    monkeypatch.setattr(uvicorn_server.logger, "error", observed_log_error)


async def _start_asgi_inert_runtime(
    *,
    boundary: str,
    side: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _AsgiInertRuntime:
    import base64
    import importlib
    from datetime import UTC, datetime
    from uuid import UUID

    from nacl.signing import SigningKey
    from pipecat.processors.frame_processor import FrameProcessor

    from projetv0_voice.admission import (
        CallRegistry,
        ProcessLeaseAuthority,
        SynchronousUnauthenticatedGate,
    )
    from projetv0_voice.app import create_app
    from projetv0_voice.config import AgentManifestV1
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.lifecycle import RuntimeProductionGraph, RuntimeSupervisor
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.persistence.relay import OutboxRelay
    from projetv0_voice.persistence.writer import PersistenceWriter
    from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer
    from projetv0_voice.session_factory import ProcessSessionFactory
    from projetv0_voice.telnyx.call_control import CallControlResult
    from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshakeService
    from projetv0_voice.telnyx.recordings import PurgeBatchResult
    from projetv0_voice.telnyx.webhooks import (
        TelnyxWebhookProcessor,
        TelnyxWebhookVerifier,
    )

    state = _AsgiInertState(boundary, side)
    token = "00" + "A" * 41
    call_control_id = f"asgi-{boundary}-{side}"
    stream_id = f"stream-{boundary}-{side}"

    class Processor(FrameProcessor):
        pass

    class SttClient:
        async def aclose(self) -> None:
            return None

    class LlmClient:
        async def close(self) -> None:
            return None

    class Llm(Processor):
        def __init__(self) -> None:
            super().__init__()
            self._client = LlmClient()

    class Recording:
        async def start(self, _identity: object) -> object:
            raise AssertionError("recording_is_disabled")

        async def cleanup(self, *_args: object, **_kwargs: object) -> None:
            return None

    class Control:
        def __init__(self) -> None:
            self.closed = False
            self.close_calls = 0
            self.stream_tokens: dict[str, str] = {}

        def _open(self, operation: str) -> None:
            if self.closed:
                state.late_calls.append(f"control:{operation}")
                raise AssertionError("call_control_used_after_close")

        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            self._open("answer")
            return CallControlResult("accepted")

        async def start_streaming(
            self,
            control_id: str,
            request: object,
            **_kwargs: object,
        ) -> CallControlResult:
            self._open("start_streaming")
            stream_token = request.stream_auth_token.get_secret_value()
            self.stream_tokens[control_id] = stream_token
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            self._open("hangup")
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            if self.closed:
                state.late_calls.append("control:aclose")
                raise AssertionError("call_control_closed_twice")
            self.close_calls += 1
            self.closed = True
            state.close_order.append("control")

    class Sink:
        def __init__(self) -> None:
            self.opened = False
            self.closed = False
            self.open_calls = 0
            self.close_calls = 0

        def _open(self, operation: str) -> None:
            if self.closed:
                state.late_calls.append(f"sink:{operation}")
                raise AssertionError("sink_used_after_close")

        async def open(self) -> None:
            self._open("open")
            self.open_calls += 1
            self.opened = True

        async def close(self) -> None:
            if self.closed:
                state.late_calls.append("sink:close")
                raise AssertionError("sink_closed_twice")
            self.close_calls += 1
            self.closed = True
            state.close_order.append("sink")

        async def ingest(self, _operation: object) -> None:
            self._open("ingest")

    class TrackingGate(SynchronousUnauthenticatedGate):
        def __init__(self) -> None:
            super().__init__(1)
            self.acquire_calls = 0
            self.release_calls = 0

        def try_acquire(self) -> Any:
            self.acquire_calls += 1
            return super().try_acquire()

        def _release(self) -> None:
            self.release_calls += 1
            super()._release()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = int(listener.getsockname()[1])
    settings = _settings(port=port, uvicorn_grace_seconds=1)
    keyring = CryptoKeyring({1: b"k" * 32}, active_version=1)
    writer = PersistenceWriter(
        tmp_path / f"asgi-inert-{boundary}-{side}.sqlite3",
        keyring,
        control_commit_timeout_seconds=10.0,
    )
    metrics = RuntimeMetrics.in_memory()
    raw_control = Control()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=raw_control,
        metrics=metrics,
        loop_interval_seconds=0.05,
        shutdown_timeout_seconds=45.0,
    )
    registry = CallRegistry(
        writer=writer,
        call_control=supervisor.call_control_facade,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/media",
        retention_days=7,
        utcnow=lambda: datetime.now(UTC),
        monotonic=lambda: asyncio.get_running_loop().time(),
        token_factory=lambda _size: token,
        prefix_factory=lambda: 1,
        uuid_factory=iter(
            UUID(int=(4 << 76) | (0b10 << 62) | index)
            for index in range(1, 1000)
        ).__next__,
    )
    sink = Sink()
    relay = OutboxRelay(
        writer,
        sink,
        on_degraded=supervisor.begin_drain,
        drain=supervisor.begin_drain,
    )
    gate = TrackingGate()

    async def purge_once() -> PurgeBatchResult:
        return PurgeBatchResult(0, 0, 0, 0, 0, 0, 0)

    async def recording_after_commit(*_args: object) -> None:
        return None

    supervisor.bind_runtime_graph(
        relay=relay,
        registry=registry,
        sink=sink,
        unauthenticated_gate=gate,
        purge_once=purge_once,
        recording_after_commit=recording_after_commit,
    )
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        _qualified_profile_json()
    ).model_copy(
        update={
            "deployment_id": "deployment-a",
            "telnyx_handshake_fixture_sha256": (
                "298459e95308c464230b646d6bc00754d87d2c7ffe4e262b3a5b42a2e74dd7f2"
            ),
        }
    )
    manifest = AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "agent_id": "agent-a",
            "revision": "revision-a",
            "tenant_id": "tenant-a",
            "dids": ["+33123456789"],
            "language": "fr-FR",
            "prompt_path": "prompt.md",
            "prompt_revision": "prompt-a",
            "greeting": "Bonjour.",
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
    session_factory = ProcessSessionFactory(
        registry=registry,
        registrar=supervisor.call_lifecycle_owners,
        runtime_metrics=metrics,
        manifest=manifest,
        profile=profile,
        writer=writer,
        keyring=keyring,
        stt_http_client_factory=SttClient,
        stt_factory=lambda _client: Processor(),
        llm_factory=Llm,
        tts_factory=Processor,
        recording_factory=lambda _identity: Recording(),
        recording_call_control_identity=supervisor.call_control_facade,
        idle_timeout_seconds=30.0,
        cleanup_phase_timeout_seconds=2.0,
    )
    lease_authority = ProcessLeaseAuthority(registry)
    handshake = AuthenticatedTelnyxHandshakeService(
        profile=profile,
        lease_authority=lease_authority,
        unauthenticated_gate=gate,
        timeout_seconds=5.0,
    )
    signing_key = SigningKey.generate()
    verifier = TelnyxWebhookVerifier(
        public_key=base64.b64encode(bytes(signing_key.verify_key)).decode("ascii"),
        call_control_required_types=(
            "call.initiated",
            "call.answered",
            "call.hangup",
        ),
    )
    webhook_processor = TelnyxWebhookProcessor(
        verifier=verifier,
        resolver=registry.resolve_webhook,
        duplicate_resolver=registry.resolve_duplicate_webhook,
        finalizer_owner=supervisor,
    )
    graph = RuntimeProductionGraph(
        settings=settings,
        manifest=manifest,
        profile=profile,
        override=None,
        keyring=keyring,
        supervisor=supervisor,
        metrics=metrics,
        writer=writer,
        sink=sink,
        relay=relay,
        raw_call_control=raw_control,
        measured_call_control=supervisor.call_control_facade,
        registry=registry,
        lease_authority=lease_authority,
        unauthenticated_gate=gate,
        handshake=handshake,
        session_factory=session_factory,
        webhook_processor=webhook_processor,
        recording_factory=lambda _identity: Recording(),
        recording_call_control_identity=supervisor.call_control_facade,
    )

    async def build_runtime(received: RuntimeSettingsV1) -> RuntimeProductionGraph:
        state.builder_calls += 1
        assert received is settings
        return graph

    coordinator = FirstSignalDrainCoordinator(
        settings=settings,
        runtime_builder=build_runtime,
    )
    app = create_app(settings, coordinator)
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    runtime = _AsgiInertRuntime(
        state=state,
        settings=settings,
        graph=graph,
        app=app,
        coordinator=coordinator,
        server=server,
        listener=listener,
        signing_key=signing_key,
        token=token,
        call_control_id=call_control_id,
        stream_id=stream_id,
        raw_control=raw_control,
        sink=sink,
        meter_provider=_AsgiInertMeterProviderGuard(metrics._provider, state),
    )
    verification = importlib.import_module("telnyx.lib.webhook_verification")
    monkeypatch.setattr(verification.time, "time", lambda: 1_777_118_400.0)
    _install_asgi_inert_common_wrappers(runtime, monkeypatch)
    _install_asgi_inert_boundary(runtime, monkeypatch)
    runtime.server_task = asyncio.create_task(
        server.serve(sockets=[listener]),
        name=f"asgi-inert-server-{boundary}-{side}",
    )
    await asyncio.wait_for(state.server_started.wait(), timeout=10.0)
    assert server.started is True
    assert app.state.runtime_graph is graph
    return runtime


def _install_asgi_inert_boundary(
    runtime: _AsgiInertRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.websockets import WebSocket, WebSocketState

    from projetv0_voice.admission import (
        CallRegistry,
        ProcessLeaseAuthority,
        ProcessLeaseClaim,
    )
    from projetv0_voice.lifecycle import (
        RuntimeSupervisor,
        _OwnedTaskSet,
        _WebhookFinalizationHandle,
    )
    from projetv0_voice.session import CallSession
    from projetv0_voice.session_factory import (
        ProcessSessionFactory,
        _CallLifecycleOwner,
        _RegistryTerminalizer,
    )
    from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshakeService
    from projetv0_voice.telnyx.webhooks import TelnyxWebhookProcessor

    state = runtime.state
    graph = runtime.graph
    supervisor = graph.supervisor
    registry = graph.registry
    handshake = graph.handshake
    session_factory = graph.session_factory
    processor = graph.webhook_processor

    processor._resolver = registry.resolve_webhook
    processor._duplicate_resolver = registry.resolve_duplicate_webhook

    if state.boundary == "finalizer":
        original_process = TelnyxWebhookProcessor.process_observed

        async def observed_process(current: Any, *args: Any, **kwargs: Any) -> Any:
            if current is not processor:
                return await original_process(current, *args, **kwargs)
            try:
                return await original_process(current, *args, **kwargs)
            except asyncio.CancelledError as error:
                state.request_task = asyncio.current_task()
                _asgi_inert_capture_cancellation(state, error)
                await _asgi_inert_reraise_after_release(state, error)

        monkeypatch.setattr(
            TelnyxWebhookProcessor,
            "process_observed",
            observed_process,
        )

        original_resolver = processor._resolver

        async def observed_resolver(event: Any) -> Any:
            result = await original_resolver(event)
            state.resolver_real_calls += 1
            state.resolver_result = result
            if state.side == "pre":
                state.resolution = result
                state.request_task = asyncio.current_task()
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    reservation = getattr(result, "reservation", None)
                    if reservation is not None:
                        reservation.abandon_before_submit()
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    _asgi_inert_capture_cancellation(state, error)
                    await _asgi_inert_reraise_after_release(state, error)
            return result

        processor._resolver = observed_resolver

        original_start = RuntimeSupervisor.start_webhook_finalization

        def observed_start(
            current: Any,
            event: Any,
            resolution: Any,
            receipt: Any = "first",
        ) -> Any:
            if current is not supervisor:
                return original_start(current, event, resolution, receipt)
            state.target_wrapper_calls += 1
            result = original_start(current, event, resolution, receipt)
            state.target_real_calls += 1
            state.request_task = asyncio.current_task()
            state.target_argument = resolution
            state.target_result = result
            tasks = tuple(current.webhook_finalizers._tasks)
            assert len(tasks) == 1
            state.finalizer_task = tasks[0]
            return result

        monkeypatch.setattr(
            RuntimeSupervisor,
            "start_webhook_finalization",
            observed_start,
        )

        if state.side == "post":
            original_wait = _WebhookFinalizationHandle.wait

            async def paused_wait(current: Any) -> Any:
                if current is not state.target_result:
                    return await original_wait(current)
                state.target_task = asyncio.current_task()
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    await _asgi_inert_reraise_after_release(state, error)
                return await original_wait(current)

            monkeypatch.setattr(_WebhookFinalizationHandle, "wait", paused_wait)
        return

    if state.boundary == "permit":
        original_accept = WebSocket.accept
        if state.side == "pre":

            async def paused_accept(
                current: WebSocket,
                subprotocol: str | None = None,
                headers: Any | None = None,
            ) -> None:
                if current.scope.get("app") is not runtime.app:
                    await original_accept(
                        current,
                        subprotocol=subprotocol,
                        headers=headers,
                    )
                    return
                await original_accept(
                    current,
                    subprotocol=subprotocol,
                    headers=headers,
                )
                assert current.application_state is WebSocketState.CONNECTED
                assert current.client_state is WebSocketState.CONNECTED
                state.request_task = asyncio.current_task()
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    _asgi_inert_capture_cancellation(state, error)
                    await _asgi_inert_reraise_after_release(state, error)

            monkeypatch.setattr(WebSocket, "accept", paused_accept)

        original_transfer = AuthenticatedTelnyxHandshakeService.transfer_authentication

        def observed_transfer(
            current: Any,
            websocket: WebSocket,
            permit: Any,
        ) -> Any:
            if current is not handshake:
                return original_transfer(current, websocket, permit)
            state.target_wrapper_calls += 1
            result = original_transfer(current, websocket, permit)
            state.target_real_calls += 1
            state.target_argument = permit
            state.target_result = result
            assert id(permit) in current._active_permits[1]
            return result

        monkeypatch.setattr(
            AuthenticatedTelnyxHandshakeService,
            "transfer_authentication",
            observed_transfer,
        )

        if state.side == "post":
            original_authenticate_owned = (
                AuthenticatedTelnyxHandshakeService._authenticate_owned
            )

            async def paused_authenticate_owned(
                current: Any,
                websocket: WebSocket,
                *,
                finish_permit: Any,
            ) -> Any:
                if current is not handshake:
                    return await original_authenticate_owned(
                        current,
                        websocket,
                        finish_permit=finish_permit,
                    )
                state.request_task = asyncio.current_task()
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    _asgi_inert_capture_cancellation(state, error)
                    await _asgi_inert_reraise_after_release(state, error)
                return await original_authenticate_owned(
                    current,
                    websocket,
                    finish_permit=finish_permit,
                )

            monkeypatch.setattr(
                AuthenticatedTelnyxHandshakeService,
                "_authenticate_owned",
                paused_authenticate_owned,
            )
        return

    if state.boundary == "claim":
        original_claim = ProcessLeaseAuthority.claim_once

        async def paused_claim(
            current: Any,
            *,
            call_control_id: str,
            token_digest: bytes,
        ) -> Any:
            if current is not graph.lease_authority:
                return await original_claim(
                    current,
                    call_control_id=call_control_id,
                    token_digest=token_digest,
                )
            state.target_wrapper_calls += 1
            state.request_task = asyncio.current_task()
            state.target_argument = (call_control_id, token_digest)
            if state.side == "pre":
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    _asgi_inert_capture_cancellation(state, error)
                    await _asgi_inert_reraise_after_release(state, error)
                raise AssertionError("claim_pre_released_without_cancellation")
            result = await original_claim(
                current,
                call_control_id=call_control_id,
                token_digest=token_digest,
            )
            state.target_real_calls += 1
            assert isinstance(result, ProcessLeaseClaim)
            assert registry._by_control[call_control_id].claim is result
            state.target_result = result
            snapshot = await registry.snapshot(call_control_id)
            assert snapshot is not None
            assert snapshot.lease_state == "active"
            assert snapshot.raw_token_retained is False
            state.claim_snapshot = snapshot
            state.boundary_reached.set()
            try:
                await state.release.wait()
            except asyncio.CancelledError as error:
                state.target_cancellation = error
                state.target_cancelled.set()
                _asgi_inert_capture_cancellation(state, error)
                await _asgi_inert_reraise_after_release(state, error)
            return result

        monkeypatch.setattr(ProcessLeaseAuthority, "claim_once", paused_claim)
        return

    if state.boundary == "lifecycle":
        original_try_start = _OwnedTaskSet.try_start

        def observed_try_start(
            current: Any,
            coroutine: Any,
            *,
            name: str,
        ) -> Any:
            if current is not supervisor.call_lifecycle_owners:
                return original_try_start(current, coroutine, name=name)
            state.target_wrapper_calls += 1
            result = original_try_start(current, coroutine, name=name)
            state.target_real_calls += 1
            assert isinstance(result, asyncio.Task)
            assert result.get_name() == "voice-call-session-owner"
            state.owner_task = result
            state.target_result = result
            return result

        monkeypatch.setattr(_OwnedTaskSet, "try_start", observed_try_start)

        original_run = ProcessSessionFactory.run
        if state.side == "pre":

            async def paused_run(current: Any, authenticated: Any) -> None:
                if current is not session_factory:
                    await original_run(current, authenticated)
                    return
                state.request_task = asyncio.current_task()
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    _asgi_inert_capture_cancellation(state, error)
                    await _asgi_inert_reraise_after_release(state, error)
                raise AssertionError("lifecycle_pre_released_without_cancellation")

            monkeypatch.setattr(ProcessSessionFactory, "run", paused_run)
        else:

            async def observed_run(current: Any, authenticated: Any) -> None:
                if current is session_factory:
                    state.request_task = asyncio.current_task()
                await original_run(current, authenticated)

            monkeypatch.setattr(ProcessSessionFactory, "run", observed_run)
            original_consume = CallRegistry.consume_claim_for_construction

            async def paused_consume(current: Any, *args: Any, **kwargs: Any) -> Any:
                if current is not registry:
                    return await original_consume(current, *args, **kwargs)
                state.boundary_reached.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError as error:
                    state.target_cancellation = error
                    state.target_cancelled.set()
                    _asgi_inert_capture_cancellation(state, error)
                    await _asgi_inert_reraise_after_release(state, error)
                return await original_consume(current, *args, **kwargs)

            monkeypatch.setattr(
                CallRegistry,
                "consume_claim_for_construction",
                paused_consume,
            )
        return

    if state.boundary != "cleanup":
        raise AssertionError("unknown_asgi_inert_boundary")

    original_factory_run = ProcessSessionFactory.run

    async def observed_factory_run(current: Any, authenticated: Any) -> None:
        if current is session_factory:
            state.request_task = asyncio.current_task()
        try:
            await original_factory_run(current, authenticated)
        except asyncio.CancelledError as error:
            if current is session_factory and state.request_cancellation is None:
                state.request_cancellation = error
            raise

    monkeypatch.setattr(ProcessSessionFactory, "run", observed_factory_run)

    original_activate = CallRegistry.activate_session

    async def observed_activate(current: Any, *args: Any, **kwargs: Any) -> bool:
        result = await original_activate(current, *args, **kwargs)
        if current is registry and result:
            owner_task = args[2]
            assert isinstance(owner_task, asyncio.Task)
            state.owner_task = owner_task
            state.activated.set()
        return result

    monkeypatch.setattr(CallRegistry, "activate_session", observed_activate)

    original_replay = CallSession._replay_pending_drain

    async def observed_replay(current: Any) -> None:
        await original_replay(current)
        terminalizer = current._registry_terminalizer
        if terminalizer is not None and terminalizer._registry is registry:
            state.session_ready.set()

    monkeypatch.setattr(CallSession, "_replay_pending_drain", observed_replay)

    original_request_drain = _CallLifecycleOwner.request_drain

    def observed_request_drain(current: Any, cause: str) -> None:
        if current._factory is session_factory and cause == "external_cancel":
            request_task = asyncio.current_task()
            assert request_task is state.request_task
            state.request_cancelled.set()
        original_request_drain(current, cause)

    monkeypatch.setattr(_CallLifecycleOwner, "request_drain", observed_request_drain)

    original_transfer_cleanup = _RegistryTerminalizer.transfer_to_cleanup

    def observed_transfer_cleanup(current: Any, cleanup_task: asyncio.Task[None]) -> None:
        if current._registry is not registry:
            original_transfer_cleanup(current, cleanup_task)
            return
        state.target_wrapper_calls += 1
        result = original_transfer_cleanup(current, cleanup_task)
        state.target_real_calls += 1
        state.target_result = result
        state.target_argument = cleanup_task
        state.cleanup_task = cleanup_task
        assert current._capability._cleanup_task is cleanup_task
        assert current._capability.permits_current_task() is False
        return result

    monkeypatch.setattr(
        _RegistryTerminalizer,
        "transfer_to_cleanup",
        observed_transfer_cleanup,
    )

    if state.side == "pre":
        original_await_cleanup = CallSession._await_owned_cleanup

        async def paused_await_cleanup(
            current: Any,
            cleanup: Any,
            **kwargs: Any,
        ) -> Any:
            terminalizer = current._registry_terminalizer
            if terminalizer is None or terminalizer._registry is not registry:
                return await original_await_cleanup(current, cleanup, **kwargs)
            state.target_task = asyncio.current_task()
            state.boundary_reached.set()
            try:
                await state.release.wait()
            except asyncio.CancelledError as error:
                state.target_cancellation = error
                state.target_cancelled.set()
                while not state.release.is_set():
                    try:
                        await state.release.wait()
                    except asyncio.CancelledError:
                        continue
                await cleanup
                state.cleanup_real_calls += 1
                state.reraised_cancellation = error
                raise error
            await cleanup
            state.cleanup_real_calls += 1
            raise AssertionError("cleanup_pre_released_without_cancellation")

        monkeypatch.setattr(
            CallSession,
            "_await_owned_cleanup",
            paused_await_cleanup,
        )
    else:
        original_cleanup = CallSession._cleanup_owned_state

        async def paused_cleanup(current: Any, **kwargs: Any) -> None:
            terminalizer = current._registry_terminalizer
            if terminalizer is None or terminalizer._registry is not registry:
                await original_cleanup(current, **kwargs)
                return
            state.target_task = asyncio.current_task()
            assert state.cleanup_task is state.target_task
            assert terminalizer._capability.permits_current_task() is True
            state.boundary_reached.set()
            try:
                await state.release.wait()
            except asyncio.CancelledError as error:
                state.target_cancellation = error
                state.target_cancelled.set()
                await _asgi_inert_reraise_after_release(state, error)
            await original_cleanup(current, **kwargs)
            state.cleanup_real_calls += 1

        monkeypatch.setattr(CallSession, "_cleanup_owned_state", paused_cleanup)


def _asgi_inert_signed_event(
    runtime: _AsgiInertRuntime,
    event_type: str,
) -> tuple[bytes, dict[str, str]]:
    import base64
    from datetime import UTC, datetime

    timestamp = 1_777_118_400
    payload: dict[str, object] = {
        "call_control_id": runtime.call_control_id,
        "call_leg_id": f"leg-{runtime.state.boundary}-{runtime.state.side}",
        "call_session_id": f"session-{runtime.state.boundary}-{runtime.state.side}",
    }
    if event_type == "call.initiated":
        payload.update({"direction": "incoming", "state": "parked"})
    elif event_type == "call.answered":
        payload["state"] = "answered"
    body = json.dumps(
        {
            "data": {
                "id": (
                    f"event-{runtime.state.boundary}-{runtime.state.side}-{event_type}"
                ),
                "event_type": event_type,
                "occurred_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "payload": payload,
            }
        },
        separators=(",", ":"),
    ).encode()
    signature = base64.b64encode(
        runtime.signing_key.sign(f"{timestamp}|".encode() + body).signature
    ).decode("ascii")
    return body, {
        "Telnyx-Signature-Ed25519": signature,
        "Telnyx-Timestamp": str(timestamp),
        "Content-Type": "application/json",
    }


async def _asgi_inert_post_event(
    runtime: _AsgiInertRuntime,
    client: httpx.AsyncClient,
    event_type: str,
) -> httpx.Response:
    body, headers = _asgi_inert_signed_event(runtime, event_type)
    return await client.post(
        f"http://127.0.0.1:{runtime.settings.bind_port}/telnyx/events",
        content=body,
        headers=headers,
    )


async def _asgi_inert_seed_call(
    runtime: _AsgiInertRuntime,
    client: httpx.AsyncClient,
) -> None:
    for event_type in ("call.initiated", "call.answered"):
        response = await _asgi_inert_post_event(runtime, client, event_type)
        assert response.status_code == 200
        assert response.content == b""
    assert runtime.raw_control.stream_tokens == {
        runtime.call_control_id: runtime.token
    }


async def _asgi_inert_send_handshake(runtime: _AsgiInertRuntime, websocket: Any) -> None:
    await websocket.send(
        json.dumps(
            {
                "event": "connected",
                "version": "1.0.0",
                "connected": {
                    "x-telnyx-streaming-auth-token": runtime.token,
                },
            }
        )
    )
    await websocket.send(
        json.dumps(
            {
                "stream_id": runtime.stream_id,
                "event": "start",
                "sequence_number": "1",
                "start": {
                    "call_control_id": runtime.call_control_id,
                    "from": "+33111111111",
                    "to": "+33222222222",
                    "media_format": {
                        "encoding": "PCMU",
                        "sample_rate": 8000,
                        "channels": 1,
                    },
                },
            }
        )
    )


async def _assert_asgi_inert_boundary(runtime: _AsgiInertRuntime) -> None:
    state = runtime.state
    graph = runtime.graph
    expected_calls = 0 if state.side == "pre" else 1
    assert state.target_real_calls == expected_calls
    expected_wrapper_calls = 1 if state.boundary == "claim" else expected_calls
    assert state.target_wrapper_calls == expected_wrapper_calls

    if state.boundary == "finalizer":
        assert state.resolver_real_calls == 1
        if state.side == "pre":
            assert state.resolution is not None
            assert state.writer_submit_calls == 0
            assert not graph.supervisor.webhook_finalizers._tasks
        else:
            assert state.target_result is not None
            assert state.resolver_result is not None
            assert state.target_argument is state.resolver_result
            assert state.finalizer_task is not None
            assert state.finalizer_task.get_name() == "voice-webhook-finalizer"
        return

    if state.boundary == "permit":
        if state.side == "pre":
            assert not graph.handshake._active_permits[1]
            assert graph.unauthenticated_gate.in_use == 1
        else:
            operation = state.target_result
            assert operation is not None
            assert operation._permit is state.target_argument
            assert id(state.target_argument) in graph.handshake._active_permits[1]
            assert len(graph.handshake._active_permits[1]) == 1
        return

    if state.boundary == "claim":
        snapshot = await graph.registry.snapshot(runtime.call_control_id)
        assert snapshot is not None
        if state.side == "pre":
            assert snapshot.lease_state == "pending"
            assert state.commit_lease_states.count("active") == 0
        else:
            assert state.target_result is not None
            assert state.claim_snapshot == snapshot
            assert snapshot.lease_state == "active"
            assert snapshot.raw_token_retained is False
            assert state.commit_lease_states.count("active") == 1
        return

    if state.boundary == "lifecycle":
        if state.side == "pre":
            assert state.owner_task is None
            assert not graph.supervisor.call_lifecycle_owners._tasks
        else:
            assert state.owner_task is state.target_result
            assert state.owner_task is not None
            assert state.owner_task in graph.supervisor.call_lifecycle_owners._tasks
            assert state.owner_task.get_name() == "voice-call-session-owner"
        return

    assert state.boundary == "cleanup"
    if state.side == "pre":
        assert state.cleanup_task is None
        assert state.cleanup_real_calls == 0
    else:
        assert state.cleanup_task is state.target_argument
        assert state.cleanup_task is not None
        assert state.cleanup_task.get_name() == "call-cleanup"
        assert state.cleanup_real_calls == 0


def _assert_asgi_inert_closed(runtime: _AsgiInertRuntime) -> None:
    state = runtime.state
    graph = runtime.graph
    supervisor = graph.supervisor
    registry = graph.registry
    writer = graph.writer
    expected_target_calls = 0 if state.side == "pre" else 1

    assert state.target_real_calls == expected_target_calls
    assert state.builder_calls == 1
    assert state.supervisor_aclose_calls == 1
    assert state.writer_close_calls == 1
    assert state.writer_drain_calls == 1
    assert state.registry_close_calls == 1
    assert state.registry_join_calls == 1
    assert runtime.raw_control.close_calls == 1
    assert runtime.sink.open_calls == 1
    assert runtime.sink.close_calls == 1
    assert runtime.meter_provider.shutdown_calls == 1
    assert state.close_order == ["registry", "writer", "control", "sink", "meter"]
    assert state.writer_closed is True
    assert state.registry_closed is True
    assert state.meter_closed is True
    assert state.late_calls == []

    assert not supervisor.webhook_finalizers._tasks
    assert not supervisor.call_lifecycle_owners._tasks
    assert not supervisor.fixed_supervisors._tasks
    assert not supervisor._startup_phase_tasks
    assert not supervisor._deadline_tasks
    assert supervisor._writer_task is not None
    assert supervisor._writer_task.done()
    assert supervisor._closed is True
    assert not registry._background_owner._tasks
    assert registry._session_close_task is None or registry._session_close_task.done()
    assert registry._permits_used == 0
    assert registry._by_control == {}
    assert writer._accepting is False
    assert graph.unauthenticated_gate.in_use == 0
    assert graph.handshake._active_permits[1] == set()
    assert runtime.server._server.server_state.tasks == set()
    assert runtime.app.state.runtime_graph is None

    if state.boundary == "cleanup":
        assert state.cleanup_real_calls == 1
        assert state.registry_terminal_calls == 1
        assert len(state.terminal_operations) == 1
    if state.boundary == "claim" and state.side == "post":
        assert state.commit_lease_states.count("active") == 1


_ASGI_INERT_CASES = (
    pytest.param("finalizer", "pre", id="finalizer-transfer-pre"),
    pytest.param("finalizer", "post", id="finalizer-transfer-post"),
    pytest.param("permit", "pre", id="permit-transfer-pre"),
    pytest.param("permit", "post", id="permit-transfer-post"),
    pytest.param("claim", "pre", id="claim-commit-pre"),
    pytest.param("claim", "post", id="claim-commit-post"),
    pytest.param("lifecycle", "pre", id="lifecycle-registration-pre"),
    pytest.param("lifecycle", "post", id="lifecycle-registration-post"),
    pytest.param("cleanup", "pre", id="session-cleanup-transfer-pre"),
    pytest.param("cleanup", "post", id="session-cleanup-transfer-post"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("boundary", "side"), _ASGI_INERT_CASES)
async def test_asgi_inert_matrix_uses_real_runtime_owners_across_uvicorn_timeout(
    boundary: str,
    side: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from websockets.asyncio.client import connect

    runtime: _AsgiInertRuntime | None = None
    client: httpx.AsyncClient | None = None
    websocket: Any | None = None
    request_driver: asyncio.Task[Any] | None = None
    close_driver: asyncio.Task[Any] | None = None
    try:
        runtime = await _start_asgi_inert_runtime(
            boundary=boundary,
            side=side,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
        state = runtime.state
        client = httpx.AsyncClient(trust_env=False, timeout=15.0)
        if boundary == "finalizer":
            request_driver = asyncio.create_task(
                _asgi_inert_post_event(runtime, client, "call.initiated"),
                name=f"asgi-inert-http-{side}",
            )
        else:
            await _asgi_inert_seed_call(runtime, client)
            websocket = await connect(
                f"ws://127.0.0.1:{runtime.settings.bind_port}/telnyx/media",
                additional_headers={
                    "x-telnyx-streaming-auth-token": runtime.token,
                },
            )
            if boundary in {"claim", "lifecycle", "cleanup"}:
                await _asgi_inert_send_handshake(runtime, websocket)
            if boundary == "cleanup":
                await asyncio.wait_for(state.session_ready.wait(), timeout=10.0)
                close_driver = asyncio.create_task(
                    websocket.close(),
                    name=f"asgi-inert-client-close-{side}",
                )

        await asyncio.wait_for(state.boundary_reached.wait(), timeout=15.0)
        assert state.request_task is not None
        await _assert_asgi_inert_boundary(runtime)

        runtime.server.should_exit = True
        await asyncio.wait_for(
            asyncio.gather(
                state.timeout_logged.wait(),
                state.request_cancelled.wait(),
            ),
            timeout=10.0,
        )
        if boundary == "cleanup" and side == "pre":
            await asyncio.wait_for(state.target_cancelled.wait(), timeout=5.0)
        assert runtime.server.force_exit is False
        assert state.request_task.done() is False
        assert state.timeout_logs == [
            "Cancel 1 running task(s), timeout graceful shutdown exceeded"
        ]

        await asyncio.wait_for(
            state.lifespan_shutdown_entered.wait(),
            timeout=5.0,
        )
        release_after_full_close = (boundary, side) not in {
            ("claim", "post"),
            ("lifecycle", "post"),
            ("cleanup", "pre"),
            ("cleanup", "post"),
        }
        if release_after_full_close:
            await asyncio.wait_for(state.lifespan_closed.wait(), timeout=20.0)
        state.release.set()

        assert runtime.server_task is not None
        await asyncio.wait_for(runtime.server_task, timeout=55.0)
        await asyncio.gather(state.request_task, return_exceptions=True)
        if state.target_task is not None:
            await asyncio.gather(state.target_task, return_exceptions=True)
        if request_driver is not None:
            await asyncio.gather(request_driver, return_exceptions=True)
        if close_driver is not None:
            await asyncio.wait_for(
                asyncio.gather(close_driver, return_exceptions=True),
                timeout=5.0,
            )
        if websocket is not None:
            await websocket.close()
            await websocket.wait_closed()
        await client.aclose()
        client = None

        assert state.reraised_cancellation is state.request_cancellation or (
            boundary == "cleanup"
        )
        if boundary != "cleanup":
            assert state.request_cancellation is not None
            assert state.request_cancellation.args == (
                "Task cancelled, timeout graceful shutdown exceeded",
            )
        assert runtime.server.force_exit is False
        _assert_asgi_inert_closed(runtime)
        captured = capfd.readouterr()
        rendered = captured.out + captured.err + caplog.text
        assert runtime.token not in rendered
        assert runtime.call_control_id not in rendered
        assert runtime.stream_id not in rendered
        assert "voice.invalid" not in rendered

        current = asyncio.current_task()
        living_new_tasks = {
            task
            for task in asyncio.all_tasks()
            if task not in state.baseline_tasks and task is not current and not task.done()
        }
        assert living_new_tasks == set()
    finally:
        if runtime is not None:
            runtime.state.release.set()
        if close_driver is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(close_driver, timeout=5.0)
        if websocket is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(websocket.close(), timeout=5.0)
        if request_driver is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(request_driver, timeout=5.0)
        if client is not None:
            await client.aclose()
        if runtime is not None:
            server_task = runtime.server_task
            if server_task is not None and not server_task.done():
                runtime.server.should_exit = True
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(server_task, timeout=55.0)
            runtime.listener.close()


@pytest.mark.asyncio
async def test_real_uvicorn_graceful_timeout_resumes_only_dependency_inert_wss_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from websockets.asyncio.client import connect

    from projetv0_voice.admission import SynchronousUnauthenticatedGate
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

    session_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    dependencies_closed = asyncio.Event()
    release_survivor = asyncio.Event()
    resumed_after_close = asyncio.Event()

    class Handshake:
        def transfer_authentication(self, _websocket: object, permit: Any) -> Any:
            async def authenticate() -> object:
                try:
                    return object()
                finally:
                    permit.release()

            return authenticate()

    class SessionFactory:
        async def run(self, _handshake: object) -> None:
            session_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                request_cancelled.set()
                await release_survivor.wait()
            assert dependencies_closed.is_set()
            resumed_after_close.set()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    settings = _settings(port=port, uvicorn_grace_seconds=1)
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    app = create_app(settings, coordinator)
    gate = SynchronousUnauthenticatedGate(1)
    app.state.runtime_graph = SimpleNamespace(
        unauthenticated_gate=gate,
        handshake=Handshake(),
        session_factory=SessionFactory(),
        metrics=RuntimeMetrics.in_memory(),
    )

    @asynccontextmanager
    async def lifespan(_app: object) -> Any:
        try:
            yield
        finally:
            dependencies_closed.set()
            release_survivor.set()

    app.router.lifespan_context = lifespan
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    server_started = _observe_server_start(server, monkeypatch)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    await asyncio.wait_for(server_started.wait(), timeout=5.0)
    assert server.started is True

    try:
        async with connect(f"ws://127.0.0.1:{port}/telnyx/media"):
            await session_started.wait()
            assert gate.in_use == 0
            server.should_exit = True
            await request_cancelled.wait()
            await dependencies_closed.wait()
            await resumed_after_close.wait()
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        if not task.done():
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5.0)
        listener.close()

    assert server.force_exit is False


@pytest.mark.asyncio
async def test_ten_sessions_traverse_real_asgi_registry_and_lifecycle_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64
    from datetime import UTC, datetime
    from uuid import UUID

    from nacl.signing import SigningKey
    from pipecat.processors.frame_processor import FrameProcessor
    from websockets.asyncio.client import connect

    from projetv0_voice.admission import (
        CallRegistry,
        ProcessLeaseAuthority,
        SynchronousUnauthenticatedGate,
    )
    from projetv0_voice.app import create_app
    from projetv0_voice.config import AgentManifestV1
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.lifecycle import RuntimeProductionGraph, RuntimeSupervisor
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.persistence.relay import OutboxRelay
    from projetv0_voice.persistence.writer import PersistenceWriter
    from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer
    from projetv0_voice.session_factory import ProcessSessionFactory
    from projetv0_voice.telnyx.call_control import CallControlResult
    from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshakeService
    from projetv0_voice.telnyx.recordings import PurgeBatchResult
    from projetv0_voice.telnyx.webhooks import (
        TelnyxWebhookProcessor,
        TelnyxWebhookVerifier,
        VerifiedWebhook,
    )

    class Processor(FrameProcessor):
        pass

    class SttClient:
        async def aclose(self) -> None:
            return None

    class Llm(Processor):
        def __init__(self) -> None:
            super().__init__()
            self._client = SimpleNamespace(close=self._close_client)

        async def _close_client(self) -> None:
            return None

    class Recording:
        async def start(self, _identity: object) -> object:
            raise AssertionError("recording is disabled")

        async def cleanup(self, *_args: object, **_kwargs: object) -> None:
            return None

    tokens = iter(f"{index:02d}" + "A" * 41 for index in range(10))
    stream_tokens: dict[str, str] = {}

    class Control:
        def __init__(self) -> None:
            self.closed = False

        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            assert not self.closed
            return CallControlResult("accepted")

        async def start_streaming(
            self, call_control_id: str, request: object, **_kwargs: object
        ) -> CallControlResult:
            assert not self.closed
            stream_tokens[call_control_id] = request.stream_auth_token.get_secret_value()
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            assert not self.closed
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            self.closed = True

    class Sink:
        def __init__(self) -> None:
            self.closed = False

        async def open(self) -> None:
            assert not self.closed
            return None

        async def close(self) -> None:
            self.closed = True

        async def ingest(self, _operation: object) -> None:
            assert not self.closed
            return None

    activated = {f"control-{index}": asyncio.Event() for index in range(10)}
    lifecycle_tasks: dict[str, asyncio.Task[None]] = {}

    class TrackingRegistry(CallRegistry):
        async def activate_session(
            self,
            grant: object,
            owner: object,
            owner_task: asyncio.Task[None],
            session: object,
        ) -> bool:
            result = await super().activate_session(
                grant,  # type: ignore[arg-type]
                owner,
                owner_task,
                session,
            )
            if result:
                control_id = grant.telnyx_call_control_id
                lifecycle_tasks[control_id] = owner_task
                activated[control_id].set()
            return result

    keyring = CryptoKeyring({1: b"k" * 32}, active_version=1)
    writer = PersistenceWriter(
        tmp_path / "real-asgi.sqlite3",
        keyring,
        control_commit_timeout_seconds=10.0,
    )
    metrics = RuntimeMetrics.in_memory()
    raw_control = Control()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=raw_control,  # type: ignore[arg-type]
        metrics=metrics,
        shutdown_timeout_seconds=30.0,
    )
    runtime_started = asyncio.Event()
    real_startup = supervisor.startup

    async def observed_startup() -> None:
        await real_startup()
        runtime_started.set()

    supervisor.startup = observed_startup  # type: ignore[method-assign]
    registry = TrackingRegistry(
        writer=writer,
        call_control=supervisor.call_control_facade,  # type: ignore[arg-type]
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=10,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/media",
        retention_days=7,
        utcnow=lambda: datetime.now(UTC),
        monotonic=lambda: asyncio.get_running_loop().time(),
        token_factory=lambda _size: next(tokens),
        prefix_factory=lambda: 1,
        uuid_factory=iter(
            UUID(int=(4 << 76) | (0b10 << 62) | index)
            for index in range(1, 1000)
        ).__next__,
    )
    session_0_terminated = asyncio.Event()
    terminal_calls = 0
    real_complete_reserved_terminal = registry.complete_reserved_terminal

    async def observed_complete_reserved_terminal(authority: object) -> bool:
        nonlocal terminal_calls
        result = await real_complete_reserved_terminal(authority)
        entry = getattr(authority, "_entry", None)
        if getattr(entry, "call_control_id", None) == "control-0":
            terminal_calls += 1
            session_0_terminated.set()
        return result

    registry.complete_reserved_terminal = (  # type: ignore[method-assign]
        observed_complete_reserved_terminal
    )
    sink = Sink()
    relay = OutboxRelay(
        writer,
        sink,  # type: ignore[arg-type]
        on_degraded=supervisor.begin_drain,
        drain=supervisor.begin_drain,
    )
    gate = SynchronousUnauthenticatedGate(10)

    async def purge_once() -> PurgeBatchResult:
        return PurgeBatchResult(0, 0, 0, 0, 0, 0, 0)

    async def recording_after_commit(*_args: object) -> None:
        return None

    supervisor.bind_runtime_graph(
        relay=relay,
        registry=registry,
        sink=sink,  # type: ignore[arg-type]
        unauthenticated_gate=gate,
        purge_once=purge_once,
        recording_after_commit=recording_after_commit,  # type: ignore[arg-type]
    )
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        _qualified_profile_json()
    ).model_copy(
        update={
            "deployment_id": "deployment-a",
            "telnyx_handshake_fixture_sha256": (
                "298459e95308c464230b646d6bc00754d87d2c7ffe4e262b3a5b42a2e74dd7f2"
            )
        }
    )
    manifest = AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "agent_id": "agent-a",
            "revision": "revision-a",
            "tenant_id": "tenant-a",
            "dids": ["+33123456789"],
            "language": "fr-FR",
            "prompt_path": "prompt.md",
            "prompt_revision": "prompt-a",
            "greeting": "Bonjour.",
            "conversation_mode": "freeform",
            "max_concurrent_calls": 10,
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
    session_factory = ProcessSessionFactory(
        registry=registry,  # type: ignore[arg-type]
        registrar=supervisor.call_lifecycle_owners,
        runtime_metrics=metrics,
        manifest=manifest,
        profile=profile,
        writer=writer,
        keyring=keyring,
        stt_http_client_factory=SttClient,
        stt_factory=lambda _client: Processor(),
        llm_factory=Llm,
        tts_factory=Processor,
        recording_factory=lambda _identity: Recording(),
        idle_timeout_seconds=30.0,
        cleanup_phase_timeout_seconds=2.0,
    )
    handshake = AuthenticatedTelnyxHandshakeService(
        profile=profile,
        lease_authority=ProcessLeaseAuthority(registry),
        unauthenticated_gate=gate,
        timeout_seconds=5.0,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(10)
    port = listener.getsockname()[1]
    settings = _settings(port=port, uvicorn_grace_seconds=2)
    signing_key = SigningKey.generate()
    verifier = TelnyxWebhookVerifier(
        public_key=base64.b64encode(bytes(signing_key.verify_key)).decode("ascii"),
        call_control_required_types=(
            "call.initiated",
            "call.answered",
            "call.hangup",
        ),
    )
    webhook_processor = TelnyxWebhookProcessor(
        verifier=verifier,
        resolver=registry.resolve_webhook,
        duplicate_resolver=registry.resolve_duplicate_webhook,
        finalizer_owner=supervisor,
    )
    graph = RuntimeProductionGraph(
        settings=settings,
        manifest=manifest,
        profile=profile,
        override=None,
        keyring=keyring,
        supervisor=supervisor,
        metrics=metrics,
        writer=writer,
        sink=sink,  # type: ignore[arg-type]
        relay=relay,
        raw_call_control=raw_control,  # type: ignore[arg-type]
        measured_call_control=supervisor.call_control_facade,
        registry=registry,
        lease_authority=ProcessLeaseAuthority(registry),
        unauthenticated_gate=gate,
        handshake=handshake,
        session_factory=session_factory,
        webhook_processor=webhook_processor,
        recording_factory=lambda _identity: Recording(),
        recording_call_control_identity=supervisor.call_control_facade,
    )

    async def build_runtime(_settings: RuntimeSettingsV1) -> RuntimeProductionGraph:
        return graph

    coordinator = FirstSignalDrainCoordinator(
        settings=settings,
        runtime_builder=build_runtime,
    )
    app = create_app(settings, coordinator)
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    server_started = _observe_server_start(server, monkeypatch)
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    await asyncio.wait_for(runtime_started.wait(), timeout=10.0)
    await asyncio.wait_for(server_started.wait(), timeout=10.0)
    assert server.started is True

    def event(index: int, event_type: str) -> VerifiedWebhook:
        return VerifiedWebhook(
            event_id=f"{event_type}-{index}",
            event_type=event_type,
            occurred_at=datetime.now(UTC),
            call_control_id=f"control-{index}",
            call_leg_id=f"leg-{index}",
            call_session_id=f"session-{index}",
            recording_id=None,
            stream_id=None,
            client_state=None,
            recording_started_at=None,
            recording_ended_at=None,
            recording_channels=None,
            semantic_fingerprint_sha256=bytes(
                [index + 1 if event_type == "call.initiated" else index + 21]
            )
            * 32,
            direction="incoming" if event_type == "call.initiated" else None,
            call_state="parked" if event_type == "call.initiated" else "answered",
        )

    for index in range(10):
        for event_type in ("call.initiated", "call.answered"):
            received = event(index, event_type)
            resolution = await registry.resolve_webhook(received)
            effect = resolution.effect
            result = await writer.submit_webhook(
                receipt={
                    "event_id": received.event_id,
                    "event_type": received.event_type,
                    "call_control_id": received.call_control_id,
                    "occurred_at": received.occurred_at,
                    "received_at": received.occurred_at,
                    "semantic_fingerprint_sha256": (
                        received.semantic_fingerprint_sha256
                    ),
                },
                lease=None if effect is None else effect.lease,
                operation=None if effect is None else effect.operation,
            ).wait()
            assert (
                await registry.reconcile_after_commit(received, resolution, result)
            ).status_code == 200
    assert len(stream_tokens) == 10

    connections = []
    try:
        for index in range(10):
            control_id = f"control-{index}"
            token = stream_tokens[control_id]
            connection = await connect(
                f"ws://127.0.0.1:{port}/telnyx/media",
                additional_headers={"x-telnyx-streaming-auth-token": token},
            )
            connections.append(connection)
            await connection.send(
                json.dumps(
                    {
                        "event": "connected",
                        "version": "1.0.0",
                        "connected": {"x-telnyx-streaming-auth-token": token},
                    }
                )
            )
            await connection.send(
                json.dumps(
                    {
                        "stream_id": f"stream-{index}",
                        "event": "start",
                        "sequence_number": "1",
                        "start": {
                            "call_control_id": control_id,
                            "from": "+33111111111",
                            "to": "+33222222222",
                            "media_format": {
                                "encoding": "PCMU",
                                "sample_rate": 8000,
                                "channels": 1,
                            },
                        },
                    }
                )
            )
            await asyncio.wait_for(
                activated[control_id].wait(),
                timeout=30.0,
            )
        assert all(ready.is_set() for ready in activated.values())
        await connections[0].close()
        await connections[0].wait_closed()
        await asyncio.wait_for(session_0_terminated.wait(), timeout=30.0)
        await asyncio.gather(lifecycle_tasks["control-0"], return_exceptions=True)
        assert terminal_calls == 1
        assert await registry.live_call_count() == 9
        expected_live_owners = {
            lifecycle_tasks[f"control-{index}"] for index in range(1, 10)
        }
        captured_owners = set(supervisor.call_lifecycle_owners._tasks)
        assert captured_owners == expected_live_owners
        assert all(not task.done() for task in expected_live_owners)
        assert lifecycle_tasks["control-0"] not in captured_owners
        assert all(connection.close_code is None for connection in connections[1:])
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=35.0)
    finally:
        await asyncio.gather(
            *(connection.close() for connection in connections),
            return_exceptions=True,
        )
        if not server_task.done():
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=35.0)
        listener.close()

    assert server.force_exit is False
    assert gate.in_use == 0
    assert raw_control.closed is True
    assert sink.closed is True


@pytest.mark.asyncio
async def test_first_and_repeated_signals_begin_one_drain_without_force_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

    class Supervisor:
        def __init__(self) -> None:
            self.calls = 0
            self.draining = asyncio.Event()

        async def begin_drain(self) -> None:
            self.calls += 1
            self.draining.set()

    settings = _settings()
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    server = VoiceUvicornServer(
        _LoopbackApp(),
        settings=settings,
        coordinator=coordinator,
    )
    coordinator.bind_server(server, asyncio.get_running_loop())
    supervisor = Supervisor()
    coordinator.publish_runtime(supervisor)  # type: ignore[arg-type]
    coordinator.publish_startup_complete()

    server.handle_exit(signal.SIGINT, None)
    server.handle_exit(signal.SIGINT, None)
    await supervisor.draining.wait()

    assert supervisor.calls == 1
    assert coordinator.urgent is True
    assert server.should_exit is True
    assert server.force_exit is False
    await coordinator.aclose()

    exit_set = asyncio.Event()
    original_should_exit = VoiceUvicornServer.should_exit
    getter = original_should_exit.fget
    setter = original_should_exit.fset
    assert getter is not None
    assert setter is not None
    observed_server: VoiceUvicornServer | None = None

    def observe_should_exit(current: VoiceUvicornServer, value: bool) -> None:
        setter(current, value)
        if current is observed_server and value:
            exit_set.set()

    monkeypatch.setattr(
        VoiceUvicornServer,
        "should_exit",
        property(getter, observe_should_exit),
    )
    second = FirstSignalDrainCoordinator(settings=settings)
    second_server = VoiceUvicornServer(
        _LoopbackApp(),
        settings=settings,
        coordinator=second,
    )
    observed_server = second_server
    second.bind_server(second_server, asyncio.get_running_loop())
    second.publish_startup_failure()
    second_server.handle_exit(signal.SIGTERM, None)
    second_server.handle_exit(signal.SIGTERM, None)
    await asyncio.wait_for(exit_set.wait(), timeout=1.0)
    assert second_server.should_exit is True
    assert second_server.force_exit is False
    await second.aclose()


@pytest.mark.asyncio
async def test_coordinator_unpublishes_only_the_exact_closed_runtime() -> None:
    from projetv0_voice.lifecycle import RuntimeSupervisor
    from projetv0_voice.server import FirstSignalDrainCoordinator

    coordinator = FirstSignalDrainCoordinator()
    published = RuntimeSupervisor()
    different = RuntimeSupervisor()

    coordinator.publish_runtime(published)
    coordinator.unpublish_runtime(different)
    assert coordinator._runtime is published  # noqa: SLF001

    coordinator.unpublish_runtime(published)
    assert coordinator._runtime is None  # noqa: SLF001
    assert coordinator._startup_owner is None  # noqa: SLF001

    await coordinator.aclose()
    await published.aclose()
    await different.aclose()


async def _read_process_marker(
    process: asyncio.subprocess.Process,
    marker: str,
) -> list[str]:
    assert process.stdout is not None
    lines: list[str] = []
    while True:
        raw = await asyncio.wait_for(process.stdout.readline(), timeout=7.0)
        if not raw:
            raise AssertionError(f"process ended before {marker}: {lines!r}")
        line = raw.decode("utf-8", errors="strict").strip()
        lines.append(line)
        if line == marker or line.startswith(marker + ":"):
            return lines


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()


@pytest.mark.asyncio
async def test_programmatic_lifespan_closes_runtime_before_shutdown_complete(
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _signal_process_script("programmatic_order"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        lines = await _read_process_marker(process, "READY")
        lines.extend(await _read_process_marker(process, "ACLOSE_STARTED"))
        assert "SHUTDOWN_COMPLETE" not in lines

        assert process.stdin is not None
        process.stdin.write(b"continue\n")
        await process.stdin.drain()
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8.0)
        lines.extend(stdout.decode("utf-8", errors="strict").splitlines())

        assert process.returncode == 0
        assert stderr == b""
        assert lines.count("ACLOSE_STARTED") == 1
        assert lines.count("ACLOSE_FINISHED") == 1
        assert lines.count("UNPUBLISHED") == 1
        assert lines.count("SHUTDOWN_COMPLETE") == 1
        assert lines.index("ACLOSE_FINISHED") < lines.index("SHUTDOWN_COMPLETE")
        assert lines.index("ACLOSE_FINISHED") < lines.index("UNPUBLISHED")
        assert lines.index("UNPUBLISHED") < lines.index("SHUTDOWN_COMPLETE")
    finally:
        await _kill_process(process)


@pytest.mark.asyncio
async def test_programmatic_lifespan_hard_deadline_waits_for_global_seam(
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _signal_process_script("programmatic_hard_deadline"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        lines = await _read_process_marker(process, "READY")
        lines.extend(await _read_process_marker(process, "ACLOSE_STARTED"))
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8.0)
        lines.extend(stdout.decode("utf-8", errors="strict").splitlines())

        assert process.returncode == 72
        assert stderr == b""
        assert lines.count("GLOBAL_HARD:72") == 1
        assert "ACLOSE_FINISHED" not in lines
        assert "SHUTDOWN_COMPLETE" not in lines
        assert not any(line.startswith("RUNTIME_HARD:") for line in lines)
        assert not any(line.startswith("SUMMARY:") for line in lines)
    finally:
        await _kill_process(process)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "signals",
    [
        (signal.SIGTERM,),
        (signal.SIGINT, signal.SIGINT),
        (signal.SIGTERM, signal.SIGTERM),
        (signal.SIGTERM, signal.SIGINT),
        (signal.SIGINT, signal.SIGTERM),
    ],
)
async def test_signal_proc_repeated_signal_permutations_preserve_lifespan(
    signals: tuple[int, ...],
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _signal_process_script("postgates"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    lines = await _read_process_marker(process, "READY")
    os.kill(process.pid, signals[0])
    lines.extend(await _read_process_marker(process, "DRAIN"))
    for selected in signals[1:]:
        os.kill(process.pid, selected)
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8.0)
    lines.extend(stdout.decode("utf-8", errors="strict").splitlines())

    assert process.returncode == 0
    assert stderr == b""
    assert lines.count("ACLOSE_STARTED") == 1
    assert lines.count("ACLOSE_FINISHED") == 1
    assert lines.count("UNPUBLISHED") == 1
    summary = next(line for line in lines if line.startswith("SUMMARY:"))
    _label, raw_calls, transitions, draining, forced, urgent, shutdown = summary.split(
        ":"
    )
    assert (raw_calls, transitions, draining, forced, shutdown) == (
        "2",
        "1",
        "1",
        "0",
        "1",
    )
    assert urgent == str(int(len(signals) > 1 and signals[1] == signal.SIGINT))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        "before_runtime_publication",
        "writer_ready",
        "writer_quick_check",
        "qualification_status",
        "sink_open",
        "stale_recovery",
        "runtime_publication",
    ],
)
async def test_signal_proc_before_every_startup_await_prevents_gate_open(
    phase: str,
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _signal_process_script(phase),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    lines = await _read_process_marker(process, "PHASE")
    os.kill(process.pid, signal.SIGTERM)
    assert process.stdin is not None
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        process.stdin.write(b"continue\n")
        await process.stdin.drain()
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8.0)
    lines.extend(stdout.decode("utf-8", errors="strict").splitlines())

    assert process.returncode != 72
    assert "GATE:1" not in lines
    assert "GATE:0" in lines
    assert lines.count("ACLOSE_STARTED") == 1
    assert lines.count("ACLOSE_FINISHED") == 1
    assert b"force_exit" not in stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.asyncio
async def test_signal_proc_real_startup_failure_unwinds_without_hard_exit() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _signal_process_script("startup_failure"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    lines = await _read_process_marker(process, "STARTUP_FAILURE")
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8.0)
    lines.extend(stdout.decode("utf-8", errors="strict").splitlines())

    assert process.returncode == 0
    assert "GATE:0" in lines
    assert lines.count("ACLOSE_STARTED") == 1
    assert lines.count("ACLOSE_FINISHED") == 1
    assert not any(line.startswith(("GLOBAL_HARD:", "RUNTIME_HARD:")) for line in lines)
    assert b"private" not in stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.asyncio
async def test_signal_proc_only_global_deadline_invokes_hard_exit_72() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _signal_process_script("hard_deadline"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    lines = await _read_process_marker(process, "READY")
    os.kill(process.pid, signal.SIGTERM)
    lines.extend(await _read_process_marker(process, "DRAIN"))
    stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=8.0)
    lines.extend(stdout.decode("utf-8", errors="strict").splitlines())

    assert process.returncode == 72
    assert lines.count("GLOBAL_HARD:72") == 1
    assert lines.count("ACLOSE_STARTED") == 1
    assert "ACLOSE_FINISHED" not in lines
    assert not any(line.startswith("RUNTIME_HARD:") for line in lines)
    assert not any(line.startswith("SUMMARY:") for line in lines)


@pytest.mark.asyncio
async def test_signal_during_startup_cancels_the_published_lifespan_owner_before_gate_open(
) -> None:
    from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

    entered = asyncio.Event()
    release = asyncio.Event()
    gate_opened = False

    class Supervisor:
        async def begin_drain(self) -> None:
            raise AssertionError("startup owner must unwind instead of racing begin_drain")

    settings = _settings()
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    server = VoiceUvicornServer(
        _LoopbackApp(),
        settings=settings,
        coordinator=coordinator,
    )
    coordinator.bind_server(server, asyncio.get_running_loop())

    async def startup_owner() -> None:
        nonlocal gate_opened
        coordinator.publish_runtime(Supervisor())  # type: ignore[arg-type]
        entered.set()
        await release.wait()
        gate_opened = True
        coordinator.publish_startup_complete()

    owner = asyncio.create_task(startup_owner())
    await entered.wait()
    server.handle_exit(signal.SIGTERM, None)

    with pytest.raises(asyncio.CancelledError):
        await owner
    assert gate_opened is False
    assert server.should_exit is True
    assert server.force_exit is False
    await coordinator.aclose()
