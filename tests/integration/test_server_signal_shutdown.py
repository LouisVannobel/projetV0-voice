from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import sys
import textwrap
from pathlib import PurePosixPath
from typing import Any

import httpx
import pytest

from projetv0_voice.runtime_config import RuntimeSettingsV1


def _signal_process_script(mode: str) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import socket
        import sys
        from pathlib import PurePosixPath

        from projetv0_voice.runtime_config import RuntimeSettingsV1
        from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer

        MODE = {mode!r}
        settings = RuntimeSettingsV1(
            runtime_mode="strict",
            deployment_id="voice-agent-a",
            runtime_contract_path=PurePosixPath("/srv/runtime.json"),
            agent_bundle_path=PurePosixPath("/srv/bundle"),
            qualified_profile_path=PurePosixPath("/srv/profile.json"),
            qualification_candidate_path=None,
            qualification_override_path=None,
            keyring_path=PurePosixPath("/run/secrets/aead_keyring_v1.json"),
            sqlite_path=PurePosixPath("/var/lib/voice.sqlite3"),
            runtime_contract_sha256="a" * 64,
            image_digest="ghcr.io/example/voice@sha256:" + "d" * 64,
            agent_bundle_sha256="b" * 64,
            inference_profile_sha256="c" * 64,
            qualification_run_id=None,
            benchmark_did_sha256=None,
            deployment_max_calls=1,
            handshake_timeout_seconds=5,
            call_idle_timeout_seconds=300,
            call_cleanup_phase_timeout_seconds=10,
            pre_drain_grace_seconds=1,
            uvicorn_grace_seconds=1,
            shutdown_grace_seconds=5,
            telnyx_api_key_file=PurePosixPath("/run/secrets/telnyx"),
            telnyx_webhook_public_key_file=PurePosixPath("/run/secrets/webhook"),
            openrouter_api_key_file=PurePosixPath("/run/secrets/openrouter"),
            postgres_dsn_file=PurePosixPath("/run/secrets/postgres"),
            telnyx_media_wss_url="wss://voice.invalid/telnyx/media",
            otlp_http_endpoint="https://collector.invalid/v1/metrics",
            bind_host="127.0.0.1",
            bind_port=18080,
        )

        class Supervisor:
            def __init__(self):
                self.calls = 0

            async def begin_drain(self):
                self.calls += 1
                print("DRAIN", flush=True)

        async def input_line():
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await asyncio.get_running_loop().connect_read_pipe(
                lambda: protocol, sys.stdin
            )
            await reader.readline()

        supervisor = Supervisor()
        coordinator = FirstSignalDrainCoordinator(settings=settings)
        shutdown_seen = False
        gate_opened = False

        class App:
            async def __call__(self, scope, receive, send):
                global shutdown_seen, gate_opened
                assert scope["type"] == "lifespan"
                assert (await receive())["type"] == "lifespan.startup"
                try:
                    if MODE == "before_runtime_publication":
                        print("PHASE", flush=True)
                        await input_line()
                    coordinator.publish_runtime(supervisor)
                    if MODE not in {{"postgates", "before_runtime_publication"}}:
                        print("PHASE", flush=True)
                        await input_line()
                    gate_opened = True
                    coordinator.publish_startup_complete()
                    await send({{"type": "lifespan.startup.complete"}})
                    assert (await receive())["type"] == "lifespan.shutdown"
                    shutdown_seen = True
                    await send({{"type": "lifespan.shutdown.complete"}})
                except asyncio.CancelledError:
                    print("CANCELLED", flush=True)
                    raise
                finally:
                    print("GATE:" + str(int(gate_opened)), flush=True)

        async def run():
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            server = VoiceUvicornServer(
                App(), settings=settings, coordinator=coordinator
            )
            task = asyncio.create_task(server.serve(sockets=[listener]))
            if MODE == "postgates":
                while not server.started:
                    await asyncio.sleep(0)
                print("READY", flush=True)
            try:
                await task
            finally:
                listener.close()
            print(
                "SUMMARY:"
                + ":".join(
                    (
                        str(supervisor.calls),
                        str(int(server.force_exit)),
                        str(int(coordinator.urgent)),
                        str(int(shutdown_seen)),
                    )
                ),
                flush=True,
            )

        asyncio.run(run())
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
    task = asyncio.create_task(server.serve(sockets=[listener]))
    await app.startup.wait()
    for _ in range(10_000):
        if server.started:
            break
        await asyncio.sleep(0)
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
async def test_real_sansio_loopback_rejects_websocket_byte_524289() -> None:
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
    task = asyncio.create_task(server.serve(sockets=[listener]))
    await app.startup.wait()
    for _ in range(10_000):
        if server.started:
            break
        await asyncio.sleep(0)
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
async def test_real_uvicorn_graceful_timeout_resumes_only_dependency_inert_wss_task() -> None:
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
    task = asyncio.create_task(server.serve(sockets=[listener]))
    for _ in range(10_000):
        if server.started:
            break
        await asyncio.sleep(0)
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
async def test_first_and_repeated_signals_begin_one_drain_without_force_exit() -> None:
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

    second = FirstSignalDrainCoordinator(settings=settings)
    second_server = VoiceUvicornServer(
        _LoopbackApp(),
        settings=settings,
        coordinator=second,
    )
    second.bind_server(second_server, asyncio.get_running_loop())
    second.publish_startup_failure()
    second_server.handle_exit(signal.SIGTERM, None)
    second_server.handle_exit(signal.SIGTERM, None)
    for _ in range(10_000):
        if second_server.should_exit:
            break
        await asyncio.sleep(0)
    assert second_server.should_exit is True
    assert second_server.force_exit is False
    await second.aclose()


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
        if line == marker:
            return lines


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
    summary = next(line for line in lines if line.startswith("SUMMARY:"))
    _label, calls, forced, urgent, shutdown = summary.split(":")
    assert (calls, forced, shutdown) == ("1", "0", "1")
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
    assert b"force_exit" not in stderr


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
