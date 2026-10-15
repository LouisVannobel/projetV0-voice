from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from projetv0_voice.admission import SynchronousUnauthenticatedGate
from projetv0_voice.app import create_app
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.server import FirstSignalDrainCoordinator, VoiceUvicornServer
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshakeService
from tests.integration.test_server_signal_shutdown import _observe_server_start, _settings
from tests.integration.test_telnyx_handshake import (
    TOKEN,
    _connected,
    _LeaseAuthority,
    _profile,
    _setup_points,
    _start,
    _token_digest,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,expected",
    [
        ("wrong_route", ("asgi_other_websocket", "denied")),
        ("gate_full", ("gate", "refused")),
        ("missing_header", ("header", "absent")),
        ("duplicate_header", ("header", "duplicate")),
        ("mismatch", ("connected", "token_mismatch")),
        ("invalid_connected_no_start", ("connected", "invalid_event")),
        ("no_connected", ("connected", "timeout")),
        ("no_start", ("start", "timeout")),
        ("peer_disconnect", ("start", "peer_disconnect")),
        ("fixture", ("fixture", "rejected")),
        ("lease_refused", ("lease", "refused")),
        ("valid", ("authentication", "success")),
    ],
)
async def test_pinned_native_uvicorn_reports_actual_setup_outcome(
    phase: str, expected: tuple[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = RuntimeMetrics.in_memory()
    gate = SynchronousUnauthenticatedGate(1)
    authority = _LeaseAuthority(expected_digest=_token_digest(), expired=phase == "lease_refused")
    profile = _profile()
    if phase == "fixture":
        profile = profile.model_copy(update={"telnyx_handshake_fixture_sha256": "a" * 64})
    handshake = AuthenticatedTelnyxHandshakeService(
        profile=profile, lease_authority=authority, unauthenticated_gate=gate,
        timeout_seconds=0.2 if phase in {"no_connected", "no_start"} else 10,
        runtime_metrics=metrics,
    )
    session_started = asyncio.Event()
    session_release = asyncio.Event()

    class Sessions:
        async def run(self, authenticated: Any) -> None:
            assert authenticated.lease_claim is authority.claim
            session_started.set()
            await session_release.wait()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    settings = _settings(port=listener.getsockname()[1], uvicorn_grace_seconds=1)
    coordinator = FirstSignalDrainCoordinator(settings=settings)
    app = create_app(settings, coordinator)
    app.state.runtime_graph = SimpleNamespace(
        metrics=metrics, handshake=handshake, unauthenticated_gate=gate, session_factory=Sessions(),
    )

    @asynccontextmanager
    async def lifespan(_app: Any) -> Any:
        yield

    app.router.lifespan_context = lifespan
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    assert server.config.ws == "websockets-sansio"
    server_started = _observe_server_start(server, monkeypatch)
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    held = gate.try_acquire() if phase == "gate_full" else None
    try:
        await asyncio.wait_for(server_started.wait(), timeout=5)
        path = "/canary-secret-route" if phase == "wrong_route" else "/telnyx/media"
        url = f"ws://127.0.0.1:{settings.bind_port}{path}"
        headers = [] if phase == "missing_header" else [("x-telnyx-streaming-auth-token", TOKEN)]
        if phase == "duplicate_header":
            headers *= 2
        if phase in {"wrong_route", "gate_full"}:
            with pytest.raises(InvalidStatus) as denied:
                async with connect(url, additional_headers=headers):
                    raise AssertionError("denied connection must not upgrade")
            assert denied.value.response.status_code == (403 if phase == "wrong_route" else 503)
        else:
            async with connect(url, additional_headers=headers) as websocket:
                if phase == "invalid_connected_no_start":
                    await websocket.send(_connected(event="canary-secret-event"))
                elif phase == "mismatch":
                    await websocket.send(_connected("canary-secret-token"))
                elif phase not in {"no_connected", "missing_header", "duplicate_header"}:
                    await websocket.send(_connected())
                    if phase not in {"no_start", "peer_disconnect"}:
                        await websocket.send(_start())
                if phase == "valid":
                    await asyncio.wait_for(session_started.wait(), timeout=1)
                    assert gate.in_use == 0
                    await websocket.close()
                    session_release.set()
                elif phase == "peer_disconnect":
                    await websocket.close()
                else:
                    with pytest.raises(ConnectionClosed) as closed:
                        await asyncio.wait_for(websocket.recv(), timeout=1)
                    assert closed.value.rcvd is not None
                    assert closed.value.rcvd.code == (1013 if phase.startswith("no_") else 1008)
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=5)
        points = _setup_points(metrics)
        assert points[expected] == 1
        assert (("authentication", "success") in points) is (phase == "valid")
        if phase == "invalid_connected_no_start":
            assert ("start", "waiting") not in points
        if held is not None:
            held.release()
        assert gate.in_use == 0
        assert not any(secret in repr(points) for secret in (TOKEN, "canary-secret"))
        assert metrics.failure_code is None
    finally:
        session_release.set()
        if held is not None:
            held.release()
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=5)
        listener.close()
        metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001
