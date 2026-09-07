from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimePublication, RuntimePublishedSnapshot
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.runtime_config import RuntimeSettingsV1
from projetv0_voice.telnyx.webhooks import WebhookDisposition

NOW = datetime(2026, 9, 1, 9, tzinfo=UTC)
KEY = bytes(range(32))


def _initial_snapshot() -> RuntimePublishedSnapshot:
    return RuntimePublishedSnapshot(
        generation=0,
        ready=False,
        draining=False,
        writer_queue_depth=0,
        writer_queue_oldest_age=0.0,
        writer_quick_check=False,
        outbox_depth=0,
        outbox_oldest_age=0.0,
        outbox_bytes=0,
        storage_bytes=0,
        startup_profile_and_stale_recovery_complete=False,
        admission_open=False,
        qualification_state_valid=False,
        writer_owner_alive_and_ready=False,
        no_writer_fatal_or_degradation=False,
        relay_supervisor_alive=False,
        no_permanent_relay_or_sink_degradation=False,
    )


def _operation() -> VoiceOperationV1:
    return VoiceOperationV1(
        schema_version=1,
        operation_id=UUID("11111111-1111-4111-8111-111111111111"),
        deployment_id="deployment-a",
        call_id=UUID("22222222-2222-4222-8222-222222222222"),
        occurred_at=NOW,
        kind="call.upsert",
        payload=CallUpsertPayloadV1(
            telnyx_call_control_id="control-a",
            telnyx_call_leg_id=None,
            telnyx_call_session_id=None,
            status="pending",
            disclosure_state="pending",
            started_at=None,
            ended_at=None,
            end_reason=None,
            retention_until=NOW + timedelta(days=7),
        ),
    )


@pytest.mark.asyncio
async def test_writer_runtime_observation_uses_one_owner_queued_outbox_aggregate(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voice.sqlite"

    def file_size(path: Path) -> int:
        return 100 if path == database else 20

    writer = PersistenceWriter(
        database,
        CryptoKeyring({1: KEY}, active_version=1),
        monotonic=lambda: 10.0,
        utcnow=lambda: NOW,
        file_size=file_size,
    )
    writer_task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    await writer.commit_control(
        PersistenceCommand("outbox", {"operation": _operation()}, None)
    )

    observed = await writer.runtime_observation()

    assert observed.writer_queue_depth == 0
    assert observed.writer_queue_oldest_age == 0.0
    assert observed.writer_quick_check is True
    assert observed.outbox_depth == 1
    assert observed.outbox_oldest_age == 0.0
    assert observed.outbox_bytes > 0
    assert observed.storage_bytes == 120

    await writer.drain(timeout_seconds=2.0)
    await writer_task
    with sqlite3.connect(database) as connection:
        expected_bytes = connection.execute(
            "SELECT COALESCE(SUM(length(nonce) + length(ciphertext)), 0) FROM outbox"
        ).fetchone()[0]
    assert observed.outbox_bytes == expected_bytes


@pytest.mark.parametrize(
    "change",
    [
        {"startup_profile_and_stale_recovery_complete": False},
        {"draining": True},
        {"admission_open": False},
        {"qualification_state_valid": False},
        {"writer_owner_alive_and_ready": False},
        {"writer_quick_check": False},
        {"no_writer_fatal_or_degradation": False},
        {"storage_bytes": 268_435_457},
        {"writer_queue_oldest_age": 1.000_001},
        {"relay_supervisor_alive": False},
        {"no_permanent_relay_or_sink_degradation": False},
        {"outbox_oldest_age": 900.000_001},
    ],
)
def test_readiness_truth_table_closes_each_required_predicate(
    change: dict[str, object],
) -> None:
    from projetv0_voice.lifecycle import publish_runtime_readiness
    from projetv0_voice.persistence.writer import WriterRuntimeObservation

    publication = RuntimePublication(_initial_snapshot())
    writer = WriterRuntimeObservation(
        writer_queue_depth=0,
        writer_queue_oldest_age=0.0,
        writer_quick_check=True,
        outbox_depth=0,
        outbox_oldest_age=0.0,
        outbox_bytes=0,
        storage_bytes=268_435_456,
    )
    values: dict[str, object] = {
        "startup_profile_and_stale_recovery_complete": True,
        "draining": False,
        "admission_open": True,
        "qualification_state_valid": True,
        "writer_owner_alive_and_ready": True,
        "no_writer_fatal_or_degradation": True,
        "relay_supervisor_alive": True,
        "no_permanent_relay_or_sink_degradation": True,
    }
    for name in tuple(change):
        if hasattr(writer, name):
            writer = replace(writer, **{name: change.pop(name)})
    values.update(change)

    published = publish_runtime_readiness(
        publication,
        writer=writer,
        **values,
    )

    assert published.ready is False
    assert publication.snapshot() is published


def test_readiness_healthy_boundaries_publish_one_complete_ready_generation() -> None:
    from projetv0_voice.lifecycle import publish_runtime_readiness
    from projetv0_voice.persistence.writer import WriterRuntimeObservation

    publication = RuntimePublication(_initial_snapshot())
    published = publish_runtime_readiness(
        publication,
        writer=WriterRuntimeObservation(
            writer_queue_depth=256,
            writer_queue_oldest_age=1.0,
            writer_quick_check=True,
            outbox_depth=1,
            outbox_oldest_age=900.0,
            outbox_bytes=12,
            storage_bytes=268_435_456,
        ),
        startup_profile_and_stale_recovery_complete=True,
        draining=False,
        admission_open=True,
        qualification_state_valid=True,
        writer_owner_alive_and_ready=True,
        no_writer_fatal_or_degradation=True,
        relay_supervisor_alive=True,
        no_permanent_relay_or_sink_degradation=True,
    )

    assert published.ready is True
    assert published.generation == 1
    assert publication.snapshot() == published


def _runtime_settings() -> RuntimeSettingsV1:
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
        pre_drain_grace_seconds=15,
        uvicorn_grace_seconds=20,
        shutdown_grace_seconds=30,
        telnyx_api_key_file=PurePosixPath("/run/secrets/telnyx-api-key"),
        telnyx_webhook_public_key_file=PurePosixPath(
            "/run/secrets/telnyx-webhook-key"
        ),
        openrouter_api_key_file=PurePosixPath("/run/secrets/openrouter-api-key"),
        postgres_dsn_file=PurePosixPath("/run/secrets/postgres-dsn"),
        telnyx_media_wss_url="wss://voice.invalid/telnyx/media",
        otlp_http_endpoint="https://collector.invalid/v1/metrics",
        bind_host="127.0.0.1",
        bind_port=8080,
    )


async def _raw_http(
    app: Any,
    *,
    path: str,
    messages: list[dict[str, Any]] | None = None,
    headers: list[tuple[bytes, bytes]] | None = None,
    method: str = "GET",
) -> list[dict[str, Any]]:
    queued = list(
        messages
        or [
            {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if queued:
            return queued.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.5"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": list(headers or []),
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8080),
        },
        receive,
        send,
    )
    return sent


async def _raw_websocket(
    app: Any,
    *,
    send_hook: Any | None = None,
) -> list[dict[str, Any]]:
    queued = [{"type": "websocket.connect"}]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if queued:
            return queued.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if send_hook is not None:
            send_hook(message)

    await app(
        {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.5"},
            "scheme": "ws",
            "path": "/telnyx/media",
            "raw_path": b"/telnyx/media",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8080),
            "subprotocols": [],
            "extensions": {"websocket.http.response": {}},
        },
        receive,
        send,
    )
    return sent


class _ReadySupervisor:
    def __init__(self, *, ready: bool) -> None:
        self._ready = ready

    def readiness_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(ready=self._ready)


class _RecordingProcessor:
    def __init__(self) -> None:
        self.received: list[tuple[bytes, list[tuple[str, str]]]] = []

    async def process_observed(
        self, *, body: bytes, headers: list[tuple[str, str]]
    ) -> object:
        from projetv0_voice.telnyx.webhooks import ObservedWebhookResult

        self.received.append((body, headers))
        return ObservedWebhookResult(
            disposition=WebhookDisposition(200),
            webhook_class="invalid",
            receipt="none",
            metric_disposition="ok",
        )


def _test_app(processor: object | None = None) -> tuple[Any, _RecordingProcessor]:
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics

    selected = processor if processor is not None else _RecordingProcessor()
    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        supervisor=_ReadySupervisor(ready=True),
        webhook_processor=selected,
        metrics=RuntimeMetrics.in_memory(),
    )
    assert isinstance(selected, _RecordingProcessor)
    return app, selected


@pytest.mark.asyncio
async def test_exact_four_parent_routes_disable_docs_openapi_and_redirects() -> None:
    app, _ = _test_app()

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert app.router.redirect_slashes is False
    assert {(route.path, type(route).__name__) for route in app.routes} == {
        ("/health/live", "APIRoute"),
        ("/health/ready", "APIRoute"),
        ("/telnyx/events", "APIRoute"),
        ("/telnyx/media", "APIWebSocketRoute"),
    }

    for path in (
        "/health/live/",
        "/health/ready/",
        "/telnyx/events/",
        "/docs",
        "/redoc",
        "/openapi.json",
    ):
        messages = await _raw_http(app, path=path)
        start = next(message for message in messages if message["type"] == "http.response.start")
        assert start["status"] == 404
        assert start["status"] not in {301, 302, 307, 308}


@pytest.mark.asyncio
@pytest.mark.parametrize("body_size, expected_status", [(65_536, 200), (65_537, 413)])
async def test_fragmented_webhook_body_bound_preserves_ordered_duplicate_headers(
    body_size: int,
    expected_status: int,
) -> None:
    app, processor = _test_app()
    body = bytes(index % 251 for index in range(body_size))
    headers = [
        (b"telnyx-signature-ed25519", b"first"),
        (b"x-ordered", b"one"),
        (b"telnyx-signature-ed25519", b"second"),
        (b"x-ordered", b"two"),
    ]
    messages = [
        {"type": "http.request", "body": body[:17], "more_body": True},
        {"type": "http.request", "body": body[17:65_530], "more_body": True},
        {"type": "http.request", "body": body[65_530:], "more_body": False},
    ]

    sent = await _raw_http(
        app,
        path="/telnyx/events",
        method="POST",
        messages=messages,
        headers=headers,
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    bodies = [message["body"] for message in sent if message["type"] == "http.response.body"]
    assert start["status"] == expected_status
    assert b"".join(bodies) == b""
    if expected_status == 200:
        assert processor.received == [
            (
                body,
                [
                    ("telnyx-signature-ed25519", "first"),
                    ("x-ordered", "one"),
                    ("telnyx-signature-ed25519", "second"),
                    ("x-ordered", "two"),
                ],
            )
        ]
    else:
        assert processor.received == []


@pytest.mark.asyncio
async def test_websocket_route_denies_before_accept_or_transfers_exact_held_permit() -> None:
    from projetv0_voice.admission import SynchronousUnauthenticatedGate
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics

    class CountingGate:
        def __init__(self) -> None:
            self.real = SynchronousUnauthenticatedGate(1)
            self.acquire_count = 0

        @property
        def in_use(self) -> int:
            return self.real.in_use

        def try_acquire(self) -> Any:
            self.acquire_count += 1
            return self.real.try_acquire()

        def close(self) -> None:
            self.real.close()

    class Handshake:
        def __init__(self, gate: CountingGate) -> None:
            self.gate = gate
            self.transferred: list[object] = []

        def transfer_authentication(self, websocket: Any, permit: Any) -> Any:
            assert websocket.application_state.name == "CONNECTED"
            assert self.gate.in_use == 1
            self.transferred.append(permit)

            async def authenticate() -> object:
                try:
                    return object()
                finally:
                    permit.release()

            return authenticate()

    class SessionFactory:
        def __init__(self) -> None:
            self.handshakes: list[object] = []

        async def run(self, handshake: object) -> None:
            self.handshakes.append(handshake)

    gate = CountingGate()
    held = gate.try_acquire()
    assert held is not None
    handshake = Handshake(gate)
    sessions = SessionFactory()
    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        unauthenticated_gate=gate,
        handshake=handshake,
        session_factory=sessions,
        metrics=RuntimeMetrics.in_memory(),
    )

    denied = await _raw_websocket(app)
    assert [message["type"] for message in denied] == [
        "websocket.http.response.start",
        "websocket.http.response.body",
    ]
    assert denied[0]["status"] == 503
    assert denied[1]["body"] == b""
    assert handshake.transferred == []
    held.release()

    accepted = await _raw_websocket(
        app,
        send_hook=lambda message: (
            gate.close() if message["type"] == "websocket.accept" else None
        ),
    )

    assert [message["type"] for message in accepted] == ["websocket.accept"]
    assert gate.acquire_count == 3
    assert len(handshake.transferred) == 1
    assert len(sessions.handshakes) == 1
    assert gate.in_use == 0


@pytest.mark.asyncio
async def test_websocket_accept_cancellation_releases_route_owned_permit() -> None:
    from projetv0_voice.admission import SynchronousUnauthenticatedGate
    from projetv0_voice.app import create_app

    gate = SynchronousUnauthenticatedGate(1)
    receive_started = asyncio.Event()
    release_receive = asyncio.Event()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        receive_started.set()
        await release_receive.wait()
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        unauthenticated_gate=gate,
        handshake=object(),
        session_factory=object(),
    )
    running = asyncio.create_task(
        app(
            {
                "type": "websocket",
                "asgi": {"version": "3.0", "spec_version": "2.5"},
                "scheme": "ws",
                "path": "/telnyx/media",
                "raw_path": b"/telnyx/media",
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8080),
                "subprotocols": [],
                "extensions": {"websocket.http.response": {}},
            },
            receive,
            send,
        )
    )
    await receive_started.wait()
    assert gate.in_use == 1

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert gate.in_use == 0
    assert sent == []


@pytest.mark.asyncio
async def test_websocket_synchronous_transfer_failure_releases_before_fixed_close() -> None:
    from projetv0_voice.admission import SynchronousUnauthenticatedGate
    from projetv0_voice.app import create_app

    class Handshake:
        def transfer_authentication(self, _websocket: object, _permit: object) -> object:
            raise RuntimeError("private-transfer-sentinel")

    gate = SynchronousUnauthenticatedGate(1)
    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        unauthenticated_gate=gate,
        handshake=Handshake(),
        session_factory=object(),
    )

    sent = await _raw_websocket(app)

    assert [message["type"] for message in sent] == [
        "websocket.accept",
        "websocket.close",
    ]
    assert sent[-1] == {
        "type": "websocket.close",
        "code": 1011,
        "reason": "runtime_failed",
    }
    assert gate.in_use == 0


@pytest.mark.asyncio
async def test_new_capacity_rejection_emits_one_exact_ingress_metric_outcome() -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.telnyx.webhooks import (
        TelnyxWebhookProcessor,
        VerifiedWebhook,
    )

    event = VerifiedWebhook(
        event_id="event-a",
        event_type="call.initiated",
        occurred_at=NOW,
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"f" * 32,
        direction="incoming",
        call_state="parked",
    )

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class Owner:
        async def classify_webhook_receipt(self, _: VerifiedWebhook) -> str:
            return "missing"

        def start_webhook_finalization(self, *_: object) -> object:
            raise AssertionError("capacity rejection must not start a finalizer")

    async def reject(_: VerifiedWebhook) -> object:
        raise CallAdmissionRejected("call_capacity_reached")

    metrics = RuntimeMetrics.in_memory()
    processor = TelnyxWebhookProcessor(
        verifier=Verifier(),  # type: ignore[arg-type]
        resolver=reject,  # type: ignore[arg-type]
        finalizer_owner=Owner(),  # type: ignore[arg-type]
    )
    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        webhook_processor=processor,
        metrics=metrics,
    )

    sent = await _raw_http(
        app,
        path="/telnyx/events",
        method="POST",
        messages=[{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert next(message for message in sent if message["type"] == "http.response.start")[
        "status"
    ] == 503

    collected = metrics._metric_reader.get_metrics_data()  # noqa: SLF001
    assert collected is not None
    points = {
        metric.name: list(metric.data.data_points)
        for resource in collected.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name
        in {
            "projetv0.voice.admission.rejections",
            "projetv0.voice.webhooks.total",
        }
    }
    assert [point.value for point in points["projetv0.voice.webhooks.total"]] == [1]
    assert dict(points["projetv0.voice.webhooks.total"][0].attributes) == {
        "webhook_class": "initiated",
        "receipt": "none",
        "disposition": "unavailable",
    }
    assert [point.value for point in points["projetv0.voice.admission.rejections"]] == [
        1
    ]
    assert dict(points["projetv0.voice.admission.rejections"][0].attributes) == {
        "reason": "capacity"
    }


@pytest.mark.asyncio
async def test_qualification_commit_rejection_emits_one_exact_ingress_metric_outcome() -> None:
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.telnyx.webhooks import (
        ResolvedWebhook,
        TelnyxWebhookProcessor,
        VerifiedWebhook,
        WebhookDisposition,
    )

    event = VerifiedWebhook(
        event_id="qualification-event",
        event_type="call.initiated",
        occurred_at=NOW,
        call_control_id="qualification-control",
        call_leg_id="qualification-leg",
        call_session_id="qualification-session",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"q" * 32,
        direction="incoming",
        call_state="parked",
    )

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class Handle:
        async def wait(self) -> WebhookDisposition:
            return WebhookDisposition(503, admission_rejection="qualification")

    class Owner:
        async def classify_webhook_receipt(self, _: VerifiedWebhook) -> str:
            return "missing"

        def start_webhook_finalization(self, *_: object) -> Handle:
            return Handle()

    metrics = RuntimeMetrics.in_memory()
    processor = TelnyxWebhookProcessor(
        verifier=Verifier(),  # type: ignore[arg-type]
        resolver=lambda _: ResolvedWebhook(None),
        finalizer_owner=Owner(),  # type: ignore[arg-type]
    )
    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        webhook_processor=processor,
        metrics=metrics,
    )

    sent = await _raw_http(
        app,
        path="/telnyx/events",
        method="POST",
        messages=[{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert next(message for message in sent if message["type"] == "http.response.start")[
        "status"
    ] == 503
    assert next(message for message in sent if message["type"] == "http.response.body")[
        "body"
    ] == b""

    collected = metrics._metric_reader.get_metrics_data()  # noqa: SLF001
    assert collected is not None
    points = {
        metric.name: list(metric.data.data_points)
        for resource in collected.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name
        in {
            "projetv0.voice.admission.rejections",
            "projetv0.voice.webhooks.total",
        }
    }
    assert [point.value for point in points["projetv0.voice.webhooks.total"]] == [1]
    assert dict(points["projetv0.voice.webhooks.total"][0].attributes) == {
        "webhook_class": "initiated",
        "receipt": "first",
        "disposition": "unavailable",
    }
    assert [point.value for point in points["projetv0.voice.admission.rejections"]] == [
        1
    ]
    assert dict(points["projetv0.voice.admission.rejections"][0].attributes) == {
        "reason": "qualification"
    }


@pytest.mark.asyncio
async def test_real_candidate_loser_preserves_qualification_ingress_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from projetv0_voice.admission import CallRegistry
    from projetv0_voice.app import create_app
    from projetv0_voice.lifecycle import RuntimeSupervisor
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.telnyx.call_control import CallControlResult
    from projetv0_voice.telnyx.webhooks import (
        TelnyxWebhookProcessor,
        VerifiedWebhook,
    )

    class Control:
        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            return CallControlResult("accepted")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def aclose(self) -> None:
            return None

    candidate_run_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    first_commit_entered = asyncio.Event()
    release_first_commit = asyncio.Event()
    failpoint_armed = False
    submissions = 0
    second_submitted = asyncio.Event()

    async def failpoint(name: str) -> None:
        if (
            failpoint_armed
            and name == "after_mutation_before_commit"
            and submissions == 1
            and not first_commit_entered.is_set()
        ):
            first_commit_entered.set()
            await release_first_commit.wait()

    writer = PersistenceWriter(
        tmp_path / "qualification-concurrent.sqlite",
        CryptoKeyring({1: KEY}, active_version=1),
        utcnow=lambda: NOW,
        failpoint=failpoint,
    )
    control = Control()
    registry = CallRegistry(
        writer=writer,
        call_control=control,
        tenant_id="tenant-a",
        agent_id="agent-a",
        deployment_id="deployment-a",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://voice.invalid/telnyx/stream",
        retention_days=7,
        utcnow=lambda: NOW,
        monotonic=lambda: 100.0,
        candidate_run_id=candidate_run_id,
    )
    metrics = RuntimeMetrics.in_memory()
    supervisor = RuntimeSupervisor(
        writer=writer,
        call_control=control,
        registry=registry,
        candidate_run_id=candidate_run_id,
        metrics=metrics,
        utcnow=lambda: NOW,
        loop_interval_seconds=0.05,
    )

    def event(event_id: str, fingerprint: bytes) -> VerifiedWebhook:
        return VerifiedWebhook(
            event_id=event_id,
            event_type="call.initiated",
            occurred_at=NOW,
            call_control_id="qualification-control",
            call_leg_id="qualification-leg",
            call_session_id="qualification-session",
            recording_id=None,
            stream_id=None,
            client_state=None,
            recording_started_at=None,
            recording_ended_at=None,
            recording_channels=None,
            semantic_fingerprint_sha256=fingerprint,
            direction="incoming",
            call_state="parked",
        )

    events = {
        b"first": event("qualification-first", b"a" * 32),
        b"second": event("qualification-second", b"b" * 32),
    }

    class Verifier:
        def verify(self, *, body: bytes, **_kwargs: object) -> VerifiedWebhook:
            return events[body]

    await supervisor.startup()
    first_request: asyncio.Task[list[dict[str, Any]]] | None = None
    second_request: asyncio.Task[list[dict[str, Any]]] | None = None
    try:
        for received in events.values():
            assert await supervisor.classify_webhook_receipt(received) == "missing"

        async def cached_missing(_event: VerifiedWebhook) -> str:
            return "missing"

        monkeypatch.setattr(supervisor, "classify_webhook_receipt", cached_missing)
        original_submit = writer.submit_webhook

        def observe_submit(*args: object, **kwargs: object) -> object:
            nonlocal submissions
            ticket = original_submit(*args, **kwargs)
            submissions += 1
            if submissions == 2:
                second_submitted.set()
            return ticket

        monkeypatch.setattr(writer, "submit_webhook", observe_submit)
        processor = TelnyxWebhookProcessor(
            verifier=Verifier(),  # type: ignore[arg-type]
            resolver=registry.resolve_webhook,
            duplicate_resolver=registry.resolve_duplicate_webhook,
            finalizer_owner=supervisor,
        )
        app = create_app(_runtime_settings(), SimpleNamespace())
        app.state.runtime_graph = SimpleNamespace(
            webhook_processor=processor,
            metrics=metrics,
        )

        async def post(body: bytes) -> list[dict[str, Any]]:
            return await _raw_http(
                app,
                path="/telnyx/events",
                method="POST",
                messages=[
                    {"type": "http.request", "body": body, "more_body": False}
                ],
            )

        failpoint_armed = True
        first_request = asyncio.create_task(post(b"first"))
        await first_commit_entered.wait()
        second_request = asyncio.create_task(post(b"second"))
        await second_submitted.wait()
        release_first_commit.set()
        first_response, second_response = await asyncio.gather(
            first_request,
            second_request,
        )

        statuses = [
            next(
                message["status"]
                for message in response
                if message["type"] == "http.response.start"
            )
            for response in (first_response, second_response)
        ]
        assert sorted(statuses) == [200, 503]
        assert all(
            next(
                message["body"]
                for message in response
                if message["type"] == "http.response.body"
            )
            == b""
            for response in (first_response, second_response)
        )

        collected = metrics._metric_reader.get_metrics_data()  # noqa: SLF001
        assert collected is not None
        points = {
            metric.name: list(metric.data.data_points)
            for resource in collected.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            if metric.name
            in {
                "projetv0.voice.admission.rejections",
                "projetv0.voice.webhooks.total",
            }
        }
        unavailable = [
            point
            for point in points["projetv0.voice.webhooks.total"]
            if dict(point.attributes)
            == {
                "webhook_class": "initiated",
                "receipt": "first",
                "disposition": "unavailable",
            }
        ]
        assert [point.value for point in unavailable] == [1]
        reasons = {
            dict(point.attributes)["reason"]: point.value
            for point in points["projetv0.voice.admission.rejections"]
        }
        assert reasons == {"qualification": 1}
    finally:
        release_first_commit.set()
        if first_request is not None:
            await asyncio.gather(first_request, return_exceptions=True)
        if second_request is not None:
            await asyncio.gather(second_request, return_exceptions=True)
        await supervisor.aclose()


@pytest.mark.asyncio
async def test_duplicate_resolver_rejection_emits_one_duplicate_and_zero_admission() -> None:
    from projetv0_voice.admission import CallAdmissionRejected
    from projetv0_voice.app import create_app
    from projetv0_voice.metrics import RuntimeMetrics
    from projetv0_voice.telnyx.webhooks import (
        TelnyxWebhookProcessor,
        VerifiedWebhook,
    )

    event = VerifiedWebhook(
        event_id="duplicate-event",
        event_type="call.initiated",
        occurred_at=NOW,
        call_control_id="control-a",
        call_leg_id="leg-a",
        call_session_id="session-a",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"n" * 32,
        legacy_v1_semantic_fingerprint_sha256=b"l" * 32,
        direction="incoming",
        call_state="parked",
    )

    class Verifier:
        def verify(self, **_: object) -> VerifiedWebhook:
            return event

    class Owner:
        async def classify_webhook_receipt(self, _: VerifiedWebhook) -> str:
            return "duplicate"

        def start_webhook_finalization(self, *_: object) -> object:
            raise AssertionError("duplicate resolver rejection has no finalizer")

    async def reject(_: VerifiedWebhook) -> object:
        raise CallAdmissionRejected("call_draining")

    metrics = RuntimeMetrics.in_memory()
    app = create_app(_runtime_settings(), SimpleNamespace())
    app.state.runtime_graph = SimpleNamespace(
        webhook_processor=TelnyxWebhookProcessor(
            verifier=Verifier(),  # type: ignore[arg-type]
            resolver=lambda _: object(),  # type: ignore[arg-type]
            duplicate_resolver=reject,  # type: ignore[arg-type]
            finalizer_owner=Owner(),  # type: ignore[arg-type]
        ),
        metrics=metrics,
    )

    sent = await _raw_http(
        app,
        path="/telnyx/events",
        method="POST",
        messages=[{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert next(message for message in sent if message["type"] == "http.response.start")[
        "status"
    ] == 503

    collected = metrics._metric_reader.get_metrics_data()  # noqa: SLF001
    assert collected is not None
    points = {
        metric.name: list(metric.data.data_points)
        for resource in collected.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name
        in {
            "projetv0.voice.admission.rejections",
            "projetv0.voice.webhooks.total",
        }
    }
    assert "projetv0.voice.admission.rejections" not in points
    assert [point.value for point in points["projetv0.voice.webhooks.total"]] == [1]
    assert dict(points["projetv0.voice.webhooks.total"][0].attributes) == {
        "webhook_class": "initiated",
        "receipt": "duplicate",
        "disposition": "unavailable",
    }


def test_integrated_runtime_metric_owner_inventory_is_exactly_thirteen() -> None:
    from projetv0_voice.app import INGRESS_METRIC_OWNERS
    from projetv0_voice.lifecycle import NON_INGRESS_METRIC_OWNERS

    assert not (set(INGRESS_METRIC_OWNERS) & set(NON_INGRESS_METRIC_OWNERS))
    assert dict(INGRESS_METRIC_OWNERS) | dict(NON_INGRESS_METRIC_OWNERS) == {
        "admission.rejections": "webhook_delivery",
        "webhooks.total": "webhook_delivery",
        "actions.total": "call_control",
        "relay.runs": "relay_supervisor",
        "writer.queue_depth": "writer_snapshot",
        "writer.queue_oldest_age": "writer_snapshot",
        "writer.quick_check": "writer_snapshot",
        "outbox.depth": "writer_snapshot",
        "outbox.oldest_age": "writer_snapshot",
        "outbox.bytes": "writer_snapshot",
        "recordings.total": "recording_transitions",
        "runtime.event_loop_lag": "lag_supervisor",
        "ready": "runtime_publication",
    }


@pytest.mark.asyncio
async def test_lifespan_publishes_startup_failure_after_runtime_startup_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.app as app_module

    events: list[str] = []

    class Supervisor:
        async def startup(self) -> None:
            events.append("startup")
            raise RuntimeError("private-startup-sentinel")

        async def aclose(self) -> None:
            events.append("close")

    class Graph:
        def __init__(self) -> None:
            self.supervisor = Supervisor()

    class Coordinator:
        async def build_runtime(self, _settings: RuntimeSettingsV1) -> Graph:
            events.append("build")
            return Graph()

        def publish_runtime(self, _supervisor: object) -> None:
            events.append("runtime")

        def publish_startup_complete(self) -> None:
            events.append("complete")

        def publish_startup_failure(self) -> None:
            events.append("failure")

        def unpublish_runtime(self, _supervisor: object) -> None:
            assert events[-1] == "close"
            events.append("unpublish")

    monkeypatch.setattr(app_module, "RuntimeProductionGraph", Graph)
    app = app_module.create_app(_runtime_settings(), Coordinator())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="private-startup-sentinel"):
        async with app.router.lifespan_context(app):
            raise AssertionError("startup failure must not yield")

    assert events == ["build", "runtime", "startup", "failure", "close", "unpublish"]


@pytest.mark.asyncio
async def test_lifespan_unpublishes_runtime_after_normal_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.app as app_module

    events: list[str] = []

    class Supervisor:
        async def startup(self) -> None:
            events.append("startup")

        async def aclose(self) -> None:
            events.append("close")

    class Graph:
        def __init__(self) -> None:
            self.supervisor = Supervisor()

    class Coordinator:
        async def build_runtime(self, _settings: RuntimeSettingsV1) -> Graph:
            events.append("build")
            return Graph()

        def publish_runtime(self, _supervisor: object) -> None:
            events.append("runtime")

        def publish_startup_complete(self) -> None:
            events.append("complete")

        def publish_startup_failure(self) -> None:
            events.append("failure")

        def unpublish_runtime(self, _supervisor: object) -> None:
            assert events[-1] == "close"
            events.append("unpublish")

    monkeypatch.setattr(app_module, "RuntimeProductionGraph", Graph)
    app = app_module.create_app(_runtime_settings(), Coordinator())  # type: ignore[arg-type]

    async with app.router.lifespan_context(app):
        assert events == ["build", "runtime", "startup", "complete"]

    assert events == ["build", "runtime", "startup", "complete", "close", "unpublish"]
