"""Closed FastAPI ingress for the production voice runtime."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import Any, Protocol, cast

from fastapi import FastAPI, Request, Response, WebSocket

from projetv0_voice.lifecycle import (
    IngressMetricObservation,
    IngressMetricOutcome,
    RuntimeProductionGraph,
    RuntimeSupervisor,
)
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.runtime_config import RuntimeSettingsV1
from projetv0_voice.telnyx.handshake import (
    TelnyxHandshakeRejectedError,
    TelnyxHandshakeTimeoutError,
)
from projetv0_voice.telnyx.webhooks import (
    MAX_WEBHOOK_BODY_BYTES,
    ObservedWebhookResult,
)

INGRESS_METRIC_OWNERS: Mapping[str, str] = MappingProxyType(
    {
        "admission.rejections": "webhook_delivery",
        "webhooks.total": "webhook_delivery",
    }
)

_PUBLIC_BODY = b""
_PUBLIC_STATUSES = frozenset({200, 400, 403, 413, 500, 503})


class _Coordinator(Protocol):
    async def build_runtime(self, settings: RuntimeSettingsV1) -> RuntimeProductionGraph: ...

    def publish_runtime(self, supervisor: RuntimeSupervisor) -> None: ...

    def publish_startup_complete(self) -> None: ...

    def publish_startup_failure(self) -> None: ...

    def unpublish_runtime(self, supervisor: RuntimeSupervisor) -> None: ...


def _runtime(app: FastAPI) -> Any:
    return getattr(app.state, "runtime_graph", None)


def _response(status_code: int) -> Response:
    selected = status_code if status_code in _PUBLIC_STATUSES else 500
    return Response(content=_PUBLIC_BODY, status_code=selected, media_type=None)


def _terminal_outcome(
    *,
    status_code: int,
    rejection: str | None = None,
) -> IngressMetricOutcome:
    disposition = {
        200: "ok",
        400: "bad_request",
        403: "forbidden",
        413: "too_large",
        503: "unavailable",
    }.get(status_code, "internal_error")
    return IngressMetricOutcome(
        webhook_class="invalid",
        receipt="none",
        disposition=cast(Any, disposition),
        admission_rejection=cast(Any, rejection),
    )


def _metric_outcome(observed: ObservedWebhookResult) -> IngressMetricOutcome:
    return IngressMetricOutcome(
        webhook_class=observed.webhook_class,
        receipt=observed.receipt,
        disposition=observed.metric_disposition,
        admission_rejection=observed.admission_rejection,
    )


def _finish_webhook_metric(
    graph: object | None,
    outcome: IngressMetricOutcome,
) -> None:
    metrics = getattr(graph, "metrics", None)
    if not isinstance(metrics, RuntimeMetrics):
        return
    IngressMetricObservation(metrics).finish(outcome)


async def _bounded_body(request: Request) -> bytes | None:
    retained = bytearray()
    async for chunk in request.stream():
        if not isinstance(chunk, bytes):
            return None
        remaining = MAX_WEBHOOK_BODY_BYTES + 1 - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(retained) > MAX_WEBHOOK_BODY_BYTES:
            return None
    return bytes(retained)


def _ordered_headers(request: Request) -> list[tuple[str, str]] | None:
    raw = request.scope.get("headers")
    if not isinstance(raw, list):
        return None
    result: list[tuple[str, str]] = []
    for item in raw:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], bytes)
            or not isinstance(item[1], bytes)
        ):
            return None
        result.append((item[0].decode("latin-1"), item[1].decode("latin-1")))
    return result


def create_app(
    settings: RuntimeSettingsV1,
    coordinator: _Coordinator,
) -> FastAPI:
    """Create the exact four-route production ASGI application."""

    if type(settings) is not RuntimeSettingsV1:
        raise ValueError("runtime_app_config_invalid") from None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        graph: RuntimeProductionGraph | None = None
        supervisor: RuntimeSupervisor | None = None
        failed = False
        try:
            graph = await coordinator.build_runtime(settings)
            if not isinstance(graph, RuntimeProductionGraph):
                raise RuntimeError("runtime_graph_invalid")
            supervisor = graph.supervisor
            app.state.runtime_graph = graph
            coordinator.publish_runtime(supervisor)
            await supervisor.startup()
            coordinator.publish_startup_complete()
            yield
        except asyncio.CancelledError:
            failed = True
            raise
        except BaseException:
            failed = True
            raise
        finally:
            if failed:
                coordinator.publish_startup_failure()
            if supervisor is not None:
                try:
                    await supervisor.aclose()
                finally:
                    coordinator.unpublish_runtime(supervisor)
            app.state.runtime_graph = None

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
        lifespan=lifespan,
    )

    @app.get("/health/live", response_class=Response)
    async def live() -> Response:
        return _response(200)

    @app.get("/health/ready", response_class=Response)
    async def ready() -> Response:
        graph = _runtime(app)
        try:
            snapshot = graph.supervisor.readiness_snapshot()
            status = 200 if snapshot.ready is True else 503
        except BaseException:
            status = 503
        return _response(status)

    @app.post("/telnyx/events", response_class=Response)
    async def webhook(request: Request) -> Response:
        graph = _runtime(app)
        status = 500
        outcome = _terminal_outcome(status_code=500)
        try:
            body = await _bounded_body(request)
            if body is None:
                status = 413
                outcome = _terminal_outcome(
                    status_code=status,
                    rejection="invalid",
                )
            else:
                headers = _ordered_headers(request)
                processor = getattr(graph, "webhook_processor", None)
                if headers is None or processor is None:
                    status = 500
                else:
                    observed = await processor.process_observed(
                        body=body,
                        headers=headers,
                    )
                    if isinstance(observed, ObservedWebhookResult):
                        status = observed.disposition.status_code
                        outcome = _metric_outcome(observed)
        except asyncio.CancelledError:
            raise
        except Exception:
            status = 500
            outcome = _terminal_outcome(status_code=500)
        _finish_webhook_metric(graph, outcome)
        return _response(status)

    @app.websocket("/telnyx/media")
    async def stream(websocket: WebSocket) -> None:
        graph = _runtime(app)
        gate = getattr(graph, "unauthenticated_gate", None)
        handshake = getattr(graph, "handshake", None)
        session_factory = getattr(graph, "session_factory", None)
        if gate is None or handshake is None or session_factory is None:
            await websocket.send_denial_response(_response(503))
            return

        permit: Any | None = None
        try:
            permit = gate.try_acquire()
        except Exception:
            await websocket.send_denial_response(_response(503))
            return
        if permit is None:
            await websocket.send_denial_response(_response(503))
            return

        transferred = False
        try:
            await websocket.accept()
            operation = handshake.transfer_authentication(websocket, permit)
            transferred = True
            authenticated = await operation
            await session_factory.run(authenticated)
        except asyncio.CancelledError:
            raise
        except TelnyxHandshakeRejectedError:
            with contextlib.suppress(Exception):
                await websocket.close(code=1008, reason="authentication_rejected")
        except TelnyxHandshakeTimeoutError:
            with contextlib.suppress(Exception):
                await websocket.close(code=1013, reason="authentication_timeout")
        except Exception:
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="runtime_failed")
        finally:
            if not transferred:
                with contextlib.suppress(Exception):
                    permit.release()

    return app


__all__ = ["INGRESS_METRIC_OWNERS", "create_app"]
