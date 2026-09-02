"""Stdlib-first Uvicorn process entrypoint and first-signal coordination."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
from collections.abc import Awaitable, Callable, Mapping
from types import FrameType
from typing import TYPE_CHECKING, Any, NoReturn, cast

from projetv0_voice.runtime_config import (
    RuntimeSettingsV1,
    capture_runtime_environment,
    parse_runtime_settings,
)

if TYPE_CHECKING:
    from projetv0_voice.lifecycle import RuntimeProductionGraph, RuntimeSupervisor

type RuntimeBuilder = Callable[[RuntimeSettingsV1], Awaitable[RuntimeProductionGraph]]

_RUNTIME_HARD_EXIT_CODE = 72
_UVICORN_FLOOR_LOGGERS = (
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
)
_UVICORN_LOGGERS = (
    "uvicorn",
    *_UVICORN_FLOOR_LOGGERS,
)


def _install_uvicorn_logging_floors() -> None:
    """Install exact stdlib-only privacy floors before any heavy import."""

    for name in _UVICORN_FLOOR_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers[:] = [logging.NullHandler()]
        logger.propagate = False
        logger.disabled = True
        logger.setLevel(logging.CRITICAL + 1)


class FirstSignalDrainCoordinator:
    """Latch the first signal before app publication and never force Uvicorn."""

    def __init__(
        self,
        *,
        settings: RuntimeSettingsV1 | None = None,
        runtime_builder: RuntimeBuilder | None = None,
        hard_exit: Callable[[int], NoReturn] = os._exit,
    ) -> None:
        if settings is not None and type(settings) is not RuntimeSettingsV1:
            raise ValueError("signal_coordinator_config_invalid") from None
        if runtime_builder is not None and not callable(runtime_builder):
            raise ValueError("signal_coordinator_config_invalid") from None
        if not callable(hard_exit):
            raise ValueError("signal_coordinator_config_invalid") from None
        self._settings = settings
        self._runtime_builder = runtime_builder
        self._hard_exit = hard_exit
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: VoiceUvicornServer | None = None
        self._runtime: RuntimeSupervisor | None = None
        self._startup_failed = False
        self._startup_complete = False
        self._startup_owner: asyncio.Task[Any] | None = None
        self._published = asyncio.Event()
        self._urgent_event = asyncio.Event()
        self._first_signal: int | None = None
        self._urgent = False
        self._signal_task: asyncio.Task[None] | None = None
        self._deadline_task: asyncio.Task[None] | None = None

    @property
    def urgent(self) -> bool:
        with self._lock:
            return self._urgent

    async def build_runtime(self, settings: RuntimeSettingsV1) -> RuntimeProductionGraph:
        if type(settings) is not RuntimeSettingsV1:
            raise RuntimeError("runtime_settings_invalid") from None
        builder = self._runtime_builder
        if builder is None:
            builder = _default_runtime_builder
        return await builder(settings)

    def bind_server(
        self,
        server: VoiceUvicornServer,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if not isinstance(server, VoiceUvicornServer) or not isinstance(
            loop, asyncio.AbstractEventLoop
        ):
            raise ValueError("signal_coordinator_binding_invalid") from None
        with self._lock:
            if self._server is not None and self._server is not server:
                raise RuntimeError("signal_coordinator_already_bound") from None
            self._server = server
            self._loop = loop
            signal_latched = self._first_signal is not None
        if signal_latched:
            self._schedule()

    def publish_runtime(self, supervisor: RuntimeSupervisor) -> None:
        owner = asyncio.current_task()
        with self._lock:
            if self._runtime is not None or self._startup_failed:
                return
            self._runtime = supervisor
            self._startup_owner = owner
        self._published.set()

    def publish_startup_complete(self) -> None:
        with self._lock:
            if self._runtime is None or self._startup_failed:
                return
            self._startup_complete = True
            self._startup_owner = None

    def publish_startup_failure(self) -> None:
        with self._lock:
            if self._startup_complete or self._startup_failed:
                return
            self._startup_failed = True
            self._startup_owner = None
        self._published.set()

    def handle_signal(self, sig: int) -> None:
        if sig not in {signal.SIGINT, signal.SIGTERM}:
            return
        urgent = False
        server: VoiceUvicornServer | None
        with self._lock:
            if self._first_signal is None:
                self._first_signal = sig
            elif sig == signal.SIGINT:
                self._urgent = True
                urgent = True
            server = self._server
        if urgent:
            self._urgent_event.set()
            if server is not None:
                server.should_exit = True
        self._schedule()

    def _schedule(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def start() -> None:
            if self._signal_task is None:
                self._signal_task = loop.create_task(
                    self._coordinate_signal(),
                    name="voice-first-signal",
                )
            if self._deadline_task is None:
                self._deadline_task = loop.create_task(
                    self._enforce_deadline(),
                    name="voice-global-shutdown-deadline",
                )

        loop.call_soon_threadsafe(start)

    async def _coordinate_signal(self) -> None:
        settings = self._settings
        server = self._server
        if settings is None or server is None:
            return
        deadline = asyncio.get_running_loop().time() + settings.shutdown_grace_seconds
        try:
            async with asyncio.timeout_at(deadline):
                await self._published.wait()
                runtime = self._runtime
                with self._lock:
                    startup_complete = self._startup_complete
                    startup_owner = self._startup_owner
                    startup_failed = self._startup_failed
                if startup_failed:
                    server.should_exit = True
                    return
                if not startup_complete and startup_owner is not None:
                    startup_owner.cancel()
                    server.should_exit = True
                    return
                if runtime is not None:
                    await runtime.begin_drain()
                if runtime is not None and not self.urgent:
                    try:
                        async with asyncio.timeout(settings.pre_drain_grace_seconds):
                            await self._urgent_event.wait()
                    except TimeoutError:
                        pass
                server.should_exit = True
        except TimeoutError:
            return

    async def _enforce_deadline(self) -> None:
        settings = self._settings
        if settings is None:
            return
        await asyncio.sleep(settings.shutdown_grace_seconds)
        self._hard_exit(_RUNTIME_HARD_EXIT_CODE)
        os._exit(_RUNTIME_HARD_EXIT_CODE)

    async def aclose(self) -> None:
        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in (self._signal_task, self._deadline_task)
            if task is not None and task is not current
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class VoiceUvicornServer:
    """A lazy wrapper retaining Uvicorn's normal lifespan and shutdown flow."""

    def __init__(
        self,
        app: Any,
        *,
        settings: RuntimeSettingsV1,
        coordinator: FirstSignalDrainCoordinator,
    ) -> None:
        if type(settings) is not RuntimeSettingsV1 or not isinstance(
            coordinator, FirstSignalDrainCoordinator
        ):
            raise ValueError("voice_server_config_invalid") from None
        import uvicorn

        config = uvicorn.Config(
            app,
            host=settings.bind_host,
            port=settings.bind_port,
            workers=1,
            reload=False,
            env_file=None,
            lifespan="on",
            loop="asyncio",
            ws="websockets-sansio",
            ws_max_size=524_288,
            ws_per_message_deflate=False,
            timeout_graceful_shutdown=settings.uvicorn_grace_seconds,
            log_config=None,
            log_level=None,
            access_log=False,
            proxy_headers=False,
            forwarded_allow_ips=[],
        )
        self._server = uvicorn.Server(config)
        cast(Any, self._server).handle_exit = self.handle_exit
        self._coordinator = coordinator

    @property
    def config(self) -> Any:
        return self._server.config

    @property
    def started(self) -> bool:
        return bool(self._server.started)

    @property
    def should_exit(self) -> bool:
        return bool(self._server.should_exit)

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._server.should_exit = bool(value)

    @property
    def force_exit(self) -> bool:
        return bool(self._server.force_exit)

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        del frame
        self._coordinator.handle_signal(sig)

    async def serve(
        self,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        previous = {
            name: logging.getLogger(name).disabled for name in _UVICORN_LOGGERS
        }
        for name in _UVICORN_LOGGERS:
            logging.getLogger(name).disabled = True
        self._coordinator.bind_server(self, asyncio.get_running_loop())
        try:
            await self._server.serve(sockets=sockets)
        finally:
            await self._coordinator.aclose()
            for name, disabled in previous.items():
                logging.getLogger(name).disabled = disabled


async def _default_runtime_builder(
    settings: RuntimeSettingsV1,
) -> RuntimeProductionGraph:
    from projetv0_voice.lifecycle import build_production_runtime
    from projetv0_voice.production_wiring import build_production_factories

    factories = build_production_factories(settings)
    return await build_production_runtime(settings, factories=factories)


def main(mapping: Mapping[str, object] = os.environ) -> None:
    """Parse and guard the process before importing any runtime dependency."""

    capture = capture_runtime_environment(mapping)
    settings = parse_runtime_settings(capture)
    _install_uvicorn_logging_floors()

    from projetv0_voice.dependency_logging import configure_dependency_logging

    configure_dependency_logging(settings.observability_token())

    from projetv0_voice.app import create_app

    coordinator = FirstSignalDrainCoordinator(settings=settings)
    app = create_app(settings, coordinator)
    server = VoiceUvicornServer(app, settings=settings, coordinator=coordinator)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()


__all__ = [
    "FirstSignalDrainCoordinator",
    "VoiceUvicornServer",
    "main",
]
