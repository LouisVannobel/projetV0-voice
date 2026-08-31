"""Atomic process factory for one owned, measured call session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import suppress
from importlib.metadata import version
from typing import Any, Protocol
from uuid import UUID

from pipecat.processors.frame_processor import FrameProcessor

from projetv0_voice.admission import (
    CallConstructionGrant,
    TerminalAuthority,
    TerminalProposal,
)
from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.metrics import RuntimeMetrics, _CallMetricLease
from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1
from projetv0_voice.persistence.commands import PersistenceCommand
from projetv0_voice.pipeline import _CallObservers
from projetv0_voice.qualified_profile import RuntimeDeploymentProfileV1
from projetv0_voice.session import (
    CallIdentity,
    CallSession,
    PublicSttHttpClient,
    RecordingBoundary,
    ServiceBundle,
    SessionWriter,
)
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake


class ProcessTaskRegistrar(Protocol):
    def try_start(
        self,
        coroutine: Coroutine[object, object, None],
        *,
        name: str,
    ) -> asyncio.Task[None] | None: ...


class _SessionRegistry(Protocol):
    async def consume_claim_for_construction(
        self,
        claim: object,
        stream_id: str,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> CallConstructionGrant | None: ...

    async def construction_is_live(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
    ) -> bool: ...

    async def activate_session(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
        session: object,
    ) -> bool: ...

    async def reserve_or_read_terminal(
        self,
        grant: CallConstructionGrant,
        owner: object,
        owner_task: asyncio.Task[None],
        proposed: TerminalProposal,
    ) -> TerminalAuthority: ...

    async def complete_reserved_terminal(self, authority: object) -> bool: ...

    async def prepare_required_recording_drain(
        self,
        call_id: UUID,
    ) -> TerminalAuthority | None: ...

    def _note_terminal_failure(self, code: str) -> None: ...


class _RegistryTerminalizer:
    __slots__ = ("_grant", "_owner", "_registry")

    def __init__(
        self,
        registry: _SessionRegistry,
        grant: CallConstructionGrant,
        owner: _CallLifecycleOwner,
    ) -> None:
        self._registry = registry
        self._grant = grant
        self._owner = owner

    async def terminalize(
        self,
        _identity: CallIdentity,
        *,
        status: str,
        reason: str,
    ) -> None:
        del status, reason

    async def reserve_or_read(
        self,
        proposed: TerminalProposal,
    ) -> TerminalAuthority:
        task = self._owner._task
        if task is None:
            raise RuntimeError("terminal_authority_unavailable") from None
        return await self._registry.reserve_or_read_terminal(
            self._grant,
            self._owner,
            task,
            proposed,
        )

    async def complete(self, authority: TerminalAuthority) -> bool:
        return await self._registry.complete_reserved_terminal(authority)

    def note_failure(self, code: str) -> None:
        self._registry._note_terminal_failure(code)


class _ConstructionStack:
    __slots__ = ("_callbacks",)

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], Any]] = []

    def push(self, callback: Callable[[], Any]) -> None:
        self._callbacks.append(callback)

    def detach_all(self) -> None:
        self._callbacks.clear()

    def discard(self, callback: Callable[[], Any]) -> None:
        self._callbacks = [retained for retained in self._callbacks if retained is not callback]

    async def aclose(self) -> None:
        while self._callbacks:
            callback = self._callbacks.pop()
            try:
                result = callback()
                if hasattr(result, "__await__"):
                    await result
            except BaseException:
                continue


class _CallLifecycleOwner:
    """One closed-gate owner and retained task for a call's entire lifecycle."""

    __slots__ = (
        "_closed",
        "_drain_cause",
        "_factory",
        "_gate",
        "_grant",
        "_handshake",
        "_metric_lease",
        "_phase",
        "_session",
        "_task",
    )

    def __init__(self, factory: ProcessSessionFactory) -> None:
        self._factory = factory
        self._closed = asyncio.Event()
        self._gate = asyncio.Event()
        self._phase = "gated"
        self._task: asyncio.Task[None] | None = None
        self._grant: CallConstructionGrant | None = None
        self._handshake: AuthenticatedTelnyxHandshake | None = None
        self._metric_lease: _CallMetricLease | None = None
        self._session: object | None = None
        self._drain_cause: str | None = None

    def bind_task(self, task: asyncio.Task[None]) -> None:
        self._task = task

    def publish(
        self,
        grant: CallConstructionGrant,
        handshake: AuthenticatedTelnyxHandshake,
        metric_lease: _CallMetricLease,
    ) -> None:
        self._grant = grant
        self._handshake = handshake
        self._metric_lease = metric_lease
        self._gate.set()

    def request_drain(self, cause: str) -> None:
        if type(cause) is not str or not cause or self._phase == "done":
            return
        if self._drain_cause is None or cause == "recording_required_error":
            self._drain_cause = cause
        task = self._task
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
            and task.cancelling() == 0
        ):
            task.cancel()

    async def wait(self) -> None:
        wait_task = asyncio.create_task(self._closed.wait())
        cancellation: asyncio.CancelledError | None = None
        while not wait_task.done():
            try:
                await asyncio.shield(wait_task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        await wait_task
        if cancellation is not None:
            raise cancellation

    async def cancel_and_join(self) -> None:
        task = self._task
        if task is None:
            self._phase = "done"
            self._closed.set()
            return
        if not task.done() and task.cancelling() == 0:
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done():
            with suppress(BaseException):
                task.result()

    async def run(self) -> None:
        try:
            await self._gate.wait()
            grant = self._grant
            handshake = self._handshake
            metric_lease = self._metric_lease
            task = self._task
            if (
                grant is None
                or handshake is None
                or metric_lease is None
                or task is None
            ):
                return
            self._phase = "constructing"
            await self._factory._construct_and_run(  # noqa: SLF001
                self,
                task,
                grant,
                handshake,
                metric_lease,
            )
        except BaseException:
            return
        finally:
            self._phase = "done"
            self._closed.set()


class ProcessSessionFactory:
    """Create and supervise one exact call from an authenticated handshake."""

    def __init__(
        self,
        *,
        registry: _SessionRegistry,
        registrar: ProcessTaskRegistrar,
        runtime_metrics: RuntimeMetrics,
        manifest: AgentManifestV1,
        profile: RuntimeDeploymentProfileV1,
        writer: SessionWriter,
        keyring: CryptoKeyring,
        stt_http_client_factory: Callable[[], PublicSttHttpClient],
        stt_factory: Callable[[PublicSttHttpClient], FrameProcessor],
        llm_factory: Callable[[], FrameProcessor],
        tts_factory: Callable[[], FrameProcessor],
        recording_factory: Callable[[CallIdentity], RecordingBoundary],
        idle_timeout_seconds: float,
        session_factory: Callable[..., CallSession] = CallSession,
        cleanup_phase_timeout_seconds: float = 5.0,
    ) -> None:
        self._registry = registry
        self._registrar = registrar
        self._runtime_metrics = runtime_metrics
        self._manifest = manifest
        self._profile = profile
        self._writer = writer
        self._keyring = keyring
        self._stt_http_client_factory = stt_http_client_factory
        self._stt_factory = stt_factory
        self._llm_factory = llm_factory
        self._tts_factory = tts_factory
        self._recording_factory = recording_factory
        self._session_factory = session_factory
        self._idle_timeout_seconds = idle_timeout_seconds
        self._cleanup_phase_timeout_seconds = cleanup_phase_timeout_seconds

    async def run(self, handshake: AuthenticatedTelnyxHandshake) -> None:
        owner = _CallLifecycleOwner(self)
        owner_coroutine = owner.run()
        owner_task = self._registrar.try_start(
            owner_coroutine,
            name="voice-call-session-owner",
        )
        if owner_task is None:
            owner._phase = "done"  # noqa: SLF001
            owner._closed.set()  # noqa: SLF001
            raise RuntimeError("session_owner_registration_failed") from None
        owner.bind_task(owner_task)
        try:
            stream_id = handshake.call_data.stream_id
            grant = await self._registry.consume_claim_for_construction(
                handshake.lease_claim,
                stream_id if type(stream_id) is str else "",
                owner,
                owner_task,
            )
        except asyncio.CancelledError:
            await owner.cancel_and_join()
            raise
        if grant is None:
            await owner.cancel_and_join()
            raise RuntimeError("process_lease_claim_unavailable") from None
        metric_lease = self._runtime_metrics.begin_call()
        owner.publish(grant, handshake, metric_lease)
        cancellation: asyncio.CancelledError | None = None
        owner_wait = asyncio.create_task(owner.wait())
        while not owner_wait.done():
            try:
                await asyncio.shield(owner_wait)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                    owner.request_drain("external_cancel")
            except BaseException:
                break
        try:
            await owner_wait
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        if owner_task.done():
            with suppress(BaseException):
                owner_task.result()
        if cancellation is not None:
            raise cancellation

    async def drain_call_by_id(self, call_id: UUID, reason: str) -> None:
        if type(reason) is not str or reason != "recording_required_error":
            raise ValueError("call_drain_reason_invalid") from None
        if type(call_id) is not UUID:
            raise ValueError("call_drain_call_id_invalid") from None
        authority = await self._registry.prepare_required_recording_drain(call_id)
        if authority is None:
            return
        await self._persist_unconsumed_terminal(authority)
        await self._registry.complete_reserved_terminal(authority)

    async def _live(
        self,
        grant: CallConstructionGrant,
        owner: _CallLifecycleOwner,
        owner_task: asyncio.Task[None],
    ) -> bool:
        live = await self._registry.construction_is_live(
            grant,
            owner,
            owner_task,
        )
        if not live and owner._drain_cause is None:  # noqa: SLF001
            owner._drain_cause = "token_deadline"  # noqa: SLF001
        return live

    async def _construct_and_run(
        self,
        owner: _CallLifecycleOwner,
        owner_task: asyncio.Task[None],
        grant: CallConstructionGrant,
        handshake: AuthenticatedTelnyxHandshake,
        metric_lease: _CallMetricLease,
    ) -> None:
        stack = _ConstructionStack()
        transferred = False
        session: CallSession | None = None
        try:
            if not await self._live(grant, owner, owner_task):
                return
            identity = CallIdentity(
                call_id=grant.call_id,
                generation=grant.generation,
                lease_claim=grant.lease_claim,
                deployment_id=grant.deployment_id,
                telnyx_call_control_id=grant.telnyx_call_control_id,
                telnyx_call_leg_id=grant.telnyx_call_leg_id,
                telnyx_call_session_id=grant.telnyx_call_session_id,
                stream_id=grant.stream_id,
                started_at=grant.started_at,
                retention_until=grant.retention_until,
            )
            if not await self._live(grant, owner, owner_task):
                return
            stt_client = self._stt_http_client_factory()
            stt_client_close = stt_client.aclose
            stack.push(stt_client_close)
            if not await self._live(grant, owner, owner_task):
                return
            stt = self._stt_factory(stt_client)
            stack.push(stt.cleanup)
            if not await self._live(grant, owner, owner_task):
                return
            llm = self._llm_factory()

            def llm_client_close() -> Any:
                return self._close_llm_client(llm)

            stack.push(llm_client_close)
            stack.push(llm.cleanup)
            if not await self._live(grant, owner, owner_task):
                return
            tts = self._tts_factory()
            stack.push(tts.cleanup)
            if not await self._live(grant, owner, owner_task):
                return
            services = ServiceBundle(
                stt=stt,
                llm=llm,
                tts=tts,
                stt_http_client=stt_client,
            )
            stack.discard(stt_client_close)
            stack.discard(llm_client_close)
            stack.push(services.aclose)
            if not await self._live(grant, owner, owner_task):
                return
            observers = _CallObservers(
                runtime_metrics=self._runtime_metrics,
                stt=stt,
                llm=llm,
                tts=tts,
            )
            if not await self._live(grant, owner, owner_task):
                return
            recording = self._recording_factory(identity)
            stack.push(
                lambda: recording.cleanup(
                    identity,
                    recording_may_be_active=False,
                    reason="session_construction_failed",
                )
            )
            if not await self._live(grant, owner, owner_task):
                return
            terminalizer = _RegistryTerminalizer(
                self._registry,
                grant,
                owner,
            )
            session = self._session_factory(
                identity=identity,
                manifest=self._manifest,
                profile=self._profile,
                services=services,
                writer=self._writer,
                keyring=self._keyring,
                recording=recording,
                lease_terminalizer=terminalizer,
                registry_terminalizer=terminalizer,
                metric_lease=metric_lease,
                runtime_metrics=self._runtime_metrics,
                observers=observers,
                idle_timeout_seconds=self._idle_timeout_seconds,
                cleanup_phase_timeout_seconds=self._cleanup_phase_timeout_seconds,
            )
            owner._session = session  # noqa: SLF001
            owner._phase = "preactivated"  # noqa: SLF001
            stack.detach_all()
            transferred = True
            if await self._registry.activate_session(
                grant,
                owner,
                owner_task,
                session,
            ):
                await session.run(handshake)
                owner._phase = "finishing"  # noqa: SLF001
                return
            if owner._drain_cause is None:  # noqa: SLF001
                owner._drain_cause = "token_deadline"  # noqa: SLF001
        finally:
            if not transferred:
                await stack.aclose()
                await self._finish_unstarted(
                    grant,
                    owner,
                    metric_lease,
                    self._pre_run_reason(owner),
                )
            elif owner._phase == "preactivated" and session is not None:  # noqa: SLF001
                owner._phase = "finishing"  # noqa: SLF001
                try:
                    await session.request_drain(self._pre_run_reason(owner))
                finally:
                    await session.aclose_unstarted()

    @staticmethod
    async def _close_llm_client(llm: FrameProcessor) -> None:
        if version("pipecat-ai") != "1.7.0":
            return
        client = getattr(llm, "_client", None)
        close = getattr(client, "close", None)
        if not callable(close):
            return
        result = close()
        if hasattr(result, "__await__"):
            await result

    @staticmethod
    def _pre_run_reason(owner: _CallLifecycleOwner) -> str:
        return owner._drain_cause or "session_construction_failed"

    async def _finish_unstarted(
        self,
        grant: CallConstructionGrant,
        owner: _CallLifecycleOwner,
        metric_lease: _CallMetricLease,
        reason: str,
    ) -> None:
        owner._phase = "finishing"  # noqa: SLF001
        if reason == "recording_required_error":
            proposed = TerminalProposal(
                status="failed",
                reason=reason,
                metric_class="failed",
                cleanup_hangup=False,
            )
        elif reason == "process_draining":
            proposed = TerminalProposal(
                status="closed",
                reason=reason,
                metric_class="drained",
                cleanup_hangup=True,
            )
        else:
            proposed = TerminalProposal(
                status="failed",
                reason=reason,
                metric_class="failed",
                cleanup_hangup=True,
            )
        terminalizer = _RegistryTerminalizer(self._registry, grant, owner)
        authority = await terminalizer.reserve_or_read(proposed)
        metric_lease.finish(authority.metric_class)
        if authority.persist_call:
            await self._persist_terminal_call(grant, authority)
        await terminalizer.complete(authority)

    async def _persist_terminal_call(
        self,
        grant: CallConstructionGrant,
        authority: TerminalAuthority,
    ) -> None:
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=authority.completion_token,
            deployment_id=grant.deployment_id,
            call_id=grant.call_id,
            occurred_at=authority._closed_at,  # noqa: SLF001
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=grant.telnyx_call_control_id,
                telnyx_call_leg_id=grant.telnyx_call_leg_id,
                telnyx_call_session_id=grant.telnyx_call_session_id,
                status=authority.status,
                disclosure_state="failed",
                started_at=grant.started_at,
                ended_at=authority._closed_at,  # noqa: SLF001
                end_reason=authority.reason,
                retention_until=grant.retention_until,
            ),
        )
        while True:
            try:
                await self._writer.commit_control(
                    PersistenceCommand("outbox", {"operation": operation}, None)
                )
            except asyncio.CancelledError:
                continue
            except BaseException:
                self._registry._note_terminal_failure(
                    "terminal_persistence_failed"
                )
                return
            return

    async def _persist_unconsumed_terminal(
        self,
        authority: TerminalAuthority,
    ) -> None:
        entry = authority._entry  # noqa: SLF001
        started_at = entry.claimed_at or entry.created_at
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=authority.completion_token,
            deployment_id=self._registry._deployment_id,  # type: ignore[attr-defined]  # noqa: SLF001
            call_id=entry.call_id,
            occurred_at=authority._closed_at,  # noqa: SLF001
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id=entry.call_control_id,
                telnyx_call_leg_id=entry.call_leg_id,
                telnyx_call_session_id=entry.call_session_id,
                status=authority.status,
                disclosure_state="failed",
                started_at=started_at,
                ended_at=authority._closed_at,  # noqa: SLF001
                end_reason=authority.reason,
                retention_until=started_at
                + self._registry._retention_delta,  # type: ignore[attr-defined]  # noqa: SLF001
            ),
        )
        while True:
            try:
                await self._writer.commit_control(
                    PersistenceCommand("outbox", {"operation": operation}, None)
                )
            except asyncio.CancelledError:
                continue
            except BaseException:
                self._registry._note_terminal_failure(
                    "terminal_persistence_failed"
                )
            return


__all__ = ["ProcessSessionFactory", "ProcessTaskRegistrar"]
