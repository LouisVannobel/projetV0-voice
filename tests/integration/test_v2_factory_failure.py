"""Native factory ownership and SQLite2 publication; all resources are offline fixtures."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from pipecat.runner.types import TelnyxCallData
from test_fixed_v2_process import NOW, registry_case
from test_process_session_factory import (
    _AudioAdmission,
    _factory,
    _metric_points,
    _OfflineTransport,
    _real_process_factory,
    _Registrar,
)
from test_sparra_admission import DID, committed, event
from test_sparra_admission import snapshot as legacy_snapshot

from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.persistence.commands import decode_operation_v2, operation_aad_from_metadata
from projetv0_voice.session import CallSession
from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake


@asynccontextmanager
async def factory_case(tmp_path, *, activation=False, failpoint=None, clock=None):
    current = [NOW] if clock is None else clock
    async with registry_case(tmp_path, utcnow=lambda: current[0], failpoint=failpoint,
                             deployment_id="deployment-a" if activation else "fixture") as case:
        assert (await committed(case.registry, case.writer,
                                event(occurred_at=NOW))).status_code == 200
        assert (await committed(case.registry, case.writer,
                                event("call.answered", occurred_at=NOW))).status_code == 200
        admitted = await case.registry.snapshot("original")
        claim = await case.registry.claim_once(call_control_id="original",
            token_digest=admitted.token_digest, abort_target_publisher=lambda _target: True,
            abort_target_clearer=lambda _target: None)
        assert claim is not None
        events, seen, capabilities = [], [], []
        metrics = RuntimeMetrics.in_memory()
        registrar = _Registrar()
        if activation:
            factory = _real_process_factory(registry=case.registry, writer=case.writer,
                metrics=metrics, events=events, registrar=registrar)
        else:
            factory = _factory(registry=case.registry, writer=case.writer, registrar=registrar,
                runtime_metrics=metrics, events=events, session_error=True)
        native_persist = factory._persist_terminal_call

        async def observe(grant, authority):
            seen.append((grant, authority))
            capabilities.append(authority._entry.terminal_capability)
            await native_persist(grant, authority)

        factory._persist_terminal_call = observe
        handshake = AuthenticatedTelnyxHandshake(call_data=TelnyxCallData(
            stream_id="stream-original", call_id="original", outbound_encoding="PCMU",
            to_number=DID, from_number=None),
            token_locator_id="telnyx-header-connected-v1", lease_claim=claim,
            transport=_OfflineTransport(events), audio_admission=_AudioAdmission())
        case.factory, case.handshake, case.events = factory, handshake, events
        case.seen, case.metrics, case.registrar, case.clock = seen, metrics, registrar, current
        case.capabilities = capabilities
        try:
            yield case
        finally:
            print("factory-v2-progress:fixture-owner-cleanup", flush=True)
            for task in registrar.tasks:
                if not task.done():
                    task.cancel()
            if registrar.tasks:
                done, pending = await asyncio.wait(registrar.tasks, timeout=2)
                await asyncio.gather(*done, return_exceptions=True)
                if pending:
                    metadata = getattr(case, "diagnostic_metadata", "unjoined_owned_factory")
                    raise AssertionError(f"factory_cleanup_pending:{metadata}")
            metrics._provider.shutdown(timeout_millis=10000)
            print("factory-v2-progress:fixture-metrics-closed", flush=True)


def publication_bytes(case):
    with sqlite3.connect(case.path) as db:
        return db.execute("SELECT op_id,deployment_id,key_version,nonce,ciphertext "
                          "FROM sparra_publications").fetchall()


async def bounded_factory_action(case, stage, action):
    """Diagnose these owned native consumers without printing stack locals or payloads."""
    print(f"factory-v2-progress:{stage}:entered", flush=True)
    route = asyncio.create_task(action)
    done, _pending = await asyncio.wait({route}, timeout=5)
    if done:
        result = await route
        print(f"factory-v2-progress:{stage}:completed", flush=True)
        return result
    stacks = tuple(
        tuple((Path(frame.f_code.co_filename).name, frame.f_code.co_name)
              for frame in task.get_stack(limit=5))
        for task in asyncio.all_tasks() if not task.done()
    )
    case.diagnostic_metadata = (stage, stacks)
    print(f"factory-v2-progress:{stage}:timeout:{stacks}", flush=True)
    owned = {route, *case.registrar.tasks}
    for task in owned:
        if not task.done():
            task.cancel()
    done, pending = await asyncio.wait(owned, timeout=2)
    await asyncio.gather(*done, return_exceptions=True)
    raise AssertionError(f"factory_action_timeout:{stage}:{stacks}:pending={bool(pending)}")


@pytest.mark.asyncio
async def test_factory_v2_constructor_failure_real_reverse_cleanup_and_one_authority(tmp_path):
    async with factory_case(tmp_path) as case:
        await case.factory.run(case.handshake)
        assert not case.writer.is_degraded
        grant, authority = case.seen[0]
        row = publication_bytes(case)[0]
        canonical = case.writer._keyring.decrypt(EncryptedValue(*row[2:5]),
            aad=operation_aad_from_metadata({"schema_version": 2, "operation_id": row[0],
                "deployment_id": row[1], "call_id": str(grant.call_id), "kind": "call.upsert"}))
        decoded = decode_operation_v2(canonical)
        assert "message_result" not in decoded.payload.model_fields_set
        frozen = await case.writer.read_frozen_call_publication_v2(grant.call_id,
            generation=grant.generation.generation)
        assert frozen.schema_version == 2 and frozen.operation_id == authority.completion_token
        assert frozen.occurred_at == authority._closed_at and frozen.payload.status == "failed"
        assert frozen.payload.end_reason == "session_construction_failed"
        assert frozen.payload.retention_until == grant.retention_until
        assert frozen.payload.message_result is None
        assert [name for name in case.events if name.endswith("-cleanup")] == [
            "recording-cleanup", "tts-cleanup", "llm-cleanup", "stt-cleanup"]
        assert case.events.count("stt-client-close") == case.events.count("llm-client-close") == 1
        assert len(publication_bytes(case)) == 1 and await case.registry.live_call_count() == 0
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        totals = _metric_points(case.metrics, "projetv0.voice.calls.total")
        assert len(totals) == 1 and totals[0].attributes["session"] == "failed"


@pytest.mark.asyncio
async def test_factory_v2_activation_failure_real_session_owns_cleanup_once(tmp_path, monkeypatch):
    async with factory_case(tmp_path, activation=True) as case:
        sessions = []
        native_session = case.factory._session_factory

        def construct(**values):
            session = native_session(**values)
            assert isinstance(session, CallSession)
            sessions.append(session)
            return session

        async def fail_activation(*_args):
            raise RuntimeError("owned-activation-fault")

        monkeypatch.setattr(case.factory, "_session_factory", construct)
        monkeypatch.setattr(case.registry, "activate_session", fail_activation)
        await bounded_factory_action(case, "activation", case.factory.run(case.handshake))
        assert len(sessions) == 1 and not case.writer.is_degraded
        session = sessions[0]
        frozen = await case.writer.read_frozen_call_publication_v2(session._identity.call_id,
            generation=session._identity.generation.generation)
        assert frozen.schema_version == 2
        assert frozen.payload.end_reason == "session_construction_failed"
        before = publication_bytes(case)
        await session.aclose_unstarted()
        assert publication_bytes(case) == before and len(before) == 1
        assert case.events.count("stt-client-close") == case.events.count("llm-client-close") == 1
        assert all(case.events.count(f"{name}-cleanup") == 1 for name in ("stt", "llm", "tts"))
        assert await case.registry.live_call_count() == 0
        assert len(_metric_points(case.metrics, "projetv0.voice.calls.total")) == 1


@pytest.mark.asyncio
async def test_factory_v2_cancelled_terminal_commit_keeps_capacity_and_frozen_token(
    tmp_path, monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with factory_case(tmp_path, failpoint=hold) as case:
        native_freeze = case.writer.freeze_call_publication_v2

        async def arm_terminal(*args, **options):
            nonlocal armed
            armed = True
            return await native_freeze(*args, **options)

        monkeypatch.setattr(case.writer, "freeze_call_publication_v2", arm_terminal)
        route = asyncio.create_task(case.factory.run(case.handshake))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert publication_bytes(case) == [] and await case.registry.live_call_count() == 1
            route.cancel("first-owned-cancel")
            await asyncio.sleep(0)
            route.cancel("second-owned-cancel")
            assert await case.registry.live_call_count() == 1
            assert not any(action[0] == "hangup" for action in case.provider.actions)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(route, 2)
        grant, authority = case.seen[0]
        frozen = await case.writer.read_frozen_call_publication_v2(grant.call_id,
            generation=grant.generation.generation)
        assert frozen.operation_id == authority.completion_token
        assert frozen.occurred_at == authority._closed_at
        assert len(publication_bytes(case)) == 1 and await case.registry.live_call_count() == 0
        assert all(task.done() for task in case.registrar.tasks)
        assert [action[0] for action in case.provider.actions].count("hangup") == 1


@pytest.mark.asyncio
async def test_factory_v2_suppression_and_stale_capability_preserve_original_phone_cleanup(
    tmp_path, monkeypatch,
):
    for label in ("expiry", "erase"):
        directory = tmp_path / label
        directory.mkdir()
        async with factory_case(directory) as case:
            native_persist = case.factory._persist_terminal_call

            async def suppress_content(grant, authority, *, variant=label, persist=native_persist):
                if variant == "expiry":
                    case.clock[0] = grant.retention_until
                else:
                    await case.writer.erase_call_content(
                        grant.call_id, lease_token=uuid4(), now=NOW
                    )
                await persist(grant, authority)

            monkeypatch.setattr(case.factory, "_persist_terminal_call", suppress_content)
            await bounded_factory_action(case, f"suppression-{label}",
                                         case.factory.run(case.handshake))
            assert not case.writer.is_degraded and publication_bytes(case) == []
            grant, authority = case.seen[0]
            assert await case.registry.live_call_count() == 0
            assert [action[0] for action in case.provider.actions].count("hangup") == 1
            assert not await case.registry.complete_reserved_terminal(authority)
            with pytest.raises(RuntimeError, match="terminal_authority_unavailable"):
                await case.registry.reserve_or_read_terminal(grant, case.capabilities[0],
                    CallSession._terminal_proposal("closed"))
    directory = tmp_path / "unadmitted"
    directory.mkdir()
    async with registry_case(directory, reply=lambda _pin, call, routing: legacy_snapshot(
        call, routing
    )) as case:
        print("factory-v2-progress:unadmitted-finalizer-start", flush=True)
        observed = event(occurred_at=NOW)
        assert (await committed(case.registry, case.writer, observed)).status_code == 503
        assert len(case.begins) == 1
        original_call_id = case.begins[0][1]
        print("factory-v2-progress:unadmitted-begin-failure-finalized", flush=True)
        metrics = RuntimeMetrics.in_memory()
        registrar = _Registrar()
        factory = _factory(registry=case.registry, registrar=registrar, runtime_metrics=metrics,
                           events=[], writer=case.writer, session_error=True)
        case.registrar = registrar
        try:
            await bounded_factory_action(case, "suppression-unadmitted",
                factory.drain_call_by_id(original_call_id, "recording_required_error"))
            assert publication_bytes(case) == [] and not case.writer.is_degraded
            assert await case.registry.live_call_count() == 0
            assert not any(action[0] in {"answer", "streaming"} for action in case.provider.actions)
        finally:
            print("factory-v2-progress:unadmitted-metrics-cleanup", flush=True)
            metrics._provider.shutdown(timeout_millis=10000)
            print("factory-v2-progress:unadmitted-metrics-closed", flush=True)
