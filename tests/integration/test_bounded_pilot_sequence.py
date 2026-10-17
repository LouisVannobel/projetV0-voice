"""Finite pilot policy against native V2 admission/finalization, with offline peers.

The construction below follows test_v2_final_call_freeze.final_case, but retains
one real supervisor/registry/writer across distinct calls. It does not exercise
the Pipecat audio pipeline or prove live provider acceptance.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from projetv0_voice.admission import CallAdmissionRejected, CallRegistry
from projetv0_voice.audio_contract import VoiceOperationV2
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.lifecycle import RuntimeSupervisor
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.models import CallUpsertPayloadV1, DisclosureEvidenceV1, MessageResultV1
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.pipeline import FirstFailure
from projetv0_voice.session import CallIdentity, CallSession, TurnRecorder, _TerminalOutcome
from projetv0_voice.session_factory import _RegistryTerminalizer
from projetv0_voice.telnyx.call_control import CallControlResult
from tests.contract.test_postgres_sink import sink_with_rows
from tests.integration.test_fixed_v2_process import KEY, NOW, WORKSPACE, fixed_policy, pin
from tests.unit.test_sparra_admission import DID, ControlledProvider, event

RUN = UUID("00000000-0000-4000-8000-000000000018")
PROFILE = b"p" * 32


class PilotProvider(ControlledProvider):
    async def aclose(self):
        return None


class FinalOwner(SimpleNamespace):
    __hash__ = object.__hash__


def observed(control, kind="call.initiated", *, instant=NOW):
    return event(kind, occurred_at=instant, call_control_id=control,
                 call_leg_id=f"{control}-leg", call_session_id=f"{control}-session")


def used(case):
    with sqlite3.connect(case.path) as database:
        return database.execute(
            "SELECT total_calls,used_calls,profile_sha256 FROM qualification_runs WHERE run_id=?",
            (str(RUN),),
        ).fetchone()


async def finalize(case, value, *, duplicate=False):
    resolution = await (case.registry.resolve_duplicate_webhook(value) if duplicate
                        else case.registry.resolve_webhook(value))
    return await case.supervisor.start_webhook_finalization(
        value, resolution, "duplicate" if duplicate else "first",
    ).wait()


@asynccontextmanager
async def pilot_case(tmp_path, *, clock=None, failpoint=None, expires_at=None):
    current = [NOW] if clock is None else clock
    path = tmp_path / "pilot.sqlite"
    writer = PersistenceWriter(path, CryptoKeyring({1: KEY}, active_version=1),
        contract_version=2, utcnow=lambda: current[0], failpoint=failpoint,
        process_agent_id="fixture", process_deployment_id="fixture")
    sink, pool, connection, _factory = sink_with_rows([])
    provider, begins, sessions = PilotProvider(), [], []

    async def begin(deployment, call_id, routing):
        begins.append(call_id)
        connection.rows = [(pin(call_id, routing).model_dump(mode="json"),)]
        return await sink.begin_call_v2(deployment, call_id, routing)

    registry = CallRegistry(writer=writer, call_control=provider, tenant_id=str(WORKSPACE),
        agent_id="fixture", deployment_id="fixture", capacity=1, lease_ttl_seconds=30,
        original_call_limit_seconds=300, stream_url="wss://fixture.invalid/media",
        retention_days=30, utcnow=lambda: current[0], monotonic=lambda: 100.0,
        sparra=fixed_policy(), called_did=DID, begin_call=begin, candidate_run_id=RUN,
        admission_not_before=NOW, admission_expires_at=expires_at or NOW + timedelta(minutes=30),
        qualification_observer=lambda state: supervisor.observe_qualification_state(state))
    # Match the existing lifecycle test's real supervisor, with no relay task:
    # durable publication is checked separately from remote SQL delivery.
    supervisor = RuntimeSupervisor(writer=writer, call_control=provider, registry=registry,
        candidate_run_id=RUN, candidate_total_calls=5, candidate_profile_sha256=PROFILE,
        deployment_id="fixture", retention_days=30, metrics=RuntimeMetrics.in_memory(),
        utcnow=lambda: current[0], loop_interval_seconds=0.05)
    case = SimpleNamespace(path=path, writer=writer, registry=registry, supervisor=supervisor,
        provider=provider, begins=begins, sessions=sessions, clock=current, pool=pool)
    try:
        await supervisor.startup()
        await writer.assert_sparra_compatible()
        yield case
    finally:
        for session in sessions:
            session._recorder.close()
            session._metric_lease.finish("failed")
        await supervisor.aclose()
        await sink.close()


async def session_for(case, control, *, declined=False):
    assert (await finalize(case, observed(control, "call.answered",
                                         instant=case.clock[0]))).status_code == 200
    admitted = await case.registry.snapshot(control)
    claim = await case.registry.claim_once(call_control_id=control,
        token_digest=admitted.token_digest, abort_target_publisher=lambda _target: True,
        abort_target_clearer=lambda _target: None)

    async def wait():
        return None

    owner = FinalOwner(_task=asyncio.current_task(), _phase="constructing", _session=None,
        _terminal_capability=None, request_drain=lambda _reason: None, wait=wait)
    grant = await case.registry.consume_claim_for_construction(claim, f"{control}-stream",
                                                              owner, owner._task)
    assert grant is not None
    identity = CallIdentity(call_id=grant.call_id, generation=grant.generation,
        lease_claim=grant.lease_claim, deployment_id=grant.deployment_id,
        telnyx_call_control_id=grant.telnyx_call_control_id,
        telnyx_call_leg_id=grant.telnyx_call_leg_id,
        telnyx_call_session_id=grant.telnyx_call_session_id, stream_id=grant.stream_id,
        started_at=grant.started_at, retention_until=grant.retention_until,
        routing=grant.routing, begin_snapshot=grant.begin_snapshot)
    evidence = DisclosureEvidenceV1(schema_version=1, started_at=NOW, completed_at=NOW,
                                   failed_at=None, input_gate_opened_at=NOW)
    active = VoiceOperationV2(schema_version=2, operation_id=uuid4(), deployment_id="fixture",
        call_id=grant.call_id, occurred_at=NOW, kind="call.upsert", payload=CallUpsertPayloadV1(
            telnyx_call_control_id=control, telnyx_call_leg_id=f"{control}-leg",
            telnyx_call_session_id=f"{control}-session", status="active",
            disclosure_state="completed", started_at=NOW, ended_at=None, end_reason=None,
            retention_until=grant.retention_until, disclosure_evidence=evidence))
    await case.writer.publish_control_v2(active, generation=grant.generation.generation)
    metrics = RuntimeMetrics.in_memory()
    recorder = TurnRecorder(identity=identity, writer=case.writer, keyring=case.writer._keyring,
        first_failure=FirstFailure(), runtime_metrics=metrics, utcnow=lambda: case.clock[0])
    recorder.record_user("Je ne donne pas mes coordonnées." if declined else
                         "Quels sont vos horaires ?", NOW.isoformat())
    await case.writer.wait_until_idle()
    retained = await case.writer.read_retained_call(grant.call_id)

    async def inference(_context, **_options):
        return json.dumps({"result": {"schema_version": 1, "quality": "partial",
            "category": "information", "summary": "Demande d'horaires.",
            "next_action": "Informer.", "contact": {"name": None, "callback_e164": None,
                "preference": "Coordonnées déclinées." if declined else None,
                "callback_source": "missing", "callback_confirmed": False},
            "evidence": [{"turn_id": str(retained.turns[0].turn_id), "role": "user"}],
            "request_confirmed": False}})

    async def close_services():
        return None

    session = CallSession.__new__(CallSession)
    session._identity, session._writer = identity, case.writer
    session._controller, session._recorder = SimpleNamespace(evidence=evidence), recorder
    session._services = SimpleNamespace(llm=SimpleNamespace(run_inference=inference),
                                       aclose=close_services)
    session._no_new_ai = session._result_inference_fenced = False
    session._partial_result = session._result_inference_task = session._terminal_publication = None
    session._cleanup_phase_timeout_seconds = 2
    session._utcnow, session._uuid_factory = lambda: case.clock[0], uuid4
    session._registry_terminalizer = _RegistryTerminalizer(case.registry, grant, owner)
    session._metric_lease, session._runtime_metrics = metrics.begin_call(), metrics
    case.sessions.append(session)
    return session


async def finish(session):
    await session._prepare_partial_result()
    await session._finish_durable_boundaries(first_failure=FirstFailure(),
        terminal_outcome=_TerminalOutcome("closed"), disclosure_completed=True)


async def assert_next_blocked(case, control, *, allow_debit=False):
    before_actions = list(case.provider.actions)
    before_begins, before_used = len(case.begins), used(case)
    try:
        disposition = await finalize(case, observed(control, instant=case.clock[0]))
    except CallAdmissionRejected:
        disposition = None
    assert case.provider.actions == before_actions and len(case.begins) == before_begins, (
        "immediate next arrival reached Answer/begin while previous outcome was unresolved",
        case.provider.actions, disposition,
    )
    after_used = used(case)
    if allow_debit:
        # A fail-closed arrival may itself consume quota; never refund it merely
        # because Answer was prevented by the native process failure observer.
        assert after_used[0] == before_used[0] and after_used[2] == before_used[2]
        assert before_used[1] <= after_used[1] <= before_used[1] + 1
    else:
        assert after_used == before_used
        assert await case.registry.snapshot(control) is None


@pytest.mark.asyncio
async def test_five_sequential_v2_calls_finish_after_quota_exhaustion(tmp_path):
    async with pilot_case(tmp_path) as case:
        calls, publications = set(), set()
        for number in range(5):
            control = f"pilot-{number}"
            initiated = observed(control)
            assert (await finalize(case, initiated)).status_code == 200
            assert used(case) == (5, number + 1, PROFILE)
            assert (await finalize(case, initiated, duplicate=True)).status_code == 200
            assert used(case) == (5, number + 1, PROFILE)
            readiness = case.supervisor.readiness_snapshot()
            assert readiness.qualification_state_valid is (number < 4)
            session = await session_for(case, control, declined=number == 2)
            await assert_next_blocked(case, f"overlap-{number}")
            if number == 4:
                # No relay is configured, so global readiness was already false;
                # only the qualification predicate transitions in this fixture.
                assert not readiness.ready and not readiness.qualification_state_valid
                assert await case.registry.live_call_count() == 1
            await finish(session)
            frozen = await case.writer.read_frozen_call_publication_v2(session._identity.call_id,
                generation=session._identity.generation.generation)
            assert frozen is not None and frozen.payload.message_result is not None
            if number == 2:
                envelope = frozen.payload.message_result
                result = MessageResultV1.model_validate_json(case.writer._keyring.decrypt(
                    EncryptedValue(envelope.key_version, base64.b64decode(envelope.nonce_b64),
                                   base64.b64decode(envelope.ciphertext_b64)),
                    aad=f"result:{session._identity.call_id}".encode("ascii")))
                assert result.contact.callback_e164 is None
                assert result.contact.preference == "Coordonnées déclinées."
            calls.add(frozen.call_id)
            publications.add(frozen.operation_id)
            assert await case.registry.live_call_count() == 0
        await assert_next_blocked(case, "sixth")
        assert len(calls) == len(publications) == 5
        assert [value[0] for value in case.provider.actions].count("answer") == 5
        assert [value[0] for value in case.provider.actions].count("streaming") == 5
        assert [value[0] for value in case.provider.actions].count("hangup") == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["result", "commit", "hangup"])
async def test_immediate_arrival_during_unresolved_final_boundary(tmp_path, monkeypatch, boundary):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def failpoint(name):
        nonlocal armed
        if boundary == "commit" and armed and name == "after_mutation_before_commit":
            armed = False
            entered.set()
            await release.wait()

    async with pilot_case(tmp_path, failpoint=failpoint) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200

        async def complete_prior():
            nonlocal armed
            session = await session_for(case, "prior")
            if boundary == "result":
                native = session._services.llm.run_inference

                async def pending(*args, **kwargs):
                    entered.set()
                    await release.wait()
                    return await native(*args, **kwargs)

                monkeypatch.setattr(session._services.llm, "run_inference", pending)
            if boundary == "hangup":
                native = case.provider.hangup

                async def pending_hangup(*args, **kwargs):
                    entered.set()
                    await release.wait()
                    return await native(*args, **kwargs)

                monkeypatch.setattr(case.provider, "hangup", pending_hangup)
            await session._prepare_partial_result()
            armed = boundary == "commit"
            await session._finish_durable_boundaries(first_failure=FirstFailure(),
                terminal_outcome=_TerminalOutcome("closed"), disclosure_completed=True)

        completing = asyncio.create_task(complete_prior())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await assert_next_blocked(case, "immediate")
        finally:
            release.set()
            await asyncio.wait_for(completing, 2)
        assert used(case) == (5, 1, PROFILE)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["outcome_unknown", "exception"])
async def test_immediate_arrival_after_unknown_final_hangup_is_not_paid(
    tmp_path, monkeypatch, outcome,
):
    async with pilot_case(tmp_path) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200
        session = await session_for(case, "prior")

        async def unknown(control, *, command_id, client_state=None):
            case.provider.actions.append(("hangup", control, command_id))
            if outcome == "exception":
                raise OSError("offline-hangup-outcome-unknown")
            return CallControlResult("outcome_unknown")

        monkeypatch.setattr(case.provider, "hangup", unknown)
        await finish(session)
        # No timer, polling interval, or guardian observation before this arrival.
        assert case.registry.internal_failure_code == "terminal_hangup_failed"
        await assert_next_blocked(case, "immediate", allow_debit=True)


@pytest.mark.asyncio
async def test_immediate_arrival_after_native_publication_rollback_is_not_paid(
    tmp_path, monkeypatch,
):
    armed = False

    def failpoint(name):
        if armed and name == "after_mutation_before_commit":
            raise OSError("offline-final-publication-rollback")

    async with pilot_case(tmp_path, failpoint=failpoint) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200
        session = await session_for(case, "prior")
        await session._prepare_partial_result()
        native = case.writer.freeze_call_publication_v2

        async def rollback(*args, **kwargs):
            nonlocal armed
            armed = True
            try:
                return await native(*args, **kwargs)
            finally:
                armed = False

        monkeypatch.setattr(case.writer, "freeze_call_publication_v2", rollback)
        await session._finish_durable_boundaries(first_failure=FirstFailure(),
            terminal_outcome=_TerminalOutcome("closed"), disclosure_completed=True)
        assert case.registry.internal_failure_code == "terminal_persistence_failed"
        assert await case.registry.live_call_count() == 1
        await assert_next_blocked(case, "immediate")
        with sqlite3.connect(case.path) as database:
            assert database.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)
        assert used(case) == (5, 1, PROFILE)


@pytest.mark.asyncio
async def test_known_summary_fallback_is_durable_before_next_admission(tmp_path, monkeypatch):
    async with pilot_case(tmp_path) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200
        session = await session_for(case, "prior")

        async def failure(*_args, **_kwargs):
            raise OSError("offline-summary-unavailable")

        monkeypatch.setattr(session._services.llm, "run_inference", failure)
        await finish(session)
        frozen = await case.writer.read_frozen_call_publication_v2(session._identity.call_id,
            generation=session._identity.generation.generation)
        assert frozen is not None and frozen.payload.message_result is None
        assert case.registry.internal_failure_code is None
        assert await case.registry.live_call_count() == 0
        assert (await finalize(case, observed("next"))).status_code == 200
        assert used(case) == (5, 2, PROFILE)


@pytest.mark.asyncio
async def test_fixed_window_rejects_before_start_without_spending(tmp_path):
    async with pilot_case(tmp_path, clock=[NOW - timedelta(microseconds=1)]) as case:
        await assert_next_blocked(case, "too-early")
        assert used(case) is None


@pytest.mark.asyncio
async def test_expiry_before_answered_prevents_delayed_streaming_without_refund(tmp_path):
    cutoff = NOW + timedelta(seconds=1)
    async with pilot_case(tmp_path, expires_at=cutoff) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200
        assert [value[0] for value in case.provider.actions] == ["answer"]
        case.clock[0] = cutoff
        with suppress(CallAdmissionRejected):
            await finalize(case, observed("prior", "call.answered", instant=cutoff))
        assert not any(value[0] == "streaming" for value in case.provider.actions)
        assert used(case) == (5, 1, PROFILE)
        await assert_next_blocked(case, "after-expiry")


@pytest.mark.asyncio
async def test_fixed_expiry_with_active_last_call_preserves_its_final_publication(tmp_path):
    cutoff = NOW + timedelta(seconds=1)
    async with pilot_case(tmp_path, expires_at=cutoff) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200
        session = await session_for(case, "prior")
        case.clock[0] = cutoff
        await assert_next_blocked(case, "expired")
        await finish(session)
        frozen = await case.writer.read_frozen_call_publication_v2(session._identity.call_id,
            generation=session._identity.generation.generation)
        assert frozen is not None and frozen.payload.ended_at == cutoff
        assert await case.registry.live_call_count() == 0
        assert used(case) == (5, 1, PROFILE)


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_owner", ["reaper", "supervisor"])
async def test_fixed_cutoff_drains_real_running_session_without_manual_finish(
    tmp_path, expiry_owner,
):
    from pipecat.runner.types import TelnyxCallData

    from projetv0_voice.telnyx.handshake import AuthenticatedTelnyxHandshake
    from tests.integration.test_call_session import _Transport
    from tests.integration.test_disclosure import _local_choice_wait
    from tests.integration.test_process_session_factory import (
        _AudioAdmission,
        _real_process_factory,
    )

    cutoff = NOW + timedelta(seconds=1)
    async with pilot_case(tmp_path, expires_at=cutoff) as case:
        assert (await finalize(case, observed("prior"))).status_code == 200
        assert (await finalize(case, observed("prior", "call.answered"))).status_code == 200
        admitted = await case.registry.snapshot("prior")
        claim = await case.registry.claim_once(call_control_id="prior",
            token_digest=admitted.token_digest, abort_target_publisher=lambda _target: True,
            abort_target_clearer=lambda _target: None)
        events = []
        metrics = RuntimeMetrics.in_memory()
        factory = _real_process_factory(registry=case.registry, writer=case.writer,
            metrics=metrics, events=events, registrar=case.supervisor.call_lifecycle_owners,
            pass_frames=True)
        factory._keyring = case.writer._keyring
        factory._profile = factory._profile.model_copy(update={"deployment_id": "fixture"})
        factory._manifest = factory._manifest.model_copy(update={
            "sparra": fixed_policy(), "dids": (DID,), "transcript_retention_days": 30})
        factory._session_factory = lambda **values: CallSession(
            **values, utcnow=lambda: case.clock[0])
        handshake = AuthenticatedTelnyxHandshake(
            call_data=TelnyxCallData(stream_id="prior-stream", call_id="prior",
                                    outbound_encoding="PCMU", to_number=DID, from_number=None),
            token_locator_id="telnyx-header-connected-v1", lease_claim=claim,
            transport=_Transport(events, echo_ack=True), audio_admission=_AudioAdmission())
        running = asyncio.create_task(factory.run(handshake), name="bounded-cutoff-session")
        try:
            def active():
                entry = case.registry._by_control.get("prior")
                session = None if entry is None else entry.session
                controller = None if session is None else session._controller
                return controller is not None and controller.is_active()

            await _local_choice_wait(active)
            await case.writer.wait_until_idle()
            assert await case.registry.live_call_count() == 1
            case.clock[0] = cutoff
            if expiry_owner == "reaper":
                assert await case.registry.reap_expired() == 1
            # The real supervisor already started both expiry and reaper loops.
            # Observe up to four native 5s cleanup phases without cancelling the
            # owner early; report stacks before any diagnostic cleanup on failure.
            done, pending = await asyncio.wait((running,), timeout=20)
            if pending:
                stacks = []
                for task in asyncio.all_tasks():
                    name = task.get_name()
                    if any(part in name for part in (
                        "voice-", "call-", "session", "writer", "runner",
                    )):
                        frames = [
                            f"{frame.f_code.co_filename}:{frame.f_code.co_name}:{frame.f_lineno}"
                            for frame in task.get_stack()
                        ]
                        stacks.append({"name": name, "done": task.done(),
                                       "cancelling": task.cancelling(), "stack": frames})
                pytest.fail("cutoff_native_tasks_pending:" + json.dumps(stacks))
            assert running in done
            await running
            frozen = await case.writer.read_frozen_call_publication_v2(admitted.call_id,
                                                                     generation=admitted.generation)
            assert frozen is not None and frozen.payload.ended_at == cutoff
            assert frozen.payload.end_reason == "token_deadline"
            assert frozen.payload.status == "failed"
            assert frozen.payload.disclosure_state == "completed"
            assert frozen.payload.message_result is None
            with sqlite3.connect(case.path) as database:
                assert database.execute("SELECT count(*) FROM sparra_publications WHERE call_id=?",
                                        (str(admitted.call_id),)).fetchone() == (1,)
            assert await case.registry.live_call_count() == 0
            assert [value[0] for value in case.provider.actions].count("hangup") == 1
            await assert_next_blocked(case, "after-cutoff")
            assert used(case) == (5, 1, PROFILE)
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            await metrics.aclose()


@pytest.mark.asyncio
async def test_pending_quota_and_fixed_expiry_survive_writer_registry_restart(tmp_path):
    cutoff = NOW + timedelta(minutes=30)
    async with pilot_case(tmp_path, expires_at=cutoff) as case:
        first = observed("prior")
        assert (await finalize(case, first)).status_code == 200
        # Interrupt before START through the supervisor's controlled shutdown;
        # this does not claim to simulate a process crash or carrier disconnect.
        assert used(case) == (5, 1, PROFILE)
    # Restart promptly: this fixture intentionally leaves publications in its
    # offline outbox; jumping 30 minutes would correctly trip the outbox-age gate.
    clock = [NOW + timedelta(seconds=1)]
    async with pilot_case(tmp_path, clock=clock, expires_at=cutoff) as restored:
        assert used(restored) == (5, 1, PROFILE)
        assert (await finalize(restored, first, duplicate=True)).status_code == 200
        assert used(restored) == (5, 1, PROFILE)
        clock[0] = cutoff
        await assert_next_blocked(restored, "after-expiry")
        for limit, fingerprint in ((4, PROFILE), (5, b"q" * 32)):
            with pytest.raises(PersistenceError, match="qualification_run_conflict"):
                await restored.writer.qualification_run_consumed(
                    RUN, total_calls=limit, profile_sha256=fingerprint)
