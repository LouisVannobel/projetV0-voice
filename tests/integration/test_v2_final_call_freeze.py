"""Actual registry authority and session final callback; SQL pool replies are offline data."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from test_audio_writer import owned
from test_disclosure import _TEST_RUNTIME_METRICS, _local_choice_wait
from test_fixed_v2_process import NOW, registry_case
from test_sparra_admission import committed, event

from projetv0_voice.audio_contract import VoiceOperationV2
from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.models import CallUpsertPayloadV1, DisclosureEvidenceV1, MessageResultV1
from projetv0_voice.persistence.commands import PersistenceError, canonical_operation_bytes
from projetv0_voice.persistence.relay import OutboxRelay
from projetv0_voice.pipeline import FirstFailure
from projetv0_voice.session import CallIdentity, CallSession, TurnRecorder, _TerminalOutcome
from projetv0_voice.session_factory import _RegistryTerminalizer


@asynccontextmanager
async def final_case(tmp_path, *, clock=None, failpoint=None, reply=None):
    current = [NOW] if clock is None else clock
    async with registry_case(tmp_path, utcnow=lambda: current[0], failpoint=failpoint,
                             reply=reply) as case:
        initiated = event(occurred_at=NOW, from_e164="+33102030407")
        assert (await committed(case.registry, case.writer, initiated)).status_code == 200
        assert (await committed(case.registry, case.writer,
                                event("call.answered", occurred_at=NOW))).status_code == 200
        admitted = await case.registry.snapshot("original")
        claim = await case.registry.claim_once(call_control_id="original",
            token_digest=admitted.token_digest, abort_target_publisher=lambda _target: True,
            abort_target_clearer=lambda _target: None)

        async def wait():
            return None

        owner = SimpleNamespace(_task=asyncio.current_task(), _phase="constructing", _session=None,
            _terminal_capability=None, request_drain=lambda _reason: None, wait=wait)
        grant = await case.registry.consume_claim_for_construction(
            claim, "stream", owner, owner._task
        )
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
        gate = VoiceOperationV2(schema_version=2, operation_id=uuid4(), deployment_id="fixture",
            call_id=grant.call_id, occurred_at=NOW, kind="call.upsert", payload=CallUpsertPayloadV1(
                telnyx_call_control_id="original", telnyx_call_leg_id="original-leg",
                telnyx_call_session_id="session", status="active", disclosure_state="completed",
                started_at=NOW, ended_at=None, end_reason=None,
                retention_until=grant.retention_until,
                disclosure_evidence=evidence))
        await case.writer.publish_control_v2(gate, generation=grant.generation.generation)
        failure = FirstFailure()
        recorder = TurnRecorder(identity=identity, writer=case.writer, keyring=case.writer._keyring,
            first_failure=failure, runtime_metrics=_TEST_RUNTIME_METRICS, utcnow=lambda: NOW)
        recorder.record_user("Pouvez-vous me rappeler ?", NOW.isoformat())
        await case.writer.wait_until_idle()
        retained = await case.writer.read_retained_call(grant.call_id)
        assert len(retained.turns) == 1 and failure.code is None
        requests = []

        async def inference(_context, **options):
            requests.append(options)
            return json.dumps({"schema_version": 1, "quality": "partial", "category": "callback",
                "summary": "Rappel demandé.", "next_action": "Rappeler.",
                "contact": {"name": None, "callback_e164": "+33102030407", "preference": None,
                    "callback_source": "provider", "callback_confirmed": False},
                "evidence": [{"turn_id": str(retained.turns[0].turn_id), "role": "user"}],
                "request_confirmed": False})

        async def close_services():
            return None

        session = CallSession.__new__(CallSession)
        session._identity, session._writer = identity, case.writer
        session._controller = SimpleNamespace(evidence=evidence)
        session._recorder = recorder
        session._services = SimpleNamespace(llm=SimpleNamespace(run_inference=inference),
                                           aclose=close_services)
        session._no_new_ai = session._result_inference_fenced = False
        session._partial_result = session._result_inference_task = None
        session._terminal_publication = None
        session._cleanup_phase_timeout_seconds = 2
        session._utcnow, session._uuid_factory = lambda: current[0], uuid4
        session._registry_terminalizer = _RegistryTerminalizer(case.registry, grant, owner)
        session._metric_lease = _TEST_RUNTIME_METRICS.begin_call()
        case.session, case.grant, case.requests, case.clock = session, grant, requests, current
        try:
            yield case
        finally:
            recorder.close()
            session._metric_lease.finish("failed")


def frozen_bytes(case):
    with sqlite3.connect(case.path) as db:
        return db.execute("SELECT op_id,deployment_id,key_version,nonce,ciphertext "
            "FROM sparra_publications WHERE call_id=?", (str(case.grant.call_id),)).fetchone()


async def reserve(case):
    return await case.session._registry_terminalizer.reserve_or_read(
        case.session._terminal_proposal("closed")
    )


def fresh_final(case, authority):
    return VoiceOperationV2(schema_version=2, operation_id=authority.completion_token,
        deployment_id="fixture", call_id=case.grant.call_id, occurred_at=authority._closed_at,
        kind="call.upsert", payload=CallUpsertPayloadV1(
            telnyx_call_control_id="original", telnyx_call_leg_id="original-leg",
            telnyx_call_session_id="session", status="closed", disclosure_state="completed",
            started_at=NOW, ended_at=authority._closed_at, end_reason=authority.reason,
            retention_until=case.grant.retention_until,
            disclosure_evidence=case.session._controller.evidence))


@pytest.mark.asyncio
async def test_final_call_v2_actual_off_session_result_and_issued_terminal_authority(tmp_path):
    async with final_case(tmp_path) as case:
        assert callable(getattr(case.writer, "freeze_call_publication_v2", None)), (
            "missing V2 final freeze"
        )
        await case.session._prepare_partial_result()
        authority = await reserve(case)
        await case.session._finish_durable_boundaries(first_failure=FirstFailure(),
            terminal_outcome=_TerminalOutcome("closed"), disclosure_completed=True)
        frozen = await case.writer.read_frozen_call_publication_v2(case.grant.call_id,
            generation=case.grant.generation.generation)
        assert frozen.schema_version == 2 and frozen.operation_id == authority.completion_token
        assert frozen.occurred_at == authority._closed_at
        assert frozen.payload.retention_until == NOW + timedelta(days=30)
        envelope = frozen.payload.message_result
        result = MessageResultV1.model_validate_json(case.writer._keyring.decrypt(EncryptedValue(
            envelope.key_version, base64.b64decode(envelope.nonce_b64),
            base64.b64decode(envelope.ciphertext_b64)),
            aad=f"result:{case.grant.call_id}".encode("ascii")))
        assert result.contact.callback_e164 == "+33102030407"
        assert result.evidence[0].role == "user" and len(case.requests) == 1
        assert await case.registry.live_call_count() == 0
        assert [action[0] for action in case.provider.actions].count("hangup") == 1


@pytest.mark.asyncio
async def test_final_call_v2_real_codec_ack_restart_reuses_frozen_result_and_cipher(
    tmp_path, monkeypatch,
):
    async with final_case(tmp_path) as case:
        assert callable(getattr(case.writer, "freeze_call_publication_v2", None)), (
            "missing V2 final freeze"
        )
        await case.session._prepare_partial_result()
        authority = await reserve(case)
        base = fresh_final(case, authority)
        frozen = await case.writer.freeze_call_publication_v2(base, case.session._partial_result,
            generation=case.grant.generation.generation, provider_callback="+33102030407",
            result_permitted=lambda: True)
        before = frozen_bytes(case)
        native_ingest = case.sink.ingest_v2

        async def ingest(operation):
            case.connection.rows = [({"schema_version": 2, "status": "applied",
                "operation_id": str(operation.operation_id), "payload_sha256":
                hashlib.sha256(canonical_operation_bytes(operation)).hexdigest()},)]
            await native_ingest(operation)

        async def unexpected():
            pytest.fail("known final-call codec receipt degraded native relay")

        monkeypatch.setattr(case.sink, "ingest_v2", ingest)
        relay = OutboxRelay(case.writer, case.sink, utcnow=lambda: NOW,
                            on_degraded=unexpected, drain=unexpected)
        assert (await relay.run_once()).status == "delivered"
        assert frozen_bytes(case) == before and case.pool.active == 0
        await case.session._prepare_partial_result()
        assert len(case.requests) == 1 and case.session._terminal_publication == frozen
        path, generation, call = case.path, case.grant.generation.generation, case.grant.call_id
    async with owned(
        path, contract_version=2, utcnow=lambda: NOW,
        process_agent_id="fixture", process_deployment_id="fixture",
    ) as (writer, _keyring):
        assert await writer.read_frozen_call_publication_v2(call, generation=generation) == frozen
        changed = base.model_copy(update={"operation_id": uuid4()})
        assert await writer.freeze_call_publication_v2(changed, None, generation=generation,
            provider_callback=None, result_permitted=lambda: False) == frozen
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT op_id,deployment_id,key_version,nonce,ciphertext "
                "FROM sparra_publications").fetchone() == before


@pytest.mark.asyncio
async def test_final_call_v2_provenance_generation_and_queued_result_fence(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    async with final_case(tmp_path, failpoint=hold) as case:
        assert callable(getattr(case.writer, "freeze_call_publication_v2", None)), (
            "missing V2 final freeze"
        )
        await case.session._prepare_partial_result()
        base = fresh_final(case, await reserve(case))
        generation = case.grant.generation.generation
        with pytest.raises(PersistenceError):
            await case.writer.freeze_call_publication_v2(base, None, generation=UUID(int=999),
                provider_callback=None, result_permitted=lambda: True)
        invalid = case.session._partial_result.model_copy(update={"evidence": [
            case.session._partial_result.evidence[0].model_copy(update={"role": "assistant"})]})
        with pytest.raises(PersistenceError):
            await case.writer.freeze_call_publication_v2(base, invalid, generation=generation,
                provider_callback="+33102030407", result_permitted=lambda: True)
        assert frozen_bytes(case) is None and not case.writer.is_degraded
        armed = True
        control = case.writer.submit_webhook(
            receipt={"event_id": "held-final-control", "event_type": "fixture.control",
                "call_control_id": None, "occurred_at": NOW, "received_at": NOW,
                "semantic_fingerprint_sha256": b"q" * 32}, lease=None, operation=None,
        )
        await asyncio.wait_for(entered.wait(), 2)
        publishing = asyncio.create_task(case.writer.freeze_call_publication_v2(base,
            case.session._partial_result, generation=generation, provider_callback="+33102030407",
            result_permitted=lambda: not case.session._result_inference_fenced))
        try:
            await _local_choice_wait(lambda: case.writer.queue_size == 1)
            case.session.stop_result_inference()
        finally:
            release.set()
        await control.wait()
        frozen = await publishing
        assert frozen.payload.message_result is None
        assert frozen.operation_id == base.operation_id and not case.writer.is_degraded


@pytest.mark.asyncio
async def test_final_call_v2_expiry_erase_and_cancelled_commit_keep_original_authority(tmp_path):
    for label in ("expiry", "erase"):
        directory = tmp_path / label
        directory.mkdir()
        async with final_case(directory) as case:
            assert callable(getattr(case.writer, "freeze_call_publication_v2", None)), (
                "missing V2 final freeze"
            )
            authority = await reserve(case)
            if label == "expiry":
                case.clock[0] = case.grant.retention_until
            else:
                await case.writer.erase_call_content(
                    case.grant.call_id, lease_token=uuid4(), now=NOW
                )
            await case.session._finish_durable_boundaries(first_failure=FirstFailure(),
                terminal_outcome=_TerminalOutcome("closed"), disclosure_completed=True)
            assert frozen_bytes(case) is None and await case.registry.live_call_count() == 0
            facts = await case.writer.read_call_lifecycle(case.grant.call_id)
            assert facts.original_ended_at in (None, authority._closed_at)
    entered, release = asyncio.Event(), asyncio.Event()
    armed = False

    async def hold(name):
        nonlocal armed
        if name == "after_mutation_before_commit" and armed:
            armed = False
            entered.set()
            await release.wait()

    directory = tmp_path / "cancelled"
    directory.mkdir()
    async with final_case(directory, failpoint=hold) as case:
        await case.session._prepare_partial_result()
        authority = await reserve(case)
        armed = True
        committing = asyncio.create_task(case.session._commit_authoritative_terminal_call(
            status=authority.status, reason=authority.reason, disclosure_completed=True,
            operation_id=authority.completion_token, ended_at=authority._closed_at))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert frozen_bytes(case) is None
            committing.cancel()
        finally:
            release.set()
        assert await asyncio.wait_for(committing, 2)
        before = frozen_bytes(case)
        assert before[0] == str(authority.completion_token)
        assert await case.session._commit_authoritative_terminal_call(status=authority.status,
            reason=authority.reason, disclosure_completed=True,
            operation_id=authority.completion_token, ended_at=authority._closed_at)
        assert frozen_bytes(case) == before and len(case.requests) == 1
