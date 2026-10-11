from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from projetv0_voice import config
from projetv0_voice.admission import CallAdmissionRejected, CallRegistry
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import (
    BeginCallSnapshotV1,
    CallUpsertPayloadV1,
    DisclosureEvidenceV1,
    VoiceOperationV1,
)
from projetv0_voice.persistence.commands import PersistenceCommand, canonical_operation_bytes
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.telnyx.call_control import CallControlResult
from projetv0_voice.telnyx.webhooks import VerifiedWebhook

NOW = datetime(2026, 10, 1, 10, 0, 0, 123456, tzinfo=UTC)
DID = "+33102030405"
TARGET = "+33102030406"


def policy():
    return config.SparraManifestV1(
        schema_version=1,
        connection_id="fixture-connection",
        original_forward_line_e164=None,
        qualified_transfer_destination_e164=TARGET,
    )


def event(kind="call.initiated", **updates):
    values = dict(
        event_id=str(uuid4()),
        event_type=kind,
        occurred_at=NOW,
        call_control_id="original",
        call_leg_id="original-leg",
        call_session_id="session",
        recording_id=None,
        stream_id=None,
        client_state=None,
        recording_started_at=None,
        recording_ended_at=None,
        recording_channels=None,
        semantic_fingerprint_sha256=b"x" * 32,
        direction="incoming" if kind == "call.initiated" else None,
        call_state="parked"
        if kind == "call.initiated"
        else "answered"
        if kind == "call.answered"
        else None,
        connection_id="fixture-connection",
        to_e164=DID,
        from_e164=None,
    )
    values.update(updates)
    return VerifiedWebhook(**values)


class ControlledProvider:
    def __init__(self):
        self.actions = []
        self.transfer_entered = asyncio.Event()
        self.transfer_release = asyncio.Event()

    async def answer(self, control, *, command_id):
        self.actions.append(("answer", control, command_id))
        return CallControlResult("accepted")

    async def start_streaming(self, control, request, *, command_id):
        self.actions.append(("streaming", control, command_id))
        return CallControlResult("accepted")

    async def hangup(self, control, *, command_id, client_state=None):
        self.actions.append(("hangup", control, command_id))
        return CallControlResult("accepted")

    async def transfer(self, control, request, *, command_id):
        self.actions.append(("transfer", control, command_id, request))
        self.transfer_entered.set()
        await self.transfer_release.wait()
        return CallControlResult("accepted")


def snapshot(call_id, routing):
    return BeginCallSnapshotV1(
        schema_version=1,
        call_id=call_id,
        configuration_revision=1,
        knowledge=dict(
            business_name="Garage",
            sector="garage",
            opening_hours="",
            services="",
            prices="",
            faq="",
            instructions="",
        ),
        transfer_destination=TARGET,
        retention_until=routing.admitted_at + timedelta(days=30),
    )


async def start(
    tmp_path, begin=None, *, utcnow=None, monotonic=None, failpoint=None,
    sparra=True, writer_utcnow=None, candidate_run_id=None,
):
    writer = PersistenceWriter(
        tmp_path / "voice.sqlite", CryptoKeyring({1: bytes(range(32))}, active_version=1),
        failpoint=failpoint,
        utcnow=writer_utcnow or utcnow or (lambda: NOW),
    )
    worker = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    if sparra:
        await writer.assert_sparra_compatible()
    provider = ControlledProvider()

    async def default_begin(deployment, call_id, routing):
        return snapshot(call_id, routing)

    registry = CallRegistry(
        writer=writer,
        call_control=provider,
        tenant_id="fixture",
        agent_id="fixture",
        deployment_id="fixture",
        capacity=1,
        lease_ttl_seconds=30,
        stream_url="wss://fixture.invalid/media",
        retention_days=30,
        original_call_limit_seconds=300 if sparra else None,
        utcnow=utcnow or (lambda: NOW),
        monotonic=monotonic or (lambda: 100.0),
        sparra=policy() if sparra else None,
        called_did=DID if sparra else None,
        begin_call=(begin or default_begin) if sparra else None,
        candidate_run_id=candidate_run_id,
    )
    return registry, writer, worker, provider


async def committed(registry, writer, observed, *, duplicate=False, candidate_limit=1):
    resolved = await (
        registry.resolve_duplicate_webhook(observed)
        if duplicate
        else registry.resolve_webhook(observed)
    )
    effect = resolved.effect
    ticket = writer.submit_webhook(
        receipt=dict(
            event_id=observed.event_id,
            event_type=observed.event_type,
            call_control_id=observed.call_control_id,
            occurred_at=observed.occurred_at,
            received_at=NOW,
            semantic_fingerprint_sha256=observed.semantic_fingerprint_sha256,
        ),
        lease=None if effect is None else effect.lease,
        operation=None if effect is None else effect.operation,
        admission_facts=None if effect is None else effect.admission_facts,
        qualification_run_id=(
            registry.candidate_run_id
            if observed.event_type == "call.initiated" and not duplicate else None
        ),
        qualification_total_calls=candidate_limit,
        qualification_profile_sha256=b"p" * 32 if registry.candidate_run_id else None,
    )
    result = await ticket.wait()
    return await registry.reconcile_after_commit(observed, resolved, result)


async def persist_original_start(registry, writer, *, started_at=NOW):
    """Unit fixture for the active call fact normally published by the native session."""
    entry = registry._by_control["original"]
    facts = await writer.read_call_lifecycle(entry.call_id)
    operation = VoiceOperationV1(schema_version=1, operation_id=uuid4(),
        deployment_id="fixture", call_id=entry.call_id, occurred_at=started_at,
        kind="call.upsert", payload=CallUpsertPayloadV1(
            telnyx_call_control_id="original", telnyx_call_leg_id="original-leg",
            telnyx_call_session_id="session", status="active", disclosure_state="pending",
            started_at=started_at, ended_at=None, end_reason=None,
            retention_until=facts.retention_until))
    await writer.commit_control(PersistenceCommand("outbox", {"operation": operation}, None))


@pytest.mark.asyncio
@pytest.mark.parametrize("begin_outcome", ["success", "failure"])
async def test_finite_candidate_rejects_next_call_before_local_intent_begin_or_answer(
    tmp_path, begin_outcome
):
    from projetv0_voice.admission import ProcessLeaseAuthority

    run_id = uuid4()
    begins = []

    async def begin(deployment, call_id, routing):
        begins.append(call_id)
        if begin_outcome == "failure":
            raise RuntimeError("offline-begin-failed")
        return snapshot(call_id, routing)

    registry, writer, worker, provider = await start(tmp_path, begin, candidate_run_id=run_id)
    try:
        for number in range(2):
            observed = event(call_control_id=f"finite-{number}")
            await committed(registry, writer, observed, candidate_limit=2)
            admitted = await registry.snapshot(f"finite-{number}")
            if begin_outcome == "success":
                assert admitted is not None
                await committed(
                    registry, writer, event("call.answered", call_control_id=f"finite-{number}")
                )
                if number == 1:
                    # Consumed readiness closes new admission; the last call's media grant survives.
                    assert await registry.qualification_state() == "consumed"
                    assert await ProcessLeaseAuthority(registry).claim_once(
                        call_control_id=f"finite-{number}", token_digest=admitted.token_digest
                    ) is not None
                await committed(
                    registry, writer, event("call.hangup", call_control_id=f"finite-{number}")
                )
            else:
                # A failed begin and carrier end never refund the admitted unit.
                await committed(
                    registry, writer, event("call.hangup", call_control_id=f"finite-{number}")
                )
                await registry.wait_background()
        before_actions, before_begins = list(provider.actions), list(begins)
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            before = tuple(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                           for table in ("call_leases", "webhook_receipts", "outbox"))
        with pytest.raises(CallAdmissionRejected, match="^qualification_run_consumed$"):
            await registry.resolve_webhook(event(call_control_id="refused-next"))
        assert provider.actions == before_actions
        assert begins == before_begins and len(begins) == 2
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert tuple(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                         for table in ("call_leases", "webhook_receipts", "outbox")) == before
            assert connection.execute(
                "SELECT used_calls,total_calls FROM qualification_runs"
            ).fetchone() == (2, 2)
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


async def terminal_publications(writer):
    items = await writer.read_relay_batch(
        batch_size=100, now=datetime.now(UTC) + timedelta(seconds=1), lease_seconds=60
    )
    return [
        item.operation
        for item in items
        if isinstance(item.operation.payload, CallUpsertPayloadV1)
        and item.operation.payload.status in {"failed", "closed"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("answered", [False, True])
async def test_unowned_sparra_expiry_publishes_terminal_call_once(tmp_path, answered):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    )
    try:
        initiated = event()
        await committed(registry, writer, initiated)
        admitted = await registry.snapshot("original")
        assert admitted is not None
        admitted_facts = await writer.read_call_lifecycle(admitted.call_id)
        assert admitted_facts is not None
        if answered:
            await committed(
                registry, writer, event("call.answered", occurred_at=NOW + timedelta(seconds=1))
            )
        elapsed[0] = 30.0
        assert await registry.reap_expired() == 1
        assert await registry.reap_expired() == 0
        await committed(
            registry, writer, event("call.hangup", occurred_at=NOW + timedelta(seconds=31))
        )
        operations = await terminal_publications(writer)
        assert len(operations) == 1, "unowned expiry must publish its actual terminal call"
        operation = operations[0]
        payload = operation.payload
        assert operation.call_id == admitted.call_id
        assert operation.deployment_id == "fixture"
        assert operation.occurred_at == NOW + timedelta(seconds=30)
        assert payload.status == "failed"
        assert payload.end_reason == "token_deadline"
        assert payload.telnyx_call_control_id == "original"
        assert payload.telnyx_call_leg_id == "original-leg"
        assert payload.telnyx_call_session_id == "session"
        assert payload.started_at == (NOW + timedelta(seconds=1) if answered else None)
        assert payload.ended_at == operation.occurred_at
        assert payload.retention_until == admitted_facts.retention_until
        assert payload.disclosure_state == "failed"
        assert payload.message_result is None
        assert await registry.live_call_count() == 0
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            lease = connection.execute("SELECT state,closed_at FROM call_leases").fetchone()
            assert lease[0] == "terminal"
            assert datetime.fromisoformat(lease[1].replace("Z", "+00:00")) == operation.occurred_at
            # Only the two actual test webhooks, plus the answered event when supplied.
            assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (
                3 if answered else 2,
            )
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expiry_preserves_retained_start_and_disclosure(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    )
    try:
        await committed(registry, writer, event())
        admitted = await registry.snapshot("original")
        assert admitted is not None
        admitted_facts = await writer.read_call_lifecycle(admitted.call_id)
        assert admitted_facts is not None
        evidence = DisclosureEvidenceV1(
            schema_version=1,
            started_at=NOW + timedelta(seconds=2),
            completed_at=NOW + timedelta(seconds=3),
            failed_at=None,
            input_gate_opened_at=NOW + timedelta(seconds=3),
        )
        # An engineering fixture already retained through the native writer,
        # proving cleanup preserves evidence rather than inventing new evidence.
        observed = VoiceOperationV1(
            schema_version=1,
            operation_id=uuid4(),
            deployment_id="fixture",
            call_id=admitted.call_id,
            occurred_at=NOW + timedelta(seconds=3),
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id="original",
                telnyx_call_leg_id="original-leg",
                telnyx_call_session_id="session",
                status="active",
                disclosure_state="completed",
                started_at=NOW + timedelta(seconds=1),
                ended_at=None,
                end_reason=None,
                retention_until=admitted_facts.retention_until,
                disclosure_evidence=evidence,
            ),
        )
        await writer.commit_control(PersistenceCommand("outbox", {"operation": observed}, None))
        retained_facts = await writer.read_call_lifecycle(admitted.call_id)
        assert retained_facts is not None
        assert retained_facts.started_at == observed.payload.started_at
        assert retained_facts.disclosure_evidence == evidence
        elapsed[0] = 30.0
        await registry.reap_expired()
        operations = await terminal_publications(writer)
        assert len(operations) == 1
        payload = operations[0].payload
        assert payload.started_at == observed.payload.started_at
        assert payload.disclosure_state == "completed"
        assert payload.disclosure_evidence == evidence
        assert payload.retention_until == observed.payload.retention_until
        assert payload.end_reason == "token_deadline"
        assert payload.message_result is None
        assert [action[0] for action in provider.actions].count("hangup") == 1
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expiry_failure_retains_identity_and_rolls_back(tmp_path):
    elapsed = [0.0]
    reject_terminal = [False]

    def failpoint(name):
        if name == "after_mutation_before_commit" and reject_terminal[0]:
            raise OSError("test-only terminal transaction failure")

    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
        failpoint=failpoint,
    )
    real_commit = writer.commit_lease
    terminal_attempts = []

    async def fail_actual_terminal_commit(**values):
        if values["state"] == "terminal":
            terminal_attempts.append(values["operation"])
            reject_terminal[0] = True
        await real_commit(**values)

    try:
        await committed(registry, writer, event())
        admitted = await registry.snapshot("original")
        assert admitted is not None
        writer.commit_lease = fail_actual_terminal_commit
        elapsed[0] = 30.0
        await registry.reap_expired()
        await asyncio.wait_for(worker, 2)
        assert registry.internal_failure_code == "terminal_persistence_failed"
        assert len(terminal_attempts) == 1
        retained = await registry.snapshot("original")
        assert retained is not None, "failed publication must retain its cleanup identity"
        assert retained.generation == admitted.generation
        assert await registry.live_call_count() == 1
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None
            )
            assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
    finally:
        writer.commit_lease = real_commit
        await registry.wait_background()
        if not worker.done():
            await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expiry_lifecycle_read_failure_keeps_cleanup_identity(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    )
    real_read = writer.read_call_lifecycle

    async def failed_read(call_id):
        raise RuntimeError("test-only lifecycle read failure")

    try:
        await committed(registry, writer, event())
        admitted = await registry.snapshot("original")
        writer.read_call_lifecycle = failed_read
        elapsed[0] = 30.0
        assert await registry.reap_expired() == 1
        writer.read_call_lifecycle = real_read
        retained = await registry.snapshot("original")
        assert retained is not None and retained.generation == admitted.generation
        assert registry.internal_failure_code == "terminal_persistence_failed"
        assert await registry.live_call_count() == 1
        assert await terminal_publications(writer) == []
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None
            )
    finally:
        writer.read_call_lifecycle = real_read
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expiry_missing_native_facts_does_not_release_identity(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    )
    try:
        await committed(registry, writer, event())
        admitted = await registry.snapshot("original")
        assert admitted is not None
        assert await writer.read_call_lifecycle(admitted.call_id) is not None
        # Corrupt only this owned disposable fixture's retained lifecycle. The
        # activated real writer must not turn missing evidence into a success.
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            connection.execute(
                "UPDATE call_leases SET lifecycle_json=NULL WHERE call_id=?",
                (str(admitted.call_id),),
            )
        assert await writer.read_call_lifecycle(admitted.call_id) is None
        elapsed[0] = 30.0
        assert await registry.reap_expired() == 1
        retained = await registry.snapshot("original")
        assert retained is not None, "missing native facts must retain actual cleanup identity"
        assert retained.generation == admitted.generation
        assert registry.internal_failure_code == "terminal_persistence_failed"
        assert await registry.live_call_count() == 1
        assert await terminal_publications(writer) == []
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None
            )
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expiry_cancellation_keeps_frozen_publication(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    )
    real_commit = writer.commit_lease
    attempts = []

    async def cancelled_after_real_commit(**values):
        if values["state"] == "terminal":
            operation = values.get("operation")
            assert operation is not None, "terminal lease and call must share one command"
            attempts.append((values["closed_at"], canonical_operation_bytes(operation)))
        await real_commit(**values)
        if values["state"] == "terminal" and len(attempts) == 1:
            elapsed[0] = 32.0
            raise asyncio.CancelledError("test-only caller cancellation after actual commit")

    try:
        await committed(registry, writer, event())
        writer.commit_lease = cancelled_after_real_commit
        elapsed[0] = 30.0
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(registry.reap_expired(), 2)
        assert len(attempts) == 2
        assert attempts[0] == attempts[1], "retry must not change terminal identity or time"
        operations = await terminal_publications(writer)
        assert len(operations) == 1
        assert canonical_operation_bytes(operations[0]) == attempts[0][1]
        assert operations[0].occurred_at == NOW + timedelta(seconds=30)
        assert await registry.live_call_count() == 0
        assert registry.internal_failure_code is None
        assert [action[0] for action in provider.actions].count("hangup") == 1
    finally:
        writer.commit_lease = real_commit
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expired_retention_terminal_failure_keeps_identity(tmp_path):
    elapsed = [0.0]
    reject_terminal = [False]

    def failpoint(name):
        if name == "after_mutation_before_commit" and reject_terminal[0]:
            raise OSError("test-only expired native lease failure")

    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        writer_utcnow=lambda: NOW,
        monotonic=lambda: 100.0 + elapsed[0],
        failpoint=failpoint,
    )
    real_commit = writer.commit_lease
    attempts = []

    async def fail_actual_expired_terminal_commit(**values):
        if values["state"] == "terminal":
            assert values.get("operation") is None
            attempts.append(values["closed_at"])
            reject_terminal[0] = True
        await real_commit(**values)

    try:
        await committed(registry, writer, event())
        admitted = await registry.snapshot("original")
        assert admitted is not None
        # Fixed writer time retains coherent engineering facts; this does not
        # qualify retention deletion or carrier-side end observation.
        facts = await writer.read_call_lifecycle(admitted.call_id)
        assert facts is not None and facts.admission_generation == admitted.generation
        writer.commit_lease = fail_actual_expired_terminal_commit
        elapsed[0] = timedelta(days=31).total_seconds()
        assert await registry.reap_expired() == 1
        await asyncio.wait_for(worker, 2)
        assert len(attempts) == 1
        retained = await registry.snapshot("original")
        assert retained is not None, "expired native lease failure still owns cleanup identity"
        assert retained.generation == admitted.generation
        assert registry.internal_failure_code == "terminal_persistence_failed"
        assert await registry.live_call_count() == 1
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None
            )
            assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
    finally:
        writer.commit_lease = real_commit
        await registry.wait_background()
        if not worker.done():
            await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expired_retention_cancellation_retries_frozen_lease(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        writer_utcnow=lambda: NOW,
        monotonic=lambda: 100.0 + elapsed[0],
    )
    real_commit = writer.commit_lease
    attempts = []

    async def cancelled_after_expired_real_commit(**values):
        if values["state"] == "terminal":
            assert values.get("operation") is None
            attempts.append(dict(values))
        await real_commit(**values)
        if values["state"] == "terminal" and len(attempts) == 1:
            elapsed[0] += 2
            raise asyncio.CancelledError("test-only expired native commit cancellation")

    try:
        await committed(registry, writer, event())
        admitted = await registry.snapshot("original")
        assert admitted is not None
        facts = await writer.read_call_lifecycle(admitted.call_id)
        assert facts is not None and facts.admission_generation == admitted.generation
        writer.commit_lease = cancelled_after_expired_real_commit
        elapsed[0] = timedelta(days=31).total_seconds()
        with pytest.raises(asyncio.CancelledError):
            # Public reaper invokes cleanup with retry_cancelled_io=False.
            await asyncio.wait_for(registry.reap_expired(), 2)
        assert len(attempts) == 2
        assert attempts[0] == attempts[1], "expired lease retry must freeze its exact close time"
        assert attempts[0]["closed_at"] == NOW + timedelta(days=31)
        assert await registry.live_call_count() == 0
        assert registry.internal_failure_code is None
        assert await terminal_publications(writer) == []
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            state, closed_at = connection.execute(
                "SELECT state,closed_at FROM call_leases"
            ).fetchone()
            assert state == "terminal"
            assert datetime.fromisoformat(closed_at.replace("Z", "+00:00")) == attempts[0][
                "closed_at"
            ]
            assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
    finally:
        writer.commit_lease = real_commit
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_sparra_expired_retention_success_omits_terminal_publication(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        writer_utcnow=lambda: NOW,
        monotonic=lambda: 100.0 + elapsed[0],
    )
    try:
        await committed(registry, writer, event())
        elapsed[0] = timedelta(days=31).total_seconds()
        assert await registry.reap_expired() == 1
        assert await registry.reap_expired() == 0
        assert await registry.live_call_count() == 0
        assert registry.internal_failure_code is None
        assert await terminal_publications(writer) == []
        assert [action[0] for action in provider.actions].count("hangup") == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert connection.execute("SELECT state FROM call_leases").fetchone() == ("terminal",)
            assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone() == (1,)
            assert connection.execute("SELECT COUNT(*) FROM webhook_receipts").fetchone() == (1,)
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_unowned_non_sparra_expiry_keeps_legacy_lease_only_cleanup(tmp_path):
    elapsed = [0.0]
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
        sparra=False,
    )
    try:
        await committed(registry, writer, event())
        elapsed[0] = 30.0
        assert await registry.reap_expired() == 1
        assert await terminal_publications(writer) == []
        assert await registry.live_call_count() == 0
        assert [action[0] for action in provider.actions].count("hangup") == 1
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_owned_sparra_expiry_leaves_publication_to_existing_owner(tmp_path):
    from types import SimpleNamespace

    elapsed = [0.0]
    drains = []
    registry, writer, worker, provider = await start(
        tmp_path,
        utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    )
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        admitted = await registry.snapshot("original")
        claim = await registry.claim_once(
            call_control_id="original",
            token_digest=admitted.token_digest,
            abort_target_publisher=lambda target: True,
            abort_target_clearer=lambda target: None,
        )
        assert claim is not None

        async def wait():
            pass

        owner = SimpleNamespace(
            _task=asyncio.current_task(),
            _phase="constructing",
            _session=None,
            _terminal_capability=None,
            request_drain=drains.append,
            wait=wait,
        )
        grant = await registry.consume_claim_for_construction(claim, "stream", owner, owner._task)
        assert grant is not None
        elapsed[0] = 30.0
        assert await registry.reap_expired() == 1
        assert drains == ["token_deadline"]
        assert await terminal_publications(writer) == []
        assert await registry.live_call_count() == 1
        assert not any(action[0] == "hangup" for action in provider.actions)
        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "active", None
            )
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled, expected_status", [(False, 200), (True, 503)])
async def test_pinned_policy_requires_qualified_audio_before_answer(
    tmp_path, enabled, expected_status
):
    begins = []
    async def begin(deployment, call_id, routing):
        begins.append(call_id)
        return BeginCallSnapshotV1.model_validate(
            {**snapshot(call_id, routing).model_dump(), "recording_enabled": enabled}
        )

    registry, writer, worker, provider = await start(tmp_path, begin)
    try:
        result = await committed(registry, writer, event())
        assert result.status_code == expected_status
        assert any(action[0] == "answer" for action in provider.actions) is (not enabled)
        facts = await writer.read_call_lifecycle(begins[0])
        assert getattr(facts, "recording_policy_revision", None) == 1
        assert getattr(facts, "recording_enabled", None) is enabled
        assert getattr(facts, "audio_reserved_bytes", None) == 0
        if enabled:
            assert not any(action[0] == "streaming" for action in provider.actions)
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_committed_transfer_target_is_acknowledged_without_original_admission(tmp_path):
    from pydantic import SecretStr

    begins = []

    async def begin(deployment, call_id, routing):
        begins.append((call_id, routing.telnyx_call_control_id))
        return snapshot(call_id, routing)

    registry, writer, worker, provider = await start(tmp_path, begin)
    requested = None
    try:
        assert (await committed(registry, writer, event())).status_code == 200
        assert (await committed(registry, writer, event("call.answered"))).status_code == 200
        generation = await registry.generation_handle("original")
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await provider.transfer_entered.wait()
        facts = await writer.read_call_lifecycle((await registry.snapshot("original")).call_id)
        target = event(
            call_control_id="target",
            call_leg_id="target-leg",
            to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation),
            direction="outgoing",
            call_state=None,
        )
        assert (await committed(registry, writer, target)).status_code == 200
        assert (await committed(registry, writer, target, duplicate=True)).status_code == 200
        assert [control for _, control in begins] == ["original"]
        assert not any(action[1] == "target" for action in provider.actions)
        assert await registry.snapshot("target") is None
        assert await registry.live_call_count() == 1
        assert registry._permits_used == 1
        assert (await committed(registry, writer, event("call.hangup"))).status_code == 200
        assert await registry.live_call_count() == 0
        assert registry._permits_used == 0
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await requested
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.parametrize(
    "bad",
    [
        {"connection_id": "wrong"},
        {"to_e164": TARGET},
        {"occurred_at": NOW - timedelta(seconds=301)},
        {"occurred_at": NOW + timedelta(seconds=31)},
    ],
)
@pytest.mark.asyncio
async def test_signed_admission_binding_and_event_age_reject_before_local_effect(tmp_path, bad):
    registry, writer, worker, provider = await start(tmp_path)
    try:
        with pytest.raises(CallAdmissionRejected):
            await registry.resolve_webhook(event(**bad))
        assert await registry.live_call_count() == 0
        assert provider.actions == []
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_normal_answer_and_early_streaming_share_one_begin_future(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def begin(deployment, call_id, routing):
        calls.append((call_id, routing))
        entered.set()
        await release.wait()
        return snapshot(call_id, routing)

    registry, writer, worker, provider = await start(tmp_path, begin)
    try:
        initiated = asyncio.create_task(committed(registry, writer, event()))
        await entered.wait()
        answered = asyncio.create_task(committed(registry, writer, event("call.answered")))
        await asyncio.sleep(0.01)
        assert provider.actions == []
        assert len(calls) == 1
        release.set()
        await asyncio.gather(initiated, answered)
        assert {action[0] for action in provider.actions} <= {"answer", "streaming"}
        facts = await writer.read_call_lifecycle(calls[0][0])
        assert facts.admitted_at.microsecond == 123000
        assert facts.retention_until == facts.admitted_at + timedelta(days=30)
    finally:
        release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


def test_manifest_extension_is_strict_and_legacy_serialization_omits_absence():
    from tests.unit.test_config import manifest_data

    legacy = config.AgentManifestV1.model_validate(manifest_data())
    assert "sparra" not in legacy.model_dump()
    values = manifest_data(
        max_concurrent_calls=1, transcript_retention_days=30, sparra=policy().model_dump()
    )
    pilot = config.AgentManifestV1.model_validate(values)
    assert pilot.sparra == policy()
    for updates in (
        {"sparra": None},
        {"dids": [DID, TARGET]},
        {"transcript_retention_days": 7},
        {"max_concurrent_calls": 2},
        {"dids": ["anonymous"]},
    ):
        with pytest.raises(ValidationError):
            config.AgentManifestV1.model_validate({**values, **updates})


@pytest.mark.asyncio
async def test_pending_transfer_fence_survives_drain_cancel_and_writer_restart(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    await committed(registry, writer, event())
    generation = await registry.generation_handle("original")
    try:
        await persist_original_start(registry, writer)
        transfer = asyncio.create_task(registry.request_human(generation))
        await provider.transfer_entered.wait()
        transfer.cancel()
        await asyncio.gather(transfer, return_exceptions=True)
        await registry.begin_drain()
        await registry.close_session_owner_registration()
        assert not any(item[0] == "hangup" for item in provider.actions)
        assert await registry.live_call_count() == 1
        call_id = (await registry.snapshot("original")).call_id
        facts = await writer.read_call_lifecycle(call_id)
        assert facts.transfer_command_id == provider.actions[-1][2]
        provider.transfer_release.set()
        await registry.wait_background()
    finally:
        provider.transfer_release.set()
        await writer.drain(2)
        await worker
    recovered = PersistenceWriter(
        tmp_path / "voice.sqlite", CryptoKeyring({1: bytes(range(32))}, active_version=1)
    )
    restarted = asyncio.create_task(recovered.run())
    assert await recovered.wait_ready()
    try:
        stale = recovered.take_stale_leases()[0]
        assert stale.lifecycle.transfer_command_id == facts.transfer_command_id
        assert stale.lifecycle.retention_until == NOW.replace(microsecond=123000) + timedelta(
            days=30
        )
    finally:
        await recovered.drain(2)
        await restarted


@pytest.mark.asyncio
async def test_nullable_lifecycle_read_completes_when_call_has_no_original_facts(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    try:
        assert await asyncio.wait_for(writer.read_call_lifecycle(uuid4()), timeout=0.2) is None
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_committed_intent_survives_local_failure_but_closed_executor_cannot_dispatch(
    tmp_path,
):
    registry, writer, worker, provider = await start(tmp_path)
    await committed(registry, writer, event())
    entered, release = asyncio.Event(), asyncio.Event()
    original_commit = writer.commit_transfer_intent

    async def pause_after_real_commit(facts):
        await original_commit(facts)
        entered.set()
        await release.wait()

    writer.commit_transfer_intent = pause_after_real_commit
    generation = await registry.generation_handle("original")
    try:
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await entered.wait()
        provider.dispatch_available = False
        release.set()
        assert await asyncio.wait_for(requested, timeout=0.5) == "unavailable_collect_message"
        assert not any(item[0] == "transfer" for item in provider.actions)
        assert await registry.live_call_count() == 1
    finally:
        release.set()
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("hangup_after", [timedelta(seconds=10), timedelta(days=31)])
async def test_paused_committed_intent_fences_real_terminal_authority_and_delayed_bridge(
    tmp_path, hangup_after
):
    from types import SimpleNamespace

    from projetv0_voice.admission import TerminalProposal

    registry, writer, worker, provider = await start(tmp_path)
    await committed(registry, writer, event())
    await committed(registry, writer, event("call.answered"))
    entry_snapshot = await registry.snapshot("original")
    claim = await registry.claim_once(
        call_control_id="original",
        token_digest=entry_snapshot.token_digest,
        abort_target_publisher=lambda target: True,
        abort_target_clearer=lambda target: None,
    )

    async def wait():
        pass

    owner = SimpleNamespace(
        _task=asyncio.current_task(),
        _phase="constructing",
        _session=None,
        _terminal_capability=None,
        request_drain=lambda cause: None,
        wait=wait,
    )
    grant = await registry.consume_claim_for_construction(claim, "stream", owner, owner._task)
    assert grant is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original_commit = writer.commit_transfer_intent

    async def pause_after_real_commit(facts):
        await original_commit(facts)
        entered.set()
        await release.wait()

    writer.commit_transfer_intent = pause_after_real_commit
    generation = await registry.generation_handle("original")
    try:
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await entered.wait()
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="pipeline_failed",
                metric_class="failed",
                cleanup_hangup=True,
            ),
        )
        assert authority.status == "closing" and not authority.cleanup_hangup
        assert await registry.complete_reserved_terminal(authority)
        await registry.begin_drain()
        release.set()
        await asyncio.wait_for(provider.transfer_entered.wait(), 2)
        assert not any(item[0] == "hangup" for item in provider.actions)
        assert await registry.live_call_count() == 1
        facts = await writer.read_call_lifecycle(grant.call_id)
        from pydantic import SecretStr

        target = dict(
            call_control_id="target",
            call_leg_id="target-leg",
            to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation),
            direction="outgoing",
            call_state=None,
        )
        await committed(registry, writer, event(**target))
        await committed(
            registry,
            writer,
            event("call.answered", **{**target, "direction": None, "call_state": "answered"}),
        )
        assert not registry._by_control["original"].no_new_ai
        await committed(registry, writer, event("call.bridged", **{**target, "direction": None}))
        assert registry._by_control["original"].no_new_ai
        await committed(registry, writer, event("call.bridged", **{**target, "direction": None}))
        durable = await writer.read_call_lifecycle(grant.call_id)
        assert durable.qualified_line_bridged_at == NOW
        assert durable.bridge_operation_id is not None
        assert await registry.live_call_count() == 1
        provider.transfer_release.set()
        assert await requested == "qualified_line_connected"
        await registry.close_session_owner_registration()
        assert not any(item[0] == "hangup" for item in provider.actions)
        await committed(registry, writer, event("call.hangup", occurred_at=NOW + hangup_after))
        assert await registry.live_call_count() == 0
        assert not any(item[0] == "hangup" for item in provider.actions)
    finally:
        release.set()
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_cleanup_reserved_first_prevents_transfer_arming(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    await committed(registry, writer, event())
    generation = await registry.generation_handle("original")
    try:
        async with registry._lock:
            entry = registry._by_control["original"]
            work = registry._mark_terminal_locked(
                entry, reason="pipeline_failed", persist_terminal=True, cleanup_hangup=True
            )
        assert work is not None
        assert await registry.request_human(generation) == "unavailable_collect_message"
        assert entry.transfer_facts is None
        assert not any(item[0] == "transfer" for item in provider.actions)
        await registry._run_terminal_cleanup(work)
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.parametrize("cause", ["user_busy", "no_answer", "timeout", "call_rejected"])
@pytest.mark.asyncio
async def test_bound_target_failure_is_distinct_and_wrong_leg_cannot_take_over(tmp_path, cause):
    from pydantic import SecretStr

    registry, writer, worker, provider = await start(tmp_path)
    await committed(registry, writer, event())
    await committed(registry, writer, event("call.answered"))
    generation = await registry.generation_handle("original")
    try:
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await provider.transfer_entered.wait()
        facts = await writer.read_call_lifecycle((await registry.snapshot("original")).call_id)
        target = dict(
            call_control_id="target",
            call_leg_id="target-leg",
            to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation),
            direction="outgoing",
            call_state=None,
        )
        await committed(registry, writer, event(**target))
        await committed(
            registry,
            writer,
            event("call.bridged", **{**target, "direction": None, "call_leg_id": "wrong"}),
        )
        assert not registry._by_control["original"].no_new_ai
        await committed(
            registry,
            writer,
            event("call.hangup", **{**target, "direction": None, "hangup_cause": cause}),
        )
        provider.transfer_release.set()
        assert await requested == f"{cause}_collect_message"
        assert await registry.live_call_count() == 1
        assert not any(action[0] == "hangup" for action in provider.actions)
        assert not registry._by_control["original"].no_new_ai
    finally:
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["call.bridged", "call.hangup"])
async def test_correlated_target_terminal_before_initiation_is_durable_once(tmp_path, terminal):
    from pydantic import SecretStr

    registry, writer, worker, provider = await start(tmp_path)
    requested = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(
            await registry.generation_handle("original")))
        await asyncio.wait_for(provider.transfer_entered.wait(), 2)
        entry = registry._by_control["original"]
        facts = await writer.read_call_lifecycle(entry.call_id)
        assert facts.target_call_control_id is None
        target = dict(call_control_id="target", call_leg_id="target-leg", to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation), call_state=None)
        observed = event(terminal, **target, direction=None, hangup_cause="no_answer")
        assert (await committed(registry, writer, observed)).status_code == 200
        durable = await writer.read_call_lifecycle(entry.call_id)
        assert durable.target_call_control_id == "target"
        assert durable.target_call_leg_id == "target-leg"
        for updates in ({"call_control_id": "other-target"}, {"call_leg_id": "other-leg"}, {}):
            assert (await committed(registry, writer,
                event(**{**target, **updates}, direction="outgoing"))).status_code == 200
        assert (await committed(registry, writer, observed, duplicate=True)).status_code == 200
        same = await writer.read_call_lifecycle(entry.call_id)
        assert same == durable
        provider.transfer_release.set()
        if terminal == "call.bridged":
            assert entry.no_new_ai and same.qualified_line_bridged_at == NOW
            assert same.bridge_operation_id is not None
            assert await requested == "qualified_line_connected"
        else:
            assert not entry.no_new_ai and same.qualified_line_bridged_at is None
            assert same.transfer_failed_at == NOW and same.transfer_failure_cause == "no_answer"
            assert await requested == "no_answer_collect_message"
        assert [action[0] for action in provider.actions].count("transfer") == 1
        assert not any(action[0] == "hangup" for action in provider.actions)
        assert await registry.live_call_count() == 1
        with sqlite3.connect(tmp_path / "voice.sqlite") as db:
            assert db.execute("SELECT count(*) FROM webhook_receipts WHERE event_id=?",
                (observed.event_id,)).fetchone() == (1,)
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.wait_for(requested, 2)
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_or_wrong", [
    {"client_state": None}, {"client_state": "wrong"},
    {"call_session_id": None}, {"call_session_id": "wrong"},
    {"connection_id": None}, {"connection_id": "wrong"},
    {"to_e164": None}, {"to_e164": "+33102030407"},
    {"call_control_id": None},
    {"call_control_id": "original"}, {"call_leg_id": None},
    {"call_leg_id": "original-leg"},
    {"event_type": "call.initiated", "direction": None},
    {"event_type": "call.initiated", "direction": "incoming", "call_state": "parked"},
])
async def test_first_target_binding_requires_complete_authority(tmp_path, missing_or_wrong):
    from pydantic import SecretStr

    registry, writer, worker, provider = await start(tmp_path)
    requested = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(
            await registry.generation_handle("original")))
        await asyncio.wait_for(provider.transfer_entered.wait(), 2)
        entry = registry._by_control["original"]
        facts = await writer.read_call_lifecycle(entry.call_id)
        target = dict(call_control_id="target", call_leg_id="target-leg", to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation), call_state=None, direction=None)
        bad = {**target, **missing_or_wrong}
        if isinstance(bad["client_state"], str):
            bad["client_state"] = SecretStr(bad["client_state"])
        kind = bad.pop("event_type", "call.bridged")
        assert (await committed(registry, writer, event(kind, **bad))).status_code == 200
        assert not entry.no_new_ai
        assert await writer.read_call_lifecycle(entry.call_id) == facts
        assert (await committed(registry, writer,
            event(**{**target, "direction": "outgoing"}))).status_code == 200
        assert (await committed(registry, writer,
            event("call.bridged", **target))).status_code == 200
        provider.transfer_release.set()
        assert await requested == "qualified_line_connected"
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.wait_for(requested, 2)
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.parametrize(
    "caller", [None, "anonymous", "unavailable", "sip:fixture@example.invalid"]
)
@pytest.mark.asyncio
async def test_real_signed_webhook_extracts_binding_and_nullable_caller_before_admission(
    tmp_path, monkeypatch, caller
):
    from tests.unit.test_telnyx_webhooks import event_body, verifier_for

    body = event_body(
        occurred_at=NOW.isoformat(),
        payload={
            "call_control_id": "original",
            "call_leg_id": "original-leg",
            "call_session_id": "session",
            "connection_id": "wrong",
            "to": DID,
            "from": caller,
        },
    )
    verifier, headers = verifier_for(monkeypatch, body, timestamp=int(NOW.timestamp()))
    verified = verifier.verify(body=body, headers=headers)
    assert verified.connection_id == "wrong" and verified.to_e164 == DID
    assert verified.from_e164 is None
    registry, writer, worker, provider = await start(tmp_path)
    try:
        with pytest.raises(CallAdmissionRejected):
            await registry.resolve_webhook(verified)
        assert provider.actions == []
        assert await registry.live_call_count() == 0
    finally:
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_begin_return_after_original_admission_deadline_cannot_answer(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def begin(deployment, call_id, routing):
        entered.set()
        await release.wait()
        return snapshot(call_id, routing)

    registry, writer, worker, provider = await start(tmp_path, begin)
    try:
        processing = asyncio.create_task(committed(registry, writer, event()))
        await entered.wait()
        registry._utcnow = lambda: NOW + timedelta(seconds=301)
        release.set()
        result = await processing
        assert result.status_code == 503
        assert provider.actions == []
    finally:
        release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.parametrize("cause", ["user_busy", "no_answer"])
@pytest.mark.asyncio
async def test_departed_owner_target_failure_preserves_actual_original_hangup(tmp_path, cause):
    from types import SimpleNamespace

    from pydantic import SecretStr

    from projetv0_voice.admission import TerminalProposal

    registry, writer, worker, provider = await start(tmp_path)
    await committed(registry, writer, event())
    await committed(registry, writer, event("call.answered"))
    snapshot_entry = await registry.snapshot("original")
    claim = await registry.claim_once(
        call_control_id="original",
        token_digest=snapshot_entry.token_digest,
        abort_target_publisher=lambda target: True,
        abort_target_clearer=lambda target: None,
    )

    async def wait():
        pass

    owner = SimpleNamespace(
        _task=asyncio.current_task(),
        _phase="constructing",
        _session=None,
        _terminal_capability=None,
        request_drain=lambda cause: None,
        wait=wait,
    )
    grant = await registry.consume_claim_for_construction(claim, "stream", owner, owner._task)
    await persist_original_start(registry, writer)
    requested = asyncio.create_task(registry.request_human(grant.generation))
    await provider.transfer_entered.wait()
    try:
        authority = await registry.reserve_or_read_terminal(
            grant,
            owner._terminal_capability,
            TerminalProposal(
                status="failed",
                reason="pipeline_failed",
                metric_class="failed",
                cleanup_hangup=True,
            ),
        )
        assert authority.status == "closing"
        await registry._persist_authority_call(authority)
        await registry.complete_reserved_terminal(authority)
        await registry.close_session_owner_registration()
        facts = await writer.read_call_lifecycle(grant.call_id)
        target = dict(
            call_control_id="target",
            call_leg_id="target-leg",
            to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation),
            direction="outgoing",
            call_state=None,
        )
        await committed(registry, writer, event(**target))
        failure = event("call.hangup", **{**target, "direction": None, "hangup_cause": cause})
        await committed(registry, writer, failure)
        await committed(registry, writer, failure)
        retained = await writer.read_call_lifecycle(grant.call_id)
        assert retained.transfer_failed_at == failure.occurred_at
        assert retained.local_closing_at == authority._closed_at
        assert retained.transfer_fenced
        assert await registry.live_call_count() == 1
        assert not any(action[0] == "hangup" for action in provider.actions)
        original = event("call.hangup", occurred_at=NOW + timedelta(seconds=10))
        await committed(registry, writer, original)
        await committed(registry, writer, original)
        assert await registry.live_call_count() == 0
        items = await writer.read_relay_batch(
            batch_size=100, now=datetime.now(UTC) + timedelta(seconds=1), lease_seconds=60
        )
        terminal = [
            item.operation.payload
            for item in items
            if item.operation.kind == "call.upsert" and item.operation.payload.ended_at is not None
        ]
        assert len(terminal) == 1 and terminal[0].ended_at == original.occurred_at
        import sqlite3

        with sqlite3.connect(tmp_path / "voice.sqlite") as connection:
            state, closed_at = connection.execute(
                "SELECT state,closed_at FROM call_leases WHERE call_id=?", (str(grant.call_id),)
            ).fetchone()
        assert state == "terminal"
        assert datetime.fromisoformat(closed_at.replace("Z", "+00:00")) == original.occurred_at
    finally:
        provider.transfer_release.set()
        await asyncio.gather(requested, return_exceptions=True)
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["false", "raise", "cancel", "missing"])
async def test_transfer_audio_boundary_failure_releases_only_unpublished_intent(tmp_path, mode):
    """Registry/SQLite boundary; session preparation is controlled, not capture qualification."""
    from types import SimpleNamespace

    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    registry, writer, worker, provider = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
            **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(uuid4()), "recording_policy": "local_30d",
            "recording_contact_phone": DID, "audio_available": True,
            "recording_id": str(uuid4()),
        })
        entered, release = asyncio.Event(), asyncio.Event()
        closes = []

        async def prepare():
            entered.set()
            await release.wait()
            if mode == "raise":
                raise RuntimeError("controlled transfer preparation failure")
            if mode == "cancel":
                raise asyncio.CancelledError()
            return False

        entry.session = SimpleNamespace(
            stop_result_inference=lambda: None,
            close_audio_for_transfer=lambda: closes.append(True) or True,
            **({"prepare_audio_for_transfer": prepare} if mode != "missing" else {}),
        )
        generation = await registry.generation_handle("original")
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        if mode != "missing":
            await asyncio.wait_for(entered.wait(), 2)
            assert closes == [True] and registry._transfer_fenced(entry)
            assert not (await writer.read_call_lifecycle(entry.call_id)).transfer_fenced
        release.set()
        if mode == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(requested, 2)
        else:
            assert await asyncio.wait_for(requested, 2) == "unavailable_collect_message"
            assert await registry.request_human(generation) == "unavailable_collect_message"
        assert entry.transfer_facts is None and not registry._transfer_fenced(entry)
        assert not (await writer.read_call_lifecycle(entry.call_id)).transfer_fenced
        assert not any(item[0] == "transfer" for item in provider.actions)
        assert await registry.live_call_count() == 1
        assert registry._permits_used == 1
    finally:
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_transfer_audio_on_missing_concrete_session_refuses_before_reservation(tmp_path):
    """Registry policy boundary only; no native capture/qualification is manufactured."""
    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    registry, writer, worker, provider = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
            **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(uuid4()), "recording_policy": "local_30d",
            "recording_contact_phone": DID, "audio_available": True,
            "recording_id": str(uuid4()),
        })
        assert entry.session is None
        generation = await registry.generation_handle("original")
        await persist_original_start(registry, writer)
        assert await registry.request_human(generation) == "unavailable_collect_message"
        assert entry.transfer_task is None and entry.transfer_facts is None
        assert not (await writer.read_call_lifecycle(entry.call_id)).transfer_fenced
        assert not any(item[0] == "transfer" for item in provider.actions)
        assert await registry.live_call_count() == 1 and registry._permits_used == 1
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_transfer_entered_real_commit_unknown_keeps_fence_without_dispatch_or_retry(tmp_path):
    """Real SQLite COMMIT then lost response; existing V1 has no capture obligation."""
    registry, writer, worker, provider = await start(tmp_path)
    commits = []
    native_commit = writer.commit_transfer_intent

    async def lose_after_commit(facts):
        await native_commit(facts)
        commits.append(facts.transfer_command_id)
        raise RuntimeError("controlled committed response loss")

    writer.commit_transfer_intent = lose_after_commit
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        generation = await registry.generation_handle("original")
        with pytest.raises(RuntimeError, match="^controlled committed response loss$"):
            await persist_original_start(registry, writer)
            await registry.request_human(generation)
        assert entry.transfer_facts is not None and registry._transfer_fenced(entry)
        facts = await writer.read_call_lifecycle(entry.call_id)
        assert facts.transfer_fenced
        assert facts.transfer_command_id == entry.transfer_facts.transfer_command_id
        with pytest.raises(RuntimeError, match="^controlled committed response loss$"):
            await registry.request_human(generation)
        assert commits == [facts.transfer_command_id]
        assert not any(item[0] == "transfer" for item in provider.actions)
        assert await registry.live_call_count() == 1 and registry._permits_used == 1
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["off", "local_30d"])
async def test_transfer_audio_off_and_unavailable_do_not_require_capture_callback(tmp_path, policy):
    """Existing provider/SQLite path with a metadata-only V2 pin; no capture claim."""
    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    registry, writer, worker, provider = await start(tmp_path)
    requested = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
            **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(uuid4()), "recording_policy": policy,
            "recording_contact_phone": DID if policy == "local_30d" else None,
            "audio_available": False, "recording_id": None,
        })
        assert entry.session is None
        generation = await registry.generation_handle("original")
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await asyncio.wait_for(provider.transfer_entered.wait(), 2)
        provider.transfer_release.set()
        assert await requested == "ringing"
        assert (await writer.read_call_lifecycle(entry.call_id)).transfer_fenced
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.gather(requested, return_exceptions=True)
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared_result", [False, True])
async def test_terminal_authority_during_audio_prepare_keeps_consumed_fence_without_dispatch(
    tmp_path, prepared_result
):
    """Real registry grant/terminal authority and SQLite; controlled session prep only."""
    from types import SimpleNamespace

    from projetv0_voice.admission import TerminalProposal
    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    registry, writer, worker, provider = await start(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    requested = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
            **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(uuid4()), "recording_policy": "local_30d",
            "recording_contact_phone": DID, "audio_available": True,
            "recording_id": str(uuid4()),
        })
        snapshot = await registry.snapshot("original")
        claim = await registry.claim_once(
            call_control_id="original", token_digest=snapshot.token_digest,
            abort_target_publisher=lambda target: True, abort_target_clearer=lambda target: None,
        )

        async def wait():
            pass

        owner = SimpleNamespace(
            _task=asyncio.current_task(), _phase="constructing", _session=None,
            _terminal_capability=None, request_drain=lambda cause: None, wait=wait,
        )
        grant = await registry.consume_claim_for_construction(claim, "stream", owner, owner._task)
        assert grant is not None

        async def prepare():
            entered.set()
            await release.wait()
            return prepared_result

        entry.session = SimpleNamespace(
            stop_result_inference=lambda: None, close_audio_for_transfer=lambda: True,
            prepare_audio_for_transfer=prepare,
        )
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(grant.generation))
        await asyncio.wait_for(entered.wait(), 2)
        reserved = entry.transfer_facts
        assert reserved is not None
        authority = await registry.reserve_or_read_terminal(
            grant, owner._terminal_capability,
            TerminalProposal(status="failed", reason="pipeline_failed", metric_class="failed",
                             cleanup_hangup=True),
        )
        assert authority.status == "closing" and not authority.cleanup_hangup
        release.set()
        assert await asyncio.wait_for(requested, 2) == "unavailable_collect_message"
        assert entry.transfer_facts is reserved and registry._transfer_fenced(entry)
        assert not (await writer.read_call_lifecycle(entry.call_id)).transfer_fenced
        assert not any(item[0] in {"transfer", "hangup"} for item in provider.actions)
        assert await registry.live_call_count() == 1 and registry._permits_used == 1
        assert await registry.complete_reserved_terminal(authority)
        assert await registry.live_call_count() == 1 and registry._permits_used == 1
        assert (await writer.read_call_lifecycle(grant.call_id)).original_ended_at is None
    finally:
        release.set()
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.gather(requested, return_exceptions=True)
        await registry.wait_background()
        await writer.drain(2)
        await worker



@pytest.mark.asyncio
async def test_transfer_rechecks_live_audio_after_contended_registry_lock_before_intent(tmp_path):
    """Registry/SQLite ordering; controlled session readiness models synchronous invalidation."""
    from types import SimpleNamespace

    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    registry, writer, worker, provider = await start(tmp_path)
    entered, release, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    ready = True
    requested = None
    locked = False
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
            **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(uuid4()), "recording_policy": "local_30d",
            "recording_contact_phone": DID, "audio_available": True,
            "recording_id": str(uuid4()),
        })

        async def prepare():
            entered.set()
            await release.wait()
            returned.set()
            return True

        entry.session = SimpleNamespace(
            stop_result_inference=lambda: None, close_audio_for_transfer=lambda: True,
            prepare_audio_for_transfer=prepare, audio_ready_for_transfer=lambda: ready,
        )
        generation = await registry.generation_handle("original")
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await asyncio.wait_for(entered.wait(), 2)
        await registry._lock.acquire()
        locked = True
        release.set()
        await asyncio.wait_for(returned.wait(), 2)
        assert not requested.done()
        ready = False
        registry._lock.release()
        locked = False
        assert await asyncio.wait_for(requested, 2) == "unavailable_collect_message"
        assert entry.transfer_facts is None
        assert not (await writer.read_call_lifecycle(entry.call_id)).transfer_fenced
        assert not any(item[0] in {"transfer", "hangup"} for item in provider.actions)
        assert await registry.live_call_count() == 1 and registry._permits_used == 1
    finally:
        if locked:
            registry._lock.release()
        release.set()
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.gather(requested, return_exceptions=True)
        await registry.wait_background()
        await writer.drain(2)
        await worker

@pytest.mark.asyncio
@pytest.mark.parametrize("invalidated", ["opposition", "process-draining"])
async def test_transfer_live_audio_readiness_is_rechecked_after_real_intent_commit(
    tmp_path, invalidated
):
    """Real registry/SQLite COMMIT; session readiness is a controlled owner boundary."""
    from types import SimpleNamespace

    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    registry, writer, worker, provider = await start(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    ready = True
    requested = None
    native_commit = writer.commit_transfer_intent

    async def hold_after_commit(facts):
        await native_commit(facts)
        entered.set()
        await release.wait()

    writer.commit_transfer_intent = hold_after_commit
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
            **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}), "schema_version": 2,
            "workspace_id": str(uuid4()), "recording_policy": "local_30d",
            "recording_contact_phone": DID, "audio_available": True,
            "recording_id": str(uuid4()),
        })

        async def prepare():
            return True

        entry.session = SimpleNamespace(
            stop_result_inference=lambda: None, close_audio_for_transfer=lambda: True,
            prepare_audio_for_transfer=prepare, audio_ready_for_transfer=lambda: ready,
        )
        generation = await registry.generation_handle("original")
        await persist_original_start(registry, writer)
        requested = asyncio.create_task(registry.request_human(generation))
        await asyncio.wait_for(entered.wait(), 2)
        if invalidated == "opposition":
            ready = False
        else:
            await registry.begin_drain()
            assert ready
        release.set()
        assert await asyncio.wait_for(requested, 2) == "unavailable_collect_message"
        facts = await writer.read_call_lifecycle(entry.call_id)
        assert facts.transfer_failure_cause == "local_not_dispatched"
        assert not facts.transfer_fenced and not registry._transfer_fenced(entry)
        assert facts.transfer_command_id == entry.transfer_facts.transfer_command_id
        assert facts.retention_until == entry.begin_snapshot.retention_until
        assert not any(item[0] in {"transfer", "hangup"} for item in provider.actions)
        assert await registry.live_call_count() == 1 and registry._permits_used == 1
    finally:
        release.set()
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.gather(requested, return_exceptions=True)
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize(("elapsed", "expected"), [(60, 218), (248, 30), (248.001, None)])
async def test_transfer_uses_remaining_original_started_budget_and_refuses_fractional_minimum(
    tmp_path, elapsed, expected,
):
    clock, monotonic = [NOW], [100.0]
    registry, writer, worker, provider = await start(
        tmp_path, utcnow=lambda: clock[0], monotonic=lambda: monotonic[0])
    try:
        await committed(registry, writer, event())
        await persist_original_start(registry, writer)
        clock[0], monotonic[0] = NOW + timedelta(seconds=elapsed), 100.0 + elapsed
        provider.transfer_release.set()
        generation = await registry.generation_handle("original")
        result = await registry.request_human(generation)
        entry = registry._by_control["original"]
        facts = await writer.read_call_lifecycle(entry.call_id)
        transfers = [action for action in provider.actions if action[0] == "transfer"]
        if expected is None:
            assert result == "unavailable_collect_message" and transfers == []
            assert registry._by_control["original"].transfer_facts is None
            assert not facts.transfer_fenced
        else:
            assert result == "ringing" and len(transfers) == 1
            assert transfers[0][3].time_limit_secs == expected
            assert transfers[0][3].timeout_secs == 20
            assert facts.started_at == NOW
            assert registry._by_control["original"].transfer_facts.started_at == NOW
    finally:
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "available"),
    [(TARGET, False), ("+33102030407", True), (None, True)],
    ids=["known-same-target", "known-different-caller", "unknown-caller"],
)
async def test_native_admission_relay_refuses_known_caller_as_target_only(
    tmp_path, caller, available
):
    registry, writer, worker, provider = await start(tmp_path)
    try:
        await committed(registry, writer, event(from_e164=caller))
        await persist_original_start(registry, writer)
        generation = await registry.generation_handle("original")
        assert await registry.human_tool_available(generation) is available
        provider.transfer_release.set()
        result = await registry.request_human(generation)
        entry = registry._by_control["original"]
        facts = await writer.read_call_lifecycle(entry.call_id)
        transfers = [action for action in provider.actions if action[0] == "transfer"]
        if available:
            assert result == "ringing" and len(transfers) == 1
            assert transfers[0][3].to_e164 == TARGET
        else:
            assert result == "unavailable_collect_message" and transfers == []
            assert entry.transfer_facts is None
            assert facts.transfer_command_id is None and not facts.transfer_fenced
        assert await registry.live_call_count() == 1
        assert not entry.no_new_ai
        assert not any(action[0] == "hangup" for action in provider.actions)
    finally:
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing", "before_admission", "future"])
async def test_transfer_missing_or_inconsistent_original_start_refuses_before_effects(
    tmp_path, invalid,
):
    registry, writer, worker, provider = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        if invalid != "missing":
            await persist_original_start(registry, writer,
                started_at=NOW + timedelta(seconds=-1 if invalid == "before_admission" else 1))
        provider.transfer_release.set()
        assert await registry.request_human(await registry.generation_handle("original")) == (
            "unavailable_collect_message")
        assert not any(action[0] == "transfer" for action in provider.actions)
        assert registry._by_control["original"].transfer_facts is None
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_backward_clock_before_first_human_request_does_not_extend_original_allocation(
    tmp_path,
):
    clock, monotonic = [NOW], [100.0]
    registry, writer, worker, provider = await start(
        tmp_path, utcnow=lambda: clock[0], monotonic=lambda: monotonic[0])
    try:
        await committed(registry, writer, event())
        await persist_original_start(registry, writer)
        clock[0], monotonic[0] = NOW + timedelta(seconds=60), 348.001
        provider.transfer_release.set()
        assert await registry.request_human(await registry.generation_handle("original")) == (
            "unavailable_collect_message")
        assert not any(action[0] == "transfer" for action in provider.actions)
        assert registry._by_control["original"].transfer_facts is None
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_forward_clock_during_native_claim_cannot_extend_transfer_planning_after_correction(
    tmp_path,
):
    from types import SimpleNamespace

    clock, monotonic = [NOW], [100.0]
    registry, writer, worker, provider = await start(
        tmp_path, utcnow=lambda: clock[0], monotonic=lambda: monotonic[0])
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        admitted = await registry.snapshot("original")
        # The native claim is permitted by elapsed monotonic time, while UTC jumps.
        clock[0], monotonic[0] = NOW + timedelta(seconds=121), 101.0
        claim = await registry.claim_once(call_control_id="original",
            token_digest=admitted.token_digest, abort_target_publisher=lambda _target: True,
            abort_target_clearer=lambda _target: None)

        async def wait():
            return None

        owner = SimpleNamespace(_task=asyncio.current_task(), _phase="constructing", _session=None,
            _terminal_capability=None, request_drain=lambda _reason: None, wait=wait)
        grant = await registry.consume_claim_for_construction(claim, "stream", owner, owner._task)
        assert grant is not None and grant.started_at == NOW + timedelta(seconds=121)
        await persist_original_start(registry, writer, started_at=grant.started_at)
        original = await writer.read_call_lifecycle(grant.call_id)
        assert original.started_at == grant.started_at
        # 249 real seconds since claim; UTC has corrected, still later than the start.
        clock[0], monotonic[0] = NOW + timedelta(seconds=250), 350.0
        provider.transfer_release.set()
        assert await registry.request_human(grant.generation) == "unavailable_collect_message"
        assert not any(action[0] == "transfer" for action in provider.actions)
        retained = await writer.read_call_lifecycle(grant.call_id)
        assert retained.started_at == original.started_at and not retained.transfer_fenced
    finally:
        provider.transfer_release.set()
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize(("audio", "lapse"), [
    (False, "late"), (False, "backward_clock"), (False, "closed"),
    (True, "late"), (True, "backward_clock"), (True, "drain"), (True, "closed"),
])
async def test_transfer_post_commit_no_dispatch_is_durable_without_provider_end_fact(
    tmp_path, audio, lapse,
):
    from types import SimpleNamespace

    from projetv0_voice.audio_contract import BeginCallSnapshotV2

    clock, monotonic = [NOW], [100.0]
    registry, writer, worker, provider = await start(
        tmp_path, utcnow=lambda: clock[0], monotonic=lambda: monotonic[0])
    entered, release = asyncio.Event(), asyncio.Event()
    original_commit = writer.commit_transfer_intent

    async def paused(facts):
        await original_commit(facts)
        entered.set()
        await release.wait()

    writer.commit_transfer_intent = paused
    requested = None
    try:
        await committed(registry, writer, event())
        await persist_original_start(registry, writer)
        entry = registry._by_control["original"]
        if lapse == "backward_clock":
            clock[0], monotonic[0] = NOW + timedelta(seconds=120), 220.0
        if audio:
            entry.begin_snapshot = BeginCallSnapshotV2.model_validate({
                **entry.begin_snapshot.model_dump(exclude={"recording_enabled"}),
                "schema_version": 2, "workspace_id": str(uuid4()), "recording_policy": "local_30d",
                "recording_contact_phone": DID, "audio_available": True,
                "recording_id": str(uuid4()),
            })

            async def prepare():
                return True

            entry.session = SimpleNamespace(close_audio_for_transfer=lambda: True,
                prepare_audio_for_transfer=prepare, audio_ready_for_transfer=lambda: True)
        requested = asyncio.create_task(registry.request_human(
            await registry.generation_handle("original")))
        await asyncio.wait_for(entered.wait(), 2)
        stale = await writer.read_call_lifecycle(entry.call_id)
        if lapse in {"late", "backward_clock"}:
            clock[0] = NOW + timedelta(seconds=248.001 if lapse == "late" else 60)
            monotonic[0] = 348.001
        elif lapse == "drain":
            await registry.begin_drain()
        else:
            provider.dispatch_available = False
        release.set()
        assert await asyncio.wait_for(requested, 2) == "unavailable_collect_message"
        facts = await writer.read_call_lifecycle(entry.call_id)
        assert facts.transfer_failed_at is not None
        assert facts.transfer_failure_cause == "local_not_dispatched"
        assert facts.transfer_command_id == stale.transfer_command_id
        assert facts.original_ended_at is None and facts.qualified_line_bridged_at is None
        assert not facts.transfer_fenced and not registry._transfer_fenced(entry)
        assert not any(action[0] == "transfer" for action in provider.actions)
        await writer.commit_transfer_observation(stale, None)
        replayed = await writer.read_call_lifecycle(entry.call_id)
        assert replayed.transfer_failed_at == facts.transfer_failed_at
        assert await registry.live_call_count() == 1
        assert await registry.request_human(await registry.generation_handle("original")) == (
            "unavailable_collect_message")
    finally:
        release.set()
        provider.transfer_release.set()
        if requested is not None:
            await asyncio.gather(requested, return_exceptions=True)
        await registry.wait_background()
        await writer.drain(2)
        await worker
