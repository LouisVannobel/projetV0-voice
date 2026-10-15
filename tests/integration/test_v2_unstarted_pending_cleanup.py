"""Native V2 registry/SQLite pending calls; no WebSocket, AI or live carrier."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import timedelta
from uuid import UUID

import pytest

from projetv0_voice.lifecycle import RuntimeSupervisor
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.telnyx.call_control import CallControlResult
from tests.integration.test_fixed_v2_process import NOW, registry_case
from tests.unit.test_sparra_admission import committed, event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expire_first", "missing_snapshot"), [(False, False), (True, False), (True, True)],
    ids=["clean-shutdown-pending", "expiry-then-clean-shutdown", "expiry-no-memory-begin"],
)
async def test_unstarted_v2_pending_call_is_durably_terminal_after_native_shutdown(
    tmp_path, monkeypatch, expire_first, missing_snapshot,
):
    elapsed = [0.0]
    observed_errors = []
    candidate = UUID("00000000-0000-4000-8000-000000000099")
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], candidate_run_id=candidate,
    ) as case:
        native_commit_lease = case.writer.commit_lease

        async def observe_native_commit(**kwargs):
            try:
                return await native_commit_lease(**kwargs)
            except PersistenceError as error:
                observed_errors.append(str(error))
                raise

        monkeypatch.setattr(case.writer, "commit_lease", observe_native_commit)
        assert (await committed(
            case.registry, case.writer, event(), candidate_limit=2,
        )).status_code == 200
        assert (await committed(
            case.registry, case.writer,
            event("call.answered", occurred_at=NOW + timedelta(seconds=1)),
        )).status_code == 200
        admitted = await case.registry.snapshot("original")
        facts = await case.writer.read_call_lifecycle(admitted.call_id)
        assert facts.started_at is None and facts.original_ended_at is None
        assert case.registry._by_control["original"].lifecycle_owner is None
        assert case.registry._by_control["original"].attached is False
        assert [action[0] for action in case.provider.actions] == ["answer", "streaming"]
        if missing_snapshot:
            case.registry._by_control["original"].begin_snapshot = None

        elapsed[0] = 31.0 if expire_first else 5.0
        if expire_first:
            assert await case.registry.reap_expired() == 1
            assert await case.registry.reap_expired() == 0
        supervisor = RuntimeSupervisor(
            writer=case.writer, registry=case.registry, sparra_enabled=True,
            retention_days=30, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        )
        await supervisor.aclose()
        assert supervisor._closed is True
        assert not case.writer.is_degraded
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        frozen = await case.writer.read_frozen_call_publication_v2(
            admitted.call_id, generation=admitted.generation,
        )
        assert frozen is not None and frozen.payload.message_result is None
        assert frozen.payload.disclosure_evidence is None
        assert frozen.payload.started_at == NOW + timedelta(seconds=1)
        path = case.path

    # The fixture has now joined the actual writer; inspect durable bytes only.
    with sqlite3.connect(path) as connection:
        lease = connection.execute("SELECT state,closed_at FROM call_leases").fetchone()
        runs = connection.execute(
            "SELECT run_id,total_calls,used_calls FROM qualification_runs",
        ).fetchall()
        publications = connection.execute("SELECT count(*) FROM sparra_publications").fetchone()
        outbox = connection.execute("SELECT schema_version FROM outbox").fetchall()
        assert connection.execute("SELECT count(*) FROM sparra_turn_decisions").fetchone() == (0,)
    assert runs == [(str(candidate), 2, 1)], "cleanup must never reset the actual admission"
    assert lease[0] == "terminal" and lease[1] is not None, (
        f"native shutdown left {lease!r}; observed terminal errors={observed_errors!r}"
    )
    assert observed_errors == []
    assert publications == (1,) and outbox == [(2,)]


@pytest.mark.asyncio
@pytest.mark.parametrize("expire_first", [False, True], ids=["pending", "after-expiry-failure"])
async def test_verified_v2_unstarted_hangup_terminalizes_original_without_new_inference(
    tmp_path, expire_first,
):
    elapsed = [0.0]
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    ) as case:
        await committed(case.registry, case.writer, event())
        await committed(case.registry, case.writer, event("call.answered"))
        if expire_first:
            elapsed[0] = 31
            await case.registry.reap_expired()
        supervisor = RuntimeSupervisor(
            writer=case.writer, registry=case.registry, sparra_enabled=True,
            retention_days=30, utcnow=lambda: NOW + timedelta(seconds=32),
        )
        observed = event("call.hangup", occurred_at=NOW + timedelta(seconds=32))
        resolution = await case.registry.resolve_webhook(observed)
        disposition = await supervisor.start_webhook_finalization(
            observed, resolution, receipt="first",
        ).wait()
        assert disposition.status_code == 200
        with sqlite3.connect(case.path) as connection:
            state = connection.execute("SELECT state,closed_at FROM call_leases").fetchone()
            ended = connection.execute(
                "SELECT json_extract(lifecycle_json,'$.original_ended_at') FROM call_leases",
            ).fetchone()
            assert connection.execute(
                "SELECT count(*) FROM webhook_receipts WHERE event_type='call.hangup'",
            ).fetchone() == (1,)
        assert state[0] == "terminal" and state[1] is not None
        assert ended[0] is not None
        assert await case.registry.live_call_count() == 0
        assert [action[0] for action in case.provider.actions].count("hangup") == int(expire_first)
        await supervisor.aclose()


@pytest.mark.asyncio
async def test_native_active_stale_phase_accepts_original_hangup_without_restarting_ai(tmp_path):
    elapsed = [0.0]
    candidate = UUID("00000000-0000-4000-8000-000000000099")
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], candidate_run_id=candidate,
    ) as original:
        await committed(original.registry, original.writer, event(), candidate_limit=2)
        await committed(original.registry, original.writer, event("call.answered"))
        admitted = await original.registry.snapshot("original")
        assert await original.registry.claim_once(
            call_control_id="original", token_digest=admitted.token_digest,
            abort_target_publisher=lambda _target: True, abort_target_clearer=lambda _target: None,
        ) is not None
        # Preserve an actually admitted, unstarted pending record at process
        # departure; creating history must not depend on the expiry bug under test.
        assert await original.registry.live_call_count() == 1
    elapsed[0] = 40
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], candidate_run_id=candidate,
    ) as recovered:
        supervisor = RuntimeSupervisor(
            writer=recovered.writer, registry=recovered.registry, call_control=recovered.provider,
            sparra_enabled=True, retention_days=30, deployment_id="fixture",
            utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        )
        # Run the unchanged actual startup recovery phase over the reopened
        # native writer; full OS/process startup qualification is a separate gate.
        await supervisor._recover_stale_leases()
        restored = recovered.registry._by_control["original"]
        assert restored.no_new_ai and restored.raw_token is None
        assert recovered.begins == [] and recovered.provider.actions == []
        observed = event("call.hangup", occurred_at=NOW + timedelta(seconds=32))
        resolution = await recovered.registry.resolve_webhook(observed)
        disposition = await supervisor.start_webhook_finalization(
            observed, resolution, receipt="first",
        ).wait()
        assert disposition.status_code == 200
        with sqlite3.connect(recovered.path) as connection:
            lease = connection.execute("SELECT state,closed_at FROM call_leases").fetchone()
            ended = connection.execute(
                "SELECT json_extract(lifecycle_json,'$.original_ended_at') FROM call_leases",
            ).fetchone()
            assert connection.execute(
                "SELECT run_id,total_calls,used_calls FROM qualification_runs",
            ).fetchall() == [(str(candidate), 2, 1)]
        assert lease[0] == "terminal" and lease[1] is not None
        assert ended[0] is not None and await recovered.registry.live_call_count() == 0
        assert recovered.begins == [] and recovered.provider.actions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hangup_outcome", ["accepted", "rejected", "outcome_unknown"])
async def test_expired_unclaimed_v2_stale_pending_can_close_without_a_new_phone_event(
    tmp_path, monkeypatch, hangup_outcome,
):
    elapsed = [0.0]
    candidate = UUID("00000000-0000-4000-8000-000000000099")
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], candidate_run_id=candidate,
    ) as original:
        await committed(original.registry, original.writer, event(), candidate_limit=2)
        await committed(original.registry, original.writer, event("call.answered"))
        admitted = await original.registry.snapshot("original")
        facts = await original.writer.read_call_lifecycle(admitted.call_id)
        assert facts.started_at is None and not facts.transfer_fenced
        assert original.registry._by_control["original"].lifecycle_owner is None
        assert not original.registry._by_control["original"].attached
        with sqlite3.connect(original.path) as connection:
            assert connection.execute("SELECT state FROM call_leases").fetchone() == ("pending",)
    elapsed[0] = 40
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], candidate_run_id=candidate,
    ) as recovered:
        native_hangup = recovered.provider.hangup

        async def provider_hangup(*args, **kwargs):
            await native_hangup(*args, **kwargs)
            return CallControlResult(hangup_outcome)

        monkeypatch.setattr(recovered.provider, "hangup", provider_hangup)
        supervisor = RuntimeSupervisor(
            writer=recovered.writer, registry=recovered.registry, call_control=recovered.provider,
            sparra_enabled=True, retention_days=30, deployment_id="fixture",
            utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        )
        # Actual native startup recovery, with no new webhook, replayed signature,
        # synthetic provider end, new Begin, or direct database mutation.
        await supervisor._recover_stale_leases()
        assert recovered.begins == []
        with sqlite3.connect(recovered.path) as connection:
            state = connection.execute("SELECT state,closed_at FROM call_leases").fetchone()
            ended = connection.execute(
                "SELECT json_extract(lifecycle_json,'$.original_ended_at') FROM call_leases",
            ).fetchone()
            assert connection.execute(
                "SELECT run_id,total_calls,used_calls FROM qualification_runs",
            ).fetchall() == [(str(candidate), 2, 1)]
            assert connection.execute("SELECT count(*) FROM webhook_receipts").fetchone() == (2,)
        assert ended == (None,), "local AI timeout cannot assert provider phone completion"
        assert state[0] == "terminal" and state[1] is not None
        assert await recovered.registry.live_call_count() == 0
        assert [action[0] for action in recovered.provider.actions] == ["hangup"]
    async with registry_case(tmp_path, utcnow=lambda: NOW + timedelta(seconds=41)) as restarted:
        supervisor = RuntimeSupervisor(
            writer=restarted.writer, registry=restarted.registry, call_control=restarted.provider,
            sparra_enabled=True, retention_days=30, deployment_id="fixture",
            utcnow=lambda: NOW + timedelta(seconds=41),
        )
        await supervisor._recover_stale_leases()
        assert restarted.begins == [] and restarted.provider.actions == []
        with sqlite3.connect(restarted.path) as connection:
            assert connection.execute(
                "SELECT run_id,total_calls,used_calls FROM qualification_runs",
            ).fetchall() == [(str(candidate), 2, 1)]
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (1,)


@pytest.mark.asyncio
async def test_pending_recovery_native_read_requires_the_exact_stale_record(tmp_path):
    elapsed = [0.0]
    async with registry_case(tmp_path, utcnow=lambda: NOW) as original:
        await committed(original.registry, original.writer, event())
        await committed(original.registry, original.writer, event("call.answered"))
    elapsed[0] = 40
    async with registry_case(tmp_path, utcnow=lambda: NOW + timedelta(seconds=40)) as recovered:
        stale, = recovered.writer.take_stale_leases()
        assert await recovered.writer.unstarted_pending_recovery_allowed(
            stale, now=NOW + timedelta(seconds=40),
        )
        for altered in (
            replace(stale, previous_state="active"),
            replace(stale, tenant_id=str(UUID(int=999))),
            replace(stale, agent_id="foreign"),
            replace(stale, call_control_id="foreign"),
            replace(stale, token_hash=b"x" * 32),
            replace(stale, created_at=stale.created_at + timedelta(milliseconds=1)),
            replace(stale, expires_at=stale.expires_at + timedelta(milliseconds=1)),
            replace(stale, lifecycle=None),
            replace(stale, lifecycle=replace(stale.lifecycle, admission_generation=UUID(int=999))),
        ):
            assert not await recovered.writer.unstarted_pending_recovery_allowed(
                altered, now=NOW + timedelta(seconds=40),
            )
        assert not await recovered.writer.unstarted_pending_recovery_allowed(stale, now=NOW)
        assert recovered.begins == [] and recovered.provider.actions == []
        with sqlite3.connect(recovered.path) as connection:
            assert connection.execute("SELECT state FROM call_leases").fetchone() == ("pending",)
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("protected", ["active", "recording", "audio", "transfer", "erased"])
async def test_pending_recovery_preserves_native_ownership_audio_and_privacy_fences(
    tmp_path, protected,
):
    elapsed = [0.0]
    async with registry_case(
        tmp_path, enabled=protected in {"recording", "audio"}, available=protected == "audio",
        utcnow=lambda: NOW,
    ) as original:
        await committed(original.registry, original.writer, event())
        await committed(original.registry, original.writer, event("call.answered"))
        admitted = await original.registry.snapshot("original")
        if protected == "active":
            assert await original.registry.claim_once(
                call_control_id="original", token_digest=admitted.token_digest,
                abort_target_publisher=lambda _target: True,
                abort_target_clearer=lambda _target: None,
            ) is not None
        elif protected == "transfer":
            facts = await original.writer.read_call_lifecycle(admitted.call_id)
            await original.writer.commit_transfer_intent(replace(
                facts, transfer_command_id=UUID(int=999), transfer_generation=admitted.generation,
                transfer_correlation="fixture", transfer_connection_sha256="a" * 64,
                transfer_destination_sha256="b" * 64,
            ))
        elif protected == "erased":
            await original.registry.stop_call_content(admitted.call_id)
    elapsed[0] = 40
    async with registry_case(tmp_path, utcnow=lambda: NOW + timedelta(seconds=40)) as recovered:
        supervisor = RuntimeSupervisor(
            writer=recovered.writer, registry=recovered.registry, call_control=recovered.provider,
            sparra_enabled=True, retention_days=30, deployment_id="fixture",
            utcnow=lambda: NOW + timedelta(seconds=40),
        )
        await supervisor._recover_stale_leases()
        assert await recovered.registry.live_call_count() == 1
        assert recovered.registry._by_control["original"].no_new_ai
        assert recovered.begins == [] and recovered.provider.actions == []
        with sqlite3.connect(recovered.path) as connection:
            expected = "active" if protected == "active" else "pending"
            assert connection.execute("SELECT state FROM call_leases").fetchone() == (expected,)
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)


@pytest.mark.asyncio
async def test_pending_recovery_without_the_durable_begin_pin_keeps_phone_fenced(tmp_path):
    async with registry_case(tmp_path, utcnow=lambda: NOW) as original:
        observed = event()
        resolved = await original.registry.resolve_webhook(observed)
        # Actual receipt/lease commit before Begin: a real crash boundary, no SQL seed.
        result = await original.writer.submit_webhook(
            receipt={"event_id": observed.event_id, "event_type": observed.event_type,
                "call_control_id": observed.call_control_id, "occurred_at": observed.occurred_at,
                "received_at": NOW,
                "semantic_fingerprint_sha256": observed.semantic_fingerprint_sha256},
            lease=resolved.effect.lease, operation=None,
            admission_facts=resolved.effect.admission_facts,
        ).wait()
        await resolved.reservation.confirm(result)
        assert original.begins == []
    async with registry_case(tmp_path, utcnow=lambda: NOW + timedelta(seconds=40)) as recovered:
        supervisor = RuntimeSupervisor(
            writer=recovered.writer, registry=recovered.registry, call_control=recovered.provider,
            sparra_enabled=True, retention_days=30, deployment_id="fixture",
            utcnow=lambda: NOW + timedelta(seconds=40),
        )
        await supervisor._recover_stale_leases()
        assert await recovered.registry.live_call_count() == 1
        assert recovered.registry._by_control["original"].no_new_ai
        assert recovered.begins == [] and recovered.provider.actions == []
        with sqlite3.connect(recovered.path) as connection:
            assert connection.execute("SELECT state FROM call_leases").fetchone() == ("pending",)
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)


@pytest.mark.asyncio
async def test_pending_recovery_erasure_between_read_and_freeze_cannot_close_lease(
    tmp_path, monkeypatch,
):
    async with registry_case(tmp_path, utcnow=lambda: NOW) as original:
        await committed(original.registry, original.writer, event())
        await committed(original.registry, original.writer, event("call.answered"))
    async with registry_case(tmp_path, utcnow=lambda: NOW + timedelta(seconds=40)) as recovered:
        native_freeze = recovered.writer.freeze_call_publication_v2

        async def erase_then_freeze(operation, *args, **kwargs):
            await recovered.writer.erase_call_content(
                operation.call_id, now=NOW + timedelta(seconds=40), generation=kwargs["generation"],
            )
            return await native_freeze(operation, *args, **kwargs)

        monkeypatch.setattr(recovered.writer, "freeze_call_publication_v2", erase_then_freeze)
        supervisor = RuntimeSupervisor(
            writer=recovered.writer, registry=recovered.registry, call_control=recovered.provider,
            sparra_enabled=True, retention_days=30, deployment_id="fixture",
            utcnow=lambda: NOW + timedelta(seconds=40),
        )
        with pytest.raises(RuntimeError, match="^stale_recovery_failed$"):
            await supervisor._recover_stale_leases()
        assert recovered.begins == [] and recovered.provider.actions == []
        with sqlite3.connect(recovered.path) as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None,
            )
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["missing-pin", "erased-between-read-and-freeze"])
async def test_v2_expiry_freeze_none_retains_pending_identity_and_capacity(
    tmp_path, monkeypatch, refusal,
):
    elapsed, freeze_results = [0.0], []
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    ) as case:
        if refusal == "missing-pin":
            observed = event()
            resolved = await case.registry.resolve_webhook(observed)
            result = await case.writer.submit_webhook(
                receipt={"event_id": observed.event_id, "event_type": observed.event_type,
                    "call_control_id": observed.call_control_id,
                    "occurred_at": observed.occurred_at, "received_at": NOW,
                    "semantic_fingerprint_sha256": observed.semantic_fingerprint_sha256},
                lease=resolved.effect.lease, operation=None,
                admission_facts=resolved.effect.admission_facts,
            ).wait()
            await resolved.reservation.confirm(result)
            assert case.begins == []
        else:
            await committed(case.registry, case.writer, event())
            await committed(case.registry, case.writer, event("call.answered"))
        admitted = await case.registry.snapshot("original")
        native_freeze = case.writer.freeze_call_publication_v2

        async def observe_native_refusal(operation, *args, **kwargs):
            if refusal != "missing-pin":
                await case.writer.erase_call_content(
                    operation.call_id, now=NOW + timedelta(seconds=31),
                    generation=kwargs["generation"],
                )
            frozen = await native_freeze(operation, *args, **kwargs)
            freeze_results.append(frozen)
            return frozen

        monkeypatch.setattr(case.writer, "freeze_call_publication_v2", observe_native_refusal)
        elapsed[0] = 31
        await case.registry.reap_expired()
        assert freeze_results == [None], "exercise the real native suppression return"
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None,
            )
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (0,)
        assert case.registry.internal_failure_code == "terminal_persistence_failed"
        retained = await case.registry.snapshot("original")
        assert retained.generation == admitted.generation
        assert await case.registry.live_call_count() == 1
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        facts = await case.writer.read_call_lifecycle(admitted.call_id)
        assert facts.original_ended_at is None


@pytest.mark.asyncio
async def test_v2_expiry_publication_rollback_drains_and_retains_pending_identity(
    tmp_path, monkeypatch,
):
    elapsed, reject = [0.0], [False]

    def failpoint(name):
        if reject[0] and name == "after_mutation_before_commit":
            raise OSError("offline-publication-rollback")

    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], failpoint=failpoint,
    ) as case:
        await committed(case.registry, case.writer, event())
        await committed(case.registry, case.writer, event("call.answered"))
        admitted = await case.registry.snapshot("original")
        native_freeze = case.writer.freeze_call_publication_v2

        async def fail_actual_publication(*args, **kwargs):
            reject[0] = True
            try:
                return await native_freeze(*args, **kwargs)
            finally:
                reject[0] = False

        monkeypatch.setattr(case.writer, "freeze_call_publication_v2", fail_actual_publication)
        elapsed[0] = 31
        supervisor = RuntimeSupervisor(
            writer=case.writer, registry=case.registry, sparra_enabled=True, retention_days=30,
            utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        )
        await case.registry.reap_expired()
        await asyncio.wait_for(supervisor._writer_fatal_loop(), 2)
        assert supervisor.readiness_snapshot().draining is True
        assert supervisor.readiness_snapshot().ready is False
        assert case.registry.internal_failure_code == "terminal_persistence_failed"
        retained = await case.registry.snapshot("original")
        assert retained is not None and retained.generation == admitted.generation
        assert await case.registry.live_call_count() == 1
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None,
            )
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before-freeze", "after-freeze", "after-lease"])
async def test_v2_expiry_defers_cancellation_until_original_completion_settles(
    tmp_path, monkeypatch, phase,
):
    elapsed, attempts = [0.0], []
    interrupted = asyncio.Event()
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    ) as case:
        await committed(case.registry, case.writer, event())
        await committed(case.registry, case.writer, event("call.answered"))
        admitted = await case.registry.snapshot("original")
        native_freeze, native_lease = (
            case.writer.freeze_call_publication_v2, case.writer.commit_lease,
        )
        blocked = False

        async def interrupt_once(boundary):
            nonlocal blocked
            if phase == boundary and not blocked:
                blocked = True
                interrupted.set()
                await asyncio.Event().wait()

        async def observe_freeze(operation, *args, **kwargs):
            attempts.append(operation.model_dump(mode="json"))
            await interrupt_once("before-freeze")
            result = await native_freeze(operation, *args, **kwargs)
            await interrupt_once("after-freeze")
            return result

        async def observe_lease(**kwargs):
            result = await native_lease(**kwargs)
            await interrupt_once("after-lease")
            return result

        monkeypatch.setattr(case.writer, "freeze_call_publication_v2", observe_freeze)
        monkeypatch.setattr(case.writer, "commit_lease", observe_lease)
        elapsed[0] = 31
        reaper = asyncio.create_task(case.registry.reap_expired())
        await asyncio.wait_for(interrupted.wait(), 2)
        assert await case.registry.reap_expired() == 0
        await asyncio.wait_for(case.registry.close_session_owner_registration(), 2)
        assert await case.registry.live_call_count() == 1
        elapsed[0] = 32
        reaper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reaper, 2)
        assert attempts and all(value == attempts[0] for value in attempts)
        assert attempts[0]["occurred_at"] == (NOW + timedelta(seconds=31)).isoformat().replace(
            "+00:00", "Z",
        )
        assert await case.registry.live_call_count() == 0
        assert case.registry.internal_failure_code is None
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        assert await case.registry.reap_expired() == 0
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT state FROM call_leases").fetchone() == ("terminal",)
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (1,)
        assert (await case.writer.read_call_lifecycle(admitted.call_id)).original_ended_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["generation", "tenant", "deployment"])
async def test_v2_expiry_identity_rejection_does_not_become_lease_only_success(
    tmp_path, mismatch,
):
    elapsed = [0.0]
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    ) as case:
        await committed(case.registry, case.writer, event())
        await committed(case.registry, case.writer, event("call.answered"))
        if mismatch == "generation":
            case.registry._by_control["original"].generation = UUID(int=999)
        elif mismatch == "tenant":
            case.registry._tenant_id = str(UUID(int=999))
        else:
            case.registry._deployment_id = "foreign-deployment"
        elapsed[0] = 31
        await case.registry.reap_expired()
        assert case.registry.internal_failure_code == "terminal_persistence_failed"
        assert await case.registry.live_call_count() == 1
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None,
            )


@pytest.mark.asyncio
async def test_v2_freeze_to_lease_failure_keeps_frozen_publication_and_pending_obligation(
    tmp_path, monkeypatch,
):
    elapsed = [0.0]
    candidate = UUID("00000000-0000-4000-8000-000000000099")
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0], candidate_run_id=candidate,
    ) as case:
        await committed(case.registry, case.writer, event(), candidate_limit=2)
        await committed(case.registry, case.writer, event("call.answered"))
        admitted = await case.registry.snapshot("original")

        async def interrupt_lease(**_kwargs):
            raise OSError("offline-after-freeze-before-lease")

        monkeypatch.setattr(case.writer, "commit_lease", interrupt_lease)
        elapsed[0] = 31
        await case.registry.reap_expired()
        assert case.registry.internal_failure_code == "terminal_persistence_failed"
        assert await case.registry.live_call_count() == 1
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        frozen = await case.writer.read_frozen_call_publication_v2(
            admitted.call_id, generation=admitted.generation,
        )
        assert frozen is not None
        assert await case.writer.freeze_call_publication_v2(
            frozen, None, generation=admitted.generation, provider_callback=None,
        ) == frozen
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                "pending", None,
            )
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (1,)
            frozen_bytes = connection.execute(
                "SELECT * FROM sparra_publications",
            ).fetchall()
    elapsed[0] = 40
    for restart in range(2):
        async with registry_case(
            tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
            candidate_run_id=candidate,
        ) as recovered:
            supervisor = RuntimeSupervisor(
                writer=recovered.writer, registry=recovered.registry,
                call_control=recovered.provider, sparra_enabled=True, retention_days=30,
                deployment_id="fixture", utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
            )
            await supervisor._recover_stale_leases()
            assert recovered.begins == [] and await recovered.registry.live_call_count() == 0
            assert len(recovered.provider.actions) == int(restart == 0)
            assert await recovered.writer.read_frozen_call_publication_v2(
                admitted.call_id, generation=admitted.generation,
            ) == frozen
            with sqlite3.connect(recovered.path) as connection:
                assert connection.execute("SELECT * FROM sparra_publications").fetchall() == (
                    frozen_bytes
                )
                assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (1,)
                assert connection.execute("SELECT state,closed_at FROM call_leases").fetchone() == (
                    "terminal", frozen.occurred_at.isoformat().replace("+00:00", "Z"),
                )
                assert connection.execute(
                    "SELECT run_id,total_calls,used_calls FROM qualification_runs",
                ).fetchall() == [(str(candidate), 2, 1)]
            assert (
                await recovered.writer.read_call_lifecycle(admitted.call_id)
            ).original_ended_at is None


@pytest.mark.asyncio
async def test_v2_expiry_after_retention_does_not_resurrect_business_content(tmp_path):
    elapsed = [0.0]
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    ) as case:
        await committed(case.registry, case.writer, event())
        await committed(case.registry, case.writer, event("call.answered"))
        elapsed[0] = timedelta(days=31).total_seconds()
        assert await case.registry.reap_expired() == 1
        assert await case.registry.live_call_count() == 0
        assert [action[0] for action in case.provider.actions].count("hangup") == 1
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT state FROM call_leases").fetchone() == ("terminal",)
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM outbox").fetchone() == (0,)


@pytest.mark.asyncio
async def test_v2_erasure_phone_fence_is_not_bypassed_by_unstarted_expiry(tmp_path):
    elapsed = [0.0]
    async with registry_case(
        tmp_path, utcnow=lambda: NOW + timedelta(seconds=elapsed[0]),
        monotonic=lambda: 100.0 + elapsed[0],
    ) as case:
        await committed(case.registry, case.writer, event())
        await committed(case.registry, case.writer, event("call.answered"))
        admitted = await case.registry.snapshot("original")
        await case.registry.stop_call_content(admitted.call_id)
        elapsed[0] = 31
        assert await case.registry.reap_expired() == 0
        assert await case.registry.live_call_count() == 1
        assert not any(action[0] == "hangup" for action in case.provider.actions)
        with sqlite3.connect(case.path) as connection:
            assert connection.execute("SELECT count(*) FROM sparra_publications").fetchone() == (0,)
            assert connection.execute(
                "SELECT count(*) FROM sparra_turn_decisions",
            ).fetchone() == (0,)
