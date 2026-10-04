from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import SecretStr
from test_sparra_admission import NOW, TARGET, committed, event, start


@pytest.mark.asyncio
@pytest.mark.parametrize("in_flight", [False, True])
async def test_bridge_observation_cannot_retain_disclosure_after_content_stop(tmp_path, in_flight):
    from projetv0_voice.models import DisclosureEvidenceV1

    registry, writer, worker, provider = await start(tmp_path)
    release = asyncio.Event()
    requested = bridge_task = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        entry = registry._by_control["original"]
        requested = asyncio.create_task(
            registry.request_human(await registry.generation_handle("original"))
        )
        await provider.transfer_entered.wait()
        facts = await writer.read_call_lifecycle(entry.call_id)
        target = dict(
            call_control_id="target",
            call_leg_id="target-leg",
            to_e164=TARGET,
            client_state=SecretStr(facts.transfer_correlation),
            direction="outgoing",
            call_state=None,
        )
        await committed(registry, writer, event(**target))
        # This component test supplies dated evidence; the connected witness
        # obtains it from real native disclosure/media events.
        evidence = DisclosureEvidenceV1(
            schema_version=1,
            started_at=NOW,
            completed_at=NOW,
            failed_at=None,
            input_gate_opened_at=NOW,
        )
        original_read = writer.read_call_lifecycle
        entered = asyncio.Event()

        async def observed_read(call_id):
            observed = await original_read(call_id)
            if call_id == entry.call_id:
                observed = replace(observed, disclosure_evidence=evidence)
                entered.set()
                if in_flight:
                    await release.wait()
            return observed

        writer.read_call_lifecycle = observed_read
        bridge_task = asyncio.create_task(
            committed(registry, writer, event("call.bridged", **{**target, "direction": None}))
        )
        await entered.wait()
        if not in_flight:
            await bridge_task
            assert entry.bridge_publication.payload.disclosure_evidence == evidence
        await registry.stop_call_content(entry.call_id)
        assert entry.bridge_publication is None
        assert entry.transfer_facts.disclosure_evidence is None
        release.set()
        await bridge_task
        assert entry.bridge_publication is None
        durable = await original_read(entry.call_id)
        assert durable.content_erased and durable.disclosure_evidence is None
        assert durable.content_departed_generation == entry.generation
        assert await registry.live_call_count() == 1
        duplicate = event("call.bridged", **{**target, "direction": None})
        await committed(registry, writer, duplicate)
        await committed(registry, writer, duplicate, duplicate=True)
        assert entry.bridge_publication is None
        assert not any(action[0] == "hangup" for action in provider.actions)
        await committed(registry, writer, event("call.hangup"))
        assert await registry.live_call_count() == 0
    finally:
        release.set()
        provider.transfer_release.set()
        for task in (requested, bridge_task):
            if task is not None:
                await task
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_original_termination_fact_is_durable_and_stale_transfer_cannot_clear_it(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    requested = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        requested = asyncio.create_task(
            registry.request_human(await registry.generation_handle("original"))
        )
        await provider.transfer_entered.wait()
        call_id = registry._by_control["original"].call_id
        stale = await writer.read_call_lifecycle(call_id)
        await committed(registry, writer, event("call.hangup"))
        durable = await writer.read_call_lifecycle(call_id)
        assert getattr(durable, "original_ended_at", None) == NOW
        await writer.commit_transfer_observation(stale, None)
        assert (await writer.read_call_lifecycle(call_id)).original_ended_at == NOW
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await requested
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_sparra_missing_admission_never_accepts_late_capture_or_frozen_result(tmp_path):
    from test_sparra_result import capture

    from projetv0_voice.models import CallUpsertPayloadV1, VoiceOperationV1

    registry, writer, worker, _ = await start(tmp_path)
    try:
        await writer.assert_sparra_compatible()
        late_call = uuid4()
        turn = capture(writer, late_call, 1)
        for _ in range(2):
            assert writer.try_enqueue_turn(turn)
            assert writer.try_enqueue_capture_loss(late_call, turn.payload.turn_id)
            retained = await writer.read_retained_call(late_call)
            assert retained.erased and not retained.turns and retained.loss_count == 0
        operation = VoiceOperationV1(
            schema_version=1,
            operation_id=uuid4(),
            deployment_id="fixture",
            call_id=late_call,
            occurred_at=NOW,
            kind="call.upsert",
            payload=CallUpsertPayloadV1(
                telnyx_call_control_id="late",
                telnyx_call_leg_id=None,
                telnyx_call_session_id=None,
                status="closed",
                disclosure_state="failed",
                started_at=NOW,
                ended_at=NOW,
                end_reason="fixture",
                retention_until=NOW + timedelta(days=30),
            ),
        )
        assert await writer.freeze_call_publication(operation, None, provider_callback=None) is None
        assert await writer.oldest_outbox_created_at() is None
        assert not writer.is_degraded
    finally:
        await registry.wait_background()
        await writer.drain(2)
        await worker


@pytest.mark.asyncio
async def test_same_writer_preserves_newer_bridge_facts_against_stale_observation(tmp_path):
    registry, writer, worker, provider = await start(tmp_path)
    requested = None
    try:
        await committed(registry, writer, event())
        await committed(registry, writer, event("call.answered"))
        requested = asyncio.create_task(
            registry.request_human(await registry.generation_handle("original"))
        )
        await provider.transfer_entered.wait()
        call_id = registry._by_control["original"].call_id
        stale = await writer.read_call_lifecycle(call_id)
        newest = replace(
            stale,
            target_call_control_id="target",
            target_call_leg_id="target-leg",
            qualified_line_bridged_at=NOW,
            bridge_operation_id=uuid4(),
        )
        await writer.commit_transfer_observation(newest, None)
        await registry.stop_call_content(call_id)
        await writer.commit_transfer_observation(stale, None)
        durable = await writer.read_call_lifecycle(call_id)
        assert durable.target_call_control_id == newest.target_call_control_id
        assert durable.qualified_line_bridged_at == newest.qualified_line_bridged_at
        assert durable.bridge_operation_id == newest.bridge_operation_id
        assert durable.content_erased
        assert durable.content_departed_generation == registry._by_control["original"].generation
    finally:
        provider.transfer_release.set()
        if requested is not None:
            await requested
        await registry.wait_background()
        await writer.drain(2)
        await worker
