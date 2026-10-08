"""Fixed admission consumers; controlled provider/pool data is offline test evidence."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from projetv0_voice.admission import CallRegistry
from projetv0_voice.audio_contract import BeginCallSnapshotV2
from projetv0_voice.config import SparraManifestV1, load_agent_manifest
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.relay import OutboxRelay
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.session import CallIdentity
from tests.contract.test_postgres_sink import sink_with_rows
from tests.unit.test_config import manifest_data, write_bundle
from tests.unit.test_sparra_admission import (
    DID,
    ControlledProvider,
    committed,
    event,
    policy,
)
from tests.unit.test_sparra_admission import (
    NOW as LEGACY_NOW,
)
from tests.unit.test_sparra_admission import (
    snapshot as legacy_snapshot,
)

NOW = LEGACY_NOW.replace(microsecond=123000)
KEY = bytes(range(32))
WORKSPACE = UUID(int=22)


def fixed_policy(version=2):
    return SparraManifestV1.model_validate({
        **policy().model_dump(), "operation_contract_version": version,
    })


def pin(call_id, routing, *, enabled=False, available=False):
    return BeginCallSnapshotV2.model_validate({
        "schema_version": 2, "workspace_id": str(WORKSPACE), "call_id": str(call_id),
        "configuration_revision": 7, "knowledge": legacy_snapshot(call_id, routing).knowledge,
        "transfer_destination": None,
        "retention_until": routing.admitted_at + timedelta(days=30),
        "recording_policy": "local_30d" if enabled else "off",
        "recording_contact_phone": DID if enabled else None, "audio_available": available,
        "recording_id": str(UUID(int=33)) if available else None,
    })


@asynccontextmanager
async def registry_case(
    tmp_path, *, enabled=False, available=False, reply=None, ambiguous=False,
    utcnow=lambda: NOW, failpoint=None,
    deployment_id="fixture",
):
    writer = PersistenceWriter(tmp_path / "voice.sqlite", CryptoKeyring({1: KEY}, active_version=1),
                               contract_version=2, utcnow=utcnow, failpoint=failpoint,
                               process_agent_id=deployment_id, process_deployment_id=deployment_id)
    task = asyncio.create_task(writer.run())
    sink, pool, connection, _factory = sink_with_rows([])
    begins = []
    provider = ControlledProvider()
    registry = None

    async def begin(deployment, call_id, routing):
        begins.append((deployment, call_id, routing))
        returned = pin(call_id, routing, enabled=enabled, available=available)
        if reply is not None:
            returned = reply(returned, call_id, routing)
        connection.rows = [(returned.model_dump(mode="json"),)]
        connection.commit_error = OSError("owned-begin-commit-unknown") if (
            ambiguous and len(begins) == 1
        ) else None
        return await sink.begin_call_v2(deployment, call_id, routing)

    try:
        assert await writer.wait_ready()
        await writer.assert_sparra_compatible()
        registry = CallRegistry(
            writer=writer, call_control=provider, tenant_id=str(WORKSPACE), agent_id=deployment_id,
            deployment_id=deployment_id, capacity=1, lease_ttl_seconds=30,
            stream_url="wss://fixture.invalid/media", retention_days=30,
            utcnow=utcnow, monotonic=lambda: 100.0,
            sparra=fixed_policy(), called_did=DID, begin_call=begin,
        )
        yield SimpleNamespace(registry=registry, writer=writer, provider=provider, begins=begins,
                              sink=sink, pool=pool, connection=connection,
                              path=tmp_path / "voice.sqlite")
    finally:
        if registry is not None:
            await registry.wait_background()
        await sink.close()
        if writer.is_degraded:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        else:
            await writer.drain(2)
            await asyncio.wait_for(task, 2)


def test_fixed_v2_process_manifest_selector_is_strict_immutable_and_changes_owned_bundle(tmp_path):
    assert policy().operation_contract_version == 1
    assert fixed_policy(1).operation_contract_version == 1
    selected = fixed_policy()
    assert selected.operation_contract_version == 2
    with pytest.raises(ValidationError):
        selected.operation_contract_version = 1
    for invalid in (True, 2.0, "2", 0, 3):
        with pytest.raises(ValidationError):
            fixed_policy(invalid)
    raw = manifest_data(max_concurrent_calls=1, transcript_retention_days=30,
                        sparra=policy().model_dump())
    first = write_bundle(tmp_path / "one", raw)
    second = write_bundle(tmp_path / "two", {**raw, "sparra": selected.model_dump()})
    first_manifest = load_agent_manifest(first, host_max_concurrent_calls=1)
    second_manifest = load_agent_manifest(second, host_max_concurrent_calls=1)
    assert first_manifest.sparra.operation_contract_version == 1
    assert second_manifest.sparra.operation_contract_version == 2
    assert hashlib.sha256((first / "manifest.yaml").read_bytes()).digest() != hashlib.sha256(
        (second / "manifest.yaml").read_bytes()
    ).digest()
    with pytest.raises(ValidationError):
        load_agent_manifest(write_bundle(tmp_path / "provider", {**raw,
            "sparra": selected.model_dump(), "recording_mode": "telnyx_dual",
            "recording_retention_days": 30}), host_max_concurrent_calls=1)


@pytest.mark.asyncio
async def test_fixed_v2_process_off_unavailable_admission_grant_has_no_legacy_poison(tmp_path):
    for label, enabled in (("off", False), ("unavailable", True)):
        directory = tmp_path / label
        directory.mkdir()
        async with registry_case(directory, enabled=enabled) as case:
            initiated = event(occurred_at=NOW)
            assert (await committed(case.registry, case.writer, initiated)).status_code == 200
            admitted = await case.registry.snapshot("original")
            assert admitted is not None
            assert (await committed(case.registry, case.writer, initiated,
                                    duplicate=True)).status_code == 200
            assert len(case.begins) == 1 and case.begins[0][1] == admitted.call_id
            assert case.begins[0][2].admitted_at == NOW
            assert (await committed(case.registry, case.writer,
                                    event("call.answered", occurred_at=NOW))).status_code == 200
            claim = await case.registry.claim_once(
                call_control_id="original", token_digest=admitted.token_digest,
                abort_target_publisher=lambda _target: True,
                abort_target_clearer=lambda _target: None,
            )
            assert claim is not None

            async def wait_for_owner():
                return None

            owner = SimpleNamespace(_task=asyncio.current_task(), _phase="constructing",
                _session=None, _terminal_capability=None, request_drain=lambda _reason: None,
                wait=wait_for_owner)
            grant = await case.registry.consume_claim_for_construction(
                claim, "stream", owner, owner._task
            )
            assert grant is not None and isinstance(grant.begin_snapshot, BeginCallSnapshotV2)
            identity = CallIdentity(
                call_id=grant.call_id, generation=grant.generation, lease_claim=grant.lease_claim,
                deployment_id=grant.deployment_id,
                telnyx_call_control_id=grant.telnyx_call_control_id,
                telnyx_call_leg_id=grant.telnyx_call_leg_id,
                telnyx_call_session_id=grant.telnyx_call_session_id, stream_id=grant.stream_id,
                started_at=grant.started_at, retention_until=grant.retention_until,
                routing=grant.routing, begin_snapshot=grant.begin_snapshot,
            )
            assert identity.begin_snapshot is grant.begin_snapshot
            assert grant.retention_until == NOW + timedelta(days=30)
            with sqlite3.connect(case.path) as db:
                assert db.execute("PRAGMA user_version").fetchone()[0] == 9
                assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
                assert db.execute("SELECT generation FROM local_audio_pin").fetchone()[0] == (
                    str(grant.generation.generation)
                )
                assert db.execute("SELECT count(*) FROM webhook_receipts").fetchone()[0] == 2
            assert case.pool.active == 0 and not case.writer.is_degraded
            assert all(call[0] == "SELECT voice.begin_call_v2(%s,%s,%s::jsonb)"
                       for call in case.connection.calls)

            async def unexpected_drain():
                pytest.fail("fixed V2 admission degraded the actual relay")

            relay = OutboxRelay(case.writer, case.sink, utcnow=lambda: NOW,
                                on_degraded=unexpected_drain, drain=unexpected_drain)
            assert (await relay.run_once()).status == "empty"


@pytest.mark.asyncio
async def test_fixed_v2_process_begin_reply_and_ambiguous_retry_never_fall_back(tmp_path):
    wrong = {
        "schema": lambda _pin, call, routing: legacy_snapshot(call, routing),
        "call": lambda value, _call, _routing: value.model_copy(update={"call_id": UUID(int=99)}),
        "expiry": lambda value, _call, _routing: value.model_copy(update={
            "retention_until": value.retention_until + timedelta(milliseconds=1),
        }),
    }
    for label, reply in wrong.items():
        directory = tmp_path / label
        directory.mkdir()
        async with registry_case(directory, reply=reply) as case:
            assert (await committed(case.registry, case.writer,
                                    event(occurred_at=NOW))).status_code == 503
            assert not any(action[0] in {"answer", "streaming"} for action in case.provider.actions)
            assert len(case.begins) == 1 and not case.writer.is_degraded
    directory = tmp_path / "ambiguous"
    directory.mkdir()
    async with registry_case(directory, ambiguous=True) as case:
        assert (await committed(case.registry, case.writer,
                                event(occurred_at=NOW))).status_code == 200
        assert len(case.begins) == 2 and case.begins[0] == case.begins[1]
        # The controlled __aexit__ raises at COMMIT; it never reports a known rollback.
        assert case.connection.transaction_rollbacks == 0
        assert case.connection.transaction_commits == 1 and case.pool.active == 0


@pytest.mark.asyncio
async def test_fixed_v2_process_mixed_selector_refuses_and_old_v1_owner_still_drains(tmp_path):
    legacy = tmp_path / "legacy.sqlite"
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    writer = PersistenceWriter(legacy, keyring, utcnow=lambda: NOW)
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    try:
        # Genuine V1 initial receipt/outbox, kept with its original owner and AEAD bytes.
        registry = CallRegistry(writer=writer, call_control=ControlledProvider(),
            tenant_id="fixture", agent_id="fixture", deployment_id="fixture", capacity=1,
            lease_ttl_seconds=30, stream_url="wss://fixture.invalid/media", retention_days=30,
            utcnow=lambda: NOW, monotonic=lambda: 100.0, sparra=policy(), called_did=DID,
            begin_call=lambda *_args: None)
        resolved = await registry.resolve_webhook(event(occurred_at=NOW))
        assert resolved.effect.operation.schema_version == 1
        await writer.submit_webhook(receipt={"event_id": "legacy-original",
            "event_type": "call.initiated", "call_control_id": "original", "occurred_at": NOW,
            "received_at": NOW, "semantic_fingerprint_sha256": b"l" * 32},
            lease=resolved.effect.lease, operation=resolved.effect.operation,
            admission_facts=resolved.effect.admission_facts).wait()
        with pytest.raises(ValueError):
            CallRegistry(writer=writer, call_control=ControlledProvider(), tenant_id="fixture",
                agent_id="fixture", deployment_id="fixture", capacity=1, lease_ttl_seconds=30,
                stream_url="wss://fixture.invalid/media", retention_days=30,
                sparra=fixed_policy(), called_did=DID, begin_call=lambda *_args: None)
    finally:
        await writer.drain(2)
        await asyncio.wait_for(task, 2)
    frozen = legacy.read_bytes()
    refused = PersistenceWriter(
        legacy, keyring, contract_version=2, utcnow=lambda: NOW,
        process_agent_id="fixture", process_deployment_id="fixture",
    )
    task = asyncio.create_task(refused.run())
    assert not await refused.wait_ready()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert legacy.read_bytes() == frozen
    writer = PersistenceWriter(legacy, keyring, utcnow=lambda: NOW)
    task = asyncio.create_task(writer.run())
    try:
        assert await writer.wait_ready()
        claimed = await writer.read_relay_batch(batch_size=1, now=NOW, lease_seconds=30)
        assert len(claimed) == 1 and claimed[0].operation.schema_version == 1
        assert (await writer.ack_outbox(queue_id=claimed[0].queue_id,
            expected_claim_attempt=claimed[0].claim_attempt)).applied
    finally:
        await writer.drain(2)
        await asyncio.wait_for(task, 2)
