"""Historical metadata compatibility only; no invented receipt or capture qualification."""

import sqlite3
from datetime import timedelta
from uuid import UUID

import pytest

from projetv0_voice.audio_contract import BeginCallSnapshotV2
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.schema import LOCAL_AUDIO_CHOICE_MIGRATION_SQL
from tests.unit.test_audio_choice_writer import historical_v6
from tests.unit.test_audio_writer import CALL, GENERATION, NOW, owned, seed_admission, snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [{"configuration_revision": 8}, {"workspace_id": UUID(int=99)}, {"recording_id": UUID(int=99)}],
)
async def test_historical_unknown_accounting_does_not_rebind_a_different_pin(tmp_path, changes):
    path = tmp_path / "historical.sqlite"
    historical_v6(path)
    with sqlite3.connect(path) as connection:
        connection.executescript(LOCAL_AUDIO_CHOICE_MIGRATION_SQL)
    async with owned(path, contract_version=2) as (writer, _keyring):
        wrong = BeginCallSnapshotV2.model_validate(
            {**snapshot().model_dump(mode="python"), **changes}
        )
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(wrong, generation=GENERATION)
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(snapshot(), generation=UUID(int=99))
        await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        with sqlite3.connect(path) as connection:
            assert connection.execute(
                "SELECT committed_last_sequence,committed_total_samples FROM local_audio_pin"
            ).fetchone() == (None, None)
            assert connection.execute("SELECT count(*) FROM webhook_receipts").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retain_pin", [False, True], ids=["fresh-no-pin", "fresh-known-accounting"]
)
async def test_fresh_pin_without_original_receipt_is_not_historical_authority(tmp_path, retain_pin):
    path = tmp_path / "fresh.sqlite"
    async with owned(path, contract_version=2) as (writer, _keyring):
        await seed_admission(writer)
        with sqlite3.connect(path) as connection:
            connection.execute("DELETE FROM webhook_receipts")
            if not retain_pin:
                connection.execute("DELETE FROM local_audio_pin")
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT count(*) FROM local_audio_pin").fetchone()[0] == int(
                retain_pin
            )
        assert not writer.is_degraded


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["lease-created", "contradictory-receipt"])
async def test_historical_unknown_rebind_refuses_conflicting_native_anchors(tmp_path, fault):
    path = tmp_path / "historical-conflict.sqlite"
    historical_v6(path)
    with sqlite3.connect(path) as connection:
        connection.executescript(LOCAL_AUDIO_CHOICE_MIGRATION_SQL)
    async with owned(path, contract_version=2) as (writer, _keyring):
        before_facts = await writer.read_call_lifecycle(CALL)
        with sqlite3.connect(path) as connection:
            before_cipher = connection.execute(
                "SELECT op_id,key_version,nonce,ciphertext FROM outbox"
            ).fetchall()
        changed = NOW + timedelta(milliseconds=1)
        if fault == "lease-created":
            # Negative corruption fixture only. No historical value is changed
            # to make a positive run pass or to reset its original retention.
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "UPDATE call_leases SET created_at=?",
                    (changed.isoformat().replace("+00:00", "Z"),),
                )
        else:
            # A genuinely submitted new receipt contradicts the retained pin;
            # it is not a fabricated original historical signed admission.
            ticket = writer.submit_webhook(
                receipt={
                    "event_id": "conflicting-admission",
                    "event_type": "call.initiated",
                    "call_control_id": "control-a",
                    "occurred_at": changed,
                    "received_at": changed,
                    "semantic_fingerprint_sha256": b"c" * 32,
                },
                lease=None,
                operation=None,
            )
            assert (await ticket.wait()).receipt == "first"
        with pytest.raises(PersistenceError, match="audio_pin_unavailable"):
            await writer.bind_audio_snapshot(snapshot(), generation=GENERATION)
        assert await writer.read_call_lifecycle(CALL) == before_facts
        with sqlite3.connect(path) as connection:
            assert (
                connection.execute(
                    "SELECT op_id,key_version,nonce,ciphertext FROM outbox"
                ).fetchall()
                == before_cipher
            )
            assert connection.execute(
                "SELECT committed_last_sequence,committed_total_samples FROM local_audio_pin"
            ).fetchone() == (None, None)
        assert not writer.is_degraded
