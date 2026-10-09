"""Fresh V2 turn producer and retained consumer over owned native SQLite."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from projetv0_voice.audio_contract import VoiceOperationV2
from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.models import TurnUpsertPayloadV1
from projetv0_voice.persistence.business_result import (
    infer_partial_result,
    validate_result_provenance,
)
from projetv0_voice.persistence.commands import (
    PersistenceError,
    canonical_operation_bytes,
    decode_operation,
    decode_operation_v2,
    operation_aad_from_metadata,
)
from projetv0_voice.pipeline import FirstFailure
from projetv0_voice.session import TurnRecorder
from tests.integration.test_disclosure import (
    _TEST_RUNTIME_METRICS,
    _local_choice_wait,
    pipeline_module,
)
from tests.integration.test_local_audio_capture import accept_local, capture_case
from tests.unit.test_audio_writer import (
    CALL,
    DEADLINE,
    GENERATION,
    NOW,
    authenticate_audio,
    owned,
    seed_admission,
)
from tests.unit.test_sparra_admission import committed, event, start
from tests.unit.test_sparra_result import capture as legacy_capture


def make_recorder(identity, writer, keyring, failure):
    return TurnRecorder(identity=identity, writer=writer, keyring=keyring,
        first_failure=failure, runtime_metrics=_TEST_RUNTIME_METRICS,
        utcnow=lambda: getattr(identity, "started_at", NOW))


def fresh_turn(keyring, number, text="Rappelez-moi."):
    turn_id = uuid4()
    inner = keyring.encrypt(text.encode("utf-8"), aad=f"turn:{turn_id}".encode("ascii"))
    return VoiceOperationV2(schema_version=2, operation_id=uuid4(), deployment_id="agent-a",
        call_id=CALL, occurred_at=NOW, kind="turn.upsert", payload=TurnUpsertPayloadV1(
            turn_id=turn_id, turn_no=number, role="user", source="stt_final", crypto_version=1,
            key_version=inner.key_version, nonce_b64=base64.b64encode(inner.nonce).decode("ascii"),
            ciphertext_b64=base64.b64encode(inner.ciphertext).decode("ascii"),
            started_at=NOW, ended_at=NOW, interrupted=False))


def decoded_rows(path, keyring, *, version=2):
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT schema_version,op_id,deployment_id,call_id,kind,"
            "key_version,nonce,ciphertext FROM outbox WHERE kind='turn.upsert' ORDER BY queue_id"
        ).fetchall()
    operations = []
    for schema, operation_id, deployment, call, kind, key, nonce, cipher in rows:
        assert schema == version
        plaintext = keyring.decrypt(
            EncryptedValue(key, nonce, cipher), aad=operation_aad_from_metadata({
                "schema_version": schema, "operation_id": operation_id,
                "deployment_id": deployment, "call_id": call, "kind": kind,
            }),
        )
        operation = decode_operation_v2(plaintext) if version == 2 else decode_operation(plaintext)
        assert canonical_operation_bytes(operation) == plaintext
        operations.append(operation)
    return operations


@pytest.mark.asyncio
async def test_turn_v2_actual_native_user_assistant_events_retained_aead_and_result(
    tmp_path, monkeypatch,
):
    # Public native external strategy consumes synthetic turn boundaries
    # without SmartTurn inference.
    monkeypatch.setattr(pipeline_module, "UserTurnStrategies", ExternalUserTurnStrategies)
    async with capture_case(tmp_path, recorder_factory=make_recorder) as case:
        assert callable(getattr(case.writer, "try_enqueue_turn_v2", None)), (
            "missing fixed V2 turn writer"
        )
        await accept_local(case)
        timestamp = (case.pin.retention_until - timedelta(days=30)).isoformat()
        def real_turn_count():
            with sqlite3.connect(case.path) as db:
                return db.execute("SELECT count(*) FROM sparra_turn_decisions").fetchone()[0]

        await case.runtime.worker.queue_frame(UserStartedSpeakingFrame())
        await case.runtime.worker.queue_frame(
            TranscriptionFrame("Pouvez-vous me rappeler ?", "", timestamp)
        )
        await case.runtime.worker.queue_frame(UserStoppedSpeakingFrame())
        await _local_choice_wait(lambda: real_turn_count() == 1)
        await case.runtime.worker.queue_frame(LLMFullResponseStartFrame())
        await case.runtime.worker.queue_frame(LLMTextFrame("Je prends votre message."))
        await case.runtime.worker.queue_frame(LLMFullResponseEndFrame())

        await _local_choice_wait(lambda: real_turn_count() == 2)
        retained = await case.writer.read_retained_call(case.pin.call_id)
        assert [turn.role for turn in retained.turns] == ["user", "assistant"]
        assert [turn.text for turn in retained.turns] == [
            "Pouvez-vous me rappeler ?", "Je prends votre message."]
        for operation, turn in zip(
            decoded_rows(case.path, case.keyring), retained.turns, strict=True
        ):
            payload = operation.payload
            assert isinstance(payload, TurnUpsertPayloadV1)
            plaintext = case.keyring.decrypt(EncryptedValue(payload.key_version,
                base64.b64decode(payload.nonce_b64), base64.b64decode(payload.ciphertext_b64)),
                aad=f"turn:{turn.turn_id}".encode("ascii"))
            assert plaintext.decode("utf-8") == turn.text
        requested = []

        class OfflineInference:
            async def run_inference(self, context, **options):
                requested.append((context, options))
                return json.dumps({"result": {
                    "schema_version": 1, "quality": "partial", "category": "callback",
                    "summary": "Rappel demandé.", "next_action": "Rappeler.",
                    "contact": {"name": None, "callback_e164": None, "preference": None,
                        "callback_source": "missing", "callback_confirmed": False},
                    "evidence": [{"turn_id": str(retained.turns[0].turn_id), "role": "user"}],
                    "request_confirmed": False}})

        result = await infer_partial_result(OfflineInference(), retained, None)
        assert result is not None and result.quality == "partial"
        assert requested[0][1]["max_tokens"] == 2048
        invalid = result.model_copy(update={"evidence": [
            result.evidence[0].model_copy(update={"turn_id": uuid4()})]})
        with pytest.raises(ValueError):
            validate_result_provenance(invalid, retained, None)
        assert case.failure.code is None and not case.writer.is_degraded


@pytest.mark.asyncio
async def test_turn_v2_native_truncation_count_and_retained_map_bounds(tmp_path):
    async with owned(tmp_path / "count.sqlite", contract_version=2) as (writer, keyring):
        assert callable(getattr(writer, "try_enqueue_turn_v2", None)), (
            "missing fixed V2 turn writer"
        )
        await seed_admission(writer)
        await authenticate_audio(writer)
        identity = SimpleNamespace(
            call_id=CALL, deployment_id="agent-a", routing=object(), begin_snapshot=None,
            generation=SimpleNamespace(generation=GENERATION),
        )
        from tests.unit.test_audio_writer import snapshot
        identity.begin_snapshot = snapshot()
        failure = FirstFailure()
        recorder = make_recorder(identity, writer, keyring, failure)
        recorder.record_user("é" * 10000, NOW.isoformat())
        await writer.wait_until_idle()
        assert len((await writer.read_retained_call(CALL)).turns[0].text.encode("utf-8")) == 16384
        for number in range(1, 202):
            recorder.record_user(f"tour {number}", NOW.isoformat())
            await writer.wait_until_idle()
        retained = await writer.read_retained_call(CALL)
        assert len(retained.turns) == 200 and retained.loss_count == 3
        assert failure.code is None and not writer.is_degraded
    async with owned(tmp_path / "map.sqlite", contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        accepted = []
        for number in range(1, 35):
            operation = fresh_turn(keyring, number, "a" * 16384)
            assert writer.try_enqueue_turn_v2(operation, generation=GENERATION, truncated=False)
            retained = await writer.read_retained_call(CALL)
            if retained.loss_count:
                break
            accepted.append(operation)
        assert len(retained.turns) < 200 and retained.loss_count == 1
        actual = {
            str(item.payload.turn_id): item.payload.model_dump(mode="json") for item in accepted
        }
        assert len(json.dumps(actual, ensure_ascii=False).encode("utf-8")) <= 524288
        actual[str(operation.payload.turn_id)] = operation.payload.model_dump(mode="json")
        assert len(json.dumps(actual, ensure_ascii=False).encode("utf-8")) > 524288
        assert not writer.is_degraded


@pytest.mark.asyncio
async def test_turn_v2_generation_retained_authentication_and_late_fence_refusal(tmp_path):
    path = tmp_path / "guard.sqlite"
    clock = [NOW]
    async with owned(path, contract_version=2, utcnow=lambda: clock[0]) as (writer, keyring):
        assert callable(getattr(writer, "try_enqueue_turn_v2", None)), (
            "missing fixed V2 turn writer"
        )
        await seed_admission(writer)
        await authenticate_audio(writer)
        operation = fresh_turn(keyring, 1)
        assert not writer.try_enqueue_turn_v2(operation, generation=UUID(int=999), truncated=False)
        assert writer.try_enqueue_turn_v2(operation, generation=GENERATION, truncated=False)
        retained = await writer.read_retained_call(CALL)
        for item in await writer.read_relay_batch(batch_size=100, now=NOW, lease_seconds=30):
            await writer.ack_outbox(
                queue_id=item.queue_id, expected_claim_attempt=item.claim_attempt
            )
        assert await writer.read_retained_call(CALL) == retained
        clock[0] = DEADLINE
        assert not writer.try_enqueue_turn_v2(fresh_turn(keyring, 2), generation=GENERATION)
        clock[0] = NOW
        await writer.erase_call_content(CALL, lease_token=uuid4(), now=NOW)
        later = writer.try_enqueue_turn_v2(fresh_turn(keyring, 3), generation=GENERATION)
        assert isinstance(later, bool)
        assert (await writer.read_retained_call(CALL)).erased and not writer.is_degraded
        assert (await writer.read_call_lifecycle(CALL)).retention_until == DEADLINE
    # Authenticated retained metadata is still checked after outbox ACK, without decoding fallback.
    async with owned(tmp_path / "tamper.sqlite", contract_version=2) as (writer, keyring):
        await seed_admission(writer)
        await authenticate_audio(writer)
        assert writer.try_enqueue_turn_v2(fresh_turn(keyring, 1), generation=GENERATION)
        await writer.read_retained_call(CALL)
    with sqlite3.connect(tmp_path / "tamper.sqlite") as db:
        db.execute("UPDATE sparra_turn_decisions SET deployment_id='owned-wrong-deployment'")
    async with owned(tmp_path / "tamper.sqlite", contract_version=2) as (writer, _keyring):
        with pytest.raises(PersistenceError):
            await writer.read_retained_call(CALL)


@pytest.mark.asyncio
async def test_turn_v2_preserves_original_v1_encrypted_producer_and_retained_bytes(tmp_path):
    async with owned(tmp_path / "mode2.sqlite", contract_version=2) as (writer, keyring):
        assert callable(getattr(writer, "try_enqueue_turn_v2", None)), (
            "missing fixed V2 turn writer"
        )
        with pytest.raises(ValueError):
            writer.try_enqueue_turn_v2(legacy_capture(writer, CALL, 1), generation=GENERATION)
        assert not writer.is_degraded
    registry, writer, task, _provider = await start(tmp_path)
    try:
        await committed(registry, writer, event())
        call = (await registry.snapshot("original")).call_id
        operation = legacy_capture(writer, call, 1)
        assert writer.try_enqueue_turn(operation)
        retained = await writer.read_retained_call(call)
        assert retained.turns[0].text == "Pouvez-vous me rappeler ?"
        assert decoded_rows(tmp_path / "voice.sqlite", writer._keyring, version=1) == [operation]
        with pytest.raises(ValueError):
            writer.try_enqueue_turn_v2(fresh_turn(writer._keyring, 1), generation=GENERATION)
        assert await writer.read_retained_call(call) == retained and not writer.is_degraded
    finally:
        await writer.drain(2)
        await asyncio.wait_for(task, 2)
