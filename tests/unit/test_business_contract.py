from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from pydantic import ValidationError

from projetv0_voice import models
from projetv0_voice.crypto import CryptoKeyring, EncryptedValue
from projetv0_voice.persistence.commands import (
    EncryptedCommandTooLarge,
    PersistenceCommand,
    canonical_operation_bytes,
    encrypt_operation,
    operation_aad,
)
from projetv0_voice.persistence.writer import PersistenceWriter

NOW = datetime(2026, 10, 1, 10, 0, 0, 123000, tzinfo=UTC)
CALL_ID = UUID(int=2)
TURN_ID = UUID(int=3)
ROOT = Path(__file__).resolve().parents[2]
BASE = "6327b576feaa61421f77a2a3e80572d16f1d8433"


def model(name: str):
    assert hasattr(models, name), f"missing native {name}"
    return getattr(models, name)


def routing(**updates: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "direction": "incoming",
        "connection_id": "connection-1",
        "to_e164": "+33123456789",
        "from_e164": None,
        "telnyx_call_control_id": "control-1",
        "telnyx_call_leg_id": None,
        "telnyx_call_session_id": None,
        "admitted_at": "2026-10-01T10:00:00.123Z",
        **updates,
    }


def result(**updates: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "quality": "partial",
        "category": "callback",
        "summary": "Rappeler le client.",
        "contact": {
            "name": None,
            "callback_e164": None,
            "preference": None,
            "callback_source": "missing",
            "callback_confirmed": False,
        },
        "next_action": "Rappeler",
        "evidence": [{"turn_id": str(TURN_ID), "role": "user"}],
        "request_confirmed": False,
        **updates,
    }


def legacy_operation(**extensions: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation_id": str(UUID(int=1)),
        "call_id": str(CALL_ID),
        "deployment_id": "agent-a",
        "occurred_at": "2026-10-01T10:00:00.123000Z",
        "kind": "call.upsert",
        "payload": {
            "telnyx_call_control_id": "control-1",
            "telnyx_call_leg_id": None,
            "telnyx_call_session_id": None,
            "status": "pending",
            "disclosure_state": "pending",
            "started_at": None,
            "ended_at": None,
            "end_reason": None,
            "retention_until": "2026-10-31T10:00:00.123000Z",
            **extensions,
        },
    }


@pytest.fixture
def original_models():
    # Reference imports suppress bytecode explicitly and execute only pinned owned source.
    sys.dont_write_bytecode = True
    source = subprocess.run(
        ["git", "show", f"{BASE}:src/projetv0_voice/models.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    original = ModuleType("_task1_original_models")
    sys.modules[original.__name__] = original
    try:
        exec(compile(source, "<pinned-native-models>", "exec"), original.__dict__)
        yield original
    finally:
        sys.modules.pop(original.__name__, None)


@pytest.mark.asyncio
async def test_legacy_serialization_and_real_sqlite_digest_match_original(
    original_models, tmp_path
):
    legacy = models.VoiceOperationV1.model_validate(legacy_operation())
    original = original_models.VoiceOperationV1.model_validate(legacy_operation())
    assert legacy.model_dump(mode="json") == original.model_dump(mode="json") == legacy_operation()
    assert all(
        name not in legacy.model_dump(mode="json")["payload"]
        for name in ("message_result", "disclosure_evidence", "transcript_loss_count")
    )
    original_bytes = json.dumps(
        original.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert canonical_operation_bytes(legacy) == original_bytes
    path = tmp_path / "voice.sqlite"
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    writer = PersistenceWriter(path, keyring)
    task = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    try:
        await writer.commit_control(PersistenceCommand("outbox", {"operation": legacy}, None))
        with sqlite3.connect(path) as connection:
            version, nonce, ciphertext = connection.execute(
                "SELECT key_version, nonce, ciphertext FROM outbox"
            ).fetchone()
        durable = keyring.decrypt(
            EncryptedValue(version, nonce, ciphertext), aad=operation_aad(legacy)
        )
        assert durable == original_bytes
        assert hashlib.sha256(durable).digest() == hashlib.sha256(original_bytes).digest()
    finally:
        await writer.drain(timeout_seconds=2)
        await asyncio.wait_for(task, 2)


@pytest.mark.parametrize(
    "field", ["message_result", "disclosure_evidence", "transcript_loss_count"]
)
def test_explicit_null_extensions_reject(field):
    with pytest.raises(ValidationError):
        models.VoiceOperationV1.model_validate(legacy_operation(**{field: None}))


@pytest.mark.parametrize("value", [True, False, -1, 2147483648, 1.0, "1", None])
def test_transcript_loss_count_is_a_strict_bounded_integer(value):
    with pytest.raises(ValidationError):
        models.VoiceOperationV1.model_validate(legacy_operation(transcript_loss_count=value))


@pytest.mark.parametrize("value", [0, 1, 2147483647])
def test_transcript_loss_count_survives_wire_serialization(value):
    parsed = models.VoiceOperationV1.model_validate(legacy_operation(transcript_loss_count=value))
    assert parsed.model_dump(mode="json")["payload"]["transcript_loss_count"] == value


def test_routing_canonicalizes_original_admission_once_to_utc_milliseconds():
    parsed = model("RoutingV1").model_validate(
        routing(admitted_at="2026-10-01T12:00:00.123456+02:00")
    )
    assert parsed.model_dump(mode="json")["admitted_at"] == "2026-10-01T10:00:00.123Z"
    assert parsed.from_e164 is None


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": True},
        {"direction": "outgoing"},
        {"to_e164": "0033123456789"},
        {"from_e164": "anonymous"},
        {"connection_id": "é" * 129},
        {"telnyx_call_control_id": "é" * 513},
        {"telnyx_call_leg_id": "bad\n"},
        {"admitted_at": "2026-02-30T10:00:00Z"},
        {"admitted_at": 0},
        {"workspace_id": "wrong"},
        {"admitted_at": "2026-10-01T10:00:00"},
    ],
)
def test_routing_rejects_invalid_bounds_dates_and_authority_fields(updates):
    with pytest.raises(ValidationError):
        model("RoutingV1").model_validate(routing(**updates))


def snapshot(**updates):
    return {
        "schema_version": 1,
        "call_id": str(CALL_ID),
        "configuration_revision": 1,
        "knowledge": {
            "business_name": "Garage",
            "sector": "garage",
            "opening_hours": "",
            "services": "",
            "prices": "",
            "faq": "",
            "instructions": "",
        },
        "transfer_destination": None,
        "retention_until": "2026-10-31T10:00:00.123Z",
        **updates,
    }


def test_begin_snapshot_is_exact_strict_reply():
    assert (
        model("BeginCallSnapshotV1").model_validate(snapshot()).model_dump(mode="json")
        == snapshot()
    )
    for updates in [
        {"configuration_revision": True},
        {"workspace_id": str(CALL_ID)},
        {"transfer_destination": "123"},
        {"retention_until": "2026-02-30T00:00:00Z"},
    ]:
        with pytest.raises(ValidationError):
            model("BeginCallSnapshotV1").model_validate(snapshot(**updates))


def disclosure(**updates):
    return {
        "schema_version": 1,
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "input_gate_opened_at": None,
        **updates,
    }


def test_disclosure_evidence_preserves_completed_facts_and_requires_all_keys():
    data = disclosure(
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
        failed_at=NOW + timedelta(seconds=2),
        input_gate_opened_at=NOW + timedelta(seconds=1),
    )
    parsed = model("DisclosureEvidenceV1").model_validate(data)
    assert parsed.completed_at == data["completed_at"]
    with pytest.raises(ValidationError):
        model("DisclosureEvidenceV1").model_validate({"schema_version": 1})


@pytest.mark.parametrize(
    "updates",
    [
        {"input_gate_opened_at": NOW},
        {"started_at": NOW, "completed_at": NOW - timedelta(microseconds=1)},
        {"completed_at": NOW, "input_gate_opened_at": NOW - timedelta(microseconds=1)},
        {"failed_at": "2026-02-30T10:00:00Z"},
        {"started_at": 0},
        {"schema_version": True},
    ],
)
def test_disclosure_rejects_invented_or_invalid_chronology(updates):
    with pytest.raises(ValidationError):
        model("DisclosureEvidenceV1").model_validate(disclosure(**updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"quality": "complete"},
        {"request_confirmed": True},
        {"request_confirmed": 0},
        {"summary": "😀" * 1501},
        {"summary": "bad\x7f"},
        {"summary": "\ud800"},
        {"schema_version": True},
        {"unknown": "field"},
        {"evidence": [{"turn_id": str(TURN_ID), "role": "user"}] * 2},
        {"evidence": [{"turn_id": str(UUID(int=i)), "role": "user"} for i in range(65)]},
        {
            "contact": {
                "name": None,
                "callback_e164": None,
                "preference": None,
                "callback_source": "provider",
                "callback_confirmed": False,
            }
        },
    ],
)
def test_pilot_result_strict_node_contract(updates):
    with pytest.raises(ValidationError):
        model("MessageResultV1").model_validate(result(**updates))


def test_result_utf8_serialized_bound_and_allowed_unicode():
    assert model("MessageResultV1").model_validate(result(summary="😀" * 1000)).summary
    with pytest.raises(ValidationError):
        model("MessageResultV1").model_validate(result(summary="漢" * 2900))


def test_result_crypto_and_evidence_use_existing_keyring_and_call_aad():
    assert importlib.util.find_spec("projetv0_voice.persistence.business_contract") is not None
    business = importlib.import_module("projetv0_voice.persistence.business_contract")
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    inner = model("MessageResultV1").model_validate(result())
    envelope = business.encrypt_message_result(
        inner, call_id=CALL_ID, keyring=keyring, authenticated_turns={TURN_ID: "user"}
    )
    raw = EncryptedValue(
        envelope.key_version,
        base64.b64decode(envelope.nonce_b64),
        base64.b64decode(envelope.ciphertext_b64),
    )
    assert json.loads(
        keyring.decrypt(raw, aad=f"result:{CALL_ID}".encode("ascii"))
    ) == inner.model_dump(mode="json")
    for retained in [{}, {TURN_ID: "assistant"}, {UUID(int=i): "user" for i in range(201)}]:
        with pytest.raises(ValueError):
            business.encrypt_message_result(
                inner, call_id=CALL_ID, keyring=keyring, authenticated_turns=retained
            )


def envelope(**updates):
    return {
        "schema_version": 1,
        "crypto_version": 1,
        "key_version": 1,
        "nonce_b64": base64.b64encode(bytes(range(12))).decode(),
        "ciphertext_b64": base64.b64encode(bytes(16)).decode(),
        **updates,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"key_version": 9007199254740992},
        {"key_version": True},
        {"crypto_version": True},
        {"nonce_b64": base64.b64encode(bytes(13)).decode()},
        {"ciphertext_b64": base64.b64encode(bytes(15)).decode()},
        {"ciphertext_b64": base64.b64encode(bytes(8209)).decode()},
        {"nonce_b64": "%%%%"},
        {"nonce_b64": None},
        {"extra": "unknown"},
    ],
)
def test_result_envelope_rejects_unknowns_and_exact_crypto_bounds(updates):
    with pytest.raises(ValidationError):
        model("MessageResultEnvelopeV1").model_validate(envelope(**updates))


def test_all_extensions_are_consumed_and_legacy_nulls_survive():
    data = legacy_operation(
        message_result=envelope(), disclosure_evidence=disclosure(), transcript_loss_count=0
    )
    parsed = models.VoiceOperationV1.model_validate(data)
    assert parsed.model_dump(mode="json") == data
    assert models.VoiceOperationV1.model_validate_json(parsed.model_dump_json()) == parsed


def test_new_turn_text_limit_is_utf8_and_does_not_rewrite_legacy_turns():
    business = importlib.import_module("projetv0_voice.persistence.business_contract")
    assert business.validate_turn_text("a" * 16384) == "a" * 16384
    assert business.validate_turn_text("😀" * 4096) == "😀" * 4096
    for invalid in ["a" * 16385, "😀" * 4097, "\ud800", "bad\x00", b"not text"]:
        with pytest.raises(ValueError):
            business.validate_turn_text(invalid)


@pytest.mark.parametrize(
    "instant", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00", "infinity"]
)
def test_new_wire_dates_reject_nonfinite_utc_conversion(instant):
    with pytest.raises(ValidationError):
        model("RoutingV1").model_validate(routing(admitted_at=instant))


def test_final_encrypted_command_gate_consumes_all_new_extensions():
    data = legacy_operation(
        message_result=envelope(), disclosure_evidence=disclosure(), transcript_loss_count=7
    )
    data["payload"].update(status="failed", ended_at=NOW, end_reason="x")
    parsed = models.VoiceOperationV1.model_validate(data)
    # Existing storage contract counts final canonical bytes, metadata AAD,
    # twelve-byte nonce and sixteen-byte authentication tag.
    fitting_reason = (
        1 + 65536 - len(canonical_operation_bytes(parsed)) - len(operation_aad(parsed)) - 12 - 16
    )
    data["payload"]["end_reason"] = "x" * fitting_reason
    exact = models.VoiceOperationV1.model_validate(data)
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    assert encrypt_operation(exact, keyring).envelope_size == 65536
    data["payload"]["end_reason"] += "x"
    with pytest.raises(EncryptedCommandTooLarge):
        encrypt_operation(models.VoiceOperationV1.model_validate(data), keyring)
