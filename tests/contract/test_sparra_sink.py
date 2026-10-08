from __future__ import annotations

from datetime import timedelta
from uuid import UUID

import pytest

from projetv0_voice import models
from projetv0_voice.persistence import postgres_sink as sink_module
from projetv0_voice.persistence.postgres_sink import (
    OperationSinkCommitAmbiguousError,
    OperationSinkContractError,
    OperationSinkPermanentError,
    OperationSinkStaleLeaseError,
)
from tests.contract.test_postgres_sink import NOW, SqlstateError, operation, sink_with_rows


def routing():
    assert hasattr(models, "RoutingV1"), "missing RoutingV1"
    return models.RoutingV1.model_validate(
        {
            "schema_version": 1,
            "direction": "incoming",
            "connection_id": "conn-1",
            "to_e164": "+33123456789",
            "from_e164": None,
            "telnyx_call_control_id": "control-1",
            "telnyx_call_leg_id": None,
            "telnyx_call_session_id": None,
            "admitted_at": NOW,
        }
    )


def snapshot():
    return {
        "schema_version": 1,
        "call_id": str(UUID(int=2)),
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
        "retention_until": (NOW + timedelta(days=30)).isoformat(),
    }


def lease(**updates):
    return {
        "schema_version": 1,
        "call_id": str(UUID(int=2)),
        "lease_token": str(UUID(int=3)),
        "deployment_id": "agent-a",
        "original_retention_until": NOW.isoformat(),
        "lease_expires_at": (NOW + timedelta(seconds=30)).isoformat(),
        **updates,
    }


@pytest.mark.asyncio
async def test_begin_uses_owned_transaction_and_exact_reply():
    sink, pool, connection, _ = sink_with_rows([(snapshot(),)])
    assert callable(getattr(sink, "begin_call", None)), "missing native begin_call"
    admitted = routing()
    value = await sink.begin_call("agent-a", UUID(int=2), admitted)
    assert value.call_id == UUID(int=2)
    assert value.recording_enabled is False
    sql, params, prepare = connection.calls[0]
    assert sql == "SELECT voice.begin_call_v1(%s,%s,%s::jsonb)"
    assert params[:2] == ("agent-a", UUID(int=2))
    assert params[2].obj == admitted.model_dump(mode="json")
    assert prepare is False and connection.transaction_commits == 1 and pool.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_begin_consumes_the_explicit_pinned_company_recording_policy(enabled):
    sink, _, _, _ = sink_with_rows([({**snapshot(), "recording_enabled": enabled},)])
    value = await sink.begin_call("agent-a", UUID(int=2), routing())
    assert value.recording_enabled is enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [None, "true", "false", 0, 1])
async def test_begin_rejects_malformed_company_recording_policy(enabled):
    sink, _, connection, _ = sink_with_rows([({**snapshot(), "recording_enabled": enabled},)])
    with pytest.raises(OperationSinkContractError):
        await sink.begin_call("agent-a", UUID(int=2), routing())
    assert connection.transaction_rollbacks == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(None,)],
        [(snapshot(), "extra")],
        [({**snapshot(), "call_id": str(UUID(int=99))},)],
        [({**snapshot(), "workspace_id": "foreign"},)],
        [({**snapshot(), "retention_until": (NOW + timedelta(days=31)).isoformat()},)],
    ],
)
async def test_begin_rejects_wrong_shape_identity_and_retention(rows):
    sink, _, connection, _ = sink_with_rows(rows)
    assert callable(getattr(sink, "begin_call", None)), "missing native begin_call"
    with pytest.raises(OperationSinkContractError):
        await sink.begin_call("agent-a", UUID(int=2), routing())
    assert connection.transaction_rollbacks == 1


@pytest.mark.asyncio
async def test_begin_unknown_commit_is_ambiguous_same_identity_is_replayable():
    sink, _, connection, _ = sink_with_rows([(snapshot(),)], commit_error=OSError("raw"))
    assert callable(getattr(sink, "begin_call", None)), "missing native begin_call"
    admitted = routing()
    with pytest.raises(OperationSinkCommitAmbiguousError):
        await sink.begin_call("agent-a", UUID(int=2), admitted)
    connection.commit_error = None
    assert (await sink.begin_call("agent-a", UUID(int=2), admitted)).call_id == UUID(int=2)
    assert connection.calls[0][1][:2] == connection.calls[1][1][:2]
    assert connection.calls[0][1][2].obj == connection.calls[1][1][2].obj


@pytest.mark.asyncio
async def test_call_erasure_lease_and_ack_native_shapes():
    sink, _, connection, _ = sink_with_rows([(lease(),)])
    assert callable(getattr(sink, "lease_call_erasures", None)), "missing native erasure lease"
    values = await sink.lease_call_erasures("worker-1", 30, 1)
    assert values[0].call_id == UUID(int=2) and values[0].lease_token == UUID(int=3)
    assert connection.calls[0] == (
        "SELECT * FROM voice.lease_call_erasure_v1(%s,%s,%s)",
        ("worker-1", 30, 1),
        False,
    )
    connection.rows = [(None,)]
    await sink.ack_call_erasure(UUID(int=2), UUID(int=3), NOW)
    assert connection.calls[1] == (
        "SELECT voice.ack_call_erasure_v1(%s,%s,%s)",
        (UUID(int=2), UUID(int=3), NOW),
        False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [(None, None)], [("null",)], [({},)], [(None,), (None,)]])
async def test_call_ack_requires_exact_single_column_null(rows):
    sink, _, _, _ = sink_with_rows(rows)
    assert callable(getattr(sink, "ack_call_erasure", None)), "missing native erasure ack"
    with pytest.raises(OperationSinkContractError):
        await sink.ack_call_erasure(UUID(int=2), UUID(int=3), NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": True},
        {"deployment_id": ""},
        {"call_id": str(UUID(int=2)).upper()},
        {"lease_token": "invalid"},
        {"lease_expires_at": "2026-02-30T00:00:00Z"},
        {"extra": 1},
    ],
)
async def test_call_lease_rejects_malformed_rows(updates):
    # A UUID with letters makes uppercase distinguishable.
    data = lease(call_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    data.update(updates)
    if "call_id" in updates:
        data["call_id"] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    sink, _, _, _ = sink_with_rows([(data,)])
    assert callable(getattr(sink, "lease_call_erasures", None)), "missing native erasure lease"
    with pytest.raises(OperationSinkContractError):
        await sink.lease_call_erasures("worker-1", 30, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,exception",
    [("PV201", OperationSinkStaleLeaseError), ("PV202", OperationSinkPermanentError)],
)
async def test_call_ack_maps_stale_and_malformed_without_ambiguous_success(state, exception):
    sink, _, _, _ = sink_with_rows([], execute_error=SqlstateError(state))
    assert callable(getattr(sink, "ack_call_erasure", None)), "missing native erasure ack"
    with pytest.raises(exception):
        await sink.ack_call_erasure(UUID(int=2), UUID(int=3), NOW)


@pytest.mark.asyncio
async def test_only_pv301_ingest_is_erased():
    assert hasattr(sink_module, "OperationSinkErasedError"), "missing erased SQLSTATE mapping"
    for state, exception in [
        ("PV301", sink_module.OperationSinkErasedError),
        ("PV202", OperationSinkCommitAmbiguousError),
    ]:
        sink, _, _, _ = sink_with_rows([], execute_error=SqlstateError(state))
        with pytest.raises(exception) as captured:
            await sink.ingest(operation())
        if state != "PV301":
            assert not isinstance(captured.value, sink_module.OperationSinkErasedError)


@pytest.mark.asyncio
async def test_call_erasure_ack_unknown_commit_remains_ambiguous():
    sink, _, _, _ = sink_with_rows([(None,)], commit_error=OSError("raw"))
    with pytest.raises(OperationSinkCommitAmbiguousError):
        await sink.ack_call_erasure(UUID(int=2), UUID(int=3), NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        [(lease(),), (lease(),)],
        [(lease(),), (lease(call_id=str(UUID(int=4))),)],
        [(None,)],
        [(lease(), "extra")],
    ],
)
async def test_call_erasure_lease_rejects_duplicate_identity_token_and_shape(rows):
    sink, _, _, _ = sink_with_rows(rows)
    with pytest.raises(OperationSinkContractError):
        await sink.lease_call_erasures("worker-1", 30, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [("worker-1", True, 1), ("worker-1", 301, 1), ("worker-1", 30, 101), ("bad worker", 30, 1)],
)
async def test_call_erasure_lease_inputs_fail_before_sql(arguments):
    sink, _, connection, _ = sink_with_rows([])
    with pytest.raises(ValueError):
        await sink.lease_call_erasures(*arguments)
    assert connection.calls == []


@pytest.mark.asyncio
async def test_new_begin_and_erasure_lease_malformed_sqlstate_is_permanent():
    sink, _, _, _ = sink_with_rows([], execute_error=SqlstateError("PV202"))
    with pytest.raises(OperationSinkPermanentError):
        await sink.begin_call("agent-a", UUID(int=2), routing())
    with pytest.raises(OperationSinkPermanentError):
        await sink.lease_call_erasures("worker-1", 30, 1)
