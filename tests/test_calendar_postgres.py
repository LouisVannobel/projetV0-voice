from uuid import uuid4

import pytest

from projetv0_voice.persistence.postgres_sink import OperationSinkPermanentError
from tests.contract.test_postgres_sink import SqlstateError, sink_with_rows


@pytest.mark.asyncio
async def test_catalog_null_and_safe_refusal():
    sink, pool, connection, _ = sink_with_rows([(None,)])
    await sink.open()
    method = getattr(sink, "calendar_catalog", None)
    assert method is not None, "missing native Calendar sink"
    assert await method(uuid4()) is None
    connection.execute_error = SqlstateError("PV202")
    with pytest.raises(OperationSinkPermanentError):
        await method(uuid4())
    await sink.close()


@pytest.mark.asyncio
async def test_calendar_canonical_operation_id_and_transaction_release():
    canonical = uuid4()
    sink, pool, connection, _ = sink_with_rows(
        [({"state": "waiting_evidence", "operationId": str(canonical)},)]
    )
    await sink.open()
    method = getattr(sink, "calendar_submit", None)
    assert method is not None, "missing native Calendar submit"
    result = await method(
        uuid4(),
        uuid4(),
        {
            "kind": "find_slots",
            "serviceId": "oil",
            "policyRevision": 1,
            "windowStart": "2026-10-26T08:00:00Z",
            "windowEnd": "2026-10-26T11:00:00Z",
        },
    )
    assert result.operationId == canonical
    assert result.state == "waiting_evidence"
    assert connection.transaction_commits == 1
    assert connection.events[-1] == "transaction-exit"
    await sink.close()


@pytest.mark.asyncio
async def test_ambiguous_and_waiting_evidence_retry_frozen_payload_then_poll_canonical_id():
    from projetv0_voice.calendar import CalendarResult, await_calendar_result
    from projetv0_voice.persistence.postgres_sink import OperationSinkCommitAmbiguousError

    original, canonical, call = uuid4(), uuid4(), uuid4()
    request = {"kind": "book_appointment", "offerId": str(uuid4())}
    submissions, polls = [], []

    class Sink:
        async def calendar_submit(self, call_id, operation_id, payload):
            submissions.append((call_id, operation_id, payload.copy()))
            payload["offerId"] = "mutated-by-peer"
            if len(submissions) == 1:
                raise OperationSinkCommitAmbiguousError("operation_commit_ambiguous")
            return CalendarResult.model_validate(
                {
                    "state": "waiting_evidence" if len(submissions) == 2 else "pending",
                    "operationId": str(canonical),
                }
            )

        async def calendar_result(self, call_id, operation_id):
            polls.append((call_id, operation_id))
            return CalendarResult.model_validate(
                {
                    "state": "unknown",
                    "operationId": str(canonical),
                    "reason": "appointment_to_confirm",
                }
            )

    result = await await_calendar_result(Sink(), call, original, request, poll_delay=0)
    assert result.state == "unknown"
    assert submissions == [(call, original, request)] * 3
    assert polls == [(call, canonical)]


@pytest.mark.asyncio
async def test_pending_poll_is_bounded_without_claiming_success_or_new_operation():
    from projetv0_voice.calendar import CalendarResult, await_calendar_result

    operation = uuid4()
    calls = []

    class Sink:
        async def calendar_submit(self, call, identifier, request):
            calls.append(identifier)
            return CalendarResult.model_validate(
                {"state": "pending", "operationId": str(operation)}
            )

        async def calendar_result(self, call, identifier):
            return CalendarResult.model_validate(
                {"state": "pending", "operationId": str(operation)}
            )

    assert (
        await await_calendar_result(
            Sink(), uuid4(), operation, {}, wait_seconds=0.02, poll_delay=0.005
        )
        is None
    )
    assert calls == [operation]
