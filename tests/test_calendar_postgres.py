import asyncio
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

import pytest

from projetv0_voice.persistence.postgres_sink import OperationSinkPermanentError
from tests.contract.test_postgres_sink import SqlstateError, sink_with_rows


@contextmanager
def _calendar_timeout_after_delays(delays_before_timeout):
    """Schedule unit branches before the real timeout; do not measure wall time."""
    loop = asyncio.get_running_loop()
    native_sleep = asyncio.sleep
    now = loop.time()
    delays = 0

    async def controlled_sleep(delay):
        nonlocal now, delays
        assert delay == 0.001
        delays += 1
        if delays > delays_before_timeout:
            now += 0.031
            # Let the native timeout callback cancel this task once due.
            await native_sleep(0)
            await native_sleep(0)
            raise AssertionError("native Calendar timeout did not cancel")
        await native_sleep(0)

    with patch.object(loop, "time", side_effect=lambda: now), patch.object(
        asyncio, "sleep", controlled_sleep,
    ):
        yield


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

    result = await await_calendar_result(
        Sink(), call, original, request, wait_seconds=0.1, poll_delay=0
    )
    assert result.state == "unknown"
    assert submissions == [(call, original, request)] * 3
    assert polls and all(poll == (call, canonical) for poll in polls)


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


@pytest.mark.asyncio
async def test_pending_unknown_confirmed_polls_one_canonical_submission():
    from projetv0_voice.calendar import CalendarResult, await_calendar_result

    original, canonical, call = uuid4(), uuid4(), uuid4()
    submissions, polls = [], []
    booking = {
        "serviceId": "oil", "serviceName": "Vidange", "timezone": "Europe/Paris",
        "start": "2026-10-26T08:00:00Z", "end": "2026-10-26T08:30:00Z",
        "providerReference": "synthetic-verified-event",
    }

    class Sink:
        async def calendar_submit(self, call_id, operation_id, request):
            submissions.append((call_id, operation_id, request))
            return CalendarResult.model_validate(
                {"state": "pending", "operationId": str(canonical)}
            )

        async def calendar_result(self, call_id, operation_id):
            polls.append((call_id, operation_id))
            return CalendarResult.model_validate(
                {"state": "unknown", "operationId": str(canonical),
                 "reason": "appointment_to_confirm"}
                if len(polls) == 1 else
                {"state": "confirmed", "operationId": str(canonical), "booking": booking}
            )

    result = await await_calendar_result(Sink(), call, original, {}, poll_delay=0)
    assert result.state == "confirmed"
    assert result.booking.providerReference == "synthetic-verified-event"
    assert submissions == [(call, original, {})]
    assert polls == [(call, canonical)] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("after_unknown", [
    "transient", "ambiguous", "missing", "waiting_evidence", "pending", "running",
    "foreign_unknown", "foreign_confirmed",
])
async def test_unknown_latches_read_only_and_canonical_identity_until_deadline(after_unknown):
    from projetv0_voice.calendar import CalendarResult, await_calendar_result
    from projetv0_voice.persistence.postgres_sink import (
        OperationSinkCommitAmbiguousError,
        OperationSinkTransientError,
    )

    original, canonical, foreign, call = uuid4(), uuid4(), uuid4(), uuid4()
    submissions, polls = [], []
    unknown = CalendarResult.model_validate(
        {"state": "unknown", "operationId": str(canonical), "reason": "appointment_to_confirm"}
    )

    class Sink:
        async def calendar_submit(self, call_id, operation_id, request):
            submissions.append((call_id, operation_id))
            return unknown

        async def calendar_result(self, call_id, operation_id):
            polls.append((call_id, operation_id))
            if after_unknown == "transient":
                raise OperationSinkTransientError("operation_transient")
            if after_unknown == "ambiguous":
                raise OperationSinkCommitAmbiguousError("operation_commit_ambiguous")
            response = {
                "state": after_unknown.removeprefix("foreign_"),
                "operationId": str(foreign if after_unknown.startswith("foreign_") else canonical),
            }
            if after_unknown == "foreign_confirmed":
                response["booking"] = {
                    "serviceId": "oil", "serviceName": "Vidange", "timezone": "Europe/Paris",
                    "start": "2026-10-26T08:00:00Z", "end": "2026-10-26T08:30:00Z",
                    "providerReference": "foreign-synthetic-event",
                }
            return CalendarResult.model_validate(response)

    # Two polls also exercise the next iteration after each post-unknown branch.
    with _calendar_timeout_after_delays(2):
        result = await await_calendar_result(
            Sink(), call, original, {}, wait_seconds=0.03, poll_delay=0.001
        )
    assert result is unknown
    assert submissions == [(call, original)]
    assert polls and all(poll == (call, canonical) for poll in polls)
    assert polls == [(call, canonical)] * 2


@pytest.mark.asyncio
async def test_unknown_is_retained_when_deadline_expires_before_first_poll():
    from projetv0_voice.calendar import CalendarResult, await_calendar_result

    original, canonical, call = uuid4(), uuid4(), uuid4()
    submissions, polls = [], []
    unknown = CalendarResult.model_validate(
        {"state": "unknown", "operationId": str(canonical), "reason": "appointment_to_confirm"}
    )

    class Sink:
        async def calendar_submit(self, call_id, operation_id, request):
            submissions.append((call_id, operation_id))
            return unknown

        async def calendar_result(self, call_id, operation_id):
            polls.append((call_id, operation_id))
            raise AssertionError("Calendar read after deadline")

    with _calendar_timeout_after_delays(0):
        result = await await_calendar_result(
            Sink(), call, original, {}, wait_seconds=0.03, poll_delay=0.001
        )
    assert result is unknown
    assert submissions == [(call, original)]
    assert polls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("after_unknown", [False, True])
async def test_waiter_permanent_failure_remains_failure(after_unknown):
    from projetv0_voice.calendar import CalendarResult, await_calendar_result

    operation = uuid4()

    class Sink:
        async def calendar_submit(self, *_args):
            if not after_unknown:
                raise OperationSinkPermanentError("operation_permanent")
            return CalendarResult.model_validate(
                {"state": "unknown", "operationId": str(operation)}
            )

        async def calendar_result(self, *_args):
            raise OperationSinkPermanentError("operation_permanent")

    assert await await_calendar_result(Sink(), uuid4(), operation, {}, poll_delay=0) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "unknown"])
async def test_waiter_cancellation_propagates(state):
    import asyncio

    from projetv0_voice.calendar import CalendarResult, await_calendar_result

    operation = uuid4()
    reading = asyncio.Event()

    class Sink:
        async def calendar_submit(self, *_args):
            return CalendarResult.model_validate({"state": state, "operationId": str(operation)})

        async def calendar_result(self, *_args):
            reading.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(await_calendar_result(Sink(), uuid4(), operation, {}, poll_delay=0))
    await asyncio.wait_for(reading.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
