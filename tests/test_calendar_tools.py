"""Bounded native Calendar contracts; no provider network."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from projetv0_voice import calendar


def api(name):
    value = getattr(calendar, name, None)
    assert value is not None, f"missing Calendar contract: {name}"
    return value


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Oui.", True),
        ("Oui, je confirme.", True),
        ("D’accord", True),
        ("oui mais jeudi", False),
        ("non", False),
        ("Il a dit oui", False),
        ('"oui"', False),
        ("oui annulez", False),
        ("", False),
    ],
)
def test_only_full_affirmative(text, expected):
    assert api("is_affirmative")(text) is expected


@pytest.mark.parametrize(
    "bad",
    [
        "2026-02-30T10:00:00Z",
        "2026-10-21",
        "2026-10-21T10:00:00",
        "2026-10-21T10:00:00Zjunk",
        "2026-10-21T10:00:00.1234567Z",
    ],
)
def test_strict_calendar_instant(bad):
    parse = api("calendar_instant")
    with pytest.raises(ValueError):
        parse(bad)
    assert parse("2026-10-21T10:00:00+02:00") == datetime(2026, 10, 21, 8, tzinfo=UTC)


def test_recap_requires_owned_audio_forwarding_and_ack_then_first_started_turn():
    playback = api("CalendarPlayback")()
    offer = uuid4()
    attempt = playback.begin(offer, "Vidange lundi 26 octobre à 9 heures. Confirmez-vous ?")
    playback.before_tts_frame(attempt.speak_frame)
    playback.bind_context("recap-context")
    playback.after_tts_frame(attempt.speak_frame)
    playback.note_audio("foreign")
    assert not playback.arm_mark(attempt.mark_name)
    playback.note_audio("recap-context")
    assert playback.arm_mark(attempt.mark_name)
    playback.accept_mark(attempt.mark_name)
    assert attempt.acknowledged_at is None
    playback.mark_forwarded(attempt.mark_name)
    playback.accept_mark(attempt.mark_name)
    assert attempt.acknowledged_at is not None
    playback.caller_started(1)
    playback.interrupted()  # ordinary native post-ACK interruption preserves proof
    assert playback.confirmation_for(1, "Oui.") is attempt
    assert playback.confirmation_for(1, "Oui.") is None


@pytest.mark.parametrize("mode", ["pre_ack", "clear", "correction", "second_turn"])
def test_recap_no_authority_after_incomplete_or_corrected_playback(mode):
    playback = api("CalendarPlayback")()
    attempt = playback.begin(uuid4(), "Confirmez-vous ?")
    playback.before_tts_frame(attempt.speak_frame)
    playback.bind_context("recap")
    playback.note_audio("recap")
    assert playback.arm_mark(attempt.mark_name)
    playback.mark_forwarded(attempt.mark_name)
    if mode == "pre_ack":
        playback.caller_started(1)
    if mode == "clear":
        playback.interrupted()
    playback.accept_mark(attempt.mark_name)
    if mode != "pre_ack":
        playback.caller_started(1)
    if mode == "second_turn":
        playback.caller_started(2)
    assert (
        playback.confirmation_for(1, "oui mais jeudi" if mode == "correction" else "Oui.") is None
    )
    assert playback.confirmation_for(1, "Oui.") is None


@pytest.mark.asyncio
async def test_calendar_native_catalog_and_unknown_service_no_submit():
    import asyncio
    import json

    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import active, caller, observe_result, tool

    class Sink:
        def __init__(self):
            self.calls = []

        async def calendar_catalog(self, call_id):
            return calendar.CalendarCatalog.model_validate(
                {
                    "schema_version": 1,
                    "call_id": str(call_id),
                    "policy_revision": 1,
                    "timezone": "Europe/Paris",
                    "services": [{"service_id": "oil", "name": "Vidange", "duration_minutes": 30}],
                }
            )

        async def calendar_submit(self, *args):
            self.calls.append(args)
            raise AssertionError("unknown service reached database")

    sink = Sink()
    async with _native_sparra_opening(calendar_sink=sink) as fixture:
        await active(fixture)
        fixture.peers.tool_calls = [
            tool(
                "find_slots",
                arguments=json.dumps(
                    {
                        "serviceId": "unknown",
                        "windowStart": "2026-10-26T08:00:00Z",
                        "windowEnd": "2026-10-26T11:00:00Z",
                    }
                ),
            )
        ]
        done = asyncio.Event()
        observe_result(fixture, done)
        await caller(fixture, "Je voudrais un rendez-vous pour une vidange lundi.")
        await asyncio.wait_for(done.wait(), 3)
        assert sink.calls == []
        schemas = fixture.peers.conversations[0]["tools"]
        assert {item["function"]["name"] for item in schemas} >= {
            "find_slots",
            "book_appointment",
        }
        prompt = str(fixture.peers.conversations[0]["messages"])
        assert "Europe/Paris" in prompt and "Vidange" in prompt


class CalendarPeer:
    def __init__(self):
        import asyncio

        self.submitted = []
        self.booked = asyncio.Event()
        self.release = asyncio.Event()
        self.hold_booking = False
        self.offer_id = uuid4()
        self.operation_id = uuid4()
        self.now = datetime.now(UTC)
        self.offer = {
            "offerId": str(self.offer_id),
            "serviceId": "oil",
            "serviceName": "Vidange",
            "timezone": "Europe/Paris",
            "start": (self.now + timedelta(days=2)).isoformat(),
            "end": (self.now + timedelta(days=2, minutes=30)).isoformat(),
            "expiresAt": (self.now + timedelta(minutes=5)).isoformat(),
        }

    async def calendar_catalog(self, call_id):
        return calendar.CalendarCatalog.model_validate(
            {
                "schema_version": 1,
                "call_id": str(call_id),
                "policy_revision": 1,
                "timezone": "Europe/Paris",
                "services": [{"service_id": "oil", "name": "Vidange", "duration_minutes": 30}],
            }
        )

    async def calendar_submit(self, call, operation, request):
        self.submitted.append((operation, request))
        if request["kind"] == "find_slots":
            return calendar.CalendarResult.model_validate(
                {"state": "offered", "operationId": str(self.operation_id), "offers": [self.offer]}
            )
        self.booked.set()
        if self.hold_booking:
            await self.release.wait()
        return calendar.CalendarResult.model_validate(
            {
                "state": "confirmed",
                "operationId": str(operation),
                "booking": {
                    key: value
                    for key, value in {
                        **self.offer,
                        "providerReference": "verified-event",
                    }.items()
                    if key not in {"offerId", "expiresAt"}
                },
            }
        )


async def until(predicate):
    import asyncio

    async with asyncio.timeout(8):
        for _ in range(800):
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("native observation deadline")


async def native_recap(fixture, sink, *, mixed=False):
    import json

    from tests.integration.test_voice_action_authority import active, caller, tool

    await active(fixture)
    fixture.peers.tool_calls = [
        tool(
            "find_slots",
            "search-1",
            json.dumps(
                {
                    "serviceId": "oil",
                    "windowStart": sink.offer["start"],
                    "windowEnd": sink.offer["end"],
                }
            ),
        )
    ]
    await caller(fixture, "Je voudrais une vidange après-demain.")
    await until(lambda: any("Quel créneau" in text for text in fixture.peers.texts))
    fixture.peers.tool_calls = [
        tool(
            "book_appointment",
            "select-1",
            json.dumps(
                {
                    "offerId": str(sink.offer_id),
                }
            ),
        )
    ]
    await caller(fixture, "Ce créneau me convient.")
    await until(
        lambda: any(
            wire.get("event") == "mark"
            and wire["mark"]["name"].startswith("sparra-calendar-recap-")
            for wire in fixture.sent
        )
    )
    return fixture.session._calendar_playback.attempt


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed", [False, True])
async def test_native_recap_ack_affirmative_submits_exact_persisted_rows_without_inference(mixed):
    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import caller

    sink = CalendarPeer()
    async with _native_sparra_opening(calendar_sink=sink) as fixture:
        attempt = await native_recap(fixture, sink, mixed=mixed)
        await until(lambda: attempt.acknowledged_at is not None)
        assert attempt.receipt is not None, "native recap receipt was not bound"
        assert attempt.receipt.settled.is_set(), "native recap receipt not settled"
        assert attempt.receipt.turn_id is not None, "native recap row rejected"
        before = len(fixture.peers.conversations)
        await caller(fixture, "Oui, je confirme.")
        assert fixture.session._calendar_task is not None, (
            "confirmation was not consumed",
            attempt.confirmation_start_id,
            attempt.invalidated,
            fixture.session._action_authority.started_id,
            fixture.session._action_failure.code,
        )
        await until(lambda: sink.booked.is_set())
        await until(
            lambda: any("Votre rendez-vous est confirmé" in text for text in fixture.peers.texts)
        )
        assert len(fixture.peers.conversations) == before
        assert len(sink.submitted) == 2
        request = sink.submitted[-1][1]
        proof = request["confirmation"]
        rows = {
            str(operation.payload.turn_id): operation.payload for operation in fixture.writer.turns
        }
        recap, confirmation = rows[proof["recapTurnId"]], rows[proof["confirmationTurnId"]]
        assert recap.role == "assistant" and recap.source == "pipecat_assistant"
        assert confirmation.role == "user" and confirmation.source == "stt_final"
        ack = datetime.fromisoformat(proof["playbackAcknowledgedAt"])
        assert recap.ended_at <= ack <= confirmation.started_at <= confirmation.ended_at
        assert confirmation.turn_no == recap.turn_no + 1
        assert not recap.interrupted and not confirmation.interrupted
        assert fixture.tts._farewell_context_id is None


@pytest.mark.asyncio
async def test_delayed_identical_native_turn_callbacks_keep_exact_admission_receipt(monkeypatch):
    import asyncio

    from pipecat.utils.base_object import BaseObject

    from projetv0_voice.action_authority import ActionAuthority
    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import active, caller

    scopes = []
    release = asyncio.Event()
    delayed = asyncio.Event()
    capture = ActionAuthority.capture
    run_handler = BaseObject._run_handler

    def capture_exact(self, *args, **kwargs):
        scope = capture(self, *args, **kwargs)
        if scope is not None:
            scopes.append(scope)
        return scope

    async def delay_first(self, name, handler, *args, **kwargs):
        if (
            name == "on_user_turn_stopped"
            and handler.__name__ == "record_user_turn"
            and not delayed.is_set()
        ):
            delayed.set()
            await release.wait()
        await run_handler(self, name, handler, *args, **kwargs)

    monkeypatch.setattr(ActionAuthority, "capture", capture_exact)
    monkeypatch.setattr(BaseObject, "_run_handler", delay_first)
    async with _native_sparra_opening() as fixture:
        try:
            await active(fixture)
            await caller(fixture, "Oui.")
            await caller(fixture, "Oui.")
            assert len(scopes) == 2
            first, second = scopes
            assert first.receipt is not second.receipt
            await until(lambda: second.receipt.settled.is_set())
            assert not first.receipt.settled.is_set()
            second_id = second.receipt.turn_id
            release.set()
            await until(lambda: first.receipt.settled.is_set())
            assert first.receipt.turn_id != second_id
            rows = {
                operation.payload.turn_id: operation.payload for operation in fixture.writer.turns
            }
            assert rows[first.receipt.turn_id].started_at < rows[second_id].started_at
            assert not fixture.session._action_authority.current(first)
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["missing_ack", "corrective_yes", "clear_after_ack", "stale_result", "hangup"]
)
async def test_native_recap_rejects_missing_or_changed_consent_and_suppresses_stale_result(case):
    import json

    from pipecat.frames.frames import InterruptionFrame

    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import caller

    sink = CalendarPeer()
    async with _native_sparra_opening(
        calendar_sink=sink, ack_calendar=case != "missing_ack"
    ) as fixture:
        attempt = await native_recap(fixture, sink)
        if case != "missing_ack":
            await until(lambda: attempt.acknowledged_at is not None)
        if case == "clear_after_ack":
            await fixture.session._active_runtime.worker.queue_frame(InterruptionFrame())
            await until(lambda: attempt.invalidated)
        if case in {"stale_result", "hangup"}:
            sink.hold_booking = True
        fixture.peers.tool_calls = None
        await caller(fixture, "Oui mais jeudi." if case == "corrective_yes" else "Oui.")
        if case == "hangup":
            await until(lambda: sink.booked.is_set())
            await fixture.session.request_drain()
            await fixture.running
            assert fixture.session._calendar_task.done()
            assert fixture.session._calendar_pending_booking is not None
            assert len(sink.submitted) == 2
        elif case == "stale_result":
            await until(lambda: sink.booked.is_set())
            await caller(fixture, "Attendez, je dois vérifier.")
            sink.release.set()
            await until(lambda: not fixture.session._calendar_submitting)
            assert len(sink.submitted) == 2
        else:
            assert not sink.booked.is_set()
            if case == "missing_ack":
                await fixture.incoming.put(
                    {
                        "type": "websocket.receive",
                        "text": json.dumps(
                            {
                                "event": "mark",
                                "stream_id": "stream-1",
                                "mark": {"name": attempt.mark_name},
                            }
                        ),
                    }
                )
                await caller(fixture, "Oui.")
                assert not sink.booked.is_set()
        assert not any("Votre rendez-vous est confirmé" in text for text in fixture.peers.texts)


@pytest.mark.parametrize(
    "name,args",
    [
        ("book_appointment", {"offerId": str(uuid4()), "confirmed": True}),
        ("book_appointment", {"offerId": "foreign-not-uuid"}),
        ("find_slots", {"serviceId": "oil", "duration": 30}),
        (
            "find_slots",
            {"serviceId": "oil", "windowStart": "tomorrow", "windowEnd": "next week"},
        ),
    ],
)
def test_model_cannot_supply_proof_duration_or_unbounded_date_arguments(name, args):
    assert not calendar.valid_calendar_arguments(name, args)


@pytest.mark.asyncio
async def test_unknown_booking_retains_original_identity_and_refuses_overlapping_intent():
    import json

    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import caller, tool

    sink = CalendarPeer()
    original_submit = sink.calendar_submit

    async def uncertain(call_id, operation_id, request):
        if request["kind"] == "find_slots":
            return await original_submit(call_id, operation_id, request)
        sink.submitted.append((operation_id, request))
        sink.booked.set()
        return calendar.CalendarResult.model_validate(
            {
                "state": "unknown",
                "operationId": str(operation_id),
                "reason": "appointment_to_confirm",
            }
        )

    sink.calendar_submit = uncertain
    async with _native_sparra_opening(calendar_sink=sink) as fixture:
        attempt = await native_recap(fixture, sink)
        await until(lambda: attempt.acknowledged_at is not None)
        await caller(fixture, "Oui.")
        await until(lambda: sink.booked.is_set() and not fixture.session._calendar_submitting)
        fixture.peers.tool_calls = [
            tool(
                "book_appointment", "repeat-intent", json.dumps({"offerId": str(sink.offer_id)})
            )
        ]
        await caller(fixture, "Réessayez ce rendez-vous.")
        await until(lambda: len(fixture.peers.conversations) >= 3)
        assert fixture.session._calendar_playback.attempt is None
        assert len(sink.submitted) == 2
        frozen = fixture.session._calendar_pending_booking
        assert frozen[0] == sink.submitted[-1][0]
        assert frozen[1] == sink.submitted[-1][1]


@pytest.mark.asyncio
async def test_native_six_second_paced_recap_has_separate_forwarding_allowance():
    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import caller

    sink = CalendarPeer()
    async with _native_sparra_opening(calendar_sink=sink) as fixture:
        fixture.peers.recap_pcm_samples = 6 * 24000
        attempt = await native_recap(fixture, sink)
        await until(lambda: attempt.acknowledged_at is not None)
        assert attempt.receipt.turn_id is not None
        rows = {operation.payload.turn_id: operation.payload for operation in fixture.writer.turns}
        recap = rows[attempt.receipt.turn_id]
        assert (attempt.acknowledged_at - recap.started_at).total_seconds() > 5
        await caller(fixture, "Oui.")
        await until(lambda: sink.booked.is_set())
        assert not attempt.invalidated


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["foreign_offer", "model_consent", "batch", "reused_id"])
async def test_native_calendar_tool_guard_refuses_foreign_consent_batch_and_reused_ids(case):
    import asyncio
    import json
    from dataclasses import replace

    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import active, caller, tool

    sink = CalendarPeer()
    async with _native_sparra_opening(calendar_sink=sink) as fixture:
        await active(fixture)
        callbacks = []
        handler = fixture.session._calendar_tool

        async def observe_callback(params):
            async def callback(result, *, properties):
                callbacks.append((params.tool_call_id, properties.run_llm))
                await params.result_callback(result, properties=properties)

            await handler(replace(params, result_callback=callback))

        fixture.llm.register_function("find_slots", observe_callback)
        fixture.llm.register_function("book_appointment", observe_callback)
        if case == "reused_id":
            fixture.peers.tool_calls = [
                tool(
                    "find_slots",
                    "same-id",
                    json.dumps(
                        {
                            "serviceId": "oil",
                            "windowStart": sink.offer["start"],
                            "windowEnd": sink.offer["end"],
                        }
                    ),
                )
            ]
            await caller(fixture, "Une vidange après-demain.")
            await until(lambda: len(callbacks) == 1)
        args = {"offerId": str(uuid4())}
        if case == "model_consent":
            args["confirmed"] = True
        fixture.peers.tool_calls = [tool("book_appointment", "same-id", json.dumps(args))]
        if case == "batch":
            fixture.peers.separate_tool_chunks = True
            fixture.peers.tool_calls.append(
                tool("book_appointment", "second-id", json.dumps(args), index=1)
            )
        before = len(callbacks)
        await caller(fixture, "Ce rendez-vous.")
        await until(lambda: len(callbacks) == before + len(fixture.peers.tool_calls))
        assert all(run_llm is False for _, run_llm in callbacks)
        assert fixture.session._calendar_playback.attempt is None
        assert not sink.booked.is_set()
        assert len(sink.submitted) == (1 if case == "reused_id" else 0)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_native_confirmation_started_before_ack_cannot_be_repaired_by_late_ack():
    import asyncio
    import json

    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_voice_action_authority import caller

    sink = CalendarPeer()
    async with _native_sparra_opening(calendar_sink=sink, ack_calendar=False) as fixture:
        attempt = await native_recap(fixture, sink)
        fixture.peers.tool_calls = None
        speaking = asyncio.create_task(caller(fixture, "Oui."))
        try:
            await until(lambda: attempt.invalidated)
            await fixture.incoming.put(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "event": "mark",
                            "stream_id": "stream-1",
                            "mark": {"name": attempt.mark_name},
                        }
                    ),
                }
            )
            await speaking
            assert attempt.acknowledged_at is None
            assert not sink.booked.is_set()
        finally:
            if not speaking.done():
                speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (
            "2026-10-26T09:00:00Z",
            "2026-10-26T09:30:00Z",
            "Vidange, lundi 26 octobre de 10 h à 10 h 30, heure de Paris",
        ),
        (
            "2026-10-25T00:30:00Z",
            "2026-10-25T01:30:00Z",
            "Vidange, dimanche 25 octobre de 2 h 30 (heure d’été) "
            "à 2 h 30 (heure d’hiver), heure de Paris",
        ),
        (
            "2026-10-26T22:30:00Z",
            "2026-10-26T23:30:00Z",
            "Vidange, lundi 26 octobre à 23 h 30, "
            "jusqu’au mardi 27 octobre à 0 h 30, heure de Paris",
        ),
    ],
)
def test_appointment_labels_are_natural_and_dst_midnight_unambiguous(start, end, expected):
    offer = calendar.CalendarOffer.model_validate(
        {
            "offerId": str(uuid4()),
            "serviceId": "oil",
            "serviceName": "Vidange",
            "timezone": "Europe/Paris",
            "start": start,
            "end": end,
            "expiresAt": "2026-10-21T12:00:00Z",
        }
    )
    assert calendar.appointment_label(offer) == expected
    assert "Europe/Paris" not in expected and "UTC" not in expected


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-21T10:00:00+00:99",
        "0001-01-01T00:00:00+01:00",
        "9999-12-31T23:59:59-01:00",
    ],
)
def test_calendar_instant_rejects_invalid_offset_minutes_instead_of_normalizing(value):
    with pytest.raises(ValueError):
        calendar.calendar_instant(value)
