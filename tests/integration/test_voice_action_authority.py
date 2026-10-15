from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    EndFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.transcriptions.language import Language

from projetv0_voice.action_authority import direct_command
from tests.integration.sparra_connected_scenario import Scenario
from tests.integration.test_call_session import (
    NOW,
    _native_sparra_opening,
    _opening_event,
    session_module,
)


async def active(fixture):
    fixture.writer.ack_release.set()
    fixture.writer.gate_release.set()
    await fixture.session._controller.join_continuations()
    await _opening_event(fixture.peers.invitation, "invitation missing")
    for _ in range(300):
        if fixture.writer.turns:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("invitation not persisted")


async def caller(fixture, text):
    runtime = fixture.session._active_runtime
    aggregator = next(p for p in runtime.pipeline.processors if isinstance(p, LLMUserAggregator))
    stopped = asyncio.Event()

    def stop(_aggregator, _strategy, _message):
        stopped.set()

    aggregator.add_event_handler("on_user_turn_stopped", stop)
    try:
        await runtime.worker.queue_frame(TranscriptionFrame(
            text=text, user_id="", timestamp=NOW.isoformat(), language=Language.FR, finalized=True,
        ))
        # Reuse the established native caller bound, separately from the opening
        # fixture's 3-second ACK helper. This event is an observation, no authority.
        await asyncio.wait_for(stopped.wait(), timeout=8)
    finally:
        aggregator.remove_event_handler("on_user_turn_stopped", stop)


def tool(name="end_call", tool_id="action-one", arguments="{}", index=0):
    return {"index": index, "id": tool_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def observe_result(fixture, event):
    def pushed(_llm, frame):
        if isinstance(frame, FunctionCallResultFrame):
            assert frame.properties.run_llm is False
            event.set()
    fixture.llm.add_event_handler("on_after_push_frame", pushed)


def transfer_consumer():
    async def available():
        return True

    async def forbidden(**_kwargs):
        raise AssertionError("proposal performed a transfer")

    return SimpleNamespace(human_tool_available=available, request_human=forbidden,
                           transfer_to_cleanup=lambda _task: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "J'ai dit au revoir.", "Au revoir, merci pour votre aide.",
    "Je vous demande de raccrocher maintenant.",
])
async def test_current_native_end_tool_starts_owned_goodbye_after_its_single_callback(text):
    # The scripted SSE proves current-turn execution, not semantic model selection.
    assert direct_command(text) is None
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.end_call = True
        results, order = [], []
        completed = asyncio.Event()
        handler = fixture.session._end_call_tool

        async def recorded_handler(params):
            async def callback(result, *, properties):
                results.append((params.arguments, result, properties.run_llm))
                await params.result_callback(result, properties=properties)
                order.append("callback")
                completed.set()

            await handler(replace(params, result_callback=callback))

        def observe(_llm, frame):
            if isinstance(frame, TTSSpeakFrame) and frame.text == "Bonne journée, au revoir.":
                order.append("programmed-goodbye")

        fixture.llm.register_function("end_call", recorded_handler)
        fixture.llm.add_event_handler("on_after_push_frame", observe)
        await caller(fixture, text)
        await _opening_event(completed, "native result callback missing")
        assert results == [({}, {"status": "ending"}, False)]
        await _opening_event(fixture.final_mark_sent, "current native tool did not play goodbye")
        assert order == ["callback", "programmed-goodbye"]
        assert len(fixture.peers.conversations) == 1
        request = fixture.peers.conversations[0]
        instruction = request["messages"][0]["content"]
        description = request["tools"][0]["function"]["description"]
        assert all(word in instruction for word in ("intention présente", "citation", "négation"))
        assert "clarification" in instruction and "sans prononcer" in instruction
        assert "current caller turn" in description and "task completion" in description
        assert "without saying another" in description
        assert fixture.peers.texts[2:] == ["Bonne journée, au revoir."]
        assert fixture.session._end_call_playback.pending
        assert not fixture.running.done()
        final_mark = fixture.sent[-1]
        await fixture.incoming.put({"type": "websocket.receive", "text": json.dumps({
            **final_mark, "stream_id": "stream-1",
        })})
        await asyncio.wait_for(fixture.running, timeout=3)
        assert fixture.writer.commands[-1].payload.end_reason == "conversation_ended"


@pytest.mark.asyncio
async def test_current_native_end_tool_with_model_text_still_has_one_programmed_ending():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.end_call = True
        fixture.peers.text_before_tool = "Très bien."
        finished = []
        fixture.session._active_runtime.worker.add_event_handler(
            "on_pipeline_finished", lambda _worker, frame: finished.append(frame),
        )
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(fixture.final_mark_sent, "mixed native response did not play goodbye")
        assert "Très bien." in fixture.peers.texts
        assert fixture.peers.texts.count("Bonne journée, au revoir.") == 1
        assert "Pour terminer l'appel, dites : au revoir." not in fixture.peers.texts
        assert len(fixture.peers.conversations) == 1
        assert len([wire for wire in fixture.sent if wire.get("event") == "mark"]) == 2
        assert not fixture.running.done() and finished == []
        await fixture.incoming.put({"type": "websocket.receive", "text": json.dumps({
            **fixture.sent[-1], "stream_id": "stream-1",
        })})
        await asyncio.wait_for(fixture.running, timeout=3)
        assert len(finished) == 1 and isinstance(finished[0], EndFrame)
        assert finished[0].reason == "conversation_ended"
        assert fixture.writer.commands[-1].payload.end_reason == "conversation_ended"


@pytest.mark.asyncio
async def test_end_tool_revalidates_scope_after_callback_and_resolves_once():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        handler = fixture.session._end_call_tool
        callbacks = []
        completed = asyncio.Event()

        async def revoke_after_callback(params):
            async def callback(result, *, properties):
                callbacks.append((result, properties.run_llm))
                await params.result_callback(result, properties=properties)
                fixture.session._action_authority.invalidate_response()

            await handler(replace(params, result_callback=callback))
            completed.set()

        fixture.llm.register_function("end_call", revoke_after_callback)
        fixture.peers.end_call = True
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(completed, "revoked native handler did not resolve")
        assert callbacks == [({"status": "ending"}, False)]
        assert fixture.session._program_action_task is None
        assert not fixture.final_mark_sent.is_set()
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
async def test_consumed_current_tool_scope_cannot_start_another_goodbye():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        captured = []

        async def handler(params):
            captured.append(params)
            await fixture.session._end_call_tool(params)

        fixture.llm.register_function("end_call", handler)
        fixture.peers.end_call = True
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(fixture.final_mark_sent, "first native tool did not play goodbye")
        original_task = fixture.session._program_action_task
        results = []

        async def callback(result, *, properties):
            results.append((result, properties.run_llm))

        await fixture.session._end_call_tool(replace(captured[0], result_callback=callback))
        assert results == [({"success": False}, False)]
        assert fixture.session._program_action_task is original_task
        assert fixture.peers.texts.count("Bonne journée, au revoir.") == 1
        assert len([wire for wire in fixture.sent if wire.get("event") == "mark"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [None, False])
async def test_native_transfer_without_qualified_availability_keeps_conversation(available):
    async def human_tool_available():
        return False

    async def request_human(**_kwargs):
        raise AssertionError("unqualified direct command invoked transfer")

    registry = None if available is None else SimpleNamespace(
        human_tool_available=human_tool_available, request_human=request_human,
        transfer_to_cleanup=lambda _task: None,
    )
    async with _native_sparra_opening(registry_terminalizer=registry) as fixture:
        await active(fixture)
        await caller(fixture, "transférez-moi")
        await _opening_event(fixture.peers.llm_requested, "unavailable relay blocked conversation")
        for _ in range(300):
            if len(fixture.peers.texts) > 2:
                break
            await asyncio.sleep(0.01)
        assert fixture.peers.texts[2:] == ["Le garage est ouvert de 8 h à 18 h."]
        assert [entry["function"]["name"] for entry in fixture.peers.conversations[0]["tools"]] == [
            "end_call",
        ]
        assert not fixture.final_mark_sent.is_set()
        assert not fixture.running.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "Au revoir.", "merci au revoir", "MERCI, AU REVOIR !", "c'est tout merci au revoir",
    "c'est tout, merci, au revoir", "  C’est tout,   merci, au revoir.  ",
    "Non, ça va, merci, au revoir.",
    "Au revoir, bonne journée.", "Merci, au revoir, bonne soirée.",
])
async def test_native_direct_farewell_bypasses_model_inference(text):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        await caller(fixture, text)
        await _opening_event(fixture.final_mark_sent, "direct command did not play goodbye")
        assert fixture.peers.conversations == []
        assert fixture.peers.texts[2:] == ["Bonne journée, au revoir."]
        assert not fixture.running.done()


@pytest.mark.asyncio
async def test_connected_helper_uses_native_completed_caller_command_without_force_flush():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        probe = SimpleNamespace(session=fixture.session, peers=fixture.peers,
                                graph=SimpleNamespace(writer=fixture.writer), call_id=NOW)
        await Scenario.captured(probe, "Au revoir.")
        await _opening_event(fixture.final_mark_sent, "helper did not drive native direct goodbye")
        assert fixture.peers.conversations == []
        assert fixture.peers.texts[2:] == ["Bonne journée, au revoir."]


@pytest.mark.asyncio
async def test_native_direct_goodbye_failed_synthesis_cannot_end_conversation_normally():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.fail_goodbye = True
        await caller(fixture, "au revoir")
        with pytest.raises(session_module.CallSessionError):
            await asyncio.wait_for(fixture.running, timeout=12)
        assert fixture.peers.conversations == []
        assert not fixture.final_mark_sent.is_set()
        assert not fixture.session._end_call_playback.pending
        terminal = fixture.writer.commands[-1]
        assert terminal.payload.status == "failed"
        assert terminal.payload.end_reason != "conversation_ended"


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", [
    [tool(), tool(tool_id="action-two", index=1)],
    [tool(), tool("request_human", "action-two", index=1)],
    [tool(), tool(tool_id="action-one", index=1)],
    [tool(arguments=json.dumps({"destination": "+33111111111"}))],
    [tool(arguments="[]")],
])
async def test_native_conflicting_duplicate_or_malformed_batch_has_no_instruction(calls):
    async with _native_sparra_opening(
        ack_end_call=False, registry_terminalizer=transfer_consumer(),
    ) as fixture:
        await active(fixture)
        fixture.peers.tool_calls = calls
        completed = asyncio.Event()
        observe_result(fixture, completed)
        await caller(fixture, "Quels sont vos horaires ?")
        await _opening_event(completed, "tool batch did not finish")
        assert fixture.peers.texts[2:] == []
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("document_text", ["au revoir", "Non, ça va, merci, au revoir."])
async def test_native_document_context_has_no_action_scope_but_recovery_still_streams(
    document_text,
):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        aggregator = next(p for p in fixture.session._active_runtime.pipeline.processors
                          if isinstance(p, LLMUserAggregator))
        fixture.peers.end_call = True
        finished = asyncio.Event()
        observe_result(fixture, finished)
        await fixture.session._active_runtime.worker.queue_frame(
            LLMContextFrame(aggregator._context),
        )
        await _opening_event(finished, "document proposal did not finish")
        assert fixture.peers.texts[2:] == []
        fixture.peers.end_call = False
        fixture.peers.llm_requested.clear()
        await fixture.session._active_runtime.worker.queue_frame(LLMContextFrame(LLMContext([
            {"role": "assistant", "content": document_text},
        ])))
        await _opening_event(fixture.peers.llm_requested, "no-scope recovery did not infer")
        for _ in range(300):
            if len(fixture.peers.texts) > 2:
                break
            await asyncio.sleep(0.01)
        assert fixture.peers.texts[2:] == ["Le garage est ouvert de 8 h à 18 h."]
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
async def test_native_capture_is_synchronous_before_context_and_deferred_turn_stop():
    async with _native_sparra_opening() as fixture:
        await active(fixture)
        aggregator = next(p for p in fixture.session._active_runtime.pipeline.processors
                          if isinstance(p, LLMUserAggregator))
        observed = []

        def pushing(_aggregator, frame):
            if isinstance(frame, LLMContextFrame):
                scope = fixture.session._action_authority.scope_for(frame)
                observed.append(("context", scope.text, scope.origin is aggregator))
            elif isinstance(frame, UserStartedSpeakingFrame):
                observed.append(("start",))

        aggregator.add_event_handler("on_before_push_frame", pushing)
        aggregator.add_event_handler("on_user_turn_stopped",
                                     lambda *_args: observed.append(("deferred-stop",)))
        await caller(fixture, "Question synthétique")
        await _opening_event(fixture.peers.llm_requested, "native context did not infer")
        assert observed[:3] == [
            ("start",), ("start",), ("context", "Question synthétique", True),
        ]
        assert ("deferred-stop",) in observed


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "transférez-moi", "transférez moi", "transferez-moi", "TRANSFEREZ MOI !",
])
async def test_native_direct_transfer_requires_no_instruction_ack_and_no_new_inference(text):
    transferred = asyncio.Event()
    predicates = []

    async def available():
        return True

    async def request_human(*, action_is_current):
        predicates.append(action_is_current())
        transferred.set()
        return "ringing"

    registry = SimpleNamespace(human_tool_available=available, request_human=request_human,
                               transfer_to_cleanup=lambda _task: None)
    async with _native_sparra_opening(registry_terminalizer=registry) as fixture:
        await active(fixture)
        await caller(fixture, text)
        await _opening_event(transferred, "direct transfer did not reach existing consumer")
        assert predicates == [True]
        assert fixture.peers.conversations == []
        assert fixture.peers.texts[2:] == []


@pytest.mark.asyncio
async def test_native_transfer_command_can_interrupt_unfinished_owned_instruction():
    transferred = asyncio.Event()

    async def available():
        return True

    async def request_human(*, action_is_current):
        assert action_is_current()
        transferred.set()
        return "ringing"

    registry = SimpleNamespace(human_tool_available=available, request_human=request_human,
                               transfer_to_cleanup=lambda _task: None)
    async with _native_sparra_opening(registry_terminalizer=registry) as fixture:
        await active(fixture)
        fixture.peers.instruction_blocked = True
        fixture.peers.tool_calls = [tool("request_human")]
        await caller(fixture, "Je souhaite un interlocuteur")
        await _opening_event(fixture.peers.instruction_entered, "owned instruction missing")
        assert not transferred.is_set()
        assert fixture.peers.texts[-1] == "Pour demander le transfert, dites : transférez-moi."
        await caller(fixture, "transférez-moi")
        await _opening_event(transferred, "command waited for an instruction ACK")
        assert len(fixture.peers.conversations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("late_tool", [False, True])
async def test_native_response_a_is_fenced_at_actual_identical_caller_b_start(late_tool):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.delayed_response = True
        fixture.peers.response_text = "Réponse tardive A."
        fixture.peers.end_call = late_tool
        await caller(fixture, "Question identique")
        await _opening_event(fixture.peers.delayed_entered, "response A did not pause")
        aggregator = next(p for p in fixture.session._active_runtime.pipeline.processors
                          if isinstance(p, LLMUserAggregator))
        crossed = asyncio.Event()
        result = asyncio.Event()
        observe_result(fixture, result)
        response_finished = asyncio.Event()

        def finished(_llm, frame):
            if isinstance(frame, LLMFullResponseEndFrame):
                response_finished.set()

        fixture.llm.add_event_handler("on_after_push_frame", finished)

        async def release_a_after_native_b_start(_aggregator, frame):
            if not isinstance(frame, UserStartedSpeakingFrame) or crossed.is_set():
                return
            # Product's synchronous hook has already revoked A, but native
            # interruption has not yet been dispatched. Let the real SSE finish.
            fixture.peers.delayed_release.set()
            await _opening_event(fixture.peers.delayed_sent, "late SSE did not complete")
            await _opening_event(response_finished, "late response did not finish")
            if late_tool:
                await _opening_event(result, "stale native tool did not settle")
            assert fixture.peers.texts[2:] == []
            fixture.peers.delayed_response = False
            fixture.peers.response_text = "Réponse courante B."
            response_finished.clear()
            result.clear()
            crossed.set()

        aggregator.add_event_handler("on_before_push_frame", release_a_after_native_b_start)
        await caller(fixture, "Question identique")
        await _opening_event(crossed, "caller B did not invalidate A before interruption")
        for _ in range(400):
            if len(fixture.peers.conversations) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(fixture.peers.conversations) == 2
        await _opening_event(response_finished, "response B did not finish")
        if late_tool:
            await _opening_event(result, "reused ID in response B did not settle")
        assert "Réponse tardive A." not in fixture.peers.texts
        assert "Pour terminer l'appel, dites : au revoir." not in fixture.peers.texts
        assert not fixture.final_mark_sent.is_set()
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
async def test_native_known_a_delayed_before_processing_is_dropped_after_identical_b():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        runtime = fixture.session._active_runtime
        aggregator = next(
            p for p in runtime.pipeline.processors if isinstance(p, LLMUserAggregator)
        )
        contexts = []

        def capture(_aggregator, frame):
            if isinstance(frame, LLMContextFrame):
                contexts.append(frame)

        aggregator.add_event_handler("on_before_push_frame", capture)

        async def pause_after_native_interruption(_llm, frame):
            if isinstance(frame, InterruptionFrame):
                # Native interruption resets its data queue. Pause only after
                # that actual reset, before the completed caller context arrives.
                await fixture.llm.pause_processing_frames()

        fixture.llm.add_event_handler("on_after_process_frame", pause_after_native_interruption)
        await caller(fixture, "Question identique")
        old = contexts[0]
        old_scope = fixture.session._action_authority.scope_for(old)
        await caller(fixture, "Question identique")
        assert fixture.peers.conversations == []
        assert fixture.session._action_authority.scope_for(old) is old_scope
        assert old_scope.epoch < fixture.session._action_authority.epoch
        fixture.llm.remove_event_handler("on_after_process_frame", pause_after_native_interruption)
        await fixture.llm.resume_processing_frames()
        await _opening_event(fixture.peers.llm_requested, "current B did not infer after resume")
        for _ in range(300):
            if len(fixture.peers.texts) > 2:
                break
            await asyncio.sleep(0.01)
        assert len(fixture.peers.conversations) == 1
        processed = asyncio.Event()

        def after(_llm, frame):
            if frame is old:
                processed.set()

        fixture.llm.add_event_handler("on_after_process_frame", after)
        fixture.peers.response_text = "Stale pre-dispatch A."
        # Deliver the retained native frame late at the actual service boundary;
        # its genuine origin remains bound even though the context now contains B.
        await fixture.llm.queue_frame(old)
        await _opening_event(processed, "delayed old native context did not settle")
        assert len(fixture.peers.conversations) == 1
        assert "Stale pre-dispatch A." not in fixture.peers.texts
