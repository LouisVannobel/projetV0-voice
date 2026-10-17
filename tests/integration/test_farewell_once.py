"""Native streamed text must not duplicate the program-owned final goodbye."""
from __future__ import annotations

import asyncio
import json

import pytest
from pipecat.frames.frames import (
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

from tests.integration.test_call_session import (
    _native_sparra_opening,
    _opening_event,
    session_module,
)
from tests.integration.test_voice_action_authority import active, caller, tool


def spoken(fixture, expected):
    event = asyncio.Event()

    def observe(_tts, frame):
        if isinstance(frame, TTSTextFrame) and frame.text == expected:
            event.set()

    fixture.tts.add_event_handler("on_after_push_frame", observe)
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", [
    "Bonne journée, au revoir.",
    ["Bonne ", "journée, ", "au revoir."],
    list("Bonne journée, au revoir."),
    ["\n", "Bonne journée, au revoir.", " "],
    " Bonne journée, au revoir.\r\n",
])
async def test_exact_streamed_goodbye_then_valid_tool_has_one_owned_playback(parts):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.end_call = True
        fixture.peers.text_before_tool = parts
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(fixture.final_mark_sent, "native goodbye mark missing")
        assert fixture.peers.texts[2:] == ["Bonne journée, au revoir."]
        assert fixture.session._end_call_playback.pending
        assert not fixture.session._end_call_playback.acknowledged
        assert not fixture.running.done()
        await fixture.incoming.put({"type": "websocket.receive", "text": json.dumps({
            **fixture.sent[-1], "stream_id": "stream-1",
        })})
        await asyncio.wait_for(fixture.running, timeout=3)
        assert fixture.writer.commands[-1].payload.end_reason == "conversation_ended"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Bonne journée, au revoir.", "Bonne journée,"])
async def test_matching_response_without_end_tool_flushes_at_native_response_end(text):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.response_text = text
        heard = spoken(fixture, text)
        await caller(fixture, "Comment dit-on cela ?")
        await _opening_event(heard, "ordinary matching response was lost")
        assert fixture.peers.texts[2:] == [text]
        assert not fixture.final_mark_sent.is_set()
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate", [False, True])
async def test_rejected_end_tool_does_not_remove_matching_model_text(duplicate):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        if duplicate:
            fixture.peers.end_call = True
            fixture.peers.duplicate_end_call = True
        else:
            fixture.peers.tool_calls = [tool(arguments='{"unexpected":true}')]
        fixture.peers.text_before_tool = ["Bonne journée, ", "au revoir."]
        heard = spoken(fixture, "Bonne journée, au revoir.")
        await caller(fixture, "Quels sont vos horaires ?")
        await _opening_event(heard, "rejected tool hid ordinary streamed text")
        assert fixture.peers.texts[2:] == ["Bonne journée, au revoir."]
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
async def test_failed_stream_discards_matching_prefix_without_speaking_it():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.response_text = "Bonne journée, au revoir."
        fixture.peers.pause_after_llm_chunks = 1
        fixture.peers.llm_suffix_failure = True
        await caller(fixture, "Comment dit-on cela ?")
        await _opening_event(fixture.peers.llm_prefix_sent, "SSE did not pause")
        assert fixture.peers.texts[2:] == []
        fixture.peers.llm_suffix_release.set()
        with pytest.raises(session_module.CallSessionError):
            await asyncio.wait_for(fixture.running, timeout=3)
        assert fixture.peers.texts[2:] == []
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
async def test_unregistered_end_tool_keeps_matching_model_text():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.llm.register_function("end_call", fixture.session._end_call_tool)
        fixture.llm.unregister_function("end_call")
        fixture.peers.end_call = True
        fixture.peers.text_before_tool = "Bonne journée, au revoir."
        heard = spoken(fixture, "Bonne journée, au revoir.")

        def next_response(_llm, frame):
            if isinstance(frame, FunctionCallResultFrame):
                fixture.peers.end_call = False
                fixture.peers.text_before_tool = None
                fixture.peers.response_text = "Je peux prendre un message."

        fixture.llm.add_event_handler("on_before_push_frame", next_response)
        await caller(fixture, "Quels sont vos horaires ?")
        await _opening_event(heard, "unavailable tool hid ordinary streamed text")
        assert fixture.peers.texts.count("Bonne journée, au revoir.") == 1
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", [["Bon", " anniversaire."], [" " * 33, " "]])
async def test_mismatch_or_padding_bound_flushes_original_frames_before_stream_finishes(parts):
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.end_call = True
        fixture.peers.text_before_tool = parts
        fixture.peers.pause_after_llm_chunks = 2
        produced, forwarded = [], []
        native_push = fixture.llm.push_frame
        flushed = asyncio.Event()

        async def remember(frame, *args, **kwargs):
            if isinstance(frame, LLMTextFrame) and frame not in produced:
                frame.metadata["offline-original"] = object()
                produced.append(frame)
            await native_push(frame, *args, **kwargs)

        def observe(_llm, frame):
            if isinstance(frame, LLMTextFrame):
                forwarded.append(frame)
                if len(forwarded) == 2:
                    flushed.set()

        fixture.llm.push_frame = remember
        fixture.llm.add_event_handler("on_after_push_frame", observe)
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(fixture.peers.llm_prefix_sent, "SSE did not pause before tool")
        await _opening_event(flushed, "ordinary text waited for the remaining SSE stream")
        assert [frame.text for frame in forwarded] == parts
        assert len(produced) == 2
        assert all(actual is original for actual, original in zip(forwarded, produced, strict=True))
        assert all("offline-original" in frame.metadata for frame in forwarded)
        assert not fixture.final_mark_sent.is_set()
        fixture.peers.llm_suffix_release.set()
        await _opening_event(fixture.final_mark_sent, "tool did not retain native playback")


@pytest.mark.asyncio
async def test_epoch_revoked_during_prefix_flush_drops_remaining_original_frames():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.end_call = True
        fixture.peers.text_before_tool = ["Bon", "ne ", "fête."]
        fixture.peers.pause_after_llm_chunks = 3
        forwarded = []
        finished = asyncio.Event()

        def revoke_after_first(_llm, frame):
            if isinstance(frame, LLMTextFrame):
                forwarded.append(frame.text)
                fixture.session._action_authority.invalidate_response()
            elif isinstance(frame, LLMFullResponseEndFrame):
                finished.set()

        fixture.llm.add_event_handler("on_after_push_frame", revoke_after_first)
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(fixture.peers.llm_prefix_sent, "SSE did not reach mismatch")
        assert forwarded == ["Bon"]
        fixture.peers.llm_suffix_release.set()
        await _opening_event(finished, "revoked response did not finish")
        assert forwarded == ["Bon"]
        assert not fixture.session._end_call_playback.pending


@pytest.mark.asyncio
async def test_no_configured_goodbye_owner_keeps_matching_text_streaming():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.llm.bind_action_authority(fixture.session._action_authority)
        fixture.peers.response_text = "Bonne journée, au revoir."
        fixture.peers.pause_after_llm_chunks = 1
        forwarded = asyncio.Event()

        def observe(_llm, frame):
            if isinstance(frame, LLMTextFrame):
                forwarded.set()

        fixture.llm.add_event_handler("on_after_push_frame", observe)
        await caller(fixture, "Comment dit-on cela ?")
        await _opening_event(fixture.peers.llm_prefix_sent, "SSE did not pause")
        await _opening_event(forwarded, "unconfigured owner delayed ordinary text")
        fixture.peers.llm_suffix_release.set()


@pytest.mark.asyncio
async def test_interrupted_matching_prefix_cannot_leak_into_next_caller_response():
    async with _native_sparra_opening(ack_end_call=False) as fixture:
        await active(fixture)
        fixture.peers.end_call = True
        fixture.peers.text_before_tool = ["Bonne journée, "]
        fixture.peers.pause_after_llm_chunks = 1
        await caller(fixture, "J'ai dit au revoir.")
        await _opening_event(fixture.peers.llm_prefix_sent, "old prefix did not pause")
        aggregator = next(p for p in fixture.session._active_runtime.pipeline.processors
                          if isinstance(p, LLMUserAggregator))
        old_finished = asyncio.Event()
        new_started = asyncio.Event()

        def finished(_llm, frame):
            if isinstance(frame, LLMFullResponseEndFrame):
                old_finished.set()

        async def release_old(_aggregator, frame):
            if isinstance(frame, UserStartedSpeakingFrame) and not new_started.is_set():
                fixture.peers.llm_suffix_release.set()
                await _opening_event(old_finished, "old response did not finish")
                assert fixture.peers.texts[2:] == []
                fixture.peers.end_call = False
                fixture.peers.text_before_tool = None
                fixture.peers.pause_after_llm_chunks = None
                fixture.peers.response_text = "Bonne journée, au revoir."
                new_started.set()

        fixture.llm.add_event_handler("on_after_push_frame", finished)
        aggregator.add_event_handler("on_before_push_frame", release_old)
        heard = spoken(fixture, "Bonne journée, au revoir.")
        await caller(fixture, "Je vous pose une nouvelle question.")
        await _opening_event(new_started, "new caller did not revoke old response")
        await _opening_event(heard, "new ordinary response lost its independent prefix")
        assert fixture.peers.texts[2:] == ["Bonne journée, au revoir."]
        assert not fixture.session._end_call_playback.pending
