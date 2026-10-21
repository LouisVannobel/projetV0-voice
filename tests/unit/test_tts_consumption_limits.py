from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from pipecat.frames.frames import FatalErrorFrame, Frame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test
from pydantic import SecretStr

from projetv0_voice.inference import openrouter_tts
from projetv0_voice.pipeline import END_CALL_GOODBYE, EndCallPlayback, FirstFailure
from projetv0_voice.qualified_profile import InferenceProfileV1

CHARACTER_LIMIT = 20_000
FAREWELL_RESERVE = len(END_CALL_GOODBYE)
CONVERSATION_LIMIT = CHARACTER_LIMIT - FAREWELL_RESERVE


class _HTTPPeer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.received = asyncio.Event()
        self.hold_first_before_pcm = False
        self.writers: set[asyncio.StreamWriter] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        try:
            first_line = await reader.readline()
            assert first_line.startswith(b"POST /audio/speech HTTP/1.1")
            headers: dict[bytes, bytes] = {}
            while (line := await reader.readline()) != b"\r\n":
                if not line:
                    return
                name, value = line.split(b":", 1)
                headers[name.lower()] = value.strip()
            length = int(headers[b"content-length"])
            assert 0 < length <= 100_000
            self.requests.append(json.loads(await reader.readexactly(length)))
            self.received.set()
            if self.hold_first_before_pcm and len(self.requests) == 1:
                await reader.read()
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                b"Content-Length: 2\r\nConnection: close\r\n\r\n\x01\x02"
            )
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.writers.discard(writer)


@pytest_asyncio.fixture
async def peer(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_HTTPPeer]:
    endpoint = _HTTPPeer()
    server = await asyncio.start_server(endpoint.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(openrouter_tts, "_TTS_ENDPOINT", f"http://127.0.0.1:{port}/audio/speech")
    try:
        yield endpoint
    finally:
        server.close()
        await server.wait_closed()
        for writer in tuple(endpoint.writers):
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()


def _service(client: httpx.AsyncClient, *, bounded: bool = True) -> Any:
    profile = InferenceProfileV1.model_validate(
        {
            "schema_version": 1,
            "stt_model": "test/stt",
            "llm_model": "test/llm",
            "tts_model": "microsoft/mai-voice-2-flash",
            "tts_voice": "fr-FR-Soleil:MAI-Voice-2",
            "tts_pcm_sample_rate": 24_000,
            "tts_pcm_channels": 1,
            "llm_provider_policy": {"sort": "latency", "allow_fallbacks": True},
            "tts_provider_options": {},
        }
    )
    options = {"enforce_consumption_limits": True} if bounded else {}
    service = openrouter_tts.OpenRouterTTSService(
        profile=profile, api_key=SecretStr("offline-unit-key"), http_client=client, **options
    )
    service._sample_rate = 24_000
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service


async def _collect(service: Any, text: str, context_id: str) -> list[Frame | None]:
    return [frame async for frame in service.run_tts(text, context_id)]


def _assert_limit(frames: list[Frame | None]) -> None:
    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == "openrouter_tts_character_limit"
    assert frames[0].exception is None


@pytest.mark.asyncio
async def test_unicode_final_body_is_built_once_and_reserved_before_post(peer: _HTTPPeer) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        service = _service(client)
        # The final POST input, rather than the caller's short text, sets the debit.
        final_input = "é😀" * (CONVERSATION_LIMIT // 2) + "界" * (CONVERSATION_LIMIT % 2)
        original_body = service._request_body
        calls = 0

        def build_body(text: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            body = original_body(text)
            body["input"] = final_input if calls == 1 else text
            return body

        service._request_body = build_body
        await _collect(service, "short caller text", "first")
        assert calls == 1
        assert peer.requests[0]["input"] == final_input
        assert len(final_input) == CONVERSATION_LIMIT
        assert len(final_input.encode()) > CONVERSATION_LIMIT
        _assert_limit(await _collect(service, "😀", "second"))
        assert len(peer.requests) == 1


@pytest.mark.asyncio
async def test_received_post_cancel_before_pcm_never_refunds_new_context(peer: _HTTPPeer) -> None:
    peer.hold_first_before_pcm = True
    async with httpx.AsyncClient(trust_env=False) as client:
        service = _service(client)
        pending = asyncio.create_task(_collect(service, "x" * CONVERSATION_LIMIT, "cancelled"))
        await asyncio.wait_for(peer.received.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await service.on_audio_context_interrupted("cancelled")
        await service.on_turn_context_created("after-cancel")
        _assert_limit(await _collect(service, "x", "after-cancel"))
        assert [len(row["input"]) for row in peer.requests] == [CONVERSATION_LIMIT]


@pytest.mark.asyncio
async def test_completion_and_cleanup_cannot_reset_budget_and_calls_are_isolated(
    peer: _HTTPPeer,
) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        first, second = _service(client), _service(client)
        await _collect(first, "x" * CONVERSATION_LIMIT, "first")
        await first.on_audio_context_completed("first")
        await first.on_turn_context_created("new-context")
        await first.cleanup()
        _assert_limit(await _collect(first, "x", "new-context"))
        assert any(
            isinstance(frame, TTSAudioRawFrame)
            for frame in await _collect(second, "x", "other-call")
        )
        assert [len(row["input"]) for row in peer.requests] == [CONVERSATION_LIMIT, 1]


def test_end_call_context_claim_requires_exact_owned_frame_not_identical_text() -> None:
    playback = EndCallPlayback(generation=UUID(int=1), first_failure=FirstFailure())
    attempt = playback.begin()
    assert attempt is not None
    fake = TTSSpeakFrame(END_CALL_GOODBYE)
    playback.before_tts_frame(fake)
    assert playback.bind_context("fake") is False
    assert attempt.context_id is None
    playback.before_tts_frame(attempt.speak_frame)
    assert playback.bind_context("owned") is True
    playback.after_tts_frame(attempt.speak_frame)
    assert playback.bind_context("late") is False
    assert attempt.context_id == "owned"


@pytest.mark.asyncio
async def test_only_native_owned_farewell_can_use_once_protected_reserve(peer: _HTTPPeer) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        service = _service(client)
        failure = FirstFailure()
        playback = EndCallPlayback(generation=UUID(int=1), first_failure=failure)
        attempt = playback.begin()
        assert attempt is not None
        service.bind_end_call_context(playback.bind_context)
        await _collect(service, "x" * CONVERSATION_LIMIT, "conversation")
        playback.before_tts_frame(TTSSpeakFrame(END_CALL_GOODBYE))
        await service.on_turn_context_created("fake-farewell")
        _assert_limit(await _collect(service, END_CALL_GOODBYE, "fake-farewell"))
        assert len(peer.requests) == 1
        assert not playback.acknowledged

        def before(_service: TTSService, frame: Frame) -> None:
            playback.before_tts_frame(frame)

        def after(_service: TTSService, frame: Frame) -> None:
            playback.after_tts_frame(frame)

        service.add_event_handler("on_before_process_frame", before)
        service.add_event_handler("on_after_process_frame", after)
        down, _up = await run_test(
            Pipeline([service]),
            frames_to_send=[attempt.speak_frame],
            pipeline_params=PipelineParams(audio_out_sample_rate=24_000),
        )
        assert peer.requests[-1]["input"] == END_CALL_GOODBYE
        assert sum(len(row["input"]) for row in peer.requests) == CHARACTER_LIMIT
        assert any(isinstance(frame, TTSAudioRawFrame) for frame in down)
        assert attempt.context_id is not None
        assert not playback.acknowledged, (
            "generated PCM cannot stand in for native wire send and ACK"
        )
        await service.on_turn_context_created("repeat-farewell")
        _assert_limit(await _collect(service, END_CALL_GOODBYE, "repeat-farewell"))
        assert len(peer.requests) == 2


@pytest.mark.asyncio
async def test_other_voice_consumers_keep_default_unbounded_behavior(peer: _HTTPPeer) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        service = _service(client, bounded=False)
        await _collect(service, "x" * (CHARACTER_LIMIT + 1), "other-project")
        assert len(peer.requests) == 1
        assert len(peer.requests[0]["input"]) == CHARACTER_LIMIT + 1


@pytest.mark.asyncio
async def test_calendar_context_observer_debits_ordinary_budget_and_preserves_farewell(peer):
    async with httpx.AsyncClient(trust_env=False) as client:
        service = _service(client)
        observed = []
        service.observe_turn_context(observed.append)
        await service.on_turn_context_created("calendar-recap")
        assert observed == ["calendar-recap"]
        assert service._farewell_context_id is None
        await _collect(service, "Recap", "calendar-recap")
        assert service._tts_characters_reserved == 5
        await _collect(service, "x" * (CONVERSATION_LIMIT - 5), "ordinary-remainder")
        _assert_limit(await _collect(service, "Recap", "another-recap"))
        assert len(peer.requests) == 2
        service.bind_end_call_context(lambda context: context == "goodbye")
        await service.on_turn_context_created("goodbye")
        await _collect(service, END_CALL_GOODBYE, "goodbye")
        assert service._farewell_request_reserved
        assert [len(request["input"]) for request in peer.requests] == [
            5, CONVERSATION_LIMIT - 5, FAREWELL_RESERVE,
        ]
