from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from importlib import import_module
from typing import Any

import httpx
import pytest
from openai import APIStatusError, DefaultAsyncHttpxClient
from pipecat.frames.frames import FatalErrorFrame, TranscriptionFrame, TTSAudioRawFrame
from pydantic import SecretStr

from projetv0_voice.qualified_profile import InferenceProfileV1


def _profile() -> InferenceProfileV1:
    return InferenceProfileV1.model_validate(
        {
            "schema_version": 1,
            "stt_model": "test/stt",
            "llm_model": "test/llm",
            "tts_model": "test/tts",
            "tts_voice": "fr-FR-Soleil:MAI-Voice-2",
            "tts_pcm_sample_rate": 24000,
            "tts_pcm_channels": 1,
            "llm_provider_policy": {"sort": "latency", "allow_fallbacks": True},
            "tts_provider_options": {},
        }
    )


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 80)
    return buffer.getvalue()


async def _collect_stt(service: Any, frames: list[object]) -> None:
    async for frame in service.run_stt(_wav()):
        frames.append(frame)


@pytest.mark.asyncio
async def test_stt_overall_deadline_cancels_native_http_without_late_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("projetv0_voice.inference.services")
    release = asyncio.Event()
    cancelled = asyncio.Event()
    requests: list[httpx.Request] = []
    blocking_request = False

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if not blocking_request:
            return httpx.Response(200, json={"text": ""})
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, json={"text": "réponse tardive"})

    frames: list[object] = []
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        # The SDK's first request performs platform discovery on a thread. Warm it
        # through the public native request before measuring cancellation of HTTP.
        await _collect_stt(service, [])
        blocking_request = True
        monkeypatch.setattr(module, "_STT_TIMEOUT_SECONDS", 0.02, raising=False)
        try:
            await asyncio.wait_for(_collect_stt(service, frames), timeout=0.25)
        except TimeoutError:
            pytest.fail("STT did not finish within its configured overall deadline")
        finally:
            release.set()
        await asyncio.sleep(0)

    assert cancelled.is_set()
    assert len(requests) == 2  # One normal warmup, one cancelled transcription.
    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == "openrouter_stt_timeout"
    assert frames[0].exception is None
    assert not any(isinstance(frame, TranscriptionFrame) for frame in frames)


@pytest.mark.asyncio
async def test_stt_external_cancellation_propagates_without_retry_or_transcription() -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, json={"text": "unreachable"})

    frames: list[object] = []
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        task = asyncio.create_task(_collect_stt(service, frames))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        await asyncio.sleep(0)

    assert cancelled.is_set()
    assert len(requests) == 1
    assert frames == []


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", ["  Bonjour  ", "", "  "])
async def test_stt_deadline_preserves_native_wav_language_and_empty_transcript_policy(
    transcript: str,
) -> None:
    module = import_module("projetv0_voice.inference.services")
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"text": transcript})

    frames: list[object] = []
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        await _collect_stt(service, frames)

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert b'name="language"\r\n\r\nfr\r\n' in request.content
    assert b'name="model"\r\n\r\ntest/stt\r\n' in request.content
    assert b'filename="audio.wav"\r\nContent-Type: audio/wav' in request.content
    assert _wav() in request.content
    if transcript.strip():
        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == "Bonjour"
    else:
        assert frames == []


@pytest.mark.asyncio
async def test_stt_native_http_error_becomes_constant_safe_fatal() -> None:
    module = import_module("projetv0_voice.inference.services")
    sentinel = "provider-secret-error"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": sentinel, "type": "invalid_request_error"}},
        )

    frames: list[object] = []
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        await _collect_stt(service, frames)

    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == "openrouter_stt_transport"
    assert frames[0].exception is None
    assert sentinel not in repr(frames[0])


@pytest.mark.asyncio
async def test_llm_native_sdk_does_not_retry_and_sends_finite_http_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("projetv0_voice.inference.services")
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            503,
            json={"error": {"message": "unit-unavailable", "type": "server_error"}},
        )

    def client_factory(**kwargs: Any) -> DefaultAsyncHttpxClient:
        return DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(module, "DefaultAsyncHttpxClient", client_factory)
    service = module.build_llm(_profile(), SecretStr("unit-secret"))
    try:
        with pytest.raises(APIStatusError):
            await service._client.chat.completions.create(
                model="test/llm", messages=[{"role": "user", "content": "ping"}]
            )
    finally:
        await service._client.close()
        await service.cleanup()

    assert len(requests) == 1
    assert requests[0].extensions["timeout"] == {
        "connect": 2.0,
        "read": 8.0,
        "write": 8.0,
        "pool": 8.0,
    }


class _DeadlinePCMStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_headers", [True, False])
async def test_tts_overall_deadline_cancels_http_and_prevents_late_audio(
    delayed_headers: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("projetv0_voice.inference.openrouter_tts")
    monkeypatch.setattr(module, "_TTS_TIMEOUT_SECONDS", 0.02, raising=False)
    stream = _DeadlinePCMStream()
    header_cancelled = asyncio.Event()
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if delayed_headers:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                header_cancelled.set()
                raise
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = module.OpenRouterTTSService(
            profile=_profile(), api_key=SecretStr("unit-secret"), http_client=client
        )
        service._sample_rate = 24000
        frames: list[object] = []

        async def collect() -> None:
            async for frame in service.run_tts("Bonjour", "deadline-context"):
                frames.append(frame)

        try:
            await asyncio.wait_for(collect(), timeout=0.25)
        except TimeoutError:
            pytest.fail("TTS did not finish within its configured overall deadline")
        await asyncio.sleep(0)
        audio = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
        fatal = [frame for frame in frames if isinstance(frame, FatalErrorFrame)]
        assert len(fatal) == 1
        assert fatal[0].error == "openrouter_tts_timeout"
        assert fatal[0].exception is None
        if delayed_headers:
            assert header_cancelled.is_set()
            assert audio == []
        else:
            assert stream.cancelled.is_set()
            assert stream.closed
            assert len(audio) == 1
            assert audio[0].audio == b"\x01\x02"
        assert [frame async for frame in service.run_tts("suite", "deadline-context")] == []
        assert len(requests) == 1
        await service.cleanup()
