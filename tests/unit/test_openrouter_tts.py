from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from importlib import import_module
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from pipecat.frames.frames import (
    FatalErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.tts_service import TTSContext, TTSService
from pipecat.tests.utils import run_test
from pydantic import SecretStr

from projetv0_voice.qualified_profile import InferenceProfileV1

TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: Sequence[bytes] = (),
        *,
        failure: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._failure = failure
        self.iterations = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterations += 1
        for chunk in self._chunks:
            yield chunk
        if self._failure is not None:
            raise self._failure

    async def aclose(self) -> None:
        self.closed = True


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.entered.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    async def aclose(self) -> None:
        self.closed = True


def _profile_data(
    *, provider_options: dict[str, dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stt_model": "test/stt",
        "llm_model": "test/llm",
        "tts_model": "microsoft/mai-voice-2-flash",
        "tts_voice": "fr-FR-Soleil:MAI-Voice-2",
        "tts_pcm_sample_rate": 24000,
        "tts_pcm_channels": 1,
        "llm_provider_policy": {"sort": "latency", "allow_fallbacks": True},
        "tts_provider_options": provider_options
        if provider_options is not None
        else {"azure": {"style": "cheerful"}},
    }


def _adapter_class() -> type[TTSService]:
    module = import_module("projetv0_voice.inference.openrouter_tts")
    adapter = getattr(module, "OpenRouterTTSService", None)
    assert isinstance(adapter, type)
    assert issubclass(adapter, TTSService)
    return adapter


def _client_for(
    *,
    status_code: int = 200,
    headers: httpx.HeaderTypes | None = ((b"content-type", b"audio/pcm"),),
    stream: httpx.AsyncByteStream | None = None,
    captured: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(
            status_code,
            headers=headers,
            stream=stream or _ChunkStream(),
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _service(
    client: httpx.AsyncClient,
    *,
    provider_options: dict[str, dict[str, object]] | None = None,
) -> Any:
    profile = InferenceProfileV1.model_validate(
        _profile_data(provider_options=provider_options)
    )
    service = _adapter_class()(
        profile=profile,
        api_key=SecretStr("unit-secret"),
        http_client=client,
    )
    # `TTSService.start()` applies this same qualified `_init_sample_rate` in a worker.
    service._sample_rate = profile.tts_pcm_sample_rate
    return service


async def _collect(service: Any, *, context_id: str = "context-one") -> list[object]:
    return [frame async for frame in service.run_tts("Bonjour", context_id)]


@pytest.mark.asyncio
async def test_streams_arbitrary_voice_pcm_and_provider_options_with_odd_chunk_guard() -> None:
    captured: list[httpx.Request] = []
    stream = _ChunkStream([b"\x01", b"\x02\x03"])
    client = _client_for(
        headers=((b"content-type", b"AuDiO/PcM; rate=8000; channels=2"),),
        stream=stream,
        captured=captured,
    )
    service = _service(client)
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    try:
        frames = await _collect(service)
    finally:
        await client.aclose()

    audio = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
    assert len(audio) == 2
    assert b"".join(frame.audio for frame in audio) == b"\x01\x02\x03\x00"
    assert all(frame.sample_rate == 24000 for frame in audio)
    assert all(frame.num_channels == 1 for frame in audio)
    assert all(frame.context_id == "context-one" for frame in audio)
    assert len(captured) == 1
    assert str(captured[0].url) == TTS_ENDPOINT
    assert json.loads(captured[0].content) == {
        "model": "microsoft/mai-voice-2-flash",
        "input": "Bonjour",
        "voice": "fr-FR-Soleil:MAI-Voice-2",
        "response_format": "pcm",
        "provider": {"options": {"azure": {"style": "cheerful"}}},
    }
    assert stream.iterations == 1
    assert stream.closed
    assert service.can_generate_metrics() is True
    assert "unit-secret" not in repr(service)
    service.start_tts_usage_metrics.assert_awaited_once_with("Bonjour")
    service.stop_ttfb_metrics.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_provider_options_omit_entire_provider_request_field() -> None:
    captured: list[httpx.Request] = []
    client = _client_for(stream=_ChunkStream([b"\x01\x02"]), captured=captured)
    service = _service(client, provider_options={})
    try:
        frames = await _collect(service)
    finally:
        await client.aclose()

    assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)
    assert "provider" not in json.loads(captured[0].content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra", [{}, {"tts_speed": None}, {"tts_speed": 1.15}], ids=["absent", "none", "explicit"]
)
async def test_optional_tts_speed_is_sent_as_standard_top_level_http_parameter(
    extra: dict[str, object],
) -> None:
    captured: list[httpx.Request] = []
    stream = _ChunkStream([b"\x01\x02", b"\x03\x04"])
    client = _client_for(stream=stream, captured=captured)
    try:
        profile = InferenceProfileV1.model_validate({**_profile_data(), **extra})
        service = _adapter_class()(
            profile=profile, api_key=SecretStr("unit-secret"), http_client=client
        )
        service._sample_rate = 24000  # Same qualified value applied by native start.
        frames = await _collect(service, context_id="speed-context")
    finally:
        await client.aclose()

    assert len(captured) == 1
    assert captured[0].method == "POST"
    assert str(captured[0].url) == TTS_ENDPOINT
    expected = {
        "model": "microsoft/mai-voice-2-flash",
        "input": "Bonjour",
        "voice": "fr-FR-Soleil:MAI-Voice-2",
        "response_format": "pcm",
        "provider": {"options": {"azure": {"style": "cheerful"}}},
    }
    if extra.get("tts_speed") is not None:
        expected["speed"] = 1.15
    assert json.loads(captured[0].content) == expected
    audio = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
    assert b"".join(frame.audio for frame in audio) == b"\x01\x02\x03\x04"
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in audio)
    assert all(frame.context_id == "speed-context" for frame in audio)
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        (),
        ((b"content-type", b"audio/pcm"), (b"content-type", b"audio/pcm")),
        ((b"content-type", b"text/event-stream"),),
        ((b"content-type", b"audio/pcm, audio/pcm"),),
        ((b"content-type", b"audio"),),
        ((b"content-type", b""),),
    ],
)
async def test_rejects_invalid_content_type_before_reading_body(
    headers: tuple[tuple[bytes, bytes], ...],
) -> None:
    stream = _ChunkStream([b"provider-body-must-not-be-read"])
    client = _client_for(headers=headers, stream=stream)
    service = _service(client)
    service.stop_ttfb_metrics = AsyncMock()
    try:
        frames = await _collect(service)
    finally:
        await client.aclose()

    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == "openrouter_tts_content_type"
    assert frames[0].exception is None
    assert stream.iterations == 0
    assert stream.closed
    service.stop_ttfb_metrics.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_non_200_is_constant_safe_and_never_reads_response_body() -> None:
    secret_body = "provider-secret-body"
    stream = _ChunkStream([secret_body.encode()])
    client = _client_for(status_code=429, stream=stream)
    service = _service(client)
    service.stop_ttfb_metrics = AsyncMock()
    try:
        frames = await _collect(service)
    finally:
        await client.aclose()

    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == "openrouter_tts_http_status"
    assert secret_body not in repr(frames[0])
    assert frames[0].exception is None
    assert stream.iterations == 0
    assert stream.closed
    service.stop_ttfb_metrics.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_transport_stream_and_empty_audio_fail_with_stable_codes() -> None:
    adapter = _adapter_class()
    profile = InferenceProfileV1.model_validate(_profile_data())

    async def transport_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider-secret-transport", request=request)

    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(transport_failure))
    transport_service = adapter(
        profile=profile,
        api_key=SecretStr("unit-secret"),
        http_client=transport_client,
    )
    transport_service._sample_rate = 24000
    transport_service.stop_ttfb_metrics = AsyncMock()

    stream = _ChunkStream(failure=httpx.ReadError("provider-secret-stream"))
    stream_client = _client_for(stream=stream)
    stream_service = _service(stream_client)
    stream_service.start_tts_usage_metrics = AsyncMock()
    stream_service.stop_ttfb_metrics = AsyncMock()

    empty_stream = _ChunkStream()
    empty_client = _client_for(stream=empty_stream)
    empty_service = _service(empty_client)
    empty_service.start_tts_usage_metrics = AsyncMock()
    empty_service.stop_ttfb_metrics = AsyncMock()
    try:
        transport_frames = await _collect(transport_service, context_id="transport")
        stream_frames = await _collect(stream_service, context_id="stream")
        empty_frames = await _collect(empty_service, context_id="empty")
    finally:
        await transport_client.aclose()
        await stream_client.aclose()
        await empty_client.aclose()

    assert [frame.error for frame in transport_frames] == ["openrouter_tts_transport"]
    assert [frame.error for frame in stream_frames] == ["openrouter_tts_stream"]
    assert [frame.error for frame in empty_frames] == ["openrouter_tts_empty_audio"]
    assert all(isinstance(frame, FatalErrorFrame) for frame in transport_frames)
    assert all(isinstance(frame, FatalErrorFrame) for frame in stream_frames)
    assert all(isinstance(frame, FatalErrorFrame) for frame in empty_frames)
    assert all(frame.exception is None for frame in transport_frames + stream_frames + empty_frames)
    transport_service.stop_ttfb_metrics.assert_awaited_once_with()
    stream_service.stop_ttfb_metrics.assert_awaited_once_with()
    empty_service.stop_ttfb_metrics.assert_awaited_once_with()
    stream_service.start_tts_usage_metrics.assert_awaited_once_with("Bonjour")
    empty_service.start_tts_usage_metrics.assert_awaited_once_with("Bonjour")
    assert stream.closed
    assert empty_stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "transport_error", "expected_code"),
    [
        (429, False, "openrouter_tts_http_status"),
        (200, True, "openrouter_tts_transport"),
    ],
)
async def test_metrics_failure_cannot_suppress_first_stable_fatal(
    status_code: int,
    transport_error: bool,
    expected_code: str,
) -> None:
    sentinel = "sentinel-provider-secret"

    async def handler(request: httpx.Request) -> httpx.Response:
        if transport_error:
            raise httpx.ConnectError("provider-transport", request=request)
        return httpx.Response(status_code, stream=_ChunkStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = _service(client)
    service.stop_ttfb_metrics = AsyncMock(side_effect=RuntimeError(sentinel))
    try:
        frames = await _collect(service)
    finally:
        await client.aclose()

    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == expected_code
    assert frames[0].exception is None
    assert sentinel not in repr(frames[0])


@pytest.mark.asyncio
async def test_metrics_cancellation_still_propagates() -> None:
    client = _client_for(status_code=429, stream=_ChunkStream())
    service = _service(client)
    service.stop_ttfb_metrics = AsyncMock(side_effect=asyncio.CancelledError)
    try:
        with pytest.raises(asyncio.CancelledError):
            await _collect(service)
        assert service._failed_contexts == {"context-one": "openrouter_tts_http_status"}
        await service.cleanup()
        assert service._failed_contexts == {}
        assert not client.is_closed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_failure_after_audio_does_not_stop_ttfb_in_adapter() -> None:
    stream = _ChunkStream(
        [b"\x01\x02"],
        failure=httpx.ReadError("provider-secret-after-audio"),
    )
    client = _client_for(stream=stream)
    service = _service(client)
    service.stop_ttfb_metrics = AsyncMock()
    try:
        frames = await _collect(service)
    finally:
        await client.aclose()

    assert isinstance(frames[0], TTSAudioRawFrame)
    assert isinstance(frames[1], FatalErrorFrame)
    assert frames[1].error == "openrouter_tts_stream"
    service.stop_ttfb_metrics.assert_not_awaited()
    assert stream.closed


@pytest.mark.asyncio
async def test_first_failure_wins_and_later_same_context_makes_no_request_or_error() -> None:
    captured: list[httpx.Request] = []
    client = _client_for(status_code=500, stream=_ChunkStream(), captured=captured)
    service = _service(client)
    try:
        first = await _collect(service)
        second = await _collect(service)
    finally:
        await client.aclose()

    assert len(captured) == 1
    assert [frame.error for frame in first] == ["openrouter_tts_http_status"]
    assert second == []


@pytest.mark.asyncio
async def test_failed_context_neutralizes_late_tts_text_without_manual_removal() -> None:
    client = _client_for(status_code=500, stream=_ChunkStream())
    service = _service(client)
    service._tts_contexts["context-one"] = TTSContext(
        append_to_context=True,
        push_assistant_aggregation=True,
    )
    service._audio_contexts = {"context-one": asyncio.Queue()}
    try:
        frames = await _collect(service)
        text = TTSTextFrame("Bonjour", "sentence")
        text.append_to_context = True
        await service.append_to_audio_context("context-one", text)
        queued = service._audio_contexts["context-one"].get_nowait()
    finally:
        await client.aclose()

    assert len(frames) == 1
    assert service._tts_contexts["context-one"].append_to_context is False
    assert service._tts_contexts["context-one"].push_assistant_aggregation is False
    assert queued is text
    assert text.append_to_context is False
    await service.on_audio_context_interrupted("context-one")
    assert service._failed_contexts == {}


@pytest.mark.asyncio
async def test_cancellation_propagates_closes_response_and_keeps_injected_client() -> None:
    stream = _BlockingStream()
    client = _client_for(stream=stream)
    service = _service(client)
    task = asyncio.create_task(_collect(service, context_id="cancelled"))
    await asyncio.wait_for(stream.entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed
    assert not client.is_closed
    assert "cancelled" not in service._failed_contexts
    await service.cleanup()
    assert not client.is_closed
    assert service._failed_contexts == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_cleanup_is_idempotent_closes_only_owned_client_and_clears_failures() -> None:
    injected_client = _client_for(status_code=500, stream=_ChunkStream())
    injected = _service(injected_client)
    await _collect(injected, context_id="failed")
    await injected.cleanup()
    await injected.cleanup()
    assert not injected_client.is_closed
    assert injected._failed_contexts == {}
    await injected_client.aclose()

    profile = InferenceProfileV1.model_validate(_profile_data())
    owned = _adapter_class()(profile=profile, api_key=SecretStr("unit-secret"))
    owned_client = owned._client
    await owned.cleanup()
    await owned.cleanup()
    assert owned_client.is_closed


@pytest.mark.asyncio
async def test_full_pipecat_failure_does_not_commit_assistant_text_and_success_does() -> None:
    failed_stream = _ChunkStream()
    failed_client = _client_for(stream=failed_stream)
    failed_service = _service(failed_client)
    failed_context = LLMContext()
    failed_pair = LLMContextAggregatorPair(failed_context)

    failed_down, failed_up = await run_test(
        Pipeline([failed_service, failed_pair.assistant()]),
        frames_to_send=[TTSSpeakFrame("Ne doit pas être persisté")],
        pipeline_params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in failed_down)
    fatal = [frame for frame in failed_up if isinstance(frame, FatalErrorFrame)]
    assert [frame.error for frame in fatal] == ["openrouter_tts_empty_audio"]
    assert failed_context.get_messages() == []
    assert failed_service._failed_contexts == {}
    assert failed_stream.closed
    assert not failed_client.is_closed
    await failed_client.aclose()

    success_stream = _ChunkStream([b"\x01\x02"])
    success_client = _client_for(stream=success_stream)
    success_service = _service(success_client)
    success_stop_ttfb = success_service.stop_ttfb_metrics
    success_service.stop_ttfb_metrics = AsyncMock(wraps=success_stop_ttfb)
    success_context = LLMContext()
    success_pair = LLMContextAggregatorPair(success_context)

    success_down, success_up = await run_test(
        Pipeline([success_service, success_pair.assistant()]),
        frames_to_send=[TTSSpeakFrame("Bonjour")],
        pipeline_params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    assert any(isinstance(frame, TTSAudioRawFrame) for frame in success_down)
    assert not any(isinstance(frame, FatalErrorFrame) for frame in success_up)
    assert success_context.get_messages() == [{"role": "assistant", "content": "Bonjour"}]
    success_service.stop_ttfb_metrics.assert_awaited_once_with()
    assert success_stream.closed
    assert not success_client.is_closed
    await success_client.aclose()


@pytest.mark.asyncio
async def test_full_pipecat_shared_turn_stops_after_first_failed_sentence() -> None:
    captured: list[httpx.Request] = []
    stream = _ChunkStream()
    client = _client_for(stream=stream, captured=captured)
    service = _service(client)
    context = LLMContext()
    pair = LLMContextAggregatorPair(context)
    tts_requests: list[tuple[str, str]] = []

    @service.event_handler("on_tts_request")
    async def capture_tts_request(
        _service: TTSService,
        context_id: str,
        text: str,
    ) -> None:
        tts_requests.append((context_id, text))

    down, up = await run_test(
        Pipeline([service, pair.assistant()]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Première phrase. "),
            LLMTextFrame("Deuxième phrase."),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    assert len(captured) == 1
    assert [text for _, text in tts_requests] == ["Première phrase.", "Deuxième phrase."]
    assert len({context_id for context_id, _ in tts_requests}) == 1
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down)
    fatal = [frame for frame in up if isinstance(frame, FatalErrorFrame)]
    assert [frame.error for frame in fatal] == ["openrouter_tts_empty_audio"]
    assert context.get_messages() == []
    assert service._failed_contexts == {}
    assert stream.closed
    assert not client.is_closed
    await client.aclose()
