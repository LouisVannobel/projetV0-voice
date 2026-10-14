from __future__ import annotations

import asyncio
import inspect
import io
import json
import struct
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import pytest
from openai import DefaultAsyncHttpxClient
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    FatalErrorFrame,
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import QueuedFrameProcessor
from pydantic import SecretStr

from projetv0_voice.inference import services
from tests.unit.test_inference_deadlines import _profile, _start_native_stt_worker


def _wav(frames: int, rate: int = 8000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames)
    return output.getvalue()


@dataclass
class _Peer:
    base_url: str = ""
    requests: list[bytes] = field(default_factory=list)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


@asynccontextmanager
async def _peer(*, status: int = 200, blocked: bool = False) -> AsyncIterator[_Peer]:
    """The actual SDK writes HTTP to a loopback socket; no provider is contacted."""
    peer = _Peer()
    connections: set[asyncio.Task[Any]] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            assert headers.startswith(b"POST /v1/audio/transcriptions HTTP/1.1\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            peer.requests.append(await reader.readexactly(length))
            peer.entered.set()
            if blocked:
                await peer.release.wait()
            body = json.dumps(
                {"text": ""} if status == 200 else {"error": {"message": "offline-unavailable"}}
            ).encode()
            writer.write(
                f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
            )
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            connections.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    assert server.sockets is not None
    peer.base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
    try:
        yield peer
    finally:
        peer.release.set()
        server.close()
        await server.wait_closed()
        if connections:
            await asyncio.wait_for(asyncio.gather(*connections), timeout=2)


def _service(client: DefaultAsyncHttpxClient, *, sparra: bool = True) -> Any:
    kwargs: dict[str, Any] = {"language": "fr-FR", "http_client": client}
    # Before the opt-in API exists, run the real old service so RED is a
    # behavioral assertion (extra POST / missing refusal), not a TypeError.
    if "sparra_audio_limits" in inspect.signature(services.build_stt).parameters:
        kwargs["sparra_audio_limits"] = sparra
    return services.build_stt(_profile(), SecretStr("offline-unit-key"), **kwargs)


async def _collect(service: Any, audio: bytes) -> list[Frame]:
    return [frame async for frame in service.run_stt(audio)]


def _assert_refused(frames: list[Frame], code: str) -> None:
    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == code
    assert frames[0].exception is None


@pytest.mark.asyncio
async def test_sparra_stt_native_retryable_503_sends_exactly_one_post(monkeypatch) -> None:
    async with _peer(status=503) as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            try:
                frames = await asyncio.wait_for(_collect(service, _wav(8000)), timeout=4)
                _assert_refused(frames, "openrouter_stt_transport")
                assert len(peer.requests) == 1
                assert service._submitted_audio_seconds == Fraction(1)
            finally:
                await service.cleanup()


@pytest.mark.asyncio
async def test_default_600_seconds_accepts_exact_limit_then_refuses_one_sample(monkeypatch) -> None:
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            try:
                exact = _wav(600 * 8000)
                frames = await _collect(service, exact)
                assert not any(isinstance(f, FatalErrorFrame) for f in frames)
                _assert_refused(await _collect(service, _wav(1)), "openrouter_stt_audio_limit")
                assert len(peer.requests) == 1
                assert exact in peer.requests[0]
                assert service._submitted_audio_seconds == Fraction(600)
            finally:
                await service.cleanup()


@pytest.mark.asyncio
async def test_fractional_pcm_duration_sums_across_sample_rates_without_rounding(
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "_SPARRA_STT_MAX_AUDIO_SECONDS", 1, raising=False)
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            try:
                await _collect(service, _wav(1, rate=3))
                await _collect(service, _wav(2, rate=3))
                _assert_refused(await _collect(service, _wav(1)), "openrouter_stt_audio_limit")
                assert len(peer.requests) == 2
                assert service._submitted_audio_seconds == Fraction(1)
            finally:
                await service.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    b"not-a-wav",
    _wav(80)[:-2],
    _wav(80)[:24] + b"\0\0\0\0" + _wav(80)[28:],
])
async def test_invalid_or_truncated_wav_is_refused_before_any_post(
    invalid, monkeypatch,
) -> None:
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            try:
                _assert_refused(await _collect(service, invalid), "openrouter_stt_audio_invalid")
                assert peer.requests == []
                assert service._submitted_audio_seconds == Fraction(0)
            finally:
                await service.cleanup()


def _additional_audio(*, second_data_chunk: bool) -> bytes:
    audio = _wav(8000)
    extra_pcm = b"\0\0" * 8000
    if not second_data_chunk:
        return audio + extra_pcm  # Bytes outside the declared RIFF boundary.
    extra_chunk = b"data" + struct.pack("<I", len(extra_pcm)) + extra_pcm
    riff_size = struct.pack("<I", len(audio) + len(extra_chunk) - 8)
    return audio[:4] + riff_size + audio[8:] + extra_chunk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_data_chunk", [True, False], ids=["second-data", "trailing-payload"],
)
async def test_noncanonical_wav_cannot_upload_pcm_omitted_from_duration(
    second_data_chunk: bool, monkeypatch,
) -> None:
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            try:
                audio = _additional_audio(second_data_chunk=second_data_chunk)
                frames = await _collect(service, audio)
                _assert_refused(frames, "openrouter_stt_audio_invalid")
                assert peer.requests == []
                assert service._submitted_audio_seconds == Fraction(0)
            finally:
                await service.cleanup()


@pytest.mark.asyncio
async def test_cancellation_after_received_post_never_refunds_reserved_audio(monkeypatch) -> None:
    monkeypatch.setattr(services, "_SPARRA_STT_MAX_AUDIO_SECONDS", 1, raising=False)
    async with _peer(blocked=True) as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            task = asyncio.create_task(_collect(service, _wav(8000)))
            try:
                await asyncio.wait_for(peer.entered.wait(), timeout=2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                monkeypatch.setattr(services, "_STT_TIMEOUT_SECONDS", 0.1)
                _assert_refused(await _collect(service, _wav(1)), "openrouter_stt_audio_limit")
                assert service._submitted_audio_seconds == Fraction(1)
                await service.cancel(CancelFrame())
                assert service._submitted_audio_seconds == Fraction(1)
                assert await _collect(service, _wav(1)) == []
                assert len(peer.requests) == 1
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                peer.release.set()
                await service.cleanup()


@pytest.mark.asyncio
async def test_start_and_end_frame_cannot_reset_warmup_or_refund_padded_native_segment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "_SPARRA_STT_MAX_AUDIO_SECONDS", 1, raising=False)
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client)
            await _collect(service, _wav(4000))  # 0.5 s before the actual native StartFrame.
            observed: asyncio.Queue[Frame] = asyncio.Queue()
            collector = QueuedFrameProcessor(
                queue=observed, queue_direction=FrameDirection.UPSTREAM,
            )
            sink = QueuedFrameProcessor(queue=observed, queue_direction=FrameDirection.DOWNSTREAM)
            worker, task = await _start_native_stt_worker(collector, service, sink)
            try:
                await worker.queue_frames([
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(b"\0\0" * 800, 8000, 1),
                    VADUserStoppedSpeakingFrame(),
                    EndFrame(),
                ])
                await asyncio.wait_for(task, timeout=3)
                errors = [observed.get_nowait() for _ in range(observed.qsize())]
                assert any(
                    isinstance(f, FatalErrorFrame) and f.error == "openrouter_stt_audio_limit"
                    for f in errors
                )
                # Native adds 0.5 s silence to 0.1 s PCM. It exceeds the 0.5 s
                # remainder during EndFrame draining and must never reach the peer.
                assert len(peer.requests) == 1
                assert service._submitted_audio_seconds == Fraction(1, 2)
            finally:
                if not task.done():
                    await worker.queue_frame(CancelFrame())
                await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_two_call_instances_have_independent_limits_and_owner_closes_shared_http_client(
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "_SPARRA_STT_MAX_AUDIO_SECONDS", 1, raising=False)
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        client = DefaultAsyncHttpxClient(trust_env=False)
        first, second = _service(client), _service(client)
        try:
            await _collect(first, _wav(8000))
            _assert_refused(await _collect(first, _wav(1)), "openrouter_stt_audio_limit")
            await _collect(second, _wav(8000))
            assert len(peer.requests) == 2
            assert first._submitted_audio_seconds == second._submitted_audio_seconds == Fraction(1)
            await first.cleanup()
            assert not client.is_closed
        finally:
            await second.cleanup()
            await client.aclose()
        assert client.is_closed


@pytest.mark.asyncio
async def test_non_sparra_consumer_keeps_unlimited_legacy_audio_and_native_retries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "_SPARRA_STT_MAX_AUDIO_SECONDS", 0, raising=False)
    async with _peer(status=503) as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        async with DefaultAsyncHttpxClient(trust_env=False) as client:
            service = _service(client, sparra=False)
            try:
                _assert_refused(await _collect(service, _wav(8000)), "openrouter_stt_transport")
                assert len(peer.requests) == 3
            finally:
                await service.cleanup()
