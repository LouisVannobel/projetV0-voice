"""Synthetic offline peer; actual pinned Pipecat -> OpenAI SDK -> HTTPX path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx
import pytest
from openai import DefaultAsyncHttpxClient
from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import ErrorFrame, FatalErrorFrame, Frame, TranscriptionFrame
from pydantic import SecretStr

from projetv0_voice.inference.completion_strategy import (
    STT_COMPLETED_SEGMENT_KEY,
    STT_DELIVERY_ALLOWANCE_SECONDS,
    STT_MAX_PENDING_SEGMENTS,
    STT_REQUEST_TIMEOUT_SECONDS,
    STT_USER_TURN_WATCHDOG_SECONDS,
)
from projetv0_voice.inference.services import build_stt
from tests.unit.test_inference_deadlines import _profile
from tests.unit.test_inference_deadlines import (
    test_stt_late_final_does_not_release_context_before_resumed_segment_transcript as _fifo_case,
)

_SYNTHETIC_TEXT = "synthetic offline deadline fixture"
_SYNTHETIC_WAV = pcm_to_wav(b"\x00\x00" * 800, 8000)


@dataclass
class _DelayedPeer:
    delay: float
    requests: list[httpx.Request] = field(default_factory=list)
    cancelled: bool = False
    responded: bool = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.responded = True
        return httpx.Response(200, json={"text": _SYNTHETIC_TEXT})


async def _native_request(peer: _DelayedPeer) -> tuple[list[Frame], list[Frame]]:
    # No service method, timeout, validator or SDK parser is patched.
    async with DefaultAsyncHttpxClient(
        transport=httpx.MockTransport(peer.handle), trust_env=False
    ) as client:
        profile = _profile().model_copy(update={"stt_model": "microsoft/mai-transcribe-2"})
        service = build_stt(
            profile, SecretStr("synthetic-offline-key"), language="fr-FR",
            http_client=client, sparra_audio_limits=True,
        )
        async with asyncio.timeout(25):
            frames = [frame async for frame in service.run_stt(_SYNTHETIC_WAV)]
            # A terminal failure must refuse later audio without another upload.
            after = []
            if any(isinstance(frame, FatalErrorFrame) for frame in frames):
                after = [frame async for frame in service.run_stt(_SYNTHETIC_WAV)]
    return frames, after


def _assert_native_multipart(peer: _DelayedPeer) -> None:
    assert len(peer.requests) == 1
    request = peer.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/audio/transcriptions"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="audio.wav"' in request.content
    assert b"Content-Type: audio/wav" in request.content
    assert _SYNTHETIC_WAV in request.content
    assert b'name="model"\r\n\r\nmicrosoft/mai-transcribe-2\r\n' in request.content
    assert b'name="language"\r\n\r\nfr\r\n' in request.content


@pytest.mark.asyncio
async def test_native_stt_response_after_old_eight_second_deadline_completes_once() -> None:
    peer = _DelayedPeer(delay=9.0)
    frames, after = await _native_request(peer)
    _assert_native_multipart(peer)
    transcripts = [frame for frame in frames if isinstance(frame, TranscriptionFrame)]
    errors = [frame for frame in frames if isinstance(frame, ErrorFrame)]
    assert len(transcripts) == 1, (
        "native 9s HTTP response was cut before transcription; "
        f"safe native error codes: {[frame.error for frame in errors]}"
    )
    assert errors == []
    assert transcripts[0].text == _SYNTHETIC_TEXT
    assert transcripts[0].metadata[STT_COMPLETED_SEGMENT_KEY] == 1
    assert peer.responded and not peer.cancelled
    assert after == []


@pytest.mark.asyncio
async def test_native_stt_real_twenty_second_deadline_cancels_and_fails_closed() -> None:
    peer = _DelayedPeer(delay=20.2)
    loop = asyncio.get_running_loop()
    started = loop.time()
    frames, after = await _native_request(peer)
    elapsed = loop.time() - started
    _assert_native_multipart(peer)
    assert peer.cancelled and not peer.responded
    assert 19.5 <= elapsed < 25, "must exercise the real 20s deadline, not a shortened clock"
    assert len(frames) == 1
    assert isinstance(frames[0], FatalErrorFrame)
    assert frames[0].error == "openrouter_stt_timeout"
    assert not any(isinstance(frame, TranscriptionFrame) for frame in frames)
    assert after == []


def test_stt_deadline_and_computed_watchdog_contract() -> None:
    assert STT_REQUEST_TIMEOUT_SECONDS == 20.0
    assert STT_MAX_PENDING_SEGMENTS == 4
    assert STT_DELIVERY_ALLOWANCE_SECONDS == 4.0
    assert STT_USER_TURN_WATCHDOG_SECONDS == (
        STT_MAX_PENDING_SEGMENTS * STT_REQUEST_TIMEOUT_SECONDS + STT_DELIVERY_ALLOWANCE_SECONDS
    ) == 84.0


@pytest.mark.asyncio
async def test_native_fifo_nine_second_response_preserves_resumed_speech_completion() -> None:
    # Reuse the native VAD/FIFO/aggregator harness and every completion assertion.
    loop = asyncio.get_running_loop()
    started = loop.time()
    async with asyncio.timeout(25):
        await _fifo_case(
            b_stopped_before_a=True,
            transcripts=("synthetic segment alpha", "synthetic segment bravo"),
            expected_text="synthetic segment alpha synthetic segment bravo",
            http_response_delay=9.0,
        )
    assert 9 <= loop.time() - started < 15
