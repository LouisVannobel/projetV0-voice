from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIStatusError, DefaultAsyncHttpxClient
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    FatalErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner
from pydantic import SecretStr

from projetv0_voice.inference.completion_strategy import (
    STT_COMPLETED_SEGMENT_KEY,
    STT_TERMINAL_PARTIAL_KEY,
    STT_USER_TURN_WATCHDOG_SECONDS,
    CompletionAwareTurnStopStrategy,
)
from projetv0_voice.pipeline import FirstFailure
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


async def _start_native_stt_worker(
    *processors: FrameProcessor,
) -> tuple[PipelineWorker, asyncio.Task[None]]:
    started = asyncio.Event()
    worker = PipelineWorker(
        Pipeline(list(processors)),
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
    )

    async def note_started(_worker: PipelineWorker, _frame: StartFrame) -> None:
        started.set()

    worker.add_event_handler("on_pipeline_started", note_started)
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    run_task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
    except BaseException:
        await worker.queue_frame(CancelFrame())
        await asyncio.wait_for(run_task, timeout=2)
        raise
    return worker, run_task


class _CompletePauseAnalyzer(BaseSmartTurn):
    """Keep native buffering/analysis, controlling only the ML verdict boundary."""

    def _predict_endpoint(self, audio_array: Any) -> dict[str, Any]:
        assert len(audio_array) > 0
        return {"prediction": 1, "probability": 1.0}


class _IncompleteThenCompleteAnalyzer(BaseSmartTurn):
    """Control only the endpoint verdict: A incomplete, resumed B complete."""

    def __init__(self) -> None:
        super().__init__()
        self._prediction_count = 0

    def _predict_endpoint(self, audio_array: Any) -> dict[str, Any]:
        assert len(audio_array) > 0
        prediction = int(self._prediction_count > 0)
        self._prediction_count += 1
        return {"prediction": prediction, "probability": float(prediction)}


class _CompletionAttemptProbe(CompletionAwareTurnStopStrategy):
    def __init__(self, attempt: asyncio.Event, first_failure: FirstFailure) -> None:
        super().__init__(
            turn_analyzer=_IncompleteThenCompleteAnalyzer(),
            on_incomplete_turn_stop=lambda: first_failure.signal("stt_failed"),
        )
        self._attempt = attempt

    async def trigger_user_turn_stopped(
        self, *, enable_user_speaking_frames: bool | None = None
    ) -> None:
        self._attempt.set()
        await super().trigger_user_turn_stopped(
            enable_user_speaking_frames=enable_user_speaking_frames
        )


class _ConfidenceBoundaryVAD(VADAnalyzer):
    """Native VAD timing/volume path with only classifier confidence controlled."""

    def __init__(self) -> None:
        super().__init__(
            sample_rate=8000,
            params=VADParams(start_secs=0.02, stop_secs=0.02, confidence=0.5, min_volume=0),
        )

    def num_frames_required(self) -> int:
        return 80

    def voice_confidence(self, buffer: bytes) -> float:
        return float(any(buffer))


class _CoverageDeliveryBridge(FrameProcessor):
    """Hold actual native notifications/receipts at their delivery boundary."""

    def __init__(self, mode: str) -> None:
        super().__init__(enable_direct_mode=True)
        self.mode = mode
        self.held = asyncio.Event()
        self._starts = 0
        self._holding_upstream = False
        self._held_frames: list[tuple[Frame, FrameDirection]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._starts += 1
                self._holding_upstream = self.mode == "upstream_vad" and self._starts == 2
            if self._holding_upstream and isinstance(
                frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
            ):
                self._held_frames.append((frame, direction))
                if isinstance(frame, VADUserStoppedSpeakingFrame):
                    self.held.set()
                return
        elif (
            self.mode == "completed_not_delivered"
            and isinstance(frame, TranscriptionFrame)
            and frame.metadata.get(STT_COMPLETED_SEGMENT_KEY) == 2
        ):
            self._held_frames.append((frame, direction))
            self.held.set()
            return
        await self.push_frame(frame, direction)

    async def deliver_held(self) -> None:
        frames, self._held_frames = self._held_frames, []
        for frame, direction in frames:
            await self.push_frame(frame, direction)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["upstream_vad", "completed_not_delivered"])
async def test_stt_coverage_uses_local_vad_and_consumed_receipt(mode: str) -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    a_delivered = asyncio.Event()
    b_local_stop = asyncio.Event()
    completion_attempt = asyncio.Event()
    context_emitted = asyncio.Event()
    context_snapshots: list[list[str]] = []
    requests: list[httpx.Request] = []
    stops_seen = 0
    first_failure = FirstFailure()
    bridge = _CoverageDeliveryBridge(mode)
    user = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=_ConfidenceBoundaryVAD(),
            user_turn_strategies=UserTurnStrategies(
                stop=[_CompletionAttemptProbe(completion_attempt, first_failure)]
            ),
            user_turn_stop_timeout=STT_USER_TURN_WATCHDOG_SECONDS,
            empty_user_turn=None,
        ),
        realtime_service_mode=False,
    ).user()

    async def handler(request: httpx.Request) -> httpx.Response:
        index = len(requests)
        requests.append(request)
        entered[index].set()
        await release[index].wait()
        return httpx.Response(200, json={"text": ["segment alpha", "segment bravo"][index]})

    def note_processed(_processor: FrameProcessor, frame: Frame) -> None:
        nonlocal stops_seen
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            stops_seen += 1
            if stops_seen == 2:
                b_local_stop.set()
        elif isinstance(frame, TranscriptionFrame):
            if frame.metadata.get(STT_COMPLETED_SEGMENT_KEY) == 1:
                a_delivered.set()

    def note_context(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, LLMContextFrame):
            context_snapshots.append(
                [str(message["content"]) for message in frame.context.get_messages()]
            )
            context_emitted.set()

    user.add_event_handler("on_after_process_frame", note_processed)
    user.add_event_handler("on_before_push_frame", note_context)
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        worker, run_task = await _start_native_stt_worker(service, bridge, user)
        try:
            await worker.queue_frame(
                STTMetadataFrame(service_name=service.name, ttfs_p99_latency=0.25)
            )
            await worker.queue_frames(
                [
                    InputAudioRawFrame(b"\x30\x75" * 800, 8000, 1),
                    InputAudioRawFrame(b"\x00\x00" * 800, 8000, 1),
                ]
            )
            await asyncio.wait_for(entered[0].wait(), timeout=5)
            if mode == "completed_not_delivered":
                release[0].set()
                await asyncio.wait_for(a_delivered.wait(), timeout=2)
            await worker.queue_frames(
                [
                    InputAudioRawFrame(b"\x30\x75" * 800, 8000, 1),
                    InputAudioRawFrame(b"\x00\x00" * 800, 8000, 1),
                ]
            )
            await asyncio.wait_for(b_local_stop.wait(), timeout=2)
            if mode == "upstream_vad":
                await asyncio.wait_for(bridge.held.wait(), timeout=2)
                assert len(requests) == 1  # B's upstream enrollment has not reached STT.
                release[0].set()
                await asyncio.wait_for(a_delivered.wait(), timeout=2)
            else:
                await asyncio.wait_for(entered[1].wait(), timeout=2)
                release[1].set()
                # This receipt exists only after B's real native generator exhausted.
                await asyncio.wait_for(bridge.held.wait(), timeout=2)
            await asyncio.wait_for(completion_attempt.wait(), timeout=2)
            assert context_snapshots == []
            assert not run_task.done()
            await bridge.deliver_held()
            if mode == "upstream_vad":
                await asyncio.wait_for(entered[1].wait(), timeout=2)
                release[1].set()
            await asyncio.wait_for(context_emitted.wait(), timeout=2)
            assert context_snapshots == [["segment alpha segment bravo"]]
            assert not run_task.done()
        finally:
            for event in release:
                event.set()
            await bridge.deliver_held()
            if not run_task.done():
                await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run_task, timeout=2)
    assert first_failure.code is None
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("timer", ["stt_safety_net", "controller_watchdog"])
@pytest.mark.parametrize("b_transcript", ["segment bravo", ""])
async def test_stt_prior_delivered_text_does_not_escape_pending_segment_via_timer(
    timer: str,
    b_transcript: str,
) -> None:
    """Witness the actual native timer while B HTTP is held, before terminal flush."""
    module = import_module("projetv0_voice.inference.services")
    native_completion_attempt = asyncio.Event()
    watchdog_fired = asyncio.Event()
    a_stop_processed = asyncio.Event()
    a_delivered = asyncio.Event()
    b_stop_processed = asyncio.Event()
    context_emitted = asyncio.Event()
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    context_snapshots: list[list[str]] = []
    provider_input: asyncio.Queue[Frame] = asyncio.Queue()
    first_failure = FirstFailure()
    a_stop = VADUserStoppedSpeakingFrame(stop_secs=0.2)
    b_stop = VADUserStoppedSpeakingFrame(stop_secs=0.2)
    strategy = _CompletionAttemptProbe(native_completion_attempt, first_failure)
    context = LLMContext()
    user = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(stop=[strategy]),
            user_turn_stop_timeout=2.0 if timer == "stt_safety_net" else 5.0,
            empty_user_turn=None,
        ),
        realtime_service_mode=False,
    ).user()
    requests: list[httpx.Request] = []
    warming = True

    async def handler(request: httpx.Request) -> httpx.Response:
        if warming:
            return httpx.Response(200, json={"text": ""})
        index = len(requests)
        requests.append(request)
        entered[index].set()
        await release[index].wait()
        return httpx.Response(200, json={"text": ["segment alpha", b_transcript][index]})

    def note_processed(_processor: FrameProcessor, frame: Frame) -> None:
        if frame is a_stop:
            a_stop_processed.set()
        elif frame is b_stop:
            b_stop_processed.set()
        elif isinstance(frame, TranscriptionFrame) and frame.text == "segment alpha":
            a_delivered.set()

    def note_context(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, LLMContextFrame):
            context_snapshots.append(
                [str(message["content"]) for message in frame.context.get_messages()]
            )
            context_emitted.set()

    def note_watchdog(_processor: FrameProcessor) -> None:
        first_failure.signal("stt_failed")
        watchdog_fired.set()

    async def allow_inference(frame: Frame) -> bool:
        return not isinstance(frame, LLMContextFrame) or first_failure.code is None

    user.add_event_handler("on_after_process_frame", note_processed)
    user.add_event_handler("on_before_push_frame", note_context)
    user.add_event_handler("on_user_turn_stop_timeout", note_watchdog)
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        # Warm SDK discovery before the native controller's real five-second clock.
        await _collect_stt(service, [])
        warming = False
        inference_gate = FunctionFilter(allow_inference)
        provider_probe = QueuedFrameProcessor(
            queue=provider_input, queue_direction=FrameDirection.DOWNSTREAM
        )
        worker, run_task = await _start_native_stt_worker(
            service, user, inference_gate, provider_probe
        )
        try:
            await worker.queue_frame(
                STTMetadataFrame(
                    service_name=service.name,
                    ttfs_p99_latency=0.25 if timer == "stt_safety_net" else 7.0,
                )
            )
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1),
                    a_stop,
                ]
            )
            await asyncio.wait_for(entered[0].wait(), timeout=2)
            await asyncio.wait_for(a_stop_processed.wait(), timeout=2)
            release[0].set()
            await asyncio.wait_for(a_delivered.wait(), timeout=2)
            assert context_snapshots == []  # A is delivered, but the endpoint is INCOMPLETE.
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(b"\x02\x00" * 800, 8000, 1),
                    b_stop,
                ]
            )
            await asyncio.wait_for(entered[1].wait(), timeout=2)
            await asyncio.wait_for(b_stop_processed.wait(), timeout=2)
            witness = native_completion_attempt if timer == "stt_safety_net" else watchdog_fired
            await asyncio.wait_for(witness.wait(), timeout=1 if timer == "stt_safety_net" else 5.5)
            with suppress(TimeoutError):
                await asyncio.wait_for(context_emitted.wait(), timeout=0.1)
            assert not run_task.done()
            if timer == "stt_safety_net":
                assert context_snapshots == [], (
                    f"{timer}_released_prior_A_while_B_pending: {context_snapshots!r}"
                )
                release[1].set()
                await asyncio.wait_for(context_emitted.wait(), timeout=2)
                assert context_snapshots == [
                    ["segment alpha segment bravo" if b_transcript else "segment alpha"]
                ]
                assert not run_task.done()  # Normal completion is proven before End.
            else:
                # Exercise actual native five-second expiry; production coordinates
                # it at 36 seconds. Even forced expiry must close inference first.
                assert first_failure.code is not None
                assert context_snapshots == [["segment alpha"]]
        finally:
            for event in release:
                event.set()
            if not run_task.done():
                await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run_task, timeout=2)

    provider_frames = [provider_input.get_nowait() for _ in range(provider_input.qsize())]
    if timer == "controller_watchdog":
        assert not any(isinstance(frame, LLMContextFrame) for frame in provider_frames)
    else:
        assert len([frame for frame in provider_frames if isinstance(frame, LLMContextFrame)]) == 1


@pytest.mark.asyncio
async def test_stt_fatal_with_queued_successor_cannot_restart_llm_during_native_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual inference gate must close before Cancel flushes delivered A."""
    services_module = import_module("projetv0_voice.inference.services")
    pipeline_module = import_module("projetv0_voice.pipeline")
    monkeypatch.setattr(
        pipeline_module, "LocalSmartTurnAnalyzerV3", _IncompleteThenCompleteAnalyzer
    )
    monkeypatch.setattr(
        pipeline_module, "SileroVADAnalyzer", lambda **_kwargs: _ConfidenceBoundaryVAD()
    )
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    a_delivered = asyncio.Event()
    c_enrolled = asyncio.Event()
    requests: list[httpx.Request] = []
    llm_input: asyncio.Queue[Frame] = asyncio.Queue()
    initial_llm_frames: list[Frame] = []
    transport_frames: asyncio.Queue[Frame] = asyncio.Queue()
    first_failure = FirstFailure()
    stops_enrolled = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        index = len(requests)
        requests.append(request)
        if index >= 2:
            return httpx.Response(200, json={"text": "segment charlie"})
        entered[index].set()
        await release[index].wait()
        if index == 0:
            return httpx.Response(200, json={"text": "segment alpha"})
        return httpx.Response(400, json={"error": {"message": "unit-B-failure"}})

    async def noop(*_args: object) -> None:
        return None

    async def allow_mark(*_args: object) -> bool:
        return True

    # Keep admission active throughout: the CallSession owner may close it only
    # after native fatal cancellation has already begun. No owner task is faked.
    controller = SimpleNamespace(
        is_active=lambda: True,
        mark_name="unit-disclosure-mark",
        accept_mark=allow_mark,
        abort=noop,
        note_disclosure_audio=noop,
        arm_expected_mark=allow_mark,
        mark_forwarded=noop,
    )
    input_processor = QueuedFrameProcessor(
        queue=transport_frames, queue_direction=FrameDirection.UPSTREAM
    )
    output_processor = QueuedFrameProcessor(
        queue=transport_frames, queue_direction=FrameDirection.DOWNSTREAM
    )
    llm = QueuedFrameProcessor(queue=llm_input, queue_direction=FrameDirection.DOWNSTREAM)
    tts = QueuedFrameProcessor(queue=transport_frames, queue_direction=FrameDirection.DOWNSTREAM)
    transport = SimpleNamespace(input=lambda: input_processor, output=lambda: output_processor)
    recorder = SimpleNamespace(
        record_user=lambda *_args: None, record_assistant=lambda *_args: None
    )

    def note_enrollment(_processor: FrameProcessor, frame: Frame) -> None:
        nonlocal stops_enrolled
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # This native hook runs after the segmented handler enrolled its WAV.
            stops_enrolled += 1
            if stops_enrolled == 3:
                c_enrolled.set()

    def note_delivery(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, TranscriptionFrame) and frame.text == "segment alpha":
            a_delivered.set()

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        stt = services_module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        stt.add_event_handler("on_after_process_frame", note_enrollment)
        pipeline = pipeline_module.build_pipeline(
            transport=transport,
            services=SimpleNamespace(stt=stt, llm=llm, tts=tts),
            controller=controller,
            turn_recorder=recorder,
            first_failure=first_failure,
        )
        user = next(
            processor
            for processor in pipeline.processors
            if isinstance(processor, LLMUserAggregator)
        )
        user.add_event_handler("on_after_process_frame", note_delivery)
        worker, run_task = await _start_native_stt_worker(pipeline)
        try:
            await worker.queue_frames(
                [
                    InputAudioRawFrame(b"\x30\x75" * 800, 8000, 1),
                    InputAudioRawFrame(b"\x00\x00" * 800, 8000, 1),
                ]
            )
            await asyncio.wait_for(entered[0].wait(), timeout=5)
            release[0].set()
            await asyncio.wait_for(a_delivered.wait(), timeout=2)
            while not llm_input.empty():
                initial_llm_frames.append(llm_input.get_nowait())
            assert not any(isinstance(frame, LLMContextFrame) for frame in initial_llm_frames)
            for _ in range(2):  # B, then C: both traverse real aggregator-originated VAD.
                await worker.queue_frames(
                    [
                        InputAudioRawFrame(b"\x30\x75" * 800, 8000, 1),
                        InputAudioRawFrame(b"\x00\x00" * 800, 8000, 1),
                    ]
                )
            await asyncio.wait_for(entered[1].wait(), timeout=2)
            await asyncio.wait_for(c_enrolled.wait(), timeout=2)
            release[1].set()
            await asyncio.wait_for(run_task, timeout=2)
        finally:
            for event in release:
                event.set()
            if not run_task.done():
                await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(run_task, timeout=2)

    observed = initial_llm_frames + [llm_input.get_nowait() for _ in range(llm_input.qsize())]
    assert len(requests) == 2  # C never starts provider work after B failed.
    assert not any(isinstance(frame, LLMContextFrame) for frame in observed), (
        "native_cancel_released_prior_A_after_B_fatal: the actual inference gate "
        "must stop provider work even while admission is still active"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("b_stopped_before_a", [True, False])
@pytest.mark.parametrize(
    ("transcripts", "expected_text"),
    [
        (("segment alpha", "segment bravo"), "segment alpha segment bravo"),
        (("", "segment bravo"), "segment bravo"),
        (("segment alpha", ""), "segment alpha"),
        (("", ""), ""),
    ],
)
async def test_stt_late_final_does_not_release_context_before_resumed_segment_transcript(
    b_stopped_before_a: bool,
    transcripts: tuple[str, str],
    expected_text: str,
) -> None:
    """Controlled VAD/COMPLETE ordering; this is not an acoustic qualification."""
    module = import_module("projetv0_voice.inference.services")
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    b_stop_processed = asyncio.Event()
    resumed_audio_processed = asyncio.Event()
    premature_context = asyncio.Event()
    first_audio_processed = asyncio.Event()
    requests: list[httpx.Request] = []
    context_snapshots: list[list[str]] = []
    emitted_transcripts: list[TranscriptionFrame] = []
    completed_receipt_processed = asyncio.Event()
    normal_context = asyncio.Event()
    first_audio = InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1)
    resumed_audio = InputAudioRawFrame(b"\x02\x00" * 800, 8000, 1)
    b_stop = VADUserStoppedSpeakingFrame(stop_secs=0.2)
    context = LLMContext()
    user = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[CompletionAwareTurnStopStrategy(turn_analyzer=_CompletePauseAnalyzer())]
            )
        ),
        realtime_service_mode=False,
    ).user()

    async def handler(request: httpx.Request) -> httpx.Response:
        index = len(requests)
        requests.append(request)
        entered[index].set()
        await release[index].wait()
        return httpx.Response(200, json={"text": transcripts[index]})

    def note_processed(_processor: FrameProcessor, frame: Frame) -> None:
        if frame is first_audio:
            first_audio_processed.set()
        elif frame is b_stop:
            b_stop_processed.set()
        elif frame is resumed_audio:
            resumed_audio_processed.set()
        elif (
            isinstance(frame, TranscriptionFrame)
            and frame.metadata.get(STT_COMPLETED_SEGMENT_KEY) == 2
        ):
            completed_receipt_processed.set()

    def note_transcript(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, TranscriptionFrame):
            emitted_transcripts.append(frame)

    def note_context(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, LLMContextFrame):
            # Snapshot at emission: native context frames share a mutable context.
            context_snapshots.append(
                [str(message["content"]) for message in frame.context.get_messages()]
            )
            if not release[1].is_set():
                premature_context.set()
            else:
                normal_context.set()

    user.add_event_handler("on_after_process_frame", note_processed)
    user.add_event_handler("on_before_push_frame", note_context)
    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        service.add_event_handler("on_before_push_frame", note_transcript)
        worker, run_task = await _start_native_stt_worker(service, user)
        try:
            await worker.queue_frames([VADUserStartedSpeakingFrame(), first_audio])
            await asyncio.wait_for(first_audio_processed.wait(), timeout=2)
            await worker.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await asyncio.wait_for(entered[0].wait(), timeout=5)
            await worker.queue_frames([VADUserStartedSpeakingFrame(), resumed_audio])
            await asyncio.wait_for(resumed_audio_processed.wait(), timeout=2)
            if b_stopped_before_a:
                await worker.queue_frame(b_stop)
                await asyncio.wait_for(b_stop_processed.wait(), timeout=2)

            # Release A while B is either still speaking or already queued for ASR.
            release[0].set()
            with suppress(TimeoutError):
                await asyncio.wait_for(premature_context.wait(), timeout=0.25)
            if not b_stopped_before_a:
                await worker.queue_frame(b_stop)
                await asyncio.wait_for(b_stop_processed.wait(), timeout=2)
            await asyncio.wait_for(entered[1].wait(), timeout=2)
            release[1].set()
            await asyncio.wait_for(completed_receipt_processed.wait(), timeout=2)
            if expected_text:
                await asyncio.wait_for(normal_context.wait(), timeout=2)
            assert not run_task.done()
            assert context_snapshots == ([[expected_text]] if expected_text else [])
        finally:
            for event in release:
                event.set()
            if not run_task.done():
                await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run_task, timeout=2)

    assert len(requests) == 2
    assert first_audio.audio in requests[0].content
    assert resumed_audio.audio in requests[1].content
    assert not premature_context.is_set(), (
        "late_final_A_released_context_while_B_pending: an old finalized segment "
        "must not certify that the resumed segment has been transcribed"
    )
    assert context_snapshots == ([[expected_text]] if expected_text else [])
    assert [message["content"] for message in context.get_messages()] == (
        [expected_text] if expected_text else []
    )
    assert [frame.text for frame in emitted_transcripts] == [expected_text]
    assert emitted_transcripts[0].finalized
    assert emitted_transcripts[0].result.text == transcripts[1]
    assert emitted_transcripts[0].metadata[STT_COMPLETED_SEGMENT_KEY] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_type", [CancelFrame, EndFrame])
async def test_stt_native_worker_terminal_preserves_cancel_or_drain_order(
    terminal_type: type[CancelFrame] | type[EndFrame],
) -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    cancelled = asyncio.Event()
    received: asyncio.Queue[Frame] = asyncio.Queue()
    observed: list[Frame] = []
    requests: list[httpx.Request] = []
    first_audio = InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1)
    resumed_audio = InputAudioRawFrame(b"\x02\x00" * 800, 8000, 1)
    b_stop = VADUserStoppedSpeakingFrame()
    terminal = terminal_type()
    terminal_queued = False

    async def handler(request: httpx.Request) -> httpx.Response:
        index = len(requests)
        requests.append(request)
        entered[index].set()
        try:
            await release[index].wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, json={"text": ["segment alpha", "segment bravo"][index]})

    async def receive_frame(target: Frame) -> None:
        while True:
            frame = await received.get()
            observed.append(frame)
            if frame is target:
                return

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        sink = QueuedFrameProcessor(queue=received, queue_direction=FrameDirection.DOWNSTREAM)
        worker, run_task = await _start_native_stt_worker(service, sink)
        try:
            await worker.queue_frames([VADUserStartedSpeakingFrame(), first_audio])
            await asyncio.wait_for(receive_frame(first_audio), timeout=2)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await asyncio.wait_for(entered[0].wait(), timeout=5)
            await worker.queue_frames([VADUserStartedSpeakingFrame(), resumed_audio, b_stop])
            await asyncio.wait_for(receive_frame(b_stop), timeout=2)
            await worker.queue_frame(terminal)
            terminal_queued = True

            if terminal_type is CancelFrame:
                await asyncio.wait_for(cancelled.wait(), timeout=1)
                await asyncio.wait_for(run_task, timeout=2)
            else:
                done, _ = await asyncio.wait({run_task}, timeout=0.05)
                assert not done, "EndFrame terminated before pending segments were drained"
                release[0].set()
                await asyncio.wait_for(entered[1].wait(), timeout=2)
                release[1].set()
                await asyncio.wait_for(run_task, timeout=2)
        finally:
            for event in release:
                event.set()
            if not run_task.done() and not terminal_queued:
                await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(run_task, timeout=2)

    while not received.empty():
        observed.append(received.get_nowait())
    transcripts = [frame.text for frame in observed if isinstance(frame, TranscriptionFrame)]
    assert any(frame is terminal for frame in observed)
    if terminal_type is CancelFrame:
        assert len(requests) == 1
        assert transcripts == []
    else:
        assert len(requests) == 2
        assert transcripts == ["segment alpha segment bravo"]
        assert observed.index(terminal) > max(
            index for index, frame in enumerate(observed) if isinstance(frame, TranscriptionFrame)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_type", [CancelFrame, EndFrame])
async def test_stt_terminal_discards_or_flushes_text_held_during_resumed_speech(
    terminal_type: type[CancelFrame] | type[EndFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = asyncio.Event()
    release = asyncio.Event()
    a_consumed = asyncio.Event()
    received: asyncio.Queue[Frame] = asyncio.Queue()
    observed: list[Frame] = []
    requests: list[httpx.Request] = []
    first_audio = InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1)
    resumed_audio = InputAudioRawFrame(b"\x02\x00" * 800, 8000, 1)
    terminal = terminal_type()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"text": "segment alpha"})

    async def receive_frame(target: Frame) -> None:
        while True:
            frame = await received.get()
            observed.append(frame)
            if frame is target:
                return

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        native_run_stt = service.run_stt

        async def observe_run_stt(audio: bytes) -> AsyncGenerator[Frame]:
            # Observe the production generator's completion; all frames and HTTP
            # still come from the real adapter running on the native segment task.
            async for frame in native_run_stt(audio):
                yield frame
            a_consumed.set()

        monkeypatch.setattr(service, "run_stt", observe_run_stt)
        sink = QueuedFrameProcessor(queue=received, queue_direction=FrameDirection.DOWNSTREAM)
        worker, run_task = await _start_native_stt_worker(service, sink)
        try:
            await worker.queue_frames([VADUserStartedSpeakingFrame(), first_audio])
            await asyncio.wait_for(receive_frame(first_audio), timeout=2)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await asyncio.wait_for(entered.wait(), timeout=5)
            await worker.queue_frames([VADUserStartedSpeakingFrame(), resumed_audio])
            await asyncio.wait_for(receive_frame(resumed_audio), timeout=2)
            release.set()
            await asyncio.wait_for(a_consumed.wait(), timeout=2)
            while not received.empty():
                observed.append(received.get_nowait())
            assert not any(isinstance(frame, TranscriptionFrame) for frame in observed)
            await worker.queue_frame(terminal)
            await asyncio.wait_for(run_task, timeout=2)
        finally:
            release.set()
            if not run_task.done():
                await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(run_task, timeout=2)
        assert [frame async for frame in service.run_stt(_wav())] == []

    while not received.empty():
        observed.append(received.get_nowait())
    assert len(requests) == 1  # B never stopped, so no untranscribed B text is invented.
    assert any(frame is terminal for frame in observed)
    transcripts = [frame for frame in observed if isinstance(frame, TranscriptionFrame)]
    if terminal_type is CancelFrame:
        assert transcripts == []
    else:
        assert [frame.text for frame in transcripts] == ["segment alpha"]
        assert transcripts[0].result.text == "segment alpha"
        assert transcripts[0].metadata[STT_TERMINAL_PARTIAL_KEY] is True
        assert observed.index(transcripts[0]) < observed.index(terminal)


@pytest.mark.asyncio
@pytest.mark.parametrize("text, expected_error", [("é" * 16384, None), ("é" * 16385, "limit")])
async def test_stt_batch_text_budget_is_32_kib_of_utf8_bytes(
    text: str, expected_error: str | None
) -> None:
    module = import_module("projetv0_voice.inference.services")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": text})

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        frames = [frame async for frame in service.run_stt(_wav())]
    assert len(frames) == 1
    if expected_error is None:
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == text
    else:
        assert isinstance(frames[0], FatalErrorFrame)
        assert frames[0].error == "openrouter_stt_text_limit"
        assert frames[0].exception is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["timeout", "http_error", "text_limit"])
async def test_stt_failed_resumed_segment_discards_batch_and_blocks_postmortem_http(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    fatal_seen = asyncio.Event()
    fatal_frames: list[FatalErrorFrame] = []
    cancelled = asyncio.Event()
    requests: list[httpx.Request] = []
    received: asyncio.Queue[Frame] = asyncio.Queue()
    b_stop = VADUserStoppedSpeakingFrame()

    async def handler(request: httpx.Request) -> httpx.Response:
        index = len(requests)
        requests.append(request)
        if index == 0:
            entered.set()
            await release_a.wait()
            return httpx.Response(200, json={"text": "segment alpha"})
        if failure_mode == "timeout":
            try:
                await release_b.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return httpx.Response(200, json={"text": "unreachable"})
        if failure_mode == "http_error":
            return httpx.Response(400, json={"error": {"message": "unit-provider-error"}})
        # 16385 two-byte characters exceed 32 KiB even without the retained A.
        return httpx.Response(200, json={"text": "é" * 16385})

    def note_fatal(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, FatalErrorFrame):
            fatal_frames.append(frame)
            fatal_seen.set()

    async def receive_frame(target: Frame) -> None:
        while await received.get() is not target:
            pass

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        service.add_event_handler("on_before_push_frame", note_fatal)
        sink = QueuedFrameProcessor(queue=received, queue_direction=FrameDirection.DOWNSTREAM)
        worker, run_task = await _start_native_stt_worker(service, sink)
        try:
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1),
                    VADUserStoppedSpeakingFrame(),
                ]
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(b"\x02\x00" * 800, 8000, 1),
                    b_stop,
                ]
            )
            await asyncio.wait_for(receive_frame(b_stop), timeout=2)
            if failure_mode == "timeout":
                # A is already inside its 8-second deadline; only B sees this budget.
                monkeypatch.setattr(module, "_STT_TIMEOUT_SECONDS", 0.03)
            release_a.set()
            await asyncio.wait_for(fatal_seen.wait(), timeout=2)
            await asyncio.wait_for(run_task, timeout=2)
        finally:
            release_a.set()
            release_b.set()
            if not run_task.done():
                await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(run_task, timeout=2)

        assert [frame async for frame in service.run_stt(_wav())] == []

    observed = [received.get_nowait() for _ in range(received.qsize())]
    assert len(requests) == 2
    assert [frame.error for frame in fatal_frames] == [
        {
            "timeout": "openrouter_stt_timeout",
            "http_error": "openrouter_stt_transport",
            "text_limit": "openrouter_stt_text_limit",
        }[failure_mode]
    ]
    assert not any(isinstance(frame, TranscriptionFrame) for frame in observed)
    assert all(frame.exception is None for frame in fatal_frames)
    if failure_mode == "timeout":
        assert cancelled.is_set()


@pytest.mark.asyncio
async def test_stt_pending_segment_limit_cancels_native_worker_without_transcript() -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    received: asyncio.Queue[Frame] = asyncio.Queue()
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, json={"text": "unreachable"})

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        sink = QueuedFrameProcessor(queue=received, queue_direction=FrameDirection.DOWNSTREAM)
        worker, run_task = await _start_native_stt_worker(service, sink)
        try:
            for index in range(5):
                await worker.queue_frames(
                    [
                        VADUserStartedSpeakingFrame(),
                        InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1),
                        VADUserStoppedSpeakingFrame(),
                    ]
                )
                if index == 0:
                    await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(run_task, timeout=2)
        finally:
            release.set()
            if not run_task.done():
                await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(run_task, timeout=2)
        assert [frame async for frame in service.run_stt(_wav())] == []

    observed = [received.get_nowait() for _ in range(received.qsize())]
    assert len(requests) == 1
    assert cancelled.is_set()
    assert [frame.error for frame in observed if isinstance(frame, FatalErrorFrame)] == [
        "openrouter_stt_segment_limit"
    ]
    assert not any(isinstance(frame, TranscriptionFrame) for frame in observed)


@pytest.mark.asyncio
async def test_stt_end_drain_deadline_cancels_http_without_late_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("projetv0_voice.inference.services")
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    received: asyncio.Queue[Frame] = asyncio.Queue()

    async def handler(_request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, json={"text": "unreachable"})

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        sink = QueuedFrameProcessor(queue=received, queue_direction=FrameDirection.DOWNSTREAM)
        worker, run_task = await _start_native_stt_worker(service, sink)
        try:
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1),
                    VADUserStoppedSpeakingFrame(),
                ]
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            monkeypatch.setattr(module, "_STT_DRAIN_TIMEOUT_SECONDS", 0.03)
            await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run_task, timeout=2)
        finally:
            release.set()
            if not run_task.done():
                await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(run_task, timeout=2)
        assert [frame async for frame in service.run_stt(_wav())] == []

    observed = [received.get_nowait() for _ in range(received.qsize())]
    assert cancelled.is_set()
    assert [frame.error for frame in observed if isinstance(frame, FatalErrorFrame)] == [
        "openrouter_stt_drain_timeout"
    ]
    assert not any(isinstance(frame, TranscriptionFrame) for frame in observed)


@pytest.mark.asyncio
async def test_stt_pending_http_allows_resumed_audio_to_reach_downstream() -> None:
    """A caller's resumed audio must reach downstream VAD while ASR is pending."""
    module = import_module("projetv0_voice.inference.services")
    started = asyncio.Event()
    asr_entered = asyncio.Event()
    release_asr = asyncio.Event()
    received: asyncio.Queue[Frame] = asyncio.Queue()
    observed: list[Frame] = []
    requests: list[httpx.Request] = []
    first_audio = InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1)
    resumed_audio = InputAudioRawFrame(b"\x02\x00" * 800, 8000, 1)
    resumed_audio_before_asr_release = False

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        asr_entered.set()
        await release_asr.wait()
        return httpx.Response(200, json={"text": "question de test"})

    async def receive_audio(target: InputAudioRawFrame) -> None:
        while True:
            frame = await received.get()
            observed.append(frame)
            if frame is target:
                return

    async with DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)) as client:
        service = module.build_stt(
            _profile(), SecretStr("unit-secret"), language="fr-FR", http_client=client
        )
        sink = QueuedFrameProcessor(
            queue=received, queue_direction=FrameDirection.DOWNSTREAM
        )
        worker = PipelineWorker(
            Pipeline([service, sink]),
            params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
        )

        async def note_started(_worker: PipelineWorker, _frame: StartFrame) -> None:
            started.set()

        worker.add_event_handler("on_pipeline_started", note_started)
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await runner.add_workers(worker)
        run_task = asyncio.create_task(runner.run())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await worker.queue_frames([VADUserStartedSpeakingFrame(), first_audio])
            await asyncio.wait_for(receive_audio(first_audio), timeout=2)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await asyncio.wait_for(asr_entered.wait(), timeout=5)

            # Exercise the native worker/input queue, not a direct process_frame call.
            # The HTTP response stays withheld throughout this observation window.
            await worker.queue_frame(resumed_audio)
            try:
                await asyncio.wait_for(receive_audio(resumed_audio), timeout=0.25)
            except TimeoutError:
                pass
            else:
                resumed_audio_before_asr_release = not release_asr.is_set()

            release_asr.set()
            if not resumed_audio_before_asr_release:
                # Establish that it was delayed by ASR, rather than lost by the probe.
                await asyncio.wait_for(receive_audio(resumed_audio), timeout=2)
        finally:
            release_asr.set()
            if not run_task.done():
                await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run_task, timeout=2)

    while not received.empty():
        observed.append(received.get_nowait())
    assert len(requests) == 1
    assert b'name="language"\r\n\r\nfr\r\n' in requests[0].content
    assert first_audio.audio in requests[0].content
    assert any(frame is resumed_audio for frame in observed)
    assert any(
        isinstance(frame, TranscriptionFrame) and frame.text == "question de test"
        for frame in observed
    )
    assert not any(isinstance(frame, FatalErrorFrame) for frame in observed)
    assert resumed_audio_before_asr_release, (
        "resumed_audio_before_asr_release=False: native STT delayed resumed audio "
        "until its pending HTTP transcription completed"
    )


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
        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == ""
        assert frames[0].metadata[STT_COMPLETED_SEGMENT_KEY] == 1


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
