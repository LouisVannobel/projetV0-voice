from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from importlib import import_module
from importlib.metadata import version
from types import SimpleNamespace

import httpx
import pytest
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.base_worker import WorkerParams
from pipecat.workers.runner import WorkerRunner
from pydantic import SecretStr

from projetv0_voice.inference.openrouter_tts import OpenRouterTTSService
from projetv0_voice.qualified_profile import InferenceProfileV1
from projetv0_voice.telnyx.frames import TelnyxMarkFrame

pipeline_module = import_module("projetv0_voice.pipeline")


class _CleanupProbe(FrameProcessor):
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        failure: str | None = None,
        cancel: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._calls = calls
        self._failure = failure
        self._cancel = cancel

    async def cleanup(self) -> None:
        self._calls.append(self.name)
        if self._cancel:
            raise asyncio.CancelledError
        if self._failure is not None:
            raise RuntimeError(self._failure)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _GateController:
    def __init__(self, *, active: bool = False, fail_messages: bool = False) -> None:
        self.active = active
        self.fail_messages = fail_messages
        self.mark_name = "expected-mark"
        self.messages: list[str] = []
        self.aborts: list[str] = []
        self.audio = 0
        self.armed = 0
        self.forwarded = 0
        self.forwarded_event = asyncio.Event()

    def is_active(self) -> bool:
        return self.active

    async def accept_mark(self, mark_name: str) -> bool:
        if self.fail_messages:
            raise RuntimeError("transport-secret")
        self.messages.append(mark_name)
        return True

    async def abort(self, code: str) -> None:
        self.aborts.append(code)
        self.active = False

    async def note_disclosure_audio(self) -> None:
        self.audio += 1

    async def arm_expected_mark(self) -> bool:
        self.armed += 1
        return self.audio > 0 and not self.aborts

    async def mark_forwarded(self) -> None:
        self.forwarded += 1
        self.forwarded_event.set()


class _TriggerFilterError(FrameProcessor):
    def __init__(self, target: FunctionFilter) -> None:
        super().__init__()
        self._target = target

    async def process_frame(self, frame: object, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)  # type: ignore[arg-type]
        if isinstance(frame, TextFrame):
            await self._target.push_error_frame(
                ErrorFrame(
                    error="filter-provider-secret",
                    exception=RuntimeError("filter-provider-secret"),
                )
            )
        await self.push_frame(frame, direction)  # type: ignore[arg-type]


class _Transport:
    def __init__(self) -> None:
        self.input_processor = _SetupProbe("transport-input")
        self.output_processor = _SetupProbe("transport-output")

    def input(self) -> FrameProcessor:
        return self.input_processor

    def output(self) -> FrameProcessor:
        return self.output_processor


class _Turns:
    def __init__(self) -> None:
        self.user: list[str | None] = []
        self.assistant: list[tuple[str, bool]] = []

    def record_user(self, _content: str | None, _timestamp: str) -> None:
        self.user.append(_content)

    def record_assistant(self, _content: str, _timestamp: str, _interrupted: bool) -> None:
        self.assistant.append((_content, _interrupted))


class _SetupProbe(FrameProcessor):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name, enable_direct_mode=True)
        self.started = asyncio.Event()
        self.setup_manager: object | None = None
        self.start_frame: StartFrame | None = None
        self.frames: list[object] = []
        self.tts_speak_event = asyncio.Event()

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        self.setup_manager = setup.task_manager

    async def process_frame(self, frame: object, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)  # type: ignore[arg-type]
        self.frames.append(frame)
        if isinstance(frame, TTSSpeakFrame):
            self.tts_speak_event.set()
        if isinstance(frame, StartFrame):
            self.start_frame = frame
            self.started.set()
        await self.push_frame(frame, direction)  # type: ignore[arg-type]


class _SegmentedSttProbe(SegmentedSTTService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.segments: list[bytes] = []

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self.started.set()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None]:
        self.segments.append(audio)
        yield TranscriptionFrame(
            text="bonjour",
            user_id="",
            timestamp="2026-08-28T18:00:00+00:00",
            language=Language.FR,
        )


class _OfflineTtsProcessor(FrameProcessor):
    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=b"\x01\x00" * 80,
                    sample_rate=8000,
                    num_channels=1,
                    context_id="greeting",
                ),
                direction,
            )
            await self.push_frame(TTSStoppedFrame(context_id="greeting"), direction)
        else:
            await self.push_frame(frame, direction)


class _FailingChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        raise RuntimeError("tts-provider-secret")

    async def aclose(self) -> None:
        self.closed = True


def test_pinned_public_runner_pipeline_filter_and_message_contracts() -> None:
    assert version("pipecat-ai") == "1.7.0"
    assert issubclass(pipeline_module.ObservedTaskManager, TaskManager)
    assert issubclass(pipeline_module.ObservedPipeline, Pipeline)
    assert issubclass(OutputTransportMessageFrame, object)
    assert list(inspect.signature(WorkerRunner).parameters) == [
        "name",
        "bus",
        "handle_sigint",
        "handle_sigterm",
        "force_gc",
        "check_dangling_tasks",
        "loop",
        "task_manager",
    ]
    assert list(inspect.signature(WorkerRunner.run).parameters) == [
        "self",
        "worker",
        "auto_end",
    ]
    assert list(inspect.signature(WorkerRunner.add_workers).parameters) == ["self", "workers"]
    assert list(inspect.signature(PipelineWorker).parameters)[-1] == "tool_resources"
    assert list(inspect.signature(WorkerParams).parameters) == ["task_manager"]
    assert list(inspect.signature(FrameProcessorSetup).parameters) == [
        "clock",
        "task_manager",
        "pipeline_worker",
        "observer",
        "tool_resources",
    ]
    assert list(inspect.signature(FunctionFilter).parameters) == [
        "filter",
        "direction",
        "filter_system_frames",
        "kwargs",
    ]


@pytest.mark.asyncio
async def test_observed_manager_turns_nested_failure_constant_safe_without_raw_log() -> None:
    failure = pipeline_module.FirstFailure()
    manager = pipeline_module.ObservedTaskManager(
        first_failure=failure,
        loop=asyncio.get_running_loop(),
    )
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    async def secret_failure() -> None:
        raise RuntimeError("provider-secret")

    try:
        task = manager.create_task(secret_failure(), "nested-secret-task")
        await task
        assert await asyncio.wait_for(failure.wait(), timeout=0.1) == "pipeline_task_failed"
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert "provider-secret" not in rendered
    assert "pipeline_task_failed" not in rendered


@pytest.mark.asyncio
async def test_observed_manager_closes_inner_coroutine_when_cancelled_before_cpu() -> None:
    failure = pipeline_module.FirstFailure()
    manager = pipeline_module.ObservedTaskManager(
        first_failure=failure,
        loop=asyncio.get_running_loop(),
    )

    async def never_started() -> None:
        await asyncio.sleep(1)

    coroutine = never_started()
    task = manager.create_task(coroutine, "cancel-before-start")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    assert failure.code is None


@pytest.mark.asyncio
async def test_observed_pipeline_attempts_every_child_and_signals_cleanup_failure() -> None:
    calls: list[str] = []
    failure = pipeline_module.FirstFailure()
    pipeline = pipeline_module.ObservedPipeline(
        [
            _CleanupProbe("first", calls),
            _CleanupProbe("middle", calls, failure="cleanup-provider-secret"),
            _CleanupProbe("cancelled", calls, cancel=True),
            _CleanupProbe("last", calls),
        ],
        first_failure=failure,
    )

    await pipeline.cleanup()

    assert failure.code == "pipeline_cleanup_failed"
    assert calls == ["first", "middle", "cancelled", "last"]


@pytest.mark.asyncio
async def test_real_worker_finishes_after_observed_pipeline_cleanup_failure() -> None:
    calls: list[str] = []
    failure = pipeline_module.FirstFailure()
    pipeline = pipeline_module.ObservedPipeline(
        [
            _CleanupProbe("middle", calls, failure="cleanup-provider-secret"),
            _CleanupProbe("last", calls),
        ],
        first_failure=failure,
    )
    runtime = pipeline_module.build_runtime(
        pipeline=pipeline,
        first_failure=failure,
        greeting="Disclosure.",
        mark_name="mark",
        idle_timeout_seconds=60.0,
    )
    started = asyncio.Event()
    finished = asyncio.Event()

    async def on_started(_worker: object, _frame: StartFrame) -> None:
        started.set()

    async def on_finished(_worker: object, _frame: Frame) -> None:
        finished.set()

    runtime.worker.add_event_handler("on_pipeline_started", on_started)
    runtime.worker.add_event_handler("on_pipeline_finished", on_finished)
    running = asyncio.create_task(runtime.runner.run(auto_end=True))
    await runtime.runner.add_workers(runtime.worker)
    await asyncio.wait_for(started.wait(), timeout=3)
    await runtime.worker.queue_frame(EndFrame())
    await asyncio.wait_for(running, timeout=3)

    assert finished.is_set()
    assert failure.code == "pipeline_cleanup_failed"
    assert calls == ["middle", "last"]


@pytest.mark.asyncio
async def test_inference_error_boundary_replaces_secret_and_preserves_disposition() -> None:
    stt = FrameProcessor(name="stt")
    llm = _SetupProbe("llm")
    tts = FrameProcessor(name="tts")
    boundary = pipeline_module.InferenceErrorBoundary(stt=stt, llm=llm, tts=tts)
    secret = RuntimeError("provider-secret")
    frame = ErrorFrame(
        error="provider-secret",
        fatal=True,
        processor=tts,
        exception=secret,
    )

    _, upstream = await run_test(
        boundary,
        frames_to_send=[frame],
        frames_to_send_direction=FrameDirection.UPSTREAM,
        expected_up_frames=[ErrorFrame],
    )

    sanitized = upstream[0]
    assert isinstance(sanitized, ErrorFrame)
    assert sanitized.error == "tts_failed"
    assert sanitized.fatal is True
    assert sanitized.processor is None
    assert sanitized.exception is None
    assert "provider-secret" not in repr(sanitized)

    already_safe = ErrorFrame(error="tts_failed", fatal=True)
    _, upstream = await run_test(
        pipeline_module.InferenceErrorBoundary(stt=stt, llm=llm, tts=tts),
        frames_to_send=[already_safe],
        frames_to_send_direction=FrameDirection.UPSTREAM,
        expected_up_frames=[ErrorFrame],
    )
    assert upstream[0].error == "tts_failed"


def test_pipeline_parameters_are_frozen_to_eight_khz_metrics_contract() -> None:
    params = pipeline_module.pipeline_params()
    assert isinstance(params, PipelineParams)
    assert params.audio_in_sample_rate == 8000
    assert params.audio_out_sample_rate == 8000
    assert params.enable_metrics is True
    assert params.enable_usage_metrics is True


@pytest.mark.asyncio
async def test_direct_input_gate_consumes_control_and_rechecks_audio_before_stt() -> None:
    controller = _GateController(active=False)
    failure = pipeline_module.FirstFailure()
    gate = pipeline_module.build_input_gate(controller=controller, first_failure=failure)

    downstream, _ = await run_test(
        gate,
        frames_to_send=[
            InputAudioRawFrame(audio=b"\x00\x00", sample_rate=8000, num_channels=1),
            VADUserStartedSpeakingFrame(),
            UserSpeakingFrame(),
            InterruptionFrame(),
            InputTransportMessageFrame(
                message={"event": "mark", "mark": {"name": "expected-mark"}}
            ),
            ErrorFrame(error="already-safe", fatal=False),
        ],
        expected_down_frames=[InterruptionFrame, ErrorFrame],
    )
    assert [type(frame) for frame in downstream] == [InterruptionFrame, ErrorFrame]
    assert controller.messages == ["expected-mark"]
    assert controller.aborts == ["disclosure_failed"]

    controller.active = True
    downstream, _ = await run_test(
        pipeline_module.build_input_gate(controller=controller, first_failure=failure),
            frames_to_send=[
                InputAudioRawFrame(
                    audio=b"\x01\x00", sample_rate=8000, num_channels=1
                ),
                UserSpeakingFrame(),
            ],
        expected_down_frames=[InputAudioRawFrame, UserSpeakingFrame],
    )
    assert len(downstream) == 2

    controller.active = False
    downstream, _ = await run_test(
        pipeline_module.build_input_gate(controller=controller, first_failure=failure),
        frames_to_send=[
            InputAudioRawFrame(audio=b"\x02\x00", sample_rate=8000, num_channels=1)
        ],
        expected_down_frames=[],
    )
    assert downstream == []


@pytest.mark.asyncio
async def test_input_gate_predicate_and_inline_error_handler_are_total_and_constant_safe() -> None:
    controller = _GateController(fail_messages=True)
    failure = pipeline_module.FirstFailure()
    gate = pipeline_module.build_input_gate(controller=controller, first_failure=failure)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        downstream, _ = await run_test(
            gate,
            frames_to_send=[
                InputTransportMessageFrame(
                    message={"event": "mark", "mark": {"name": "expected-mark"}}
                )
            ],
            expected_down_frames=[],
        )
        assert downstream == []

        linked = Pipeline([gate, _TriggerFilterError(gate)])
        _, upstream = await run_test(
            linked,
            frames_to_send=[TextFrame("trigger")],
            expected_up_frames=[ErrorFrame],
        )
    finally:
        logger.remove(sink)

    sanitized = upstream[0]
    assert isinstance(sanitized, ErrorFrame)
    assert sanitized.error == "input_gate_failed"
    assert sanitized.fatal is True
    assert sanitized.exception is None
    assert sanitized.processor is None
    assert failure.code == "input_gate_failed"
    rendered = "\n".join(messages)
    assert "transport-secret" not in rendered
    assert "filter-provider-secret" not in rendered


@pytest.mark.asyncio
async def test_input_gate_public_lifecycle_and_upstream_error_paths_are_preserved() -> None:
    controller = _GateController(active=False)
    failure = pipeline_module.FirstFailure()

    downstream, _ = await run_test(
        pipeline_module.build_input_gate(controller=controller, first_failure=failure),
        frames_to_send=[EndFrame()],
        ignore_start=False,
        send_end_frame=False,
        expected_down_frames=[StartFrame, EndFrame],
    )
    assert [type(frame) for frame in downstream] == [StartFrame, EndFrame]

    downstream, _ = await run_test(
        pipeline_module.build_input_gate(controller=controller, first_failure=failure),
        frames_to_send=[CancelFrame()],
        ignore_start=False,
        send_end_frame=False,
        expected_down_frames=[StartFrame, CancelFrame],
    )
    assert [type(frame) for frame in downstream] == [StartFrame, CancelFrame]

    _, upstream = await run_test(
        pipeline_module.build_input_gate(controller=controller, first_failure=failure),
        frames_to_send=[ErrorFrame(error="safe", fatal=False)],
        frames_to_send_direction=FrameDirection.UPSTREAM,
        expected_up_frames=[ErrorFrame],
    )
    assert upstream[0].error == "safe"


@pytest.mark.asyncio
async def test_disclosure_barrier_requires_audio_ignores_stop_and_forwards_one_expected_mark(
) -> None:
    controller = _GateController()
    barrier = pipeline_module.DisclosureOutputBarrier(controller=controller)
    audio = TTSAudioRawFrame(
        audio=b"\x01\x00\x02\x00",
        sample_rate=8000,
        num_channels=1,
        context_id="greeting",
    )
    stopped = TTSStoppedFrame(context_id="greeting")
    mark = TelnyxMarkFrame("expected-mark")

    downstream, _ = await run_test(
        barrier,
        frames_to_send=[audio, stopped, mark],
        expected_down_frames=[TTSAudioRawFrame, TTSStoppedFrame, TelnyxMarkFrame],
    )

    assert downstream == [audio, stopped, mark]
    assert controller.audio == 1
    assert controller.armed == 1
    assert controller.forwarded == 1

    empty = _GateController()
    downstream, _ = await run_test(
        pipeline_module.DisclosureOutputBarrier(controller=empty),
        frames_to_send=[TTSStoppedFrame(context_id="greeting"), TelnyxMarkFrame("expected-mark")],
        expected_down_frames=[TTSStoppedFrame],
    )
    assert len(downstream) == 1
    assert empty.aborts == ["disclosure_failed"]


def test_build_pipeline_has_exact_native_context_and_project_boundary_order() -> None:
    transport = _Transport()
    stt = FrameProcessor(name="stt")
    llm = FrameProcessor(name="llm")
    tts = FrameProcessor(name="tts")
    controller = _GateController()
    pipeline = pipeline_module.build_pipeline(
        transport=transport,
        services=SimpleNamespace(stt=stt, llm=llm, tts=tts),
        controller=controller,
        turn_recorder=_Turns(),
        first_failure=pipeline_module.FirstFailure(),
    )

    public = pipeline.processors
    assert public[1] is transport.input_processor
    assert isinstance(public[2], FunctionFilter)
    assert isinstance(public[3], pipeline_module.InferenceErrorBoundary)
    assert public[4] is stt
    assert isinstance(public[5], LLMUserAggregator)
    assert public[6] is llm
    assert public[7] is tts
    assert isinstance(public[8], pipeline_module.DisclosureOutputBarrier)
    assert public[9] is transport.output_processor
    assert isinstance(public[10], LLMAssistantAggregator)
    assert all(type(processor).__name__ != "TranscriptProcessor" for processor in public)
    assert all(type(processor).__name__ != "AudioBufferProcessor" for processor in public)


@pytest.mark.asyncio
async def test_runtime_propagates_runner_manager_and_queues_exact_disclosure_pair() -> None:
    failure = pipeline_module.FirstFailure()
    probe = _SetupProbe()
    runtime = pipeline_module.build_runtime(
        pipeline=pipeline_module.ObservedPipeline([probe], first_failure=failure),
        first_failure=failure,
        greeting="Bonjour, appel automatise.",
        mark_name="expected-mark",
        idle_timeout_seconds=60.0,
    )
    with pytest.raises(Exception, match="TaskManager is not initialized"):
        _ = runtime.worker.task_manager

    runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
    await runtime.runner.add_workers(runtime.worker)
    await probe.started.wait()
    assert probe.setup_manager is runtime.task_manager
    assert runtime.worker.task_manager is runtime.task_manager
    assert probe.start_frame is not None
    assert probe.start_frame.audio_in_sample_rate == 8000
    assert probe.start_frame.audio_out_sample_rate == 8000
    assert probe.start_frame.enable_metrics is True
    assert probe.start_frame.enable_usage_metrics is True

    await asyncio.wait_for(probe.tts_speak_event.wait(), timeout=1)
    queued = [
        frame for frame in probe.frames if isinstance(frame, (TTSSpeakFrame, TelnyxMarkFrame))
    ]
    assert len(queued) == 2
    assert isinstance(queued[0], TTSSpeakFrame)
    assert queued[0].text == "Bonjour, appel automatise."
    assert queued[0].append_to_context is False
    assert isinstance(queued[1], TelnyxMarkFrame)
    assert queued[1].mark_name == "expected-mark"

    await runtime.worker.queue_frame(EndFrame())
    await asyncio.wait_for(runner_task, timeout=1)


@pytest.mark.asyncio
async def test_build_runtime_returns_owned_unregistered_runtime_before_add_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = pipeline_module.FirstFailure()
    pipeline = pipeline_module.ObservedPipeline(
        [_SetupProbe()],
        first_failure=failure,
    )

    async def unexpected_add(_runner: object, *_workers: object) -> None:
        raise AssertionError("build_runtime registered a worker")

    monkeypatch.setattr(pipeline_module.WorkerRunner, "add_workers", unexpected_add)
    built = pipeline_module.build_runtime(
        pipeline=pipeline,
        first_failure=failure,
        greeting="Disclosure.",
        mark_name="mark",
        idle_timeout_seconds=60.0,
    )
    assert isinstance(built, pipeline_module.CallRuntime)


@pytest.mark.asyncio
async def test_real_worker_sets_silero_to_eight_khz_and_vad_stop_segments_stt() -> None:
    transport = _Transport()
    stt = _SegmentedSttProbe()
    llm = _SetupProbe("llm")
    tts = _OfflineTtsProcessor()
    controller = _GateController(active=True)
    failure = pipeline_module.FirstFailure()
    turns = _Turns()
    pipeline = pipeline_module.build_pipeline(
        transport=transport,
        services=SimpleNamespace(stt=stt, llm=llm, tts=tts),
        controller=controller,
        turn_recorder=turns,
        first_failure=failure,
    )
    user_aggregator = pipeline.processors[5]
    runtime = pipeline_module.build_runtime(
        pipeline=pipeline,
        first_failure=failure,
        greeting="Disclosure seulement.",
        mark_name=controller.mark_name,
        idle_timeout_seconds=60.0,
    )
    runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
    await runtime.runner.add_workers(runtime.worker)
    try:
        await asyncio.wait_for(stt.started.wait(), timeout=5)
        stt_rate = stt.sample_rate

        await asyncio.wait_for(controller.forwarded_event.wait(), timeout=5)
        greeting_messages = list(user_aggregator.context.get_messages())
        await runtime.worker.queue_frames(
            [
                VADUserStartedSpeakingFrame(),
                InputAudioRawFrame(
                    audio=b"\x01\x00" * 400,
                    sample_rate=8000,
                    num_channels=1,
                ),
                VADUserStoppedSpeakingFrame(),
                EndFrame(),
            ]
        )
        await asyncio.wait_for(runner_task, timeout=5)
    finally:
        if not runner_task.done():
            await runtime.runner.cancel(reason="test_cleanup")
            await asyncio.wait_for(runner_task, timeout=5)

    assert stt_rate == 8000
    assert greeting_messages == []
    assert turns.assistant == []
    assert turns.user == ["bonjour"]
    assert len(stt.segments) == 1
    assert len(stt.segments[0]) > 44


@pytest.mark.asyncio
async def test_actual_task7_partial_tts_failure_never_forwards_disclosure_mark() -> None:
    stream = _FailingChunkStream([b"\x01\x00" * 80])

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/pcm"},
            stream=stream,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    profile = InferenceProfileV1.model_validate(
        {
            "schema_version": 1,
            "stt_model": "test/stt",
            "llm_model": "test/llm",
            "tts_model": "test/tts",
            "tts_voice": "test-voice",
            "tts_pcm_sample_rate": 8000,
            "tts_pcm_channels": 1,
            "llm_provider_policy": {"allow_fallbacks": False},
            "tts_provider_options": {},
        }
    )
    tts = OpenRouterTTSService(
        profile=profile,
        api_key=SecretStr("offline-secret"),
        http_client=client,
    )
    transport = _Transport()
    stt = _SetupProbe("stt")
    llm = _SetupProbe("llm")
    controller = _GateController()
    failure = pipeline_module.FirstFailure()
    pipeline = pipeline_module.build_pipeline(
        transport=transport,
        services=SimpleNamespace(stt=stt, llm=llm, tts=tts),
        controller=controller,
        turn_recorder=_Turns(),
        first_failure=failure,
    )
    output = transport.output_processor
    runtime = pipeline_module.build_runtime(
        pipeline=pipeline,
        first_failure=failure,
        greeting="Disclosure.",
        mark_name=controller.mark_name,
        idle_timeout_seconds=60.0,
    )
    messages: list[str] = []
    log_sink = logger.add(messages.append, format="{message}")
    runner_task = asyncio.create_task(runtime.runner.run(auto_end=True))
    await runtime.runner.add_workers(runtime.worker)
    try:
        assert await asyncio.wait_for(failure.wait(), timeout=5) == "tts_failed"
        await runtime.runner.cancel(reason="local_failure")
        await asyncio.wait_for(runner_task, timeout=5)
    finally:
        logger.remove(log_sink)
        if not runner_task.done():
            await runtime.runner.cancel(reason="test_cleanup")
            await asyncio.wait_for(runner_task, timeout=5)
        await client.aclose()

    assert any(isinstance(frame, TTSAudioRawFrame) for frame in output.frames)
    assert not any(isinstance(frame, TelnyxMarkFrame) for frame in output.frames)
    assert controller.active is False
    assert "tts_failed" in controller.aborts
    assert stream.closed is True
    assert "tts-provider-secret" not in "\n".join(messages)
