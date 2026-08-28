"""Per-call Pipecat pipeline composition and safe public-surface adapters."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Coroutine, Sequence
from contextvars import Context
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.runner import WorkerRunner

from projetv0_voice.telnyx.frames import TelnyxMarkFrame


class FirstFailure:
    """One constant-safe failure signal shared by a call's native tasks."""

    _SAFE_CODES = frozenset(
        {
            "call_failed",
            "disclosure_commit_failed",
            "disclosure_failed",
            "disclosure_timeout",
            "input_gate_failed",
            "lease_terminalization_failed",
            "persistence_failed",
            "pipeline_cleanup_failed",
            "pipeline_task_failed",
            "recording_cleanup_failed",
            "recording_failed",
            "service_close_failed",
            "transport_disconnected",
            "transport_session_timeout",
            "tts_failed",
            "writer_failed",
        }
    )

    def __init__(self, *, shared_failure_event: asyncio.Event | None = None) -> None:
        self._event = asyncio.Event()
        self._code: str | None = None
        self._shared_failure_event = shared_failure_event

    @property
    def code(self) -> str | None:
        return self._code

    def signal(self, code: str) -> None:
        safe_code = code if code in self._SAFE_CODES else "call_failed"
        if self._code is None:
            self._code = safe_code
            self._event.set()

    async def wait(self) -> str:
        shared = self._shared_failure_event
        if shared is None:
            await self._event.wait()
            return self._code or "call_failed"
        if shared.is_set():
            self.signal("writer_failed")
            return self._code or "writer_failed"
        local_wait = asyncio.create_task(self._event.wait(), name="wait-local-call-failure")
        shared_wait = asyncio.create_task(shared.wait(), name="wait-shared-writer-failure")
        waits = (local_wait, shared_wait)
        try:
            done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            if shared_wait in done and shared.is_set():
                self.signal("writer_failed")
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            for task in waits:
                task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
            raise
        return self._code or "call_failed"


class ObservedTaskManager(TaskManager):
    """Suppress raw nested-task failures and surface one safe call failure."""

    def __init__(self, *, first_failure: FirstFailure, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._first_failure = first_failure

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        name: str,
        context: Context | None = None,
    ) -> asyncio.Task[Any]:
        async def observe() -> Any:
            try:
                return await coroutine
            except asyncio.CancelledError:
                raise
            except Exception:
                self._first_failure.signal("pipeline_task_failed")
                return None

        task = super().create_task(observe(), name, context)

        def close_unstarted_inner(_task: asyncio.Task[Any]) -> None:
            if inspect.iscoroutine(coroutine) and (
                inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
            ):
                coroutine.close()

        task.add_done_callback(close_unstarted_inner)
        return task


class ObservedPipeline(Pipeline):
    """Use only public cleanup surfaces while attempting every child."""

    def __init__(
        self,
        processors: Sequence[FrameProcessor],
        *,
        first_failure: FirstFailure,
        source: FrameProcessor | None = None,
        sink: FrameProcessor | None = None,
    ) -> None:
        super().__init__(processors, source=source, sink=sink)
        self._first_failure = first_failure

    async def cleanup(self) -> None:
        failed = False
        try:
            await FrameProcessor.cleanup(self)  # type: ignore[no-untyped-call]
        except asyncio.CancelledError:
            failed = True
        except Exception:
            failed = True

        for processor in self.processors:
            try:
                await cast(_CleanupProcessor, processor).cleanup()
            except asyncio.CancelledError:
                failed = True
            except Exception:
                failed = True

        if failed:
            self._first_failure.signal("pipeline_cleanup_failed")


class InferenceErrorBoundary(FrameProcessor):
    """Replace upstream inference errors before they cross the project boundary."""

    def __init__(
        self,
        *,
        stt: FrameProcessor,
        llm: FrameProcessor,
        tts: FrameProcessor,
    ) -> None:
        super().__init__(name="InferenceErrorBoundary", enable_direct_mode=True)
        self._stages = {
            id(stt): "stt_failed",
            id(llm): "llm_failed",
            id(tts): "tts_failed",
        }

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame) and direction is FrameDirection.UPSTREAM:
            existing_stage = (
                frame.error
                if frame.error in {"stt_failed", "llm_failed", "tts_failed"}
                else None
            )
            safe = ErrorFrame(
                error=self._stages.get(
                    id(frame.processor), existing_stage or "inference_failed"
                ),
                fatal=frame.fatal,
                processor=None,
                exception=None,
            )
            await self.push_frame(safe, FrameDirection.UPSTREAM)
            return
        await self.push_frame(frame, direction)


class _CleanupProcessor(Protocol):
    async def cleanup(self) -> None: ...


class GateController(Protocol):
    mark_name: str

    def is_active(self) -> bool: ...

    async def accept_mark(self, mark_name: str) -> bool: ...

    async def abort(self, code: str) -> None: ...

    async def note_disclosure_audio(self) -> None: ...

    async def arm_expected_mark(self) -> bool: ...

    async def mark_forwarded(self) -> None: ...


_CLOSED_INPUT_FRAMES = (
    InputDTMFFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)


def build_input_gate(
    *,
    controller: GateController,
    first_failure: FirstFailure,
) -> FunctionFilter:
    """Build the one direct, total, call-owned input/control gate."""

    async def fail_closed() -> None:
        first_failure.signal("input_gate_failed")
        try:
            await controller.abort("input_gate_failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def predicate(frame: Frame) -> bool:
        try:
            if isinstance(frame, (StartFrame, EndFrame, CancelFrame, ErrorFrame)):
                return True
            if isinstance(frame, InputTransportMessageFrame):
                message = frame.message
                if not isinstance(message, dict):
                    raise ValueError
                event = message.get("event")
                if event == "mark":
                    mark = message.get("mark")
                    if not isinstance(mark, dict) or not isinstance(mark.get("name"), str):
                        raise ValueError
                    await controller.accept_mark(mark["name"])
                elif event in {"stop", "error"}:
                    await controller.abort("call_failed")
                else:
                    raise ValueError
                return False
            if isinstance(frame, InputAudioRawFrame):
                return controller.is_active()
            if isinstance(frame, InterruptionFrame) and not controller.is_active():
                await controller.abort("disclosure_failed")
                return True
            return not (
                isinstance(frame, _CLOSED_INPUT_FRAMES) and not controller.is_active()
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            await fail_closed()
            return False

    gate = FunctionFilter(
        filter=predicate,
        direction=FrameDirection.DOWNSTREAM,
        filter_system_frames=True,
        enable_direct_mode=True,
        name="CallInputGate",
    )

    async def sanitize_gate_error(_gate: FunctionFilter, error: ErrorFrame) -> None:
        error.error = "input_gate_failed"
        error.fatal = True
        error.exception = None
        error.processor = None
        await fail_closed()

    gate.add_event_handler("on_error", sanitize_gate_error)
    return gate


class DisclosureOutputBarrier(FrameProcessor):
    """Keep the expected Telnyx mark behind observed standalone TTS audio."""

    def __init__(self, *, controller: GateController) -> None:
        super().__init__(name="DisclosureOutputBarrier", enable_direct_mode=True)
        self._controller = controller

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSAudioRawFrame):
            try:
                await self._controller.note_disclosure_audio()
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._abort_safely()
                return
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TelnyxMarkFrame):
            if frame.mark_name != self._controller.mark_name:
                await self._abort_safely()
                return
            try:
                armed = await self._controller.arm_expected_mark()
                if not armed:
                    await self._abort_safely()
                    return
                await self.push_frame(frame, direction)
                await self._controller.mark_forwarded()
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._abort_safely()
            return
        await self.push_frame(frame, direction)

    async def _abort_safely(self) -> None:
        try:
            await self._controller.abort("disclosure_failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            return


class PipelineTransport(Protocol):
    def input(self) -> FrameProcessor: ...

    def output(self) -> FrameProcessor: ...


class PipelineServices(Protocol):
    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor


class PipelineTurnRecorder(Protocol):
    def record_user(self, content: str | None, timestamp: str) -> None: ...

    def record_assistant(self, content: str, timestamp: str, interrupted: bool) -> None: ...


def _sanitize_inline_error(error: ErrorFrame, code: str) -> None:
    error.error = code
    error.fatal = True
    error.exception = None
    error.processor = None


def build_pipeline(
    *,
    transport: PipelineTransport,
    services: PipelineServices,
    controller: GateController,
    turn_recorder: PipelineTurnRecorder,
    first_failure: FirstFailure,
) -> ObservedPipeline:
    """Compose exactly one Task 8 call pipeline from native processors."""

    input_gate = build_input_gate(controller=controller, first_failure=first_failure)
    inference_boundary = InferenceErrorBoundary(
        stt=services.stt,
        llm=services.llm,
        tts=services.tts,
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(sample_rate=8000),
            user_turn_strategies=UserTurnStrategies(),
        ),
        realtime_service_mode=False,
    )
    user_aggregator = aggregators.user()
    assistant_aggregator = aggregators.assistant()

    def record_user_turn(
        _aggregator: FrameProcessor,
        _strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ) -> None:
        try:
            turn_recorder.record_user(message.content, message.timestamp)
        except Exception:
            first_failure.signal("persistence_failed")

    def record_assistant_turn(
        _aggregator: FrameProcessor,
        message: AssistantTurnStoppedMessage,
    ) -> None:
        try:
            turn_recorder.record_assistant(
                message.content,
                message.timestamp,
                message.interrupted,
            )
        except Exception:
            first_failure.signal("persistence_failed")

    async def sanitize_tts_error(_tts: FrameProcessor, error: ErrorFrame) -> None:
        if controller.is_active():
            return
        _sanitize_inline_error(error, "tts_failed")
        first_failure.signal("tts_failed")
        try:
            await controller.abort("tts_failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    user_aggregator.add_event_handler("on_user_turn_stopped", record_user_turn)
    assistant_aggregator.add_event_handler(
        "on_assistant_turn_stopped", record_assistant_turn
    )
    services.tts.add_event_handler("on_error", sanitize_tts_error)

    barrier = DisclosureOutputBarrier(controller=controller)
    return ObservedPipeline(
        [
            transport.input(),
            input_gate,
            inference_boundary,
            services.stt,
            user_aggregator,
            services.llm,
            services.tts,
            barrier,
            transport.output(),
            assistant_aggregator,
        ],
        first_failure=first_failure,
    )


@dataclass(frozen=True, slots=True)
class CallRuntime:
    pipeline: ObservedPipeline
    worker: PipelineWorker
    runner: WorkerRunner
    task_manager: ObservedTaskManager


async def build_runtime(
    *,
    pipeline: ObservedPipeline,
    first_failure: FirstFailure,
    greeting: str,
    mark_name: str,
    idle_timeout_seconds: float,
) -> CallRuntime:
    """Create and register one native worker/runner pair owned by the call."""

    if (
        not isinstance(greeting, str)
        or not greeting
        or not isinstance(mark_name, str)
        or not mark_name
        or not isinstance(idle_timeout_seconds, int | float)
        or isinstance(idle_timeout_seconds, bool)
        or not math.isfinite(idle_timeout_seconds)
        or idle_timeout_seconds <= 0
    ):
        raise ValueError("call_runtime_config_invalid")
    task_manager = ObservedTaskManager(
        first_failure=first_failure,
        loop=asyncio.get_running_loop(),
    )
    worker = PipelineWorker(
        pipeline,
        params=pipeline_params(),
        enable_rtvi=False,
        enable_tracing=False,
        idle_timeout_secs=float(idle_timeout_seconds),
    )

    async def queue_disclosure(_worker: PipelineWorker, _frame: StartFrame) -> None:
        try:
            await worker.queue_frames(
                [
                    TTSSpeakFrame(greeting, append_to_context=False),
                    TelnyxMarkFrame(mark_name),
                ]
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            first_failure.signal("disclosure_failed")

    worker.add_event_handler("on_pipeline_started", queue_disclosure)

    runner = WorkerRunner(
        handle_sigint=False,
        handle_sigterm=False,
        task_manager=task_manager,
    )
    await runner.add_workers(worker)
    return CallRuntime(
        pipeline=pipeline,
        worker=worker,
        runner=runner,
        task_manager=task_manager,
    )


def pipeline_params() -> PipelineParams:
    """Return the frozen qualified Task 8 audio and metrics contract."""

    return PipelineParams(
        audio_in_sample_rate=8000,
        audio_out_sample_rate=8000,
        enable_metrics=True,
        enable_usage_metrics=True,
    )


__all__ = [
    "FirstFailure",
    "DisclosureOutputBarrier",
    "CallRuntime",
    "GateController",
    "InferenceErrorBoundary",
    "ObservedPipeline",
    "ObservedTaskManager",
    "PipelineServices",
    "PipelineTransport",
    "PipelineTurnRecorder",
    "build_pipeline",
    "build_input_gate",
    "build_runtime",
    "pipeline_params",
]
