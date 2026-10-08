"""Native recorder characterization, not qualification of product capture or ON."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio import audio_buffer_processor as native_audio
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.base_object import BaseObject


@pytest.fixture
def recorder_time(monkeypatch):
    state = SimpleNamespace(now=10.0, checkpoint=None, reads=[])

    def monotonic():
        state.reads.append((state.now, state.checkpoint.is_set() if state.checkpoint else False))
        return state.now

    # Replace this module's binding only; asyncio deadlines retain the real clock.
    monkeypatch.setattr(native_audio, "time", SimpleNamespace(monotonic=monotonic))
    return state


@asynccontextmanager
async def native_recorder():
    manager = TaskManager(loop=asyncio.get_running_loop())
    clock = SystemClock()
    clock.start()
    worker = PipelineWorker(
        Pipeline([]),
        task_manager=manager,
        clock=clock,
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
        enable_rtvi=False,
        enable_turn_tracking=False,
    )
    recorder = AudioBufferProcessor(
        sample_rate=8000, num_channels=2, buffer_size=16000,
        enable_turn_audio=False, auto_start_recording=False, enable_direct_mode=True,
    )
    await recorder.setup(FrameProcessorSetup(
        clock=clock, task_manager=manager, pipeline_worker=worker, observer=None,
    ))
    await recorder.queue_frame(StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=8000))
    try:
        yield recorder, manager, clock, worker
    finally:
        await recorder.stop_recording()
        await FrameProcessor.cleanup(recorder)
        await worker.cleanup()
        await asyncio.sleep(0)
        assert not manager.current_tasks()


async def emit_one_native_event(recorder, timing):
    for _ in range(9):
        timing.now += 0.1
        await recorder.queue_frame(InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1))
        await recorder.queue_frame(OutputAudioRawFrame(b"\x02\x00" * 800, 8000, 1))
    timing.now += 0.1
    await recorder.queue_frame(InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1))


@pytest.mark.asyncio
async def test_native_event_join_waits_for_held_callback_and_remains_reusable(recorder_time):
    entered, release = asyncio.Event(), asyncio.Event()
    callback_tasks, delivered = [], []

    async def receive(_recorder, audio, sample_rate, channels):
        callback_tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        delivered.append((len(audio), sample_rate, channels))

    async with native_recorder() as (recorder, manager, _clock, _worker):
        recorder.add_event_handler("on_audio_data", receive)
        await recorder.queue_frame(InputAudioRawFrame(b"\x01\x00" * 800, 8000, 1))
        assert not recorder.has_audio() and not delivered  # auto_start_recording=False
        await recorder.start_recording()
        join = None
        try:
            await emit_one_native_event(recorder, recorder_time)
            await asyncio.wait_for(entered.wait(), 1)
            assert len(callback_tasks) == 1 and not callback_tasks[0].done()
            assert callback_tasks[0] not in manager.current_tasks()
            await recorder.stop_recording()
            assert not callback_tasks[0].done() and delivered == []
            join = asyncio.create_task(BaseObject.cleanup(recorder))
            await asyncio.sleep(0)
            assert not join.done()
            release.set()
            await asyncio.wait_for(join, 1)
            assert delivered == [(32000, 8000, 2)]
            assert recorder.task_manager is manager
            entered.clear()
            release = asyncio.Event()
            await recorder.start_recording()
            await emit_one_native_event(recorder, recorder_time)
            await asyncio.wait_for(entered.wait(), 1)
            assert len(callback_tasks) == 2
            assert callback_tasks[0] is not callback_tasks[1] and callback_tasks[0].done()
            assert sum(not task.done() for task in callback_tasks) == 1
            release.set()
            await asyncio.wait_for(BaseObject.cleanup(recorder), 1)
            assert delivered == [(32000, 8000, 2), (32000, 8000, 2)]
        finally:
            release.set()
            if join is not None:
                await asyncio.wait_for(join, 1)


@pytest.mark.asyncio
async def test_native_16000_post_append_threshold_can_emit_oversized_stereo(recorder_time):
    delivered = []

    async def receive(_recorder, audio, sample_rate, channels):
        delivered.append((len(audio), sample_rate, channels))

    async with native_recorder() as (recorder, _manager, _clock, _worker):
        recorder.add_event_handler("on_audio_data", receive)
        await recorder.start_recording()
        # Every mono frame is <=100ms and aligned; none is an oversized input.
        for size in [1600] * 9 + [1400, 1600]:
            recorder_time.now += size / 16000
            await recorder.queue_frame(InputAudioRawFrame(b"\x01\x00" * (size // 2), 8000, 1))
        await asyncio.wait_for(BaseObject.cleanup(recorder), 1)
        assert delivered == [(34800, 8000, 2)]
        assert delivered[0][0] > 32000  # Characterizes the unsafe native threshold.


class HeldOuterObserver(BaseObserver):
    def __init__(self):
        super().__init__()
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls = 0

    async def on_process_frame(self, data):
        if isinstance(data.frame, InputAudioRawFrame):
            self.calls += 1
            self.entered.set()
            await self.release.wait()


@pytest.mark.asyncio
async def test_outer_observer_hold_precedes_fresh_inner_clock_read_without_inner_suspension(
    recorder_time,
):
    observer = HeldOuterObserver()
    outer = FrameProcessor(enable_direct_mode=True)
    delivered = []

    async def receive(_recorder, audio, _sample_rate, _channels):
        delivered.append(len(audio))

    async with native_recorder() as (recorder, manager, clock, worker):
        await outer.setup(FrameProcessorSetup(
            clock=clock, task_manager=manager, pipeline_worker=worker, observer=observer,
        ))
        await outer.queue_frame(StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=8000))
        recorder.add_event_handler("on_audio_data", receive)
        await recorder.start_recording()
        frame = InputAudioRawFrame(b"\x01\x00" * 160, 8000, 1)
        recorder_time.now += 0.02
        await recorder.queue_frame(frame)
        early_check = recorder_time.now
        held = asyncio.create_task(outer.queue_frame(frame))
        try:
            await asyncio.wait_for(observer.entered.wait(), 1)
            recorder_time.now += 2.0
            assert not held.done()
            observer.release.set()
            await asyncio.wait_for(held, 1)
            fresh_check = recorder_time.now
            assert fresh_check - early_check == 2.0
            checkpoint = asyncio.Event()

            async def next_loop_turn():
                checkpoint.set()

            recorder_time.reads.clear()
            recorder_time.checkpoint = checkpoint
            tick = asyncio.create_task(next_loop_turn())
            await recorder.queue_frame(frame)
            # Actual native same-rate resampling/clock padding ran before another task.
            assert recorder_time.reads and all(
                value == fresh_check and not ran for value, ran in recorder_time.reads
            )
            await asyncio.wait_for(tick, 1)
            assert observer.calls == 1
            await asyncio.wait_for(BaseObject.cleanup(recorder), 1)
            assert delivered and delivered[0] > 32000
        finally:
            observer.release.set()
            await asyncio.wait_for(held, 1)
            await FrameProcessor.cleanup(outer)
