"""RED consumers for optional bounded capture; no policy, provider or writer cutover."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio import audio_buffer_processor as native_audio
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.base_object import BaseObject

from projetv0_voice.audio_capture import BoundedAudioBufferTap


@pytest.fixture
def timing(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(native_audio, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


class ForwardedFrames(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)


class HeldObservation(BaseObserver):
    def __init__(self):
        super().__init__()
        self.hold = False
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def on_process_frame(self, data):
        if self.hold and isinstance(data.frame, InputAudioRawFrame):
            self.entered.set()
            await self.release.wait()


@asynccontextmanager
async def native_tap(offer_chunk, timing, observer=None):
    manager = TaskManager(loop=asyncio.get_running_loop())
    clock = SystemClock()
    clock.start()
    worker = PipelineWorker(
        Pipeline([]), clock=clock, task_manager=manager,
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
        enable_rtvi=False, enable_turn_tracking=False,
    )
    tap = BoundedAudioBufferTap(offer_chunk=offer_chunk, monotonic=lambda: timing.now)
    forwarded = ForwardedFrames()
    tap.link(forwarded)
    await forwarded.setup(FrameProcessorSetup(
        clock=clock, task_manager=manager, pipeline_worker=worker, observer=None,
    ))
    await tap.setup(FrameProcessorSetup(
        clock=clock, task_manager=manager, pipeline_worker=worker, observer=observer,
    ))
    await tap.queue_frame(StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=8000))
    try:
        yield tap, forwarded
    finally:
        await asyncio.wait_for(tap.quiesce(), 1)
        await tap.cleanup()
        await FrameProcessor.cleanup(forwarded)
        await worker.cleanup()
        await asyncio.sleep(0)
        assert not manager.current_tasks()


async def fill_event_without_yielding(tap, clock):
    for _ in range(9):
        clock.now += 0.1
        await tap.queue_frame(InputAudioRawFrame(
            audio=b"\x01\x00" * 800, sample_rate=8000, num_channels=1
        ))


@pytest.mark.asyncio
async def test_deferred_native_event_permit_bounds_burst_and_forwards_ordinary_control(timing):
    offered = []

    def offer(pcm, rate, channels):
        offered.append((len(pcm), rate, channels))
        return True

    async with native_tap(offer, timing) as (tap, forwarded):
        await tap.start_capture()
        baseline = set(asyncio.all_tasks())
        await fill_event_without_yielding(tap, timing)
        assert offered == []  # The real native event exists before its handler gets CPU.
        for _ in range(12):
            timing.now += 0.1
            await tap.queue_frame(InputAudioRawFrame(
                audio=b"\x02\x00" * 800, sample_rate=8000, num_channels=1
            ))
        control = TextFrame("fixture control remains live")
        await tap.queue_frame(control)
        pending = [task for task in set(asyncio.all_tasks()) - baseline if not task.done()]
        assert 1 <= len(pending) <= 2  # One native event and at most one owned join task.
        assert control in forwarded.frames and tap.state == "partial"
        assert not any(isinstance(frame, ErrorFrame) and frame.fatal for frame in forwarded.frames)
        assert await tap.quiesce()
        assert len(offered) == 1 and 0 < offered[0][0] <= 32000
        assert offered[0][1:] == (8000, 2)


@pytest.mark.asyncio
async def test_post_observation_guard_refuses_stale_track_padding_before_native_dispatch(
    timing, monkeypatch,
):
    entered, offered = [], []
    original = AudioBufferProcessor.process_frame

    async def observe_native(recorder, frame, direction):
        entered.append(frame.id)
        return await original(recorder, frame, direction)

    monkeypatch.setattr(AudioBufferProcessor, "process_frame", observe_native)
    observer = HeldObservation()
    async with native_tap(
        lambda pcm, rate, channels: offered.append(len(pcm)) or True, timing, observer
    ) as (tap, forwarded):
        await tap.start_capture()
        timing.now += 0.1
        await tap.queue_frame(InputAudioRawFrame(
            audio=b"\x01\x00" * 800, sample_rate=8000, num_channels=1
        ))
        frame = InputAudioRawFrame(audio=b"\x02\x00" * 800, sample_rate=8000, num_channels=1)
        observer.hold = True
        held = asyncio.create_task(tap.queue_frame(frame))
        try:
            await asyncio.wait_for(observer.entered.wait(), 1)
            timing.now += 2.1
            observer.release.set()
            await asyncio.wait_for(held, 1)
            assert frame.id not in entered and frame in forwarded.frames
            assert tap.state == "partial"
            assert await tap.quiesce()
            assert all(size <= 32000 for size in offered)
        finally:
            observer.release.set()
            await asyncio.wait_for(held, 1)


@pytest.mark.asyncio
async def test_oversize_format_and_muted_track_gap_stop_optional_capture_before_native_allocation(
    timing, monkeypatch,
):
    entered, offered = [], []
    original = AudioBufferProcessor.process_frame

    async def observe_native(recorder, frame, direction):
        entered.append(frame.id)
        return await original(recorder, frame, direction)

    monkeypatch.setattr(AudioBufferProcessor, "process_frame", observe_native)
    for bad in [
        InputAudioRawFrame(audio=b"x" * 1602, sample_rate=8000, num_channels=1),
        InputAudioRawFrame(audio=b"x" * 1600, sample_rate=16000, num_channels=1),
    ]:
        async with native_tap(
            lambda pcm, rate, channels: offered.append(len(pcm)) or True, timing
        ) as (tap, forwarded):
            await tap.start_capture()
            await tap.queue_frame(bad)
            assert bad.id not in entered and bad in forwarded.frames and tap.state == "partial"
            assert await tap.quiesce()
    async with native_tap(
        lambda pcm, rate, channels: offered.append(len(pcm)) or True, timing
    ) as (tap, forwarded):
        await tap.start_capture()
        timing.now += 0.1
        await tap.queue_frame(InputAudioRawFrame(
            audio=b"\x01\x00" * 800, sample_rate=8000, num_channels=1
        ))
        await tap.queue_frame(UserStartedSpeakingFrame())
        # The opposite track stays active: a global last-frame gap guard cannot catch this.
        for _ in range(15):
            timing.now += 0.1
            output = OutputAudioRawFrame(
                audio=b"\x02\x00" * 80, sample_rate=8000, num_channels=1
            )
            await tap.queue_frame(output)
            assert output.id in entered
        assert tap.state == "recording"
        resumed = InputAudioRawFrame(audio=b"\x03\x00" * 800, sample_rate=8000, num_channels=1)
        await tap.queue_frame(resumed)
        assert resumed.id not in entered and resumed in forwarded.frames and tap.state == "partial"
        assert await tap.quiesce()
        assert all(size <= 32000 for size in offered)


@pytest.mark.asyncio
async def test_unknown_native_event_join_remains_pending_until_proved_and_stop_cannot_revive(
    timing, monkeypatch,
):
    offered = []
    async with native_tap(
        lambda pcm, rate, channels: offered.append(len(pcm)) or True, timing
    ) as (tap, forwarded):
        await tap.start_capture()
        await fill_event_without_yielding(tap, timing)
        assert offered == []  # Actual queued native handler, not a fabricated pending flag.
        original_cleanup = BaseObject.cleanup

        async def unknown_join(owner):
            if isinstance(owner, AudioBufferProcessor):
                raise TimeoutError("fixture event join outcome unknown")
            await original_cleanup(owner)

        monkeypatch.setattr(BaseObject, "cleanup", unknown_join)
        try:
            assert await tap.stop_capture() is False
            assert tap.state == "partial" and tap.pending_join
            await asyncio.sleep(0)
            assert tap.pending_join
            end = EndFrame()
            await tap.queue_frame(end)
            assert end in forwarded.frames and tap.pending_join
            assert not any(
                isinstance(frame, ErrorFrame) and frame.fatal for frame in forwarded.frames
            )
        finally:
            monkeypatch.setattr(BaseObject, "cleanup", original_cleanup)
        assert await tap.quiesce()
        assert not tap.pending_join and len(offered) == 1 and offered[0] <= 32000
        await tap.start_capture()
        after_stop = InputAudioRawFrame(audio=b"\x04\x00" * 800, sample_rate=8000, num_channels=1)
        await tap.queue_frame(after_stop)
        assert after_stop in forwarded.frames
        assert await tap.quiesce()
        assert len(offered) == 1 and tap.state == "partial"


@pytest.mark.asyncio
async def test_failed_native_stop_must_be_retried_before_tail_join_can_complete(
    timing, monkeypatch
):
    offered, seen_recorders = [], []

    def offer(pcm, rate, channels):
        offered.append((len(pcm), rate, channels))
        return True

    async with native_tap(offer, timing) as (tap, _forwarded):
        await tap.start_capture()
        timing.now += 0.02
        await tap.queue_frame(InputAudioRawFrame(
            audio=b"\x01\x00" * 160, sample_rate=8000, num_channels=1
        ))
        assert offered == []  # Genuine native subthreshold tail has not emitted an event.
        original_stop = AudioBufferProcessor.stop_recording

        async def fail_before_stop(recorder):
            seen_recorders.append(recorder)
            raise TimeoutError("fixture native stop did not execute")

        monkeypatch.setattr(AudioBufferProcessor, "stop_recording", fail_before_stop)
        try:
            assert await tap.stop_capture() is False
            assert tap.state == "partial" and tap.pending_join
            assert len(seen_recorders) == 1 and seen_recorders[0].has_audio()
            assert offered == []
        finally:
            monkeypatch.setattr(AudioBufferProcessor, "stop_recording", original_stop)
        assert await tap.quiesce()
        assert offered == [(640, 8000, 2)]
        assert not seen_recorders[0].has_audio() and not tap.pending_join
