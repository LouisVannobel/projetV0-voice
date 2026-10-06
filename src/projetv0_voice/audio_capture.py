"""Call-owned optional capture over the pinned native AudioBufferProcessor."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Literal

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.base_object import BaseObject

CaptureState = Literal["off", "recording", "partial", "stopped"]
_MONO_BYTES_PER_SECOND = 16000
_MAX_MONO_BYTES = 16000
_NATIVE_THRESHOLD_BYTES = 14400
_MAX_FRAME_BYTES = 1600
_MAX_SAMPLES = 4_800_000


class BoundedAudioBufferTap(FrameProcessor):
    """Forward every frame and refuse optional capture before native expansion.

    Position/time counters describe only admitted frames and public native events.
    They never read or alter the recorder's private buffers or event/task fields.
    """

    def __init__(
        self,
        *,
        offer_chunk: Callable[[bytes, int, int], bool],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(enable_direct_mode=True)
        self._offer_chunk = offer_chunk
        self._monotonic = monotonic
        self._recorder = AudioBufferProcessor(
            sample_rate=8000, num_channels=2, buffer_size=_NATIVE_THRESHOLD_BYTES,
            enable_turn_audio=False, auto_start_recording=False, enable_direct_mode=True,
        )
        self._recorder.add_event_handler(
            "on_audio_data", self._receive_audio
        )
        self._state: CaptureState = "off"
        self._started = False
        self._closed = False
        self._stop_called = False
        self._pending_join = False
        self._permit = False
        self._completion: asyncio.Task[bool] | None = None
        self._user_position = self._bot_position = 0
        self._last_user: float | None = None
        self._last_bot: float | None = None
        self._user_speaking = self._bot_speaking = False
        self._total_samples = 0

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def pending_join(self) -> bool:
        return self._pending_join

    # Pipecat 1.7 FrameProcessor narrows BaseObject's TaskManager setup to this public shape.
    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        await super().setup(setup)
        await self._recorder.setup(FrameProcessorSetup(
            clock=setup.clock, task_manager=setup.task_manager,
            pipeline_worker=setup.pipeline_worker, observer=None,
        ))

    async def start_capture(self) -> None:
        if self._started or self._closed:
            return
        self._started = True
        try:
            await self._recorder.start_recording()  # type: ignore[no-untyped-call]
            self._state = "recording"
        except Exception:
            self._refuse_capture()

    def _refuse_capture(self) -> None:
        self._closed = True
        self._state = "partial"
        if self._completion is None or self._completion.done():
            self._completion = asyncio.create_task(
                self._stop_and_join(), name="sparra-audio-event-join"
            )

    async def _stop_and_join(self) -> bool:
        self._permit = True
        self._pending_join = True
        try:
            if not self._stop_called:
                await self._recorder.stop_recording()  # type: ignore[no-untyped-call]
                self._stop_called = True
            await BaseObject.cleanup(self._recorder)  # type: ignore[no-untyped-call]
        except Exception:
            self._closed = True
            self._state = "partial"
            return False
        self._pending_join = False
        self._permit = False
        if self._state != "partial":
            self._state = "stopped"
        return True

    async def _join_event(self) -> bool:
        self._pending_join = True
        try:
            await BaseObject.cleanup(self._recorder)  # type: ignore[no-untyped-call]
        except Exception:
            self._closed = True
            self._state = "partial"
            return False
        self._pending_join = False
        if self._closed:
            return await self._stop_and_join()
        self._permit = False
        return True

    async def stop_capture(self) -> bool:
        self._closed = True
        if self._state != "partial":
            self._state = "stopped"
        return await self.quiesce()

    async def quiesce(self) -> bool:
        self._closed = True
        task = self._completion
        if task is not None and not task.done() and not await asyncio.shield(task):
            return False
        self._completion = None
        return await self._stop_and_join()

    async def cleanup(self) -> None:
        if not await self.quiesce():
            raise RuntimeError("audio_event_join_pending")
        await self._recorder.cleanup()  # type: ignore[no-untyped-call]
        await super().cleanup()  # type: ignore[no-untyped-call]

    async def _receive_audio(
        self, _recorder: AudioBufferProcessor, pcm: bytes, rate: int, channels: int
    ) -> None:
        try:
            if (
                rate != 8000 or channels != 2 or not pcm or len(pcm) % 4
                or len(pcm) > 32000 or self._total_samples + len(pcm) // 4 > _MAX_SAMPLES
            ):
                self._closed = True
                self._state = "partial"
                return
            self._total_samples += len(pcm) // 4
            if self._offer_chunk(pcm, rate, channels) is not True:
                self._closed = True
                self._state = "partial"
        except Exception:
            # Native BaseObject logs callback exceptions, so none may escape here.
            self._closed = True
            self._state = "partial"

    def _project_audio(
        self, frame: InputAudioRawFrame | OutputAudioRawFrame, now: float
    ) -> tuple[int, int, float | None, float | None] | None:
        size = len(frame.audio)
        if (
            frame.sample_rate != 8000 or frame.num_channels != 1
            or not 0 < size <= _MAX_FRAME_BYTES or size % 2 or not math.isfinite(now)
        ):
            return None
        user, bot = self._user_position, self._bot_position
        user_time, bot_time = self._last_user, self._last_bot
        previous = user_time if isinstance(frame, InputAudioRawFrame) else bot_time
        elapsed = 0.0 if previous is None else now - previous
        if elapsed < 0 or elapsed > 2.0:
            return None
        gap = elapsed - size / _MONO_BYTES_PER_SECOND
        padding = int(gap * _MONO_BYTES_PER_SECOND) if gap > 0.2 else 0
        padding -= padding % 2
        if isinstance(frame, InputAudioRawFrame):
            user += padding
            if not self._bot_speaking:
                bot, bot_time = max(bot, user), now
            user, user_time = user + size, now
        else:
            bot += padding
            if not self._user_speaking:
                user, user_time = max(user, bot), now
            bot, bot_time = bot + size, now
        maximum = max(user, bot)
        if maximum > _MAX_MONO_BYTES or self._total_samples + maximum // 2 > _MAX_SAMPLES:
            return None
        return user, bot, user_time, bot_time

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Observation may await. Admission must use the clock only after it returns.
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._recorder.queue_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop_capture()
        elif not self._closed and self._state == "recording":
            if isinstance(frame, (InputAudioRawFrame, OutputAudioRawFrame)):
                if self._permit:
                    self._refuse_capture()
                else:
                    projected = self._project_audio(frame, self._monotonic())
                    if projected is None:
                        self._refuse_capture()
                    else:
                        self._permit = True
                        await self._recorder.queue_frame(
                            frame, direction
                        )
                        (self._user_position, self._bot_position,
                         self._last_user, self._last_bot) = projected
                        if max(projected[0], projected[1]) >= _NATIVE_THRESHOLD_BYTES:
                            self._user_position = self._bot_position = 0
                            self._last_user = self._last_bot = self._monotonic()
                            self._completion = asyncio.create_task(
                                self._join_event(), name="sparra-audio-event-join"
                            )
                        else:
                            self._permit = False
            elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                    BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
                    self._user_speaking = isinstance(frame, UserStartedSpeakingFrame)
                else:
                    self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
                await self._recorder.queue_frame(frame, direction)
        await self.push_frame(frame, direction)
