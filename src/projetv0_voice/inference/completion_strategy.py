"""HTTP-segment coverage around Pipecat's native endpoint strategy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

STT_COMPLETED_SEGMENT_KEY = "projetv0.stt.completed_segment_seq"
STT_TERMINAL_PARTIAL_KEY = "projetv0.stt.terminal_partial"
STT_MAX_PENDING_SEGMENTS = 4
STT_REQUEST_TIMEOUT_SECONDS = 8.0
STT_DELIVERY_ALLOWANCE_SECONDS = 4.0
# This coordinates the independent controller watchdog, not normal endpoint latency.
STT_USER_TURN_WATCHDOG_SECONDS = (
    STT_MAX_PENDING_SEGMENTS * STT_REQUEST_TIMEOUT_SECONDS + STT_DELIVERY_ALLOWANCE_SECONDS
)


class CompletionAwareTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Require consumed HTTP completion to cover the latest local VAD segment.

    The producer and consumer count the same single ordered VAD stop stream.
    Distinct upstream/local VAD frame objects need not arrive atomically.
    A receipt is consumed here, after the user aggregator consumed its text.
    Native endpoint decisions and timers still decide when completion is useful.
    """

    def __init__(
        self, *, on_incomplete_turn_stop: Callable[[], None] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._latest_segment = 0
        self._delivered_segment = 0
        self._speech_open = False
        self._terminal = False
        self._last_nonempty_text = ""
        self._on_incomplete_turn_stop = on_incomplete_turn_stop

    async def handle_user_turn_started(self) -> None:
        self._last_nonempty_text = ""
        await super().handle_user_turn_started()  # type: ignore[no-untyped-call]

    async def handle_user_turn_stopped(self) -> None:
        # The independent controller can force a stop without asking this
        # strategy. This public hook runs before the aggregator publishes it.
        if not self._terminal and (
            self._speech_open or self._delivered_segment < self._latest_segment
        ):
            self._terminal = True
            if self._on_incomplete_turn_stop is not None:
                self._on_incomplete_turn_stop()
        self._last_nonempty_text = ""
        await super().handle_user_turn_stopped()  # type: ignore[no-untyped-call]

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, (CancelFrame, EndFrame)):
            self._terminal = True
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_open = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_open = False
            self._latest_segment += 1
        elif isinstance(frame, TranscriptionFrame):
            completed = frame.metadata.get(STT_COMPLETED_SEGMENT_KEY)
            if isinstance(completed, int) and not isinstance(completed, bool) and completed >= 0:
                self._delivered_segment = max(self._delivered_segment, completed)
            if frame.metadata.get(STT_TERMINAL_PARTIAL_KEY) is True:
                self._terminal = True
            if frame.text.strip():
                self._last_nonempty_text = frame.text
            elif self._last_nonempty_text:
                # Only the stop strategy sees this clone. The original empty
                # receipt contributes no user text and cannot append A again.
                frame = replace(frame, text=self._last_nonempty_text)
        return await super().process_frame(frame)

    async def trigger_user_turn_stopped(
        self, *, enable_user_speaking_frames: bool | None = None
    ) -> None:
        # Native safety-net requests must pass coverage before firing inference,
        # not merely before finalizing an already-started inference.
        if (
            self._terminal
            or self._speech_open
            or self._delivered_segment < self._latest_segment
        ):
            return
        await super().trigger_user_turn_stopped(
            enable_user_speaking_frames=enable_user_speaking_frames
        )
