from __future__ import annotations

import pytest
from pipecat.frames.frames import CancelFrame, EndFrame, VADUserStartedSpeakingFrame

from projetv0_voice.inference.completion_strategy import CompletionAwareTurnStopStrategy
from tests.unit.test_inference_deadlines import _CompletePauseAnalyzer


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [VADUserStartedSpeakingFrame(), CancelFrame(), EndFrame()])
async def test_incomplete_or_terminal_native_strategy_cannot_grant_completed_turn(frame):
    completed = []
    strategy = CompletionAwareTurnStopStrategy(
        turn_analyzer=_CompletePauseAnalyzer(), on_completed_turn=lambda: completed.append(True),
    )
    await strategy.process_frame(frame)
    await strategy.trigger_user_turn_stopped()
    assert completed == []
