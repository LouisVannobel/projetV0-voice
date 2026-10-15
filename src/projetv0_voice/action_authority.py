"""Call-owned origin and exact transcript policy for the two Sparra actions."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pipecat.frames.frames import LLMContextFrame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext

Action = Literal["end_call", "request_human"]
_FAREWELL = re.compile(
    r"(?:(?:non ça va|c'est tout|c'est bon|d'accord|ok|très bien) )?"
    r"(?:merci(?: beaucoup)? )?au revoir"
)


def direct_command(text: str) -> Action | None:
    normalized = " ".join(text.casefold().replace("’", "'").split())
    normalized = normalized.rstrip(".!").rstrip()
    if _FAREWELL.fullmatch(" ".join(normalized.replace(",", " ").split())):
        return "end_call"
    if normalized in {"transférez-moi", "transférez moi", "transferez-moi", "transferez moi"}:
        return "request_human"
    return None


@dataclass(frozen=True, slots=True)
class CallerScope:
    epoch: int
    text: str
    context: LLMContext
    frame: LLMContextFrame
    origin: object


class ActionAuthority:
    """Synchronous native frame capture; response/task consumers never read latest text."""

    def __init__(self, *, is_active: Callable[[], bool]) -> None:
        self._is_active = is_active
        self.epoch = 0
        self._terminal = False
        self._started = False
        self._completed = False
        self._claimed = False
        self._message_count = 0
        self._frames: dict[int, CallerScope] = {}
        self._origin: object | None = None
        self._last_start_id: int | None = None

    def current(self, scope: CallerScope) -> bool:
        return not self._terminal and self.epoch == scope.epoch and self._is_active()

    def response_current(self, epoch: int) -> bool:
        return not self._terminal and self.epoch == epoch and self._is_active()

    def caller_started(self, context: LLMContext, frame: UserStartedSpeakingFrame) -> None:
        if frame.broadcast_sibling_id == self._last_start_id and self._last_start_id is not None:
            # Native broadcasts the same turn start upstream and downstream.
            return
        self._last_start_id = frame.id
        self.epoch += 1
        self._started = True
        self._completed = False
        self._claimed = False
        self._message_count = len(context.messages)
        self._frames.clear()

    def caller_completed(self) -> None:
        self._completed = self._started and not self._terminal and self._is_active()
        self._started = False

    def invalidate_response(self) -> None:
        self.epoch += 1
        self._frames.clear()

    def close(self) -> None:
        self._terminal = True
        self.epoch += 1
        self._frames.clear()

    def capture(self, frame: LLMContextFrame, *, context: LLMContext, origin: object) -> None:
        if (
            self._terminal or not self._is_active() or not self._completed
            or frame.speculation or frame.context is not context or len(self._frames) >= 8
            or len(context.messages) <= self._message_count
        ):
            return
        # The exact producer just appended this message. No backwards role search.
        message = context.messages[-1]
        self._message_count = len(context.messages)
        self._completed = False
        if not isinstance(message, dict) or message.get("role") != "user":
            return
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            return
        self._origin = origin
        scope = CallerScope(self.epoch, text, context, frame, origin)
        self._frames[frame.id] = scope
        frame.metadata["sparra.caller_scope"] = scope

    def scope_for(self, frame: LLMContextFrame, *, take: bool = False) -> CallerScope | None:
        scope = frame.metadata.get("sparra.caller_scope")
        if take:
            self._frames.pop(frame.id, None)
        return scope if (
            isinstance(scope, CallerScope) and scope.frame is frame
            and scope.context is frame.context and scope.origin is self._origin
        ) else None

    def claim(self, scope: CallerScope) -> bool:
        if self._claimed or not self.current(scope):
            return False
        self._claimed = True
        return True
