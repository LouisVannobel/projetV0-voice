"""Bounded, input-redacting Telnyx Call Control wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import telnyx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

ATTEMPT_DEADLINE_SECONDS = 0.500
CLOSE_DEADLINE_SECONDS = 1.0
MAX_CALL_CONTROL_ID_CHARS = 1_024
MAX_STREAM_URL_CHARS = 2_048
MAX_STREAM_AUTH_TOKEN_CHARS = 4_000

CallControlOutcome = Literal[
    "accepted",
    "rejected",
    "rate_limited",
    "retryable_not_sent",
    "outcome_unknown",
]


class CallControlError(RuntimeError):
    """A Call Control error carrying only a constant project code."""


class CallControlConfigurationError(CallControlError):
    """The owned SDK client could not be constructed safely."""


class CallControlInputError(CallControlError):
    """A local caller violated the strict action contract."""


class CallControlClosedError(CallControlError):
    """The owned client has started closing."""


class CallControlCloseError(CallControlError):
    """The owned SDK client did not close inside its fixed bound."""


class _RedactedFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class StreamingStartV1(_RedactedFrozenModel):
    stream_url: str = Field(min_length=1, max_length=MAX_STREAM_URL_CHARS, repr=False)
    stream_auth_token: str = Field(
        min_length=1,
        max_length=MAX_STREAM_AUTH_TOKEN_CHARS,
        repr=False,
    )

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise ValueError("stream_url_invalid") from None
        if (
            parsed.scheme != "wss"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "?" in value
            or "#" in value
            or "\\" in parsed.netloc
            or any(character.isspace() for character in value)
            or port is not None and not 1 <= port <= 65_535
        ):
            raise ValueError("stream_url_invalid")
        return value


class RecordingStartV1(_RedactedFrozenModel):
    play_beep: StrictBool = Field(repr=False)


@dataclass(frozen=True, slots=True)
class CallControlResult:
    outcome: CallControlOutcome


@dataclass(frozen=True, slots=True)
class _Observation:
    outcome: CallControlOutcome
    retry: bool
    ambiguous: bool


def _valid_api_key(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_STREAM_AUTH_TOKEN_CHARS


def _valid_call_control_id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_CALL_CONTROL_ID_CHARS


def _valid_command_id(value: object) -> bool:
    return isinstance(value, UUID) and value.version == 4


def _status_observation(status_code: int) -> _Observation:
    if status_code in {408, 500}:
        return _Observation("outcome_unknown", retry=True, ambiguous=True)
    if status_code == 429:
        return _Observation("rate_limited", retry=False, ambiguous=False)
    if 400 <= status_code < 500:
        return _Observation("rejected", retry=False, ambiguous=False)
    return _Observation("outcome_unknown", retry=False, ambiguous=True)


def _connection_observation(error: telnyx.APIConnectionError) -> _Observation:
    cause = error.__cause__
    if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
        return _Observation("retryable_not_sent", retry=True, ambiguous=False)
    if isinstance(cause, httpx.PoolTimeout):
        return _Observation("retryable_not_sent", retry=False, ambiguous=False)
    return _Observation("outcome_unknown", retry=True, ambiguous=True)


class CallControlClient:
    """Own exactly one long-lived Telnyx SDK client and one immediate retry."""

    __slots__ = ("_client", "_close_lock", "_closed", "_closing", "_timeout")

    def __init__(self, *, api_key: str) -> None:
        if not _valid_api_key(api_key):
            raise CallControlConfigurationError("call_control_config_invalid")
        sdk_client: telnyx.AsyncTelnyx | None = None
        construction_failed = False
        try:
            sdk_client = telnyx.AsyncTelnyx(api_key=api_key, max_retries=0)
        except Exception:
            construction_failed = True
        if construction_failed or sdk_client is None:
            raise CallControlConfigurationError("call_control_config_invalid")
        self._client = sdk_client
        self._timeout = httpx.Timeout(
            ATTEMPT_DEADLINE_SECONDS,
            connect=ATTEMPT_DEADLINE_SECONDS,
            read=ATTEMPT_DEADLINE_SECONDS,
            write=ATTEMPT_DEADLINE_SECONDS,
            pool=ATTEMPT_DEADLINE_SECONDS,
        )
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    def __repr__(self) -> str:
        return "CallControlClient()"

    def _validated_ids(self, call_control_id: object, command_id: object) -> tuple[str, str]:
        if self._closing or self._closed:
            raise CallControlClosedError("call_control_client_closed")
        if not _valid_call_control_id(call_control_id) or not _valid_command_id(command_id):
            raise CallControlInputError("call_control_input_invalid")
        return cast(str, call_control_id), str(command_id)

    async def answer(self, call_control_id: str, *, command_id: UUID) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)

        async def action() -> object:
            return await self._client.calls.actions.answer(
                control_id,
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def start_streaming(
        self,
        call_control_id: str,
        request: StreamingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if not isinstance(request, StreamingStartV1):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            return await self._client.calls.actions.start_streaming(
                control_id,
                stream_track="inbound_track",
                stream_bidirectional_mode="rtp",
                stream_bidirectional_codec="PCMU",
                stream_bidirectional_sampling_rate=8000,
                stream_bidirectional_target_legs="self",
                stream_codec="PCMU",
                stream_url=request.stream_url,
                stream_auth_token=request.stream_auth_token,
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def start_recording(
        self,
        call_control_id: str,
        request: RecordingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if not isinstance(request, RecordingStartV1):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            return await self._client.calls.actions.start_recording(
                control_id,
                channels="dual",
                format="wav",
                max_length=0,
                recording_track="both",
                timeout_secs=0,
                transcription=False,
                play_beep=request.play_beep,
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def hangup(self, call_control_id: str, *, command_id: UUID) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)

        async def action() -> object:
            return await self._client.calls.actions.hangup(
                control_id,
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def _attempt(self, action: Callable[[], Awaitable[object]]) -> _Observation:
        loop = asyncio.get_running_loop()
        response: object | None = None
        try:
            async with asyncio.timeout_at(loop.time() + ATTEMPT_DEADLINE_SECONDS):
                response = await action()
        except TimeoutError:
            return _Observation("outcome_unknown", retry=True, ambiguous=True)
        except telnyx.APIStatusError as error:
            return _status_observation(error.status_code)
        except telnyx.APIResponseValidationError:
            return _Observation("outcome_unknown", retry=False, ambiguous=True)
        except telnyx.APIConnectionError as error:
            return _connection_observation(error)
        except Exception:
            return _Observation("outcome_unknown", retry=False, ambiguous=True)

        try:
            data = getattr(response, "data", None)
            result = getattr(data, "result", None)
        except Exception:
            return _Observation("outcome_unknown", retry=False, ambiguous=True)
        if result == "ok":
            return _Observation("accepted", retry=False, ambiguous=False)
        return _Observation("outcome_unknown", retry=False, ambiguous=True)

    async def _execute(self, action: Callable[[], Awaitable[object]]) -> CallControlResult:
        first = await self._attempt(action)
        if not first.retry:
            return CallControlResult(first.outcome)
        second = await self._attempt(action)
        if second.outcome == "accepted":
            return CallControlResult("accepted")
        if first.ambiguous:
            return CallControlResult("outcome_unknown")
        return CallControlResult(second.outcome)

    async def aclose(self) -> None:
        close_failed = False
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            try:
                async with asyncio.timeout(CLOSE_DEADLINE_SECONDS):
                    await self._client.close()
            except asyncio.CancelledError:
                self._closed = True
                raise
            except Exception:
                close_failed = True
            self._closed = True
        if close_failed:
            raise CallControlCloseError("call_control_close_failed")
