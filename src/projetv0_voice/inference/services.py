"""Native Pipecat inference service construction."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from contextvars import ContextVar
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, NotGiven
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FatalErrorFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService, OpenRouterLLMSettings
from pipecat.services.whisper.base_stt import language_to_whisper_language
from pipecat.transcriptions.language import Language
from pydantic import SecretStr

from projetv0_voice.action_authority import ActionAuthority, CallerScope
from projetv0_voice.inference.completion_strategy import (
    STT_COMPLETED_SEGMENT_KEY,
    STT_MAX_PENDING_SEGMENTS,
    STT_REQUEST_TIMEOUT_SECONDS,
    STT_TERMINAL_PARTIAL_KEY,
)
from projetv0_voice.qualified_profile import InferenceProfileV1

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ISO_639_1 = re.compile(r"^[a-z]{2}$")
_STT_TIMEOUT_SECONDS = STT_REQUEST_TIMEOUT_SECONDS
_STT_DRAIN_TIMEOUT_SECONDS = 8.0
# Pilot overload bounds; these are not provider throughput guarantees.
_STT_MAX_PENDING_SEGMENTS = STT_MAX_PENDING_SEGMENTS
_STT_MAX_BATCH_TEXT_BYTES = 32 * 1024
_HTTP_TIMEOUT = httpx.Timeout(8.0, connect=2.0)


class _BoundedOpenAISTTService(OpenAISTTService):
    """Bound native HTTP segments and emit only a fully transcribed speech batch.

    Pipecat owns the audio buffer, segment queue, and serial transcription task.
    An older segment's final transcript cannot finalize resumed speech while
    its audio or queued transcription remains outstanding. Retain text only;
    the last native transcript supplies the completed batch's metadata.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pending_segments = 0
        self._accepted_segment = 0
        self._completed_segment = 0
        self._batch_text = ""
        self._last_transcript: TranscriptionFrame | None = None
        self._caller_speaking = False
        self._closing = False
        self._closing_with_open_speech = False
        self._terminal = False

    def _discard_batch(self) -> None:
        self._pending_segments = 0
        self._batch_text = ""
        self._last_transcript = None

    def _fail_batch(self, code: str) -> FatalErrorFrame | None:
        if self._terminal:
            return None
        self._terminal = True
        self._discard_batch()
        return FatalErrorFrame(error=code, processor=self)

    def _take_completed_batch(self) -> TranscriptionFrame | None:
        if self._pending_segments or self._caller_speaking or self._terminal:
            return None
        frame = self._last_transcript
        if frame is not None:
            frame.text = self._batch_text
        self._batch_text = ""
        self._last_transcript = None
        if frame is not None:
            frame.metadata[STT_COMPLETED_SEGMENT_KEY] = self._completed_segment
            if self._closing_with_open_speech:
                frame.metadata[STT_TERMINAL_PARTIAL_KEY] = True
        return frame

    async def start(self, frame: StartFrame) -> None:
        # A worker owns one call. Direct SDK warmup precedes its VAD stream.
        self._accepted_segment = 0
        self._completed_segment = 0
        await super().start(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(
            frame, (InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
        ):
            if self._terminal or self._closing:
                return
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._caller_speaking = True
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._caller_speaking = False
                if self._pending_segments >= _STT_MAX_PENDING_SEGMENTS:
                    failure = self._fail_batch("openrouter_stt_segment_limit")
                    if failure is not None:
                        await self.push_frame(failure)
                    return
                # Count before native dispatch: its background task can run as
                # soon as the public process_frame call yields to the event loop.
                self._pending_segments += 1
                self._accepted_segment += 1
        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame]:
        if self._terminal:
            return
        # Preserve public direct run_stt usage as a single segment, too.
        if not self._pending_segments:
            self._pending_segments = 1
            self._accepted_segment += 1
        try:
            async with asyncio.timeout(_STT_TIMEOUT_SECONDS):
                async with aclosing(super().run_stt(audio)) as frames:
                    async for frame in frames:
                        if self._terminal:
                            return
                        if isinstance(frame, ErrorFrame):
                            failure = self._fail_batch("openrouter_stt_transport")
                            if failure is not None:
                                yield failure
                            return
                        if isinstance(frame, TranscriptionFrame):
                            text = (
                                f"{self._batch_text} {frame.text}".strip()
                                if frame.text
                                else self._batch_text
                            )
                            if len(text.encode("utf-8")) > _STT_MAX_BATCH_TEXT_BYTES:
                                failure = self._fail_batch("openrouter_stt_text_limit")
                                if failure is not None:
                                    yield failure
                                return
                            self._batch_text = text
                            self._last_transcript = frame
                        else:
                            yield frame
            if self._terminal:
                return
            self._pending_segments -= 1
            # Native FIFO means successful exhaustion advances this frontier.
            self._completed_segment = self._accepted_segment - self._pending_segments
            completed = self._take_completed_batch()
            if completed is not None:
                yield completed
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            failure = self._fail_batch("openrouter_stt_timeout")
            if failure is not None:
                yield failure
        except UnicodeError:
            failure = self._fail_batch("openrouter_stt_text_invalid")
            if failure is not None:
                yield failure

    async def stop(self, frame: EndFrame) -> None:
        self._closing = True
        self._closing_with_open_speech = self._caller_speaking
        self._caller_speaking = False
        try:
            async with asyncio.timeout(_STT_DRAIN_TIMEOUT_SECONDS):
                await super().stop(frame)
            completed = self._take_completed_batch()
            if completed is not None:
                await self.push_frame(completed)
        except TimeoutError:
            failure = self._fail_batch("openrouter_stt_drain_timeout")
            await super().cancel(CancelFrame())
            if failure is not None:
                await self.push_frame(failure)
        finally:
            self._terminal = True
            self._discard_batch()

    async def cancel(self, frame: CancelFrame) -> None:
        self._terminal = True
        self._discard_batch()
        await super().cancel(frame)

    async def cleanup(self) -> None:
        self._terminal = True
        self._discard_batch()
        try:
            await super().cancel(CancelFrame())
        finally:
            await super().cleanup()  # type: ignore[no-untyped-call]


class _TrustlessOpenRouterLLMService(OpenRouterLLMService):
    """Pin-aware OpenRouter client factory without ambient HTTP authority."""

    _settings: OpenRouterLLMSettings

    def __init__(self, **kwargs: Any) -> None:
        self._schema_routing: ContextVar[bool] = ContextVar("result_schema_routing", default=False)
        self._action_authority: ActionAuthority | None = None
        self._response_scope: ContextVar[tuple[int, CallerScope | None] | None] = ContextVar(
            "caller_response_scope", default=None,
        )
        self._tool_scopes: dict[str, tuple[str, CallerScope]] = {}
        self._seen_action_ids: set[str] = set()
        super().__init__(**kwargs)

    def bind_action_authority(self, authority: ActionAuthority) -> None:
        self._action_authority = authority

    def consume_action_scope(
        self, *, tool_call_id: str, function_name: str, context: object,
    ) -> CallerScope | None:
        binding = self._tool_scopes.pop(tool_call_id, None)
        authority = self._action_authority
        if binding is None or authority is None:
            return None
        name, scope = binding
        if name != function_name or scope.context is not context or not authority.claim(scope):
            return None
        return scope

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        authority = self._action_authority
        if authority is not None and isinstance(frame, (CancelFrame, EndFrame, ErrorFrame)):
            authority.close()
            self._tool_scopes.clear()
        elif isinstance(frame, InterruptionFrame):
            self._tool_scopes.clear()
            if authority is not None:
                authority.invalidate_response()
        if authority is None or not isinstance(frame, LLMContextFrame):
            await super().process_frame(frame, direction)
            return
        scope = authority.scope_for(frame, take=True)
        if scope is not None and not authority.current(scope):
            # A known revoked caller frame is not a no-scope recovery request.
            return
        token = self._response_scope.set((authority.epoch if scope is None else scope.epoch, scope))
        try:
            await super().process_frame(frame, direction)
        finally:
            self._response_scope.reset(token)

    async def run_function_calls(self, function_calls: Sequence[FunctionCallFromLLM]) -> None:
        authority, response = self._action_authority, self._response_scope.get()
        actions = [fc for fc in function_calls if fc.function_name in {"end_call", "request_human"}]
        ids = [fc.tool_call_id for fc in actions]
        valid = (
            authority is not None and response is not None and response[1] is not None
            and authority.current(response[1]) and len(actions) == 1
            and len(self._seen_action_ids) < 256 and len(self._tool_scopes) < 8
            and all(isinstance(fc.arguments, dict) and not fc.arguments for fc in actions)
            and all(tool_id and tool_id not in self._seen_action_ids for tool_id in ids)
        )
        for tool_id in ids:
            self._tool_scopes.pop(tool_id, None)
            if len(self._seen_action_ids) < 256:
                self._seen_action_ids.add(tool_id)
        if valid and response is not None and response[1] is not None:
            fc = actions[0]
            if fc.context is response[1].context:
                self._tool_scopes[fc.tool_call_id] = (fc.function_name, response[1])
        # Binding happens before the native service creates background tool tasks.
        await super().run_function_calls(function_calls)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        response, authority = self._response_scope.get(), self._action_authority
        if (
            isinstance(frame, LLMTextFrame) and response is not None and authority is not None
            and not authority.response_current(response[0])
        ):
            return
        await super().push_frame(frame, direction)

    async def cleanup(self) -> None:
        if self._action_authority is not None:
            self._action_authority.close()
        self._tool_scopes.clear()
        self._seen_action_ids.clear()
        await super().cleanup()  # type: ignore[no-untyped-call]

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str | None:
        if response_schema is not None and (
            not self.supports_response_schema
            or not self.model_supports_response_schema(self._settings.model or "")
        ):
            raise RuntimeError("result_response_schema_unsupported")
        token = self._schema_routing.set(response_schema is not None)
        try:
            return await super().run_inference(
                context, max_tokens=max_tokens, system_instruction=system_instruction,
                response_schema=response_schema,
            )
        finally:
            self._schema_routing.reset(token)

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams,
    ) -> dict[str, Any]:
        params = super().build_chat_completion_params(params_from_context)
        # Pipecat's one-shot override selects its token field by key presence.
        # Omit only the unused native sentinel so it overrides the active ceiling.
        if isinstance(params.get("max_tokens"), int) and isinstance(
            params.get("max_completion_tokens"), NotGiven,
        ):
            params.pop("max_completion_tokens")
        if self._schema_routing.get():
            extra_body = dict(params.get("extra_body", {}))
            # The structured summary retains its previous reasoning defaults.
            if extra_body.get("reasoning") == {"enabled": False}:
                extra_body.pop("reasoning")
            provider = dict(extra_body.get("provider", {}))
            provider["require_parameters"] = True
            extra_body["provider"] = provider
            params["extra_body"] = extra_body
        return params

    def create_client(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        **_kwargs: Any,
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            max_retries=0,
            timeout=_HTTP_TIMEOUT,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000,
                    keepalive_expiry=None,
                ),
                timeout=_HTTP_TIMEOUT,
                trust_env=False,
            ),
            default_headers=default_headers,
        )


def build_stt(
    profile: InferenceProfileV1,
    api_key: SecretStr,
    *,
    language: str,
    http_client: DefaultAsyncHttpxClient | None = None,
) -> OpenAISTTService:
    """Build Pipecat's native segmented STT service against OpenRouter."""

    try:
        manifest_language = Language(language)
    except ValueError:
        raise ValueError("unsupported_stt_language") from None
    service_language = language_to_whisper_language(manifest_language)
    if _ISO_639_1.fullmatch(service_language) is None:
        raise ValueError("unsupported_stt_language")
    resolved_language = Language(service_language)

    return _BoundedOpenAISTTService(
        api_key=api_key.get_secret_value(),
        base_url=_OPENROUTER_BASE_URL,
        settings=OpenAISTTService.Settings(
            model=profile.stt_model,
            language=resolved_language,
        ),
        # Empty segment metadata is needed when the last segment is empty;
        # its ordered completion receipt contributes no text to the user aggregator.
        push_empty_transcripts=True,
        http_client=http_client,
    )


def build_llm(
    profile: InferenceProfileV1,
    api_key: SecretStr,
    *,
    sparra_conversation: bool = False,
) -> OpenRouterLLMService:
    """Build Pipecat's native OpenRouter LLM with qualified routing policy."""

    dumped = profile.model_dump(mode="json")
    policy = cast(dict[str, Any], dumped["llm_provider_policy"])
    extra_body = {"provider": policy}
    if sparra_conversation:
        extra_body["reasoning"] = {"enabled": False}
    settings = OpenRouterLLMService.Settings(
        model=profile.llm_model,
        extra={"extra_body": extra_body},
    )
    if sparra_conversation:
        settings.max_tokens = 256
    return _TrustlessOpenRouterLLMService(
        api_key=api_key.get_secret_value(),
        base_url=_OPENROUTER_BASE_URL,
        settings=settings,
    )
