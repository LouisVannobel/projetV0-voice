"""Native Pipecat inference service construction."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from contextvars import ContextVar
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FatalErrorFrame,
    Frame,
    InputAudioRawFrame,
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
        super().__init__(**kwargs)

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
        if self._schema_routing.get():
            extra_body = dict(params.get("extra_body", {}))
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
) -> OpenRouterLLMService:
    """Build Pipecat's native OpenRouter LLM with qualified routing policy."""

    dumped = profile.model_dump(mode="json")
    policy = cast(dict[str, Any], dumped["llm_provider_policy"])
    return _TrustlessOpenRouterLLMService(
        api_key=api_key.get_secret_value(),
        base_url=_OPENROUTER_BASE_URL,
        settings=OpenRouterLLMService.Settings(
            model=profile.llm_model,
            extra={"extra_body": {"provider": policy}},
        ),
    )
