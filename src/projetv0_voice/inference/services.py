"""Native Pipecat inference service construction."""

from __future__ import annotations

import asyncio
import json
import re
import struct
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal, NoReturn, cast

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, NotGiven, OpenAIError
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
    LLMFullResponseEndFrame,
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
from pipecat.utils.errors import ErrorCategory
from pydantic import SecretStr

from projetv0_voice.action_authority import ActionAuthority, CallerScope
from projetv0_voice.inference.completion_strategy import (
    STT_COMPLETED_SEGMENT_KEY,
    STT_MAX_PENDING_SEGMENTS,
    STT_REQUEST_TIMEOUT_SECONDS,
    STT_TERMINAL_PARTIAL_KEY,
)
from projetv0_voice.models import MessageResultV1
from projetv0_voice.persistence.business_result import (
    ResultCompletionRejected,
    RetainedCall,
    infer_partial_result,
)
from projetv0_voice.qualified_profile import InferenceProfileV1

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ISO_639_1 = re.compile(r"^[a-z]{2}$")
_STT_TIMEOUT_SECONDS = STT_REQUEST_TIMEOUT_SECONDS
_STT_DRAIN_TIMEOUT_SECONDS = 8.0
# Pilot overload bounds; these are not provider throughput guarantees.
_STT_MAX_PENDING_SEGMENTS = STT_MAX_PENDING_SEGMENTS
_STT_MAX_BATCH_TEXT_BYTES = 32 * 1024
_SPARRA_STT_MAX_AUDIO_SECONDS = 600
_HTTP_TIMEOUT = httpx.Timeout(8.0, connect=2.0)
_FAREWELL_PADDING_CHARACTERS = 8


class _BoundedOpenAISTTService(OpenAISTTService):
    """Bound native HTTP segments and emit only a fully transcribed speech batch.

    Pipecat owns the audio buffer, segment queue, and serial transcription task.
    An older segment's final transcript cannot finalize resumed speech while
    its audio or queued transcription remains outstanding. Retain text only;
    the last native transcript supplies the completed batch's metadata.
    """

    def __init__(self, *, sparra_audio_limits: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sparra_audio_limits = sparra_audio_limits
        self._submitted_audio_seconds = Fraction(0)
        self._audio_limit_seconds = Fraction(_SPARRA_STT_MAX_AUDIO_SECONDS)
        if sparra_audio_limits:
            # Pipecat 1.12 BaseWhisperSTTService directly constructs this private
            # client without a retry hook. OpenAI's public copy API preserves
            # the same call-owned HTTP client while disabling hidden uploads.
            self._client = self._client.with_options(max_retries=0)
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
        if self._sparra_audio_limits:
            refusal = self._reserve_audio(audio)
            if refusal is not None:
                failure = self._fail_batch(refusal)
                if failure is not None:
                    yield failure
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

    def _reserve_audio(self, audio: bytes) -> str | None:
        if not isinstance(audio, bytes) or len(audio) < 44:
            return "openrouter_stt_audio_invalid"
        # Pipecat 1.12 pcm_to_wav emits this exact 44-byte PCM16 header.
        # Validate the complete upload, not wave's first data chunk: additional
        # chunks or bytes outside RIFF must not bypass the duration reservation.
        (
            riff, riff_size, wave_id, fmt, fmt_size, encoding, channels,
            rate, byte_rate, alignment, bits, data, data_size,
        ) = struct.unpack_from("<4sI4s4sIHHIIHH4sI", audio)
        if (
            riff != b"RIFF" or wave_id != b"WAVE" or fmt != b"fmt " or data != b"data"
            or fmt_size != 16 or encoding != 1 or channels <= 0 or rate <= 0 or bits != 16
            or alignment != 2 * channels or byte_rate != rate * alignment
            or riff_size != len(audio) - 8 or data_size != len(audio) - 44
            or data_size <= 0 or data_size % alignment != 0
        ):
            return "openrouter_stt_audio_invalid"
        # Header-only validation establishes all PCM bytes and bounds duration
        # without allocating or copying the potentially large audio payload.
        duration = Fraction(data_size // alignment, rate)
        # No await between check and reservation. Timeouts, cancellation and
        # native Start/End/Cancel/cleanup never refund or reset this counter.
        if self._submitted_audio_seconds + duration > self._audio_limit_seconds:
            return "openrouter_stt_audio_limit"
        self._submitted_audio_seconds += duration
        return None

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


class LLMOutputLimitError(OpenAIError):
    """Constant-safe refusal before an unreserved LLM request can be dispatched."""

    def __init__(self) -> None:
        super().__init__("llm_output_limit")


_FinalFinishReason = Literal["stop", "length", "content_filter", "tool_calls", "error", "unknown"]


@dataclass(frozen=True, slots=True)
class FinalCompletionDiagnostic:
    """Call-local completion facts without response text, identifiers or errors."""

    finish_reason: _FinalFinishReason
    content_nonempty: bool
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(slots=True)
class _FarewellPrefix:
    """Keep only the bounded goodbye prefix and its original frames."""

    frames: list[tuple[LLMTextFrame, FrameDirection]] = field(default_factory=list)
    text: str = ""
    enabled: bool = True

    def discard(self) -> None:
        self.frames.clear()
        self.text = ""
        self.enabled = False


class _TrustlessOpenRouterLLMService(OpenRouterLLMService):
    """Pin-aware OpenRouter client factory without ambient HTTP authority."""

    _settings: OpenRouterLLMSettings

    def __init__(self, *, enforce_output_limits: bool = False, **kwargs: Any) -> None:
        self._enforce_output_limits = enforce_output_limits
        self._ordinary_output_reserved = 0
        self._ordinary_closed = False
        self._final_result_attempted = False
        self._final_post_reserved = False
        self._cleanup_completed = False
        self._final_completion_diagnostic: FinalCompletionDiagnostic | None = None
        self._final_result_scope: ContextVar[bool] = ContextVar(
            "llm_final_result_scope", default=False,
        )
        self._schema_routing: ContextVar[bool] = ContextVar("result_schema_routing", default=False)
        self._action_authority: ActionAuthority | None = None
        self._response_scope: ContextVar[tuple[int, CallerScope | None] | None] = ContextVar(
            "caller_response_scope", default=None,
        )
        self._end_call_goodbye: str | None = None
        self._farewell_prefix: ContextVar[_FarewellPrefix | None] = ContextVar(
            "caller_farewell_prefix", default=None,
        )
        self._tool_scopes: dict[str, tuple[str, CallerScope]] = {}
        self._seen_action_ids: set[str] = set()
        if enforce_output_limits:
            kwargs["retry_on_timeout"] = False
        super().__init__(**kwargs)

    @property
    def final_completion_diagnostic(self) -> FinalCompletionDiagnostic | None:
        return self._final_completion_diagnostic

    async def _observe_final_completion(self, response: httpx.Response) -> None:
        """Read only the actual final JSON response, preserving native HTTP errors."""
        if not (
            self._enforce_output_limits and self._final_result_scope.get()
            and self._schema_routing.get() and response.is_success
        ):
            return
        await response.aread()
        diagnostic = FinalCompletionDiagnostic("unknown", False)
        try:
            body = response.json()
        except (ValueError, UnicodeError):
            self._final_completion_diagnostic = diagnostic
            return
        choices = body.get("choices") if type(body) is dict else None
        choice = choices[0] if type(choices) is list and choices else None
        if type(choice) is dict:
            raw_reason = choice.get("finish_reason")
            reason: _FinalFinishReason = "unknown"
            if type(raw_reason) is str and raw_reason in {
                "stop", "length", "content_filter", "tool_calls", "error",
            }:
                reason = cast(_FinalFinishReason, raw_reason)
            message = choice.get("message")
            content = message.get("content") if type(message) is dict else None
            usage = body.get("usage")
            completion = usage.get("completion_tokens") if type(usage) is dict else None
            details = usage.get("completion_tokens_details") if type(usage) is dict else None
            reasoning = details.get("reasoning_tokens") if type(details) is dict else None
            diagnostic = FinalCompletionDiagnostic(
                reason, type(content) is str and bool(content.strip()),
                completion if type(completion) is int and completion >= 0 else None,
                reasoning if type(reasoning) is int and reasoning >= 0 else None,
            )
        self._final_completion_diagnostic = diagnostic

    def _refuse_output(self) -> NoReturn:
        self._ordinary_closed = True
        raise LLMOutputLimitError()

    async def _reserve_final_request(self, request: httpx.Request) -> None:
        """Inspect buffered SDK-final JSON and debit without any suspension or refund."""
        if not self._enforce_output_limits:
            return
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            self._refuse_output()

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            if len(pairs) != len(dict(pairs)):
                raise ValueError("duplicate_llm_request_key")
            return dict(pairs)

        def finite(_value: str) -> NoReturn:
            raise ValueError("nonfinite_llm_request_json")

        try:
            body = json.loads(request.content, object_pairs_hook=unique, parse_constant=finite)
        except (ValueError, UnicodeError, httpx.RequestNotRead):
            self._refuse_output()
        if type(body) is not dict:
            self._refuse_output()
        fields = [key for key in ("max_tokens", "max_completion_tokens") if key in body]
        if len(fields) != 1 or ("n" in body and (type(body["n"]) is not int or body["n"] != 1)):
            self._refuse_output()
        ceiling = body[fields[0]]
        if type(ceiling) is not int or ceiling <= 0:
            self._refuse_output()
        if self._final_result_scope.get():
            response_format = body.get("response_format")
            if (
                not self._cleanup_completed or not self._final_result_attempted
                or self._final_post_reserved or ceiling != 2048 or body.get("stream") is not False
                or type(response_format) is not dict
                or response_format.get("type") != "json_schema"
                or type(response_format.get("json_schema")) is not dict
                or type(response_format["json_schema"].get("schema")) is not dict
            ):
                self._refuse_output()
            self._final_post_reserved = True
        else:
            if (
                self._ordinary_closed or ceiling > 256
                or self._ordinary_output_reserved + ceiling > 13_952
            ):
                self._refuse_output()
            self._ordinary_output_reserved += ceiling

    async def infer_final_result_once(
        self, retained: RetainedCall, provider_callback: str | None,
    ) -> MessageResultV1 | None:
        """Bind only the actual post-cleanup retained-call consumer to its final allowance."""
        if not self._enforce_output_limits:
            return await infer_partial_result(self, retained, provider_callback)
        if not self._cleanup_completed or self._final_result_attempted:
            self._refuse_output()
        self._final_result_attempted = True
        token = self._final_result_scope.set(True)
        try:
            return await infer_partial_result(self, retained, provider_callback)
        finally:
            self._final_result_scope.reset(token)

    async def push_error(
        self,
        error_msg: str,
        exception: Exception | None = None,
        fatal: bool = False,
        category: ErrorCategory | None = None,
        force_treat_as_permanent: bool = False,
    ) -> None:
        if type(exception) is LLMOutputLimitError or (
            exception is not None and type(exception.__cause__) is LLMOutputLimitError
        ):
            await super().push_error("llm_failed", exception=None, fatal=True)
            return
        await super().push_error(
            error_msg, exception=exception, fatal=fatal, category=category,
            force_treat_as_permanent=force_treat_as_permanent,
        )

    def bind_action_authority(
        self, authority: ActionAuthority, *, end_call_goodbye: str | None = None,
    ) -> None:
        self._action_authority = authority
        self._end_call_goodbye = end_call_goodbye

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
        prefix = _FarewellPrefix() if scope is not None and self._end_call_goodbye else None
        prefix_token = self._farewell_prefix.set(prefix)
        try:
            await super().process_frame(frame, direction)
        finally:
            if prefix is not None:
                prefix.discard()
            self._farewell_prefix.reset(prefix_token)
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
                prefix = self._farewell_prefix.get()
                if (
                    fc.function_name == "end_call" and prefix is not None
                    and self.has_function("end_call")
                ):
                    # Native streaming text precedes tool dispatch. Keep the
                    # owned TTSSpeakFrame as the only final-goodbye producer.
                    prefix.discard()
        prefix = self._farewell_prefix.get()
        if prefix is not None:
            await self._flush_farewell_prefix(prefix)
        # Binding happens before the native service creates background tool tasks.
        await super().run_function_calls(function_calls)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        response, authority = self._response_scope.get(), self._action_authority
        prefix = self._farewell_prefix.get()
        if (
            isinstance(frame, LLMTextFrame) and response is not None and authority is not None
            and not authority.response_current(response[0])
        ):
            if prefix is not None:
                prefix.discard()
            return
        if prefix is not None:
            if isinstance(frame, (CancelFrame, EndFrame, ErrorFrame, InterruptionFrame)):
                prefix.discard()
            elif (
                direction == FrameDirection.DOWNSTREAM
                and isinstance(frame, LLMTextFrame) and prefix.enabled and frame.text
            ):
                candidate = prefix.text + frame.text
                if (
                    self._end_call_goodbye is not None
                    and len(candidate) <= len(self._end_call_goodbye) + _FAREWELL_PADDING_CHARACTERS
                    and self._end_call_goodbye.startswith(candidate.strip(" \t\r\n"))
                ):
                    prefix.text = candidate
                    prefix.frames.append((frame, direction))
                    return
                await self._flush_farewell_prefix(prefix)
                # A caller may have interrupted while original frames flushed.
                await self.push_frame(frame, direction)
                return
            elif (
                direction == FrameDirection.DOWNSTREAM
                and isinstance(frame, LLMFullResponseEndFrame)
            ):
                await self._flush_farewell_prefix(prefix)
        await super().push_frame(frame, direction)

    async def _flush_farewell_prefix(self, prefix: _FarewellPrefix) -> None:
        frames = tuple(prefix.frames)
        prefix.discard()
        for original, direction in frames:
            await self.push_frame(original, direction)

    async def cleanup(self) -> None:
        if self._enforce_output_limits:
            self._ordinary_closed = True
        if self._action_authority is not None:
            self._action_authority.close()
        self._tool_scopes.clear()
        self._seen_action_ids.clear()
        await super().cleanup()  # type: ignore[no-untyped-call]
        self._cleanup_completed = True

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
        inspect_final = (
            self._enforce_output_limits and self._final_result_scope.get()
            and response_schema is not None
        )
        if inspect_final:
            self._final_completion_diagnostic = None
        try:
            result = await super().run_inference(
                context, max_tokens=max_tokens, system_instruction=system_instruction,
                response_schema=response_schema,
            )
            if inspect_final:
                diagnostic = self._final_completion_diagnostic
                if diagnostic is None:
                    raise ResultCompletionRejected("unknown")
                reason = diagnostic.finish_reason
                if reason != "stop":
                    raise ResultCompletionRejected(reason)
                if not diagnostic.content_nonempty:
                    raise ResultCompletionRejected("empty")
            return result
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
                event_hooks=(
                    {"request": [self._reserve_final_request],
                     "response": [self._observe_final_completion]}
                    if self._enforce_output_limits else None
                ),
                follow_redirects=not self._enforce_output_limits,
            ),
            default_headers=default_headers,
        )


def build_stt(
    profile: InferenceProfileV1,
    api_key: SecretStr,
    *,
    language: str,
    http_client: DefaultAsyncHttpxClient | None = None,
    sparra_audio_limits: bool = False,
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
        sparra_audio_limits=sparra_audio_limits,
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
        enforce_output_limits=sparra_conversation,
    )
