"""OpenRouter HTTP text-to-speech integration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any, cast

import httpx
from pipecat.frames.frames import (
    FatalErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import assert_given
from pydantic import SecretStr

from projetv0_voice.qualified_profile import InferenceProfileV1

_TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
# OpenRouter PCM and Pipecat share a fixed signed little-endian 16-bit invariant.

_HTTP_STATUS_ERROR = "openrouter_tts_http_status"
_CONTENT_TYPE_ERROR = "openrouter_tts_content_type"
_TRANSPORT_ERROR = "openrouter_tts_transport"
_STREAM_ERROR = "openrouter_tts_stream"
_EMPTY_AUDIO_ERROR = "openrouter_tts_empty_audio"
_TIMEOUT_ERROR = "openrouter_tts_timeout"
_TTS_TIMEOUT_SECONDS = 15.0


class OpenRouterTTSService(TTSService):
    """Thin OpenRouter PCM byte-stream adapter over Pipecat's native TTS lifecycle."""

    def __init__(
        self,
        *,
        profile: InferenceProfileV1,
        api_key: SecretStr,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=profile.tts_pcm_sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(
                model=profile.tts_model,
                voice=profile.tts_voice,
                language=None,
            ),
            **kwargs,
        )
        dumped = profile.model_dump(mode="json")
        self._provider_options = cast(
            dict[str, dict[str, Any]], dumped["tts_provider_options"]
        )
        self._input_sample_rate = profile.tts_pcm_sample_rate
        self._speed = profile.tts_speed
        self._api_key = api_key
        self._client = http_client or httpx.AsyncClient(trust_env=False)
        self._owns_client = http_client is None
        self._failed_contexts: dict[str, str] = {}
        self._end_call_context_binder: Callable[[str], None] | None = None

    def bind_end_call_context(self, binder: Callable[[str], None]) -> None:
        """Bind final playback through the native context-creation hook."""
        self._end_call_context_binder = binder

    async def on_turn_context_created(self, context_id: str) -> None:
        await super().on_turn_context_created(context_id)
        if self._end_call_context_binder is not None:
            self._end_call_context_binder(context_id)

    def can_generate_metrics(self) -> bool:
        """Return true because the adapter participates in native TTS metrics."""

        return True

    def _request_body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": cast(str, assert_given(self._settings.model)),
            "input": text,
            "voice": cast(str, assert_given(self._settings.voice)),
            "response_format": "pcm",
        }
        if self._provider_options:
            body["provider"] = {"options": self._provider_options}
        if self._speed is not None:
            body["speed"] = self._speed
        return body

    @staticmethod
    def _has_pcm_content_type(response: httpx.Response) -> bool:
        values = response.headers.get_list("content-type", split_commas=False)
        if len(values) != 1:
            return False
        value = values[0]
        if not value or "," in value or "\r" in value or "\n" in value:
            return False
        base_media_type = value.partition(";")[0].strip()
        return base_media_type.casefold() == "audio/pcm"

    def _neutralize_failed_context(self, context_id: str) -> None:
        tts_context = self._tts_contexts.get(context_id)
        if tts_context is not None:
            tts_context.append_to_context = False
            tts_context.push_assistant_aggregation = False

    async def _first_failure(
        self,
        context_id: str,
        code: str,
        *,
        audio_emitted: bool,
    ) -> FatalErrorFrame | None:
        if context_id in self._failed_contexts:
            self._neutralize_failed_context(context_id)
            return None
        self._failed_contexts[context_id] = code
        self._neutralize_failed_context(context_id)
        if not audio_emitted:
            try:
                await self.stop_ttfb_metrics()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Observability must not suppress an already-classified fatal failure.
                pass
        return FatalErrorFrame(error=code)

    @traced_tts
    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None]:
        """Stream raw PCM without buffering provider response bodies."""

        if context_id in self._failed_contexts:
            self._neutralize_failed_context(context_id)
            return

        audio_emitted = False
        try:
            async with asyncio.timeout(_TTS_TIMEOUT_SECONDS):
                async with self._client.stream(
                    "POST",
                    _TTS_ENDPOINT,
                    headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                    json=self._request_body(text),
                ) as response:
                    if response.status_code != 200:
                        failure = await self._first_failure(
                            context_id,
                            _HTTP_STATUS_ERROR,
                            audio_emitted=False,
                        )
                        if failure is not None:
                            yield failure
                        return
                    if not self._has_pcm_content_type(response):
                        failure = await self._first_failure(
                            context_id,
                            _CONTENT_TYPE_ERROR,
                            audio_emitted=False,
                        )
                        if failure is not None:
                            yield failure
                        return

                    await self.start_tts_usage_metrics(text)
                    try:
                        frames = self._stream_audio_frames_from_iterator(
                            response.aiter_bytes(),
                            in_sample_rate=self._input_sample_rate,
                            context_id=context_id,
                        )
                        async for frame in frames:
                            if isinstance(frame, TTSAudioRawFrame):
                                # Preserve the context on native padded leftovers too.
                                if frame.context_id is None:
                                    frame.context_id = context_id
                                audio_emitted = True
                            yield frame
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        failure = await self._first_failure(
                            context_id,
                            _STREAM_ERROR,
                            audio_emitted=audio_emitted,
                        )
                        if failure is not None:
                            yield failure
                        return

                    if not audio_emitted:
                        failure = await self._first_failure(
                            context_id,
                            _EMPTY_AUDIO_ERROR,
                            audio_emitted=False,
                        )
                        if failure is not None:
                            yield failure
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            failure = await self._first_failure(
                context_id,
                _TIMEOUT_ERROR,
                audio_emitted=audio_emitted,
            )
            if failure is not None:
                yield failure
        except Exception:
            failure = await self._first_failure(
                context_id,
                _TRANSPORT_ERROR,
                audio_emitted=audio_emitted,
            )
            if failure is not None:
                yield failure

    async def append_to_audio_context(
        self,
        context_id: str | None,
        frame: Any,
    ) -> None:
        """Keep failed text observable while preventing assistant-context commitment."""

        if context_id is not None and context_id in self._failed_contexts:
            self._neutralize_failed_context(context_id)
            if isinstance(frame, TTSTextFrame):
                frame.append_to_context = False
        await super().append_to_audio_context(context_id, frame)

    async def on_audio_context_completed(self, context_id: str) -> None:
        self._failed_contexts.pop(context_id, None)
        await super().on_audio_context_completed(context_id)

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        self._failed_contexts.pop(context_id, None)
        await super().on_audio_context_interrupted(context_id)

    async def cleanup(self) -> None:
        try:
            await super().cleanup()  # type: ignore[no-untyped-call]
        finally:
            self._failed_contexts.clear()
            if self._owns_client and not self._client.is_closed:
                await self._client.aclose()
