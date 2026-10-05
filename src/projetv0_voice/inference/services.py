"""Native Pipecat inference service construction."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.frames.frames import ErrorFrame, FatalErrorFrame, Frame
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.whisper.base_stt import language_to_whisper_language
from pipecat.transcriptions.language import Language
from pydantic import SecretStr

from projetv0_voice.qualified_profile import InferenceProfileV1

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ISO_639_1 = re.compile(r"^[a-z]{2}$")
_STT_TIMEOUT_SECONDS = 8.0
_HTTP_TIMEOUT = httpx.Timeout(8.0, connect=2.0)


class _BoundedOpenAISTTService(OpenAISTTService):
    """Bound native segmented transcription without changing its scheduling."""

    # Pipecat 1.7.0's abstract STTService lacks yield; its native Whisper method yields.
    # Preserve the native async-generator contract exercised by HTTP deadline tests.
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame]:  # type: ignore[override]
        try:
            async with asyncio.timeout(_STT_TIMEOUT_SECONDS):
                async with aclosing(super().run_stt(audio)) as frames:
                    async for frame in frames:
                        if isinstance(frame, ErrorFrame):
                            yield FatalErrorFrame(error="openrouter_stt_transport")
                            return
                        yield frame
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            yield FatalErrorFrame(error="openrouter_stt_timeout")


class _TrustlessOpenRouterLLMService(OpenRouterLLMService):
    """Pin-aware OpenRouter client factory without ambient HTTP authority."""

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
