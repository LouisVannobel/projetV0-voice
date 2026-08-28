"""Native Pipecat inference service construction."""

from __future__ import annotations

import re
from typing import Any, cast

from openai import DefaultAsyncHttpxClient
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.whisper.base_stt import language_to_whisper_language
from pipecat.transcriptions.language import Language
from pydantic import SecretStr

from projetv0_voice.qualified_profile import InferenceProfileV1

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ISO_639_1 = re.compile(r"^[a-z]{2}$")


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

    return OpenAISTTService(
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
    return OpenRouterLLMService(
        api_key=api_key.get_secret_value(),
        base_url=_OPENROUTER_BASE_URL,
        settings=OpenRouterLLMService.Settings(
            model=profile.llm_model,
            extra={"extra_body": {"provider": policy}},
        ),
    )
