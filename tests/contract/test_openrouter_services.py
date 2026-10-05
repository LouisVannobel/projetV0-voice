from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pydantic import SecretStr

from projetv0_voice.qualified_profile import InferenceProfileV1

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _profile_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stt_model": "test/stt",
        "llm_model": "test/llm",
        "tts_model": "test/tts",
        "tts_voice": "fr-FR-Soleil:MAI-Voice-2",
        "tts_pcm_sample_rate": 24000,
        "tts_pcm_channels": 1,
        "llm_provider_policy": {"sort": "latency", "allow_fallbacks": True},
        "tts_provider_options": {"azure": {"style": "cheerful"}},
    }


def _services_module():
    module = import_module("projetv0_voice.inference.services")
    assert callable(getattr(module, "build_stt", None))
    assert callable(getattr(module, "build_llm", None))
    return module


@pytest.mark.asyncio
async def test_build_stt_uses_native_openrouter_multipart_with_manifest_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _services_module()
    profile = InferenceProfileV1.model_validate(_profile_data())
    service = module.build_stt(profile, SecretStr("unit-secret"), language="fr-FR")
    create = AsyncMock(return_value=SimpleNamespace(text="bonjour"))
    monkeypatch.setattr(service._client.audio.transcriptions, "create", create)
    try:
        result = await service._transcribe(b"\x00\x00")
    finally:
        await service._client.close()

    assert isinstance(service, OpenAISTTService)
    assert str(service._client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert service._settings.model == "test/stt"
    assert service._settings.language == "fr"
    assert result.text == "bonjour"
    assert create.await_args.kwargs == {
        "file": ("audio.wav", b"\x00\x00", "audio/wav"),
        "model": "test/stt",
        "language": "fr",
    }


@pytest.mark.asyncio
async def test_build_stt_passes_the_publicly_injected_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _services_module()
    profile = InferenceProfileV1.model_validate(_profile_data())
    client = DefaultAsyncHttpxClient()
    captured: dict[str, object] = {}

    def capture_init(_self: OpenAISTTService, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(OpenAISTTService, "__init__", capture_init)
    try:
        module.build_stt(
            profile,
            SecretStr("unit-secret"),
            language="fr-FR",
            http_client=client,
        )
        assert captured["http_client"] is client
    finally:
        await client.aclose()


@pytest.mark.parametrize("language", ["not-a-language", "ast"])
def test_build_stt_rejects_non_iso_639_1_language_before_client_construction(
    language: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _services_module()
    profile = InferenceProfileV1.model_validate(_profile_data())

    def unexpected_constructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("native STT client constructed for a rejected language")

    monkeypatch.setattr(OpenAISTTService, "__init__", unexpected_constructor)
    with pytest.raises(ValueError, match="unsupported_stt_language"):
        module.build_stt(profile, SecretStr("unit-secret"), language=language)


@pytest.mark.asyncio
async def test_build_llm_serializes_one_openrouter_provider_object_via_native_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _services_module()
    source = _profile_data()
    profile = InferenceProfileV1.model_validate(source)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def create_client(
        _self: OpenRouterLLMService,
        api_key: str | None = None,
        base_url: str | None = None,
        **_kwargs: object,
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=transport_client,
        )

    trustless = module._TrustlessOpenRouterLLMService
    monkeypatch.setattr(trustless, "create_client", create_client)
    service = module.build_llm(profile, SecretStr("unit-secret"))
    source_policy = source["llm_provider_policy"]
    assert isinstance(source_policy, dict)
    source_policy["sort"] = "price"

    params = service.build_chat_completion_params(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": NOT_GIVEN,
            "tool_choice": NOT_GIVEN,
        }
    )
    stream = await service._client.chat.completions.create(**params)
    await stream.close()
    await service._client.close()

    assert isinstance(service, OpenRouterLLMService)
    assert type(service) is trustless
    assert str(service._client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert service._settings.model == "test/llm"
    assert service._settings.extra == {
        "extra_body": {
            "provider": {"sort": "latency", "allow_fallbacks": True},
        }
    }
    assert len(captured) == 1
    body = json.loads(captured[0].content)
    assert body["provider"] == {"sort": "latency", "allow_fallbacks": True}
    assert "extra_body" not in body
    assert "provider" not in body["provider"]


@pytest.mark.asyncio
async def test_native_service_factories_do_not_expose_secret_in_repr_settings_or_logs() -> None:
    module = _services_module()
    profile = InferenceProfileV1.model_validate(_profile_data())
    secret = "sentinel-unit-secret"
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        stt = module.build_stt(profile, SecretStr(secret), language="fr-FR")
        llm = module.build_llm(profile, SecretStr(secret))
    finally:
        logger.remove(sink)

    try:
        visible = "\n".join(
            [repr(stt), repr(stt._settings), repr(llm), repr(llm._settings), *messages]
        )
        assert secret not in visible
    finally:
        await stt._client.close()
        await llm._client.close()
