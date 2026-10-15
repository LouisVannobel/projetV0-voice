from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import httpx
import pytest
from openai import DefaultAsyncHttpxClient, OpenAIError
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pydantic import SecretStr

from projetv0_voice import production_wiring
from projetv0_voice.inference import services
from projetv0_voice.persistence.business_result import (
    RetainedCall,
    RetainedTurn,
)
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from tests.unit.test_config import manifest_data, write_bundle
from tests.unit.test_production_wiring import _settings

PROVIDER_POLICY = {
    "only": ["baseten", "together"],
    "allow_fallbacks": True,
    "require_parameters": True,
    "sort": "latency",
}


def _profile() -> QualifiedDeploymentProfileV1:
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(encoding="utf-8")
    )
    profile_data = profile.model_dump()
    profile_data["inference"]["llm_provider_policy"] = PROVIDER_POLICY
    return QualifiedDeploymentProfileV1.model_validate(profile_data)


@asynccontextmanager
async def _recorded_production_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sparra: bool,
) -> AsyncIterator[tuple[OpenRouterLLMService, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body["stream"]:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content="data: [DONE]\n\n",
            )
        return httpx.Response(200, json={
            "id": "unit-summary", "object": "chat.completion", "created": 0,
            "model": "test/llm", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"result":null}'},
            }],
        })

    monkeypatch.setattr(services, "DefaultAsyncHttpxClient", lambda **kwargs:
        DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler), **kwargs))
    # This test supplies a local manifest; descriptor/digest admission has its own gates.
    monkeypatch.setattr(production_wiring, "_validate_artifacts", lambda settings: None)
    extension = {"sparra": {
        "schema_version": 1, "connection_id": "fixture", "original_forward_line_e164": None,
        "qualified_transfer_destination_e164": None,
    }} if sparra else {}
    bundle = write_bundle(tmp_path / "bundle", manifest_data(
        max_concurrent_calls=1, transcript_retention_days=30 if sparra else 7, **extension,
    ))
    settings = _settings(agent_bundle_path=PurePosixPath(str(bundle)))
    factories = production_wiring.build_production_factories(settings)
    factories.validate_artifacts(settings)
    factories.load_manifest(settings)
    llm = factories.inference_factory(SecretStr("unit-secret"), _profile(), "fr-FR").llm_factory()
    try:
        yield llm, requests
    finally:
        await llm._client.close()
        await llm.cleanup()


async def _conversation_request(llm: OpenRouterLLMService) -> None:
    stream = await llm.get_chat_completions(LLMContext([{"role": "user", "content": "Bonjour"}]))
    await stream.close()


@pytest.mark.asyncio
async def test_sparra_conversation_sends_one_256_token_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _recorded_production_llm(tmp_path, monkeypatch, sparra=True) as (llm, requests):
        await _conversation_request(llm)
    assert len(requests) == 1
    assert requests[0].get("max_tokens") == 256
    assert "max_completion_tokens" not in requests[0]
    assert requests[0].get("reasoning") == {"enabled": False}
    assert requests[0]["provider"] == PROVIDER_POLICY


@pytest.mark.asyncio
async def test_shared_non_sparra_conversation_keeps_native_unset_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _recorded_production_llm(tmp_path, monkeypatch, sparra=False) as (llm, requests):
        await _conversation_request(llm)
    assert len(requests) == 1
    assert "max_tokens" not in requests[0]
    assert "max_completion_tokens" not in requests[0]
    assert "reasoning" not in requests[0]
    assert requests[0]["provider"] == PROVIDER_POLICY


@pytest.mark.asyncio
async def test_shared_non_sparra_native_summary_keeps_existing_2048_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _recorded_production_llm(tmp_path, monkeypatch, sparra=False) as (llm, requests):
        await llm.run_inference(
            LLMContext([{"role": "user", "content": "Rappelez-moi"}]), max_tokens=2048,
            response_schema={"type": "object"},
        )
    assert len(requests) == 1
    assert requests[0]["max_completion_tokens"] == 2048
    assert "max_tokens" not in requests[0]
    assert "reasoning" not in requests[0]
    assert requests[0]["provider"] == PROVIDER_POLICY


@pytest.mark.asyncio
async def test_sparra_final_summary_preserves_ceiling_and_keeps_conversation_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = RetainedCall((RetainedTurn(
        UUID("00000000-0000-4000-8000-000000000001"), 1, "user", "Rappelez-moi", False,
    ),), 0)
    async with _recorded_production_llm(tmp_path, monkeypatch, sparra=True) as (llm, requests):
        await _conversation_request(llm)
        await llm.cleanup()
        assert await llm.infer_final_result_once(retained, None) is None
        assert llm._settings.max_tokens == 256
        with pytest.raises(OpenAIError, match="llm_output_limit"):
            await _conversation_request(llm)
    assert len(requests) == 2
    summary = requests[1]
    assert summary["stream"] is False
    assert summary["response_format"]["type"] == "json_schema"
    assert summary["response_format"]["json_schema"]["strict"] is True
    assert summary.get("max_tokens") == 2048
    assert "max_completion_tokens" not in summary
    assert summary.get("reasoning") == {"enabled": False}
    assert summary["provider"] == PROVIDER_POLICY
    assert requests[0]["max_tokens"] == 256
    assert requests[0]["stream"] is True
    assert requests[0].get("reasoning") == {"enabled": False}
