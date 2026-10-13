"""Offline connected-peer wire contract through actual native result inference."""

from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from openai import DefaultAsyncHttpxClient
from pydantic import SecretStr

from projetv0_voice.inference import services
from projetv0_voice.persistence.business_result import (
    RetainedCall,
    RetainedTurn,
    infer_partial_result,
)
from projetv0_voice.qualified_profile import InferenceProfileV1
from tests.integration.sparra_connected_scenario import Peers


@pytest.mark.asyncio
async def test_connected_peer_satisfies_native_partial_result_envelope(monkeypatch):
    peer = Peers()
    http = DefaultAsyncHttpxClient(
        transport=httpx.MockTransport(peer.http), trust_env=False
    )
    monkeypatch.setattr(services, "DefaultAsyncHttpxClient", lambda **_kwargs: http)
    profile = InferenceProfileV1.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/inference-profile-v1.json").read_text()
    )
    llm = services.build_llm(profile, SecretStr("owned-offline-key"))
    turn_id = uuid4()
    retained = RetainedCall(
        turns=(RetainedTurn(turn_id, 1, "user", "Pouvez-vous me rappeler ?", False),),
        loss_count=0,
    )
    try:
        result = await infer_partial_result(llm, retained, None)
        assert result is not None
        assert result.summary == "Demande de rappel"
        assert result.quality == "partial" and not result.request_confirmed
        assert result.contact.callback_source == "missing"
        assert len(result.evidence) == 1
        assert result.evidence[0].turn_id == turn_id and result.evidence[0].role == "user"
        assert peer.inferences == 1 and len(peer.requests) == 1
        request = peer.requests[0]
        assert request["stream"] is False and not request.get("tools")
        response_format = request["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert schema["required"] == ["result"]
    finally:
        await llm._client.close()
        assert http.is_closed
