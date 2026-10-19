from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio
from openai import OpenAIError
from pipecat.processors.aggregators.llm_context import LLMContext
from pydantic import SecretStr

from projetv0_voice.inference import services
from projetv0_voice.persistence.business_result import RetainedCall, RetainedTurn
from projetv0_voice.qualified_profile import InferenceProfileV1


class _LLMHTTPPeer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.writers: set[asyncio.StreamWriter] = set()
        self.received = asyncio.Event()
        self.hold_first_headers = False

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        try:
            first_line = await reader.readline()
            assert first_line.startswith(b"POST /v1/chat/completions HTTP/1.1")
            headers: dict[bytes, bytes] = {}
            while (line := await reader.readline()) != b"\r\n":
                if not line:
                    return
                name, value = line.split(b":", 1)
                headers[name.lower()] = value.strip()
            length = int(headers[b"content-length"])
            assert 0 < length <= 262_144
            body = json.loads(await reader.readexactly(length))
            self.requests.append(body)
            self.received.set()
            if self.hold_first_headers and len(self.requests) == 1:
                await reader.read()
                return
            common = {"id": "offline", "created": 1, "model": "test/llm"}
            if body.get("stream") is True:
                chunk = {
                    **common,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                                 "finish_reason": "stop"}],
                }
                response = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
                content_type = b"text/event-stream"
            else:
                response = json.dumps(
                    {**common, "object": "chat.completion", "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": '{"result":null}'},
                         "finish_reason": "stop"}
                    ]}
                ).encode()
                content_type = b"application/json"
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: " + content_type
                + b"\r\nContent-Length: " + str(len(response)).encode()
                + b"\r\nConnection: close\r\n\r\n" + response
            )
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.writers.discard(writer)


@pytest_asyncio.fixture
async def peer(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_LLMHTTPPeer]:
    endpoint = _LLMHTTPPeer()
    server = await asyncio.start_server(endpoint.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", f"http://127.0.0.1:{port}/v1")
    try:
        yield endpoint
    finally:
        server.close()
        await server.wait_closed()
        for writer in tuple(endpoint.writers):
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()


def _service(*, sparra: bool = True) -> Any:
    profile = InferenceProfileV1.model_validate(
        {
            "schema_version": 1,
            "stt_model": "test/stt",
            "llm_model": "test/llm",
            "tts_model": "test/tts",
            "tts_voice": "test-voice",
            "tts_pcm_sample_rate": 24_000,
            "tts_pcm_channels": 1,
            "llm_provider_policy": {"sort": "latency", "allow_fallbacks": True},
            "tts_provider_options": {},
        }
    )
    return services.build_llm(profile, SecretStr("offline-unit-key"), sparra_conversation=sparra)


async def _ordinary(service: Any) -> None:
    stream = await service.get_chat_completions(LLMContext([{"role": "user", "content": "ping"}]))
    async for _chunk in stream:
        pass


async def _close(service: Any) -> None:
    await service.cleanup()
    await service._client.close()


@pytest.mark.asyncio
async def test_paid_fence_refuses_actual_completion_dispatch(peer: _LLMHTTPPeer) -> None:
    service = _service()
    # This is the final HTTP boundary, after SDK/context preparatory awaits.
    service.paid_work_permitted = lambda: False
    try:
        with pytest.raises((services.LLMOutputLimitError, OpenAIError)):
            await _ordinary(service)
        assert peer.requests == []
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_paid_summary_has_no_exemption_after_cleanup(peer: _LLMHTTPPeer) -> None:
    service = _service()
    retained = RetainedCall((RetainedTurn(UUID(int=1), 1, "user", "Please call back", False),), 0)
    try:
        await service.cleanup()
        service.paid_work_permitted = lambda: False
        with pytest.raises((services.LLMOutputLimitError, OpenAIError)):
            await service.infer_final_result_once(retained, None)
        assert peer.requests == []
    finally:
        await service._client.close()


@pytest.mark.asyncio
async def test_committed_unaffordable_floor_blocks_wire_before_refresh_resumes(
    tmp_path, monkeypatch, peer: _LLMHTTPPeer,
) -> None:
    from decimal import Decimal

    from tests.integration.test_fixed_v2_process import NOW, registry_case
    from tests.integration.test_owned_pilot_budget_admission import (
        attach,
        budget_rows,
        committed,
        completed_observe_barrier,
        saved_permit,
    )
    from tests.unit.test_owned_pilot_budget import observation
    from tests.unit.test_sparra_admission import event

    async with registry_case(tmp_path) as case:
        gate = await attach(case)
        await committed(case.registry, case.writer, event(occurred_at=NOW))
        permit = saved_permit(case.path)
        service = _service()
        service.paid_work_permitted = lambda: gate.permits(permit)
        async def collect():
            return observation(openrouter_reported_spend_usd=Decimal("1.30"))
        gate.collect = collect
        refresh = None
        try:
            async with completed_observe_barrier(case.writer, monkeypatch) as (entered, _release):
                refresh = asyncio.create_task(gate.refresh())
                await asyncio.wait_for(entered.wait(), 2)
                assert budget_rows(case.path)[0][1] == "1.30"
                with pytest.raises((services.LLMOutputLimitError, OpenAIError)):
                    await _ordinary(service)
                assert peer.requests == []
        finally:
            if refresh is not None:
                await refresh
            await _close(service)


@pytest.mark.asyncio
async def test_fifty_fifth_native_256_ceiling_refuses_before_wire(peer: _LLMHTTPPeer) -> None:
    service = _service()
    try:
        for _ in range(54):
            await _ordinary(service)
        assert len(peer.requests) == 54
        assert sum(body["max_tokens"] for body in peer.requests) == 13_824
        with pytest.raises(OpenAIError, match="llm_output_limit"):
            await _ordinary(service)
        assert len(peer.requests) == 54
    finally:
        await _close(service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_body",
    [
        {"n": 2},
        {"max_tokens": 512},
        {"max_completion_tokens": 256},
    ],
)
async def test_final_sdk_extra_body_is_guarded_without_clamping(
    peer: _LLMHTTPPeer, extra_body: dict[str, int]
) -> None:
    service = _service()
    service._settings.extra["extra_body"].update(extra_body)
    try:
        with pytest.raises(OpenAIError, match="llm_output_limit"):
            await _ordinary(service)
        assert peer.requests == [], "the final merged body must refuse before the TCP peer"
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_ordinary_schema_cannot_claim_protected_2048_reserve(peer: _LLMHTTPPeer) -> None:
    service = _service()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    try:
        with pytest.raises(OpenAIError, match="llm_output_limit"):
            await service.run_inference(
                LLMContext([{"role": "user", "content": "ordinary structured request"}]),
                max_tokens=2048,
                response_schema=schema,
            )
        assert peer.requests == []
    finally:
        await _close(service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_body",
    [
        {"max_tokens": None}, {"max_tokens": True}, {"max_tokens": "256"},
        {"max_tokens": 256.0}, {"max_tokens": 0}, {"max_tokens": -1},
        {"max_tokens": 257}, {"max_completion_tokens": None}, {"n": True}, {"n": 1.0},
    ],
)
async def test_invalid_final_ceilings_and_n_are_exact_typed_sdk_refusals(
    peer: _LLMHTTPPeer, extra_body: dict[str, Any]
) -> None:
    service = _service()
    service._settings.extra["extra_body"].update(extra_body)
    try:
        with pytest.raises(services.LLMOutputLimitError) as refusal:
            await _ordinary(service)
        assert type(refusal.value) is services.LLMOutputLimitError
        assert str(refusal.value) == "llm_output_limit"
        assert peer.requests == []
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_actual_warmup_and_final_override_reach_exact_ordinary_limit(
    peer: _LLMHTTPPeer,
) -> None:
    service = _service()
    try:
        await service.run_inference(
            LLMContext([{"role": "user", "content": "warmup"}]), max_tokens=256
        )
        for _ in range(53):
            await _ordinary(service)
        await service.run_inference(
            LLMContext([{"role": "user", "content": "remainder"}]), max_tokens=128
        )
        assert sum(body["max_tokens"] for body in peer.requests) == 13_952
        assert peer.requests[-1]["max_tokens"] == 128
        with pytest.raises(services.LLMOutputLimitError):
            await service.run_inference(
                LLMContext([{"role": "user", "content": "over"}]), max_tokens=1
            )
        assert len(peer.requests) == 55
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_two_native_requests_race_for_only_one_remaining_ceiling(peer: _LLMHTTPPeer) -> None:
    service = _service()
    try:
        for _ in range(53):
            await _ordinary(service)
        outcomes = await asyncio.gather(
            _ordinary(service), _ordinary(service), return_exceptions=True
        )
        assert sum(type(item) is services.LLMOutputLimitError for item in outcomes) == 1
        assert sum(item is None for item in outcomes) == 1
        assert len(peer.requests) == 54
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_received_post_cancel_preserves_reserved_ceiling_and_does_not_retry(
    peer: _LLMHTTPPeer,
) -> None:
    peer.hold_first_headers = True
    service = _service()
    try:
        pending = asyncio.create_task(_ordinary(service))
        await asyncio.wait_for(peer.received.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        for _ in range(53):
            await _ordinary(service)
        with pytest.raises(services.LLMOutputLimitError):
            await _ordinary(service)
        assert len(peer.requests) == 54
        assert sum(body["max_tokens"] for body in peer.requests) == 13_824
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_calls_are_isolated_and_generic_final_method_keeps_legacy_behavior(
    peer: _LLMHTTPPeer,
) -> None:
    guarded, another, generic = _service(), _service(), _service(sparra=False)
    retained = RetainedCall(
        (RetainedTurn(UUID(int=1), 1, "user", "synthetic caller request", False),), 0
    )
    try:
        guarded._settings.extra["extra_body"]["max_tokens"] = 257
        with pytest.raises(services.LLMOutputLimitError):
            await _ordinary(guarded)
        await _ordinary(another)
        assert peer.requests[-1]["max_tokens"] == 256
        generic._settings.extra["extra_body"]["max_tokens"] = 512
        await _ordinary(generic)
        assert peer.requests[-1]["max_tokens"] == 512
        generic._settings.extra["extra_body"].pop("max_tokens")
        assert await generic.infer_final_result_once(retained, None) is None
        assert await generic.infer_final_result_once(retained, None) is None
        assert len(peer.requests) == 4
        assert all(body["stream"] is False for body in peer.requests[-2:])
    finally:
        for service in (guarded, another, generic):
            await _close(service)
