from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import httpx
import pytest
from openai import APIStatusError, DefaultAsyncHttpxClient
from pipecat.processors.aggregators.llm_context import LLMContext

from projetv0_voice.inference import services
from projetv0_voice.persistence.business_result import (
    ResultCompletionRejected,
    RetainedCall,
    RetainedTurn,
)
from tests.unit.test_llm_consumption_limits import _service

PRIVATE = "private-response-sentinel-not-a-diagnostic"
USER = UUID("00000000-0000-4000-8000-000000000001")
RETAINED = RetainedCall((RetainedTurn(USER, 1, "user", PRIVATE, False),), 0)
RESULT = {
    "schema_version": 1, "quality": "partial", "category": "callback",
    "summary": "Demande de devis", "next_action": "Rappeler",
    "contact": {"name": None, "callback_e164": None, "preference": None,
                "callback_source": "missing", "callback_confirmed": False},
    "evidence": [{"turn_id": str(USER), "role": "user"}], "request_confirmed": False,
}


def _completion(content: str | None, reason: str | None = "stop") -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": content}}
    if reason is not None:
        choice["finish_reason"] = reason
    return {"id": PRIVATE, "object": "chat.completion", "created": 1,
            "model": "test/llm", "choices": [choice]}


@asynccontextmanager
async def _native_service(
    monkeypatch: pytest.MonkeyPatch, response: Callable[[], httpx.Response], *,
    guarded: bool = True,
) -> AsyncIterator[tuple[Any, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response()

    monkeypatch.setattr(services, "DefaultAsyncHttpxClient", lambda **kwargs:
        DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler), **kwargs))
    service = _service(sparra=guarded)
    try:
        yield service, requests
    finally:
        await service._client.close()
        await service.cleanup()


async def _final(service: Any) -> Any:
    await service.cleanup()
    return await service.infer_final_result_once(RETAINED, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [RESULT, None])
async def test_stop_valid_result_or_explicit_null_keeps_native_contract(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any] | None,
) -> None:
    payload = _completion(json.dumps({"result": result}))
    payload["usage"] = {"completion_tokens": 17,
                        "completion_tokens_details": {"reasoning_tokens": 0}}
    async with _native_service(monkeypatch, lambda: httpx.Response(200, json=payload)) as (
        service, requests,
    ):
        actual = await _final(service)
        assert (actual is None) == (result is None)
        diagnostic = service.final_completion_diagnostic
        assert diagnostic == services.FinalCompletionDiagnostic("stop", True, 17, 0)
        assert PRIVATE not in repr(diagnostic)
        assert len(requests) == 1
        assert requests[0]["stream"] is False and requests[0]["max_tokens"] == 2048
        assert requests[0]["reasoning"] == {"enabled": False}
        assert requests[0]["provider"]["require_parameters"] is True
        assert service._final_post_reserved is True and service._client.max_retries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    "{" + PRIVATE, json.dumps({"result": RESULT}), '{"result":null}',
])
async def test_length_is_rejected_even_when_content_is_valid_business_json(
    monkeypatch: pytest.MonkeyPatch, content: str,
) -> None:
    async with _native_service(monkeypatch, lambda: httpx.Response(
        200, json=_completion(content, "length"),
    )) as (service, requests):
        with pytest.raises(
            ResultCompletionRejected, match="^result_completion_rejected$",
        ) as rejected:
            await _final(service)
        assert rejected.value.reason == "length"
        assert service.final_completion_diagnostic.finish_reason == "length"
        assert len(requests) == 1 and service._final_post_reserved is True
        with pytest.raises(services.LLMOutputLimitError):
            await service.infer_final_result_once(RETAINED, None)
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["content_filter", "tool_calls", "error", None, PRIVATE])
async def test_nonstop_or_missing_reason_is_closed_before_schema_validation(
    monkeypatch: pytest.MonkeyPatch, reason: str | None,
) -> None:
    async with _native_service(monkeypatch, lambda: httpx.Response(
        200, json=_completion('{"result":null}', reason),
    )) as (service, requests):
        with pytest.raises(ResultCompletionRejected) as rejected:
            await _final(service)
        expected = "unknown" if reason is None or reason == PRIVATE else reason
        assert rejected.value.reason == expected
        assert service.final_completion_diagnostic.finish_reason == expected
        assert PRIVATE not in repr(service.final_completion_diagnostic)
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", " \n\t"])
async def test_stop_with_no_nonempty_content_is_not_a_successful_null_wrapper(
    monkeypatch: pytest.MonkeyPatch, content: str | None,
) -> None:
    async with _native_service(monkeypatch, lambda: httpx.Response(
        200, json=_completion(content),
    )) as (service, requests):
        with pytest.raises(ResultCompletionRejected) as rejected:
            await _final(service)
        assert rejected.value.reason == "empty"
        assert service.final_completion_diagnostic.content_nonempty is False
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, {}, {"completion_tokens": True,
    "completion_tokens_details": {"reasoning_tokens": PRIVATE}},
    {"completion_tokens": -1, "completion_tokens_details": {"reasoning_tokens": 1.5}}])
async def test_missing_or_untrusted_usage_is_optional_and_never_retained_raw(
    monkeypatch: pytest.MonkeyPatch, usage: dict[str, Any] | None,
) -> None:
    payload = _completion('{"result":null}')
    if usage is not None:
        payload["usage"] = usage
    async with _native_service(monkeypatch, lambda: httpx.Response(200, json=payload)) as (
        service, _requests,
    ):
        assert await _final(service) is None
        diagnostic = service.final_completion_diagnostic
        assert diagnostic.completion_tokens is None and diagnostic.reasoning_tokens is None
        assert PRIVATE not in repr(diagnostic)


class _StreamingBody(httpx.AsyncByteStream):
    def __init__(self, *, hold: bool = False) -> None:
        self.reads = 0
        self.reading = asyncio.Event()
        self.closed = False
        self.hold = hold

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.reads += 1
        if self.hold:
            yield b'{"id":"offline",'
            self.reading.set()
            await asyncio.Event().wait()
        else:
            yield b'data: [DONE]\n\n'

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ordinary_sse_is_not_buffered_by_the_final_response_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _StreamingBody()
    async with _native_service(monkeypatch, lambda: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body,
    )) as (service, requests):
        stream = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": PRIVATE}]),
        )
        assert body.reads == 0 and service.final_completion_diagnostic is None
        async for _chunk in stream:
            pass
        await stream.close()
        assert body.reads == 1 and len(requests) == 1 and requests[0]["stream"] is True
        assert service._ordinary_output_reserved == 256


@pytest.mark.asyncio
async def test_real_async_stream_cancelled_inside_aread_does_not_retry_or_refund(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _StreamingBody(hold=True)
    response = httpx.Response(200, headers={"content-type": "application/json"}, stream=body)
    assert response.is_stream_consumed is False
    async with _native_service(monkeypatch, lambda: response) as (service, requests):
        pending = asyncio.create_task(_final(service))
        await asyncio.wait_for(body.reading.wait(), timeout=2)
        assert response.is_stream_consumed is True and body.reads == 1
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert body.closed and service.final_completion_diagnostic is None
        assert service._final_post_reserved is True and len(requests) == 1
        with pytest.raises(services.LLMOutputLimitError):
            await service.infer_final_result_once(RETAINED, None)
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_native_http_error_is_not_replaced_by_completion_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _native_service(monkeypatch, lambda: httpx.Response(
        400, json={"error": {"message": PRIVATE, "type": "bad_request"}},
    )) as (service, requests):
        with pytest.raises(APIStatusError):
            await _final(service)
        assert service.final_completion_diagnostic is None
        assert service._final_post_reserved is True and len(requests) == 1


@pytest.mark.asyncio
async def test_missing_completion_observation_refuses_without_another_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _native_service(monkeypatch, lambda: httpx.Response(
        200, json=_completion('{"result":null}'),
    )) as (service, requests):
        service._client._client.event_hooks["response"] = []
        with pytest.raises(ResultCompletionRejected) as rejected:
            await _final(service)
        assert rejected.value.reason == "unknown"
        assert service.final_completion_diagnostic is None
        assert service._final_post_reserved is True and len(requests) == 1


@pytest.mark.asyncio
async def test_diagnostics_are_call_local_and_generic_services_keep_native_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def length() -> httpx.Response:
        return httpx.Response(200, json=_completion('{"result":null}', "length"))

    def stop() -> httpx.Response:
        return httpx.Response(200, json=_completion('{"result":null}'))
    async with _native_service(monkeypatch, length) as (first, _requests):
        async with _native_service(monkeypatch, stop) as (second, _other_requests):
            with pytest.raises(ResultCompletionRejected):
                await _final(first)
            first_diagnostic = first.final_completion_diagnostic
            assert second.final_completion_diagnostic is None
            assert await _final(second) is None
            assert first.final_completion_diagnostic is first_diagnostic
            assert second.final_completion_diagnostic.finish_reason == "stop"
        async with _native_service(monkeypatch, length, guarded=False) as (
            generic, _generic_requests,
        ):
            assert await _final(generic) is None
            assert generic.final_completion_diagnostic is None
