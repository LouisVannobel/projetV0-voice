from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, get_args
from uuid import UUID, uuid4

import httpx
import pytest
import telnyx
from pydantic import SecretStr, ValidationError
from telnyx.resources.calls.actions import AsyncActionsResource
from telnyx.types import (
    StreamBidirectionalSamplingRate,
    StreamBidirectionalTargetLegs,
)

API_KEY = "RAW-API-KEY-SENTINEL"
TOKEN = "RAW-STREAM-TOKEN-SENTINEL"
URL = "wss://voice.example.test/telnyx/media"
CALL_CONTROL_ID = "RAW-CALL-CONTROL-SENTINEL"
COMMAND_ID = UUID("12345678-1234-4abc-8def-1234567890ab")


def call_control() -> Any:
    return importlib.import_module("projetv0_voice.telnyx.call_control")


class FakeActions:
    def __init__(self, observations: list[object]) -> None:
        self.observations = deque(observations)
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def _invoke(self, name: str, call_control_id: str, kwargs: dict[str, object]) -> object:
        self.calls.append((name, call_control_id, kwargs))
        observation = self.observations.popleft()
        if isinstance(observation, BaseException):
            raise observation
        if callable(observation):
            observation = observation()
        if inspect.isawaitable(observation):
            return await observation
        return observation

    async def answer(self, call_control_id: str, **kwargs: object) -> object:
        return await self._invoke("answer", call_control_id, kwargs)

    async def start_streaming(self, call_control_id: str, **kwargs: object) -> object:
        return await self._invoke("start_streaming", call_control_id, kwargs)

    async def start_recording(self, call_control_id: str, **kwargs: object) -> object:
        return await self._invoke("start_recording", call_control_id, kwargs)

    async def hangup(self, call_control_id: str, **kwargs: object) -> object:
        return await self._invoke("hangup", call_control_id, kwargs)


class FakeSDK:
    def __init__(
        self,
        observations: list[object],
        constructor_kwargs: dict[str, object],
        *,
        close_impl: Any | None = None,
    ) -> None:
        self.constructor_kwargs = constructor_kwargs
        self.max_retries = constructor_kwargs["max_retries"]
        self.actions = FakeActions(observations)
        self.calls = SimpleNamespace(actions=self.actions)
        self.close_count = 0
        self._close_impl = close_impl

    async def close(self) -> None:
        self.close_count += 1
        if self._close_impl is not None:
            await self._close_impl()


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    observations: list[object],
    *,
    close_impl: Any | None = None,
) -> tuple[list[dict[str, object]], list[FakeSDK]]:
    constructions: list[dict[str, object]] = []
    instances: list[FakeSDK] = []

    def factory(**kwargs: object) -> FakeSDK:
        constructions.append(dict(kwargs))
        sdk = FakeSDK(observations, dict(kwargs), close_impl=close_impl)
        instances.append(sdk)
        return sdk

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", factory)
    return constructions, instances


def ok_response() -> object:
    return SimpleNamespace(data=SimpleNamespace(result="ok"))


def malformed_response(result: str | None = None) -> object:
    return SimpleNamespace(data=None if result is None else SimpleNamespace(result=result))


def status_error(status_code: int) -> telnyx.APIStatusError:
    request = httpx.Request("POST", "https://api.telnyx.com/v2/calls/RAW-REQUEST-SENTINEL")
    response = httpx.Response(status_code, request=request, text="RAW-BODY-SENTINEL")
    return telnyx.APIStatusError(
        "RAW-STATUS-SENTINEL",
        response=response,
        body={"secret": "RAW-BODY-SENTINEL"},
    )


def response_validation_error() -> telnyx.APIResponseValidationError:
    request = httpx.Request("POST", "https://api.telnyx.com/v2/calls/RAW-REQUEST-SENTINEL")
    response = httpx.Response(200, request=request, text="RAW-BODY-SENTINEL")
    return telnyx.APIResponseValidationError(
        response,
        {"secret": "RAW-BODY-SENTINEL"},
        message="RAW-VALIDATION-SENTINEL",
    )


def connection_error(cause: BaseException) -> telnyx.APIConnectionError:
    request = httpx.Request("POST", "https://api.telnyx.com/v2/calls/RAW-REQUEST-SENTINEL")
    error = telnyx.APIConnectionError(message="RAW-CONNECTION-SENTINEL", request=request)
    error.__cause__ = cause
    return error


def test_locked_sdk_surface_matches_task5_contract() -> None:
    assert telnyx.__version__ == "4.176.0"
    assert set(get_args(StreamBidirectionalTargetLegs)) == {"both", "self", "opposite"}
    assert 8000 in get_args(StreamBidirectionalSamplingRate)
    recording = inspect.signature(AsyncActionsResource.start_recording)
    streaming = inspect.signature(AsyncActionsResource.start_streaming)
    assert "max_length" in recording.parameters
    assert "stream_auth_token" in streaming.parameters
    assert inspect.signature(telnyx.AsyncTelnyx).parameters["max_retries"].default == 2


@pytest.mark.asyncio
async def test_real_sdk_serializes_exact_paths_bodies_and_omissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    real_constructor = telnyx.AsyncTelnyx
    requests: list[tuple[str, str, dict[str, object], bytes]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads((await request.aread()).decode("utf-8"))
        requests.append((request.method, request.url.path, body, request.url.query))
        return httpx.Response(200, json={"data": {"result": "ok"}}, request=request)

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    def factory(**kwargs: object) -> telnyx.AsyncTelnyx:
        return real_constructor(**kwargs, http_client=http_client)  # type: ignore[arg-type]

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", factory)
    client = module.CallControlClient(api_key=API_KEY)
    await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)
    await client.start_streaming(
        CALL_CONTROL_ID,
        module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN),
        command_id=COMMAND_ID,
    )
    await client.start_recording(
        CALL_CONTROL_ID,
        module.RecordingStartV1(play_beep=False),
        command_id=COMMAND_ID,
    )
    await client.hangup(CALL_CONTROL_ID, command_id=COMMAND_ID)
    await client.aclose()

    command = str(COMMAND_ID)
    assert [(method, path, query) for method, path, _, query in requests] == [
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/answer", b""),
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/streaming_start", b""),
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/record_start", b""),
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/hangup", b""),
    ]
    assert [body for _, _, body, _ in requests] == [
        {"command_id": command},
        {
            "command_id": command,
            "stream_auth_token": TOKEN,
            "stream_bidirectional_codec": "PCMU",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_sampling_rate": 8000,
            "stream_bidirectional_target_legs": "self",
            "stream_codec": "PCMU",
            "stream_track": "inbound_track",
            "stream_url": URL,
        },
        {
            "channels": "dual",
            "command_id": command,
            "format": "wav",
            "max_length": 0,
            "play_beep": False,
            "recording_track": "both",
            "timeout_secs": 0,
            "transcription": False,
        },
        {"command_id": command},
    ]


def test_request_models_are_strict_frozen_and_input_redacting() -> None:
    module = call_control()
    streaming = module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN)
    recording = module.RecordingStartV1(play_beep=True)

    assert streaming.stream_url == URL
    assert isinstance(streaming.stream_auth_token, SecretStr)
    assert streaming.stream_auth_token.get_secret_value() == TOKEN
    assert recording.play_beep is True
    assert URL not in repr(streaming)
    assert TOKEN not in repr(streaming)
    with pytest.raises(ValidationError, match="frozen"):
        streaming.stream_url = "wss://changed.example.test"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        module.RecordingStartV1(play_beep=True, trim="trim-silence")


def test_stream_transport_fields_are_absent_from_dumps_and_safe_in_repr_and_str() -> None:
    module = call_control()
    streaming = module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN)

    assert streaming.model_dump() == {}
    assert streaming.model_dump_json() == "{}"
    for rendered in (repr(streaming), str(streaming)):
        assert URL not in rendered
        assert TOKEN not in rendered


@pytest.mark.parametrize(
    "url",
    [
        "http://voice.example.test/media",
        "wss:///missing-host",
        "wss://user:pass@voice.example.test/media",
        "wss://voice.example.test/media?token=RAW-QUERY-SENTINEL",
        "wss://voice.example.test/media?",
        "wss://voice.example.test/media#fragment",
        "wss://voice.example.test/media#",
        "wss://voice.example.test/" + "x" * 2048,
    ],
)
def test_stream_url_rejects_unqualified_shapes_without_reflecting_input(url: str) -> None:
    module = call_control()
    with pytest.raises(ValidationError) as raised:
        module.StreamingStartV1(stream_url=url, stream_auth_token=TOKEN)
    assert url not in str(raised.value)
    assert TOKEN not in str(raised.value)


@pytest.mark.parametrize(
    "url",
    [
        r"wss://voice.example.test/telnyx\media",
        r"wss://voice.example.test/\telnyx/media",
    ],
)
def test_stream_url_rejects_backslash_anywhere_in_path_without_reflecting_input(
    url: str,
) -> None:
    module = call_control()

    with pytest.raises(ValidationError) as raised:
        module.StreamingStartV1(stream_url=url, stream_auth_token=TOKEN)

    assert url not in str(raised.value)
    assert TOKEN not in str(raised.value)


@pytest.mark.parametrize("token", ["", "x" * 4001, 123])
def test_stream_token_is_bounded_strict_and_redacted(token: object) -> None:
    module = call_control()
    with pytest.raises(ValidationError) as raised:
        module.StreamingStartV1(stream_url=URL, stream_auth_token=token)
    if str(token):
        assert str(token) not in str(raised.value)


@pytest.mark.asyncio
async def test_one_owned_sdk_client_and_exact_action_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    constructions, instances = install_fake(monkeypatch, module, [ok_response()] * 4)
    client = module.CallControlClient(api_key=API_KEY)
    streaming = module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN)
    recording = module.RecordingStartV1(play_beep=True)

    results = [
        await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID),
        await client.start_streaming(CALL_CONTROL_ID, streaming, command_id=COMMAND_ID),
        await client.start_recording(CALL_CONTROL_ID, recording, command_id=COMMAND_ID),
        await client.hangup(CALL_CONTROL_ID, command_id=COMMAND_ID),
    ]

    assert [result.outcome for result in results] == ["accepted"] * 4
    assert constructions == [{"api_key": API_KEY, "max_retries": 0}]
    assert len(instances) == 1
    calls = instances[0].actions.calls
    assert [call[:2] for call in calls] == [
        ("answer", CALL_CONTROL_ID),
        ("start_streaming", CALL_CONTROL_ID),
        ("start_recording", CALL_CONTROL_ID),
        ("hangup", CALL_CONTROL_ID),
    ]
    command = str(COMMAND_ID)
    assert set(calls[0][2]) == {"command_id", "timeout"}
    assert calls[0][2]["command_id"] == command
    assert calls[1][2] == {
        "stream_track": "inbound_track",
        "stream_bidirectional_mode": "rtp",
        "stream_bidirectional_codec": "PCMU",
        "stream_bidirectional_sampling_rate": 8000,
        "stream_bidirectional_target_legs": "self",
        "stream_codec": "PCMU",
        "stream_url": URL,
        "stream_auth_token": TOKEN,
        "command_id": command,
        "timeout": calls[1][2]["timeout"],
    }
    assert calls[2][2] == {
        "channels": "dual",
        "format": "wav",
        "max_length": 0,
        "recording_track": "both",
        "timeout_secs": 0,
        "transcription": False,
        "play_beep": True,
        "command_id": command,
        "timeout": calls[2][2]["timeout"],
    }
    assert "trim" not in calls[2][2]
    assert set(calls[3][2]) == {"command_id", "timeout"}
    for _, _, kwargs in calls:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (0.5,) * 4
    rendered = repr(client) + "".join(repr(result) for result in results)
    for secret in (API_KEY, TOKEN, URL, CALL_CONTROL_ID, str(COMMAND_ID)):
        assert secret not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_call_id", ["", "x" * 1025, 123])
async def test_call_control_id_is_bounded_opaque_not_uuid_coerced(
    monkeypatch: pytest.MonkeyPatch, bad_call_id: object
) -> None:
    module = call_control()
    _, instances = install_fake(monkeypatch, module, [ok_response()])
    client = module.CallControlClient(api_key=API_KEY)

    with pytest.raises(module.CallControlInputError, match="call_control_input_invalid") as raised:
        await client.answer(bad_call_id, command_id=COMMAND_ID)

    if str(bad_call_id):
        assert str(bad_call_id) not in repr(raised.value)
    assert instances[0].actions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_command", [UUID(int=1), str(COMMAND_ID), uuid4().hex])
async def test_command_id_requires_caller_owned_uuidv4_object(
    monkeypatch: pytest.MonkeyPatch, bad_command: object
) -> None:
    module = call_control()
    _, instances = install_fake(monkeypatch, module, [ok_response()])
    client = module.CallControlClient(api_key=API_KEY)

    with pytest.raises(module.CallControlInputError, match="call_control_input_invalid"):
        await client.answer(CALL_CONTROL_ID, command_id=bad_command)

    assert instances[0].actions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("observations", "outcome", "calls"),
    [
        ([ok_response()], "accepted", 1),
        ([malformed_response()], "outcome_unknown", 1),
        ([malformed_response("queued")], "outcome_unknown", 1),
        ([status_error(400)], "rejected", 1),
        ([status_error(409)], "rejected", 1),
        ([status_error(429)], "rate_limited", 1),
        ([status_error(503)], "outcome_unknown", 1),
        ([response_validation_error()], "outcome_unknown", 1),
        ([connection_error(httpx.PoolTimeout("pool"))], "retryable_not_sent", 1),
        ([status_error(500), ok_response()], "accepted", 2),
        ([status_error(408), status_error(429)], "outcome_unknown", 2),
        ([status_error(500), status_error(400)], "outcome_unknown", 2),
        (
            [
                connection_error(httpx.ConnectError("connect")),
                connection_error(httpx.ConnectTimeout("connect")),
            ],
            "retryable_not_sent",
            2,
        ),
        (
            [connection_error(httpx.ConnectError("connect")), status_error(429)],
            "rate_limited",
            2,
        ),
        (
            [connection_error(httpx.WriteError("write")), status_error(400)],
            "outcome_unknown",
            2,
        ),
    ],
)
async def test_outcome_and_retry_matrix_is_bounded_and_ambiguity_is_monotonic(
    monkeypatch: pytest.MonkeyPatch,
    observations: list[object],
    outcome: str,
    calls: int,
) -> None:
    module = call_control()
    _, instances = install_fake(monkeypatch, module, list(observations))
    client = module.CallControlClient(api_key=API_KEY)

    result = await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)

    assert result.outcome == outcome
    assert len(instances[0].actions.calls) == calls
    assert len(instances[0].actions.calls) <= 2
    rendered = repr(result)
    for sentinel in ("RAW-STATUS-SENTINEL", "RAW-BODY-SENTINEL", "RAW-REQUEST-SENTINEL"):
        assert sentinel not in rendered


@pytest.mark.asyncio
async def test_private_retry_reuses_exact_streaming_closure_and_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    _, instances = install_fake(monkeypatch, module, [status_error(500), status_error(500)])
    client = module.CallControlClient(api_key=API_KEY)
    request = module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN)

    result = await client.start_streaming(CALL_CONTROL_ID, request, command_id=COMMAND_ID)

    assert result.outcome == "outcome_unknown"
    assert len(instances[0].actions.calls) == 2
    assert instances[0].actions.calls[0] == instances[0].actions.calls[1]


@pytest.mark.asyncio
async def test_each_attempt_has_true_monotonic_500ms_deadline_without_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    cancelled = asyncio.Event()

    async def no_response() -> object:
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    _, instances = install_fake(monkeypatch, module, [no_response, ok_response()])
    client = module.CallControlClient(api_key=API_KEY)
    started = time.monotonic()

    result = await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)

    elapsed = time.monotonic() - started
    assert result.outcome == "accepted"
    assert 0.45 <= elapsed < 0.8
    assert cancelled.is_set()
    assert len(instances[0].actions.calls) == 2


@pytest.mark.asyncio
async def test_external_cancellation_propagates_without_retry_or_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked() -> object:
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    _, instances = install_fake(monkeypatch, module, [blocked, ok_response()])
    client = module.CallControlClient(api_key=API_KEY)
    pending = asyncio.create_task(client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID))
    await entered.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancelled.is_set()
    assert len(instances[0].actions.calls) == 1


@pytest.mark.asyncio
async def test_close_is_owned_concurrent_idempotent_and_rejects_new_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def close_impl() -> None:
        entered.set()
        await release.wait()

    _, instances = install_fake(monkeypatch, module, [ok_response()], close_impl=close_impl)
    client = module.CallControlClient(api_key=API_KEY)
    first = asyncio.create_task(client.aclose())
    await entered.wait()
    second = asyncio.create_task(client.aclose())
    with pytest.raises(module.CallControlClosedError, match="call_control_client_closed"):
        await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)
    release.set()
    await asyncio.gather(first, second)
    await client.aclose()

    assert instances[0].close_count == 1
    assert instances[0].actions.calls == []


@pytest.mark.asyncio
async def test_close_has_one_second_bound_and_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    module = call_control()
    cancelled = asyncio.Event()

    async def never_closes() -> None:
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    _, instances = install_fake(monkeypatch, module, [], close_impl=never_closes)
    client = module.CallControlClient(api_key=API_KEY)
    started = time.monotonic()

    with pytest.raises(module.CallControlCloseError, match="call_control_close_failed") as raised:
        await client.aclose()

    assert 0.9 <= time.monotonic() - started < 1.3
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert cancelled.is_set()
    assert instances[0].close_count == 1
