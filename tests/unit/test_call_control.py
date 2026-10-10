from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import time
from collections import deque
from datetime import UTC, datetime
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
CLIENT_STATE = "RAW-CLIENT-STATE-SENTINEL"
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

    async def stop_recording(self, call_control_id: str, **kwargs: object) -> object:
        return await self._invoke("stop_recording", call_control_id, kwargs)

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
        self.recordings = SimpleNamespace()
        self.close_count = 0
        self._close_impl = close_impl

    async def close(self) -> None:
        self.close_count += 1
        if self._close_impl is not None:
            await self._close_impl()
        http_client = self.constructor_kwargs.get("http_client")
        close = getattr(http_client, "aclose", None)
        if callable(close):
            await close()


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


@pytest.mark.parametrize("region", ["", "eu", "GLOBAL", " EU", "unknown", None, 1, True])
def test_invalid_api_region_never_constructs_client(monkeypatch, region):
    module = call_control()
    constructions = []

    def forbidden_factory(**kwargs):
        constructions.append(kwargs)
        raise AssertionError("invalid region reached a client constructor")

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", forbidden_factory)
    monkeypatch.setattr(module.telnyx, "DefaultAsyncHttpxClient", forbidden_factory)
    with pytest.raises(module.CallControlConfigurationError, match="^call_control_config_invalid$"):
        module.CallControlClient(api_key=API_KEY, api_region=region)
    assert constructions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("region", "host"),
    [(None, "api.telnyx.com"), ("global", "api.telnyx.com"), ("EU", "api.telnyx.eu")],
)
async def test_pinned_sdk_explicit_region_requests_use_exact_host_and_paths(
    monkeypatch, region, host
):
    module = call_control()
    monkeypatch.setenv("TELNYX_BASE_URL", "https://hostile.invalid/override")
    requests = []
    closes = []

    async def handler(request):
        body = json.loads(await request.aread()) if request.method == "POST" else {}
        requests.append((request.method, request.url.scheme, request.url.host, request.url.path,
                         body.get("command_id")))
        if len(requests) == 1:
            return httpx.Response(500, json={"errors": []}, request=request)
        data = {"id": "recording_Ab-12"} if request.method == "DELETE" else {"result": "ok"}
        return httpx.Response(200, json={"data": data}, request=request)

    class ObservedClient(httpx.AsyncClient):
        async def aclose(self):
            closes.append(True)
            await super().aclose()

    http_client = ObservedClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module.telnyx, "DefaultAsyncHttpxClient", lambda **_kwargs: http_client)
    client = None
    try:
        options = {} if region is None else {"api_region": region}
        client = module.CallControlClient(api_key=API_KEY, **options)
        assert (await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)).outcome == "accepted"
        assert (await client.hangup(CALL_CONTROL_ID, command_id=COMMAND_ID)).outcome == "accepted"
        transfer = module.TransferRequestV1(
            to_e164="+33102030406", target_leg_client_state="Zml4dHVyZQ==",
            time_limit_secs=180,
        )
        transferred = await client.transfer(CALL_CONTROL_ID, transfer, command_id=COMMAND_ID)
        assert transferred.outcome == "accepted"
        deleted = await client.delete_recording("recording_Ab-12", timeout_seconds=0.75)
        assert deleted.outcome == "deleted"
        command = str(COMMAND_ID)
        assert requests == [
            ("POST", "https", host, f"/v2/calls/{CALL_CONTROL_ID}/actions/answer", command),
            ("POST", "https", host, f"/v2/calls/{CALL_CONTROL_ID}/actions/answer", command),
            ("POST", "https", host, f"/v2/calls/{CALL_CONTROL_ID}/actions/hangup", command),
            ("POST", "https", host, f"/v2/calls/{CALL_CONTROL_ID}/actions/transfer", command),
            ("DELETE", "https", host, "/v2/recordings/recording_Ab-12", None),
        ]
        await client.aclose()
        await client.aclose()
        assert closes == [True]
    finally:
        if client is not None:
            await client.aclose()
        elif not http_client.is_closed:
            await http_client.aclose()


@pytest.mark.asyncio
async def test_real_pinned_sdk_qualified_transfer_preserves_native_answered_leg_limit(monkeypatch):
    module = call_control()
    requests = []

    async def handler(request):
        requests.append((request.url.path, json.loads(await request.aread())))
        return httpx.Response(200, json={"data": {"result": "ok"}}, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module.telnyx, "DefaultAsyncHttpxClient", lambda **kwargs: http_client)
    client = module.CallControlClient(api_key=API_KEY)
    try:
        assert client.dispatch_available
        request = module.TransferRequestV1(
            to_e164="+33102030406",
            target_leg_client_state="Zml4dHVyZQ==",
            time_limit_secs=180,
        )
        first = await client.transfer(CALL_CONTROL_ID, request, command_id=COMMAND_ID)
        second = await client.transfer(CALL_CONTROL_ID, request, command_id=COMMAND_ID)
        assert first.outcome == second.outcome == "accepted"
        expected = (
            f"/v2/calls/{CALL_CONTROL_ID}/actions/transfer",
            {
                "to": "+33102030406",
                "command_id": str(COMMAND_ID),
                "target_leg_client_state": "Zml4dHVyZQ==",
                "timeout_secs": 20,
                "time_limit_secs": 180,
            },
        )
        assert requests == [expected, expected]
    finally:
        await client.aclose()
    assert not client.dispatch_available


@pytest.mark.parametrize("duration", [None, True, False, "180", 180.0, 180.5, 0, 29, 14401])
def test_transfer_duration_is_a_required_strict_provider_bounded_integer(duration):
    module = call_control()
    with pytest.raises(ValidationError, match="time_limit_secs"):
        module.TransferRequestV1(
            to_e164="+33102030406", target_leg_client_state="Zml4dHVyZQ==",
            time_limit_secs=duration,
        )


def test_transfer_duration_cannot_be_omitted():
    module = call_control()
    with pytest.raises(ValidationError, match="time_limit_secs"):
        module.TransferRequestV1(
            to_e164="+33102030406", target_leg_client_state="Zml4dHVyZQ==",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [30, 14400])
async def test_real_pinned_sdk_transfer_bounds_and_ambiguous_retry_count(monkeypatch, duration):
    module = call_control()
    requests = []

    async def handler(request):
        requests.append(json.loads(await request.aread()))
        return httpx.Response(500, json={"errors": []}, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module.telnyx, "DefaultAsyncHttpxClient", lambda **_kwargs: http_client)
    client = module.CallControlClient(api_key=API_KEY)
    try:
        result = await client.transfer(CALL_CONTROL_ID, module.TransferRequestV1(
            to_e164="+33102030406", target_leg_client_state="Zml4dHVyZQ==",
            time_limit_secs=duration,
        ), command_id=COMMAND_ID)
        assert result.outcome == "outcome_unknown"
        # Wrapper owns its one immediate retry; SDK max_retries=0 adds none.
        assert requests == [{"to": "+33102030406", "command_id": str(COMMAND_ID),
            "target_leg_client_state": "Zml4dHVyZQ==", "timeout_secs": 20,
            "time_limit_secs": duration}] * 2
    finally:
        await client.aclose()


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


def not_found_error() -> telnyx.NotFoundError:
    request = httpx.Request("GET", "https://api.telnyx.com/v2/recordings/RAW-SENTINEL")
    response = httpx.Response(404, request=request, text="RAW-BODY-SENTINEL")
    return telnyx.NotFoundError(
        "RAW-NOT-FOUND-SENTINEL",
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
    recording_stop = inspect.signature(AsyncActionsResource.stop_recording)
    streaming = inspect.signature(AsyncActionsResource.start_streaming)
    assert "max_length" in recording.parameters
    assert "recording_id" in recording_stop.parameters
    assert "stream_auth_token" in streaming.parameters
    assert inspect.signature(telnyx.AsyncTelnyx).parameters["max_retries"].default == 2


@pytest.mark.asyncio
async def test_real_sdk_serializes_exact_paths_bodies_and_omissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    requests: list[tuple[str, str, dict[str, object], bytes]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads((await request.aread()).decode("utf-8"))
        requests.append((request.method, request.url.path, body, request.url.query))
        return httpx.Response(200, json={"data": {"result": "ok"}}, request=request)

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    constructed: list[dict[str, object]] = []

    def transport_factory(**kwargs: object) -> httpx.AsyncClient:
        constructed.append(dict(kwargs))
        return http_client

    monkeypatch.setattr(
        module.telnyx,
        "DefaultAsyncHttpxClient",
        transport_factory,
    )
    client = module.CallControlClient(api_key=API_KEY)
    await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)
    await client.start_streaming(
        CALL_CONTROL_ID,
        module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN),
        command_id=COMMAND_ID,
    )
    await client.start_recording(
        CALL_CONTROL_ID,
        module.RecordingStartV1(
            play_beep=False,
            client_state=SecretStr(CLIENT_STATE),
        ),
        command_id=COMMAND_ID,
    )
    await client.stop_recording(
        CALL_CONTROL_ID,
        module.RecordingStopV1(
            client_state=SecretStr(CLIENT_STATE),
            recording_id=None,
        ),
        command_id=COMMAND_ID,
    )
    await client.hangup(
        CALL_CONTROL_ID,
        command_id=COMMAND_ID,
        client_state=SecretStr(CLIENT_STATE),
    )
    await client.aclose()

    assert constructed == [{"trust_env": False}]

    command = str(COMMAND_ID)
    assert [(method, path, query) for method, path, _, query in requests] == [
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/answer", b""),
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/streaming_start", b""),
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/record_start", b""),
        ("POST", f"/v2/calls/{CALL_CONTROL_ID}/actions/record_stop", b""),
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
            "client_state": CLIENT_STATE,
            "command_id": command,
            "format": "wav",
            "max_length": 0,
            "play_beep": False,
            "recording_track": "both",
            "timeout_secs": 0,
            "transcription": False,
        },
        {"client_state": CLIENT_STATE, "command_id": command},
        {"client_state": CLIENT_STATE, "command_id": command},
    ]


def test_request_models_are_strict_frozen_and_input_redacting() -> None:
    module = call_control()
    streaming = module.StreamingStartV1(stream_url=URL, stream_auth_token=TOKEN)
    recording = module.RecordingStartV1(
        play_beep=True,
        client_state=SecretStr(CLIENT_STATE),
    )

    assert streaming.stream_url == URL
    assert isinstance(streaming.stream_auth_token, SecretStr)
    assert streaming.stream_auth_token.get_secret_value() == TOKEN
    assert recording.play_beep is True
    assert URL not in repr(streaming)
    assert TOKEN not in repr(streaming)
    with pytest.raises(ValidationError, match="frozen"):
        streaming.stream_url = "wss://changed.example.test"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        module.RecordingStartV1(
            play_beep=True,
            client_state=SecretStr(CLIENT_STATE),
            trim="trim-silence",
        )


def test_recording_stop_provider_id_uses_shared_url_safe_opaque_contract() -> None:
    module = call_control()
    provider_id = "r." + "A" * 251 + "~_-"

    accepted = module.RecordingStopV1(
        client_state=SecretStr(CLIENT_STATE),
        recording_id=provider_id,
    )
    assert accepted.recording_id == provider_id

    with pytest.raises(ValidationError, match="recording_id"):
        module.RecordingStopV1(
            client_state=SecretStr(CLIENT_STATE),
            recording_id="segment/child",
        )


@pytest.mark.parametrize("provider_id", [".", ".."])
def test_recording_stop_rejects_exact_dot_segments(provider_id: str) -> None:
    module = call_control()

    with pytest.raises(ValidationError, match="recording_id"):
        module.RecordingStopV1(
            client_state=SecretStr(CLIENT_STATE),
            recording_id=provider_id,
        )


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
    recording = module.RecordingStartV1(
        play_beep=True,
        client_state=SecretStr(CLIENT_STATE),
    )

    results = [
        await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID),
        await client.start_streaming(CALL_CONTROL_ID, streaming, command_id=COMMAND_ID),
        await client.start_recording(CALL_CONTROL_ID, recording, command_id=COMMAND_ID),
        await client.hangup(CALL_CONTROL_ID, command_id=COMMAND_ID),
    ]

    assert [result.outcome for result in results] == ["accepted"] * 4
    assert len(constructions) == 1
    assert constructions[0]["api_key"] == API_KEY
    assert constructions[0]["max_retries"] == 0
    transport = constructions[0]["http_client"]
    assert isinstance(transport, telnyx.DefaultAsyncHttpxClient)
    assert transport._mounts == {}  # noqa: SLF001
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
        "client_state": CLIENT_STATE,
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
async def test_recording_actions_send_one_redacted_capsule_and_omit_unknown_recording_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    _, instances = install_fake(monkeypatch, module, [ok_response()] * 3)
    client = module.CallControlClient(api_key=API_KEY)
    capsule = SecretStr(CLIENT_STATE)

    await client.start_recording(
        CALL_CONTROL_ID,
        module.RecordingStartV1(play_beep=False, client_state=capsule),
        command_id=COMMAND_ID,
    )
    await client.stop_recording(
        CALL_CONTROL_ID,
        module.RecordingStopV1(client_state=capsule, recording_id=None),
        command_id=COMMAND_ID,
    )
    await client.hangup(
        CALL_CONTROL_ID,
        command_id=COMMAND_ID,
        client_state=capsule,
    )

    calls = instances[0].actions.calls
    assert [call[:2] for call in calls] == [
        ("start_recording", CALL_CONTROL_ID),
        ("stop_recording", CALL_CONTROL_ID),
        ("hangup", CALL_CONTROL_ID),
    ]
    assert calls[0][2]["client_state"] == CLIENT_STATE
    assert calls[1][2]["client_state"] == CLIENT_STATE
    assert "recording_id" not in calls[1][2]
    assert calls[2][2]["client_state"] == CLIENT_STATE
    request = module.RecordingStopV1(client_state=capsule, recording_id=None)
    assert request.model_dump() == {}
    assert CLIENT_STATE not in repr(request) + str(request)


@pytest.mark.asyncio
async def test_recording_catalog_adapter_awaits_exactly_one_page_and_never_reads_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    await_count = 0

    class PoisonRow:
        id = "recording_Ab-12"
        call_control_id = CALL_CONTROL_ID
        call_leg_id = "leg-1"
        call_session_id = "session-1"
        channels = "dual"
        status = "completed"
        source = "call"
        initiated_by = "StartCallRecordingAPI"
        recording_started_at = "2026-08-29T12:00:00Z"
        recording_ended_at = "2026-08-29T12:01:00Z"

        @property
        def download_urls(self) -> object:
            raise AssertionError("recording URL must never be read")

    page = SimpleNamespace(
        meta=SimpleNamespace(page_number=1, total_pages=1),
        data=[PoisonRow()],
    )

    class OnePageAwaitable:
        def __await__(self):  # type: ignore[no-untyped-def]
            nonlocal await_count
            await_count += 1
            if await_count != 1:
                raise AssertionError("paginator awaited more than once")
            if False:
                yield None
            return page

        def __aiter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("paginator iteration is forbidden")

        async def get_next_page(self) -> object:
            raise AssertionError("next page is forbidden")

    class FakeRecordings:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def list(self, **kwargs: object) -> OnePageAwaitable:
            self.calls.append(dict(kwargs))
            return OnePageAwaitable()

    class CatalogSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    sdk = CatalogSDK()
    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: sdk)
    client = module.CallControlClient(api_key=API_KEY)

    result = await client.list_recordings_one_page(
        call_control_id=CALL_CONTROL_ID,
        call_leg_id="leg-1",
        call_session_id="session-1",
        start_gte_iso="2026-08-29T11:59:55Z",
        start_lte_iso="2026-08-29T12:00:05Z",
        end_gte_iso="2026-08-29T12:00:55Z",
        end_lte_iso="2026-08-29T12:01:05Z",
        timeout_seconds=0.75,
    )

    assert await_count == 1
    assert sdk.recordings.calls == [
        {
            "filter": {
                "call_control_id": CALL_CONTROL_ID,
                "call_leg_id": "leg-1",
                "call_session_id": "session-1",
                "start_time": {
                    "gte": "2026-08-29T11:59:55Z",
                    "lte": "2026-08-29T12:00:05Z",
                },
                "end_time": {
                    "gte": "2026-08-29T12:00:55Z",
                    "lte": "2026-08-29T12:01:05Z",
                },
            },
            "page_size": 2,
            "timeout": 0.75,
        }
    ]
    assert result.page_number == 1
    assert result.total_pages == 1
    assert len(result.items) == 1
    assert result.items[0].recording_id == "recording_Ab-12"
    assert result.items[0].recording_started_at == datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [257, 1024])
async def test_catalog_list_and_retrieve_preserve_long_call_control_id(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
) -> None:
    module = call_control()
    call_control_id = "c" * length
    row = SimpleNamespace(
        id="recording_Ab-12",
        call_control_id=call_control_id,
        call_leg_id="leg-1",
        call_session_id="session-1",
        channels="dual",
        status="completed",
        source="call",
        initiated_by="StartCallRecordingAPI",
        recording_started_at="2026-08-29T12:00:00Z",
        recording_ended_at="2026-08-29T12:01:00Z",
    )
    page = SimpleNamespace(
        meta=SimpleNamespace(page_number=1, total_pages=1),
        data=[row],
    )

    class OnePageAwaitable:
        def __await__(self):  # type: ignore[no-untyped-def]
            if False:
                yield None
            return page

    class FakeRecordings:
        def __init__(self) -> None:
            self.list_calls: list[dict[str, object]] = []

        def list(self, **kwargs: object) -> OnePageAwaitable:
            self.list_calls.append(dict(kwargs))
            return OnePageAwaitable()

        async def retrieve(self, recording_id: str, **kwargs: object) -> object:
            assert recording_id == "recording_Ab-12"
            assert kwargs == {"timeout": 0.75}
            return SimpleNamespace(data=row)

    class CatalogSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    sdk = CatalogSDK()
    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: sdk)
    client = module.CallControlClient(api_key=API_KEY)

    listed = await client.list_recordings_one_page(
        call_control_id=call_control_id,
        call_leg_id="leg-1",
        call_session_id="session-1",
        start_gte_iso="2026-08-29T11:59:55Z",
        start_lte_iso="2026-08-29T12:00:05Z",
        end_gte_iso="2026-08-29T12:00:55Z",
        end_lte_iso="2026-08-29T12:01:05Z",
        timeout_seconds=0.75,
    )
    retrieved = await client.retrieve_recording(
        "recording_Ab-12",
        timeout_seconds=0.75,
    )

    assert listed.items[0].call_control_id == call_control_id
    assert retrieved.call_control_id == call_control_id
    assert sdk.recordings.list_calls[0]["filter"]["call_control_id"] == call_control_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_leg_id", "call_session_id"),
    [("", None), (None, "s" * 257)],
)
async def test_recording_list_rejects_invalid_optional_ids_without_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    call_leg_id: str | None,
    call_session_id: str | None,
) -> None:
    module = call_control()

    class ForbiddenRecordings:
        def list(self, **_: object) -> None:
            raise AssertionError("recording list must not dispatch")

    class CatalogSDK:
        def __init__(self) -> None:
            self.recordings = ForbiddenRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: CatalogSDK())
    client = module.CallControlClient(api_key=API_KEY)

    with pytest.raises(module.CallControlInputError, match="call_control_input_invalid"):
        await client.list_recordings_one_page(
            call_control_id=CALL_CONTROL_ID,
            call_leg_id=call_leg_id,
            call_session_id=call_session_id,
            start_gte_iso="2026-08-29T11:59:55Z",
            start_lte_iso="2026-08-29T12:00:05Z",
            end_gte_iso="2026-08-29T12:00:55Z",
            end_lte_iso="2026-08-29T12:01:05Z",
            timeout_seconds=0.75,
        )


@pytest.mark.asyncio
async def test_catalog_list_and_retrieve_reject_returned_call_control_id_over_1024(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    row = SimpleNamespace(
        id="recording_Ab-12",
        call_control_id="c" * 1025,
        call_leg_id="leg-1",
        call_session_id="session-1",
        channels="dual",
        status="completed",
        source="call",
        initiated_by="StartCallRecordingAPI",
        recording_started_at="2026-08-29T12:00:00Z",
        recording_ended_at="2026-08-29T12:01:00Z",
    )
    page = SimpleNamespace(
        meta=SimpleNamespace(page_number=1, total_pages=1),
        data=[row],
    )

    class OnePageAwaitable:
        def __await__(self):  # type: ignore[no-untyped-def]
            if False:
                yield None
            return page

    class FakeRecordings:
        def list(self, **_: object) -> OnePageAwaitable:
            return OnePageAwaitable()

        async def retrieve(self, *_: object, **__: object) -> object:
            return SimpleNamespace(data=row)

    class CatalogSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: CatalogSDK())
    client = module.CallControlClient(api_key=API_KEY)

    with pytest.raises(module.RecordingCatalogInvalidError):
        await client.list_recordings_one_page(
            call_control_id=CALL_CONTROL_ID,
            call_leg_id="leg-1",
            call_session_id="session-1",
            start_gte_iso="2026-08-29T11:59:55Z",
            start_lte_iso="2026-08-29T12:00:05Z",
            end_gte_iso="2026-08-29T12:00:55Z",
            end_lte_iso="2026-08-29T12:01:05Z",
            timeout_seconds=0.75,
        )
    with pytest.raises(module.RecordingCatalogInvalidError):
        await client.retrieve_recording("recording_Ab-12", timeout_seconds=0.75)


@pytest.mark.asyncio
async def test_recording_retrieve_404_is_transient_not_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()

    class FakeRecordings:
        async def retrieve(self, recording_id: str, **kwargs: object) -> object:
            del recording_id, kwargs
            raise not_found_error()

    class RetrieveSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    sdk = RetrieveSDK()
    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: sdk)
    client = module.CallControlClient(api_key=API_KEY)

    with pytest.raises(
        module.RecordingCatalogTransientError,
        match="recording_catalog_transient",
    ) as raised:
        await client.retrieve_recording("recording_Ab-12", timeout_seconds=0.75)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "RAW-BODY-SENTINEL" not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", [".", ".."])
async def test_recording_retrieve_and_delete_reject_dot_segments_without_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
) -> None:
    module = call_control()
    calls: list[str] = []

    class ForbiddenRecordings:
        async def retrieve(self, *_: object, **__: object) -> object:
            calls.append("retrieve")
            raise AssertionError("retrieve must not dispatch")

        async def delete(self, *_: object, **__: object) -> object:
            calls.append("delete")
            raise AssertionError("delete must not dispatch")

    class DotSegmentSDK:
        def __init__(self) -> None:
            self.recordings = ForbiddenRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: DotSegmentSDK())
    client = module.CallControlClient(api_key=API_KEY)

    with pytest.raises(module.CallControlInputError, match="call_control_input_invalid"):
        await client.retrieve_recording(provider_id, timeout_seconds=0.75)
    with pytest.raises(module.CallControlInputError, match="call_control_input_invalid"):
        await client.delete_recording(provider_id, timeout_seconds=0.75)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "error_name"),
    [
        ("provider_status", "RecordingCatalogTransientError"),
        ("invalid_datetime", "RecordingCatalogInvalidError"),
        ("poison_scalar", "RecordingCatalogInvalidError"),
    ],
)
async def test_recording_list_errors_discard_provider_exception_graph(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    error_name: str,
) -> None:
    module = call_control()

    class PoisonRow:
        id = "recording_Ab-12"
        call_leg_id = "leg-1"
        call_session_id = "session-1"
        channels = "dual"
        status = "completed"
        source = "call"
        initiated_by = "StartCallRecordingAPI"
        recording_started_at = "2026-08-29T12:00:00Z"
        recording_ended_at = "2026-08-29T12:01:00Z"

        @property
        def call_control_id(self) -> str:
            if scenario == "poison_scalar":
                raise RuntimeError("RAW-SCALAR-SENTINEL")
            return CALL_CONTROL_ID

    row = PoisonRow()
    if scenario == "invalid_datetime":
        row.recording_started_at = "RAW-DATETIME-SENTINEL"
    page = SimpleNamespace(
        meta=SimpleNamespace(page_number=1, total_pages=1),
        data=[row],
    )

    class OnePageAwaitable:
        def __await__(self):  # type: ignore[no-untyped-def]
            if False:
                yield None
            return page

    class FakeRecordings:
        def list(self, **kwargs: object) -> OnePageAwaitable:
            del kwargs
            if scenario == "provider_status":
                raise status_error(500)
            return OnePageAwaitable()

    class ListSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: ListSDK())
    client = module.CallControlClient(api_key=API_KEY)
    error_type = getattr(module, error_name)

    with pytest.raises(error_type) as raised:
        await client.list_recordings_one_page(
            call_control_id=CALL_CONTROL_ID,
            call_leg_id="leg-1",
            call_session_id="session-1",
            start_gte_iso="2026-08-29T11:59:55Z",
            start_lte_iso="2026-08-29T12:00:05Z",
            end_gte_iso="2026-08-29T12:00:55Z",
            end_lte_iso="2026-08-29T12:01:05Z",
            timeout_seconds=0.75,
        )

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    rendered = repr(raised.value)
    for sentinel in (
        "RAW-BODY-SENTINEL",
        "RAW-SCALAR-SENTINEL",
        "RAW-DATETIME-SENTINEL",
    ):
        assert sentinel not in rendered


@pytest.mark.asyncio
async def test_recording_delete_extracts_only_exact_id_and_maps_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()

    class FakeRecordings:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.responses: deque[object] = deque(
                [
                    SimpleNamespace(data=SimpleNamespace(id="recording_Ab-12")),
                    status_error(404),
                ]
            )

        async def delete(self, recording_id: str, **kwargs: object) -> object:
            self.calls.append((recording_id, dict(kwargs)))
            result = self.responses.popleft()
            if isinstance(result, BaseException):
                raise result
            return result

    class DeleteSDK:
        def __init__(self) -> None:
            self.recordings = FakeRecordings()
            self.calls = SimpleNamespace(actions=FakeActions([]))

        async def close(self) -> None:
            return None

    sdk = DeleteSDK()
    monkeypatch.setattr(module.telnyx, "AsyncTelnyx", lambda **_: sdk)
    client = module.CallControlClient(api_key=API_KEY)

    deleted = await client.delete_recording("recording_Ab-12", timeout_seconds=0.75)
    missing = await client.delete_recording("recording_Ab-12", timeout_seconds=0.75)

    assert deleted.outcome == "deleted"
    assert deleted.recording_id == "recording_Ab-12"
    assert missing.outcome == "not_found"
    assert missing.recording_id is None
    assert sdk.recordings.calls == [
        ("recording_Ab-12", {"timeout": 0.75}),
        ("recording_Ab-12", {"timeout": 0.75}),
    ]


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
async def test_cancelled_close_stays_closing_and_later_close_retries_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = call_control()
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    attempts = 0

    async def close_impl() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    _, instances = install_fake(monkeypatch, module, [], close_impl=close_impl)
    client = module.CallControlClient(api_key=API_KEY)
    interrupted = asyncio.create_task(client.aclose())
    await entered.wait()
    interrupted.cancel()

    with pytest.raises(asyncio.CancelledError):
        await interrupted
    assert cancelled.is_set()
    with pytest.raises(module.CallControlClosedError, match="call_control_client_closed"):
        await client.answer(CALL_CONTROL_ID, command_id=COMMAND_ID)

    await client.aclose()

    assert instances[0].close_count == 2
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
