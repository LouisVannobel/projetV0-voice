from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pipecat.runner.types import TelnyxCallData
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from starlette.websockets import WebSocket, WebSocketState

import projetv0_voice.telnyx.handshake as handshake_module
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.telnyx.handshake import (
    AuthenticatedTelnyxHandshakeService,
    TelnyxHandshakeCapacityError,
    TelnyxHandshakeError,
    TelnyxHandshakeRejectedError,
    TelnyxHandshakeTimeoutError,
)
from projetv0_voice.telnyx.serializer import ProjetV0TelnyxFrameSerializer

TOKEN = "synthetic-auth-token"
CALL_CONTROL_ID = "sensitive-call-control-id"
STREAM_ID = "sensitive-stream-id"
FROM_NUMBER = "+33111111111"
TO_NUMBER = "+33222222222"
LOCATOR_ID = "telnyx-header-connected-v1"
EXPECTED_CANONICAL_FIXTURE = (
    b'{"authentication":{"connected.connected.x-telnyx-streaming-auth-token":'
    b'{"type":"string","utf8_length":20},"header.x-telnyx-streaming-auth-token":'
    b'{"type":"string","utf8_length":20}},"connected":{"event":"connected",'
    b'"version":"1.0.0"},"provider":"telnyx","schema":'
    b'"projetv0.telnyx.handshake.v1","start":{"event":"start",'
    b'"sequence_number":{"format":"decimal","type":"string"},'
    b'"start.call_control_id":{"type":"string"},"start.from":{"type":"string"},'
    b'"start.media_format.channels":1,"start.media_format.encoding":"PCMU",'
    b'"start.media_format.sample_rate":8000,"start.to":{"type":"string"},'
    b'"stream_id":{"type":"string"}},"token_locator_id":'
    b'"telnyx-header-connected-v1"}'
)
EXPECTED_CANONICAL_SHA256 = "6ef76b42d6f0db4a60fdeda1e6c4a202363674e4eea3d04a82977cba9ef6da80"


@dataclass(frozen=True)
class _Claim:
    secret_note: str = "lease-secret-note"


class _Permit:
    def __init__(self) -> None:
        self.release_count = 0

    def release(self) -> None:
        self.release_count += 1


class _RaisingPermit(_Permit):
    def release(self) -> None:
        super().release()
        raise RuntimeError("permit-provider-secret")


class _Gate:
    def __init__(self, permit: _Permit | None) -> None:
        self.permit = permit
        self.acquire_count = 0

    def try_acquire(self) -> _Permit | None:
        self.acquire_count += 1
        return self.permit


class _RaisingGate:
    def try_acquire(self) -> None:
        raise RuntimeError("gate-provider-secret")


class _CancellingGate:
    def try_acquire(self) -> None:
        raise asyncio.CancelledError


class _LeaseAuthority:
    def __init__(
        self,
        *,
        expected_digest: bytes,
        expected_call_control_id: str = CALL_CONTROL_ID,
        expired: bool = False,
    ) -> None:
        self.expected_digest = expected_digest
        self.expected_call_control_id = expected_call_control_id
        self.expired = expired
        self.claimed = False
        self.claim_calls: list[tuple[str, bytes]] = []
        self.abort_calls: list[tuple[str, bytes]] = []
        self.claim = _Claim()

    async def claim_once(self, *, call_control_id: str, token_digest: bytes) -> _Claim | None:
        self.claim_calls.append((call_control_id, token_digest))
        if (
            self.expired
            or self.claimed
            or call_control_id != self.expected_call_control_id
            or token_digest != self.expected_digest
        ):
            return None
        self.claimed = True
        return self.claim

    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        self.abort_calls.append((call_control_id, token_digest))


class _BlockingLeaseAuthority(_LeaseAuthority):
    def __init__(self, *, expected_digest: bytes) -> None:
        super().__init__(expected_digest=expected_digest)
        self.started = asyncio.Event()
        self.never = asyncio.Event()

    async def claim_once(self, *, call_control_id: str, token_digest: bytes) -> _Claim | None:
        self.claim_calls.append((call_control_id, token_digest))
        self.claimed = True
        self.started.set()
        await self.never.wait()
        return self.claim


class _RaisingAbortAuthority(_LeaseAuthority):
    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        super().schedule_abort_if_matches(
            call_control_id=call_control_id, token_digest=token_digest
        )
        raise RuntimeError("scheduler-provider-secret")


class _RaisingBlockingLeaseAuthority(_BlockingLeaseAuthority):
    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        super().schedule_abort_if_matches(
            call_control_id=call_control_id, token_digest=token_digest
        )
        raise RuntimeError("scheduler-provider-secret")


def _token_digest(token: str = TOKEN) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _connected(token: str = TOKEN, **extras: object) -> str:
    return json.dumps(
        {
            "protocol": "Call",
            "version": "1.0.0",
            "event": "connected",
            "connected": {"x-telnyx-streaming-auth-token": token},
            **extras,
        }
    )


def _root_connected(token: str = TOKEN) -> str:
    return json.dumps(
        {
            "event": "connected",
            "version": "1.0.0",
            "x-telnyx-streaming-auth-token": token,
        }
    )


def _start(**start_extras: object) -> str:
    return json.dumps(
        {
            "stream_id": STREAM_ID,
            "event": "start",
            "sequence_number": "1",
            "start": {
                "call_control_id": CALL_CONTROL_ID,
                "from": FROM_NUMBER,
                "to": TO_NUMBER,
                "media_format": {
                    "encoding": "PCMU",
                    "sample_rate": 8000,
                    "channels": 1,
                },
                **start_extras,
            },
            "documented_extra": "ignored",
        }
    )


def _profile() -> QualifiedDeploymentProfileV1:
    original = QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(
            encoding="utf-8"
        )
    )
    return original.model_copy(
        update={
            "telnyx_handshake_fixture_sha256": EXPECTED_CANONICAL_SHA256
        }
    )


def _websocket(
    *,
    header_values: tuple[tuple[bytes, bytes], ...] = (
        (b"x-telnyx-streaming-auth-token", TOKEN.encode("utf-8")),
    ),
    messages: tuple[str | dict[str, Any], ...] = (_connected(), _start()),
    query_string: bytes = b"stream_auth_token=query-guess-is-ignored",
) -> tuple[WebSocket, dict[str, int]]:
    queued = list(messages)
    state = {"receive_count": 0}

    async def receive() -> dict[str, Any]:
        state["receive_count"] += 1
        if not queued:
            return {"type": "websocket.disconnect", "code": 1000}
        item = queued.pop(0)
        if isinstance(item, dict):
            return item
        return {"type": "websocket.receive", "text": item}

    async def send(_message: dict[str, Any]) -> None:
        return None

    scope: dict[str, Any] = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "wss",
        "server": ("voice.invalid", 443),
        "client": ("127.0.0.1", 12345),
        "root_path": "",
        "path": "/telnyx/media",
        "raw_path": b"/telnyx/media",
        "query_string": query_string,
        "headers": list(header_values),
        "subprotocols": [],
    }
    websocket = WebSocket(scope, receive=receive, send=send)
    websocket.application_state = WebSocketState.CONNECTED
    websocket.client_state = WebSocketState.CONNECTED
    return websocket, state


def _service(
    authority: _LeaseAuthority,
    gate: _Gate,
    *,
    profile: QualifiedDeploymentProfileV1 | None = None,
    timeout_seconds: float = 0.5,
) -> AuthenticatedTelnyxHandshakeService:
    return AuthenticatedTelnyxHandshakeService(
        profile=profile or _profile(),
        lease_authority=authority,
        unauthenticated_gate=gate,
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.parametrize("timeout_seconds", [float("inf"), float("nan")])
def test_nonfinite_global_deadline_is_rejected(timeout_seconds: float) -> None:
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeError) as raised:
        _service(authority, _Gate(_Permit()), timeout_seconds=timeout_seconds)

    assert str(raised.value) == "telnyx_handshake_config_invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_authenticate_builds_exact_native_call_data_and_transport_after_claim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    websocket, state = _websocket(
        header_values=((b"X-Telnyx-Streaming-Auth-Token", TOKEN.encode("utf-8")),),
        messages=(
            _connected(TOKEN, documented_object={"type": "streaming"}),
            _start(stream_auth_token="start-guess-is-ignored"),
        ),
    )
    permit = _Permit()
    gate = _Gate(permit)
    authority = _LeaseAuthority(expected_digest=_token_digest())

    result = await _service(authority, gate).authenticate(websocket)

    assert state["receive_count"] == 2
    assert hashlib.sha256(EXPECTED_CANONICAL_FIXTURE).hexdigest() == EXPECTED_CANONICAL_SHA256
    assert authority.claim_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert authority.abort_calls == []
    assert gate.acquire_count == 1
    assert permit.release_count == 1
    assert isinstance(result.call_data, TelnyxCallData)
    assert result.call_data.stream_id == STREAM_ID
    assert result.call_data.call_id == CALL_CONTROL_ID
    assert result.call_data.outbound_encoding == "PCMU"
    assert result.call_data.from_number == FROM_NUMBER
    assert result.call_data.to_number == TO_NUMBER
    assert result.token_locator_id == LOCATOR_ID
    assert result.lease_claim is authority.claim
    assert isinstance(result.transport, FastAPIWebsocketTransport)
    assert result.transport._client._websocket is websocket
    params = result.transport._params
    assert params.audio_in_enabled is True
    assert params.audio_out_enabled is True
    assert params.add_wav_header is False
    assert isinstance(params.serializer, ProjetV0TelnyxFrameSerializer)
    assert result.audio_admission.is_bound is False
    assert params.serializer.audio_admission is result.audio_admission
    assert params.serializer._stream_id == STREAM_ID
    assert params.serializer._expected_call_control_id == CALL_CONTROL_ID
    assert params.serializer._call_control_id is None
    assert params.serializer._api_key is None
    rendered = repr(result)
    for secret in (
        TOKEN,
        STREAM_ID,
        CALL_CONTROL_ID,
        FROM_NUMBER,
        TO_NUMBER,
        "voice.invalid",
        authority.claim.secret_note,
    ):
        assert secret not in rendered
        assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,connected",
    [
        ((), _connected()),
        (
            (
                (b"x-telnyx-streaming-auth-token", TOKEN.encode()),
                (b"X-Telnyx-Streaming-Auth-Token", TOKEN.encode()),
            ),
            _connected(),
        ),
        (
            ((b"x-telnyx-streaming-auth-token", TOKEN.encode()),),
            _connected("different-connected-secret"),
        ),
        (((b"x-telnyx-streaming-auth-token", b""),), _connected("")),
        (
            ((b"x-telnyx-streaming-auth-token", ("t" * 4001).encode()),),
            _connected("t" * 4001),
        ),
        (
            ((b"x-telnyx-streaming-auth-token", TOKEN.encode()),),
            _root_connected(),
        ),
    ],
)
async def test_composite_token_contract_fails_closed_before_claim(
    headers: tuple[tuple[bytes, bytes], ...], connected: str
) -> None:
    websocket, _ = _websocket(header_values=headers, messages=(connected, _start()))
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeRejectedError) as raised:
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert authority.claim_calls == []
    assert authority.abort_calls == []
    assert permit.release_count == 1
    assert str(raised.value) == "telnyx_handshake_rejected"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connected,start",
    [
        (_start(), _connected()),
        (
            _connected(),
            _start(
                media_format={"encoding": "PCMA", "sample_rate": 8000, "channels": 1}
            ),
        ),
        (_connected(), _start(sequence_number="not-in-start-is-ignored")),
        (_connected(), _start()),
    ],
)
async def test_strict_message_contract_rejects_order_media_and_duplicate_keys(
    connected: str, start: str
) -> None:
    if connected == _connected() and start == _start():
        start = (
            '{"event":"start","event":"start","sequence_number":"1",'
            f'"stream_id":"{STREAM_ID}","start":{{"call_control_id":"{CALL_CONTROL_ID}",'
            '"media_format":{"encoding":"PCMU","sample_rate":8000,"channels":1}}}'
            "}"
        )
    elif "not-in-start-is-ignored" in start:
        parsed = json.loads(start)
        parsed["sequence_number"] = "not-decimal"
        start = json.dumps(parsed)
    websocket, _ = _websocket(messages=(connected, start))
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeRejectedError):
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert authority.claim_calls == []
    assert permit.release_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_message",
    [
        (
            '{"event":"connected","version":"1.0.0",'
            '"connected":{'
            f'"x-telnyx-streaming-auth-token":"{TOKEN}",'
            f'"x-telnyx-streaming-auth-token":"{TOKEN}"}}}}'
        ),
        json.dumps(
            {
                "event": "connected",
                "version": "1.0.0",
                "connected": {"x-telnyx-streaming-auth-token": TOKEN},
                "oversized_extra": "x" * 65_536,
            }
        ),
        {"type": "websocket.receive", "bytes": _connected().encode("utf-8")},
        {"type": "websocket.disconnect", "code": 1000},
    ],
    ids=["duplicate-connected-token", "oversized", "binary", "close"],
)
async def test_duplicate_oversized_binary_and_close_fail_before_claim(
    first_message: str | dict[str, Any]
) -> None:
    websocket, _ = _websocket(messages=(first_message, _start()))
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeRejectedError):
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert authority.claim_calls == []
    assert authority.abort_calls == []
    assert permit.release_count == 1


@pytest.mark.asyncio
async def test_fixture_drift_and_authority_rejection_do_not_construct_transport() -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest(), expired=True)
    drifted = _profile().model_copy(
        update={"telnyx_handshake_fixture_sha256": "0" * 64}
    )

    with pytest.raises(TelnyxHandshakeRejectedError):
        await _service(authority, _Gate(permit), profile=drifted).authenticate(websocket)
    assert authority.claim_calls == []
    assert authority.abort_calls == []

    websocket, _ = _websocket()
    with pytest.raises(TelnyxHandshakeRejectedError):
        await _service(authority, _Gate(permit)).authenticate(websocket)
    assert authority.claim_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert authority.abort_calls == []
    assert permit.release_count == 2


@pytest.mark.asyncio
async def test_capacity_rejection_is_synchronous_and_consumes_nothing() -> None:
    websocket, state = _websocket()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeCapacityError) as raised:
        await _service(authority, _Gate(None)).authenticate(websocket)

    assert state["receive_count"] == 0
    assert authority.claim_calls == []
    assert str(raised.value) == "telnyx_handshake_capacity"


@pytest.mark.asyncio
async def test_gate_provider_exception_is_constant_safe_without_context() -> None:
    websocket, state = _websocket()
    authority = _LeaseAuthority(expected_digest=_token_digest())
    service = AuthenticatedTelnyxHandshakeService(
        profile=_profile(),
        lease_authority=authority,
        unauthenticated_gate=_RaisingGate(),
        timeout_seconds=0.5,
    )

    with pytest.raises(TelnyxHandshakeError) as raised:
        await service.authenticate(websocket)

    assert state["receive_count"] == 0
    assert authority.claim_calls == []
    assert str(raised.value) == "telnyx_handshake_gate_failed"
    assert "gate-provider-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_gate_cancellation_is_never_wrapped() -> None:
    authority = _LeaseAuthority(expected_digest=_token_digest())
    service = AuthenticatedTelnyxHandshakeService(
        profile=_profile(),
        lease_authority=authority,
        unauthenticated_gate=_CancellingGate(),
        timeout_seconds=0.5,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.authenticate(_websocket()[0])


@pytest.mark.asyncio
async def test_global_timeout_before_claim_releases_permit_without_abort() -> None:
    async def never_receive() -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(_message: dict[str, Any]) -> None:
        return None

    websocket, _ = _websocket()
    websocket._receive = never_receive
    websocket._send = send
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeTimeoutError) as raised:
        await _service(authority, _Gate(permit), timeout_seconds=0.01).authenticate(websocket)

    assert permit.release_count == 1
    assert authority.claim_calls == []
    assert authority.abort_calls == []
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_cancellation_during_unknown_claim_schedules_abort_and_releases() -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _BlockingLeaseAuthority(expected_digest=_token_digest())
    task = asyncio.create_task(_service(authority, _Gate(permit)).authenticate(websocket))
    await authority.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert permit.release_count == 1
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]


@pytest.mark.asyncio
async def test_timeout_during_unknown_claim_schedules_abort_and_releases() -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _BlockingLeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeTimeoutError):
        await _service(authority, _Gate(permit), timeout_seconds=0.01).authenticate(websocket)

    assert permit.release_count == 1
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]


@pytest.mark.asyncio
async def test_cancellation_after_claim_return_before_handoff_schedules_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    def cancel_transport(*_args: object, **_kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(handshake_module, "FastAPIWebsocketTransport", cancel_transport)

    with pytest.raises(asyncio.CancelledError):
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert authority.claim_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert permit.release_count == 1


@pytest.mark.asyncio
async def test_failure_after_claim_schedules_abort_before_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    def fail_transport(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("provider-url-sensitive")

    monkeypatch.setattr(handshake_module, "FastAPIWebsocketTransport", fail_transport)

    with pytest.raises(TelnyxHandshakeError) as raised:
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert authority.claim_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert permit.release_count == 1
    assert "provider-url-sensitive" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_raising_abort_scheduler_cannot_skip_permit_release_or_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _RaisingAbortAuthority(expected_digest=_token_digest())

    def fail_transport(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("transport-provider-secret")

    monkeypatch.setattr(handshake_module, "FastAPIWebsocketTransport", fail_transport)

    with pytest.raises(TelnyxHandshakeError) as raised:
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert permit.release_count == 1
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert str(raised.value) == "telnyx_handshake_cleanup_failed"
    assert "scheduler-provider-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_raising_abort_scheduler_cannot_mask_external_cancellation() -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _RaisingBlockingLeaseAuthority(expected_digest=_token_digest())
    task = asyncio.create_task(_service(authority, _Gate(permit)).authenticate(websocket))
    await authority.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert permit.release_count == 1
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]


@pytest.mark.asyncio
async def test_raising_permit_release_aborts_unreturned_handoff_constant_safely() -> None:
    websocket, _ = _websocket()
    permit = _RaisingPermit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeError) as raised:
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert permit.release_count == 1
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert str(raised.value) == "telnyx_handshake_cleanup_failed"
    assert "permit-provider-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_release_cleanup_failure_takes_precedence_over_stored_rejection() -> None:
    websocket, _ = _websocket(messages=(_root_connected(), _start()))
    permit = _RaisingPermit()
    authority = _LeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeError) as raised:
        await _service(authority, _Gate(permit)).authenticate(websocket)

    assert permit.release_count == 1
    assert authority.claim_calls == []
    assert str(raised.value) == "telnyx_handshake_cleanup_failed"
    assert "permit-provider-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_abort_cleanup_failure_takes_precedence_over_stored_timeout() -> None:
    websocket, _ = _websocket()
    permit = _Permit()
    authority = _RaisingBlockingLeaseAuthority(expected_digest=_token_digest())

    with pytest.raises(TelnyxHandshakeError) as raised:
        await _service(authority, _Gate(permit), timeout_seconds=0.01).authenticate(websocket)

    assert permit.release_count == 1
    assert authority.abort_calls == [(CALL_CONTROL_ID, _token_digest())]
    assert str(raised.value) == "telnyx_handshake_cleanup_failed"
    assert "scheduler-provider-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
