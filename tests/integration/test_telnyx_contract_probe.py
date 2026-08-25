from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pytest
from starlette.websockets import WebSocket, WebSocketState

from projetv0_voice.discovery.telnyx_contract import (
    ContractProbeResult,
    TelnyxContractProbe,
    TelnyxContractProbeRejectedError,
    TelnyxContractProbeTimeoutError,
    TelnyxContractProbeUsedError,
)

TOKEN = "synthetic-probe-token"
CALL_CONTROL_ID = "probe-call-control-secret"
STREAM_ID = "probe-stream-secret"
FROM_NUMBER = "+33333333333"
TO_NUMBER = "+33444444444"
LOCATOR_ID = "telnyx-header-connected-v1"


@dataclass(frozen=True)
class _Claim:
    secret_note: str = "probe-lease-secret"


class _LeaseAuthority:
    def __init__(self, *, claim_result: bool = True, block: bool = False) -> None:
        self.claim_result = claim_result
        self.block = block
        self.claim_calls: list[tuple[str, bytes]] = []
        self.abort_calls: list[tuple[str, bytes]] = []
        self.started = asyncio.Event()
        self.never = asyncio.Event()
        self.claim = _Claim()

    async def claim_once(self, *, call_control_id: str, token_digest: bytes) -> _Claim | None:
        self.claim_calls.append((call_control_id, token_digest))
        self.started.set()
        if self.block:
            await self.never.wait()
        return self.claim if self.claim_result else None

    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        self.abort_calls.append((call_control_id, token_digest))


class _RaisingAbortAuthority(_LeaseAuthority):
    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        super().schedule_abort_if_matches(
            call_control_id=call_control_id, token_digest=token_digest
        )
        raise RuntimeError("probe-scheduler-secret")


def _digest() -> bytes:
    return hashlib.sha256(TOKEN.encode("utf-8")).digest()


def _connected(*, reverse_order: bool = False) -> str:
    pairs: list[tuple[str, object]] = [
        ("event", "connected"),
        ("version", "1.0.0"),
        ("x-telnyx-streaming-auth-token", TOKEN),
        ("documented_extra", {"type": "streaming"}),
    ]
    if reverse_order:
        pairs.reverse()
    return json.dumps(dict(pairs))


def _start(*, reverse_order: bool = False) -> str:
    pairs: list[tuple[str, object]] = [
        ("event", "start"),
        ("sequence_number", "7"),
        ("stream_id", STREAM_ID),
        (
            "start",
            {
                "call_control_id": CALL_CONTROL_ID,
                "from": FROM_NUMBER,
                "to": TO_NUMBER,
                "media_format": {
                    "encoding": "PCMU",
                    "sample_rate": 8000,
                    "channels": 1,
                },
                "stream_auth_token": "unsupported-start-guess",
            },
        ),
    ]
    if reverse_order:
        pairs.reverse()
    return json.dumps(dict(pairs))


def _fixture() -> bytes:
    fixture = {
        "authentication": {
            "connected.x-telnyx-streaming-auth-token": {
                "type": "string",
                "utf8_length": len(TOKEN.encode("utf-8")),
            },
            "header.x-telnyx-streaming-auth-token": {
                "type": "string",
                "utf8_length": len(TOKEN.encode("utf-8")),
            },
        },
        "connected": {"event": "connected", "version": "1.0.0"},
        "provider": "telnyx",
        "schema": "projetv0.telnyx.handshake.v1",
        "start": {
            "event": "start",
            "sequence_number": {"format": "decimal", "type": "string"},
            "start.call_control_id": {"type": "string"},
            "start.from": {"type": "string"},
            "start.media_format.channels": 1,
            "start.media_format.encoding": "PCMU",
            "start.media_format.sample_rate": 8000,
            "start.to": {"type": "string"},
            "stream_id": {"type": "string"},
        },
        "token_locator_id": LOCATOR_ID,
    }
    return json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _websocket(
    *,
    messages: tuple[str, str] | None = None,
) -> WebSocket:
    queued = list(messages or (_connected(), _start()))

    async def receive() -> dict[str, Any]:
        if not queued:
            return {"type": "websocket.disconnect", "code": 1000}
        return {"type": "websocket.receive", "text": queued.pop(0)}

    async def send(_message: dict[str, Any]) -> None:
        return None

    scope: dict[str, Any] = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "wss",
        "server": ("probe.invalid", 443),
        "client": ("127.0.0.1", 12346),
        "root_path": "",
        "path": "/telnyx/media",
        "raw_path": b"/telnyx/media",
        "query_string": b"stream_auth_token=unsupported-query-guess",
        "headers": [(b"x-telnyx-streaming-auth-token", TOKEN.encode("utf-8"))],
        "subprotocols": [],
    }
    websocket = WebSocket(scope, receive=receive, send=send)
    websocket.application_state = WebSocketState.CONNECTED
    websocket.client_state = WebSocketState.CONNECTED
    return websocket


@pytest.mark.asyncio
async def test_probe_returns_only_exact_redacted_conformance_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    authority = _LeaseAuthority()
    probe = TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.5)

    result = await probe.inspect(_websocket(), expected_token_digest=_digest())

    assert isinstance(result, ContractProbeResult)
    assert result.token_locator_id == LOCATOR_ID
    assert result.redacted_fixture_bytes == _fixture()
    assert result.fixture_sha256 == hashlib.sha256(_fixture()).hexdigest()
    assert result.safe_summary == (
        ("connected.event", "string"),
        ("connected.version", "string"),
        ("header.x-telnyx-streaming-auth-token", "string"),
        ("connected.x-telnyx-streaming-auth-token", "string"),
        ("token.utf8_length", len(TOKEN.encode("utf-8"))),
        ("start.event", "string"),
        ("start.sequence_number", "decimal-string"),
        ("start.stream_id", "string"),
        ("start.start.call_control_id", "string"),
        ("start.start.media_format.encoding", "PCMU"),
        ("start.start.media_format.sample_rate", 8000),
        ("start.start.media_format.channels", 1),
        ("start.start.from", "string"),
        ("start.start.to", "string"),
    )
    assert authority.claim_calls == [(CALL_CONTROL_ID, _digest())]
    assert authority.abort_calls == [(CALL_CONTROL_ID, _digest())]
    for secret in (
        TOKEN,
        CALL_CONTROL_ID,
        STREAM_ID,
        FROM_NUMBER,
        TO_NUMBER,
        authority.claim.secret_note,
    ):
        assert secret not in repr(result)
        assert secret not in caplog.text
        assert secret.encode("utf-8") not in result.redacted_fixture_bytes


@pytest.mark.asyncio
async def test_probe_fixture_is_independent_of_json_key_order_and_bounded_extras() -> None:
    first_authority = _LeaseAuthority()
    second_authority = _LeaseAuthority()
    first = await TelnyxContractProbe(
        lease_authority=first_authority, timeout_seconds=0.5
    ).inspect(_websocket(), expected_token_digest=_digest())
    second = await TelnyxContractProbe(
        lease_authority=second_authority, timeout_seconds=0.5
    ).inspect(
        _websocket(messages=(_connected(reverse_order=True), _start(reverse_order=True))),
        expected_token_digest=_digest(),
    )

    assert second.redacted_fixture_bytes == first.redacted_fixture_bytes
    assert second.fixture_sha256 == first.fixture_sha256


@pytest.mark.asyncio
async def test_probe_digest_mismatch_fails_before_claim_and_is_permanently_one_shot() -> None:
    authority = _LeaseAuthority()
    probe = TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.5)

    with pytest.raises(TelnyxContractProbeRejectedError) as rejected:
        await probe.inspect(_websocket(), expected_token_digest=b"0" * 32)

    assert authority.claim_calls == []
    assert authority.abort_calls == []
    assert str(rejected.value) == "telnyx_contract_probe_rejected"
    assert rejected.value.__cause__ is None
    assert rejected.value.__context__ is None
    with pytest.raises(TelnyxContractProbeUsedError):
        await probe.inspect(_websocket(), expected_token_digest=_digest())


@pytest.mark.asyncio
async def test_probe_rejected_claim_is_not_aborted() -> None:
    authority = _LeaseAuthority(claim_result=False)

    with pytest.raises(TelnyxContractProbeRejectedError):
        await TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.5).inspect(
            _websocket(), expected_token_digest=_digest()
        )

    assert authority.claim_calls == [(CALL_CONTROL_ID, _digest())]
    assert authority.abort_calls == []


@pytest.mark.asyncio
async def test_probe_timeout_during_unknown_claim_schedules_abort() -> None:
    authority = _LeaseAuthority(block=True)

    with pytest.raises(TelnyxContractProbeTimeoutError) as raised:
        await TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.01).inspect(
            _websocket(), expected_token_digest=_digest()
        )

    assert authority.abort_calls == [(CALL_CONTROL_ID, _digest())]
    assert str(raised.value) == "telnyx_contract_probe_timeout"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_probe_cancellation_during_unknown_claim_schedules_abort() -> None:
    authority = _LeaseAuthority(block=True)
    probe = TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.5)
    task = asyncio.create_task(probe.inspect(_websocket(), expected_token_digest=_digest()))
    await authority.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert authority.abort_calls == [(CALL_CONTROL_ID, _digest())]
    with pytest.raises(TelnyxContractProbeUsedError):
        await probe.inspect(_websocket(), expected_token_digest=_digest())


@pytest.mark.asyncio
async def test_probe_raising_abort_scheduler_is_constant_safe() -> None:
    authority = _RaisingAbortAuthority()

    with pytest.raises(Exception) as raised:
        await TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.5).inspect(
            _websocket(), expected_token_digest=_digest()
        )

    assert type(raised.value).__name__ == "TelnyxContractProbeError"
    assert str(raised.value) == "telnyx_contract_probe_cleanup_failed"
    assert "probe-scheduler-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_probe_raising_abort_scheduler_cannot_mask_cancellation() -> None:
    authority = _RaisingAbortAuthority(block=True)
    task = asyncio.create_task(
        TelnyxContractProbe(lease_authority=authority, timeout_seconds=0.5).inspect(
            _websocket(), expected_token_digest=_digest()
        )
    )
    await authority.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert authority.abort_calls == [(CALL_CONTROL_ID, _digest())]
