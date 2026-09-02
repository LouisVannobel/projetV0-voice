"""Authenticated, one-consumer Telnyx WebSocket handshake boundary."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from pipecat.runner.types import TelnyxCallData
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pydantic import SecretStr
from starlette.websockets import WebSocket, WebSocketState

from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    QualifiedDeploymentProfileV1,
    RuntimeDeploymentProfileV1,
)
from projetv0_voice.telnyx.serializer import AudioAdmission, ProjetV0TelnyxFrameSerializer

TOKEN_HEADER = b"x-telnyx-streaming-auth-token"
TOKEN_FIELD = "x-telnyx-streaming-auth-token"
TOKEN_LOCATOR_ID = "telnyx-header-connected-v1"
MAX_HANDSHAKE_MESSAGE_BYTES = 65_536
MAX_TOKEN_CHARACTERS = 4_000
MAX_OPAQUE_FIELD_BYTES = 1_024
MAX_PHONE_FIELD_BYTES = 256
MAX_SEQUENCE_BYTES = 32


class TelnyxHandshakeError(RuntimeError):
    """A public constant-safe handshake failure."""


class TelnyxHandshakeRejectedError(TelnyxHandshakeError):
    """The handshake did not satisfy the authenticated contract."""


class TelnyxHandshakeCapacityError(TelnyxHandshakeError):
    """No unauthenticated handshake permit was available."""


class TelnyxHandshakeTimeoutError(TelnyxHandshakeError):
    """The single global handshake deadline elapsed."""


class LeaseClaim(Protocol):
    """Opaque Task 10 lease claim returned after the atomic transition."""


class LeaseAuthority(Protocol):
    async def claim_once(
        self, *, call_control_id: str, token_digest: bytes
    ) -> LeaseClaim | None: ...

    def schedule_abort_if_matches(
        self, *, call_control_id: str, token_digest: bytes
    ) -> None:
        """Schedule cleanup without raising; Task 10 reports internal defects."""
        ...


class UnauthenticatedPermit(Protocol):
    def release(self) -> None:
        """Release synchronously without raising; Task 10 reports internal defects."""
        ...


class UnauthenticatedGate(Protocol):
    def try_acquire(self) -> UnauthenticatedPermit | None: ...


@dataclass(frozen=True, slots=True)
class AuthenticatedTelnyxHandshake:
    call_data: TelnyxCallData = field(repr=False)
    token_locator_id: str
    lease_claim: LeaseClaim = field(repr=False)
    transport: FastAPIWebsocketTransport = field(repr=False)
    audio_admission: AudioAdmission = field(repr=False)

    def __repr__(self) -> str:
        return "AuthenticatedTelnyxHandshake()"


@dataclass(frozen=True, slots=True)
class _CapturedTelnyxHandshake:
    call_data: TelnyxCallData = field(repr=False)
    token: SecretStr = field(repr=False)
    token_digest: bytes = field(repr=False)
    redacted_fixture_bytes: bytes = field(repr=False)
    fixture_sha256: str
    safe_summary: tuple[tuple[str, str | int], ...]


class _TransferredAuthentication(
    Coroutine[Any, Any, AuthenticatedTelnyxHandshake]
):
    """An eager-safe permit owner whose close path releases before first send."""

    __slots__ = (
        "_active_permits",
        "_permit",
        "_permit_id",
        "_release_failed",
        "_released",
        "_runner",
    )

    def __init__(
        self,
        service: AuthenticatedTelnyxHandshakeService,
        websocket: WebSocket,
        permit: UnauthenticatedPermit,
        permit_id: int,
    ) -> None:
        self._active_permits = service._active_permits  # noqa: SLF001
        self._permit = permit
        self._permit_id = permit_id
        self._release_failed = False
        self._released = False
        self._runner = self._run(service, websocket)

    def __await__(self) -> _TransferredAuthentication:
        return self

    def __iter__(self) -> _TransferredAuthentication:
        return self

    def __next__(self) -> Any:
        return self.send(None)

    def send(self, value: Any) -> Any:
        return self._runner.send(value)

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> Any:
        if val is None and tb is None:
            return self._runner.throw(typ)
        return self._runner.throw(typ, val, tb)

    def close(self) -> None:
        try:
            self._runner.close()
        finally:
            self._finish()

    def _finish(self) -> bool:
        if self._released:
            return self._release_failed
        self._released = True
        try:
            self._permit.release()
        except Exception:
            self._release_failed = True
        finally:
            with self._active_permits[0]:
                self._active_permits[1].discard(self._permit_id)
        return self._release_failed

    async def _run(
        self,
        service: AuthenticatedTelnyxHandshakeService,
        websocket: WebSocket,
    ) -> AuthenticatedTelnyxHandshake:
        try:
            return await service._authenticate_owned(  # noqa: SLF001
                websocket,
                finish_permit=self._finish,
            )
        finally:
            self._finish()


class _InvalidJson(ValueError):
    pass


def _reject_constant(_value: str) -> None:
    raise _InvalidJson


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJson
        result[key] = value
    return result


def _strict_json_object(raw: str) -> dict[str, Any]:
    parsed: object | None = None
    invalid = False
    try:
        raw_bytes = raw.encode("utf-8")
        if len(raw_bytes) > MAX_HANDSHAKE_MESSAGE_BYTES:
            invalid = True
        else:
            parsed = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (UnicodeEncodeError, json.JSONDecodeError, _InvalidJson):
        invalid = True
    if invalid or not isinstance(parsed, dict):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    return parsed


def _bounded_utf8(value: object, *, maximum_bytes: int) -> str:
    encoded: bytes | None = None
    if isinstance(value, str) and value:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            encoded = None
    if encoded is None or len(encoded) > maximum_bytes:
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    return cast(str, value)


def _token_bytes(value: object) -> tuple[SecretStr, bytes]:
    encoded: bytes | None = None
    if isinstance(value, str) and 1 <= len(value) <= MAX_TOKEN_CHARACTERS:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            encoded = None
    if encoded is None:
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    return SecretStr(cast(str, value)), encoded


def _single_header_token(websocket: WebSocket) -> tuple[SecretStr, bytes]:
    matches: list[bytes] = []
    headers = websocket.scope.get("headers", [])
    if isinstance(headers, list):
        for item in headers:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], bytes)
                and isinstance(item[1], bytes)
                and item[0].lower() == TOKEN_HEADER
            ):
                matches.append(item[1])
    if len(matches) != 1:
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    decoded: str | None = None
    invalid = False
    try:
        decoded = matches[0].decode("utf-8")
    except UnicodeDecodeError:
        invalid = True
    if invalid or decoded is None:
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    return _token_bytes(decoded)


async def _receive_text(websocket: WebSocket) -> str:
    failed = False
    message: str | None = None
    try:
        message = await websocket.receive_text()
    except asyncio.CancelledError:
        raise
    except Exception:
        failed = True
    if failed or not isinstance(message, str):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    return message


def _optional_bounded(value: object, *, maximum_bytes: int) -> str | None:
    if value is None:
        return None
    return _bounded_utf8(value, maximum_bytes=maximum_bytes)


def _redacted_fixture(
    *, token_byte_length: int, from_number: str | None, to_number: str | None
) -> bytes:
    fixture = {
        "authentication": {
            "connected.connected.x-telnyx-streaming-auth-token": {
                "type": "string",
                "utf8_length": token_byte_length,
            },
            "header.x-telnyx-streaming-auth-token": {
                "type": "string",
                "utf8_length": token_byte_length,
            },
        },
        "connected": {
            "event": "connected",
            "version": "1.0.0",
        },
        "provider": "telnyx",
        "schema": "projetv0.telnyx.handshake.v1",
        "start": {
            "event": "start",
            "sequence_number": {"format": "decimal", "type": "string"},
            "start.call_control_id": {"type": "string"},
            "start.from": {"type": "string" if from_number is not None else "null"},
            "start.media_format.channels": 1,
            "start.media_format.encoding": "PCMU",
            "start.media_format.sample_rate": 8000,
            "start.to": {"type": "string" if to_number is not None else "null"},
            "stream_id": {"type": "string"},
        },
        "token_locator_id": TOKEN_LOCATOR_ID,
    }
    return json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe_summary(
    *, token_byte_length: int, from_number: str | None, to_number: str | None
) -> tuple[tuple[str, str | int], ...]:
    return (
        ("connected.event", "string"),
        ("connected.version", "string"),
        ("header.x-telnyx-streaming-auth-token", "string"),
        ("connected.connected.x-telnyx-streaming-auth-token", "string"),
        ("token.utf8_length", token_byte_length),
        ("start.event", "string"),
        ("start.sequence_number", "decimal-string"),
        ("start.stream_id", "string"),
        ("start.start.call_control_id", "string"),
        ("start.start.media_format.encoding", "PCMU"),
        ("start.start.media_format.sample_rate", 8000),
        ("start.start.media_format.channels", 1),
        ("start.start.from", "string" if from_number is not None else "null"),
        ("start.start.to", "string" if to_number is not None else "null"),
    )


async def _capture_telnyx_handshake(websocket: WebSocket) -> _CapturedTelnyxHandshake:
    header_token, header_bytes = _single_header_token(websocket)
    connected = _strict_json_object(await _receive_text(websocket))
    start_message = _strict_json_object(await _receive_text(websocket))

    if connected.get("event") != "connected" or connected.get("version") != "1.0.0":
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    connected_payload = connected.get("connected")
    if not isinstance(connected_payload, dict):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    _, connected_bytes = _token_bytes(connected_payload.get(TOKEN_FIELD))
    if not hmac.compare_digest(header_bytes, connected_bytes):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")

    if start_message.get("event") != "start":
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    sequence_number = _bounded_utf8(
        start_message.get("sequence_number"), maximum_bytes=MAX_SEQUENCE_BYTES
    )
    if not sequence_number.isdecimal():
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    stream_id = _bounded_utf8(
        start_message.get("stream_id"), maximum_bytes=MAX_OPAQUE_FIELD_BYTES
    )
    start = start_message.get("start")
    if not isinstance(start, dict):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    call_control_id = _bounded_utf8(
        start.get("call_control_id"), maximum_bytes=MAX_OPAQUE_FIELD_BYTES
    )
    media_format = start.get("media_format")
    if not isinstance(media_format, dict):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    if (
        media_format.get("encoding") != "PCMU"
        or type(media_format.get("sample_rate")) is not int
        or media_format.get("sample_rate") != 8000
        or type(media_format.get("channels")) is not int
        or media_format.get("channels") != 1
    ):
        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
    from_number = _optional_bounded(
        start.get("from"), maximum_bytes=MAX_PHONE_FIELD_BYTES
    )
    to_number = _optional_bounded(start.get("to"), maximum_bytes=MAX_PHONE_FIELD_BYTES)

    token_digest = hashlib.sha256(header_bytes).digest()
    fixture = _redacted_fixture(
        token_byte_length=len(header_bytes),
        from_number=from_number,
        to_number=to_number,
    )
    call_data = TelnyxCallData.model_validate(
        {
            "stream_id": stream_id,
            "call_id": call_control_id,
            "outbound_encoding": "PCMU",
            "from": from_number,
            "to": to_number,
        }
    )
    return _CapturedTelnyxHandshake(
        call_data=call_data,
        token=header_token,
        token_digest=token_digest,
        redacted_fixture_bytes=fixture,
        fixture_sha256=hashlib.sha256(fixture).hexdigest(),
        safe_summary=_safe_summary(
            token_byte_length=len(header_bytes),
            from_number=from_number,
            to_number=to_number,
        ),
    )


class AuthenticatedTelnyxHandshakeService:
    """Authenticate a Telnyx handshake before allocating its native transport."""

    def __init__(
        self,
        *,
        profile: RuntimeDeploymentProfileV1,
        lease_authority: LeaseAuthority,
        unauthenticated_gate: UnauthenticatedGate,
        timeout_seconds: float,
    ) -> None:
        if (
            not isinstance(
                profile, QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1
            )
            or profile.token_locator_id != TOKEN_LOCATOR_ID
            or not isinstance(timeout_seconds, int | float)
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise TelnyxHandshakeError("telnyx_handshake_config_invalid")
        self._profile = profile
        self._lease_authority = lease_authority
        self._unauthenticated_gate = unauthenticated_gate
        self._timeout_seconds = float(timeout_seconds)
        self._active_permits: tuple[threading.Lock, set[int]] = (
            threading.Lock(),
            set(),
        )

    def __repr__(self) -> str:
        return "AuthenticatedTelnyxHandshakeService()"

    async def authenticate(self, websocket: WebSocket) -> AuthenticatedTelnyxHandshake:
        permit: UnauthenticatedPermit | None = None
        gate_failed = False
        try:
            permit = self._unauthenticated_gate.try_acquire()
        except asyncio.CancelledError:
            raise
        except Exception:
            gate_failed = True
        if gate_failed:
            raise TelnyxHandshakeError("telnyx_handshake_gate_failed")
        if permit is None:
            raise TelnyxHandshakeCapacityError("telnyx_handshake_capacity")
        try:
            operation = self.transfer_authentication(websocket, permit)
        except BaseException:
            try:
                permit.release()
            except Exception:
                raise TelnyxHandshakeError("telnyx_handshake_cleanup_failed") from None
            raise
        return await operation

    def transfer_authentication(
        self,
        websocket: WebSocket,
        permit: UnauthenticatedPermit,
    ) -> Coroutine[Any, Any, AuthenticatedTelnyxHandshake]:
        """Synchronously transfer one already-acquired accepted-WebSocket permit."""

        if (
            not isinstance(websocket, WebSocket)
            or websocket.application_state is not WebSocketState.CONNECTED
            or websocket.client_state is not WebSocketState.CONNECTED
        ):
            raise TelnyxHandshakeError("telnyx_handshake_context_invalid") from None
        release = getattr(permit, "release", None)
        if not callable(release):
            raise TelnyxHandshakeError("telnyx_handshake_permit_invalid") from None
        permit_id = id(permit)
        lock, active = self._active_permits
        with lock:
            if permit_id in active:
                raise TelnyxHandshakeError("telnyx_handshake_permit_invalid") from None
            active.add(permit_id)
        try:
            return _TransferredAuthentication(self, websocket, permit, permit_id)
        except BaseException:
            with lock:
                active.discard(permit_id)
            raise TelnyxHandshakeError("telnyx_handshake_permit_invalid") from None

    async def _authenticate_owned(
        self,
        websocket: WebSocket,
        *,
        finish_permit: Callable[[], bool],
    ) -> AuthenticatedTelnyxHandshake:

        call_control_id: str | None = None
        token_digest: bytes | None = None
        abort_required = False
        handoff_complete = False
        result: AuthenticatedTelnyxHandshake | None = None
        timed_out = False
        unexpected_failure = False
        cleanup_failed = False
        stored_error: TelnyxHandshakeError | None = None
        try:
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    captured = await _capture_telnyx_handshake(websocket)
                    call_control_id = captured.call_data.call_id
                    token_digest = captured.token_digest
                    if (
                        call_control_id is None
                        or not hmac.compare_digest(
                            captured.fixture_sha256,
                            self._profile.telnyx_handshake_fixture_sha256,
                        )
                    ):
                        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")

                    abort_required = True
                    lease_claim = await self._lease_authority.claim_once(
                        call_control_id=call_control_id,
                        token_digest=token_digest,
                    )
                    if lease_claim is None:
                        abort_required = False
                        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")

                    audio_admission = AudioAdmission()
                    serializer = ProjetV0TelnyxFrameSerializer(
                        captured.call_data.stream_id or "",
                        expected_call_control_id=call_control_id,
                        audio_admission=audio_admission,
                    )
                    params = FastAPIWebsocketParams(
                        audio_in_enabled=True,
                        audio_out_enabled=True,
                        add_wav_header=False,
                        serializer=serializer,
                    )
                    transport = FastAPIWebsocketTransport(websocket, params)
                    result = AuthenticatedTelnyxHandshake(
                        call_data=captured.call_data,
                        token_locator_id=TOKEN_LOCATOR_ID,
                        lease_claim=lease_claim,
                        transport=transport,
                        audio_admission=audio_admission,
                    )
                    handoff_complete = True
                    abort_required = False
            except TimeoutError:
                timed_out = True
            except asyncio.CancelledError:
                raise
            except TelnyxHandshakeError as error:
                stored_error = error
            except Exception:
                unexpected_failure = True
        finally:
            try:
                if (
                    abort_required
                    and not handoff_complete
                    and call_control_id is not None
                    and token_digest is not None
                ):
                    self._lease_authority.schedule_abort_if_matches(
                        call_control_id=call_control_id,
                        token_digest=token_digest,
                    )
            except Exception:
                cleanup_failed = True
            finally:
                permit_release_failed = finish_permit()
                if permit_release_failed:
                    cleanup_failed = True
                    if (
                        handoff_complete
                        and call_control_id is not None
                        and token_digest is not None
                    ):
                        try:
                            self._lease_authority.schedule_abort_if_matches(
                                call_control_id=call_control_id,
                                token_digest=token_digest,
                            )
                        except Exception:
                            cleanup_failed = True

        if cleanup_failed:
            raise TelnyxHandshakeError("telnyx_handshake_cleanup_failed")
        if stored_error is not None:
            raise stored_error
        if timed_out:
            raise TelnyxHandshakeTimeoutError("telnyx_handshake_timeout")
        if unexpected_failure or result is None:
            raise TelnyxHandshakeError("telnyx_handshake_failed")
        return result


__all__ = [
    "AuthenticatedTelnyxHandshake",
    "AuthenticatedTelnyxHandshakeService",
    "LeaseAuthority",
    "LeaseClaim",
    "TelnyxHandshakeCapacityError",
    "TelnyxHandshakeError",
    "TelnyxHandshakeRejectedError",
    "TelnyxHandshakeTimeoutError",
    "UnauthenticatedGate",
    "UnauthenticatedPermit",
]
