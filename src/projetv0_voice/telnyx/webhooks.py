"""Verified, transport-neutral Telnyx webhook processing."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import importlib
import inspect
import json
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType, ModuleType
from typing import Literal, NoReturn, Protocol, cast
from uuid import UUID

from pydantic import SecretStr

from projetv0_voice.admission import (
    CallAdmissionRejected,
    WebhookFinalizationHandle,
    WebhookFinalizerOwner,
)
from projetv0_voice.models import (
    MAX_PROVIDER_RECORDING_ID_CHARS,
    VoiceOperationV1,
    is_valid_provider_recording_id,
)
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.telnyx.call_control import MAX_CALL_CONTROL_ID_CHARS

MAX_WEBHOOK_BODY_BYTES = 65_536
MAX_EVENT_ID_CHARS = 256
MAX_EVENT_TYPE_CHARS = 128
MAX_PROVIDER_ID_CHARS = MAX_PROVIDER_RECORDING_ID_CHARS
MAX_CLIENT_STATE_B64_CHARS = 4_096
HANDLED_WEBHOOK_TYPES = frozenset(
    {
        "call.initiated",
        "call.answered",
        "call.hangup",
        "call.recording.saved",
        "call.recording.error",
    }
)
_LEASE_KEYS = frozenset(
    {
        "action",
        "call_control_id",
        "call_id",
        "tenant_id",
        "agent_id",
        "state",
        "token_hash",
        "created_at",
        "expires_at",
        "closed_at",
    }
)


class WebhookError(RuntimeError):
    """A webhook error carrying only a constant project code."""


class WebhookConfigurationError(WebhookError):
    """The verifier cannot be safely constructed."""


class WebhookBodyTooLarge(WebhookError):
    """The raw request exceeds the Task 5 parsing bound."""


class InvalidWebhookSignature(WebhookError):
    """The request signature boundary rejected the request."""


class InvalidWebhookPayload(WebhookError):
    """A verified request did not match the strict minimal envelope."""


class WebhookInternalError(WebhookError):
    """An unexpected verifier/runtime defect occurred."""


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedWebhook:
    event_id: str = field(repr=False)
    event_type: str = field(repr=False)
    occurred_at: datetime = field(repr=False)
    call_control_id: str | None = field(repr=False)
    call_leg_id: str | None = field(repr=False)
    call_session_id: str | None = field(repr=False)
    recording_id: str | None = field(repr=False)
    stream_id: str | None = field(repr=False)
    client_state: SecretStr | None = field(repr=False)
    recording_started_at: datetime | None = field(repr=False)
    recording_ended_at: datetime | None = field(repr=False)
    recording_channels: str | None = field(repr=False)
    semantic_fingerprint_sha256: bytes = field(repr=False)
    legacy_v1_semantic_fingerprint_sha256: bytes | None = field(
        default=None, repr=False
    )
    direction: Literal["incoming"] | None = field(default=None, repr=False)
    call_state: Literal["parked", "answered"] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            self.client_state is not None
            and not isinstance(self.client_state, SecretStr)
            or self.recording_started_at is not None
            and (
                not isinstance(self.recording_started_at, datetime)
                or self.recording_started_at.tzinfo is None
                or self.recording_started_at.utcoffset() is None
            )
            or self.recording_ended_at is not None
            and (
                not isinstance(self.recording_ended_at, datetime)
                or self.recording_ended_at.tzinfo is None
                or self.recording_ended_at.utcoffset() is None
            )
            or self.recording_channels is not None
            and not _valid_recording_channels(self.recording_channels)
            or self.direction not in {None, "incoming"}
            or self.call_state not in {None, "parked", "answered"}
            or type(self.semantic_fingerprint_sha256) is not bytes
            or len(self.semantic_fingerprint_sha256) != 32
            or self.legacy_v1_semantic_fingerprint_sha256 is not None
            and (
                type(self.legacy_v1_semantic_fingerprint_sha256) is not bytes
                or len(self.legacy_v1_semantic_fingerprint_sha256) != 32
            )
        ):
            raise ValueError("verified_webhook_invalid")

    def __repr__(self) -> str:
        return "VerifiedWebhook()"


@dataclass(frozen=True, slots=True, repr=False)
class WebhookDurableEffect:
    lease: Mapping[str, object] | None = field(default=None, repr=False)
    operation: VoiceOperationV1 | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        normalized_lease = _validated_lease(self.lease)
        if self.lease is not None and normalized_lease is None:
            raise TypeError("invalid_webhook_effect")
        if self.operation is not None and not isinstance(self.operation, VoiceOperationV1):
            raise TypeError("invalid_webhook_effect")
        object.__setattr__(self, "lease", normalized_lease)

    def __repr__(self) -> str:
        return "WebhookDurableEffect()"


class WebhookLocalReservation(Protocol):
    def abandon_before_submit(self) -> None:
        """Release a reservation synchronously before any durable submission."""


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedWebhook:
    effect: WebhookDurableEffect | None
    reservation: WebhookLocalReservation | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.effect is not None and not isinstance(self.effect, WebhookDurableEffect):
            raise TypeError("invalid_webhook_resolution")

    def __repr__(self) -> str:
        return "ResolvedWebhook()"


def _validated_lease(
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    try:
        if set(value) != _LEASE_KEYS or value.get("action") != "upsert":
            return None
        call_control_id = value.get("call_control_id")
        tenant_id = value.get("tenant_id")
        agent_id = value.get("agent_id")
        state = value.get("state")
        token_hash = value.get("token_hash")
        created_at = value.get("created_at")
        expires_at = value.get("expires_at")
        closed_at = value.get("closed_at")
        if (
            not isinstance(call_control_id, str)
            or not call_control_id
            or not isinstance(value.get("call_id"), UUID)
            or not isinstance(tenant_id, str)
            or not tenant_id
            or not isinstance(agent_id, str)
            or not agent_id
            or state not in {"pending", "active", "terminal"}
            or type(token_hash) is not bytes
            or len(token_hash) != 32
            or not isinstance(created_at, datetime)
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
            or not isinstance(expires_at, datetime)
            or expires_at.tzinfo is None
            or expires_at.utcoffset() is None
            or expires_at <= created_at
            or closed_at is not None
            and (
                not isinstance(closed_at, datetime)
                or closed_at.tzinfo is None
                or closed_at.utcoffset() is None
            )
            or (state == "terminal") != (closed_at is not None)
        ):
            return None
        return MappingProxyType(dict(value))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class WebhookDisposition:
    status_code: int
    body: bytes = b""


WebhookResolver = Callable[
    [VerifiedWebhook], ResolvedWebhook | Awaitable[ResolvedWebhook]
]


def _raise_constant(error_type: type[WebhookError], code: str) -> NoReturn:
    raise error_type(code)


def _decode_public_key(value: object) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(decoded) != 32:
        return None
    return decoded


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("non_standard_json_constant")


def _bounded_string(value: object, maximum: int) -> str | None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        return None
    return value


def _optional_provider_id(payload: Mapping[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    bounded = _bounded_string(value, MAX_PROVIDER_ID_CHARS)
    if bounded is None:
        raise ValueError("invalid_provider_id")
    return bounded


def _optional_call_control_id(payload: Mapping[str, object]) -> str | None:
    value = payload.get("call_control_id")
    if value is None:
        return None
    bounded = _bounded_string(value, MAX_CALL_CONTROL_ID_CHARS)
    if bounded is None:
        raise ValueError("invalid_call_control_id")
    return bounded


def _optional_recording_id(payload: Mapping[str, object]) -> str | None:
    value = payload.get("recording_id")
    if value is None:
        return None
    if not is_valid_provider_recording_id(value):
        raise ValueError("invalid_recording_id")
    return value


def _optional_client_state(payload: Mapping[str, object]) -> SecretStr | None:
    value = payload.get("client_state")
    if value is None:
        return None
    if not isinstance(value, str) or not 0 < len(value) <= MAX_CLIENT_STATE_B64_CHARS:
        raise ValueError("invalid_client_state")
    return SecretStr(value)


def _optional_utc_datetime(payload: Mapping[str, object], name: str) -> datetime | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not 0 < len(value) <= 64:
        raise ValueError("invalid_recording_time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("invalid_recording_time")
    return parsed.astimezone(UTC)


def _valid_recording_channels(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 16


def _optional_recording_channels(payload: Mapping[str, object]) -> str | None:
    value = payload.get("channels")
    if value is None:
        return None
    if not _valid_recording_channels(value):
        raise ValueError("invalid_recording_channels")
    return cast(str, value)


def _optional_direction(payload: Mapping[str, object]) -> Literal["incoming"] | None:
    value = payload.get("direction")
    if value is None:
        return None
    if value != "incoming":
        raise ValueError("invalid_direction")
    return "incoming"


def _optional_call_state(
    payload: Mapping[str, object],
) -> Literal["parked", "answered"] | None:
    value = payload.get("state")
    if value is None:
        return None
    if value not in {"parked", "answered"}:
        raise ValueError("invalid_call_state")
    return value


def _canonical_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _semantic_fingerprint(
    *,
    event_id: str,
    event_type: str,
    occurred_at: datetime,
    call_control_id: str | None,
    call_leg_id: str | None,
    call_session_id: str | None,
    recording_id: str | None,
    stream_id: str | None,
    client_state: SecretStr | None,
    recording_started_at: datetime | None,
    recording_ended_at: datetime | None,
    recording_channels: str | None,
    direction: Literal["incoming"] | None,
    call_state: Literal["parked", "answered"] | None,
    include_action_fields: bool,
) -> bytes:
    values: list[object] = [
        "projetv0.voice.webhook.semantic",
        1,
        event_id,
        event_type,
        _canonical_time(occurred_at),
        call_control_id,
        call_leg_id,
        call_session_id,
        recording_id,
        stream_id,
        None if client_state is None else client_state.get_secret_value(),
        _canonical_time(recording_started_at),
        _canonical_time(recording_ended_at),
        recording_channels,
    ]
    if include_action_fields and event_type in {"call.initiated", "call.answered"}:
        values.extend((direction, call_state))
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _strict_envelope(body: bytes, required_types: frozenset[str]) -> VerifiedWebhook | None:
    try:
        parsed = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(parsed, Mapping):
            return None
        data = parsed.get("data")
        if not isinstance(data, Mapping):
            return None
        event_id = _bounded_string(data.get("id"), MAX_EVENT_ID_CHARS)
        event_type = _bounded_string(data.get("event_type"), MAX_EVENT_TYPE_CHARS)
        occurred_raw = data.get("occurred_at")
        payload = data.get("payload")
        if event_id is None or event_type is None:
            return None
        if not isinstance(occurred_raw, str) or not occurred_raw or len(occurred_raw) > 64:
            return None
        if not isinstance(payload, Mapping):
            return None
        occurred_at = datetime.fromisoformat(occurred_raw.replace("Z", "+00:00"))
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            return None
        normalized_time = occurred_at.astimezone(UTC)
        call_control_id = _optional_call_control_id(payload)
        if (
            event_type in required_types
            or event_type in {"call.initiated", "call.answered", "call.hangup"}
        ) and call_control_id is None:
            return None
        direction: Literal["incoming"] | None = None
        call_state: Literal["parked", "answered"] | None = None
        if event_type == "call.initiated":
            direction = _optional_direction(payload)
            call_state = _optional_call_state(payload)
        elif event_type == "call.answered":
            call_state = _optional_call_state(payload)
        if event_type == "call.initiated" and (
            direction != "incoming" or call_state != "parked"
        ):
            return None
        if event_type == "call.answered" and call_state != "answered":
            return None
        call_leg_id = _optional_provider_id(payload, "call_leg_id")
        call_session_id = _optional_provider_id(payload, "call_session_id")
        recording_id = _optional_recording_id(payload)
        stream_id = _optional_provider_id(payload, "stream_id")
        client_state = _optional_client_state(payload)
        recording_started_at = _optional_utc_datetime(payload, "recording_started_at")
        recording_ended_at = _optional_utc_datetime(payload, "recording_ended_at")
        recording_channels = _optional_recording_channels(payload)
        return VerifiedWebhook(
            event_id=event_id,
            event_type=event_type,
            occurred_at=normalized_time,
            call_control_id=call_control_id,
            call_leg_id=call_leg_id,
            call_session_id=call_session_id,
            recording_id=recording_id,
            stream_id=stream_id,
            client_state=client_state,
            recording_started_at=recording_started_at,
            recording_ended_at=recording_ended_at,
            recording_channels=recording_channels,
            direction=direction,
            call_state=call_state,
            semantic_fingerprint_sha256=_semantic_fingerprint(
                event_id=event_id,
                event_type=event_type,
                occurred_at=normalized_time,
                call_control_id=call_control_id,
                call_leg_id=call_leg_id,
                call_session_id=call_session_id,
                recording_id=recording_id,
                stream_id=stream_id,
                client_state=client_state,
                recording_started_at=recording_started_at,
                recording_ended_at=recording_ended_at,
                recording_channels=recording_channels,
                direction=direction,
                call_state=call_state,
                include_action_fields=True,
            ),
            legacy_v1_semantic_fingerprint_sha256=(
                _semantic_fingerprint(
                    event_id=event_id,
                    event_type=event_type,
                    occurred_at=normalized_time,
                    call_control_id=call_control_id,
                    call_leg_id=call_leg_id,
                    call_session_id=call_session_id,
                    recording_id=recording_id,
                    stream_id=stream_id,
                    client_state=client_state,
                    recording_started_at=recording_started_at,
                    recording_ended_at=recording_ended_at,
                    recording_channels=recording_channels,
                    direction=direction,
                    call_state=call_state,
                    include_action_fields=False,
                )
                if event_type in {"call.initiated", "call.answered"}
                else None
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


class TelnyxWebhookVerifier:
    """Verify with the installed SDK before parsing a minimal envelope."""

    __slots__ = ("_public_key", "_required_types", "_verification", "_verification_error")

    def __init__(
        self,
        *,
        public_key: str,
        call_control_required_types: Collection[str] = (),
    ) -> None:
        verification: ModuleType | None = None
        with contextlib.suppress(Exception):
            verification = importlib.import_module("telnyx.lib.webhook_verification")
        decoded_key = _decode_public_key(public_key)
        verify_key_preflighted = False
        with contextlib.suppress(Exception):
            signing = importlib.import_module("nacl.signing")
            verify_key = getattr(signing, "VerifyKey", None)
            if callable(verify_key) and decoded_key is not None:
                verify_key(decoded_key)
                verify_key_preflighted = True
        verification_error = (
            None
            if verification is None
            else getattr(verification, "WebhookVerificationError", None)
        )
        valid_types = all(
            isinstance(item, str) and 0 < len(item) <= MAX_EVENT_TYPE_CHARS
            for item in call_control_required_types
        )
        if (
            decoded_key is None
            or not verify_key_preflighted
            or not valid_types
            or verification is None
            or not callable(getattr(verification, "verify_webhook_signature", None))
            or not isinstance(verification_error, type)
            or not issubclass(verification_error, Exception)
        ):
            _raise_constant(WebhookConfigurationError, "webhook_config_invalid")
        self._public_key = public_key
        self._required_types = frozenset(call_control_required_types)
        self._verification = verification
        self._verification_error = verification_error

    def __repr__(self) -> str:
        return "TelnyxWebhookVerifier()"

    def verify(
        self,
        *,
        body: bytes,
        headers: Sequence[tuple[str, str]],
    ) -> VerifiedWebhook:
        if not isinstance(body, bytes) or len(body) > MAX_WEBHOOK_BODY_BYTES:
            _raise_constant(WebhookBodyTooLarge, "body_too_large")

        normalized_headers: dict[str, str] = {}
        signature_count = 0
        timestamp_count = 0
        structurally_invalid = False
        try:
            for pair in headers:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    structurally_invalid = True
                    break
                name, value = pair
                if not isinstance(name, str) or not isinstance(value, str) or not name or not value:
                    structurally_invalid = True
                    break
                folded = name.casefold()
                if folded == "telnyx-signature-ed25519":
                    signature_count += 1
                elif folded == "telnyx-timestamp":
                    timestamp_count += 1
                normalized_headers[name] = value
        except (TypeError, ValueError):
            structurally_invalid = True
        if structurally_invalid or signature_count != 1 or timestamp_count != 1:
            _raise_constant(InvalidWebhookSignature, "invalid_signature")

        verification_error = False
        internal_error = False
        try:
            self._verification.verify_webhook_signature(
                body,
                normalized_headers,
                self._public_key,
            )
        except (self._verification_error, UnicodeDecodeError):
            verification_error = True
        except Exception:
            internal_error = True
        if verification_error:
            _raise_constant(InvalidWebhookSignature, "invalid_signature")
        if internal_error:
            _raise_constant(WebhookInternalError, "webhook_internal_error")

        verified = _strict_envelope(body, self._required_types)
        if verified is None:
            _raise_constant(InvalidWebhookPayload, "invalid_payload")
        return verified


class TelnyxWebhookProcessor:
    """Verify, resolve, and synchronously transfer finalization to the process owner."""

    __slots__ = ("_duplicate_resolver", "_finalizer_owner", "_resolver", "_verifier")

    def __init__(
        self,
        *,
        verifier: TelnyxWebhookVerifier,
        resolver: WebhookResolver,
        finalizer_owner: WebhookFinalizerOwner,
        duplicate_resolver: WebhookResolver | None = None,
    ) -> None:
        self._verifier = verifier
        self._resolver = resolver
        self._duplicate_resolver = duplicate_resolver
        self._finalizer_owner = finalizer_owner

    def __repr__(self) -> str:
        return "TelnyxWebhookProcessor()"

    @staticmethod
    def _abandon(resolution: ResolvedWebhook) -> None:
        if resolution.reservation is not None:
            with contextlib.suppress(Exception):
                resolution.reservation.abandon_before_submit()

    def start_webhook_finalization(
        self,
        event: VerifiedWebhook,
        resolution: ResolvedWebhook,
    ) -> WebhookFinalizationHandle | None:
        """Synchronously transfer ownership or abandon before any submission."""

        try:
            handle = self._finalizer_owner.start_webhook_finalization(event, resolution)
        except Exception:
            self._abandon(resolution)
            return None
        if inspect.isawaitable(handle):
            close = getattr(handle, "close", None)
            if callable(close):
                close()
            self._abandon(resolution)
            return None
        wait = getattr(handle, "wait", None)
        if not callable(wait):
            self._abandon(resolution)
            return None
        return handle

    async def process(
        self,
        *,
        body: bytes,
        headers: Sequence[tuple[str, str]],
    ) -> WebhookDisposition:
        try:
            event = self._verifier.verify(body=body, headers=headers)
        except WebhookBodyTooLarge:
            return WebhookDisposition(413)
        except InvalidWebhookSignature:
            return WebhookDisposition(403)
        except InvalidWebhookPayload:
            return WebhookDisposition(400)
        except WebhookInternalError:
            return WebhookDisposition(500)
        except Exception:
            return WebhookDisposition(500)

        try:
            classification = await self._finalizer_owner.classify_webhook_receipt(event)
        except (PersistenceError, TimeoutError):
            return WebhookDisposition(503)
        except Exception:
            return WebhookDisposition(500)
        if classification == "conflict":
            return WebhookDisposition(400)
        if classification == "duplicate":
            if self._duplicate_resolver is None:
                resolution: ResolvedWebhook | Awaitable[ResolvedWebhook] = ResolvedWebhook(None)
            else:
                try:
                    resolution = self._duplicate_resolver(event)
                    if inspect.isawaitable(resolution):
                        resolution = await resolution
                except CallAdmissionRejected as error:
                    return WebhookDisposition(error.status_code)
                except Exception:
                    return WebhookDisposition(500)
        elif classification == "missing":
            try:
                resolution = self._resolver(event)
                if inspect.isawaitable(resolution):
                    resolution = await resolution
            except CallAdmissionRejected as error:
                return WebhookDisposition(error.status_code)
            except Exception:
                return WebhookDisposition(500)
        else:
            return WebhookDisposition(500)
        if not isinstance(resolution, ResolvedWebhook):
            return WebhookDisposition(500)
        handle = self.start_webhook_finalization(event, resolution)
        if handle is None:
            return WebhookDisposition(503)
        try:
            disposition = await asyncio.shield(handle.wait())
        except asyncio.CancelledError:
            raise
        except Exception:
            return WebhookDisposition(500)
        if (
            not isinstance(disposition, WebhookDisposition)
            or disposition.status_code not in {200, 400, 500, 503}
            or disposition.body != b""
        ):
            return WebhookDisposition(500)
        return disposition
