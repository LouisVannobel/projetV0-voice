"""Bounded, input-redacting Telnyx Call Control wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import telnyx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool, field_validator

from projetv0_voice.models import (
    MAX_PROVIDER_RECORDING_ID_CHARS,
    is_valid_provider_recording_id,
)

if TYPE_CHECKING:
    from projetv0_voice.telnyx.recordings import (
        ProviderDeleteResultV1,
        ProviderRecordingPageV1,
        ProviderRecordingV1,
    )

ATTEMPT_DEADLINE_SECONDS = 0.500
CLOSE_DEADLINE_SECONDS = 1.0
MAX_CALL_CONTROL_ID_CHARS = 1_024
MAX_STREAM_URL_CHARS = 2_048
MAX_STREAM_AUTH_TOKEN_CHARS = 4_000
MAX_CLIENT_STATE_CHARS = 4_096
MAX_RECORDING_ID_CHARS = MAX_PROVIDER_RECORDING_ID_CHARS

CallControlOutcome = Literal[
    "accepted",
    "rejected",
    "rate_limited",
    "retryable_not_sent",
    "outcome_unknown",
]


class CallControlError(RuntimeError):
    """A Call Control error carrying only a constant project code."""


class CallControlConfigurationError(CallControlError):
    """The owned SDK client could not be constructed safely."""


class CallControlInputError(CallControlError):
    """A local caller violated the strict action contract."""


class CallControlClosedError(CallControlError):
    """The owned client has started closing."""


class CallControlCloseError(CallControlError):
    """The owned SDK client did not close inside its fixed bound."""


class RecordingCatalogError(CallControlError):
    """The bounded Telnyx recording catalog operation failed safely."""


class RecordingCatalogTransientError(RecordingCatalogError):
    """The catalog may be incomplete or temporarily unavailable."""


class RecordingCatalogInvalidError(RecordingCatalogError):
    """The catalog response cannot identify one recording safely."""


class _RecordingCatalogInvalid(Exception):
    """Internal sentinel reduced before any public error is constructed."""


class _RedactedFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class StreamingStartV1(_RedactedFrozenModel):
    stream_url: str = Field(
        min_length=1,
        max_length=MAX_STREAM_URL_CHARS,
        exclude=True,
        repr=False,
    )
    stream_auth_token: SecretStr = Field(
        min_length=1,
        max_length=MAX_STREAM_AUTH_TOKEN_CHARS,
        exclude=True,
        repr=False,
    )

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise ValueError("stream_url_invalid") from None
        if (
            parsed.scheme != "wss"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "?" in value
            or "#" in value
            or "\\" in value
            or any(character.isspace() for character in value)
            or port is not None and not 1 <= port <= 65_535
        ):
            raise ValueError("stream_url_invalid")
        return value


class RecordingStartV1(_RedactedFrozenModel):
    play_beep: StrictBool = Field(repr=False)
    client_state: SecretStr = Field(
        min_length=1,
        max_length=MAX_CLIENT_STATE_CHARS,
        exclude=True,
        repr=False,
    )


class TransferRequestV1(_RedactedFrozenModel):
    to_e164: str = Field(repr=False)
    target_leg_client_state: str = Field(min_length=1, max_length=4096, repr=False)
    timeout_secs: Literal[20] = 20

    @field_validator("to_e164")
    @classmethod
    def destination(cls, value: str) -> str:
        from projetv0_voice.models import _e164

        return _e164(value)


class RecordingStopV1(_RedactedFrozenModel):
    client_state: SecretStr = Field(
        min_length=1,
        max_length=MAX_CLIENT_STATE_CHARS,
        exclude=True,
        repr=False,
    )
    recording_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RECORDING_ID_CHARS,
        exclude=True,
        repr=False,
    )

    @field_validator("recording_id")
    @classmethod
    def validate_recording_id(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_provider_recording_id(value):
            raise ValueError("recording_id_invalid")
        return value


@dataclass(frozen=True, slots=True)
class CallControlResult:
    outcome: CallControlOutcome


@dataclass(frozen=True, slots=True)
class _Observation:
    outcome: CallControlOutcome
    retry: bool
    ambiguous: bool


def _valid_api_key(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_STREAM_AUTH_TOKEN_CHARS


def _valid_call_control_id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_CALL_CONTROL_ID_CHARS


def _valid_command_id(value: object) -> bool:
    return isinstance(value, UUID) and value.version == 4


def _status_observation(status_code: int) -> _Observation:
    if status_code in {408, 500}:
        return _Observation("outcome_unknown", retry=True, ambiguous=True)
    if status_code == 429:
        return _Observation("rate_limited", retry=False, ambiguous=False)
    if 400 <= status_code < 500:
        return _Observation("rejected", retry=False, ambiguous=False)
    return _Observation("outcome_unknown", retry=False, ambiguous=True)


def _connection_observation(error: telnyx.APIConnectionError) -> _Observation:
    cause = error.__cause__
    if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
        return _Observation("retryable_not_sent", retry=True, ambiguous=False)
    if isinstance(cause, httpx.PoolTimeout):
        return _Observation("retryable_not_sent", retry=False, ambiguous=False)
    return _Observation("outcome_unknown", retry=True, ambiguous=True)


CatalogFailureKind = Literal["transient", "invalid"]


def _recording_catalog_failure_kind(
    error: BaseException,
    *,
    not_found_transient: bool = False,
) -> CatalogFailureKind:
    if isinstance(error, telnyx.APIStatusError):
        if (
            not_found_transient
            and error.status_code == 404
            or error.status_code in {408, 409, 429}
            or error.status_code >= 500
        ):
            return "transient"
        return "invalid"
    if isinstance(error, (telnyx.APIConnectionError, TimeoutError)):
        return "transient"
    return "invalid"


def _raise_recording_catalog_failure(kind: CatalogFailureKind) -> NoReturn:
    if kind == "transient":
        raise RecordingCatalogTransientError("recording_catalog_transient") from None
    raise RecordingCatalogInvalidError("recording_catalog_invalid") from None


def _strict_catalog_text(value: object, maximum: int = 256) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 0 < len(value) <= maximum:
        raise _RecordingCatalogInvalid
    return value


def _valid_optional_catalog_filter_id(value: object) -> bool:
    return value is None or isinstance(value, str) and 0 < len(value) <= 256


def _strict_catalog_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 0 < len(value) <= 64:
        raise _RecordingCatalogInvalid
    invalid = False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        invalid = True
        parsed = datetime.min
    if invalid:
        raise _RecordingCatalogInvalid from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _RecordingCatalogInvalid
    return parsed.astimezone(UTC)


def _catalog_item(value: object) -> ProviderRecordingV1:
    from projetv0_voice.telnyx.recordings import ProviderRecordingV1

    try:
        return ProviderRecordingV1(
            recording_id=_strict_catalog_text(getattr(value, "id", None)),
            call_control_id=_strict_catalog_text(
                getattr(value, "call_control_id", None),
                MAX_CALL_CONTROL_ID_CHARS,
            ),
            call_leg_id=_strict_catalog_text(getattr(value, "call_leg_id", None)),
            call_session_id=_strict_catalog_text(
                getattr(value, "call_session_id", None)
            ),
            channels=_strict_catalog_text(getattr(value, "channels", None)),
            status=_strict_catalog_text(getattr(value, "status", None)),
            source=_strict_catalog_text(getattr(value, "source", None)),
            initiated_by=_strict_catalog_text(getattr(value, "initiated_by", None)),
            recording_started_at=_strict_catalog_datetime(
                getattr(value, "recording_started_at", None)
            ),
            recording_ended_at=_strict_catalog_datetime(
                getattr(value, "recording_ended_at", None)
            ),
        )
    except _RecordingCatalogInvalid:
        raise
    except Exception:
        pass
    raise _RecordingCatalogInvalid from None


def _valid_catalog_filter_time(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 64:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(None)


class CallControlClient:
    """Own exactly one long-lived Telnyx SDK client and one immediate retry."""

    __slots__ = ("_client", "_close_lock", "_closed", "_closing", "_timeout")

    def __init__(self, *, api_key: str, api_region: Literal["global", "EU"] = "global") -> None:
        if (
            not _valid_api_key(api_key)
            or type(api_region) not in (str,)
            or api_region not in {"global", "EU"}
        ):
            raise CallControlConfigurationError("call_control_config_invalid")
        sdk_client: telnyx.AsyncTelnyx | None = None
        construction_failed = False
        try:
            sdk_client = telnyx.AsyncTelnyx(
                api_key=api_key,
                base_url="https://api.telnyx.eu/v2" if api_region == "EU" else "https://api.telnyx.com/v2",
                max_retries=0,
                http_client=telnyx.DefaultAsyncHttpxClient(trust_env=False),
            )
        except Exception:
            construction_failed = True
        if construction_failed or sdk_client is None:
            raise CallControlConfigurationError("call_control_config_invalid")
        self._client = sdk_client
        self._timeout = httpx.Timeout(
            ATTEMPT_DEADLINE_SECONDS,
            connect=ATTEMPT_DEADLINE_SECONDS,
            read=ATTEMPT_DEADLINE_SECONDS,
            write=ATTEMPT_DEADLINE_SECONDS,
            pool=ATTEMPT_DEADLINE_SECONDS,
        )
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    def __repr__(self) -> str:
        return "CallControlClient()"

    def _validated_ids(self, call_control_id: object, command_id: object) -> tuple[str, str]:
        if self._closing or self._closed:
            raise CallControlClosedError("call_control_client_closed")
        if not _valid_call_control_id(call_control_id) or not _valid_command_id(command_id):
            raise CallControlInputError("call_control_input_invalid")
        return cast(str, call_control_id), str(command_id)

    @property
    def dispatch_available(self) -> bool:
        return not self._closing and not self._closed

    async def answer(self, call_control_id: str, *, command_id: UUID) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)

        async def action() -> object:
            return await self._client.calls.actions.answer(
                control_id,
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def transfer(
        self,
        call_control_id: str,
        request: TransferRequestV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if not isinstance(request, TransferRequestV1):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            return await self._client.calls.actions.transfer(
                control_id,
                to=request.to_e164,
                command_id=canonical_command,
                target_leg_client_state=request.target_leg_client_state,
                timeout_secs=20,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def start_streaming(
        self,
        call_control_id: str,
        request: StreamingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if not isinstance(request, StreamingStartV1):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            return await self._client.calls.actions.start_streaming(
                control_id,
                stream_track="inbound_track",
                stream_bidirectional_mode="rtp",
                stream_bidirectional_codec="PCMU",
                stream_bidirectional_sampling_rate=8000,
                stream_bidirectional_target_legs="self",
                stream_codec="PCMU",
                stream_url=request.stream_url,
                stream_auth_token=request.stream_auth_token.get_secret_value(),
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def start_recording(
        self,
        call_control_id: str,
        request: RecordingStartV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if not isinstance(request, RecordingStartV1):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            return await self._client.calls.actions.start_recording(
                control_id,
                channels="dual",
                format="wav",
                max_length=0,
                recording_track="both",
                timeout_secs=0,
                transcription=False,
                play_beep=request.play_beep,
                client_state=request.client_state.get_secret_value(),
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def stop_recording(
        self,
        call_control_id: str,
        request: RecordingStopV1,
        *,
        command_id: UUID,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if not isinstance(request, RecordingStopV1):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            if request.recording_id is None:
                return await self._client.calls.actions.stop_recording(
                    control_id,
                    client_state=request.client_state.get_secret_value(),
                    command_id=canonical_command,
                    timeout=self._timeout,
                )
            return await self._client.calls.actions.stop_recording(
                control_id,
                client_state=request.client_state.get_secret_value(),
                command_id=canonical_command,
                recording_id=request.recording_id,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def hangup(
        self,
        call_control_id: str,
        *,
        command_id: UUID,
        client_state: SecretStr | None = None,
    ) -> CallControlResult:
        control_id, canonical_command = self._validated_ids(call_control_id, command_id)
        if client_state is not None and (
            not isinstance(client_state, SecretStr)
            or not 0 < len(client_state.get_secret_value()) <= MAX_CLIENT_STATE_CHARS
        ):
            raise CallControlInputError("call_control_input_invalid")

        async def action() -> object:
            if client_state is not None:
                return await self._client.calls.actions.hangup(
                    control_id,
                    client_state=client_state.get_secret_value(),
                    command_id=canonical_command,
                    timeout=self._timeout,
                )
            return await self._client.calls.actions.hangup(
                control_id,
                command_id=canonical_command,
                timeout=self._timeout,
            )

        return await self._execute(action)

    async def list_recordings_one_page(
        self,
        *,
        call_control_id: str,
        call_leg_id: str | None,
        call_session_id: str | None,
        start_gte_iso: str,
        start_lte_iso: str,
        end_gte_iso: str,
        end_lte_iso: str,
        timeout_seconds: float,
    ) -> ProviderRecordingPageV1:
        from projetv0_voice.telnyx.recordings import ProviderRecordingPageV1

        if self._closing or self._closed:
            raise CallControlClosedError("call_control_client_closed")
        if (
            not _valid_call_control_id(call_control_id)
            or not _valid_optional_catalog_filter_id(call_leg_id)
            or not _valid_optional_catalog_filter_id(call_session_id)
            or not all(
                _valid_catalog_filter_time(value)
                for value in (
                    start_gte_iso,
                    start_lte_iso,
                    end_gte_iso,
                    end_lte_iso,
                )
            )
            or type(timeout_seconds) not in {float, int}
            or not 0 < timeout_seconds <= 1.0
        ):
            raise CallControlInputError("call_control_input_invalid")
        filters: dict[str, object] = {
            "call_control_id": call_control_id,
            "start_time": {"gte": start_gte_iso, "lte": start_lte_iso},
            "end_time": {"gte": end_gte_iso, "lte": end_lte_iso},
        }
        if call_leg_id is not None:
            filters["call_leg_id"] = call_leg_id
        if call_session_id is not None:
            filters["call_session_id"] = call_session_id
        failure: CatalogFailureKind | None = None
        result: ProviderRecordingPageV1 | None = None
        try:
            paginator = self._client.recordings.list(
                filter=cast(Any, filters),
                page_size=2,
                timeout=float(timeout_seconds),
            )
            async with asyncio.timeout(float(timeout_seconds)):
                page = await paginator
            meta = getattr(page, "meta", None)
            page_number = None if meta is None else getattr(meta, "page_number", None)
            total_pages = None if meta is None else getattr(meta, "total_pages", None)
            if page_number is not None and type(page_number) is not int:
                raise _RecordingCatalogInvalid
            if total_pages is not None and type(total_pages) is not int:
                raise _RecordingCatalogInvalid
            data = getattr(page, "data", None)
            if not isinstance(data, list):
                raise _RecordingCatalogInvalid
            items = tuple(_catalog_item(item) for item in data)
            result = ProviderRecordingPageV1(page_number, total_pages, items)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = _recording_catalog_failure_kind(error)
        if failure is not None:
            _raise_recording_catalog_failure(failure)
        if result is None:
            _raise_recording_catalog_failure("invalid")
        return result

    async def retrieve_recording(
        self,
        recording_id: str,
        *,
        timeout_seconds: float,
    ) -> ProviderRecordingV1:
        if self._closing or self._closed:
            raise CallControlClosedError("call_control_client_closed")
        if (
            not is_valid_provider_recording_id(recording_id)
            or type(timeout_seconds) not in {float, int}
            or not 0 < timeout_seconds <= 1.0
        ):
            raise CallControlInputError("call_control_input_invalid")
        failure: CatalogFailureKind | None = None
        result: ProviderRecordingV1 | None = None
        try:
            async with asyncio.timeout(float(timeout_seconds)):
                response = await self._client.recordings.retrieve(
                    recording_id,
                    timeout=float(timeout_seconds),
                )
            result = _catalog_item(getattr(response, "data", None))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = _recording_catalog_failure_kind(
                error,
                not_found_transient=True,
            )
        if failure is not None:
            _raise_recording_catalog_failure(failure)
        if result is None:
            _raise_recording_catalog_failure("invalid")
        return result

    async def delete_recording(
        self,
        recording_id: str,
        *,
        timeout_seconds: float,
    ) -> ProviderDeleteResultV1:
        from projetv0_voice.telnyx.recordings import ProviderDeleteResultV1

        if self._closing or self._closed:
            raise CallControlClosedError("call_control_client_closed")
        if (
            not is_valid_provider_recording_id(recording_id)
            or type(timeout_seconds) not in {float, int}
            or not 0 < timeout_seconds <= 1.0
        ):
            raise CallControlInputError("call_control_input_invalid")
        try:
            async with asyncio.timeout(float(timeout_seconds)):
                response = await self._client.recordings.delete(
                    recording_id,
                    timeout=float(timeout_seconds),
                )
            returned_id = _strict_catalog_text(
                getattr(getattr(response, "data", None), "id", None)
            )
            if returned_id is None:
                return ProviderDeleteResultV1("retry")
            return ProviderDeleteResultV1("deleted", returned_id)
        except asyncio.CancelledError:
            raise
        except telnyx.APIStatusError as error:
            status_code = error.status_code
            del error
            if status_code == 404:
                return ProviderDeleteResultV1("not_found")
            if status_code in {408, 409, 429} or status_code >= 500:
                return ProviderDeleteResultV1("retry")
            if 400 <= status_code < 500:
                return ProviderDeleteResultV1("failed")
            return ProviderDeleteResultV1("retry")
        except Exception:
            return ProviderDeleteResultV1("retry")

    async def _attempt(self, action: Callable[[], Awaitable[object]]) -> _Observation:
        loop = asyncio.get_running_loop()
        response: object | None = None
        try:
            async with asyncio.timeout_at(loop.time() + ATTEMPT_DEADLINE_SECONDS):
                response = await action()
        except TimeoutError:
            return _Observation("outcome_unknown", retry=True, ambiguous=True)
        except telnyx.APIStatusError as error:
            return _status_observation(error.status_code)
        except telnyx.APIResponseValidationError:
            return _Observation("outcome_unknown", retry=False, ambiguous=True)
        except telnyx.APIConnectionError as error:
            return _connection_observation(error)
        except Exception:
            return _Observation("outcome_unknown", retry=False, ambiguous=True)

        try:
            data = getattr(response, "data", None)
            result = getattr(data, "result", None)
        except Exception:
            return _Observation("outcome_unknown", retry=False, ambiguous=True)
        if result == "ok":
            return _Observation("accepted", retry=False, ambiguous=False)
        return _Observation("outcome_unknown", retry=False, ambiguous=True)

    async def _execute(self, action: Callable[[], Awaitable[object]]) -> CallControlResult:
        first = await self._attempt(action)
        if not first.retry:
            return CallControlResult(first.outcome)
        second = await self._attempt(action)
        if second.outcome == "accepted":
            return CallControlResult("accepted")
        if first.ambiguous:
            return CallControlResult("outcome_unknown")
        return CallControlResult(second.outcome)

    async def aclose(self) -> None:
        close_failed = False
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            try:
                async with asyncio.timeout(CLOSE_DEADLINE_SECONDS):
                    await self._client.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                close_failed = True
            self._closed = True
        if close_failed:
            raise CallControlCloseError("call_control_close_failed")
