"""Thin, upstream-first Telnyx serializer extension."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import Frame, InputTransportMessageFrame
from pipecat.serializers.telnyx import TelnyxFrameSerializer

from projetv0_voice.telnyx.frames import (
    MAX_MARK_NAME_BYTES,
    TelnyxMarkFrame,
    TelnyxSerializerError,
    bounded_utf8_text,
)

MAX_WEBSOCKET_MESSAGE_BYTES = 524_288
MAX_EXTENSION_MESSAGE_BYTES = 65_536
MAX_STREAM_ID_BYTES = 1_024
MAX_CALL_CONTROL_ID_BYTES = 1_024
MAX_ERROR_FIELD_BYTES = 256
MAX_ERROR_CODE = 2_147_483_647


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


def _load_json(
    data: str | bytes,
    *,
    maximum_bytes: int,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], dict[str, Any]] | None = None,
) -> object:
    parsed: object | None = None
    invalid = False
    try:
        raw_bytes = data.encode("utf-8") if isinstance(data, str) else data
        if len(raw_bytes) > maximum_bytes:
            invalid = True
        else:
            parsed = json.loads(
                data,
                object_pairs_hook=object_pairs_hook,
                parse_constant=_reject_constant,
            )
    except (UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError, _InvalidJson):
        invalid = True
    if invalid:
        raise TelnyxSerializerError("telnyx_serializer_invalid")
    return parsed


def _require_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TelnyxSerializerError("telnyx_serializer_invalid")
    return value


class ProjetV0TelnyxFrameSerializer(TelnyxFrameSerializer):
    """Add only bounded Telnyx mark/stop/error messages to the native serializer."""

    def __init__(self, stream_id: str, *, expected_call_control_id: str) -> None:
        authenticated_stream_id = bounded_utf8_text(
            stream_id,
            maximum_bytes=MAX_STREAM_ID_BYTES,
        )
        self._expected_call_control_id = bounded_utf8_text(
            expected_call_control_id,
            maximum_bytes=MAX_CALL_CONTROL_ID_BYTES,
        )
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=8000,
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
            auto_hang_up=False,
        )
        super().__init__(
            stream_id=authenticated_stream_id,
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
            params=params,
        )

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, TelnyxMarkFrame):
            mark_name = bounded_utf8_text(
                frame.mark_name,
                maximum_bytes=MAX_MARK_NAME_BYTES,
            )
            return json.dumps(
                {"event": "mark", "mark": {"name": mark_name}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return await super().serialize(frame)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        initial = _load_json(data, maximum_bytes=MAX_WEBSOCKET_MESSAGE_BYTES)
        if not isinstance(initial, dict) or initial.get("event") not in {
            "mark",
            "stop",
            "error",
        }:
            return await super().deserialize(data)

        message = _require_mapping(
            _load_json(
                data,
                maximum_bytes=MAX_EXTENSION_MESSAGE_BYTES,
                object_pairs_hook=_unique_object,
            )
        )
        event = message.get("event")
        stream_id = bounded_utf8_text(
            message.get("stream_id"),
            maximum_bytes=MAX_STREAM_ID_BYTES,
        )
        if stream_id != self._stream_id:
            raise TelnyxSerializerError("telnyx_serializer_invalid")

        if event == "mark":
            mark = _require_mapping(message.get("mark"))
            name = bounded_utf8_text(mark.get("name"), maximum_bytes=MAX_MARK_NAME_BYTES)
            return InputTransportMessageFrame(
                message={"event": "mark", "mark": {"name": name}}
            )
        if event == "stop":
            stop = _require_mapping(message.get("stop"))
            call_control_id = bounded_utf8_text(
                stop.get("call_control_id"),
                maximum_bytes=MAX_CALL_CONTROL_ID_BYTES,
            )
            if not hmac.compare_digest(
                call_control_id.encode("utf-8"),
                self._expected_call_control_id.encode("utf-8"),
            ):
                raise TelnyxSerializerError("telnyx_serializer_invalid")
            return InputTransportMessageFrame(message={"event": "stop"})
        if event == "error":
            payload = _require_mapping(message.get("payload"))
            code = payload.get("code")
            if type(code) is not int or not 0 <= code <= MAX_ERROR_CODE:
                raise TelnyxSerializerError("telnyx_serializer_invalid")
            title = bounded_utf8_text(
                payload.get("title"),
                maximum_bytes=MAX_ERROR_FIELD_BYTES,
            )
            return InputTransportMessageFrame(
                message={"event": "error", "payload": {"code": code, "title": title}}
            )
        raise TelnyxSerializerError("telnyx_serializer_invalid")


__all__ = ["ProjetV0TelnyxFrameSerializer", "TelnyxSerializerError"]
