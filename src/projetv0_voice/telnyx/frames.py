"""Project-owned Telnyx transport message frames."""

from __future__ import annotations

from typing import cast

from pipecat.frames.frames import OutputTransportMessageFrame

MAX_MARK_NAME_BYTES = 256


class TelnyxSerializerError(RuntimeError):
    """A constant-safe Telnyx serializer contract error."""


def bounded_utf8_text(value: object, *, maximum_bytes: int) -> str:
    encoded: bytes | None = None
    if isinstance(value, str) and value:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            encoded = None
    if encoded is None or len(encoded) > maximum_bytes:
        raise TelnyxSerializerError("telnyx_serializer_invalid")
    return cast(str, value)


class TelnyxMarkFrame(OutputTransportMessageFrame):
    """An ordered outbound Telnyx mark message."""

    def __init__(self, name: str) -> None:
        mark_name = bounded_utf8_text(name, maximum_bytes=MAX_MARK_NAME_BYTES)
        self.mark_name = mark_name
        super().__init__(message={"event": "mark", "mark": {"name": mark_name}})
