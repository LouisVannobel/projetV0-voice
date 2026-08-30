"""Stdlib-only validation before any observability dependency import."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class ObservabilityBootstrapToken:
    """Opaque evidence that the real process environment passed the guard."""


_FORBIDDEN_EXACT = frozenset({"TELNYX_LOG", "OPENAI_LOG"})
_ASCII_WHITESPACE = frozenset(" \t\n\v\f\r")


def _valid_endpoint(endpoint: object) -> bool:
    if type(endpoint) is not str:
        return False
    if not endpoint or "\\" in endpoint or any(char in _ASCII_WHITESPACE for char in endpoint):
        return False
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and (port is None or 0 <= port <= 65535)
        and parsed.path.endswith("/v1/metrics")
        and not parsed.query
        and not parsed.fragment
    )


def _validate_observability_mapping(
    environment: Mapping[Any, Any],
) -> ObservabilityBootstrapToken:
    for key in environment:
        if (
            type(key) is not str
            or key.startswith(("OTEL_", "LOGURU_"))
            or key in _FORBIDDEN_EXACT
        ):
            raise RuntimeError("observability_environment_forbidden") from None

    endpoint = environment.get("VOICE_OTLP_HTTP_ENDPOINT")
    if not _valid_endpoint(endpoint):
        raise RuntimeError("observability_endpoint_invalid") from None
    return ObservabilityBootstrapToken()


def validate_observability_environment() -> ObservabilityBootstrapToken:
    """Validate the unfiltered real process environment."""

    return _validate_observability_mapping(os.environ)


__all__ = ["ObservabilityBootstrapToken", "validate_observability_environment"]
