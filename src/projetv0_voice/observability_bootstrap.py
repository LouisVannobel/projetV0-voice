"""Stdlib-only validation before any observability dependency import."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from hmac import compare_digest
from typing import Any

from projetv0_voice.runtime_config import (
    RuntimeEnvironmentCapture,
    _valid_otlp_http_endpoint,
)


@dataclass(frozen=True, slots=True, repr=False)
class ObservabilityBootstrapToken:
    """Opaque evidence that the real process environment passed the guard."""

    _endpoint: str | None = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        return "ObservabilityBootstrapToken()"


_FORBIDDEN_EXACT = frozenset(
    {
        "TELNYX_LOG",
        "OPENAI_LOG",
        "VOICE_TELNYX_API_KEY",
        "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY",
        "VOICE_OPENROUTER_API_KEY",
        "VOICE_POSTGRES_DSN",
    }
)


def _valid_endpoint(endpoint: object) -> bool:
    return _valid_otlp_http_endpoint(endpoint)


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
    assert type(endpoint) is str
    return ObservabilityBootstrapToken(endpoint)


def issue_observability_token(
    capture: RuntimeEnvironmentCapture,
    endpoint: str,
) -> ObservabilityBootstrapToken:
    """Issue opaque evidence only for the parsed endpoint from this capture."""

    if type(capture) is not RuntimeEnvironmentCapture:
        raise RuntimeError("runtime_capture_invalid") from None
    captured_endpoint = capture._value("VOICE_OTLP_HTTP_ENDPOINT")  # noqa: SLF001
    if (
        type(endpoint) is not str
        or type(captured_endpoint) is not str
        or not _valid_endpoint(endpoint)
        or not compare_digest(captured_endpoint, endpoint)
    ):
        raise RuntimeError("observability_endpoint_mismatch") from None
    return ObservabilityBootstrapToken(endpoint)


def _token_matches_endpoint(token: object, endpoint: object) -> bool:
    return (
        type(token) is ObservabilityBootstrapToken
        and type(endpoint) is str
        and token._endpoint is not None  # noqa: SLF001
        and compare_digest(token._endpoint, endpoint)  # noqa: SLF001
    )


def validate_observability_environment() -> ObservabilityBootstrapToken:
    """Validate the unfiltered real process environment."""

    return _validate_observability_mapping(os.environ)


__all__ = [
    "ObservabilityBootstrapToken",
    "issue_observability_token",
    "validate_observability_environment",
]
