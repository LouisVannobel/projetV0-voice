"""Stdlib-only validation before any observability dependency import."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from hmac import compare_digest
from typing import Any, TypeGuard

from projetv0_voice.runtime_config import _valid_otlp_http_endpoint

_TOKEN_CONSTRUCTOR_SENTINEL = object()
_PRODUCTION_TOKEN_ISSUER = object()


@dataclass(frozen=True, slots=True, repr=False, init=False)
class ObservabilityBootstrapToken:
    """Opaque evidence that the real process environment passed the guard."""

    _issuer: object | None = field(default=None, repr=False, compare=False)
    _endpoint: str | None = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        sentinel: object,
        *,
        endpoint: str,
        issuer: object | None = None,
    ) -> None:
        if sentinel is not _TOKEN_CONSTRUCTOR_SENTINEL:
            raise TypeError("observability_token_constructor_private")
        object.__setattr__(self, "_issuer", issuer)
        object.__setattr__(self, "_endpoint", endpoint)

    def __repr__(self) -> str:
        return "ObservabilityBootstrapToken()"


def _new_token(
    *,
    endpoint: str,
    issuer: object | None = None,
) -> ObservabilityBootstrapToken:
    return ObservabilityBootstrapToken(
        _TOKEN_CONSTRUCTOR_SENTINEL,
        endpoint=endpoint,
        issuer=issuer,
    )


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
    return _new_token(endpoint=endpoint)


def _issue_parser_observability_token(endpoint: str) -> ObservabilityBootstrapToken:
    """Issue production evidence for the fully validated parser result."""

    if not _valid_endpoint(endpoint):
        raise RuntimeError("observability_endpoint_invalid") from None
    return _new_token(
        endpoint=endpoint,
        issuer=_PRODUCTION_TOKEN_ISSUER,
    )


def _is_production_token(
    token: object,
) -> TypeGuard[ObservabilityBootstrapToken]:
    return (
        type(token) is ObservabilityBootstrapToken
        and token._issuer is _PRODUCTION_TOKEN_ISSUER  # noqa: SLF001
        and type(token._endpoint) is str  # noqa: SLF001
    )


def _token_matches_endpoint(token: object, endpoint: object) -> bool:
    if not _is_production_token(token) or type(endpoint) is not str:
        return False
    token_endpoint = token._endpoint  # noqa: SLF001
    assert type(token_endpoint) is str
    return compare_digest(token_endpoint, endpoint)


def validate_observability_environment() -> ObservabilityBootstrapToken:
    """Validate the unfiltered real process environment."""

    return _validate_observability_mapping(os.environ)


__all__ = [
    "ObservabilityBootstrapToken",
    "validate_observability_environment",
]
