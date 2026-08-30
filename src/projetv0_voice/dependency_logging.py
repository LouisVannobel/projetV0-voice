"""Close dependency logging before loading observability/runtime dependencies."""

from __future__ import annotations

import logging

from projetv0_voice.observability_bootstrap import ObservabilityBootstrapToken

_DEPENDENCY_LOGGERS = (
    "telnyx",
    "openai",
    "httpx",
    "httpcore",
    "psycopg",
    "psycopg.pool",
    "urllib3",
    "opentelemetry.exporter.otlp.proto.http.metric_exporter",
    "opentelemetry.sdk.metrics._internal.export",
)


def configure_dependency_logging(token: ObservabilityBootstrapToken) -> None:
    """Disable all approved dependency logging namespaces idempotently."""

    if type(token) is not ObservabilityBootstrapToken:
        raise ValueError("observability_bootstrap_token_invalid") from None
    for name in _DEPENDENCY_LOGGERS:
        dependency_logger = logging.getLogger(name)
        dependency_logger.handlers[:] = [logging.NullHandler()]
        dependency_logger.propagate = False
        dependency_logger.disabled = True
        dependency_logger.setLevel(logging.CRITICAL + 1)

    from loguru import logger

    logger.disable("pipecat")


__all__ = ["configure_dependency_logging"]
