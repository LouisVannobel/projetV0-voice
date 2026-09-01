"""Close dependency logging before loading observability/runtime dependencies."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading

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
_configured = False


def _flush_stderr() -> bool:
    try:
        if sys.stderr is not None:
            sys.stderr.flush()
    except BaseException:
        return False
    return True


def _configure_onnxruntime_native_logging() -> bool:
    saved_stderr: int | None = None
    devnull: int | None = None
    redirected = False
    failed = False
    original_inheritable = True
    try:
        if not _flush_stderr():
            failed = True
        else:
            original_inheritable = os.get_inheritable(2)
            saved_stderr = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2, inheritable=original_inheritable)
            redirected = True
            onnxruntime = importlib.import_module("onnxruntime")
            onnxruntime.set_default_logger_severity(4)
    except BaseException:
        failed = True
    finally:
        if redirected:
            if not _flush_stderr():
                failed = True
            try:
                assert saved_stderr is not None
                os.dup2(saved_stderr, 2, inheritable=original_inheritable)
            except BaseException:
                failed = True
        for descriptor in (devnull, saved_stderr):
            if descriptor is None:
                continue
            try:
                os.close(descriptor)
            except BaseException:
                failed = True
    return not failed


def _configure_python_dependency_logging() -> bool:
    try:
        for name in _DEPENDENCY_LOGGERS:
            dependency_logger = logging.getLogger(name)
            dependency_logger.handlers[:] = [logging.NullHandler()]
            dependency_logger.propagate = False
            dependency_logger.disabled = True
            dependency_logger.setLevel(logging.CRITICAL + 1)

        from loguru import logger

        logger.disable("pipecat")
    except BaseException:
        return False
    return True


def configure_dependency_logging(token: ObservabilityBootstrapToken) -> None:
    """Disable all approved dependency logging namespaces idempotently."""

    global _configured

    if type(token) is not ObservabilityBootstrapToken:
        raise ValueError("observability_bootstrap_token_invalid") from None
    if _configured:
        return
    if threading.active_count() != 1:
        raise RuntimeError("dependency_logging_phase_invalid") from None
    if not _configure_onnxruntime_native_logging() or not _configure_python_dependency_logging():
        raise RuntimeError("dependency_logging_configuration_failed") from None
    _configured = True


__all__ = ["configure_dependency_logging"]
