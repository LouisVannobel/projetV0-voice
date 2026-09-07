"""Close dependency logging before loading observability/runtime dependencies."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable
from enum import Enum, auto
from typing import NoReturn

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
_hard_exit: Callable[[int], NoReturn] = os._exit


class _DescriptorAuthority(Enum):
    NONE = auto()
    ACQUIRE_UNCERTAIN = auto()
    OWNED = auto()
    CLOSE_UNCERTAIN = auto()
    CLOSED = auto()


class _FD2Authority(Enum):
    ORIGINAL = auto()
    REDIRECT_UNCERTAIN = auto()
    REDIRECTED = auto()
    RESTORE_UNCERTAIN = auto()
    RESTORED = auto()


class _NativeBoundaryResult(Enum):
    OK = auto()
    SOFT_FAILURE = auto()
    FATAL = auto()


def _flush_stderr() -> bool:
    try:
        if sys.stderr is not None:
            sys.stderr.flush()
    except BaseException:
        return False
    return True


def _configure_onnxruntime_native_logging() -> bool:
    saved_stderr: int | None = None
    saved_stderr_authority = _DescriptorAuthority.NONE
    devnull: int | None = None
    devnull_authority = _DescriptorAuthority.NONE
    fd2_authority = _FD2Authority.ORIGINAL
    result = _NativeBoundaryResult.OK
    original_inheritable = True

    try:
        if not _flush_stderr():
            result = _NativeBoundaryResult.SOFT_FAILURE
        else:
            original_inheritable = os.get_inheritable(2)

            saved_stderr_authority = _DescriptorAuthority.ACQUIRE_UNCERTAIN
            saved_stderr = os.dup(2)
            saved_stderr_authority = _DescriptorAuthority.OWNED

            devnull_authority = _DescriptorAuthority.ACQUIRE_UNCERTAIN
            devnull = os.open(os.devnull, os.O_WRONLY)
            devnull_authority = _DescriptorAuthority.OWNED

            fd2_authority = _FD2Authority.REDIRECT_UNCERTAIN
            _redirect_result = os.dup2(
                devnull,
                2,
                inheritable=original_inheritable,
            )
            fd2_authority = _FD2Authority.REDIRECTED

            onnxruntime = importlib.import_module("onnxruntime")
            onnxruntime.set_default_logger_severity(4)
    except BaseException:
        uncertain = (
            saved_stderr_authority is _DescriptorAuthority.ACQUIRE_UNCERTAIN
            or devnull_authority is _DescriptorAuthority.ACQUIRE_UNCERTAIN
            or fd2_authority is _FD2Authority.REDIRECT_UNCERTAIN
        )
        if uncertain:
            result = _NativeBoundaryResult.FATAL
        elif result is _NativeBoundaryResult.OK:
            result = _NativeBoundaryResult.SOFT_FAILURE

    if fd2_authority in (
        _FD2Authority.REDIRECT_UNCERTAIN,
        _FD2Authority.REDIRECTED,
    ):
        if (
            result is not _NativeBoundaryResult.FATAL
            and not _flush_stderr()
            and result is _NativeBoundaryResult.OK
        ):
            result = _NativeBoundaryResult.SOFT_FAILURE

        if (
            saved_stderr_authority is _DescriptorAuthority.OWNED
            and saved_stderr is not None
        ):
            try:
                fd2_authority = _FD2Authority.RESTORE_UNCERTAIN
                _restore_result = os.dup2(
                    saved_stderr,
                    2,
                    inheritable=original_inheritable,
                )
                fd2_authority = _FD2Authority.RESTORED
            except BaseException:
                result = _NativeBoundaryResult.FATAL
        else:
            result = _NativeBoundaryResult.FATAL

    if devnull_authority is _DescriptorAuthority.OWNED and devnull is not None:
        try:
            devnull_authority = _DescriptorAuthority.CLOSE_UNCERTAIN
            os.close(devnull)
            devnull_authority = _DescriptorAuthority.CLOSED
        except BaseException:
            result = _NativeBoundaryResult.FATAL

    if (
        saved_stderr_authority is _DescriptorAuthority.OWNED
        and saved_stderr is not None
    ):
        try:
            saved_stderr_authority = _DescriptorAuthority.CLOSE_UNCERTAIN
            os.close(saved_stderr)
            saved_stderr_authority = _DescriptorAuthority.CLOSED
        except BaseException:
            result = _NativeBoundaryResult.FATAL

    if saved_stderr_authority not in (
        _DescriptorAuthority.NONE,
        _DescriptorAuthority.CLOSED,
    ):
        result = _NativeBoundaryResult.FATAL
    if devnull_authority not in (
        _DescriptorAuthority.NONE,
        _DescriptorAuthority.CLOSED,
    ):
        result = _NativeBoundaryResult.FATAL
    if fd2_authority not in (
        _FD2Authority.ORIGINAL,
        _FD2Authority.RESTORED,
    ):
        result = _NativeBoundaryResult.FATAL

    if result is _NativeBoundaryResult.FATAL:
        _hard_exit(71)
        os._exit(71)

    return result is _NativeBoundaryResult.OK


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
    if (
        threading.active_count() != 1
        or threading.current_thread() is not threading.main_thread()
    ):
        raise RuntimeError("dependency_logging_phase_invalid") from None
    if not _configure_onnxruntime_native_logging() or not _configure_python_dependency_logging():
        raise RuntimeError("dependency_logging_configuration_failed") from None
    _configured = True


__all__ = ["configure_dependency_logging"]
