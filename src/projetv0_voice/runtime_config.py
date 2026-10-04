"""Stdlib-only parse-once production runtime configuration boundary."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit
from uuid import UUID

if TYPE_CHECKING:
    from pydantic import SecretStr

    from projetv0_voice.observability_bootstrap import ObservabilityBootstrapToken

type RuntimeMode = Literal[
    "strict",
    "qualification_candidate",
    "qualification_override",
]

_ALLOWED_ENVIRONMENT_NAMES = (
    "VOICE_RUNTIME_MODE",
    "VOICE_DEPLOYMENT_ID",
    "VOICE_RUNTIME_CONTRACT_PATH",
    "VOICE_AGENT_BUNDLE_PATH",
    "VOICE_QUALIFIED_PROFILE_PATH",
    "VOICE_QUALIFICATION_CANDIDATE_PATH",
    "VOICE_QUALIFICATION_OVERRIDE_PATH",
    "VOICE_KEYRING_PATH",
    "VOICE_SQLITE_PATH",
    "VOICE_RECORDING_ARCHIVE_DIRECTORY",
    "VOICE_RECORDING_DOWNLOAD_ORIGINS",
    "VOICE_RUNTIME_CONTRACT_SHA256",
    "VOICE_IMAGE_DIGEST",
    "VOICE_AGENT_BUNDLE_SHA256",
    "VOICE_INFERENCE_PROFILE_SHA256",
    "VOICE_QUALIFICATION_RUN_ID",
    "VOICE_BENCHMARK_DID_SHA256",
    "VOICE_DEPLOYMENT_MAX_CALLS",
    "VOICE_HANDSHAKE_TIMEOUT_SECONDS",
    "VOICE_CALL_IDLE_TIMEOUT_SECONDS",
    "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS",
    "VOICE_PRE_DRAIN_GRACE_SECONDS",
    "VOICE_UVICORN_GRACE_SECONDS",
    "VOICE_SHUTDOWN_GRACE_SECONDS",
    "VOICE_TELNYX_API_KEY_FILE",
    "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE",
    "VOICE_OPENROUTER_API_KEY_FILE",
    "VOICE_POSTGRES_DSN_FILE",
    "VOICE_TELNYX_MEDIA_WSS_URL",
    "VOICE_OTLP_HTTP_ENDPOINT",
    "VOICE_BIND_HOST",
    "VOICE_BIND_PORT",
)
_ENVIRONMENT_INDEX = {
    name: index for index, name in enumerate(_ALLOWED_ENVIRONMENT_NAMES)
}
_DIRECT_SECRET_NAMES = frozenset(
    {
        "VOICE_TELNYX_API_KEY",
        "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY",
        "VOICE_OPENROUTER_API_KEY",
        "VOICE_POSTGRES_DSN",
    }
)
_FORBIDDEN_EXACT = _DIRECT_SECRET_NAMES | {"TELNYX_LOG", "OPENAI_LOG", "TELNYX_BASE_URL"}
_MISSING = object()
_ASCII_WHITESPACE = frozenset(" \t\n\v\f\r")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*"
    r"(?::[0-9]{1,5})?"
    r"/[a-z0-9]+(?:[._-][a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"@sha256:[0-9a-f]{64}$"
)
_DECIMAL_INTEGER = re.compile(r"^[1-9][0-9]*$")
_MAX_SECRET_BYTES = 16_384
_TRUSTED_SECRET_DIRECTORY = PurePosixPath("/run/secrets")


class _SettingsInvalid(Exception):
    pass


class _EnvironmentForbidden(Exception):
    pass


class _SecretInvalid(Exception):
    pass


class _PlatformUnsupported(Exception):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeEnvironmentCapture:
    """Private immutable snapshot populated only after the key-only scan."""

    _values: tuple[object, ...]

    def __repr__(self) -> str:
        return "RuntimeEnvironmentCapture()"

    def _value(self, name: str) -> object:
        return self._values[_ENVIRONMENT_INDEX[name]]


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeSettingsV1:
    """Sole immutable authority for every Task 10C runtime setting."""

    runtime_mode: RuntimeMode
    deployment_id: str
    runtime_contract_path: PurePosixPath
    agent_bundle_path: PurePosixPath
    qualified_profile_path: PurePosixPath | None
    qualification_candidate_path: PurePosixPath | None
    qualification_override_path: PurePosixPath | None
    keyring_path: PurePosixPath
    sqlite_path: PurePosixPath
    runtime_contract_sha256: str
    image_digest: str
    agent_bundle_sha256: str
    inference_profile_sha256: str
    qualification_run_id: UUID | None
    benchmark_did_sha256: str | None
    deployment_max_calls: int
    handshake_timeout_seconds: int
    call_idle_timeout_seconds: int
    call_cleanup_phase_timeout_seconds: int
    pre_drain_grace_seconds: int
    uvicorn_grace_seconds: int
    shutdown_grace_seconds: int
    telnyx_api_key_file: PurePosixPath
    telnyx_webhook_public_key_file: PurePosixPath
    openrouter_api_key_file: PurePosixPath
    postgres_dsn_file: PurePosixPath
    telnyx_media_wss_url: str
    otlp_http_endpoint: str
    bind_host: str
    bind_port: int
    recording_archive_directory: PurePosixPath | None = None
    recording_download_origins: tuple[str, ...] = ()
    _observability_token: ObservabilityBootstrapToken | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        return "RuntimeSettingsV1()"

    def observability_token(self) -> ObservabilityBootstrapToken:
        token = self._observability_token
        if token is None:
            raise RuntimeError("observability_token_unavailable") from None
        return token


def capture_runtime_environment(
    mapping: Mapping[str, object],
) -> RuntimeEnvironmentCapture:
    """Scan names first, then fetch each allowed value exactly once."""

    forbidden = False
    scan_failed = False
    try:
        for key in mapping:
            if (
                type(key) is not str
                or key in _FORBIDDEN_EXACT
                or key.startswith(("OTEL_", "LOGURU_"))
            ):
                raise _EnvironmentForbidden
    except _EnvironmentForbidden:
        forbidden = True
    except BaseException:
        scan_failed = True
    if forbidden:
        raise RuntimeError("runtime_environment_forbidden") from None
    if scan_failed:
        raise RuntimeError("runtime_environment_scan_failed") from None

    captured: list[object] = []
    for name in _ALLOWED_ENVIRONMENT_NAMES:
        capture_failed = False
        value: object = _MISSING
        try:
            value = mapping[name]
        except KeyError:
            pass
        except BaseException:
            capture_failed = True
        if capture_failed:
            raise RuntimeError("runtime_environment_capture_failed") from None
        captured.append(value)
    return RuntimeEnvironmentCapture(tuple(captured))


def _required_string(capture: RuntimeEnvironmentCapture, name: str) -> str:
    value = capture._value(name)  # noqa: SLF001
    if type(value) is not str or not value or value != value.strip(" \t\n\v\f\r"):
        raise _SettingsInvalid
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise _SettingsInvalid
    return value


def _optional_string(capture: RuntimeEnvironmentCapture, name: str) -> str | None:
    value = capture._value(name)  # noqa: SLF001
    if value is _MISSING:
        return None
    return _required_string(capture, name)


def _path(value: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if (
        not candidate.is_absolute()
        or candidate == PurePosixPath("/")
        or any(part in {".", ".."} for part in value.split("/"))
        or "\\" in value
    ):
        raise _SettingsInvalid
    return candidate


def _required_path(capture: RuntimeEnvironmentCapture, name: str) -> PurePosixPath:
    return _path(_required_string(capture, name))


def _optional_path(
    capture: RuntimeEnvironmentCapture,
    name: str,
) -> PurePosixPath | None:
    value = _optional_string(capture, name)
    return None if value is None else _path(value)


def _secret_path(capture: RuntimeEnvironmentCapture, name: str) -> PurePosixPath:
    value = _required_string(capture, name)
    candidate = _path(value)
    trusted = PurePosixPath("/run/secrets")
    if candidate.parent != trusted or value != f"{trusted}/{candidate.name}":
        raise _SettingsInvalid
    return candidate


def _sha256(capture: RuntimeEnvironmentCapture, name: str) -> str:
    value = _required_string(capture, name)
    if _SHA256.fullmatch(value) is None:
        raise _SettingsInvalid
    return value


def _optional_sha256(capture: RuntimeEnvironmentCapture, name: str) -> str | None:
    value = _optional_string(capture, name)
    if value is None:
        return None
    if _SHA256.fullmatch(value) is None:
        raise _SettingsInvalid
    return value


def _integer(
    capture: RuntimeEnvironmentCapture,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _required_string(capture, name)
    if (
        _DECIMAL_INTEGER.fullmatch(value) is None
        or len(value) > len(str(maximum))
    ):
        raise _SettingsInvalid
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise _SettingsInvalid
    return parsed


def _valid_web_url(value: object, *, schemes: frozenset[str]) -> bool:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(char in _ASCII_WHITESPACE for char in value)
    ):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in schemes
        and bool(parsed.netloc)
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and (port is None or 0 <= port <= 65535)
        and not parsed.query
        and not parsed.fragment
    )


def _valid_otlp_http_endpoint(value: object) -> bool:
    if not _valid_web_url(value, schemes=frozenset({"http", "https"})):
        return False
    assert type(value) is str
    return urlsplit(value).path.endswith("/v1/metrics")


def _runtime_mode(capture: RuntimeEnvironmentCapture) -> RuntimeMode:
    value = _required_string(capture, "VOICE_RUNTIME_MODE")
    if value == "strict":
        return "strict"
    if value == "qualification_candidate":
        return "qualification_candidate"
    if value == "qualification_override":
        return "qualification_override"
    raise _SettingsInvalid


def _optional_uuid(capture: RuntimeEnvironmentCapture, name: str) -> UUID | None:
    value = _optional_string(capture, name)
    if value is None:
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise _SettingsInvalid from None
    if str(parsed) != value:
        raise _SettingsInvalid
    return parsed


def _validate_mode(
    mode: RuntimeMode,
    *,
    qualified_profile_path: PurePosixPath | None,
    qualification_candidate_path: PurePosixPath | None,
    qualification_override_path: PurePosixPath | None,
    qualification_run_id: UUID | None,
    benchmark_did_sha256: str | None,
    deployment_max_calls: int,
) -> None:
    if mode == "strict":
        valid = (
            qualified_profile_path is not None
            and qualification_candidate_path is None
            and qualification_override_path is None
            and qualification_run_id is None
            and benchmark_did_sha256 is None
        )
    elif mode == "qualification_candidate":
        valid = (
            qualified_profile_path is None
            and qualification_candidate_path is not None
            and qualification_override_path is None
            and qualification_run_id is not None
            and benchmark_did_sha256 is not None
        )
    else:
        valid = (
            qualified_profile_path is not None
            and qualification_candidate_path is None
            and qualification_override_path is not None
            and qualification_run_id is not None
            and benchmark_did_sha256 is None
            and deployment_max_calls in {15, 20}
        )
    if not valid:
        raise _SettingsInvalid


def _recording_origins(capture: RuntimeEnvironmentCapture) -> tuple[str, ...]:
    directory = _optional_string(capture,"VOICE_RECORDING_ARCHIVE_DIRECTORY")
    origins = _optional_string(capture,"VOICE_RECORDING_DOWNLOAD_ORIGINS")
    if (directory is None) != (origins is None):
        raise _SettingsInvalid
    if origins is None:
        return ()
    values = tuple(origins.split(","))
    if not 1 <= len(values) <= 8 or len(set(values)) != len(values):
        raise _SettingsInvalid
    for value in values:
        parsed=urlsplit(value)
        if (parsed.scheme!="https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.port is not None or parsed.path
            or parsed.query or parsed.fragment or value!="https://"+parsed.hostname.lower()):
            raise _SettingsInvalid
    return values


def parse_runtime_settings(
    capture: RuntimeEnvironmentCapture,
    *,
    geteuid: Callable[[], int] | None = None,
    getegid: Callable[[], int] | None = None,
) -> RuntimeSettingsV1:
    """Parse the captured values exactly once under the runtime identity."""

    if type(capture) is not RuntimeEnvironmentCapture:
        raise RuntimeError("runtime_capture_invalid") from None
    uid_function = geteuid if geteuid is not None else getattr(os, "geteuid", None)
    gid_function = getegid if getegid is not None else getattr(os, "getegid", None)
    if uid_function is None or gid_function is None:
        raise RuntimeError("runtime_platform_unsupported") from None
    identity_failed = False
    uid: object = _MISSING
    gid: object = _MISSING
    try:
        uid = uid_function()
        gid = gid_function()
    except BaseException:
        identity_failed = True
    if identity_failed:
        raise RuntimeError("runtime_identity_invalid") from None
    if type(uid) is not int or type(gid) is not int or (uid, gid) != (10001, 10001):
        raise RuntimeError("runtime_identity_invalid") from None

    try:
        mode = _runtime_mode(capture)
        qualified_profile_path = _optional_path(capture, "VOICE_QUALIFIED_PROFILE_PATH")
        qualification_candidate_path = _optional_path(
            capture,
            "VOICE_QUALIFICATION_CANDIDATE_PATH",
        )
        qualification_override_path = _optional_path(
            capture,
            "VOICE_QUALIFICATION_OVERRIDE_PATH",
        )
        qualification_run_id = _optional_uuid(capture, "VOICE_QUALIFICATION_RUN_ID")
        benchmark_did_sha256 = _optional_sha256(
            capture,
            "VOICE_BENCHMARK_DID_SHA256",
        )
        deployment_max_calls = _integer(
            capture,
            "VOICE_DEPLOYMENT_MAX_CALLS",
            minimum=1,
            maximum=(1 << 31) - 1,
        )
        _validate_mode(
            mode,
            qualified_profile_path=qualified_profile_path,
            qualification_candidate_path=qualification_candidate_path,
            qualification_override_path=qualification_override_path,
            qualification_run_id=qualification_run_id,
            benchmark_did_sha256=benchmark_did_sha256,
            deployment_max_calls=deployment_max_calls,
        )
        image_digest = _required_string(capture, "VOICE_IMAGE_DIGEST")
        if _IMAGE_DIGEST.fullmatch(image_digest) is None:
            raise _SettingsInvalid
        telnyx_media_wss_url = _required_string(
            capture,
            "VOICE_TELNYX_MEDIA_WSS_URL",
        )
        if not _valid_web_url(telnyx_media_wss_url, schemes=frozenset({"wss"})):
            raise _SettingsInvalid
        otlp_http_endpoint = _required_string(capture, "VOICE_OTLP_HTTP_ENDPOINT")
        if not _valid_otlp_http_endpoint(otlp_http_endpoint):
            raise _SettingsInvalid
        bind_host = _required_string(capture, "VOICE_BIND_HOST")
        if any(char in bind_host for char in "/\\:[]") and bind_host != "::":
            raise _SettingsInvalid
        settings = RuntimeSettingsV1(
            runtime_mode=mode,
            deployment_id=_required_string(capture, "VOICE_DEPLOYMENT_ID"),
            runtime_contract_path=_required_path(capture, "VOICE_RUNTIME_CONTRACT_PATH"),
            agent_bundle_path=_required_path(capture, "VOICE_AGENT_BUNDLE_PATH"),
            qualified_profile_path=qualified_profile_path,
            qualification_candidate_path=qualification_candidate_path,
            qualification_override_path=qualification_override_path,
            keyring_path=_required_path(capture, "VOICE_KEYRING_PATH"),
            sqlite_path=_required_path(capture, "VOICE_SQLITE_PATH"),
            recording_archive_directory=_optional_path(capture,"VOICE_RECORDING_ARCHIVE_DIRECTORY"),
            recording_download_origins=_recording_origins(capture),
            runtime_contract_sha256=_sha256(capture, "VOICE_RUNTIME_CONTRACT_SHA256"),
            image_digest=image_digest,
            agent_bundle_sha256=_sha256(capture, "VOICE_AGENT_BUNDLE_SHA256"),
            inference_profile_sha256=_sha256(
                capture,
                "VOICE_INFERENCE_PROFILE_SHA256",
            ),
            qualification_run_id=qualification_run_id,
            benchmark_did_sha256=benchmark_did_sha256,
            deployment_max_calls=deployment_max_calls,
            handshake_timeout_seconds=_integer(
                capture,
                "VOICE_HANDSHAKE_TIMEOUT_SECONDS",
                minimum=1,
                maximum=30,
            ),
            call_idle_timeout_seconds=_integer(
                capture,
                "VOICE_CALL_IDLE_TIMEOUT_SECONDS",
                minimum=5,
                maximum=3600,
            ),
            call_cleanup_phase_timeout_seconds=_integer(
                capture,
                "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS",
                minimum=1,
                maximum=30,
            ),
            pre_drain_grace_seconds=_integer(
                capture,
                "VOICE_PRE_DRAIN_GRACE_SECONDS",
                minimum=1,
                maximum=300,
            ),
            uvicorn_grace_seconds=_integer(
                capture,
                "VOICE_UVICORN_GRACE_SECONDS",
                minimum=1,
                maximum=60,
            ),
            shutdown_grace_seconds=_integer(
                capture,
                "VOICE_SHUTDOWN_GRACE_SECONDS",
                minimum=1,
                maximum=300,
            ),
            telnyx_api_key_file=_secret_path(capture, "VOICE_TELNYX_API_KEY_FILE"),
            telnyx_webhook_public_key_file=_secret_path(
                capture,
                "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE",
            ),
            openrouter_api_key_file=_secret_path(
                capture,
                "VOICE_OPENROUTER_API_KEY_FILE",
            ),
            postgres_dsn_file=_secret_path(capture, "VOICE_POSTGRES_DSN_FILE"),
            telnyx_media_wss_url=telnyx_media_wss_url,
            otlp_http_endpoint=otlp_http_endpoint,
            bind_host=bind_host,
            bind_port=_integer(
                capture,
                "VOICE_BIND_PORT",
                minimum=1,
                maximum=65535,
            ),
        )
        from projetv0_voice.observability_bootstrap import (
            _issue_parser_observability_token,
        )

        token = _issue_parser_observability_token(settings.otlp_http_endpoint)
        object.__setattr__(settings, "_observability_token", token)
        return settings
    except _SettingsInvalid:
        pass
    raise RuntimeError("runtime_settings_invalid") from None


def _secret_basename(path: Path | PurePosixPath) -> str:
    try:
        raw_path = os.fspath(path)
    except TypeError:
        raise _SecretInvalid from None
    if type(raw_path) is not str or "\\" in raw_path:
        raise _SecretInvalid
    candidate = PurePosixPath(raw_path)
    if (
        not candidate.is_absolute()
        or candidate.parent != _TRUSTED_SECRET_DIRECTORY
        or candidate.name in {"", ".", ".."}
        or raw_path != f"{_TRUSTED_SECRET_DIRECTORY}/{candidate.name}"
    ):
        raise _SecretInvalid
    return candidate.name


def _linux_open_flags() -> tuple[int, int]:
    close_on_exec = getattr(os, "O_CLOEXEC", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    nonblocking = getattr(os, "O_NONBLOCK", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if any(
        type(value) is not int
        for value in (close_on_exec, no_follow, nonblocking, directory)
    ):
        raise _PlatformUnsupported from None
    assert type(close_on_exec) is int
    assert type(no_follow) is int
    assert type(nonblocking) is int
    assert type(directory) is int
    directory_flags = os.O_RDONLY | close_on_exec | no_follow | directory
    secret_flags = os.O_RDONLY | close_on_exec | no_follow | nonblocking
    return directory_flags, secret_flags


def _read_runtime_secret(path: Path | PurePosixPath) -> SecretStr:
    basename = _secret_basename(path)
    directory_flags, secret_flags = _linux_open_flags()
    directory_descriptor = os.open(_TRUSTED_SECRET_DIRECTORY, directory_flags)
    try:
        directory_stat = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise _SecretInvalid
        secret_descriptor = os.open(
            basename,
            secret_flags,
            dir_fd=directory_descriptor,
        )
        try:
            opened_stat = os.fstat(secret_descriptor)
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_uid != 0
                or opened_stat.st_gid != 10001
                or stat.S_IMODE(opened_stat.st_mode) != 0o440
                or opened_stat.st_size > _MAX_SECRET_BYTES
            ):
                raise _SecretInvalid
            chunks: list[bytes] = []
            remaining = _MAX_SECRET_BYTES + 1
            while remaining:
                chunk = os.read(secret_descriptor, min(remaining, 8192))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            if len(content) > _MAX_SECRET_BYTES or not content:
                raise _SecretInvalid
            if any(byte in content for byte in (0, 10, 13)):
                raise _SecretInvalid
            outer_ascii_whitespace = b" \t\v\f"
            if content[:1] in outer_ascii_whitespace or content[-1:] in outer_ascii_whitespace:
                raise _SecretInvalid
            try:
                value = content.decode("utf-8")
            except UnicodeDecodeError:
                raise _SecretInvalid from None
        finally:
            os.close(secret_descriptor)
    finally:
        os.close(directory_descriptor)

    from pydantic import SecretStr

    return SecretStr(value)


def read_runtime_secret(path: Path | PurePosixPath) -> SecretStr:
    """Read one bounded Linux secret through the trusted directory descriptor."""

    unsupported = False
    try:
        return _read_runtime_secret(path)
    except _PlatformUnsupported:
        unsupported = True
    except BaseException:
        pass
    if unsupported:
        raise RuntimeError("runtime_platform_unsupported") from None
    raise RuntimeError("runtime_secret_invalid") from None


__all__ = [
    "RuntimeEnvironmentCapture",
    "RuntimeMode",
    "RuntimeSettingsV1",
    "capture_runtime_environment",
    "parse_runtime_settings",
    "read_runtime_secret",
]
