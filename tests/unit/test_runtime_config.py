from __future__ import annotations

import importlib
import json
import os
import socket
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields, replace
from pathlib import Path, PurePosixPath
from types import ModuleType
from uuid import uuid4

import pytest

ALLOWED_NAMES = (
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
HEX_A = "a" * 64


def test_archive_settings_are_explicit_optional_and_consume_exact_https_origins():
    module = _runtime_config()
    values = valid_environment()
    values.update(
        {
            "VOICE_RECORDING_ARCHIVE_DIRECTORY": "/var/lib/projetv0/audio",
            "VOICE_RECORDING_DOWNLOAD_ORIGINS": "https://recordings.example.invalid",
        }
    )
    settings = module.parse_runtime_settings(
        module.capture_runtime_environment(values), geteuid=lambda: 10001, getegid=lambda: 10001
    )
    assert getattr(settings, "recording_archive_directory", None) == PurePosixPath(
        "/var/lib/projetv0/audio"
    )
    assert getattr(settings, "recording_download_origins", None) == (
        "https://recordings.example.invalid",
    )


@pytest.mark.parametrize(
    "directory,origins",
    [
        ("/var/lib/projetv0/audio", None),
        (None, "https://recordings.example.invalid"),
        ("/var/lib/projetv0/audio", "http://recordings.example.invalid"),
        ("/var/lib/projetv0/audio", "https://recordings.example.invalid/private?token=sentinel"),
    ],
)
def test_archive_settings_fail_closed_for_partial_or_noncanonical_scope(directory, origins):
    module = _runtime_config()
    values = valid_environment()
    if directory is not None:
        values["VOICE_RECORDING_ARCHIVE_DIRECTORY"] = directory
    if origins is not None:
        values["VOICE_RECORDING_DOWNLOAD_ORIGINS"] = origins
    with pytest.raises(RuntimeError, match="runtime_settings_invalid"):
        module.parse_runtime_settings(
            module.capture_runtime_environment(values), geteuid=lambda: 10001, getegid=lambda: 10001
        )


HEX_B = "b" * 64
HEX_C = "c" * 64
IMAGE = f"ghcr.io/louisvannobel/projetv0-voice@sha256:{'d' * 64}"
ENDPOINT = "https://collector.invalid/tenant/v1/metrics"
RUN_ID = "11111111-1111-4111-8111-111111111111"


def _runtime_config() -> ModuleType:
    try:
        return importlib.import_module("projetv0_voice.runtime_config")
    except ModuleNotFoundError:
        pytest.fail("runtime configuration boundary is unavailable", pytrace=False)


def valid_environment(mode: str = "strict") -> dict[str, str]:
    values = {
        "VOICE_RUNTIME_MODE": mode,
        "VOICE_DEPLOYMENT_ID": "voice-agent-a",
        "VOICE_RUNTIME_CONTRACT_PATH": "/srv/projetv0/runtime-contract.json",
        "VOICE_AGENT_BUNDLE_PATH": "/srv/projetv0/agent-bundle",
        "VOICE_KEYRING_PATH": "/srv/projetv0/keyring.json",
        "VOICE_SQLITE_PATH": "/var/lib/projetv0/voice.sqlite3",
        "VOICE_RUNTIME_CONTRACT_SHA256": HEX_A,
        "VOICE_IMAGE_DIGEST": IMAGE,
        "VOICE_AGENT_BUNDLE_SHA256": HEX_B,
        "VOICE_INFERENCE_PROFILE_SHA256": HEX_C,
        "VOICE_DEPLOYMENT_MAX_CALLS": "10",
        "VOICE_HANDSHAKE_TIMEOUT_SECONDS": "5",
        "VOICE_CALL_IDLE_TIMEOUT_SECONDS": "300",
        "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS": "10",
        "VOICE_PRE_DRAIN_GRACE_SECONDS": "15",
        "VOICE_UVICORN_GRACE_SECONDS": "20",
        "VOICE_SHUTDOWN_GRACE_SECONDS": "30",
        "VOICE_TELNYX_API_KEY_FILE": "/run/secrets/telnyx-api-key",
        "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE": "/run/secrets/telnyx-webhook-key",
        "VOICE_OPENROUTER_API_KEY_FILE": "/run/secrets/openrouter-api-key",
        "VOICE_POSTGRES_DSN_FILE": "/run/secrets/postgres-dsn",
        "VOICE_TELNYX_MEDIA_WSS_URL": "wss://voice.invalid/telnyx/media",
        "VOICE_OTLP_HTTP_ENDPOINT": ENDPOINT,
        "VOICE_BIND_HOST": "127.0.0.1",
        "VOICE_BIND_PORT": "8080",
    }
    if mode == "strict":
        values["VOICE_QUALIFIED_PROFILE_PATH"] = "/srv/projetv0/qualified.json"
    elif mode == "qualification_candidate":
        values.update(
            {
                "VOICE_QUALIFICATION_CANDIDATE_PATH": "/srv/projetv0/candidate.json",
                "VOICE_QUALIFICATION_RUN_ID": RUN_ID,
                "VOICE_BENCHMARK_DID_SHA256": HEX_A,
                "VOICE_DEPLOYMENT_MAX_CALLS": "1",
            }
        )
    elif mode == "qualification_override":
        values.update(
            {
                "VOICE_QUALIFIED_PROFILE_PATH": "/srv/projetv0/qualified.json",
                "VOICE_QUALIFICATION_OVERRIDE_PATH": "/srv/projetv0/override.json",
                "VOICE_QUALIFICATION_RUN_ID": RUN_ID,
                "VOICE_DEPLOYMENT_MAX_CALLS": "15",
            }
        )
    return values


class _CountingMapping(Mapping[object, object]):
    def __init__(self, values: Mapping[str, str], *extra_keys: object) -> None:
        self.values = dict(values)
        self.keys = (*self.values, *extra_keys)
        self.reads: dict[object, int] = {}

    def __getitem__(self, key: object) -> object:
        self.reads[key] = self.reads.get(key, 0) + 1
        if key in self.values:
            return self.values[key]  # type: ignore[index]
        if key in self.keys:
            raise AssertionError("unknown environment value was read")
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)


class _FailingIteratorMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"value read for {type(key).__name__}")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("mapping-iterator-secret-sentinel")

    def __len__(self) -> int:
        return 1


class _HostileBoundaryFailure(BaseException):
    pass


class _FailingAllowedValueMapping(Mapping[str, object]):
    def __init__(self, failure_type: type[BaseException], sentinel: str) -> None:
        self.failure_type = failure_type
        self.sentinel = sentinel

    def __getitem__(self, key: str) -> object:
        raise self.failure_type(self.sentinel)

    def __iter__(self) -> Iterator[str]:
        return iter(ALLOWED_NAMES)

    def __len__(self) -> int:
        return len(ALLOWED_NAMES)


@pytest.mark.parametrize(
    "key",
    [
        object(),
        "VOICE_TELNYX_API_KEY",
        "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY",
        "VOICE_OPENROUTER_API_KEY",
        "VOICE_POSTGRES_DSN",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "LOGURU_DIAGNOSE",
        "TELNYX_LOG",
        "TELNYX_BASE_URL",
        "OPENAI_LOG",
    ],
)
def test_hostile_environment_key_is_rejected_without_reading_its_value(key: object) -> None:
    runtime_config = _runtime_config()
    mapping = _CountingMapping(valid_environment(), key)

    with pytest.raises(RuntimeError, match="^runtime_environment_forbidden$") as caught:
        runtime_config.capture_runtime_environment(mapping)

    assert mapping.reads == {}
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("value", ["", "https://hostile.invalid/v2"])
def test_ambient_base_url_forbidden_before_capture(value: str) -> None:
    runtime_config = _runtime_config()
    mapping = _CountingMapping({**valid_environment(), "TELNYX_BASE_URL": value})
    with pytest.raises(RuntimeError, match="^runtime_environment_forbidden$"):
        runtime_config.capture_runtime_environment(mapping)
    assert mapping.reads == {}


def test_capture_reads_every_allowed_value_exactly_once_and_no_unknown_value() -> None:
    runtime_config = _runtime_config()
    mapping = _CountingMapping(valid_environment(), "UNRELATED_PROCESS_VALUE")

    capture = runtime_config.capture_runtime_environment(mapping)

    assert mapping.reads == {name: 1 for name in ALLOWED_NAMES}
    assert repr(capture) == "RuntimeEnvironmentCapture()"
    with pytest.raises((AttributeError, TypeError)):
        capture.values = ()


def test_hostile_mapping_iterator_failure_is_collapsed_without_text_or_chain() -> None:
    runtime_config = _runtime_config()

    with pytest.raises(RuntimeError, match="^runtime_environment_scan_failed$") as caught:
        runtime_config.capture_runtime_environment(_FailingIteratorMapping())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "mapping-iterator-secret-sentinel" not in str(caught.value)
    assert "mapping-iterator-secret-sentinel" not in repr(caught.value)


@pytest.mark.parametrize("failure_type", [RuntimeError, _HostileBoundaryFailure])
def test_hostile_allowed_value_failure_is_collapsed_without_disclosure(
    failure_type: type[BaseException],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_config = _runtime_config()
    sentinel = "allowed-value-secret-sentinel"
    mapping = _FailingAllowedValueMapping(failure_type, sentinel)

    with pytest.raises(RuntimeError, match="^runtime_environment_capture_failed$") as caught:
        runtime_config.capture_runtime_environment(mapping)

    captured = capsys.readouterr()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert sentinel not in caplog.text
    assert sentinel not in captured.out
    assert sentinel not in captured.err


@pytest.mark.parametrize(
    ("mode", "expected_profile_field"),
    [
        ("strict", "qualified_profile_path"),
        ("qualification_candidate", "qualification_candidate_path"),
        ("qualification_override", "qualification_override_path"),
    ],
)
def test_parse_once_builds_frozen_typed_settings_for_each_mode(
    mode: str,
    expected_profile_field: str,
) -> None:
    runtime_config = _runtime_config()
    mapping = _CountingMapping(valid_environment(mode))
    capture = runtime_config.capture_runtime_environment(mapping)

    settings = runtime_config.parse_runtime_settings(
        capture,
        geteuid=lambda: 10001,
        getegid=lambda: 10001,
    )

    assert settings.runtime_mode == mode
    assert settings.runtime_contract_path == PurePosixPath("/srv/projetv0/runtime-contract.json")
    assert getattr(settings, expected_profile_field) is not None
    assert settings.bind_port == 8080
    assert settings.otlp_http_endpoint == ENDPOINT
    assert repr(settings) == "RuntimeSettingsV1()"
    assert mapping.reads == {name: 1 for name in ALLOWED_NAMES}
    with pytest.raises((AttributeError, TypeError)):
        settings.bind_port = 9090


def test_parse_uses_injected_identity_on_windows_safe_path() -> None:
    runtime_config = _runtime_config()
    capture = runtime_config.capture_runtime_environment(valid_environment())
    calls: list[str] = []

    settings = runtime_config.parse_runtime_settings(
        capture,
        geteuid=lambda: calls.append("uid") or 10001,
        getegid=lambda: calls.append("gid") or 10001,
    )

    assert calls == ["uid", "gid"]
    assert settings.deployment_id == "voice-agent-a"


@pytest.mark.parametrize("host_cap", [1, 10, (1 << 31) - 1])
def test_candidate_mode_accepts_every_positive_runtime_host_cap(host_cap: int) -> None:
    runtime_config = _runtime_config()
    environment = valid_environment("qualification_candidate")
    environment["VOICE_DEPLOYMENT_MAX_CALLS"] = str(host_cap)
    capture = runtime_config.capture_runtime_environment(environment)

    settings = runtime_config.parse_runtime_settings(
        capture,
        geteuid=lambda: 10001,
        getegid=lambda: 10001,
    )

    assert settings.runtime_mode == "qualification_candidate"
    assert settings.deployment_max_calls == host_cap


def test_parse_reports_one_unsupported_platform_error_when_posix_identity_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _runtime_config()
    capture = runtime_config.capture_runtime_environment(valid_environment())
    monkeypatch.delattr(runtime_config.os, "geteuid", raising=False)
    monkeypatch.delattr(runtime_config.os, "getegid", raising=False)

    with pytest.raises(RuntimeError, match="^runtime_platform_unsupported$") as caught:
        runtime_config.parse_runtime_settings(capture)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(("uid", "gid"), [(0, 10001), (10001, 0), (10000, 10001)])
def test_parse_rejects_every_non_runtime_effective_identity(uid: int, gid: int) -> None:
    runtime_config = _runtime_config()
    capture = runtime_config.capture_runtime_environment(valid_environment())

    with pytest.raises(RuntimeError, match="^runtime_identity_invalid$") as caught:
        runtime_config.parse_runtime_settings(
            capture,
            geteuid=lambda: uid,
            getegid=lambda: gid,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("failure_type", [RuntimeError, _HostileBoundaryFailure])
@pytest.mark.parametrize("failing_identity", ["geteuid", "getegid"])
def test_hostile_identity_failure_is_collapsed_without_disclosure(
    failure_type: type[BaseException],
    failing_identity: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_config = _runtime_config()
    capture = runtime_config.capture_runtime_environment(valid_environment())
    sentinel = f"{failing_identity}-secret-sentinel"

    def hostile_identity() -> int:
        raise failure_type(sentinel)

    identity_functions: dict[str, Callable[[], int]] = {
        "geteuid": hostile_identity if failing_identity == "geteuid" else lambda: 10001,
        "getegid": hostile_identity if failing_identity == "getegid" else lambda: 10001,
    }

    with pytest.raises(RuntimeError, match="^runtime_identity_invalid$") as caught:
        runtime_config.parse_runtime_settings(capture, **identity_functions)

    captured = capsys.readouterr()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert sentinel not in caplog.text
    assert sentinel not in captured.out
    assert sentinel not in captured.err


@pytest.mark.parametrize(
    ("mode", "name", "value"),
    [
        ("strict", "VOICE_QUALIFICATION_RUN_ID", RUN_ID),
        ("strict", "VOICE_QUALIFICATION_CANDIDATE_PATH", "/srv/candidate.json"),
        ("qualification_candidate", "VOICE_QUALIFIED_PROFILE_PATH", "/srv/strict.json"),
        ("qualification_candidate", "VOICE_QUALIFICATION_OVERRIDE_PATH", "/srv/override.json"),
        ("qualification_override", "VOICE_BENCHMARK_DID_SHA256", HEX_A),
        ("qualification_override", "VOICE_QUALIFICATION_CANDIDATE_PATH", "/srv/candidate.json"),
    ],
)
def test_mode_inapplicable_values_are_rejected_not_ignored(
    mode: str,
    name: str,
    value: str,
) -> None:
    runtime_config = _runtime_config()
    environment = valid_environment(mode)
    environment[name] = value
    capture = runtime_config.capture_runtime_environment(environment)

    with pytest.raises(RuntimeError, match="^runtime_settings_invalid$") as caught:
        runtime_config.parse_runtime_settings(
            capture,
            geteuid=lambda: 10001,
            getegid=lambda: 10001,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("VOICE_HANDSHAKE_TIMEOUT_SECONDS", "0"),
        ("VOICE_HANDSHAKE_TIMEOUT_SECONDS", "31"),
        ("VOICE_CALL_IDLE_TIMEOUT_SECONDS", "4"),
        ("VOICE_CALL_IDLE_TIMEOUT_SECONDS", "3601"),
        ("VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS", "01"),
        ("VOICE_PRE_DRAIN_GRACE_SECONDS", "301"),
        ("VOICE_UVICORN_GRACE_SECONDS", "+1"),
        ("VOICE_SHUTDOWN_GRACE_SECONDS", " 30"),
        ("VOICE_BIND_PORT", "65536"),
        ("VOICE_BIND_PORT", "9" * 5000),
        ("VOICE_RUNTIME_CONTRACT_SHA256", "A" * 64),
        ("VOICE_TELNYX_MEDIA_WSS_URL", "https://voice.invalid/telnyx/media"),
        ("VOICE_OTLP_HTTP_ENDPOINT", "https://user@collector.invalid/v1/metrics"),
    ],
)
def test_exact_scalar_contract_rejects_coercion_and_out_of_range_values(
    name: str,
    invalid: str,
) -> None:
    runtime_config = _runtime_config()
    environment = valid_environment()
    environment[name] = invalid
    capture = runtime_config.capture_runtime_environment(environment)

    with pytest.raises(RuntimeError, match="^runtime_settings_invalid$") as caught:
        runtime_config.parse_runtime_settings(
            capture,
            geteuid=lambda: 10001,
            getegid=lambda: 10001,
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_full_parse_attaches_one_opaque_production_token_to_settings() -> None:
    runtime_config = _runtime_config()
    bootstrap = importlib.import_module("projetv0_voice.observability_bootstrap")
    capture = runtime_config.capture_runtime_environment(valid_environment())
    settings = runtime_config.parse_runtime_settings(
        capture,
        geteuid=lambda: 10001,
        getegid=lambda: 10001,
    )

    token = settings.observability_token()

    assert repr(token) == "ObservabilityBootstrapToken()"
    assert not hasattr(bootstrap, "issue_observability_token")


def test_direct_and_replaced_settings_have_no_production_token_or_provenance_field() -> None:
    runtime_config = _runtime_config()
    capture = runtime_config.capture_runtime_environment(valid_environment())
    settings = runtime_config.parse_runtime_settings(
        capture,
        geteuid=lambda: 10001,
        getegid=lambda: 10001,
    )
    token = settings.observability_token()
    direct_values = {
        model_field.name: getattr(settings, model_field.name)
        for model_field in fields(settings)
        if model_field.init
    }
    direct = runtime_config.RuntimeSettingsV1(**direct_values)
    replaced = replace(settings)

    for value in (direct, replaced):
        with pytest.raises(RuntimeError, match="^observability_token_unavailable$") as caught:
            value.observability_token()
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    with pytest.raises(TypeError):
        replace(settings, _capture_identity=object())
    with pytest.raises(TypeError, match="init=False"):
        replace(settings, _observability_token=token)


def test_endpoint_only_capture_cannot_create_settings_or_access_an_issuer() -> None:
    runtime_config = _runtime_config()
    bootstrap = importlib.import_module("projetv0_voice.observability_bootstrap")
    capture = runtime_config.capture_runtime_environment({"VOICE_OTLP_HTTP_ENDPOINT": ENDPOINT})

    with pytest.raises(RuntimeError, match="^runtime_settings_invalid$"):
        runtime_config.parse_runtime_settings(
            capture,
            geteuid=lambda: 10001,
            getegid=lambda: 10001,
        )
    assert not hasattr(bootstrap, "issue_observability_token")


def _run_isolated(code: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = {
        "APPDATA": os.environ.get("APPDATA", ""),
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        "PYTHONIOENCODING": "utf-8",
        **extra_env,
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )


def test_rejected_capture_imports_no_application_or_third_party_module() -> None:
    result = _run_isolated(
        """
import json
import os
import sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from projetv0_voice.runtime_config import capture_runtime_environment
class Hostile(dict):
    def __getitem__(self, key):
        raise AssertionError("hostile-value-sentinel")
try:
    capture_runtime_environment(Hostile({"OTEL_HOSTILE": object()}))
except RuntimeError as error:
    print(json.dumps({
        "error": str(error),
        "third_party": [
            name for name in (
                "fastapi", "uvicorn", "pydantic", "pipecat", "telnyx", "openai",
                "opentelemetry", "requests", "loguru", "psycopg"
            ) if name in sys.modules
        ],
    }))
""",
        {},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "error": "runtime_environment_forbidden",
        "third_party": [],
    }
    assert "hostile-value-sentinel" not in result.stdout
    assert "hostile-value-sentinel" not in result.stderr


@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath("/run/secrets"),
        PurePosixPath("/run/secrets/../secret"),
        PurePosixPath("/run/secrets/nested/secret"),
        PurePosixPath("/tmp/secret"),
        PurePosixPath("relative-secret"),
    ],
)
def test_runtime_secret_rejects_every_non_direct_trusted_child(path: PurePosixPath) -> None:
    runtime_config = _runtime_config()

    with pytest.raises(RuntimeError, match="^runtime_secret_invalid$") as caught:
        runtime_config.read_runtime_secret(path)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def _require_privileged_linux() -> None:
    if os.name == "nt":
        pytest.skip("Linux kernel descriptor contract")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        if os.environ.get("PROJETV0_PRIVILEGED_FILES_GATE") == "1":
            pytest.fail("privileged Linux FILES gate did not run as root", pytrace=False)
        pytest.skip("privileged Linux fixture ownership is required")


def _secret_fixture_path() -> Path:
    _require_privileged_linux()
    trusted = Path("/run/secrets")
    trusted.mkdir(mode=0o755, parents=True, exist_ok=True)
    run_stat = trusted.parent.stat()
    trusted_stat = trusted.stat()
    run_permission = (
        (run_stat.st_mode >> 6) & 0o7
        if run_stat.st_uid == 10001
        else (run_stat.st_mode >> 3) & 0o7
        if run_stat.st_gid == 10001
        else run_stat.st_mode & 0o7
    )
    trusted_permission = (
        (trusted_stat.st_mode >> 6) & 0o7
        if trusted_stat.st_uid == 10001
        else (trusted_stat.st_mode >> 3) & 0o7
        if trusted_stat.st_gid == 10001
        else trusted_stat.st_mode & 0o7
    )
    assert run_permission & 0o1
    assert trusted_permission & 0o5 == 0o5
    return trusted / f"projetv0-test-{uuid4().hex}"


def _remove_secret_fixture(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def _write_secret(
    path: Path,
    content: bytes,
    *,
    uid: int = 0,
    gid: int = 10001,
    mode: int = 0o440,
) -> None:
    path.write_bytes(content)
    os.chown(path, uid, gid)
    os.chmod(path, mode)


def _read_secret_subprocess(path: Path) -> dict[str, object]:
    code = """
import json
import os
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
sys.path.insert(0, os.environ["PYTHONPATH"])
from projetv0_voice.runtime_config import (
    capture_runtime_environment,
    parse_runtime_settings,
    read_runtime_secret,
)
import projetv0_voice.observability_bootstrap as preloaded_observability_bootstrap
import pydantic.types as preloaded_pydantic_types
from pydantic import SecretStr as PreloadedSecretStr
preloaded_modules = [
    preloaded_observability_bootstrap.__name__,
    preloaded_pydantic_types.__name__,
]
preloaded_secret_type = {
    "name": PreloadedSecretStr.__name__,
    "module": PreloadedSecretStr.__module__,
    "materialized": PreloadedSecretStr is preloaded_pydantic_types.SecretStr,
}
os.setgroups([])
os.setgid(10001)
os.setuid(10001)
identity = {
    "uid": os.getuid(),
    "gid": os.getgid(),
    "euid": os.geteuid(),
    "egid": os.getegid(),
    "groups": os.getgroups(),
}
expected_identity = {
    "uid": 10001,
    "gid": 10001,
    "euid": 10001,
    "egid": 10001,
    "groups": [],
}
blocked_post_drop_imports = []
class PostDropImportGuard(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "projetv0_voice.observability_bootstrap",
            "pydantic.types",
        }:
            blocked_post_drop_imports.append(fullname)
            raise ModuleNotFoundError("post_drop_filesystem_import_forbidden")
        return None
sys.meta_path.insert(0, PostDropImportGuard())
post_drop_calls = []
def call_after_identity_drop(name, operation, argument):
    current_identity = {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "euid": os.geteuid(),
        "egid": os.getegid(),
        "groups": os.getgroups(),
    }
    post_drop_calls.append({"name": name, "identity": current_identity})
    if current_identity != expected_identity:
        raise RuntimeError("runtime_secret_test_identity_invalid")
    return operation(argument)
try:
    capture = capture_runtime_environment(json.loads(sys.argv[2]))
    settings = call_after_identity_drop(
        "parse_runtime_settings",
        parse_runtime_settings,
        capture,
    )
    value = call_after_identity_drop(
        "read_runtime_secret",
        read_runtime_secret,
        Path(sys.argv[1]),
    )
except BaseException as error:
    print(json.dumps({
        "kind": "error",
        "error": str(error),
        "cause": error.__cause__ is None,
        "context": error.__context__ is None,
        "identity": identity,
        "settings": locals().get("settings").deployment_id if "settings" in locals() else None,
        "preloaded_modules": preloaded_modules,
        "preloaded_secret_type": preloaded_secret_type,
        "blocked_post_drop_imports": blocked_post_drop_imports,
        "post_drop_calls": post_drop_calls,
    }))
else:
    print(json.dumps({
        "kind": "value",
        "value": value.get_secret_value(),
        "identity": identity,
        "settings": settings.deployment_id,
        "preloaded_modules": preloaded_modules,
        "preloaded_secret_type": preloaded_secret_type,
        "blocked_post_drop_imports": blocked_post_drop_imports,
        "post_drop_calls": post_drop_calls,
    }))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    runtime_environment = valid_environment()
    for name in (
        "VOICE_TELNYX_API_KEY_FILE",
        "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE",
        "VOICE_OPENROUTER_API_KEY_FILE",
        "VOICE_POSTGRES_DSN_FILE",
    ):
        runtime_environment[name] = path.as_posix()
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-I",
                "-c",
                code,
                str(path),
                json.dumps(runtime_environment),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        return {"kind": "timeout"}
    if result.returncode != 0:
        return {"kind": "process_error", "stderr": result.stderr}
    payload = json.loads(result.stdout)
    assert payload.pop("identity") == {
        "uid": 10001,
        "gid": 10001,
        "euid": 10001,
        "egid": 10001,
        "groups": [],
    }
    assert payload.pop("preloaded_modules") == [
        "projetv0_voice.observability_bootstrap",
        "pydantic.types",
    ]
    assert payload.pop("preloaded_secret_type") == {
        "name": "SecretStr",
        "module": "pydantic.types",
        "materialized": True,
    }
    assert payload.pop("blocked_post_drop_imports") == []
    assert payload.pop("post_drop_calls") == [
        {
            "name": "parse_runtime_settings",
            "identity": {
                "uid": 10001,
                "gid": 10001,
                "euid": 10001,
                "egid": 10001,
                "groups": [],
            },
        },
        {
            "name": "read_runtime_secret",
            "identity": {
                "uid": 10001,
                "gid": 10001,
                "euid": 10001,
                "egid": 10001,
                "groups": [],
            },
        },
    ]
    assert payload.pop("settings") == "voice-agent-a"
    return payload


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
def test_linux_kernel_privileged_gate_preconditions_are_enforced() -> None:
    path = _secret_fixture_path()

    assert os.geteuid() == 0
    assert path.parent == Path("/run/secrets")


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
def test_linux_kernel_runtime_secret_dropped_identity_reads_compliant_file() -> None:
    path = _secret_fixture_path()
    _write_secret(path, b"secret-value")
    try:
        result = _read_secret_subprocess(path)
    finally:
        _remove_secret_fixture(path)

    assert result == {"kind": "value", "value": "secret-value"}


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
def test_linux_kernel_runtime_secret_regular_file_uses_trusted_dir_fd_and_exact_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_privileged_linux()
    runtime_config = _runtime_config()
    path = _secret_fixture_path()
    _write_secret(path, b"secret-value")
    real_open = os.open
    calls: list[tuple[object, int, int | None]] = []

    def recording_open(
        open_path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        calls.append((open_path, flags, dir_fd))
        return real_open(open_path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(runtime_config.os, "open", recording_open)
    try:
        secret = runtime_config.read_runtime_secret(path)
    finally:
        _remove_secret_fixture(path)

    assert secret.get_secret_value() == "secret-value"
    assert len(calls) == 2
    assert os.fspath(calls[0][0]) == "/run/secrets"
    directory_fd = calls[1][2]
    assert type(directory_fd) is int
    assert calls[1][0] == path.name
    assert calls[1][1] == os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
@pytest.mark.parametrize("kind", ["fifo", "socket", "directory", "symlink"])
def test_linux_kernel_runtime_secret_nonregular_and_symlink_fail_before_deadline(
    kind: str,
) -> None:
    _require_privileged_linux()
    path = _secret_fixture_path()
    socket_owner: socket.socket | None = None
    target: Path | None = None
    try:
        if kind == "fifo":
            os.mkfifo(path, 0o440)
            os.chown(path, 0, 10001)
        elif kind == "socket":
            socket_owner = socket.socket(socket.AF_UNIX)
            socket_owner.bind(str(path))
        elif kind == "directory":
            path.mkdir(mode=0o440)
            os.chown(path, 0, 10001)
        else:
            target = path.with_name(path.name + "-target")
            _write_secret(target, b"target-secret")
            path.symlink_to(target)

        result = _read_secret_subprocess(path)
    finally:
        if socket_owner is not None:
            socket_owner.close()
        _remove_secret_fixture(path)
        if target is not None:
            _remove_secret_fixture(target)

    assert result == {
        "kind": "error",
        "error": "runtime_secret_invalid",
        "cause": True,
        "context": True,
    }


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
@pytest.mark.parametrize(
    ("uid", "gid", "mode"),
    [
        (10001, 10001, 0o440),
        (0, 0, 0o440),
        (0, 10001, 0o400),
        (0, 10001, 0o640),
        (0, 10001, 0o444),
        (0, 10001, 0o1440),
    ],
)
def test_linux_kernel_runtime_secret_rejects_wrong_owner_group_or_mode(
    uid: int,
    gid: int,
    mode: int,
) -> None:
    path = _secret_fixture_path()
    _write_secret(path, b"secret-value", uid=uid, gid=gid, mode=mode)
    try:
        result = _read_secret_subprocess(path)
    finally:
        _remove_secret_fixture(path)

    assert result == {
        "kind": "error",
        "error": "runtime_secret_invalid",
        "cause": True,
        "context": True,
    }


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
@pytest.mark.parametrize(("size", "accepted"), [(16_384, True), (16_385, False)])
def test_linux_kernel_runtime_secret_enforces_read_bound(size: int, accepted: bool) -> None:
    path = _secret_fixture_path()
    _write_secret(path, b"x" * size)
    try:
        result = _read_secret_subprocess(path)
    finally:
        _remove_secret_fixture(path)

    if accepted:
        assert result == {"kind": "value", "value": "x" * size}
    else:
        assert result == {
            "kind": "error",
            "error": "runtime_secret_invalid",
            "cause": True,
            "context": True,
        }


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
def test_linux_kernel_runtime_secret_loops_over_partial_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _runtime_config()
    path = _secret_fixture_path()
    _write_secret(path, b"partial-secret")
    real_read = os.read
    read_sizes: list[int] = []

    def partial_read(descriptor: int, count: int) -> bytes:
        read_sizes.append(count)
        return real_read(descriptor, min(count, 3))

    monkeypatch.setattr(runtime_config.os, "read", partial_read)
    try:
        secret = runtime_config.read_runtime_secret(path)
    finally:
        _remove_secret_fixture(path)

    assert secret.get_secret_value() == "partial-secret"
    assert len(read_sizes) > 2


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
def test_linux_kernel_runtime_secret_detects_growth_after_fstat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _runtime_config()
    path = _secret_fixture_path()
    _write_secret(path, b"x" * 16_384)
    real_fstat = os.fstat
    grew = False

    def growing_fstat(descriptor: int) -> os.stat_result:
        nonlocal grew
        result = real_fstat(descriptor)
        if stat.S_ISREG(result.st_mode) and not grew:
            grew = True
            with path.open("ab") as stream:
                stream.write(b"y")
        return result

    monkeypatch.setattr(runtime_config.os, "fstat", growing_fstat)
    try:
        with pytest.raises(RuntimeError, match="^runtime_secret_invalid$"):
            runtime_config.read_runtime_secret(path)
    finally:
        _remove_secret_fixture(path)

    assert grew is True


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
def test_linux_kernel_runtime_secret_closes_each_descriptor_once_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _runtime_config()
    path = _secret_fixture_path()
    _write_secret(path, b"x" * 16_385)
    real_close = os.close
    closed: list[int] = []

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(runtime_config.os, "close", recording_close)
    try:
        with pytest.raises(RuntimeError, match="^runtime_secret_invalid$"):
            runtime_config.read_runtime_secret(path)
    finally:
        _remove_secret_fixture(path)

    assert len(closed) == 2
    assert len(set(closed)) == 2


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
@pytest.mark.parametrize("content", [b" secret", b"secret ", b"a\x00b", b"a\rb", b"a\nb"])
def test_linux_kernel_runtime_secret_rejects_content_without_leaking_it(
    content: bytes,
) -> None:
    runtime_config = _runtime_config()
    path = _secret_fixture_path()
    _write_secret(path, content)
    try:
        with pytest.raises(RuntimeError, match="^runtime_secret_invalid$") as caught:
            runtime_config.read_runtime_secret(path)
    finally:
        _remove_secret_fixture(path)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert content.decode("utf-8", errors="ignore") not in str(caught.value)
