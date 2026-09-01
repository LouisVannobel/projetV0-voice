from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from projetv0_voice import dependency_logging, observability_bootstrap

VALID_ENDPOINT = "https://collector.invalid/tenant/v1/metrics"


class _ValueSentinelMapping(Mapping[object, object]):
    def __init__(self, keys: tuple[object, ...]) -> None:
        self._keys = keys

    def __getitem__(self, key: object) -> object:
        raise AssertionError(f"value accessed for {type(key).__name__}")

    def __iter__(self) -> Iterator[object]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class _EndpointMapping(Mapping[object, object]):
    def __init__(self, endpoint: object, *extra_keys: object) -> None:
        self._endpoint = endpoint
        self._keys = (*extra_keys, "VOICE_OTLP_HTTP_ENDPOINT")
        self.reads: list[object] = []

    def __getitem__(self, key: object) -> object:
        self.reads.append(key)
        if key == "VOICE_OTLP_HTTP_ENDPOINT":
            return self._endpoint
        raise AssertionError("forbidden value accessed")

    def __iter__(self) -> Iterator[object]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


@pytest.mark.parametrize(
    "key",
    [
        object(),
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_",
        "LOGURU_AUTOINIT",
        "LOGURU_LEVEL",
        "LOGURU_FORMAT",
        "LOGURU_DIAGNOSE",
        "LOGURU_ENQUEUE",
        "LOGURU_CONTEXT",
        "TELNYX_LOG",
        "OPENAI_LOG",
    ],
)
def test_guard_rejects_forbidden_keys_without_accessing_values(key: object) -> None:
    with pytest.raises(
        RuntimeError,
        match="^observability_environment_forbidden$",
    ) as caught:
        observability_bootstrap._validate_observability_mapping(  # noqa: SLF001
            _ValueSentinelMapping((key,))
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_guard_scans_all_keys_before_reading_endpoint() -> None:
    mapping = _EndpointMapping(VALID_ENDPOINT, "VOICE_SAFE", "OTEL_HOSTILE")

    with pytest.raises(RuntimeError, match="^observability_environment_forbidden$"):
        observability_bootstrap._validate_observability_mapping(mapping)  # noqa: SLF001

    assert mapping.reads == []


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        1,
        _StringSubclass(VALID_ENDPOINT),
        "",
        "collector.invalid/v1/metrics",
        "ftp://collector.invalid/v1/metrics",
        "https:///v1/metrics",
        "https://user@collector.invalid/v1/metrics",
        "https://collector.invalid/v1/metrics?token=x",
        "https://collector.invalid/v1/metrics#secret",
        "https://collector.invalid\\v1\\metrics",
        "https://collector.invalid/v1/metrics ",
        "https://collector.invalid/v1/metric",
        "https://collector.invalid:bad/v1/metrics",
        "https://collector.invalid:70000/v1/metrics",
    ],
)
def test_guard_rejects_invalid_endpoint_with_one_constant_error(endpoint: object) -> None:
    mapping = _EndpointMapping(endpoint)

    with pytest.raises(RuntimeError, match="^observability_endpoint_invalid$") as caught:
        observability_bootstrap._validate_observability_mapping(mapping)  # noqa: SLF001

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert mapping.reads == ["VOICE_OTLP_HTTP_ENDPOINT"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:4318/v1/metrics",
        "https://collector.invalid/root/v1/metrics",
        "https://[::1]:4318/v1/metrics",
    ],
)
def test_guard_returns_opaque_frozen_token_for_valid_endpoint(endpoint: str) -> None:
    token = observability_bootstrap._validate_observability_mapping(  # noqa: SLF001
        {"VOICE_OTLP_HTTP_ENDPOINT": endpoint}
    )

    assert type(token) is observability_bootstrap.ObservabilityBootstrapToken
    assert repr(token) == "ObservabilityBootstrapToken()"
    assert str(token) == "ObservabilityBootstrapToken()"
    assert not hasattr(token, "endpoint")
    with pytest.raises((AttributeError, TypeError)):
        token.endpoint = endpoint  # type: ignore[misc]


def _run_isolated(code: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {
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
        env=env,
        timeout=30,
    )


def test_real_process_guard_cannot_be_bypassed_by_a_filtered_copy() -> None:
    result = _run_isolated(
        """
import json
import os
import sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from projetv0_voice.observability_bootstrap import (
    _validate_observability_mapping,
    validate_observability_environment,
)
copy_ok = type(_validate_observability_mapping({
    "VOICE_OTLP_HTTP_ENDPOINT": "https://collector.invalid/v1/metrics"
})).__name__
try:
    validate_observability_environment()
except RuntimeError as error:
    print(json.dumps({"copy": copy_ok, "real": str(error), "cause": error.__cause__}))
""",
        {
            "VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT,
            "OTEL_HOSTILE_KEY": "process-secret-sentinel",
        },
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "copy": "ObservabilityBootstrapToken",
        "real": "observability_environment_forbidden",
        "cause": None,
    }
    assert "process-secret-sentinel" not in result.stdout
    assert "process-secret-sentinel" not in result.stderr


def test_import_phases_keep_third_party_modules_out_until_configure() -> None:
    result = _run_isolated(
        """
import json
import os
import sys
sys.path.insert(0, os.environ["PYTHONPATH"])
third_party = (
    "loguru", "opentelemetry", "requests", "pipecat", "openai", "httpx", "telnyx"
)
import projetv0_voice.observability_bootstrap as bootstrap
phase_guard = [name for name in third_party if name in sys.modules]
import projetv0_voice.dependency_logging as dependency_logging
phase_logging_import = [name for name in third_party if name in sys.modules]
token = bootstrap.validate_observability_environment()
dependency_logging.configure_dependency_logging(token)
phase_configured = [name for name in third_party if name in sys.modules]
import projetv0_voice.metrics
phase_metrics = [name for name in third_party if name in sys.modules]
import projetv0_voice.pipeline
phase_pipeline = [name for name in third_party if name in sys.modules]
import projetv0_voice.inference.services
import projetv0_voice.inference.openrouter_tts
import projetv0_voice.telnyx.call_control
phase_providers = [name for name in third_party if name in sys.modules]
print(json.dumps([
    phase_guard,
    phase_logging_import,
    phase_configured,
    phase_metrics,
    phase_pipeline,
    phase_providers,
]))
""",
        {"VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        [],
        [],
        ["loguru"],
        ["loguru", "opentelemetry", "requests"],
        ["loguru", "opentelemetry", "requests", "pipecat", "openai", "httpx"],
        [
            "loguru",
            "opentelemetry",
            "requests",
            "pipecat",
            "openai",
            "httpx",
            "telnyx",
        ],
    ]
    assert result.stderr == ""


@pytest.mark.parametrize(
    "name",
    [
        "LOGURU_AUTOINIT",
        "LOGURU_LEVEL",
        "LOGURU_FORMAT",
        "LOGURU_DIAGNOSE",
        "LOGURU_ENQUEUE",
        "LOGURU_CONTEXT",
    ],
)
def test_hostile_loguru_values_are_never_interpreted_or_emitted(name: str) -> None:
    result = _run_isolated(
        """
import os
import sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from projetv0_voice.observability_bootstrap import validate_observability_environment
try:
    validate_observability_environment()
except RuntimeError as error:
    print(str(error))
""",
        {
            "VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT,
            name: "malformed-value-secret-sentinel",
        },
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "observability_environment_forbidden"
    assert "malformed-value-secret-sentinel" not in result.stdout
    assert "malformed-value-secret-sentinel" not in result.stderr
    assert "loguru" not in result.stderr.lower()


def test_dependency_logging_requires_exact_token() -> None:
    class _TokenSubclass(observability_bootstrap.ObservabilityBootstrapToken):
        pass

    for value in (object(), object.__new__(_TokenSubclass)):
        with pytest.raises(ValueError, match="^observability_bootstrap_token_invalid$"):
            dependency_logging.configure_dependency_logging(value)  # type: ignore[arg-type]


def test_dependency_logging_is_idempotent_and_closes_exact_logger_set() -> None:
    names = (
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
    token = observability_bootstrap._validate_observability_mapping(  # noqa: SLF001
        {"VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT}
    )

    dependency_logging.configure_dependency_logging(token)
    dependency_logging.configure_dependency_logging(token)

    for name in names:
        configured = logging.getLogger(name)
        assert len(configured.handlers) == 1
        assert type(configured.handlers[0]) is logging.NullHandler
        assert configured.propagate is False
        assert configured.disabled is True
        assert configured.level == logging.CRITICAL + 1


def test_dependency_logging_disables_only_pipecat_loguru_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled: list[str] = []
    fake_logger = type("_FakeLogger", (), {"disable": disabled.append})()
    fake_loguru = type("_FakeLoguru", (), {"logger": fake_logger})()
    monkeypatch.setitem(sys.modules, "loguru", fake_loguru)
    token = observability_bootstrap._validate_observability_mapping(  # noqa: SLF001
        {"VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT}
    )

    dependency_logging.configure_dependency_logging(token)

    assert disabled == ["pipecat"]
