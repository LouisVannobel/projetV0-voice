from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import threading
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
    "loguru", "onnxruntime", "opentelemetry", "requests", "pipecat", "openai",
    "httpx", "telnyx"
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
        ["loguru", "onnxruntime"],
        ["loguru", "onnxruntime", "opentelemetry", "requests"],
        [
            "loguru",
            "onnxruntime",
            "opentelemetry",
            "requests",
            "pipecat",
            "openai",
            "httpx",
        ],
        [
            "loguru",
            "onnxruntime",
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

    imported: list[str] = []
    severity: list[int] = []

    class _FakeOnnxRuntime:
        set_default_logger_severity = staticmethod(severity.append)

    real_import = importlib.import_module
    active_threads = 1

    def controlled_import(name: str) -> object:
        if name == "onnxruntime":
            imported.append(name)
            return _FakeOnnxRuntime()
        return real_import(name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(importlib, "import_module", controlled_import)
    monkeypatch.setattr(threading, "active_count", lambda: active_threads)
    try:
        dependency_logging.configure_dependency_logging(token)
        active_threads = 2
        dependency_logging.configure_dependency_logging(token)
    finally:
        monkeypatch.undo()

    assert imported == ["onnxruntime"]
    assert severity == [4]

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
    fake_onnx = type(
        "_FakeOnnxRuntime",
        (),
        {"set_default_logger_severity": lambda _self, _severity: None},
    )()
    real_import = importlib.import_module
    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_onnx if name == "onnxruntime" else real_import(name),
    )
    token = observability_bootstrap._validate_observability_mapping(  # noqa: SLF001
        {"VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT}
    )

    dependency_logging.configure_dependency_logging(token)

    assert disabled == ["pipecat"]


def _guard_token() -> observability_bootstrap.ObservabilityBootstrapToken:
    return observability_bootstrap._validate_observability_mapping(  # noqa: SLF001
        {"VOICE_OTLP_HTTP_ENDPOINT": VALID_ENDPOINT}
    )


def test_dependency_logging_suppresses_only_controlled_native_fd2_import_window(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    native_sentinel = b"onnx-native-import-sentinel"
    stdout_sentinel = b"fd1-remains-visible-sentinel"
    restored_sentinel = b"fd2-restored-sentinel"
    imported: list[str] = []
    severity: list[int] = []

    class _FakeOnnxRuntime:
        set_default_logger_severity = staticmethod(severity.append)

    real_import = importlib.import_module

    def controlled_import(name: str) -> object:
        if name == "onnxruntime":
            imported.append(name)
            os.write(1, stdout_sentinel)
            os.write(2, native_sentinel)
            return _FakeOnnxRuntime()
        return real_import(name)

    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    monkeypatch.setattr(importlib, "import_module", controlled_import)

    dependency_logging.configure_dependency_logging(_guard_token())
    os.write(2, restored_sentinel)
    captured = capfd.readouterr()

    assert imported == ["onnxruntime"]
    assert severity == [4]
    assert stdout_sentinel.decode() in captured.out
    assert native_sentinel.decode() not in captured.err
    assert restored_sentinel.decode() in captured.err


@pytest.mark.parametrize("failure_phase", ["import", "severity"])
def test_dependency_logging_restores_fd2_and_closes_descriptors_on_native_failure(
    failure_phase: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    native_sentinel = b"native-failure-secret-sentinel"
    restored_sentinel = b"fd2-restored-after-failure"
    real_import = importlib.import_module
    real_dup = os.dup
    real_open = os.open
    real_dup2 = os.dup2
    real_close = os.close
    created: list[int] = []
    closed: list[int] = []

    class _FakeOnnxRuntime:
        def set_default_logger_severity(self, severity: int) -> None:
            assert severity == 4
            if failure_phase == "severity":
                raise RuntimeError("severity-secret-sentinel")

    def controlled_import(name: str) -> object:
        if name != "onnxruntime":
            return real_import(name)
        os.write(2, native_sentinel)
        if failure_phase == "import":
            raise RuntimeError("import-secret-sentinel")
        return _FakeOnnxRuntime()

    def recording_dup(descriptor: int) -> int:
        duplicated = real_dup(descriptor)
        created.append(duplicated)
        return duplicated

    def recording_open(path: str, flags: int, mode: int = 0o777) -> int:
        opened = real_open(path, flags, mode)
        if path == os.devnull:
            created.append(opened)
        return opened

    def recording_dup2(
        source: int,
        destination: int,
        inheritable: bool = True,
    ) -> int:
        return real_dup2(source, destination, inheritable=inheritable)

    def recording_close(descriptor: int) -> None:
        if descriptor in created:
            closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    monkeypatch.setattr(importlib, "import_module", controlled_import)
    monkeypatch.setattr(os, "dup", recording_dup)
    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "dup2", recording_dup2)
    monkeypatch.setattr(os, "close", recording_close)

    with pytest.raises(
        RuntimeError,
        match="^dependency_logging_configuration_failed$",
    ) as caught:
        dependency_logging.configure_dependency_logging(_guard_token())
    os.write(2, restored_sentinel)
    captured = capfd.readouterr()

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert dependency_logging._configured is False  # noqa: SLF001
    assert native_sentinel.decode() not in captured.err
    assert "secret-sentinel" not in captured.err
    assert restored_sentinel.decode() in captured.err
    assert len(created) == 2
    assert sorted(closed) == sorted(created)
    assert len(set(closed)) == 2


def test_dependency_logging_rejects_native_boundary_after_threads_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    real_import = importlib.import_module
    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 2)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imports.append(name) or real_import(name),
    )

    with pytest.raises(RuntimeError, match="^dependency_logging_phase_invalid$") as caught:
        dependency_logging.configure_dependency_logging(_guard_token())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert imports == []
