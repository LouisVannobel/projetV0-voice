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
guarded_imports = set()

def observe_import(event, args):
    if event != "import":
        return
    name = args[0].partition(".")[0]
    if name in third_party[2:]:
        configured = sys.modules.get("projetv0_voice.dependency_logging")
        assert configured is not None and configured._configured, name
        guarded_imports.add(name)

sys.addaudithook(observe_import)
import projetv0_voice.observability_bootstrap as bootstrap
phase_guard = [name for name in third_party if name in sys.modules]
import projetv0_voice.dependency_logging as dependency_logging
phase_logging_import = [name for name in third_party if name in sys.modules]
token = bootstrap.validate_observability_environment()
phase_validated = [name for name in third_party if name in sys.modules]
dependency_logging.configure_dependency_logging(token)
phase_configured = [name for name in third_party if name in sys.modules]
import projetv0_voice.metrics
phase_metrics = [name for name in third_party if name in sys.modules]
# Pipecat 1.12 keeps LLMContext's OpenAI aliases under TYPE_CHECKING.
# SDKs load at the provider phase, after dependency logging is configured.
import projetv0_voice.pipeline
phase_pipeline = [name for name in third_party if name in sys.modules]
import projetv0_voice.inference.services
import projetv0_voice.inference.openrouter_tts
import projetv0_voice.telnyx.call_control
phase_providers = [name for name in third_party if name in sys.modules]
assert guarded_imports == set(third_party[2:])
print(json.dumps([
    phase_guard,
    phase_logging_import,
    phase_validated,
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
        [],
        ["loguru", "onnxruntime"],
        ["loguru", "onnxruntime", "opentelemetry", "requests"],
        [
            "loguru",
            "onnxruntime",
            "opentelemetry",
            "requests",
            "pipecat",
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
        monkeypatch.setattr(dependency_logging, "os", object())
        monkeypatch.setattr(dependency_logging, "importlib", object())
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
    original_inheritable = os.get_inheritable(2)

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
    assert os.get_inheritable(2) is original_inheritable


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


@pytest.mark.parametrize("phase", ["active_threads", "non_main_thread"])
def test_dependency_logging_rejects_invalid_phase_before_any_fd_operation(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []

    def forbidden(name: str) -> object:
        operations.append(name)
        raise AssertionError(name)

    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(
        threading,
        "active_count",
        lambda: 2 if phase == "active_threads" else 1,
    )
    if phase == "non_main_thread":
        monkeypatch.setattr(threading, "current_thread", object)
    class _ForbiddenOS:
        get_inheritable = staticmethod(lambda _fd: forbidden("get_inheritable"))
        dup = staticmethod(lambda _fd: forbidden("dup"))
        open = staticmethod(lambda *_args: forbidden("open"))
        dup2 = staticmethod(lambda *_args, **_kwargs: forbidden("dup2"))
        close = staticmethod(lambda _fd: forbidden("close"))

    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(dependency_logging, "os", _ForbiddenOS())

    with pytest.raises(RuntimeError, match="^dependency_logging_phase_invalid$") as caught:
        dependency_logging.configure_dependency_logging(_guard_token())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert operations == []


@pytest.mark.parametrize(
    "failure_phase",
    ["initial_flush", "get_inheritable", "cleanup_flush"],
)
def test_dependency_logging_preflight_or_flush_failure_is_soft_only_with_proved_cleanup(
    failure_phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_dup = os.dup
    real_open = os.open
    real_dup2 = os.dup2
    real_close = os.close
    real_get_inheritable = os.get_inheritable
    calls = {"dup": 0, "open": 0, "dup2": 0, "close": 0, "flush": 0}

    class _FakeOnnxRuntime:
        @staticmethod
        def set_default_logger_severity(severity: int) -> None:
            assert severity == 4

    def controlled_flush() -> bool:
        calls["flush"] += 1
        return not (
            failure_phase == "initial_flush"
            and calls["flush"] == 1
            or failure_phase == "cleanup_flush"
            and calls["flush"] == 2
        )

    def recording_dup(fd: int) -> int:
        calls["dup"] += 1
        return real_dup(fd)

    def recording_open(path: str, flags: int, mode: int = 0o777) -> int:
        calls["open"] += 1
        return real_open(path, flags, mode)

    def recording_dup2(
        source: int,
        destination: int,
        inheritable: bool = True,
    ) -> int:
        calls["dup2"] += 1
        return real_dup2(source, destination, inheritable=inheritable)

    def recording_close(fd: int) -> None:
        calls["close"] += 1
        real_close(fd)

    def controlled_get_inheritable(fd: int) -> bool:
        if failure_phase == "get_inheritable":
            raise KeyboardInterrupt("get-inheritable-secret-sentinel")
        return real_get_inheritable(fd)

    class _ControlledOS:
        O_WRONLY = os.O_WRONLY
        devnull = os.devnull
        get_inheritable = staticmethod(controlled_get_inheritable)
        dup = staticmethod(recording_dup)
        open = staticmethod(recording_open)
        dup2 = staticmethod(recording_dup2)
        close = staticmethod(recording_close)

    monkeypatch.setattr(dependency_logging, "_configured", False, raising=False)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    monkeypatch.setattr(dependency_logging, "_flush_stderr", controlled_flush)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: _FakeOnnxRuntime()
        if name == "onnxruntime"
        else importlib.import_module(name),
    )
    monkeypatch.setattr(dependency_logging, "os", _ControlledOS())

    with pytest.raises(
        RuntimeError,
        match="^dependency_logging_configuration_failed$",
    ) as caught:
        dependency_logging.configure_dependency_logging(_guard_token())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    if failure_phase in ("initial_flush", "get_inheritable"):
        assert calls == {"dup": 0, "open": 0, "dup2": 0, "close": 0, "flush": 1}
    else:
        assert calls == {"dup": 1, "open": 1, "dup2": 2, "close": 2, "flush": 2}


_NATIVE_FATAL_FAULT_PROBE = r"""
import importlib
import json
import os
import sys
import threading

sys.path.insert(0, os.environ["PYTHONPATH"])
from projetv0_voice import dependency_logging, observability_bootstrap

fault = os.environ["PROJETV0_NATIVE_FAULT"]
real_hard_exit = os.environ["PROJETV0_REAL_HARD_EXIT"] == "1"
returning_hard_exit = os.environ.get("PROJETV0_RETURNING_HARD_EXIT") == "1"
token = observability_bootstrap._validate_observability_mapping({
    "VOICE_OTLP_HTTP_ENDPOINT": "https://collector.invalid/tenant/v1/metrics"
})
real_dup = os.dup
real_open = os.open
real_dup2 = os.dup2
real_close = os.close
real_flush_stderr = dependency_logging._flush_stderr
calls = {
    "dup": 0,
    "open": 0,
    "dup2": [],
    "close": [],
    "flush": 0,
    "exit": [],
}
original_inheritable = os.get_inheritable(2)

class FakeOnnxRuntime:
    @staticmethod
    def set_default_logger_severity(severity):
        if severity != 4:
            raise AssertionError(severity)

def controlled_import(name):
    if name != "onnxruntime":
        raise AssertionError(name)
    return FakeOnnxRuntime()

def recording_flush_stderr():
    calls["flush"] += 1
    return real_flush_stderr()

def faulting_dup(fd):
    calls["dup"] += 1
    result = real_dup(fd)
    if fault == "dup":
        raise KeyboardInterrupt("dup-effect-real")
    return result

def faulting_open(path, flags, mode=0o777):
    calls["open"] += 1
    result = real_open(path, flags, mode)
    if fault == "open":
        raise KeyboardInterrupt("open-effect-real")
    return result

def faulting_dup2(source, destination, inheritable=True):
    calls["dup2"].append([source, destination, inheritable])
    call_number = len(calls["dup2"])
    if fault == "restore_fail" and call_number == 2:
        raise OSError("restore-failed-before-effect")
    result = real_dup2(source, destination, inheritable=inheritable)
    if fault == "redirect" and call_number == 1:
        raise KeyboardInterrupt("redirect-effect-real")
    if fault == "restore" and call_number == 2:
        raise KeyboardInterrupt("restore-effect-real")
    return result

def faulting_close(fd):
    calls["close"].append(fd)
    call_number = len(calls["close"])
    if fault == "close_devnull_fail" and call_number == 1:
        raise OSError("close-devnull-failed-before-effect")
    if fault == "close_saved_fail" and call_number == 2:
        raise OSError("close-saved-failed-before-effect")
    real_close(fd)
    if fault == "close_devnull" and call_number == 1:
        raise KeyboardInterrupt("close-devnull-effect-real")
    if fault == "close_saved" and call_number == 2:
        raise KeyboardInterrupt("close-saved-effect-real")

class TerminalExit(BaseException):
    pass

def nonreturning_hard_exit(code):
    calls["exit"].append(code)
    raise TerminalExit()

def invalid_returning_hard_exit(code):
    calls["exit"].append(code)

dependency_logging._configured = False
dependency_logging._flush_stderr = recording_flush_stderr
dependency_logging.importlib.import_module = controlled_import
dependency_logging.threading.active_count = lambda: 1
dependency_logging.threading.current_thread = threading.main_thread
dependency_logging.os.dup = faulting_dup
dependency_logging.os.open = faulting_open
dependency_logging.os.dup2 = faulting_dup2
dependency_logging.os.close = faulting_close
if returning_hard_exit:
    dependency_logging._hard_exit = invalid_returning_hard_exit
elif not real_hard_exit:
    dependency_logging._hard_exit = nonreturning_hard_exit

terminal = False
public_error = None
try:
    dependency_logging.configure_dependency_logging(token)
except TerminalExit:
    terminal = True
except RuntimeError as error:
    public_error = str(error)

if real_hard_exit or returning_hard_exit:
    os.write(1, b"POST-CALL-MARKER")
else:
    print(json.dumps({
        "terminal": terminal,
        "public_error": public_error,
        "calls": calls,
        "configured": dependency_logging._configured,
        "original_inheritable": original_inheritable,
        "final_inheritable": os.get_inheritable(2),
    }))
"""


@pytest.mark.parametrize(
    ("fault", "expected_dup2", "expected_close_count", "expected_flush_count"),
    [
        ("dup", 0, 0, 1),
        ("open", 0, 1, 1),
        ("redirect", 2, 2, 1),
        ("restore_fail", 2, 2, 2),
        ("restore", 2, 2, 2),
        ("close_devnull_fail", 2, 2, 2),
        ("close_devnull", 2, 2, 2),
        ("close_saved_fail", 2, 2, 2),
        ("close_saved", 2, 2, 2),
    ],
)
def test_dependency_logging_native_uncertain_effect_uses_nonreturning_fatal_seam(
    fault: str,
    expected_dup2: int,
    expected_close_count: int,
    expected_flush_count: int,
) -> None:
    result = _run_isolated(
        _NATIVE_FATAL_FAULT_PROBE,
        {"PROJETV0_NATIVE_FAULT": fault, "PROJETV0_REAL_HARD_EXIT": "0"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["terminal"] is True
    assert payload["public_error"] is None
    assert payload["calls"]["exit"] == [71]
    assert payload["calls"]["dup"] == 1
    assert payload["calls"]["open"] == (0 if fault == "dup" else 1)
    assert len(payload["calls"]["dup2"]) == expected_dup2
    assert len(payload["calls"]["close"]) == expected_close_count
    assert payload["calls"]["flush"] == expected_flush_count
    assert len(payload["calls"]["close"]) == len(set(payload["calls"]["close"]))
    assert all(
        call[2] is payload["original_inheritable"]
        for call in payload["calls"]["dup2"]
    )
    assert payload["final_inheritable"] is payload["original_inheritable"]
    assert payload["configured"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "dup",
        "open",
        "redirect",
        "restore_fail",
        "restore",
        "close_devnull_fail",
        "close_devnull",
        "close_saved_fail",
        "close_saved",
    ],
)
def test_dependency_logging_native_uncertain_effect_hard_exits_real_process(
    fault: str,
) -> None:
    result = _run_isolated(
        _NATIVE_FATAL_FAULT_PROBE,
        {"PROJETV0_NATIVE_FAULT": fault, "PROJETV0_REAL_HARD_EXIT": "1"},
    )

    assert result.returncode == 71
    assert "POST-CALL-MARKER" not in result.stdout


def test_dependency_logging_rejects_a_returning_hard_exit_double() -> None:
    result = _run_isolated(
        _NATIVE_FATAL_FAULT_PROBE,
        {
            "PROJETV0_NATIVE_FAULT": "dup",
            "PROJETV0_REAL_HARD_EXIT": "0",
            "PROJETV0_RETURNING_HARD_EXIT": "1",
        },
    )

    assert result.returncode == 71
    assert "POST-CALL-MARKER" not in result.stdout
