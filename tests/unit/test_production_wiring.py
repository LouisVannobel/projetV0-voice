from __future__ import annotations

import json
import os
import socket
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.runtime_config import RuntimeSettingsV1

KEYRING_PATH = PurePosixPath("/run/secrets/aead_keyring_v1.json")
DEFAULT_RUNTIME_PATH = PurePosixPath("/srv/projetv0/runtime-contract.json")
DEFAULT_BUNDLE_PATH = PurePosixPath("/srv/projetv0/agent-bundle")
RUNTIME_BYTES = b"runtime\n"
RUNTIME_SHA256 = "b45fd73aa413c685440da5ae5c63c69398b339f9cf7a73f4d4324ac8577dc3d1"
BUNDLE_GOLDEN_SHA256 = "ff95531825d43b758ac6cf8110966a8b8e95ef13140f5f0f56d1d47f1b07b684"
EMPTY_BUNDLE_SHA256 = "1d82e411b7a8587b2e105d924b9031d4fca7dfed9373c64cc5ebad71cdfa7c08"
PRIVILEGED_FILES = (
    os.name == "posix"
    and os.environ.get("PROJETV0_PRIVILEGED_FILES_GATE") == "1"
    and getattr(os, "geteuid", lambda: -1)() == 0
)


def _settings(
    *,
    runtime_contract_path: PurePosixPath = DEFAULT_RUNTIME_PATH,
    agent_bundle_path: PurePosixPath = DEFAULT_BUNDLE_PATH,
    keyring_path: PurePosixPath = KEYRING_PATH,
    runtime_sha256: str = "a" * 64,
    bundle_sha256: str = "b" * 64,
) -> RuntimeSettingsV1:
    return RuntimeSettingsV1(
        runtime_mode="strict",
        deployment_id="voice-agent-a",
        runtime_contract_path=runtime_contract_path,
        agent_bundle_path=agent_bundle_path,
        qualified_profile_path=PurePosixPath("/srv/projetv0/qualified.json"),
        qualification_candidate_path=None,
        qualification_override_path=None,
        keyring_path=keyring_path,
        sqlite_path=PurePosixPath("/var/lib/projetv0/voice.sqlite3"),
        runtime_contract_sha256=runtime_sha256,
        image_digest=f"ghcr.io/example/voice@sha256:{'d' * 64}",
        agent_bundle_sha256=bundle_sha256,
        inference_profile_sha256="c" * 64,
        qualification_run_id=None,
        benchmark_did_sha256=None,
        deployment_max_calls=1,
        handshake_timeout_seconds=5,
        call_idle_timeout_seconds=300,
        call_cleanup_phase_timeout_seconds=10,
        pre_drain_grace_seconds=15,
        uvicorn_grace_seconds=20,
        shutdown_grace_seconds=30,
        telnyx_api_key_file=PurePosixPath("/run/secrets/telnyx-api-key"),
        telnyx_webhook_public_key_file=PurePosixPath(
            "/run/secrets/telnyx-webhook-key"
        ),
        openrouter_api_key_file=PurePosixPath("/run/secrets/openrouter-api-key"),
        postgres_dsn_file=PurePosixPath("/run/secrets/postgres-dsn"),
        telnyx_media_wss_url="wss://voice.invalid/telnyx/media",
        otlp_http_endpoint="https://collector.invalid/v1/metrics",
        bind_host="127.0.0.1",
        bind_port=8080,
    )


def _keyring_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "active_version": 2,
            "keys": [
                {"version": 1, "aes256_key_hex": "01" * 32},
                {"version": 2, "aes256_key_hex": "02" * 32},
            ],
        },
        separators=(",", ":"),
    )


def test_keyring_v1_decodes_closed_schema_and_retains_old_versions() -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from projetv0_voice.crypto import EncryptedValue
    from projetv0_voice.production_wiring import _decode_keyring_secret

    keyring = _decode_keyring_secret(SecretStr(_keyring_json()))
    nonce = bytes(range(12))
    old = EncryptedValue(
        key_version=1,
        nonce=nonce,
        ciphertext=AESGCM(bytes.fromhex("01" * 32)).encrypt(
            nonce,
            b"old ciphertext",
            b"restore",
        ),
    )

    assert keyring.active_version == 2
    assert keyring.decrypt(old, aad=b"restore") == b"old ciphertext"
    current = keyring.encrypt(b"current", aad=b"turn")
    assert current.key_version == 2
    assert keyring.decrypt(current, aad=b"turn") == b"current"


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1,"active_version":1,"keys":[]}',
        '{"schema_version":NaN,"active_version":1,"keys":[]}',
        '{"schema_version":1,"active_version":true,"keys":[]}',
        '{"schema_version":1,"active_version":0,"keys":[]}',
        '{"schema_version":1,"active_version":9223372036854775808,"keys":[]}',
        '{"schema_version":1,"active_version":1,"keys":[]}',
        (
            '{"schema_version":1,"active_version":2,"keys":['
            '{"version":1,"aes256_key_hex":"' + "01" * 32 + '"},'
            '{"version":1,"aes256_key_hex":"' + "02" * 32 + '"}]}'
        ),
        (
            '{"schema_version":1,"active_version":1,"keys":['
            '{"version":1,"aes256_key_hex":"' + "AA" * 32 + '"}]}'
        ),
        (
            '{"schema_version":1,"active_version":1,"keys":['
            '{"version":1,"aes256_key_hex":"' + "01" * 32 + '","extra":0}]}'
        ),
        '{"schema_version":1,"active_version":1,"keys":[],"extra":0}',
    ],
)
def test_keyring_v1_rejects_ambiguous_values_without_raw_traceback_locals(
    raw: str,
) -> None:
    from projetv0_voice.production_wiring import _decode_keyring_secret

    sentinel = "RAW-KEYRING-SENTINEL"
    with pytest.raises(RuntimeError, match="^runtime_keyring_invalid$") as raised:
        _decode_keyring_secret(SecretStr(raw + sentinel))

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    rendered = repr(raised.value) + "".join(traceback.format_exception(raised.value))
    assert sentinel not in rendered
    current = raised.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "projetv0_voice.production_wiring"
        ):
            assert sentinel not in repr(current.tb_frame.f_locals)
        current = current.tb_next


def test_keyring_file_is_exact_and_read_once_without_hot_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

    calls: list[PurePosixPath] = []

    def read(path: PurePosixPath) -> SecretStr:
        calls.append(path)
        return SecretStr(_keyring_json())

    monkeypatch.setattr(wiring, "read_runtime_secret", read)
    factories = wiring.build_production_factories(_settings())

    first = factories.load_keyring(_settings())
    second = factories.load_keyring(_settings())

    assert first is second
    assert calls == [KEYRING_PATH]
    with pytest.raises(ValueError, match="^production_wiring_config_invalid$"):
        wiring.build_production_factories(
            _settings(keyring_path=PurePosixPath("/run/secrets/other.json"))
        )


def test_bundle_consumer_matches_literal_task11_golden_without_a_producer() -> None:
    from projetv0_voice.production_wiring import _bundle_digest

    assert (
        _bundle_digest(
            (
                (
                    b".env",
                    1,
                    bytes.fromhex(
                        "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
                    ),
                ),
                (
                    b"manifest.yaml",
                    5,
                    bytes.fromhex(
                        "d4f0bc5a29de06b510f9aa428f1eedba926012b591fef7a518e776a7c9bd1824"
                    ),
                ),
            )
        )
        == BUNDLE_GOLDEN_SHA256
    )


def test_rejected_ancestor_open_closes_every_acquired_descriptor_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

    opened: list[object] = []
    closed: list[int] = []

    def open_path(path: object, _flags: int, **_kwargs: object) -> int:
        opened.append(path)
        if len(opened) == 1:
            return 101
        raise OSError("synthetic open failure")

    monkeypatch.setattr(wiring, "_linux_flags", lambda *, directory: 0)
    monkeypatch.setattr(wiring.os, "open", open_path)
    monkeypatch.setattr(wiring.os, "fstat", lambda _descriptor: object())
    monkeypatch.setattr(wiring, "_safe_mode", lambda _value, *, directory: True)
    monkeypatch.setattr(wiring.os, "close", closed.append)

    with pytest.raises(OSError, match="synthetic open failure"):
        wiring._open_parent(PurePosixPath("/trusted/leaf"))

    assert opened == ["/", "trusted"]
    assert closed == [101]


def test_bundle_entry_stat_must_match_snapshot_before_descent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stat

    import projetv0_voice.production_wiring as wiring

    before = os.stat_result((stat.S_IFREG | 0o644, 10, 1, 1, 0, 0, 0, 0, 0, 0))
    changed = os.stat_result((stat.S_IFREG | 0o644, 11, 1, 1, 0, 0, 0, 0, 0, 0))
    snapshot = {"file": wiring._directory_stat(before)}
    monkeypatch.setattr(
        wiring,
        "os",
        SimpleNamespace(
            fstat=lambda _descriptor: before,
            stat=lambda _name, *, dir_fd, follow_symlinks: changed,
        ),
    )
    monkeypatch.setattr(wiring, "_safe_mode", lambda _value, *, directory: True)
    monkeypatch.setattr(wiring, "_snapshot_directory", lambda _descriptor: snapshot)
    monkeypatch.setattr(
        wiring,
        "_read_stable_file",
        lambda *_args, **_kwargs: b"x",
    )

    with pytest.raises(wiring._WiringInvalid):
        wiring._walk_bundle(1, parts=(), entries=[], total=[0])


def test_provider_factories_are_lazy_and_create_fresh_per_session_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

    profile = QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(
            encoding="utf-8"
        )
    )
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        wiring,
        "DefaultAsyncHttpxClient",
        lambda: events.append(("http", object())) or events[-1][1],
    )
    monkeypatch.setattr(
        wiring,
        "build_stt",
        lambda selected, key, *, language, http_client: (
            events.append(("stt", (selected, key, language, http_client))) or object()
        ),
    )
    monkeypatch.setattr(
        wiring,
        "build_llm",
        lambda selected, key: events.append(("llm", (selected, key))) or object(),
    )
    monkeypatch.setattr(
        wiring,
        "OpenRouterTTSService",
        lambda *, profile, api_key: (
            events.append(("tts", (profile, api_key))) or object()
        ),
    )
    monkeypatch.setattr(
        wiring,
        "PsycopgOperationSink",
        lambda dsn: events.append(("sink", dsn)) or object(),
    )
    monkeypatch.setattr(
        wiring,
        "CallControlClient",
        lambda *, api_key: events.append(("control", api_key)) or object(),
    )

    factories = wiring.build_production_factories(_settings())
    assert events == []
    inference = factories.inference_factory(SecretStr("openrouter"), profile, "fr")
    assert events == []

    first_client = inference.stt_http_client_factory()
    second_client = inference.stt_http_client_factory()
    assert first_client is not second_client
    inference.stt_factory(first_client)
    first_llm = inference.llm_factory()
    second_llm = inference.llm_factory()
    first_tts = inference.tts_factory()
    second_tts = inference.tts_factory()
    assert first_llm is not second_llm
    assert first_tts is not second_tts
    factories.sink_factory(SecretStr("postgres"))
    factories.call_control_factory(SecretStr("telnyx"))

    assert [name for name, _ in events] == [
        "http",
        "http",
        "stt",
        "llm",
        "llm",
        "tts",
        "tts",
        "sink",
        "control",
    ]
    assert events[2][1][2] == "fr"  # type: ignore[index]


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
def test_linux_kernel_exact_runtime_and_bundle_golden_are_accepted() -> None:
    from projetv0_voice.production_wiring import build_production_factories

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        runtime = trusted / "runtime.json"
        runtime.write_bytes(RUNTIME_BYTES)
        bundle = trusted / "bundle"
        bundle.mkdir()
        (bundle / ".env").write_bytes(b"x")
        (bundle / "manifest.yaml").write_bytes(b"agent")
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=RUNTIME_SHA256,
            bundle_sha256=BUNDLE_GOLDEN_SHA256,
        )

        build_production_factories(settings).validate_artifacts(settings)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize("size,accepted", [(1_048_576, True), (1_048_577, False)])
def test_linux_kernel_runtime_exact_byte_bound(
    size: int,
    accepted: bool,
) -> None:
    import hashlib

    from projetv0_voice.production_wiring import build_production_factories

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        runtime = trusted / "runtime.json"
        content = b"r" * size
        runtime.write_bytes(content)
        bundle = trusted / "bundle"
        bundle.mkdir()
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=hashlib.sha256(content).hexdigest(),
            bundle_sha256=EMPTY_BUNDLE_SHA256,
        )
        validate = build_production_factories(settings).validate_artifacts

        if accepted:
            validate(settings)
        else:
            with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
                validate(settings)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize("kind", ["fifo", "socket"])
def test_linux_fifo_or_socket_bundle_leaf_is_rejected(
    kind: str,
) -> None:
    from projetv0_voice.production_wiring import build_production_factories

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        runtime = trusted / "runtime.json"
        runtime.write_bytes(RUNTIME_BYTES)
        bundle = trusted / "bundle"
        bundle.mkdir()
        special = bundle / "special"
        owner: socket.socket | None = None
        if kind == "fifo":
            os.mkfifo(special)
        else:
            owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            owner.bind(str(special))
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=RUNTIME_SHA256,
        )
        try:
            with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
                build_production_factories(settings).validate_artifacts(settings)
        finally:
            if owner is not None:
                owner.close()


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
def test_linux_runtime_grows_after_same_fd_read_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        runtime = trusted / "runtime.json"
        runtime.write_bytes(RUNTIME_BYTES)
        bundle = trusted / "bundle"
        bundle.mkdir()
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=RUNTIME_SHA256,
        )
        original_read = wiring.os.read
        mutated = False

        def read_and_grow(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            chunk = original_read(descriptor, count)
            if chunk and not mutated:
                mutated = True
                with runtime.open("ab") as stream:
                    stream.write(b"growth")
            return chunk

        monkeypatch.setattr(wiring.os, "read", read_and_grow)
        with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
            wiring.build_production_factories(settings).validate_artifacts(settings)
        assert mutated is True
