from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import SecretStr

from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    QualifiedDeploymentProfileV1,
)
from projetv0_voice.runtime_config import RuntimeSettingsV1

KEYRING_PATH = PurePosixPath("/run/secrets/aead_keyring_v1.json")
DEFAULT_RUNTIME_PATH = PurePosixPath("/srv/projetv0/runtime-contract.json")
DEFAULT_BUNDLE_PATH = PurePosixPath("/srv/projetv0/agent-bundle")
RUNTIME_BYTES = b"runtime\n"
RUNTIME_SHA256 = "fae9d8f386d67956867dedef7c89476199a4a25ee9ffe13560a6bfae7ae6c407"
BUNDLE_GOLDEN_SHA256 = "ff95531825d43b758ac6cf8110966a8b8e95ef13140f5f0f56d1d47f1b07b684"
EMPTY_BUNDLE_SHA256 = "1d82e411b7a8587b2e105d924b9031d4fca7dfed9373c64cc5ebad71cdfa7c08"
MANIFEST_BYTES = (
    b'{"schema_version":1,"tenant_id":"t","agent_id":"a","revision":"r",'
    b'"dids":["+331"],"language":"fr","prompt_path":"prompt.md",'
    b'"prompt_revision":"p","greeting":"Bonjour","conversation_mode":"freeform",'
    b'"max_concurrent_calls":1,"direction":"inbound_only","transport_codec":"PCMU",'
    b'"transport_sample_rate_hz":8000,"transcript_retention_days":7,'
    b'"recording_mode":"off","recording_format":"wav",'
    b'"recording_retention_days":null,"recording_required":false,'
    b'"recording_play_beep":false}'
)
UTF8_BUNDLE_GOLDEN = "12cb8fc96a5fd5a9105538aeede00dfcd2335090a759106982807f04e36163ca"
TASK11_GOLDEN_PATH = Path("tests/fixtures/agent-bundle-v1-golden.json")
PRIVILEGED_FILES = (
    os.name == "posix"
    and os.environ.get("PROJETV0_PRIVILEGED_FILES_GATE") == "1"
    and getattr(os, "geteuid", lambda: -1)() == 0
)


def test_runtime_fixture_matches_literal_sha256() -> None:
    assert hashlib.sha256(RUNTIME_BYTES).hexdigest() == RUNTIME_SHA256


def _settings(
    *,
    runtime_contract_path: PurePosixPath = DEFAULT_RUNTIME_PATH,
    agent_bundle_path: PurePosixPath = DEFAULT_BUNDLE_PATH,
    keyring_path: PurePosixPath = KEYRING_PATH,
    runtime_sha256: str = "a" * 64,
    bundle_sha256: str = "b" * 64,
    deployment_max_calls: int = 1,
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
        deployment_max_calls=deployment_max_calls,
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


@pytest.mark.parametrize("configured", [False, True])
def test_recording_archive_factory_keeps_off_without_unverified_descriptor(
    tmp_path, monkeypatch, configured
):
    from projetv0_voice import production_wiring as wiring
    from projetv0_voice.crypto import CryptoKeyring
    from projetv0_voice.persistence.writer import PersistenceWriter

    settings = replace(
        _settings(),
        recording_archive_directory=(
            PurePosixPath("/var/lib/projetv0/audio") if configured else None
        ),
        recording_download_origins=("https://recordings.example.invalid",) if configured else (),
    )
    observations = []

    def invalid_parent(path):
        observations.append(path)
        raise wiring._WiringInvalid

    def forbidden_consumer(**_kwargs):
        pytest.fail("unverified storage constructed a media consumer")

    monkeypatch.setattr(wiring, "_open_parent", invalid_parent)
    monkeypatch.setattr(wiring, "RecordingArchive", forbidden_consumer)
    factories = wiring.build_production_factories(settings)
    assert callable(factories.archive_factory)
    keyring = CryptoKeyring({1: bytes(range(32))}, active_version=1)
    writer = PersistenceWriter(tmp_path / "factory.sqlite", keyring)
    assert factories.archive_factory(
        settings, writer, keyring, object(), lambda: datetime(2026, 10, 4, tzinfo=UTC)
    ) is None
    assert observations == ([settings.recording_archive_directory] if configured else [])


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

    golden = json.loads(TASK11_GOLDEN_PATH.read_bytes())
    assert [item["path"] for item in golden["files"]] != sorted(
        (item["path"] for item in golden["files"]),
        key=lambda path: path.encode("utf-8"),
    )
    entries: list[tuple[bytes, int, bytes]] = []
    for item in golden["files"]:
        content = bytes.fromhex(item["content_hex"])
        assert len(content) == item["size"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        entries.append(
            (
                item["path"].encode("utf-8"),
                item["size"],
                bytes.fromhex(item["sha256"]),
            )
        )

    entries.sort(key=lambda entry: entry[0])
    assert _bundle_digest(tuple(entries)) == golden["expected_bundle_sha256"]


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
    directory = os.stat_result((0o40700, 1, 0, 0, 0, 0, 0, 0, 0, 0))
    child = os.stat_result((0o40700, 2, 0, 0, 0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(wiring.os, "fstat", lambda _descriptor: directory)
    monkeypatch.setattr(wiring, "_safe_mode", lambda _value, *, directory: True)
    monkeypatch.setattr(
        wiring,
        "_snapshot_directory",
        lambda _descriptor: {"trusted": wiring._directory_stat(child)},
    )
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


def test_opened_child_directory_must_match_snapshotted_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stat

    import projetv0_voice.production_wiring as wiring

    parent = os.stat_result((stat.S_IFDIR | 0o755, 10, 1, 1, 0, 0, 0, 0, 0, 0))
    entry = os.stat_result((stat.S_IFDIR | 0o755, 20, 1, 1, 0, 0, 0, 0, 0, 0))
    opened = os.stat_result((stat.S_IFDIR | 0o755, 21, 1, 1, 0, 0, 0, 0, 0, 0))
    snapshots = {
        1: {"child": wiring._directory_stat(entry)},
        2: {},
    }
    fake_os = SimpleNamespace(
        fstat=lambda descriptor: parent if descriptor == 1 else opened,
        stat=lambda _name, *, dir_fd, follow_symlinks: entry,
        open=lambda _name, _flags, *, dir_fd: 2,
        close=lambda _descriptor: None,
    )
    monkeypatch.setattr(wiring, "os", fake_os)
    monkeypatch.setattr(wiring, "_safe_mode", lambda _value, *, directory: True)
    monkeypatch.setattr(
        wiring, "_snapshot_directory", lambda descriptor: snapshots[descriptor]
    )
    monkeypatch.setattr(wiring, "_linux_flags", lambda *, directory: 0)

    with pytest.raises(wiring._WiringInvalid):
        wiring._walk_bundle(1, parts=(), entries=[], total=[0])


def test_bundle_count_and_cumulative_content_bounds_are_closed() -> None:
    from projetv0_voice.production_wiring import _bundle_digest

    digest = b"d" * 32
    accepted_count = tuple(
        (f"f{index:03d}".encode(), 0, digest) for index in range(512)
    )
    assert len(_bundle_digest(accepted_count)) == 64
    with pytest.raises(ValueError, match="^bundle_entry_invalid$"):
        _bundle_digest((*accepted_count, (b"overflow", 0, digest)))

    assert len(_bundle_digest(((b"max", 16_777_216, digest),))) == 64
    with pytest.raises(ValueError, match="^bundle_entry_invalid$"):
        _bundle_digest(((b"overflow", 16_777_217, digest),))


def test_relative_path_exact_component_depth_and_byte_bounds() -> None:
    from projetv0_voice.production_wiring import _relative_bytes, _WiringInvalid

    assert len(_relative_bytes(tuple("x" * 255 for _ in range(16)))) == 4_095
    with pytest.raises(_WiringInvalid):
        _relative_bytes(tuple("x" for _ in range(17)))
    with pytest.raises(_WiringInvalid):
        _relative_bytes(("x" * 256,))
    with pytest.raises(_WiringInvalid):
        _relative_bytes((*tuple("x" * 255 for _ in range(15)), "y" * 256))


def test_ancestor_authority_rejects_reverse_name_or_stat_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

    stable = os.stat_result((0, 2, 1, 1, 0, 0, 0, 0, 0, 0))
    authority_type = wiring._AncestorAuthority
    verify = wiring._verify_ancestor_authorities
    authority = authority_type(
        descriptor=10,
        directory_stat=wiring._directory_stat(stable),
        entries={"child": (1, 3, 4, 5, 6, 7, 0, 0)},
    )
    monkeypatch.setattr(
        wiring,
        "_snapshot_directory",
        lambda _descriptor: {"mutated": (1, 3, 4, 5, 6, 7, 0, 0)},
    )
    monkeypatch.setattr(
        wiring.os,
        "fstat",
        lambda _descriptor: stable,
    )

    with pytest.raises(wiring._WiringInvalid):
        verify((authority,))


def test_qualified_and_candidate_profile_select_eu(monkeypatch: pytest.MonkeyPatch) -> None:
    import projetv0_voice.production_wiring as wiring

    qualified = QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(encoding="utf-8")
    )
    candidate = QualificationCandidateProfileV1.model_validate({
        **qualified.model_dump(exclude={"telnyx_data_locality", "qualified_at"}),
        "run_id": UUID("00000000-0000-4000-8000-000000000001"),
        "expires_at": qualified.qualified_at,
        "benchmark_did_hash": "b" * 64,
        "max_concurrent_calls": 1,
        "call_lease_ttl_seconds": 30,
        "disclosure_mark_timeout_ms": 10000,
    })
    constructions = []
    monkeypatch.setattr(wiring, "CallControlClient", lambda **kwargs: constructions.append(kwargs))
    factories = wiring.build_production_factories(_settings())
    assert constructions == []
    for profile in (qualified, candidate):
        factories.call_control_factory(SecretStr("synthetic-telnyx"), profile)
    assert constructions == [
        {"api_key": "synthetic-telnyx", "api_region": "EU"},
        {"api_key": "synthetic-telnyx", "api_region": "EU"},
    ]
    for key, profile in (
        (SecretStr("synthetic-telnyx"), object()),
        (SecretStr("synthetic-telnyx"),
         qualified.model_copy(update={"telnyx_data_locality": "US"})),
        ("synthetic-telnyx", qualified),
        (SecretStr(""), qualified),
    ):
        with pytest.raises(RuntimeError, match="^runtime_call_control_invalid$"):
            factories.call_control_factory(key, profile)
    assert len(constructions) == 2


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

    def http_client_factory(**kwargs: object) -> object:
        client = object()
        events.append(("http", (client, kwargs)))
        return client

    monkeypatch.setattr(wiring, "DefaultAsyncHttpxClient", http_client_factory)
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
        lambda *, api_key, api_region: (
            events.append(("control", (api_key, api_region))) or object()
        ),
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
    factories.call_control_factory(SecretStr("telnyx"), profile)

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
    assert events[0][1][1] == {"trust_env": False}  # type: ignore[index]
    assert events[1][1][1] == {"trust_env": False}  # type: ignore[index]


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

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
        original_walk = wiring._walk_bundle
        walk_reached = False

        def walk_special_bundle(*args: object, **kwargs: object) -> None:
            nonlocal walk_reached
            walk_reached = True
            original_walk(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(wiring, "_walk_bundle", walk_special_bundle)
        try:
            wiring._validate_runtime_contract(settings)
            with pytest.raises(wiring._WiringInvalid):
                wiring._validate_agent_bundle(settings)
            assert walk_reached is True
            with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
                wiring.build_production_factories(settings).validate_artifacts(settings)
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
        wiring._validate_runtime_contract(settings)
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


def _walk_real_bundle(path: Path) -> tuple[list[tuple[bytes, int, bytes]], int]:
    import projetv0_voice.production_wiring as wiring

    descriptor = os.open(path, wiring._linux_flags(directory=True))
    entries: list[tuple[bytes, int, bytes]] = []
    total = [0]
    try:
        wiring._walk_bundle(descriptor, parts=(), entries=entries, total=total)
    finally:
        os.close(descriptor)
    return entries, total[0]


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize("count,accepted", [(512, True), (513, False)])
def test_linux_kernel_bundle_file_count_bound(count: int, accepted: bool) -> None:
    import projetv0_voice.production_wiring as wiring

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        bundle = Path(raw) / "bundle"
        bundle.mkdir()
        for index in range(count):
            (bundle / f"f{index:03d}").write_bytes(b"")
        if accepted:
            entries, total = _walk_real_bundle(bundle)
            assert (len(entries), total) == (512, 0)
        else:
            with pytest.raises(wiring._WiringInvalid):
                _walk_real_bundle(bundle)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize("components,accepted", [(16, True), (17, False)])
def test_linux_kernel_bundle_component_depth_bound(
    components: int,
    accepted: bool,
) -> None:
    import projetv0_voice.production_wiring as wiring

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        bundle = Path(raw) / "bundle"
        bundle.mkdir()
        parent = bundle
        for index in range(components - 1):
            parent /= f"d{index}"
            parent.mkdir()
        (parent / "leaf").write_bytes(b"")
        if accepted:
            entries, _total = _walk_real_bundle(bundle)
            assert len(entries) == 1
        else:
            with pytest.raises(wiring._WiringInvalid):
                _walk_real_bundle(bundle)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize(
    "size,accepted",
    [(16_777_216, True), (16_777_217, False)],
)
def test_linux_kernel_bundle_cumulative_content_bound(
    size: int,
    accepted: bool,
) -> None:
    import projetv0_voice.production_wiring as wiring

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        bundle = Path(raw) / "bundle"
        bundle.mkdir()
        (bundle / "leaf").write_bytes(b"x" * size)
        if accepted:
            entries, total = _walk_real_bundle(bundle)
            assert (len(entries), total) == (1, 16_777_216)
        else:
            with pytest.raises(wiring._WiringInvalid):
                _walk_real_bundle(bundle)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_linux_kernel_runtime_contract_non_regular_leaf_is_rejected(kind: str) -> None:
    from projetv0_voice.production_wiring import build_production_factories

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        target = trusted / "target"
        target.write_bytes(RUNTIME_BYTES)
        runtime = trusted / "runtime.json"
        if kind == "symlink":
            runtime.symlink_to(target)
        else:
            runtime.mkdir()
        bundle = trusted / "bundle"
        bundle.mkdir()
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=RUNTIME_SHA256,
            bundle_sha256=EMPTY_BUNDLE_SHA256,
        )
        with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
            build_production_factories(settings).validate_artifacts(settings)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
@pytest.mark.parametrize("mutation", ["mode", "owner"])
def test_linux_kernel_runtime_contract_permissions_and_ownership(
    mutation: str,
) -> None:
    from projetv0_voice.production_wiring import build_production_factories

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        runtime = trusted / "runtime.json"
        runtime.write_bytes(RUNTIME_BYTES)
        if mutation == "mode":
            runtime.chmod(0o666)
        else:
            os.chown(runtime, 1000, 0)
        bundle = trusted / "bundle"
        bundle.mkdir()
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=RUNTIME_SHA256,
            bundle_sha256=EMPTY_BUNDLE_SHA256,
        )
        with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
            build_production_factories(settings).validate_artifacts(settings)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
def test_linux_kernel_ancestor_mutation_after_descent_is_rejected(
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
            bundle_sha256=EMPTY_BUNDLE_SHA256,
        )
        original_read = wiring.os.read
        original_verify = wiring._verify_ancestor_authorities
        mutated = False
        verified = False

        def read_and_mutate(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            chunk = original_read(descriptor, count)
            if chunk and not mutated:
                mutated = True
                (trusted / "new-entry").write_bytes(b"mutation")
            return chunk

        def verify_mutated_authority(
            authorities: tuple[wiring._AncestorAuthority, ...],
        ) -> None:
            nonlocal verified
            verified = True
            original_verify(authorities)

        monkeypatch.setattr(wiring.os, "read", read_and_mutate)
        monkeypatch.setattr(
            wiring,
            "_verify_ancestor_authorities",
            verify_mutated_authority,
        )
        with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
            wiring.build_production_factories(settings).validate_artifacts(settings)
        assert mutated is True
        assert verified is True


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
def test_linux_kernel_raw_utf8_order_and_real_manifest_load_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projetv0_voice.production_wiring as wiring

    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-wiring-") as raw:
        trusted = Path(raw)
        runtime = trusted / "runtime.json"
        runtime.write_bytes(RUNTIME_BYTES)
        bundle = trusted / "bundle"
        bundle.mkdir()
        for name, content in {
            ".dot": b"d",
            "manifest.yaml": MANIFEST_BYTES,
            "prompt.md": b"prompt",
            "z.txt": b"z",
            "é.txt": b"e",
        }.items():
            (bundle / name).write_bytes(content)
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=RUNTIME_SHA256,
            bundle_sha256=UTF8_BUNDLE_GOLDEN,
        )
        real_load = wiring.load_agent_manifest
        calls = 0

        def counted_load(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return real_load(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(wiring, "load_agent_manifest", counted_load)
        factories = wiring.build_production_factories(settings)
        factories.validate_artifacts(settings)
        first = factories.load_manifest(settings)
        second = factories.load_manifest(settings)

        assert first is second
        assert first.language == "fr"
        assert calls == 1


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
def test_linux_kernel_success_and_failure_close_every_descriptor() -> None:
    from projetv0_voice.production_wiring import build_production_factories

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
            bundle_sha256=EMPTY_BUNDLE_SHA256,
        )
        before = len(os.listdir("/proc/self/fd"))
        build_production_factories(settings).validate_artifacts(settings)
        after_success = len(os.listdir("/proc/self/fd"))
        invalid = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256="0" * 64,
            bundle_sha256=EMPTY_BUNDLE_SHA256,
        )
        with pytest.raises(RuntimeError, match="^runtime_artifacts_invalid$"):
            build_production_factories(invalid).validate_artifacts(invalid)
        after_failure = len(os.listdir("/proc/self/fd"))

        assert (after_success, after_failure) == (before, before)


@pytest.mark.skipif(not PRIVILEGED_FILES, reason="privileged Linux descriptor gate")
def test_linux_kernel_accepts_real_task11_export_without_importing_producer() -> None:
    from projetv0_voice.production_wiring import build_production_factories

    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(dir="/root", prefix="voice-export-") as raw:
        trusted = Path(raw)
        output = trusted / "dist"
        completed = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "export_runtime_contract.py"),
                "--repo-root",
                str(repository),
                "--output-dir",
                str(output),
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""

        archive_path = output / "agent-a-bundle.tar.gz"
        bundle = trusted / "bundle"
        bundle.mkdir()
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            assert members
            assert all(member.isreg() for member in members)
            archive.extractall(bundle, filter="data")

        runtime = output / "runtime-contract.json"
        manifest = json.loads(
            (output / "agent-a-bundle-v1.manifest.json").read_bytes()
        )
        settings = _settings(
            runtime_contract_path=PurePosixPath(str(runtime)),
            agent_bundle_path=PurePosixPath(str(bundle)),
            runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
            bundle_sha256=manifest["bundle_sha256"],
            deployment_max_calls=10,
        )

        factories = build_production_factories(settings)
        factories.validate_artifacts(settings)
        assert factories.load_manifest(settings).agent_id == "agent-a"
