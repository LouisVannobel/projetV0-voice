from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_runtime_contract.py"
GOLDEN = ROOT / "tests" / "fixtures" / "agent-bundle-v1-golden.json"

ENVIRONMENT_VARIABLE_NAMES = [
    "VOICE_RUNTIME_MODE",
    "VOICE_DEPLOYMENT_ID",
    "VOICE_RUNTIME_CONTRACT_PATH",
    "VOICE_AGENT_BUNDLE_PATH",
    "VOICE_QUALIFIED_PROFILE_PATH",
    "VOICE_QUALIFICATION_CANDIDATE_PATH",
    "VOICE_QUALIFICATION_OVERRIDE_PATH",
    "VOICE_KEYRING_PATH",
    "VOICE_SQLITE_PATH",
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
]

METRIC_NAMES = [
    "projetv0.voice.actions.total",
    "projetv0.voice.admission.rejections",
    "projetv0.voice.calls.active",
    "projetv0.voice.calls.total",
    "projetv0.voice.disclosure.mark_ack",
    "projetv0.voice.disclosure.timeouts",
    "projetv0.voice.outbox.bytes",
    "projetv0.voice.outbox.depth",
    "projetv0.voice.outbox.oldest_age",
    "projetv0.voice.ready",
    "projetv0.voice.recordings.total",
    "projetv0.voice.relay.runs",
    "projetv0.voice.runtime.event_loop_lag",
    "projetv0.voice.service_ttfb",
    "projetv0.voice.sessions.duration",
    "projetv0.voice.transcript.turns_lost",
    "projetv0.voice.user_bot_latency",
    "projetv0.voice.webhooks.total",
    "projetv0.voice.writer.queue_depth",
    "projetv0.voice.writer.queue_oldest_age",
    "projetv0.voice.writer.quick_check",
]

EXPECTED_CONTRACT: dict[str, object] = {
    "schema_version": 1,
    "environment_variable_names": ENVIRONMENT_VARIABLE_NAMES,
    "secret_files": {
        "directory": "/run/secrets",
        "fixed_paths": ["/run/secrets/aead_keyring_v1.json"],
        "path_environment_variables": [
            "VOICE_TELNYX_API_KEY_FILE",
            "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE",
            "VOICE_OPENROUTER_API_KEY_FILE",
            "VOICE_POSTGRES_DSN_FILE",
        ],
    },
    "filesystem": {
        "durable_writable_directories": ["/var/lib/projetv0-voice"],
        "ephemeral_writable_tmpfs": ["/tmp"],
        "read_only_directories": [
            "/run/secrets",
            "/opt/projetv0-voice/agents/active",
        ],
    },
    "endpoints": {
        "port": 8080,
        "routes": [
            {"method": "GET", "path": "/health/live"},
            {"method": "GET", "path": "/health/ready"},
            {"method": "POST", "path": "/telnyx/events"},
            {"method": "WSS", "path": "/telnyx/media"},
        ],
        "absent_paths": ["/metrics", "/docs", "/redoc", "/openapi.json"],
        "outbound_destination_environment_variables": [
            "VOICE_TELNYX_MEDIA_WSS_URL",
            "VOICE_OTLP_HTTP_ENDPOINT",
        ],
    },
    "metrics": METRIC_NAMES,
    "schemas": {
        "versions": {
            "runtime_contract": 1,
            "agent_manifest": 1,
            "inference_profile": 1,
            "qualified_profile": 1,
            "qualification_candidate": 1,
            "qualification_override": 1,
            "keyring": 1,
            "operation": 1,
        },
        "exported_profile_schema_files": [
            "qualified-deployment-profile-v1.schema.json",
            "qualification-candidate-profile-v1.schema.json",
            "qualification-override-v1.schema.json",
        ],
    },
    "agent_bundle": {
        "format": "projetv0-agent-bundle-v1",
        "path_encoding": "utf-8-strict",
        "path_style": "posix-relative",
        "unicode_normalization": "none",
        "max_files": 512,
        "max_components": 16,
        "max_component_utf8_bytes": 255,
        "max_relative_path_utf8_bytes": 4095,
        "max_cumulative_content_bytes": 16_777_216,
    },
    "token_locator_ids": ["telnyx-header-connected-v1"],
    "readiness": {
        "live": {
            "path": "/health/live",
            "status_while_event_loop_serves": 200,
        },
        "ready": {
            "path": "/health/ready",
            "status_when_all_factors_true": 200,
            "status_otherwise": 503,
            "required_factors": [
                "startup_profile_and_stale_recovery_complete",
                "not_draining",
                "admission_open",
                "qualification_state_valid",
                "writer_owner_alive_and_ready",
                "writer_quick_check_true",
                "no_writer_fatal_or_degradation",
                "relay_supervisor_alive",
                "no_permanent_relay_or_sink_degradation",
            ],
            "maximums": {
                "storage_bytes": 268_435_456,
                "writer_queue_oldest_age_seconds": 1.0,
                "outbox_oldest_age_seconds": 900.0,
            },
            "provider_availability_is_factor": False,
            "provider_io_is_factor": False,
        },
    },
    "digest_bindings": {
        "qualified_profile": [
            "runtime_contract_sha256",
            "image_digest",
            "agent_bundle_sha256",
            "inference_profile_sha256",
        ],
        "qualification_candidate_profile": [
            "runtime_contract_sha256",
            "image_digest",
            "agent_bundle_sha256",
            "inference_profile_sha256",
        ],
        "qualification_override": ["qualified_profile_sha256"],
    },
}


def _exporter() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.fail("Task 11A producer script is missing", pytrace=False)
    spec = importlib.util.spec_from_file_location("task11_exporter", SCRIPT)
    if spec is None or spec.loader is None:
        pytest.fail("Task 11A producer script cannot be loaded", pytrace=False)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_bytes())


def _golden_files(exporter: ModuleType) -> list[object]:
    golden = _golden()
    files = []
    for item in golden["files"]:
        content = bytes.fromhex(item["content_hex"])
        assert len(content) == item["size"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        files.append(exporter.BundleFile(item["path"], content))
    return files


def _write_golden_tree(root: Path, exporter: ModuleType, *, reverse: bool) -> None:
    files = _golden_files(exporter)
    if reverse:
        files.reverse()
    for index, file in enumerate(files):
        target = root.joinpath(*file.path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.content)
        stamp = 1_500_000_000 + index * 97
        os.utime(target, (stamp, stamp))


@contextmanager
def _opened_archive(raw: bytes) -> Iterator[tarfile.TarFile]:
    with tarfile.open(fileobj=__import__("io").BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        assert all(member.isreg() for member in members)
        yield archive


def test_runtime_contract_is_the_exact_canonical_v1_object() -> None:
    exporter = _exporter()

    observed = exporter.runtime_contract_bytes()

    assert observed == _canonical(EXPECTED_CONTRACT)
    assert len(observed) <= 1_048_576
    assert observed.startswith(b"{")
    assert observed.endswith(b"}\n")
    assert not observed.startswith(b"\xef\xbb\xbf")
    assert json.loads(observed) == EXPECTED_CONTRACT
    assert len(EXPECTED_CONTRACT["environment_variable_names"]) == 30  # type: ignore[arg-type]
    assert EXPECTED_CONTRACT["metrics"] == sorted(METRIC_NAMES)
    assert len(METRIC_NAMES) == 21


def test_golden_exercises_out_of_order_dotfile_and_distinct_nfc_nfd_paths() -> None:
    exporter = _exporter()
    golden = _golden()
    paths = [item["path"] for item in golden["files"]]

    assert paths != sorted(paths, key=lambda value: value.encode("utf-8"))
    assert any(path.startswith(".") for path in paths)
    assert "é.txt" in paths
    assert "é.txt" in paths
    assert "é.txt".encode() != "é.txt".encode()
    assert exporter.bundle_digest(_golden_files(exporter)) == golden[
        "expected_bundle_sha256"
    ]


def test_bundle_path_and_digest_bounds_are_closed() -> None:
    exporter = _exporter()
    digest = b"d" * 32

    assert len(exporter.validate_relative_path(tuple("x" * 255 for _ in range(16)))) == 4095
    for parts in [
        tuple("x" for _ in range(17)),
        ("x" * 256,),
        (".",),
        ("..",),
        ("",),
        ("slash/name",),
        ("\udcff",),
    ]:
        with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
            exporter.validate_relative_path(parts)

    accepted = [
        exporter.BundleLeaf(f"f{index:03d}", 0, digest) for index in range(512)
    ]
    assert len(exporter.bundle_digest_from_leaves(accepted)) == 64
    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.bundle_digest_from_leaves(
            [*accepted, exporter.BundleLeaf("overflow", 0, digest)]
        )
    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.bundle_digest_from_leaves(
            [
                exporter.BundleLeaf("same", 0, digest),
                exporter.BundleLeaf("same", 0, digest),
            ]
        )
    assert len(
        exporter.bundle_digest_from_leaves(
            [exporter.BundleLeaf("max", 16_777_216, digest)]
        )
    ) == 64
    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.bundle_digest_from_leaves(
            [exporter.BundleLeaf("overflow", 16_777_217, digest)]
        )


@pytest.mark.parametrize("count,accepted", [(512, True), (513, False)])
def test_source_tree_file_count_bound(tmp_path: Path, count: int, accepted: bool) -> None:
    exporter = _exporter()
    for index in range(count):
        (tmp_path / f"f{index:03d}").write_bytes(b"")

    if accepted:
        assert len(exporter.snapshot_agent_tree(tmp_path)) == 512
    else:
        with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
            exporter.snapshot_agent_tree(tmp_path)


@pytest.mark.parametrize("components,accepted", [(16, True), (17, False)])
def test_source_tree_component_depth_bound(
    tmp_path: Path, components: int, accepted: bool
) -> None:
    exporter = _exporter()
    parent = tmp_path
    for index in range(components - 1):
        parent /= f"d{index}"
        parent.mkdir()
    (parent / "leaf").write_bytes(b"")

    if accepted:
        assert len(exporter.snapshot_agent_tree(tmp_path)) == 1
    else:
        with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
            exporter.snapshot_agent_tree(tmp_path)


@pytest.mark.parametrize("size,accepted", [(16_777_216, True), (16_777_217, False)])
def test_source_tree_cumulative_content_bound(
    tmp_path: Path, size: int, accepted: bool
) -> None:
    exporter = _exporter()
    (tmp_path / "leaf").write_bytes(b"x" * size)

    if accepted:
        files = exporter.snapshot_agent_tree(tmp_path)
        assert (len(files), len(files[0].content)) == (1, 16_777_216)
    else:
        with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
            exporter.snapshot_agent_tree(tmp_path)


def test_source_tree_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    exporter = _exporter()
    target = tmp_path / "target"
    target.write_bytes(b"target")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.snapshot_agent_tree(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation unavailable")
def test_source_tree_rejects_fifo(tmp_path: Path) -> None:
    exporter = _exporter()
    os.mkfifo(tmp_path / "special")

    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.snapshot_agent_tree(tmp_path)


def test_source_tree_rejects_file_growth_on_same_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _exporter()
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"original")
    original_read = exporter.os.read
    mutated = False

    def read_and_grow(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            with leaf.open("ab") as stream:
                stream.write(b"growth")
        return chunk

    monkeypatch.setattr(exporter.os, "read", read_and_grow)
    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.snapshot_agent_tree(tmp_path)
    assert mutated is True


def test_source_tree_rejects_leaf_swap_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _exporter()
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"original")
    original_open = exporter.os.open
    swapped = False

    def open_and_swap(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if Path(path) == leaf and not swapped:
            swapped = True
            leaf.unlink()
            leaf.write_bytes(b"replacement")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(exporter.os, "open", open_and_swap)
    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.snapshot_agent_tree(tmp_path)
    assert swapped is True


def test_source_tree_rejects_directory_relist_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _exporter()
    (tmp_path / "leaf").write_bytes(b"stable")
    original_read = exporter.os.read
    mutated = False

    def read_and_add_sibling(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            (tmp_path / "late").write_bytes(b"late")
        return chunk

    monkeypatch.setattr(exporter.os, "read", read_and_add_sibling)
    with pytest.raises(exporter.ExportError, match="^agent_bundle_invalid$"):
        exporter.snapshot_agent_tree(tmp_path)
    assert mutated is True


def test_archive_is_deterministic_pax_with_explicit_regular_metadata(
    tmp_path: Path,
) -> None:
    exporter = _exporter()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_golden_tree(first, exporter, reverse=False)
    _write_golden_tree(second, exporter, reverse=True)

    first_raw = exporter.build_agent_archive(exporter.snapshot_agent_tree(first))
    second_raw = exporter.build_agent_archive(exporter.snapshot_agent_tree(second))

    assert first_raw == second_raw
    assert hashlib.sha256(first_raw).hexdigest() == hashlib.sha256(second_raw).hexdigest()
    assert first_raw[:3] == b"\x1f\x8b\x08"
    assert first_raw[3] & 0x08 == 0
    assert first_raw[4:8] == b"\x00\x00\x00\x00"
    with _opened_archive(first_raw) as archive:
        members = archive.getmembers()
        assert archive.pax_headers == {}
        assert [member.name for member in members] == sorted(
            [file.path for file in _golden_files(exporter)],
            key=lambda value: value.encode("utf-8"),
        )
        assert all(
            (
                member.uid,
                member.gid,
                member.uname,
                member.gname,
                member.mtime,
                stat.S_IMODE(member.mode),
            )
            == (0, 0, "", "", 0, 0o644)
            for member in members
        )
        extracted = tmp_path / "extracted"
        extracted.mkdir()
        archive.extractall(extracted, filter="data")
    assert (extracted / ".agent").read_bytes() == b"hidden"
    assert (extracted / "é.txt").read_bytes() == b"nfc"
    assert (extracted / "é.txt").read_bytes() == b"nfd"


def test_export_writes_exact_six_artifacts_and_authoritative_schema_bytes(
    tmp_path: Path,
) -> None:
    exporter = _exporter()
    output = tmp_path / "dist"

    result = exporter.export_all(ROOT, output)

    expected_names = {
        "runtime-contract.json",
        "qualified-deployment-profile-v1.schema.json",
        "qualification-candidate-profile-v1.schema.json",
        "qualification-override-v1.schema.json",
        "agent-a-bundle.tar.gz",
        "agent-a-bundle-v1.manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected_names
    assert result.runtime_contract_sha256 == hashlib.sha256(
        _canonical(EXPECTED_CONTRACT)
    ).hexdigest()
    assert (output / "runtime-contract.json").read_bytes() == _canonical(
        EXPECTED_CONTRACT
    )
    for source_name, output_name in {
        "qualified-v1.schema.json": "qualified-deployment-profile-v1.schema.json",
        "qualification-candidate-v1.schema.json": (
            "qualification-candidate-profile-v1.schema.json"
        ),
        "qualification-override-v1.schema.json": "qualification-override-v1.schema.json",
    }.items():
        assert (output / output_name).read_bytes() == (
            ROOT / "deployment-profiles" / source_name
        ).read_bytes()

    manifest_raw = (output / "agent-a-bundle-v1.manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical(manifest)
    assert set(manifest) == {
        "schema_version",
        "bundle_format",
        "bundle_sha256",
        "archive_sha256",
        "files",
    }
    assert manifest["schema_version"] == 1
    assert manifest["bundle_format"] == "projetv0-agent-bundle-v1"
    assert manifest["bundle_sha256"] == result.agent_bundle_sha256
    assert manifest["archive_sha256"] == hashlib.sha256(
        (output / "agent-a-bundle.tar.gz").read_bytes()
    ).hexdigest()
    assert manifest["files"] == sorted(
        manifest["files"], key=lambda item: item["path"].encode("utf-8")
    )
    assert all(set(item) == {"path", "size", "sha256"} for item in manifest["files"])
    with _opened_archive((output / "agent-a-bundle.tar.gz").read_bytes()) as archive:
        assert [member.name for member in archive.getmembers()] == [
            item["path"] for item in manifest["files"]
        ]


def test_cli_exports_without_live_configuration(tmp_path: Path) -> None:
    output = tmp_path / "dist"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(ROOT),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert len(list(output.iterdir())) == 6
