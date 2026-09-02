"""Export the immutable Voice runtime contract and agent bundle handoff."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_FORMAT = "projetv0-agent-bundle-v1"
BUNDLE_DOMAIN = b"projetv0-agent-bundle-v1\x00"
MAX_CONTRACT_BYTES = 1_048_576
MAX_BUNDLE_FILES = 512
MAX_BUNDLE_COMPONENTS = 16
MAX_COMPONENT_UTF8_BYTES = 255
MAX_RELATIVE_PATH_UTF8_BYTES = 4_095
MAX_BUNDLE_CONTENT_BYTES = 16_777_216

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

RUNTIME_CONTRACT: dict[str, object] = {
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
        "format": BUNDLE_FORMAT,
        "path_encoding": "utf-8-strict",
        "path_style": "posix-relative",
        "unicode_normalization": "none",
        "max_files": MAX_BUNDLE_FILES,
        "max_components": MAX_BUNDLE_COMPONENTS,
        "max_component_utf8_bytes": MAX_COMPONENT_UTF8_BYTES,
        "max_relative_path_utf8_bytes": MAX_RELATIVE_PATH_UTF8_BYTES,
        "max_cumulative_content_bytes": MAX_BUNDLE_CONTENT_BYTES,
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

SCHEMA_EXPORTS = {
    "qualified-v1.schema.json": "qualified-deployment-profile-v1.schema.json",
    "qualification-candidate-v1.schema.json": (
        "qualification-candidate-profile-v1.schema.json"
    ),
    "qualification-override-v1.schema.json": "qualification-override-v1.schema.json",
}


class ExportError(Exception):
    """A source or output cannot satisfy the immutable export contract."""


@dataclass(frozen=True, slots=True)
class BundleFile:
    path: str
    content: bytes


@dataclass(frozen=True, slots=True)
class BundleLeaf:
    path: str
    size: int
    sha256: bytes


@dataclass(frozen=True, slots=True)
class ExportResult:
    runtime_contract_sha256: str
    agent_bundle_sha256: str
    archive_sha256: str


def _invalid() -> ExportError:
    return ExportError("agent_bundle_invalid")


def _canonical_json_bytes(value: object) -> bytes:
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


def runtime_contract_bytes() -> bytes:
    """Return the exact canonical runtime-contract v1 bytes."""

    encoded = _canonical_json_bytes(RUNTIME_CONTRACT)
    if len(encoded) > MAX_CONTRACT_BYTES:
        raise ExportError("runtime_contract_invalid")
    return encoded


def validate_relative_path(parts: Sequence[str]) -> bytes:
    """Validate and encode one strict relative POSIX path without normalization."""

    if not parts or len(parts) > MAX_BUNDLE_COMPONENTS:
        raise _invalid()
    encoded: list[bytes] = []
    for part in parts:
        if not isinstance(part, str) or part in {"", ".", ".."} or "/" in part:
            raise _invalid()
        try:
            raw = part.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise _invalid() from None
        if not raw or len(raw) > MAX_COMPONENT_UTF8_BYTES:
            raise _invalid()
        encoded.append(raw)
    relative = b"/".join(encoded)
    if len(relative) > MAX_RELATIVE_PATH_UTF8_BYTES:
        raise _invalid()
    return relative


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
    )


def _snapshot_directory(path: Path) -> dict[str, tuple[int, int, int, int, int, int]]:
    snapshot: dict[str, tuple[int, int, int, int, int, int]] = {}
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                name = entry.name
                if not isinstance(name, str) or name in {"", ".", ".."} or name in snapshot:
                    raise _invalid()
                try:
                    name.encode("utf-8", errors="strict")
                    observed = os.lstat(path / name)
                except (OSError, UnicodeError):
                    raise _invalid() from None
                snapshot[name] = _stable_identity(observed)
    except ExportError:
        raise
    except OSError:
        raise _invalid() from None
    return snapshot


def _open_flags() -> int:
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        value = getattr(os, name, 0)
        if isinstance(value, int):
            flags |= value
    return flags


def _read_stable_file(
    path: Path,
    expected: tuple[int, int, int, int, int, int],
    maximum: int,
) -> bytes:
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            _stable_identity(before) != expected
            or not stat.S_ISREG(before.st_mode)
            or before.st_size > maximum
        ):
            raise _invalid()
        descriptor = os.open(path, _open_flags())
        opened = os.fstat(descriptor)
        if (
            _path_descriptor_identity(opened) != _path_descriptor_identity(before)
            or not stat.S_ISREG(opened.st_mode)
        ):
            raise _invalid()
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(content) > maximum
            or len(content) != opened.st_size
            or _stable_identity(after) != _stable_identity(opened)
            or _stable_identity(os.lstat(path)) != _stable_identity(before)
        ):
            raise _invalid()
        return content
    except ExportError:
        raise
    except (OSError, ValueError):
        raise _invalid() from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _walk_source_tree(
    directory: Path,
    parts: tuple[str, ...],
    files: list[BundleFile],
    total: list[int],
) -> None:
    try:
        directory_before = os.lstat(directory)
    except OSError:
        raise _invalid() from None
    if not stat.S_ISDIR(directory_before.st_mode):
        raise _invalid()
    snapshot = _snapshot_directory(directory)
    ordered_names: list[tuple[bytes, str]] = []
    for name in snapshot:
        try:
            ordered_names.append((name.encode("utf-8", errors="strict"), name))
        except UnicodeEncodeError:
            raise _invalid() from None

    for _raw_name, name in sorted(ordered_names):
        child = directory / name
        try:
            observed = os.lstat(child)
        except OSError:
            raise _invalid() from None
        if _stable_identity(observed) != snapshot[name]:
            raise _invalid()
        child_parts = (*parts, name)
        if len(child_parts) > MAX_BUNDLE_COMPONENTS:
            raise _invalid()
        if stat.S_ISDIR(observed.st_mode):
            _walk_source_tree(child, child_parts, files, total)
        elif stat.S_ISREG(observed.st_mode):
            relative = validate_relative_path(child_parts).decode("utf-8", errors="strict")
            remaining = MAX_BUNDLE_CONTENT_BYTES - total[0]
            content = _read_stable_file(child, snapshot[name], remaining)
            total[0] += len(content)
            files.append(BundleFile(relative, content))
            if len(files) > MAX_BUNDLE_FILES:
                raise _invalid()
        else:
            raise _invalid()

    try:
        directory_after = os.lstat(directory)
    except OSError:
        raise _invalid() from None
    if (
        _stable_identity(directory_after) != _stable_identity(directory_before)
        or _snapshot_directory(directory) != snapshot
    ):
        raise _invalid()


def snapshot_agent_tree(source: Path) -> tuple[BundleFile, ...]:
    """Read one stable, bounded snapshot of all regular agent source files."""

    files: list[BundleFile] = []
    _walk_source_tree(Path(source), (), files, [0])
    ordered = sorted(files, key=lambda file: file.path.encode("utf-8"))
    if len({file.path.encode("utf-8") for file in ordered}) != len(ordered):
        raise _invalid()
    return tuple(ordered)


def _ordered_leaves(leaves: Iterable[BundleLeaf]) -> list[tuple[bytes, BundleLeaf]]:
    materialized = list(leaves)
    if len(materialized) > MAX_BUNDLE_FILES:
        raise _invalid()
    ordered: list[tuple[bytes, BundleLeaf]] = []
    total = 0
    for leaf in materialized:
        if (
            not isinstance(leaf, BundleLeaf)
            or type(leaf.size) is not int
            or leaf.size < 0
            or not isinstance(leaf.sha256, bytes)
            or len(leaf.sha256) != 32
        ):
            raise _invalid()
        raw = validate_relative_path(tuple(leaf.path.split("/")))
        total += leaf.size
        if total > MAX_BUNDLE_CONTENT_BYTES:
            raise _invalid()
        ordered.append((raw, leaf))
    ordered.sort(key=lambda item: item[0])
    if any(
        current[0] == previous[0]
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise _invalid()
    return ordered


def bundle_digest_from_leaves(leaves: Iterable[BundleLeaf]) -> str:
    """Calculate the independently implemented domain-framed bundle digest."""

    ordered = _ordered_leaves(leaves)
    digest = hashlib.sha256()
    digest.update(BUNDLE_DOMAIN)
    digest.update(len(ordered).to_bytes(4, "big"))
    for raw_path, leaf in ordered:
        digest.update(len(raw_path).to_bytes(4, "big"))
        digest.update(raw_path)
        digest.update(leaf.size.to_bytes(8, "big"))
        digest.update(leaf.sha256)
    return digest.hexdigest()


def _ordered_files(files: Iterable[BundleFile]) -> list[BundleFile]:
    materialized = list(files)
    leaves: list[BundleLeaf] = []
    for file in materialized:
        if not isinstance(file, BundleFile) or not isinstance(file.content, bytes):
            raise _invalid()
        leaves.append(
            BundleLeaf(file.path, len(file.content), hashlib.sha256(file.content).digest())
        )
    ordered_leaves = _ordered_leaves(leaves)
    by_path = {file.path.encode("utf-8"): file for file in materialized}
    if len(by_path) != len(materialized):
        raise _invalid()
    return [by_path[raw] for raw, _leaf in ordered_leaves]


def bundle_digest(files: Iterable[BundleFile]) -> str:
    """Calculate the semantic digest for complete in-memory files."""

    ordered = _ordered_files(files)
    return bundle_digest_from_leaves(
        BundleLeaf(file.path, len(file.content), hashlib.sha256(file.content).digest())
        for file in ordered
    )


def build_agent_archive(files: Iterable[BundleFile]) -> bytes:
    """Build a deterministic PAX tar, then a deterministic gzip stream."""

    ordered = _ordered_files(files)
    raw_tar = io.BytesIO()
    with tarfile.open(
        fileobj=raw_tar,
        mode="w",
        format=tarfile.PAX_FORMAT,
        encoding="utf-8",
        errors="strict",
        pax_headers={},
    ) as archive:
        for file in ordered:
            member = tarfile.TarInfo(file.path)
            member.type = tarfile.REGTYPE
            member.size = len(file.content)
            member.mode = 0o644
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.pax_headers = {}
            archive.addfile(member, io.BytesIO(file.content))
    return gzip.compress(raw_tar.getvalue(), compresslevel=9, mtime=0)


def _bundle_manifest_bytes(files: Iterable[BundleFile], archive: bytes) -> bytes:
    ordered = _ordered_files(files)
    value: dict[str, Any] = {
        "schema_version": 1,
        "bundle_format": BUNDLE_FORMAT,
        "bundle_sha256": bundle_digest(ordered),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "files": [
            {
                "path": file.path,
                "size": len(file.content),
                "sha256": hashlib.sha256(file.content).hexdigest(),
            }
            for file in ordered
        ],
    }
    return _canonical_json_bytes(value)


def _read_schema_authority(path: Path) -> bytes:
    try:
        observed = os.lstat(path)
    except OSError:
        raise ExportError("schema_authority_invalid") from None
    if not stat.S_ISREG(observed.st_mode) or observed.st_size > MAX_CONTRACT_BYTES:
        raise ExportError("schema_authority_invalid")
    try:
        return _read_stable_file(path, _stable_identity(observed), MAX_CONTRACT_BYTES)
    except ExportError:
        raise ExportError("schema_authority_invalid") from None


def export_all(repo_root: Path, output_dir: Path) -> ExportResult:
    """Export all six pre-image artifacts from checked-in authorities."""

    repository = Path(repo_root)
    destination = Path(output_dir)
    files = snapshot_agent_tree(repository / "agents" / "agent-a")
    contract = runtime_contract_bytes()
    archive = build_agent_archive(files)
    manifest = _bundle_manifest_bytes(files, archive)
    schemas = {
        output_name: _read_schema_authority(
            repository / "deployment-profiles" / source_name
        )
        for source_name, output_name in SCHEMA_EXPORTS.items()
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "runtime-contract.json").write_bytes(contract)
    for output_name, content in schemas.items():
        (destination / output_name).write_bytes(content)
    (destination / "agent-a-bundle.tar.gz").write_bytes(archive)
    (destination / "agent-a-bundle-v1.manifest.json").write_bytes(manifest)
    return ExportResult(
        runtime_contract_sha256=hashlib.sha256(contract).hexdigest(),
        agent_bundle_sha256=bundle_digest(files),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir or args.repo_root / "dist"
    try:
        export_all(args.repo_root, output_dir)
    except ExportError as error:
        raise SystemExit(str(error)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
