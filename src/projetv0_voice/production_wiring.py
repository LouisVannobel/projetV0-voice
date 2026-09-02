"""Closed concrete production consumers for runtime artifacts and providers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import threading
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from openai import DefaultAsyncHttpxClient
from pydantic import SecretStr

from projetv0_voice.config import AgentManifestV1, load_agent_manifest
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.inference.openrouter_tts import OpenRouterTTSService
from projetv0_voice.inference.services import build_llm, build_stt
from projetv0_voice.lifecycle import (
    RuntimeInferenceFactories,
    RuntimeProductionFactories,
    RuntimeProfileSelection,
)
from projetv0_voice.persistence.postgres_sink import PsycopgOperationSink
from projetv0_voice.qualified_profile import (
    QualificationCandidateProfileV1,
    QualifiedDeploymentProfileV1,
    RuntimeDeploymentProfileV1,
    canonical_qualified_profile_sha256,
    load_qualification_candidate_profile,
    load_qualification_override,
    load_qualified_deployment_profile,
)
from projetv0_voice.runtime_config import (
    RuntimeSettingsV1,
    read_runtime_secret,
)
from projetv0_voice.session import PublicSttHttpClient
from projetv0_voice.telnyx.call_control import CallControlClient

_KEYRING_PATH = PurePosixPath("/run/secrets/aead_keyring_v1.json")
_KEY_HEX = re.compile(r"^[0-9a-f]{64}$")
_MAX_KEY_VERSION = (1 << 63) - 1
_MAX_RUNTIME_CONTRACT_BYTES = 1_048_576
_MAX_BUNDLE_FILES = 512
_MAX_BUNDLE_COMPONENTS = 16
_MAX_COMPONENT_BYTES = 255
_MAX_RELATIVE_PATH_BYTES = 4_095
_MAX_BUNDLE_CONTENT_BYTES = 16_777_216
_BUNDLE_DOMAIN = b"projetv0-agent-bundle-v1\x00"
_SPECIAL_BITS = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX

type _BundleEntry = tuple[bytes, int, bytes]


class _WiringInvalid(Exception):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _WiringInvalid
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise _WiringInvalid


def _exact_version(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_KEY_VERSION:
        raise _WiringInvalid
    return value


def _decode_keyring_value(raw: str) -> CryptoKeyring:
    parsed = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict) or set(parsed) != {
        "schema_version",
        "active_version",
        "keys",
    }:
        raise _WiringInvalid
    if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
        raise _WiringInvalid
    active_version = _exact_version(parsed["active_version"])
    items = parsed["keys"]
    if not isinstance(items, list) or not items:
        raise _WiringInvalid
    keys: dict[int, bytes] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "version",
            "aes256_key_hex",
        }:
            raise _WiringInvalid
        version = _exact_version(item["version"])
        encoded = item["aes256_key_hex"]
        if (
            version in keys
            or not isinstance(encoded, str)
            or _KEY_HEX.fullmatch(encoded) is None
        ):
            raise _WiringInvalid
        keys[version] = bytes.fromhex(encoded)
    if active_version not in keys:
        raise _WiringInvalid
    return CryptoKeyring(keys, active_version=active_version)


def _decode_keyring_secret(secret: SecretStr) -> CryptoKeyring:
    """Reduce all raw keyring failures to one traceback-safe public code."""

    result: CryptoKeyring | None = None
    invalid = False
    raw: str | None = None
    try:
        if not isinstance(secret, SecretStr):
            raise _WiringInvalid
        raw = secret.get_secret_value()
        result = _decode_keyring_value(raw)
    except BaseException:
        invalid = True
    finally:
        raw = None
        del secret
    if invalid or result is None:
        raise RuntimeError("runtime_keyring_invalid") from None
    return result


def _safe_mode(value: os.stat_result, *, directory: bool) -> bool:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    return (
        expected_type(value.st_mode)
        and value.st_uid == 0
        and value.st_mode & _SPECIAL_BITS == 0
        and value.st_mode & 0o022 == 0
    )


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_uid,
        value.st_gid,
    )


def _linux_flags(*, directory: bool) -> int:
    required = (
        getattr(os, "O_CLOEXEC", None),
        getattr(os, "O_NOFOLLOW", None),
        getattr(os, "O_NONBLOCK", None),
        getattr(os, "O_DIRECTORY", None),
    )
    if any(type(value) is not int for value in required):
        raise _WiringInvalid
    close_on_exec, no_follow, nonblocking, directory_flag = cast(
        tuple[int, int, int, int], required
    )
    flags = os.O_RDONLY | close_on_exec | no_follow | nonblocking
    if directory:
        flags |= directory_flag
    return flags


def _open_parent(path: PurePosixPath) -> tuple[list[int], int, str]:
    if not path.is_absolute() or path == PurePosixPath("/"):
        raise _WiringInvalid
    components = path.parts[1:]
    if not components or any(part in {"", ".", ".."} for part in components):
        raise _WiringInvalid
    descriptors: list[int] = []
    try:
        current = os.open("/", _linux_flags(directory=True))
        descriptors.append(current)
        if not _safe_mode(os.fstat(current), directory=True):
            raise _WiringInvalid
        for component in components[:-1]:
            child = os.open(
                component,
                _linux_flags(directory=True),
                dir_fd=current,
            )
            descriptors.append(child)
            current = child
            if not _safe_mode(os.fstat(current), directory=True):
                raise _WiringInvalid
        return descriptors, current, components[-1]
    except BaseException:
        _close_all(descriptors)
        raise


def _close_all(descriptors: list[int]) -> None:
    while descriptors:
        descriptor = descriptors.pop()
        with suppress(OSError):
            os.close(descriptor)


def _read_fd(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 65_536))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    if len(content) > maximum:
        raise _WiringInvalid
    return content


def _read_stable_file(
    parent: int,
    name: str,
    *,
    maximum: int,
) -> bytes:
    descriptor = os.open(
        name,
        _linux_flags(directory=False),
        dir_fd=parent,
    )
    try:
        before = os.fstat(descriptor)
        if not _safe_mode(before, directory=False) or before.st_size > maximum:
            raise _WiringInvalid
        content = _read_fd(descriptor, maximum)
        after = os.fstat(descriptor)
        if _stable_stat(before) != _stable_stat(after) or len(content) != before.st_size:
            raise _WiringInvalid
        return content
    finally:
        os.close(descriptor)


def _validate_runtime_contract(settings: RuntimeSettingsV1) -> None:
    descriptors: list[int] = []
    try:
        descriptors, parent, name = _open_parent(settings.runtime_contract_path)
        content = _read_stable_file(
            parent,
            name,
            maximum=_MAX_RUNTIME_CONTRACT_BYTES,
        )
        observed = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(observed, settings.runtime_contract_sha256):
            raise _WiringInvalid
    finally:
        _close_all(descriptors)


def _snapshot_directory(descriptor: int) -> dict[str, tuple[int, ...]]:
    names = os.listdir(descriptor)
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise _WiringInvalid
    snapshot: dict[str, tuple[int, ...]] = {}
    for name in names:
        if name in {"", ".", ".."}:
            raise _WiringInvalid
        snapshot[name] = _directory_stat(
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        )
    return snapshot


def _relative_bytes(parts: tuple[str, ...]) -> bytes:
    if not parts or len(parts) > _MAX_BUNDLE_COMPONENTS:
        raise _WiringInvalid
    encoded: list[bytes] = []
    for part in parts:
        if part in {"", ".", ".."}:
            raise _WiringInvalid
        raw = part.encode("utf-8", errors="strict")
        if not raw or len(raw) > _MAX_COMPONENT_BYTES:
            raise _WiringInvalid
        encoded.append(raw)
    relative = b"/".join(encoded)
    if len(relative) > _MAX_RELATIVE_PATH_BYTES:
        raise _WiringInvalid
    return relative


def _walk_bundle(
    descriptor: int,
    *,
    parts: tuple[str, ...],
    entries: list[_BundleEntry],
    total: list[int],
) -> None:
    directory_before = os.fstat(descriptor)
    if not _safe_mode(directory_before, directory=True):
        raise _WiringInvalid
    snapshot = _snapshot_directory(descriptor)
    names: list[tuple[bytes, str]] = []
    for name in snapshot:
        try:
            names.append((name.encode("utf-8", errors="strict"), name))
        except UnicodeEncodeError:
            raise _WiringInvalid from None
    for _encoded_name, name in sorted(names):
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if _directory_stat(observed) != snapshot[name]:
            raise _WiringInvalid
        child_parts = (*parts, name)
        if stat.S_ISDIR(observed.st_mode):
            if len(child_parts) > _MAX_BUNDLE_COMPONENTS:
                raise _WiringInvalid
            child = os.open(
                name,
                _linux_flags(directory=True),
                dir_fd=descriptor,
            )
            try:
                _walk_bundle(
                    child,
                    parts=child_parts,
                    entries=entries,
                    total=total,
                )
            finally:
                os.close(child)
        elif stat.S_ISREG(observed.st_mode):
            relative = _relative_bytes(child_parts)
            remaining = _MAX_BUNDLE_CONTENT_BYTES - total[0]
            content = _read_stable_file(
                descriptor,
                name,
                maximum=remaining,
            )
            total[0] += len(content)
            entries.append((relative, len(content), hashlib.sha256(content).digest()))
            if len(entries) > _MAX_BUNDLE_FILES:
                raise _WiringInvalid
        else:
            raise _WiringInvalid
    after_snapshot = _snapshot_directory(descriptor)
    directory_after = os.fstat(descriptor)
    if snapshot != after_snapshot or _directory_stat(directory_before) != _directory_stat(
        directory_after
    ):
        raise _WiringInvalid


def _bundle_digest(entries: Sequence[_BundleEntry]) -> str:
    digest = hashlib.sha256()
    digest.update(_BUNDLE_DOMAIN)
    digest.update(len(entries).to_bytes(4, "big"))
    for path, size, file_digest in entries:
        if (
            not isinstance(path, bytes)
            or not isinstance(size, int)
            or not isinstance(file_digest, bytes)
            or len(file_digest) != 32
        ):
            raise ValueError("bundle_entry_invalid") from None
        digest.update(len(path).to_bytes(4, "big"))
        digest.update(path)
        digest.update(size.to_bytes(8, "big"))
        digest.update(file_digest)
    return digest.hexdigest()


def _validate_agent_bundle(settings: RuntimeSettingsV1) -> None:
    descriptors: list[int] = []
    try:
        descriptors, parent, name = _open_parent(settings.agent_bundle_path)
        root = os.open(
            name,
            _linux_flags(directory=True),
            dir_fd=parent,
        )
        descriptors.append(root)
        entries: list[_BundleEntry] = []
        _walk_bundle(root, parts=(), entries=entries, total=[0])
        entries.sort(key=lambda item: item[0])
        if not hmac.compare_digest(
            _bundle_digest(entries),
            settings.agent_bundle_sha256,
        ):
            raise _WiringInvalid
    finally:
        _close_all(descriptors)


def _validate_artifacts(settings: RuntimeSettingsV1) -> None:
    failed = False
    try:
        _validate_runtime_contract(settings)
        _validate_agent_bundle(settings)
    except BaseException:
        failed = True
    if failed:
        raise RuntimeError("runtime_artifacts_invalid") from None


def _root_owned_regular(value: os.stat_result) -> bool:
    return _safe_mode(value, directory=False)


def _load_profile(
    settings: RuntimeSettingsV1,
    manifest: AgentManifestV1,
    now: datetime,
) -> RuntimeProfileSelection:
    if settings.runtime_mode == "qualification_candidate":
        if (
            settings.qualification_candidate_path is None
            or settings.qualification_run_id is None
            or settings.benchmark_did_sha256 is None
        ):
            raise RuntimeError("runtime_profile_invalid") from None
        candidate = load_qualification_candidate_profile(
            Path(str(settings.qualification_candidate_path)),
            qualification_mode=True,
            expected_run_id=settings.qualification_run_id,
            expected_benchmark_did_hash=settings.benchmark_did_sha256,
            expected_deployment_id=settings.deployment_id,
            expected_runtime_contract_sha256=settings.runtime_contract_sha256,
            expected_image_digest=settings.image_digest,
            expected_agent_bundle_sha256=settings.agent_bundle_sha256,
            expected_inference_profile_sha256=settings.inference_profile_sha256,
            manifest=manifest,
            now=now,
            ownership_check=_root_owned_regular,
        )
        return RuntimeProfileSelection(candidate, None)
    if settings.qualified_profile_path is None:
        raise RuntimeError("runtime_profile_invalid") from None
    profile = load_qualified_deployment_profile(
        Path(str(settings.qualified_profile_path)),
        expected_deployment_id=settings.deployment_id,
        expected_runtime_contract_sha256=settings.runtime_contract_sha256,
        expected_image_digest=settings.image_digest,
        expected_agent_bundle_sha256=settings.agent_bundle_sha256,
        expected_inference_profile_sha256=settings.inference_profile_sha256,
        ownership_check=_root_owned_regular,
    )
    override = None
    if settings.runtime_mode == "qualification_override":
        if settings.qualification_override_path is None or settings.qualification_run_id is None:
            raise RuntimeError("runtime_profile_invalid") from None
        override = load_qualification_override(
            Path(str(settings.qualification_override_path)),
            qualification_mode=True,
            expected_run_id=settings.qualification_run_id,
            expected_deployment_id=settings.deployment_id,
            expected_qualified_profile_sha256=canonical_qualified_profile_sha256(
                profile
            ),
            strict_qualified_at=profile.qualified_at,
            ownership_check=_root_owned_regular,
            now=now,
        )
    return RuntimeProfileSelection(profile, override)


def _inference_factories(
    api_key: SecretStr,
    deployment_profile: RuntimeDeploymentProfileV1,
    language: str,
) -> RuntimeInferenceFactories:
    if (
        not isinstance(api_key, SecretStr)
        or not isinstance(
            deployment_profile,
            QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1,
        )
        or not isinstance(language, str)
        or not language
    ):
        raise ValueError("runtime_inference_factory_invalid") from None
    profile = deployment_profile.inference

    def stt_http_client_factory() -> PublicSttHttpClient:
        return cast(PublicSttHttpClient, DefaultAsyncHttpxClient())

    def stt_factory(client: PublicSttHttpClient) -> Any:
        return build_stt(
            profile,
            api_key,
            language=language,
            http_client=cast(Any, client),
        )

    return RuntimeInferenceFactories(
        stt_http_client_factory=stt_http_client_factory,
        stt_factory=stt_factory,
        llm_factory=lambda: build_llm(profile, api_key),
        tts_factory=lambda: OpenRouterTTSService(profile=profile, api_key=api_key),
    )


def build_production_factories(
    settings: RuntimeSettingsV1,
) -> RuntimeProductionFactories:
    """Return the one concrete, construction-inert production factory set."""

    if type(settings) is not RuntimeSettingsV1 or settings.keyring_path != _KEYRING_PATH:
        raise ValueError("production_wiring_config_invalid") from None
    artifact_validated = False
    manifest_value: AgentManifestV1 | None = None
    keyring_value: CryptoKeyring | None = None
    keyring_attempted = False
    keyring_failed = False
    lock = threading.Lock()

    def validate_artifacts(received: RuntimeSettingsV1) -> None:
        nonlocal artifact_validated
        if received is not settings:
            raise RuntimeError("runtime_artifacts_invalid") from None
        _validate_artifacts(received)
        artifact_validated = True

    def load_manifest(received: RuntimeSettingsV1) -> AgentManifestV1:
        nonlocal manifest_value
        if received is not settings or not artifact_validated:
            raise RuntimeError("runtime_artifacts_invalid") from None
        if manifest_value is None:
            manifest_value = load_agent_manifest(
                Path(str(received.agent_bundle_path)),
                host_max_concurrent_calls=received.deployment_max_calls,
            )
        return manifest_value

    def load_keyring(received: RuntimeSettingsV1) -> CryptoKeyring:
        nonlocal keyring_attempted, keyring_failed, keyring_value
        if received != settings:
            raise RuntimeError("runtime_keyring_invalid") from None
        with lock:
            if not keyring_attempted:
                keyring_attempted = True
                try:
                    keyring_value = _decode_keyring_secret(
                        read_runtime_secret(received.keyring_path)
                    )
                except BaseException:
                    keyring_failed = True
            if keyring_failed or keyring_value is None:
                raise RuntimeError("runtime_keyring_invalid") from None
            return keyring_value

    return RuntimeProductionFactories(
        validate_artifacts=validate_artifacts,
        load_manifest=load_manifest,
        load_profile=_load_profile,
        read_secret=read_runtime_secret,
        load_keyring=load_keyring,
        sink_factory=lambda dsn: PsycopgOperationSink(dsn.get_secret_value()),
        call_control_factory=lambda key: CallControlClient(
            api_key=key.get_secret_value()
        ),
        inference_factory=_inference_factories,
    )


__all__ = ["build_production_factories"]
