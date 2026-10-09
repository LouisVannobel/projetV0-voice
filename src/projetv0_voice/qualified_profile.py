"""Qualified deployment and bounded qualification profile contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from projetv0_voice.config import AgentManifestV1

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ImageDigest = Annotated[
    str,
    Field(
        pattern=(
            r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
            r"(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*"
            r"(?::[0-9]{1,5})?"
            r"/[a-z0-9]+(?:[._-][a-z0-9]+)*"
            r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
            r"@sha256:[0-9a-f]{64}$"
        )
    ),
]
TokenLocatorId = Literal["telnyx-header-connected-v1"]
_OPENROUTER_PROVIDER_SLUG = re.compile(r"^[a-z0-9]+(?:[./_-][a-z0-9]+)*$")


def _require_exact_int(value: object) -> object:
    if type(value) is not int:
        raise ValueError("value must be an exact integer")
    return value


SchemaVersionV1 = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
PositiveInt = Annotated[int, Field(gt=0), BeforeValidator(_require_exact_int)]
MonoChannel = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
CandidateTimeout = Annotated[Literal[10000], BeforeValidator(_require_exact_int)]
CandidateLeaseTtl = Annotated[Literal[30], BeforeValidator(_require_exact_int)]
CandidateMaxCalls = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
CandidateTotalCalls = Annotated[int, Field(ge=1, le=10), BeforeValidator(_require_exact_int)]
OverrideMaxCalls = Annotated[Literal[15, 20], BeforeValidator(_require_exact_int)]
StrictLeaseTtl = Annotated[
    int, Field(ge=5, le=300), BeforeValidator(_require_exact_int)
]


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _validate_datetime_input(value: object) -> object:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("datetime input must be an ISO-8601 string") from error
        return value
    raise ValueError("datetime input must be a datetime object or ISO-8601 string")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class _StrictFrozenProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class InferenceProfileV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    stt_model: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    tts_model: str = Field(min_length=1)
    tts_voice: str = Field(min_length=1)
    tts_pcm_sample_rate: PositiveInt
    tts_pcm_channels: MonoChannel
    tts_speed: float | None = Field(
        default=None, ge=0.5, le=2.0, strict=True, exclude_if=lambda value: value is None
    )
    llm_provider_policy: Mapping[str, JsonValue] = Field(
        json_schema_extra={
            "properties": {"allow_fallbacks": {"type": "boolean"}},
            "allOf": [
                {"not": {"required": ["fallbacks"]}},
                {"not": {"required": ["provider"]}},
            ],
        }
    )
    tts_provider_options: Mapping[str, Mapping[str, JsonValue]] = Field(
        json_schema_extra={
            "propertyNames": {
                "allOf": [
                    {"pattern": r"^[a-z0-9]+(?:[./_-][a-z0-9]+)*$"},
                    {"not": {"enum": ["provider", "options"]}},
                ]
            }
        }
    )

    @field_validator("llm_provider_policy", mode="after")
    @classmethod
    def validate_and_freeze_llm_provider_policy(
        cls, value: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]:
        if "fallbacks" in value:
            raise ValueError("llm_provider_policy uses allow_fallbacks, not fallbacks")
        if "provider" in value:
            raise ValueError("llm_provider_policy must not contain a provider wrapper")
        if "allow_fallbacks" in value and type(value["allow_fallbacks"]) is not bool:
            raise ValueError("llm_provider_policy allow_fallbacks must be an exact boolean")
        return cast(Mapping[str, JsonValue], _freeze_json(value))

    @field_validator("tts_provider_options", mode="after")
    @classmethod
    def validate_and_freeze_tts_provider_options(
        cls, value: Mapping[str, Mapping[str, JsonValue]]
    ) -> Mapping[str, Mapping[str, JsonValue]]:
        for provider_slug in value:
            if _OPENROUTER_PROVIDER_SLUG.fullmatch(provider_slug) is None:
                raise ValueError("tts_provider_options contains an invalid provider slug")
            if provider_slug in {"provider", "options"}:
                raise ValueError("tts_provider_options must not contain an outer wrapper")
        return cast(Mapping[str, Mapping[str, JsonValue]], _freeze_json(value))

    @field_serializer("llm_provider_policy")
    def serialize_llm_provider_policy(
        self, value: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], _thaw_json(value))

    @field_serializer("tts_provider_options")
    def serialize_tts_provider_options(
        self, value: Mapping[str, Mapping[str, JsonValue]]
    ) -> dict[str, dict[str, JsonValue]]:
        return cast(dict[str, dict[str, JsonValue]], _thaw_json(value))


class QualifiedDeploymentProfileV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    deployment_id: str = Field(min_length=1)
    runtime_contract_sha256: Sha256
    image_digest: ImageDigest
    agent_bundle_sha256: Sha256
    inference_profile_sha256: Sha256
    inference: InferenceProfileV1
    token_locator_id: TokenLocatorId
    telnyx_api_key_sha256: Sha256
    telnyx_data_locality: Literal["EU"] = Field(
        description=(
            "Operator-attested EU Voice API and media routing. Does not attest "
            "Telnyx CDR/MDR storage location or external inference residency."
        )
    )
    telnyx_handshake_fixture_sha256: Sha256
    disclosure_mark_timeout_ms: Annotated[
        int, Field(ge=3000, le=10000), BeforeValidator(_require_exact_int)
    ]
    call_lease_ttl_seconds: StrictLeaseTtl
    qualified_at: datetime

    _validate_qualified_at_input = field_validator("qualified_at", mode="before")(
        _validate_datetime_input
    )
    _normalize_qualified_at = field_validator("qualified_at", mode="after")(_utc_datetime)


class QualificationCandidateProfileV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    run_id: UUID
    deployment_id: str = Field(min_length=1)
    expires_at: datetime
    admission_not_before: datetime | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    benchmark_did_hash: Sha256
    runtime_contract_sha256: Sha256
    image_digest: ImageDigest
    agent_bundle_sha256: Sha256
    inference_profile_sha256: Sha256
    inference: InferenceProfileV1
    token_locator_id: TokenLocatorId
    telnyx_api_key_sha256: Sha256
    telnyx_handshake_fixture_sha256: Sha256
    disclosure_mark_timeout_ms: CandidateTimeout
    call_lease_ttl_seconds: CandidateLeaseTtl
    max_concurrent_calls: CandidateMaxCalls
    total_calls: CandidateTotalCalls = Field(default=1, exclude_if=lambda value: value == 1)

    _validate_expires_at_input = field_validator("expires_at", mode="before")(
        _validate_datetime_input
    )
    _normalize_expires_at = field_validator("expires_at", mode="after")(_utc_datetime)
    _validate_admission_not_before_input = field_validator(
        "admission_not_before", mode="before"
    )(lambda value: None if value is None else _validate_datetime_input(value))
    _normalize_admission_not_before = field_validator("admission_not_before", mode="after")(
        lambda value: None if value is None else _utc_datetime(value)
    )

    @model_validator(mode="after")
    def validate_admission_window(self) -> Self:
        if self.admission_not_before is not None and self.admission_not_before >= self.expires_at:
            raise ValueError("admission_not_before must precede expires_at")
        return self


class QualificationOverrideV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    run_id: UUID
    deployment_id: str = Field(min_length=1)
    qualified_profile_sha256: Sha256
    benchmark_max_calls: OverrideMaxCalls
    created_at: datetime
    expires_at: datetime

    _validate_datetime_inputs = field_validator("created_at", "expires_at", mode="before")(
        _validate_datetime_input
    )
    _normalize_datetimes = field_validator("created_at", "expires_at", mode="after")(
        _utc_datetime
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("override expires_at must follow created_at")
        return self


RuntimeDeploymentProfileV1 = QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1


def canonical_inference_profile_sha256(profile: InferenceProfileV1) -> str:
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_qualified_profile_sha256(profile: QualifiedDeploymentProfileV1) -> str:
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_candidate_profile_sha256(profile: QualificationCandidateProfileV1) -> str:
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_model_schema_json(model: type[BaseModel]) -> str:
    """Serialize a generated schema with the single committed-file convention."""

    return (
        json.dumps(
            model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


_MAX_PROFILE_BYTES = 65_536


def _is_symlink_or_junction(path: Path) -> bool:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return False
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(file_stat, "st_file_attributes", 0)
    return path.is_symlink() or path.is_junction() or bool(file_attributes & reparse_attribute)


def _has_reparse_component(path: Path) -> bool:
    absolute_path = path.absolute()
    return any(
        _is_symlink_or_junction(component)
        for component in (absolute_path, *absolute_path.parents)
    )


def _read_profile_atomic[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    ownership_check: Callable[[os.stat_result], bool],
) -> ModelT:
    if _has_reparse_component(path):
        raise ValueError("profile artifact path must not contain a symlink or junction")
    try:
        before_open = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("profile artifact file does not exist") from error

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("profile artifact could not be opened safely") from error
    try:
        opened_stat = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            raise ValueError("profile artifact changed before atomic open")
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError("profile artifact must be a regular file")
        if not ownership_check(opened_stat):
            raise ValueError("profile artifact must be root-owned")
        if os.name != "nt" and opened_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("profile artifact must not be group or other writable")
        chunks: list[bytes] = []
        remaining = _MAX_PROFILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > _MAX_PROFILE_BYTES:
            raise ValueError("profile artifact file is too large")
    finally:
        os.close(descriptor)
    return model.model_validate_json(content)


def _validate_bindings(
    profile: QualifiedDeploymentProfileV1 | QualificationCandidateProfileV1,
    *,
    expected_deployment_id: str,
    expected_runtime_contract_sha256: str,
    expected_image_digest: str,
    expected_agent_bundle_sha256: str,
    expected_inference_profile_sha256: str,
) -> None:
    expected = {
        "deployment_id": expected_deployment_id,
        "runtime_contract_sha256": expected_runtime_contract_sha256,
        "image_digest": expected_image_digest,
        "agent_bundle_sha256": expected_agent_bundle_sha256,
        "inference_profile_sha256": expected_inference_profile_sha256,
    }
    for field_name, expected_value in expected.items():
        if getattr(profile, field_name) != expected_value:
            raise ValueError(f"profile {field_name} does not match the injected value")
    actual_inference_hash = canonical_inference_profile_sha256(profile.inference)
    if profile.inference_profile_sha256 != actual_inference_hash:
        raise ValueError("profile inference_profile_sha256 does not match canonical inference")


def load_qualified_deployment_profile(
    path: Path,
    *,
    expected_deployment_id: str,
    expected_runtime_contract_sha256: str,
    expected_image_digest: str,
    expected_agent_bundle_sha256: str,
    expected_inference_profile_sha256: str,
    ownership_check: Callable[[os.stat_result], bool],
) -> QualifiedDeploymentProfileV1:
    profile = _read_profile_atomic(path, QualifiedDeploymentProfileV1, ownership_check)
    _validate_bindings(
        profile,
        expected_deployment_id=expected_deployment_id,
        expected_runtime_contract_sha256=expected_runtime_contract_sha256,
        expected_image_digest=expected_image_digest,
        expected_agent_bundle_sha256=expected_agent_bundle_sha256,
        expected_inference_profile_sha256=expected_inference_profile_sha256,
    )
    return profile


def load_qualification_candidate_profile(
    path: Path,
    *,
    qualification_mode: bool,
    expected_run_id: UUID,
    expected_deployment_id: str,
    expected_benchmark_did_hash: str,
    expected_runtime_contract_sha256: str,
    expected_image_digest: str,
    expected_agent_bundle_sha256: str,
    expected_inference_profile_sha256: str,
    manifest: AgentManifestV1,
    now: datetime,
    ownership_check: Callable[[os.stat_result], bool],
) -> QualificationCandidateProfileV1:
    if not qualification_mode:
        raise ValueError("candidate profiles require explicit qualification mode")
    profile = _read_profile_atomic(path, QualificationCandidateProfileV1, ownership_check)
    current_time = _utc_datetime(now)
    if profile.expires_at <= current_time:
        raise ValueError("qualification candidate has expired")
    if profile.run_id != expected_run_id:
        raise ValueError("qualification candidate run_id does not match the active run")
    if profile.benchmark_did_hash != expected_benchmark_did_hash:
        raise ValueError("qualification candidate does not match the benchmark DID")
    if manifest.recording_mode != "off":
        raise ValueError("qualification candidate requires recording off")
    if manifest.recording_required:
        raise ValueError("qualification candidate requires recording_required=false")
    if manifest.recording_play_beep:
        raise ValueError("qualification candidate requires recording_play_beep=false")
    _validate_bindings(
        profile,
        expected_deployment_id=expected_deployment_id,
        expected_runtime_contract_sha256=expected_runtime_contract_sha256,
        expected_image_digest=expected_image_digest,
        expected_agent_bundle_sha256=expected_agent_bundle_sha256,
        expected_inference_profile_sha256=expected_inference_profile_sha256,
    )
    return profile


def load_qualification_override(
    path: Path,
    *,
    qualification_mode: bool,
    expected_run_id: UUID,
    expected_deployment_id: str,
    expected_qualified_profile_sha256: str,
    strict_qualified_at: datetime,
    ownership_check: Callable[[os.stat_result], bool],
    now: datetime,
) -> QualificationOverrideV1:
    if not qualification_mode:
        raise ValueError("qualification overrides require explicit qualification mode")
    override = _read_profile_atomic(path, QualificationOverrideV1, ownership_check)
    if override.run_id != expected_run_id:
        raise ValueError("qualification override run ID does not match")
    if override.deployment_id != expected_deployment_id:
        raise ValueError("qualification override deployment_id does not match")
    if override.qualified_profile_sha256 != expected_qualified_profile_sha256:
        raise ValueError("qualification override qualified profile hash does not match")
    current_time = _utc_datetime(now)
    qualified_at = _utc_datetime(strict_qualified_at)
    if not qualified_at <= override.created_at < current_time < override.expires_at:
        raise ValueError("qualification override is not active")
    return override
