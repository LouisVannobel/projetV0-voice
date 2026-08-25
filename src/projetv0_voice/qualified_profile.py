"""Qualified deployment and bounded qualification profile contracts."""

from __future__ import annotations

import hashlib
import json
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
TokenLocatorId = Literal[
    "telnyx-http-header-v1",
    "telnyx-query-v1",
    "telnyx-start-v1",
]


def _require_exact_int(value: object) -> object:
    if type(value) is not int:
        raise ValueError("value must be an exact integer")
    return value


SchemaVersionV1 = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
PositiveInt = Annotated[int, BeforeValidator(_require_exact_int), Field(gt=0)]
MonoChannel = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
CandidateTimeout = Annotated[Literal[10000], BeforeValidator(_require_exact_int)]
CandidateLeaseTtl = Annotated[Literal[30], BeforeValidator(_require_exact_int)]
CandidateMaxCalls = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
OverrideMaxCalls = Annotated[Literal[15, 20], BeforeValidator(_require_exact_int)]


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


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
    llm_provider_policy: Mapping[str, JsonValue]
    tts_provider_options: Mapping[str, JsonValue]

    @field_validator("llm_provider_policy", "tts_provider_options", mode="after")
    @classmethod
    def freeze_provider_maps(
        cls, value: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]:
        return cast(Mapping[str, JsonValue], _freeze_json(value))

    @field_serializer("llm_provider_policy", "tts_provider_options")
    def serialize_provider_maps(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], _thaw_json(value))


class QualifiedDeploymentProfileV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    deployment_id: str = Field(min_length=1)
    runtime_contract_sha256: Sha256
    image_digest: ImageDigest
    agent_bundle_sha256: Sha256
    inference_profile_sha256: Sha256
    inference: InferenceProfileV1
    token_locator_id: TokenLocatorId
    telnyx_handshake_fixture_sha256: Sha256
    disclosure_mark_timeout_ms: Annotated[
        int, BeforeValidator(_require_exact_int), Field(ge=3000, le=10000)
    ]
    call_lease_ttl_seconds: PositiveInt
    qualified_at: datetime

    _normalize_qualified_at = field_validator("qualified_at", mode="after")(_utc_datetime)


class QualificationCandidateProfileV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    run_id: UUID
    deployment_id: str = Field(min_length=1)
    expires_at: datetime
    benchmark_did_hash: Sha256
    runtime_contract_sha256: Sha256
    image_digest: ImageDigest
    agent_bundle_sha256: Sha256
    inference_profile_sha256: Sha256
    inference: InferenceProfileV1
    token_locator_id: TokenLocatorId
    telnyx_handshake_fixture_sha256: Sha256
    disclosure_mark_timeout_ms: CandidateTimeout
    call_lease_ttl_seconds: CandidateLeaseTtl
    max_concurrent_calls: CandidateMaxCalls

    _normalize_expires_at = field_validator("expires_at", mode="after")(_utc_datetime)


class QualificationOverrideV1(_StrictFrozenProfile):
    schema_version: SchemaVersionV1
    run_id: UUID
    benchmark_max_calls: OverrideMaxCalls
    created_at: datetime
    expires_at: datetime

    _normalize_datetimes = field_validator("created_at", "expires_at", mode="after")(
        _utc_datetime
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("override expires_at must follow created_at")
        return self


def canonical_inference_profile_sha256(profile: InferenceProfileV1) -> str:
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


def _read_json_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    if not path.is_file():
        raise ValueError(f"profile file does not exist: {path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


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
) -> QualifiedDeploymentProfileV1:
    profile = _read_json_model(path, QualifiedDeploymentProfileV1)
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
) -> QualificationCandidateProfileV1:
    if not qualification_mode:
        raise ValueError("candidate profiles require explicit qualification mode")
    profile = _read_json_model(path, QualificationCandidateProfileV1)
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
    ownership_check: Callable[[Path], bool],
    now: datetime,
) -> QualificationOverrideV1:
    if not qualification_mode:
        raise ValueError("qualification overrides require explicit qualification mode")
    if not ownership_check(path):
        raise ValueError("qualification override must be root-owned")
    override = _read_json_model(path, QualificationOverrideV1)
    if override.run_id != expected_run_id:
        raise ValueError("qualification override run ID does not match")
    current_time = _utc_datetime(now)
    if not override.created_at <= current_time < override.expires_at:
        raise ValueError("qualification override is not active")
    return override
