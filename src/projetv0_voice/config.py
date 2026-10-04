"""Strict, versioned configuration for one static voice-agent bundle."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

from projetv0_voice.models import _e164, _provider_id


def _require_exact_int(value: object) -> object:
    if type(value) is not int:
        raise ValueError("value must be an exact integer")
    return value


SchemaVersionV1 = Annotated[Literal[1], BeforeValidator(_require_exact_int)]
PositiveInt = Annotated[int, Field(gt=0), BeforeValidator(_require_exact_int)]
PcmuSampleRate = Annotated[Literal[8000], BeforeValidator(_require_exact_int)]


class _UniqueKeySafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    pass


def _construct_unique_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _is_symlink_or_junction(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


class SparraManifestV1(BaseModel):
    """Operator-qualified pilot routing; caller data never supplies these facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: SchemaVersionV1
    connection_id: StrictStr
    original_forward_line_e164: StrictStr | None
    qualified_transfer_destination_e164: StrictStr | None

    @field_validator("connection_id")
    @classmethod
    def connection(cls, value: str) -> str:
        return _provider_id(value, 256)

    @field_validator("original_forward_line_e164", "qualified_transfer_destination_e164")
    @classmethod
    def number(cls, value: str | None) -> str | None:
        return None if value is None else _e164(value)


class AgentManifestV1(BaseModel):
    """Immutable, non-secret policy shipped with an agent bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersionV1
    tenant_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dids: tuple[str, ...] = Field(min_length=1)
    language: str = Field(min_length=1)
    prompt_path: Path
    prompt_revision: str = Field(min_length=1)
    greeting: str = Field(min_length=1)
    conversation_mode: Literal["freeform"]
    max_concurrent_calls: PositiveInt
    direction: Literal["inbound_only"]
    transport_codec: Literal["PCMU"]
    transport_sample_rate_hz: PcmuSampleRate
    transcript_retention_days: PositiveInt
    recording_mode: Literal["off", "telnyx_dual"]
    recording_format: Literal["wav"]
    recording_retention_days: PositiveInt | None
    recording_required: StrictBool
    recording_play_beep: StrictBool
    sparra: SparraManifestV1 | None = None

    @field_validator("sparra", mode="before")
    @classmethod
    def present_sparra(cls, value: object) -> object:
        if value is None:
            raise ValueError("present sparra extension must not be null")
        return value

    @model_serializer(mode="wrap")
    def omit_absent_sparra(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized: dict[str, object] = handler(self)
        if "sparra" not in self.model_fields_set:
            serialized.pop("sparra", None)
        return serialized

    @field_validator("dids")
    @classmethod
    def validate_unique_dids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not did.strip() for did in value):
            raise ValueError("DIDs must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("duplicate DID entries are forbidden")
        return value

    @model_validator(mode="after")
    def validate_recording_policy(self) -> Self:
        if self.sparra is not None:
            if (
                len(self.dids) != 1
                or self.transcript_retention_days != 30
                or self.max_concurrent_calls != 1
                or self.recording_mode != "off"
            ):
                raise ValueError("Sparra requires one DID, thirty days, concurrency one, audio off")
            _e164(self.dids[0])
        if self.recording_mode == "off":
            if self.recording_retention_days is not None:
                raise ValueError("recording retention must be null when recording is off")
            if self.recording_required:
                raise ValueError("recording_required must be false when recording is off")
            if self.recording_play_beep:
                raise ValueError("recording_play_beep must be false when recording is off")
        elif self.recording_retention_days is None:
            raise ValueError("recording retention must be positive for telnyx_dual")
        return self


def load_agent_manifest(
    bundle_root: Path,
    *,
    host_max_concurrent_calls: int,
) -> AgentManifestV1:
    """Load one explicitly selected bundle and resolve its prompt safely."""

    if host_max_concurrent_calls < 1:
        raise ValueError("host allocation must be positive")
    if _is_symlink_or_junction(bundle_root):
        raise ValueError("agent bundle root must not be a symlink or junction")
    resolved_root = bundle_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("agent bundle root must be a directory")

    manifest_path = resolved_root / "manifest.yaml"
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("agent manifest is missing") from error
    if not resolved_manifest.is_relative_to(resolved_root) or not resolved_manifest.is_file():
        raise ValueError("agent manifest must resolve to a file inside the bundle")
    raw = yaml.load(
        resolved_manifest.read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    )
    if not isinstance(raw, dict):
        raise ValueError("agent manifest must be a YAML mapping")
    manifest = AgentManifestV1.model_validate(raw)
    if manifest.max_concurrent_calls > host_max_concurrent_calls:
        raise ValueError("manifest concurrency exceeds injected host allocation")

    configured_prompt = manifest.prompt_path
    if (
        configured_prompt.is_absolute()
        or PureWindowsPath(str(configured_prompt)).is_absolute()
        or ".." in configured_prompt.parts
    ):
        raise ValueError("prompt path must be relative and remain inside the agent bundle")
    try:
        resolved_prompt = (resolved_root / configured_prompt).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("prompt file is missing") from error
    if not resolved_prompt.is_relative_to(resolved_root) or not resolved_prompt.is_file():
        raise ValueError("prompt path must resolve to a file inside the agent bundle")

    return manifest.model_copy(update={"prompt_path": resolved_prompt})
