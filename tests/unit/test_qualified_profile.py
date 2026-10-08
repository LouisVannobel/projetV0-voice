from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from projetv0_voice import qualified_profile as qualified_profile_module
from projetv0_voice.config import AgentManifestV1
from projetv0_voice.qualified_profile import (
    InferenceProfileV1,
    QualificationCandidateProfileV1,
    QualificationOverrideV1,
    QualifiedDeploymentProfileV1,
    canonical_inference_profile_sha256,
    canonical_model_schema_json,
    canonical_qualified_profile_sha256,
    load_qualification_candidate_profile,
    load_qualification_override,
    load_qualified_deployment_profile,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
IMAGE = f"ghcr.io/louisvannobel/projetv0-voice@sha256:{'d' * 64}"
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENTED_TOKEN_LOCATOR = "telnyx-header-connected-v1"
REJECTED_TOKEN_LOCATORS = (
    "telnyx-http-header-v1",
    "telnyx-query-v1",
    "telnyx-start-v1",
    "telnyx-magic-v1",
)


def inference_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stt_model": "test/stt",
        "llm_model": "test/llm",
        "tts_model": "test/tts",
        "tts_voice": "fr-test",
        "tts_pcm_sample_rate": 24000,
        "tts_pcm_channels": 1,
        "llm_provider_policy": {"sort": "latency", "allow_fallbacks": True},
        "tts_provider_options": {"azure": {"temperature": 0.2}},
    }


def qualified_data(now: datetime | None = None) -> dict[str, object]:
    inference = InferenceProfileV1.model_validate(inference_data())
    return {
        "schema_version": 1,
        "deployment_id": "voice-agent-a",
        "runtime_contract_sha256": HEX_A,
        "image_digest": IMAGE,
        "agent_bundle_sha256": HEX_B,
        "inference_profile_sha256": canonical_inference_profile_sha256(inference),
        "inference": inference.model_dump(mode="json"),
        "token_locator_id": DOCUMENTED_TOKEN_LOCATOR,
        "telnyx_api_key_sha256": HEX_A,
        "telnyx_data_locality": "EU",
        "telnyx_handshake_fixture_sha256": HEX_C,
        "disclosure_mark_timeout_ms": 5000,
        "call_lease_ttl_seconds": 30,
        "qualified_at": (now or datetime(2026, 8, 25, tzinfo=UTC)).isoformat(),
    }


def candidate_data(now: datetime | None = None) -> dict[str, object]:
    now = now or datetime(2026, 8, 25, tzinfo=UTC)
    data = qualified_data(now)
    return {
        "schema_version": 1,
        "run_id": str(RUN_ID),
        "deployment_id": data["deployment_id"],
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "benchmark_did_hash": HEX_C,
        "runtime_contract_sha256": data["runtime_contract_sha256"],
        "image_digest": data["image_digest"],
        "agent_bundle_sha256": data["agent_bundle_sha256"],
        "inference_profile_sha256": data["inference_profile_sha256"],
        "inference": data["inference"],
        "token_locator_id": data["token_locator_id"],
        "telnyx_api_key_sha256": data["telnyx_api_key_sha256"],
        "telnyx_handshake_fixture_sha256": data["telnyx_handshake_fixture_sha256"],
        "disclosure_mark_timeout_ms": 10000,
        "call_lease_ttl_seconds": 30,
        "max_concurrent_calls": 1,
    }


def manifest(
    *,
    calls: int = 10,
    recording_mode: str = "off",
    recording_required: bool = False,
    recording_play_beep: bool = False,
) -> AgentManifestV1:
    return AgentManifestV1.model_validate(
        {
            "schema_version": 1,
            "tenant_id": "tenant-a",
            "agent_id": "agent-a",
            "revision": "r1",
            "dids": ["+33102030405"],
            "language": "fr-FR",
            "prompt_path": "prompt.md",
            "prompt_revision": "p1",
            "greeting": "Bonjour",
            "conversation_mode": "freeform",
            "max_concurrent_calls": calls,
            "direction": "inbound_only",
            "transport_codec": "PCMU",
            "transport_sample_rate_hz": 8000,
            "transcript_retention_days": 7,
            "recording_mode": recording_mode,
            "recording_format": "wav",
            "recording_retention_days": None if recording_mode == "off" else 7,
            "recording_required": recording_required,
            "recording_play_beep": recording_play_beep,
        }
    )


def write_json(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def create_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(  # noqa: S603 - fixed executable and bounded temp paths
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


def remove_directory_link(link: Path) -> None:
    if os.name == "nt":
        link.rmdir()
    else:
        link.unlink()


def expected_hashes() -> dict[str, object]:
    profile = InferenceProfileV1.model_validate(inference_data())
    return {
        "expected_deployment_id": "voice-agent-a",
        "expected_runtime_contract_sha256": HEX_A,
        "expected_image_digest": IMAGE,
        "expected_agent_bundle_sha256": HEX_B,
        "expected_inference_profile_sha256": canonical_inference_profile_sha256(profile),
        "ownership_check": lambda _: True,
    }


def override_bindings(now: datetime) -> dict[str, object]:
    return {
        "expected_deployment_id": "voice-agent-a",
        "expected_qualified_profile_sha256": HEX_B,
        "strict_qualified_at": now - timedelta(days=1),
    }


def test_inference_hash_uses_stable_canonical_json() -> None:
    first = InferenceProfileV1.model_validate(inference_data())
    reordered = InferenceProfileV1.model_validate(
        {key: inference_data()[key] for key in reversed(tuple(inference_data()))}
    )

    assert canonical_inference_profile_sha256(first) == canonical_inference_profile_sha256(
        reordered
    )
    assert len(canonical_inference_profile_sha256(first)) == 64


def test_inference_hash_matches_independent_literal_witness() -> None:
    profile = InferenceProfileV1.model_validate(inference_data())

    assert (
        canonical_inference_profile_sha256(profile)
        == "a452d4b5d530f36f13a3bbd63a5f5aafb7c76b3a9cfba3d9bfac1ab066d4edda"
    )


@pytest.mark.parametrize("speed", [None, 0.5, 1.0, 1.15, 2.0])
def test_tts_speed_accepts_only_optional_bounded_numbers(speed: float | None) -> None:
    profile = InferenceProfileV1.model_validate({**inference_data(), "tts_speed": speed})

    assert profile.tts_speed == speed
    restored = InferenceProfileV1.model_validate_json(profile.model_dump_json())
    assert restored.tts_speed == speed


@pytest.mark.parametrize(
    "speed",
    [
        True,
        False,
        "1.15",
        "1",
        b"1.15",
        0.4999,
        2.0001,
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_tts_speed_rejects_coercion_nonfinite_and_out_of_range_values(speed: object) -> None:
    with pytest.raises(ValidationError) as caught:
        InferenceProfileV1.model_validate({**inference_data(), "tts_speed": speed})
    assert caught.value.errors()[0]["loc"] == ("tts_speed",)


@pytest.mark.parametrize("extra", [{}, {"tts_speed": None}], ids=["absent", "none"])
def test_tts_speed_unset_preserves_exact_legacy_serialization_and_digest(
    extra: dict[str, object],
) -> None:
    legacy = inference_data()
    profile = InferenceProfileV1.model_validate({**legacy, **extra})
    assert profile.tts_speed is None
    assert profile.model_dump() == legacy
    assert profile.model_dump(mode="json") == legacy
    assert profile.model_dump_json() == json.dumps(
        legacy, ensure_ascii=False, separators=(",", ":")
    )
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    witness = (REPOSITORY_ROOT / "tests/fixtures/inference-profile-v1.json").read_bytes().strip()
    assert canonical == witness
    assert canonical_inference_profile_sha256(profile) == (
        "a452d4b5d530f36f13a3bbd63a5f5aafb7c76b3a9cfba3d9bfac1ab066d4edda"
    )


def test_tts_speed_explicit_is_bound_to_an_independent_canonical_digest() -> None:
    profile = InferenceProfileV1.model_validate({**inference_data(), "tts_speed": 1.15})
    assert profile.model_dump(mode="json")["tts_speed"] == 1.15
    assert canonical_inference_profile_sha256(profile) == (
        "467b88e4e48ee6954e605f34a4eb464842d1f7939b6fb7336c9a33432e7acb7c"
    )
    restored = InferenceProfileV1.model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert canonical_inference_profile_sha256(restored) == (
        "467b88e4e48ee6954e605f34a4eb464842d1f7939b6fb7336c9a33432e7acb7c"
    )


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (QualifiedDeploymentProfileV1, "qualified-deployment-profile-v1.json"),
        (QualificationCandidateProfileV1, "qualification-candidate-v1.json"),
    ],
)
def test_tts_speed_none_preserves_nested_profile_legacy_bytes(
    model: type[QualifiedDeploymentProfileV1] | type[QualificationCandidateProfileV1],
    filename: str,
) -> None:
    witness = (REPOSITORY_ROOT / "tests/fixtures" / filename).read_bytes().strip()
    legacy = json.loads(witness)
    profile = model.model_validate(
        {**legacy, "inference": {**legacy["inference"], "tts_speed": None}}
    )
    assert profile.inference.tts_speed is None
    assert profile.model_dump(mode="json") == legacy
    assert json.loads(profile.model_dump_json()) == legacy
    assert json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") == witness


@pytest.mark.parametrize("candidate", [False, True], ids=["qualified", "candidate"])
def test_tts_speed_explicit_round_trips_through_bound_nested_profile_loader(
    candidate: bool, tmp_path: Path
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    data = candidate_data(now) if candidate else qualified_data(now)
    data["inference"] = {**inference_data(), "tts_speed": 1.15}
    data["inference_profile_sha256"] = (
        "467b88e4e48ee6954e605f34a4eb464842d1f7939b6fb7336c9a33432e7acb7c"
    )
    bindings = {
        **expected_hashes(),
        "expected_inference_profile_sha256": (
            "467b88e4e48ee6954e605f34a4eb464842d1f7939b6fb7336c9a33432e7acb7c"
        ),
    }
    path = write_json(tmp_path / "speed-profile.json", data)
    path.chmod(0o600)
    if candidate:
        profile = load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            expected_benchmark_did_hash=HEX_C,
            manifest=manifest(),
            now=now,
            **bindings,
        )
        restored = QualificationCandidateProfileV1.model_validate_json(profile.model_dump_json())
    else:
        profile = load_qualified_deployment_profile(path, **bindings)
        restored = QualifiedDeploymentProfileV1.model_validate_json(profile.model_dump_json())
    assert profile.inference.tts_speed == 1.15
    assert restored == profile
    assert canonical_inference_profile_sha256(restored.inference) == (
        "467b88e4e48ee6954e605f34a4eb464842d1f7939b6fb7336c9a33432e7acb7c"
    )


@pytest.mark.parametrize("model", [QualifiedDeploymentProfileV1, QualificationCandidateProfileV1])
def test_nested_profile_schema_expresses_optional_tts_speed_bounds(
    model: type[QualifiedDeploymentProfileV1] | type[QualificationCandidateProfileV1],
) -> None:
    schema = model.model_json_schema()
    inference_schema = schema["$defs"]["InferenceProfileV1"]
    assert "tts_speed" not in inference_schema["required"]
    validator = Draft202012Validator(schema)
    legacy = qualified_data() if model is QualifiedDeploymentProfileV1 else candidate_data()
    for speed in (None, 0.5, 1.15, 2.0):
        validator.validate({**legacy, "inference": {**inference_data(), "tts_speed": speed}})
    for speed in (True, "1.15", 0.4999, 2.0001):
        with pytest.raises(JsonSchemaValidationError):
            validator.validate({**legacy, "inference": {**inference_data(), "tts_speed": speed}})


def test_inference_provider_policy_is_deeply_immutable_and_round_trips() -> None:
    data = inference_data()
    data["llm_provider_policy"] = {
        "routing": {"providers": ["first", "second"]},
        "allow_fallbacks": True,
    }
    data["tts_provider_options"] = {
        "azure": {"nested": {"voices": ["first", "second"]}}
    }
    profile = InferenceProfileV1.model_validate(data)
    original_hash = canonical_inference_profile_sha256(profile)

    with pytest.raises(TypeError):
        profile.llm_provider_policy["allow_fallbacks"] = False  # type: ignore[index]
    routing = profile.llm_provider_policy["routing"]
    assert isinstance(routing, Mapping)
    with pytest.raises(TypeError):
        routing["providers"] = []  # type: ignore[index]
    providers = routing["providers"]
    assert isinstance(providers, tuple)
    with pytest.raises(TypeError):
        providers[0] = "changed"  # type: ignore[index]
    provider_options = profile.tts_provider_options["azure"]
    nested = provider_options["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["voices"] = []  # type: ignore[index]

    dumped = profile.model_dump_json()
    restored = InferenceProfileV1.model_validate_json(dumped)
    assert restored == profile
    assert canonical_inference_profile_sha256(restored) == original_hash


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), float("-inf")])
def test_inference_profile_rejects_non_finite_nested_numbers(invalid_number: float) -> None:
    data = inference_data()
    data["tts_provider_options"] = {"azure": {"invalid": invalid_number}}
    with pytest.raises(ValidationError):
        InferenceProfileV1.model_validate(data)


@pytest.mark.parametrize(
    "policy",
    [
        {"fallbacks": True},
        {"provider": {"sort": "latency"}},
        {"allow_fallbacks": 1},
        {"allow_fallbacks": "true"},
        {"allow_fallbacks": None},
    ],
)
def test_inference_profile_rejects_ambiguous_openrouter_llm_policy(
    policy: dict[str, object],
) -> None:
    data = inference_data()
    data["llm_provider_policy"] = policy

    with pytest.raises(ValidationError, match="llm_provider_policy"):
        InferenceProfileV1.model_validate(data)


@pytest.mark.parametrize(
    "options",
    [
        {"": {}},
        {" test-provider": {}},
        {"test-provider ": {}},
        {"Azure": {}},
        {"azure!": {}},
        {"provider": {}},
        {"options": {}},
        {"test-provider": 1},
        {"test-provider": []},
        {"test-provider": None},
    ],
)
def test_inference_profile_requires_provider_slug_option_objects(
    options: dict[str, object],
) -> None:
    data = inference_data()
    data["tts_provider_options"] = options

    with pytest.raises(ValidationError, match="tts_provider_options"):
        InferenceProfileV1.model_validate(data)


def test_qualified_profile_loader_accepts_exact_digest_bindings(tmp_path: Path) -> None:
    path = write_json(tmp_path / "qualified.json", qualified_data())

    profile = load_qualified_deployment_profile(path, **expected_hashes())

    assert profile.deployment_id == "voice-agent-a"
    with pytest.raises(ValidationError, match="frozen"):
        profile.deployment_id = "changed"  # type: ignore[misc]


def test_task10_profiles_bind_telnyx_locality_ttl_and_strict_profile_hash() -> None:
    strict_data = qualified_data()
    strict_data.update(
        {
            "telnyx_api_key_sha256": HEX_A,
            "telnyx_data_locality": "EU",
            "call_lease_ttl_seconds": 300,
        }
    )
    strict = QualifiedDeploymentProfileV1.model_validate(strict_data)
    assert strict.telnyx_api_key_sha256 == HEX_A
    assert strict.telnyx_data_locality == "EU"

    candidate = candidate_data()
    candidate["telnyx_api_key_sha256"] = HEX_A
    assert (
        QualificationCandidateProfileV1.model_validate(candidate).telnyx_api_key_sha256
        == HEX_A
    )

    override = QualificationOverrideV1.model_validate(
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "deployment_id": "voice-agent-a",
            "qualified_profile_sha256": HEX_B,
            "benchmark_max_calls": 15,
            "created_at": "2026-08-29T09:00:00Z",
            "expires_at": "2026-08-29T11:00:00Z",
        }
    )
    assert (override.deployment_id, override.qualified_profile_sha256) == (
        "voice-agent-a",
        HEX_B,
    )

    for invalid_ttl in (4, 301):
        invalid = dict(strict_data)
        invalid["call_lease_ttl_seconds"] = invalid_ttl
        with pytest.raises(ValidationError, match="call_lease_ttl_seconds"):
            QualifiedDeploymentProfileV1.model_validate(invalid)


def test_strict_and_candidate_use_the_same_injected_safe_owner_loader(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    strict_path = write_json(tmp_path / "strict.json", qualified_data(now))
    candidate_path = write_json(tmp_path / "candidate.json", candidate_data(now))

    with pytest.raises(ValueError, match="root-owned"):
        load_qualified_deployment_profile(
            strict_path,
            **{**expected_hashes(), "ownership_check": lambda _: False},
        )
    with pytest.raises(ValueError, match="root-owned"):
        load_qualification_candidate_profile(
            candidate_path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            expected_benchmark_did_hash=HEX_C,
            manifest=manifest(),
            now=now,
            **{**expected_hashes(), "ownership_check": lambda _: False},
        )

    if os.name != "nt":
        strict_path.chmod(0o666)
        with pytest.raises(ValueError, match="group or other writable"):
            load_qualified_deployment_profile(
                strict_path,
                **{**expected_hashes(), "ownership_check": lambda _: True},
            )


def test_override_loader_binds_canonical_strict_profile_and_exact_time_order(
    tmp_path: Path,
) -> None:
    strict = QualifiedDeploymentProfileV1.model_validate(qualified_data())
    strict_hash = canonical_qualified_profile_sha256(strict)
    now = datetime(2026, 8, 29, 10, tzinfo=UTC)
    data = {
        "schema_version": 1,
        "run_id": str(RUN_ID),
        "deployment_id": strict.deployment_id,
        "qualified_profile_sha256": strict_hash,
        "benchmark_max_calls": 15,
        "created_at": (now - timedelta(seconds=1)).isoformat(),
        "expires_at": (now + timedelta(seconds=1)).isoformat(),
    }
    path = write_json(tmp_path / "override.json", data)
    kwargs = {
        "qualification_mode": True,
        "expected_run_id": RUN_ID,
        "expected_deployment_id": strict.deployment_id,
        "expected_qualified_profile_sha256": strict_hash,
        "strict_qualified_at": strict.qualified_at,
        "ownership_check": lambda _: True,
        "now": now,
    }

    assert load_qualification_override(path, **kwargs).benchmark_max_calls == 15
    for field, value in (
        ("deployment_id", "voice-agent-b"),
        ("qualified_profile_sha256", HEX_B),
        ("created_at", now.isoformat()),
        ("expires_at", now.isoformat()),
    ):
        write_json(path, {**data, field: value})
        with pytest.raises(ValueError):
            load_qualification_override(path, **kwargs)


def test_profile_models_accept_only_documented_telnyx_token_locator() -> None:
    for model, data in (
        (QualifiedDeploymentProfileV1, qualified_data()),
        (QualificationCandidateProfileV1, candidate_data()),
    ):
        data["token_locator_id"] = DOCUMENTED_TOKEN_LOCATOR
        assert model.model_validate(data).token_locator_id == DOCUMENTED_TOKEN_LOCATOR

        for rejected_locator in REJECTED_TOKEN_LOCATORS:
            data["token_locator_id"] = rejected_locator
            with pytest.raises(ValidationError, match="token_locator_id"):
                model.model_validate(data)


def test_profile_loaders_accept_only_documented_telnyx_token_locator(
    tmp_path: Path,
) -> None:
    qualified = qualified_data()
    qualified["token_locator_id"] = DOCUMENTED_TOKEN_LOCATOR
    qualified_path = write_json(tmp_path / "qualified.json", qualified)
    assert (
        load_qualified_deployment_profile(qualified_path, **expected_hashes()).token_locator_id
        == DOCUMENTED_TOKEN_LOCATOR
    )

    now = datetime(2026, 8, 25, tzinfo=UTC)
    candidate = candidate_data(now)
    candidate["token_locator_id"] = DOCUMENTED_TOKEN_LOCATOR
    candidate_path = write_json(tmp_path / "candidate.json", candidate)
    candidate_loader_kwargs = {
        "qualification_mode": True,
        "expected_run_id": RUN_ID,
        "expected_benchmark_did_hash": HEX_C,
        "manifest": manifest(),
        "now": now,
        **expected_hashes(),
    }
    assert (
        load_qualification_candidate_profile(candidate_path, **candidate_loader_kwargs)
        .token_locator_id
        == DOCUMENTED_TOKEN_LOCATOR
    )

    for rejected_locator in REJECTED_TOKEN_LOCATORS:
        qualified["token_locator_id"] = rejected_locator
        write_json(qualified_path, qualified)
        with pytest.raises(ValidationError, match="token_locator_id"):
            load_qualified_deployment_profile(qualified_path, **expected_hashes())

        candidate["token_locator_id"] = rejected_locator
        write_json(candidate_path, candidate)
        with pytest.raises(ValidationError, match="token_locator_id"):
            load_qualification_candidate_profile(candidate_path, **candidate_loader_kwargs)


@pytest.mark.parametrize(
    "model",
    [QualifiedDeploymentProfileV1, QualificationCandidateProfileV1],
)
def test_profile_schemas_allow_only_documented_telnyx_token_locator(
    model: type[QualifiedDeploymentProfileV1] | type[QualificationCandidateProfileV1],
) -> None:
    locator_schema = model.model_json_schema()["properties"]["token_locator_id"]

    assert locator_schema["const"] == DOCUMENTED_TOKEN_LOCATOR
    assert "enum" not in locator_schema


@pytest.mark.parametrize(
    "model",
    [QualifiedDeploymentProfileV1, QualificationCandidateProfileV1],
)
def test_inference_schema_expresses_openrouter_policy_key_contracts(
    model: type[QualifiedDeploymentProfileV1] | type[QualificationCandidateProfileV1],
) -> None:
    inference_properties = model.model_json_schema()["$defs"]["InferenceProfileV1"][
        "properties"
    ]
    llm_policy = inference_properties["llm_provider_policy"]
    tts_options = inference_properties["tts_provider_options"]

    assert llm_policy["additionalProperties"] == {"$ref": "#/$defs/JsonValue"}
    assert llm_policy["properties"] == {"allow_fallbacks": {"type": "boolean"}}
    assert llm_policy["allOf"] == [
        {"not": {"required": ["fallbacks"]}},
        {"not": {"required": ["provider"]}},
    ]
    assert tts_options["propertyNames"] == {
        "allOf": [
            {"pattern": r"^[a-z0-9]+(?:[./_-][a-z0-9]+)*$"},
            {"not": {"enum": ["provider", "options"]}},
        ]
    }


def test_fake_profiles_are_explicit_test_fixtures_with_consistent_hashes() -> None:
    fixture_root = REPOSITORY_ROOT / "tests" / "fixtures"
    fixture_inference = InferenceProfileV1.model_validate_json(
        (fixture_root / "inference-profile-v1.json").read_text(encoding="utf-8")
    )
    fixture_hash = canonical_inference_profile_sha256(fixture_inference)
    assert fixture_hash == "a452d4b5d530f36f13a3bbd63a5f5aafb7c76b3a9cfba3d9bfac1ab066d4edda"
    hashes = {
        "expected_deployment_id": "voice-agent-a",
        "expected_runtime_contract_sha256": HEX_A,
        "expected_image_digest": IMAGE,
        "expected_agent_bundle_sha256": HEX_B,
        "expected_inference_profile_sha256": fixture_hash,
        "ownership_check": lambda _: True,
    }

    qualified = load_qualified_deployment_profile(
        fixture_root / "qualified-deployment-profile-v1.json", **hashes
    )
    candidate = load_qualification_candidate_profile(
        fixture_root / "qualification-candidate-v1.json",
        qualification_mode=True,
        expected_run_id=RUN_ID,
        expected_benchmark_did_hash=HEX_C,
        manifest=manifest(),
        now=datetime(2026, 8, 25, tzinfo=UTC),
        **hashes,
    )

    assert qualified.inference == fixture_inference
    assert candidate.inference == fixture_inference


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("runtime_contract_sha256", "e" * 64),
        ("image_digest", f"ghcr.io/test/wrong@sha256:{'e' * 64}"),
        ("agent_bundle_sha256", "e" * 64),
        ("inference_profile_sha256", "e" * 64),
    ],
)
def test_qualified_profile_loader_rejects_any_digest_mismatch(
    tmp_path: Path, field: str, replacement: str
) -> None:
    data = qualified_data()
    data[field] = replacement
    path = write_json(tmp_path / "qualified.json", data)

    with pytest.raises(ValueError, match=field):
        load_qualified_deployment_profile(path, **expected_hashes())


def test_qualified_profile_loader_rejects_wrong_deployment_binding(tmp_path: Path) -> None:
    path = write_json(tmp_path / "qualified.json", qualified_data())

    with pytest.raises(ValueError, match="deployment_id"):
        load_qualified_deployment_profile(
            path, **{**expected_hashes(), "expected_deployment_id": "voice-agent-b"}
        )


def test_qualified_loader_recomputes_inference_hash_from_canonical_payload(
    tmp_path: Path,
) -> None:
    data = qualified_data()
    inference = dict(data["inference"])  # type: ignore[arg-type]
    inference["tts_voice"] = "mutated-voice"
    data["inference"] = inference
    path = write_json(tmp_path / "qualified.json", data)

    with pytest.raises(ValueError, match="canonical inference"):
        load_qualified_deployment_profile(path, **expected_hashes())


def test_qualified_profile_rejects_noncanonical_hash_image_and_locator() -> None:
    for field, value in (
        ("runtime_contract_sha256", "A" * 64),
        ("image_digest", "projetv0-voice:latest"),
        ("token_locator_id", "telnyx-magic-v1"),
    ):
        data = qualified_data()
        data[field] = value
        with pytest.raises(ValidationError, match=field):
            QualifiedDeploymentProfileV1.model_validate(data)


@pytest.mark.parametrize(
    "invalid_image",
    [
        f"projetv0-voice@sha256:{'d' * 64}",
        f"ghcr.io//projetv0-voice@sha256:{'d' * 64}",
        f"ghcr.io/louis/../projetv0-voice@sha256:{'d' * 64}",
        f"ghcr.io/louis/projetv0-voice/@sha256:{'d' * 64}",
        "ghcr.io/louis/projetv0-voice:latest",
        f"GHCR.io/louis/projetv0-voice@sha256:{'d' * 64}",
    ],
)
def test_image_digest_requires_a_lowercase_registry_namespace_and_digest(
    invalid_image: str,
) -> None:
    data = qualified_data()
    data["image_digest"] = invalid_image
    with pytest.raises(ValidationError, match="image_digest"):
        QualifiedDeploymentProfileV1.model_validate(data)


def test_profile_rejects_unknown_fields_schema_versions_and_naive_datetimes() -> None:
    for updates in (
        {"unknown": True},
        {"schema_version": 2},
        {"qualified_at": "2026-08-25T10:00:00"},
    ):
        data = qualified_data()
        data.update(updates)
        with pytest.raises(ValidationError):
            QualifiedDeploymentProfileV1.model_validate(data)


@pytest.mark.parametrize("invalid_datetime", [True, 1, 1.5, "1", "1724572800"])
def test_profiles_reject_numeric_datetime_coercion(invalid_datetime: object) -> None:
    qualified = qualified_data()
    qualified["qualified_at"] = invalid_datetime
    with pytest.raises(ValidationError, match="qualified_at"):
        QualifiedDeploymentProfileV1.model_validate(qualified)

    candidate = candidate_data()
    candidate["expires_at"] = invalid_datetime
    with pytest.raises(ValidationError, match="expires_at"):
        QualificationCandidateProfileV1.model_validate(candidate)

    override = {
        "schema_version": 1,
        "run_id": str(RUN_ID),
        "deployment_id": "voice-agent-a",
        "qualified_profile_sha256": HEX_B,
        "benchmark_max_calls": 15,
        "created_at": invalid_datetime,
        "expires_at": "2099-01-01T00:00:00Z",
    }
    with pytest.raises(ValidationError, match="created_at"):
        QualificationOverrideV1.model_validate(override)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_profiles_do_not_coerce_integer_contract_fields(value: object) -> None:
    data = qualified_data()
    data["schema_version"] = value
    with pytest.raises(ValidationError, match="schema_version"):
        QualifiedDeploymentProfileV1.model_validate(data)

    candidate = candidate_data()
    candidate["max_concurrent_calls"] = value
    with pytest.raises(ValidationError, match="max_concurrent_calls"):
        QualificationCandidateProfileV1.model_validate(candidate)

    for field in (
        "call_lease_ttl_seconds",
        "disclosure_mark_timeout_ms",
    ):
        qualified = qualified_data()
        qualified[field] = value
        with pytest.raises(ValidationError, match=field):
            QualifiedDeploymentProfileV1.model_validate(qualified)

    inference = inference_data()
    inference["tts_pcm_sample_rate"] = value
    with pytest.raises(ValidationError, match="tts_pcm_sample_rate"):
        InferenceProfileV1.model_validate(inference)


@pytest.mark.parametrize("timeout_ms", [2999, 10001])
def test_qualified_disclosure_timeout_is_clamped_to_3_through_10_seconds(
    timeout_ms: int,
) -> None:
    data = qualified_data()
    data["disclosure_mark_timeout_ms"] = timeout_ms
    with pytest.raises(ValidationError, match="disclosure_mark_timeout_ms"):
        QualifiedDeploymentProfileV1.model_validate(data)


def test_candidate_loader_is_bound_to_mode_time_did_hashes_and_one_call(tmp_path: Path) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    path = write_json(tmp_path / "candidate.json", candidate_data(now))
    kwargs = {
        **expected_hashes(),
        "expected_run_id": RUN_ID,
        "expected_benchmark_did_hash": HEX_C,
        "manifest": manifest(),
        "now": now,
    }

    accepted = load_qualification_candidate_profile(path, qualification_mode=True, **kwargs)
    assert accepted.max_concurrent_calls == 1

    with pytest.raises(ValueError, match="qualification mode"):
        load_qualification_candidate_profile(path, qualification_mode=False, **kwargs)
    with pytest.raises(ValueError, match="benchmark DID"):
        load_qualification_candidate_profile(
            path, qualification_mode=True, **{**kwargs, "expected_benchmark_did_hash": "e" * 64}
        )
    with pytest.raises(ValueError, match="run_id"):
        load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            **{**kwargs, "expected_run_id": uuid4()},
        )
    with pytest.raises(ValueError, match="deployment_id"):
        load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            **{**kwargs, "expected_deployment_id": "voice-agent-b"},
        )
    with pytest.raises(ValueError, match="expired"):
        load_qualification_candidate_profile(
            path, qualification_mode=True, **{**kwargs, "now": now + timedelta(hours=2)}
        )
    with pytest.raises(ValueError, match="recording off"):
        load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            **{**kwargs, "manifest": manifest(recording_mode="telnyx_dual")},
        )
    with pytest.raises(ValueError, match="recording_required"):
        load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            **{
                **kwargs,
                "manifest": manifest().model_copy(update={"recording_required": True}),
            },
        )
    with pytest.raises(ValueError, match="recording_play_beep"):
        load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            **{
                **kwargs,
                "manifest": manifest().model_copy(update={"recording_play_beep": True}),
            },
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("disclosure_mark_timeout_ms", 9000),
        ("call_lease_ttl_seconds", 31),
        ("max_concurrent_calls", 2),
    ],
)
def test_candidate_rejects_limits_other_than_10s_30s_and_one_call(
    field: str, value: int
) -> None:
    data = candidate_data()
    data[field] = value
    with pytest.raises(ValidationError, match=field):
        QualificationCandidateProfileV1.model_validate(data)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("runtime_contract_sha256", "e" * 64),
        ("image_digest", f"ghcr.io/test/wrong@sha256:{'e' * 64}"),
        ("agent_bundle_sha256", "e" * 64),
        ("inference_profile_sha256", "e" * 64),
    ],
)
def test_candidate_loader_rejects_each_digest_mismatch(
    tmp_path: Path, field: str, replacement: str
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    data = candidate_data(now)
    data[field] = replacement
    path = write_json(tmp_path / "candidate.json", data)
    with pytest.raises(ValueError, match=field):
        load_qualification_candidate_profile(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            expected_benchmark_did_hash=HEX_C,
            manifest=manifest(),
            now=now,
            **expected_hashes(),
        )


def test_override_loader_requires_root_ownership_run_binding_and_freshness(tmp_path: Path) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    data = {
        "schema_version": 1,
        "run_id": str(RUN_ID),
        "deployment_id": "voice-agent-a",
        "qualified_profile_sha256": HEX_B,
        "benchmark_max_calls": 15,
        "created_at": (now - timedelta(seconds=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    path = write_json(tmp_path / "override.json", data)

    assert (
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now,
        ).benchmark_max_calls
        == 15
    )
    with pytest.raises(ValueError, match="root-owned"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: False,
            now=now,
        )
    with pytest.raises(ValueError, match="run ID"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=uuid4(),
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now,
        )
    with pytest.raises(ValueError, match="not active"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now + timedelta(hours=2),
        )
    with pytest.raises(ValueError, match="qualification mode"):
        load_qualification_override(
            path,
            qualification_mode=False,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now,
        )
    with pytest.raises(ValueError, match="not active"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now - timedelta(seconds=1),
        )


def test_override_ownership_check_receives_the_open_file_stat(tmp_path: Path) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    path = write_json(
        tmp_path / "override.json",
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "deployment_id": "voice-agent-a",
            "qualified_profile_sha256": HEX_B,
            "benchmark_max_calls": 15,
            "created_at": (now - timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )
    received: list[os.stat_result] = []

    load_qualification_override(
        path,
        qualification_mode=True,
        expected_run_id=RUN_ID,
        **override_bindings(now),
        ownership_check=lambda file_stat: received.append(file_stat) is None,
        now=now,
    )

    assert len(received) == 1
    assert isinstance(received[0], os.stat_result)
    assert received[0].st_ino == path.stat().st_ino


def test_override_atomic_open_rejects_a_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    path = write_json(
        tmp_path / "override.json",
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "deployment_id": "voice-agent-a",
            "qualified_profile_sha256": HEX_B,
            "benchmark_max_calls": 15,
            "created_at": (now - timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )
    replacement = write_json(tmp_path / "replacement.json", json.loads(path.read_text()))
    real_open = os.open

    def swapping_open(open_path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        replacement.replace(path)
        return real_open(open_path, flags, mode)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(ValueError, match="changed before atomic open"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now,
        )


def test_override_loader_reads_from_open_fd_not_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    path = write_json(
        tmp_path / "override.json",
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "deployment_id": "voice-agent-a",
            "qualified_profile_sha256": HEX_B,
            "benchmark_max_calls": 20,
            "created_at": (now - timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )

    def reject_path_reread(*args: object, **kwargs: object) -> str:
        raise AssertionError("path reread")

    monkeypatch.setattr(Path, "read_text", reject_path_reread)
    loaded = load_qualification_override(
        path,
        qualification_mode=True,
        expected_run_id=RUN_ID,
        **override_bindings(now),
        ownership_check=lambda _: True,
        now=now,
    )
    assert loaded.benchmark_max_calls == 20


def test_public_profile_open_adds_nonblocking_and_close_on_exec_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    path = write_json(
        tmp_path / "override.json",
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "deployment_id": "voice-agent-a",
            "qualified_profile_sha256": HEX_B,
            "benchmark_max_calls": 20,
            "created_at": (now - timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )
    close_on_exec = 1 << 27
    nonblocking = 1 << 28
    real_open = os.open
    seen: list[int] = []
    monkeypatch.setattr(qualified_profile_module.os, "O_CLOEXEC", close_on_exec, raising=False)
    monkeypatch.setattr(qualified_profile_module.os, "O_NONBLOCK", nonblocking, raising=False)

    def recording_open(
        open_path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        seen.append(flags)
        return real_open(open_path, flags & ~(close_on_exec | nonblocking), mode)

    monkeypatch.setattr(qualified_profile_module.os, "open", recording_open)

    loaded = load_qualification_override(
        path,
        qualification_mode=True,
        expected_run_id=RUN_ID,
        **override_bindings(now),
        ownership_check=lambda _: True,
        now=now,
    )

    assert loaded.benchmark_max_calls == 20
    assert len(seen) == 1
    assert seen[0] & close_on_exec
    assert seen[0] & nonblocking


def _run_public_profile_subprocess(path: Path) -> dict[str, object]:
    code = """
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID
sys.path.insert(0, os.environ["PYTHONPATH"])
from projetv0_voice.qualified_profile import load_qualification_override
now = datetime(2026, 8, 25, tzinfo=UTC)
try:
    load_qualification_override(
        Path(sys.argv[1]),
        qualification_mode=True,
        expected_run_id=UUID("11111111-1111-4111-8111-111111111111"),
        expected_deployment_id="voice-agent-a",
        expected_qualified_profile_sha256="b" * 64,
        strict_qualified_at=now - timedelta(days=1),
        ownership_check=lambda _: True,
        now=now,
    )
except BaseException as error:
    print(json.dumps({"kind": "error", "message": str(error)}))
else:
    print(json.dumps({"kind": "value"}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", code, str(path)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        return {"kind": "timeout"}
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(os.name == "nt", reason="Linux kernel descriptor contract")
@pytest.mark.parametrize("kind", ["fifo", "socket", "directory", "symlink"])
def test_linux_kernel_public_profile_nonregular_and_symlink_fail_before_deadline(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "profile.json"
    socket_owner: socket.socket | None = None
    if kind == "fifo":
        os.mkfifo(path, 0o440)
    elif kind == "socket":
        socket_owner = socket.socket(socket.AF_UNIX)
        socket_owner.bind(str(path))
    elif kind == "directory":
        path.mkdir()
    else:
        target = write_json(tmp_path / "target.json", {})
        path.symlink_to(target)
    try:
        result = _run_public_profile_subprocess(path)
    finally:
        if socket_owner is not None:
            socket_owner.close()

    assert result["kind"] == "error"


def test_override_loader_rejects_reparse_ancestor_and_oversized_file(tmp_path: Path) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    outside = tmp_path / "outside"
    outside.mkdir()
    write_json(
        outside / "override.json",
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "deployment_id": "voice-agent-a",
            "qualified_profile_sha256": HEX_B,
            "benchmark_max_calls": 15,
            "created_at": (now - timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )
    linked = tmp_path / "linked"
    create_directory_link(linked, outside)
    try:
        with pytest.raises(ValueError, match="symlink or junction"):
            load_qualification_override(
                linked / "override.json",
                qualification_mode=True,
                expected_run_id=RUN_ID,
                **override_bindings(now),
                ownership_check=lambda _: True,
                now=now,
            )
    finally:
        remove_directory_link(linked)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * 65_536 + b"}")
    with pytest.raises(ValueError, match="too large"):
        load_qualification_override(
            oversized,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            **override_bindings(now),
            ownership_check=lambda _: True,
            now=now,
        )


def test_override_rejects_unsupported_limit_and_impossible_time_order() -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    base = {
        "schema_version": 1,
        "run_id": str(RUN_ID),
        "deployment_id": "voice-agent-a",
        "qualified_profile_sha256": HEX_B,
        "benchmark_max_calls": 15,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    for updates in (
        {"benchmark_max_calls": 10},
        {"expires_at": (now - timedelta(seconds=1)).isoformat()},
    ):
        with pytest.raises(ValidationError):
            QualificationOverrideV1.model_validate({**base, **updates})


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (QualifiedDeploymentProfileV1, "qualified-v1.schema.json"),
        (QualificationCandidateProfileV1, "qualification-candidate-v1.schema.json"),
        (QualificationOverrideV1, "qualification-override-v1.schema.json"),
    ],
)
def test_committed_profile_schema_is_the_exact_deterministic_model_schema(
    model: type[QualifiedDeploymentProfileV1]
    | type[QualificationCandidateProfileV1]
    | type[QualificationOverrideV1],
    filename: str,
) -> None:
    committed_path = REPOSITORY_ROOT / "deployment-profiles" / filename
    committed_text = committed_path.read_text(encoding="utf-8")
    committed = json.loads(committed_text)

    assert committed == model.model_json_schema()
    assert committed_text == canonical_model_schema_json(model)
    assert committed["additionalProperties"] is False
    for definition in committed.get("$defs", {}).values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False


def test_eu_attestation_schema_limits_its_scope_to_voice_api_and_media() -> None:
    locality = QualifiedDeploymentProfileV1.model_json_schema()["properties"][
        "telnyx_data_locality"
    ]

    assert locality["const"] == "EU"
    assert locality["description"] == (
        "Operator-attested EU Voice API and media routing. Does not attest "
        "Telnyx CDR/MDR storage location or external inference residency."
    )


@pytest.mark.parametrize(
    ("schema_filename", "fixture_filename"),
    [
        ("qualified-v1.schema.json", "qualified-deployment-profile-v1.json"),
        ("qualification-candidate-v1.schema.json", "qualification-candidate-v1.json"),
    ],
)
def test_committed_profile_schema_semantically_accepts_its_fixture(
    schema_filename: str,
    fixture_filename: str,
) -> None:
    schema = json.loads(
        (REPOSITORY_ROOT / "deployment-profiles" / schema_filename).read_text(encoding="utf-8")
    )
    fixture = json.loads(
        (REPOSITORY_ROOT / "tests" / "fixtures" / fixture_filename).read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(fixture)


@pytest.mark.parametrize(
    ("schema_filename", "fixture_filename", "field_path", "rejected_value"),
    [
        (
            "qualified-v1.schema.json",
            "qualified-deployment-profile-v1.json",
            ("inference", "tts_pcm_sample_rate"),
            0,
        ),
        (
            "qualified-v1.schema.json",
            "qualified-deployment-profile-v1.json",
            ("inference", "tts_pcm_sample_rate"),
            -1,
        ),
        (
            "qualification-candidate-v1.schema.json",
            "qualification-candidate-v1.json",
            ("inference", "tts_pcm_sample_rate"),
            0,
        ),
        (
            "qualification-candidate-v1.schema.json",
            "qualification-candidate-v1.json",
            ("inference", "tts_pcm_sample_rate"),
            -1,
        ),
        (
            "qualified-v1.schema.json",
            "qualified-deployment-profile-v1.json",
            ("call_lease_ttl_seconds",),
            0,
        ),
        (
            "qualified-v1.schema.json",
            "qualified-deployment-profile-v1.json",
            ("call_lease_ttl_seconds",),
            -1,
        ),
        (
            "qualified-v1.schema.json",
            "qualified-deployment-profile-v1.json",
            ("disclosure_mark_timeout_ms",),
            2999,
        ),
        (
            "qualified-v1.schema.json",
            "qualified-deployment-profile-v1.json",
            ("disclosure_mark_timeout_ms",),
            10001,
        ),
    ],
)
def test_committed_profile_schema_semantically_rejects_numeric_contract_violations(
    schema_filename: str,
    fixture_filename: str,
    field_path: tuple[str, ...],
    rejected_value: int,
) -> None:
    schema = json.loads(
        (REPOSITORY_ROOT / "deployment-profiles" / schema_filename).read_text(encoding="utf-8")
    )
    instance = json.loads(
        (REPOSITORY_ROOT / "tests" / "fixtures" / fixture_filename).read_text(
            encoding="utf-8"
        )
    )
    target = instance
    for component in field_path[:-1]:
        target = target[component]
    target[field_path[-1]] = rejected_value

    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(instance)
