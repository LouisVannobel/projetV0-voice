from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from projetv0_voice.config import AgentManifestV1
from projetv0_voice.qualified_profile import (
    InferenceProfileV1,
    QualificationCandidateProfileV1,
    QualificationOverrideV1,
    QualifiedDeploymentProfileV1,
    canonical_inference_profile_sha256,
    canonical_model_schema_json,
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


def inference_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stt_model": "test/stt",
        "llm_model": "test/llm",
        "tts_model": "test/tts",
        "tts_voice": "fr-test",
        "tts_pcm_sample_rate": 24000,
        "tts_pcm_channels": 1,
        "llm_provider_policy": {"sort": "latency", "fallbacks": True},
        "tts_provider_options": {"temperature": 0.2},
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
        "token_locator_id": "telnyx-http-header-v1",
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


def expected_hashes() -> dict[str, str]:
    profile = InferenceProfileV1.model_validate(inference_data())
    return {
        "expected_deployment_id": "voice-agent-a",
        "expected_runtime_contract_sha256": HEX_A,
        "expected_image_digest": IMAGE,
        "expected_agent_bundle_sha256": HEX_B,
        "expected_inference_profile_sha256": canonical_inference_profile_sha256(profile),
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


def test_inference_provider_policy_is_deeply_immutable_and_round_trips() -> None:
    data = inference_data()
    data["llm_provider_policy"] = {
        "routing": {"providers": ["first", "second"]},
        "fallbacks": True,
    }
    profile = InferenceProfileV1.model_validate(data)
    original_hash = canonical_inference_profile_sha256(profile)

    with pytest.raises(TypeError):
        profile.llm_provider_policy["fallbacks"] = False  # type: ignore[index]
    routing = profile.llm_provider_policy["routing"]
    assert isinstance(routing, Mapping)
    with pytest.raises(TypeError):
        routing["providers"] = []  # type: ignore[index]
    providers = routing["providers"]
    assert isinstance(providers, tuple)
    with pytest.raises(TypeError):
        providers[0] = "changed"  # type: ignore[index]

    dumped = profile.model_dump_json()
    restored = InferenceProfileV1.model_validate_json(dumped)
    assert restored == profile
    assert canonical_inference_profile_sha256(restored) == original_hash


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), float("-inf")])
def test_inference_profile_rejects_non_finite_nested_numbers(invalid_number: float) -> None:
    data = inference_data()
    data["tts_provider_options"] = {"invalid": invalid_number}
    with pytest.raises(ValidationError):
        InferenceProfileV1.model_validate(data)


def test_qualified_profile_loader_accepts_exact_digest_bindings(tmp_path: Path) -> None:
    path = write_json(tmp_path / "qualified.json", qualified_data())

    profile = load_qualified_deployment_profile(path, **expected_hashes())

    assert profile.deployment_id == "voice-agent-a"
    with pytest.raises(ValidationError, match="frozen"):
        profile.deployment_id = "changed"  # type: ignore[misc]


def test_fake_profiles_are_explicit_test_fixtures_with_consistent_hashes() -> None:
    fixture_root = REPOSITORY_ROOT / "tests" / "fixtures"
    fixture_inference = InferenceProfileV1.model_validate_json(
        (fixture_root / "inference-profile-v1.json").read_text(encoding="utf-8")
    )
    fixture_hash = canonical_inference_profile_sha256(fixture_inference)
    hashes = {
        "expected_deployment_id": "voice-agent-a",
        "expected_runtime_contract_sha256": HEX_A,
        "expected_image_digest": IMAGE,
        "expected_agent_bundle_sha256": HEX_B,
        "expected_inference_profile_sha256": fixture_hash,
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
        "benchmark_max_calls": 15,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    path = write_json(tmp_path / "override.json", data)

    assert (
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
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
            ownership_check=lambda _: False,
            now=now,
        )
    with pytest.raises(ValueError, match="run ID"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=uuid4(),
            ownership_check=lambda _: True,
            now=now,
        )
    with pytest.raises(ValueError, match="not active"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
            ownership_check=lambda _: True,
            now=now + timedelta(hours=2),
        )
    with pytest.raises(ValueError, match="qualification mode"):
        load_qualification_override(
            path,
            qualification_mode=False,
            expected_run_id=RUN_ID,
            ownership_check=lambda _: True,
            now=now,
        )
    with pytest.raises(ValueError, match="not active"):
        load_qualification_override(
            path,
            qualification_mode=True,
            expected_run_id=RUN_ID,
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
            "benchmark_max_calls": 15,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )
    received: list[os.stat_result] = []

    load_qualification_override(
        path,
        qualification_mode=True,
        expected_run_id=RUN_ID,
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
            "benchmark_max_calls": 15,
            "created_at": now.isoformat(),
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
            "benchmark_max_calls": 20,
            "created_at": now.isoformat(),
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
        ownership_check=lambda _: True,
        now=now,
    )
    assert loaded.benchmark_max_calls == 20


def test_override_loader_rejects_reparse_ancestor_and_oversized_file(tmp_path: Path) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    outside = tmp_path / "outside"
    outside.mkdir()
    write_json(
        outside / "override.json",
        {
            "schema_version": 1,
            "run_id": str(RUN_ID),
            "benchmark_max_calls": 15,
            "created_at": now.isoformat(),
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
            ownership_check=lambda _: True,
            now=now,
        )


def test_override_rejects_unsupported_limit_and_impossible_time_order() -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    base = {
        "schema_version": 1,
        "run_id": str(RUN_ID),
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
