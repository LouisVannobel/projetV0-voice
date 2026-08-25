from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from projetv0_voice.config import AgentManifestV1, load_agent_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def manifest_data(**updates: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": 1,
        "tenant_id": "tenant-a",
        "agent_id": "agent-a",
        "revision": "2026-08-25.1",
        "dids": ["+33102030405"],
        "language": "fr-FR",
        "prompt_path": "prompt.md",
        "prompt_revision": "prompt-1",
        "greeting": "Bonjour, comment puis-je vous aider ?",
        "conversation_mode": "freeform",
        "max_concurrent_calls": 10,
        "direction": "inbound_only",
        "transport_codec": "PCMU",
        "transport_sample_rate_hz": 8000,
        "transcript_retention_days": 7,
        "recording_mode": "off",
        "recording_format": "wav",
        "recording_retention_days": None,
        "recording_required": False,
        "recording_play_beep": False,
    }
    data.update(updates)
    return data


def write_bundle(root: Path, data: dict[str, object] | None = None) -> Path:
    root.mkdir()
    (root / "prompt.md").write_text("Tu es un assistant vocal.\n", encoding="utf-8")
    (root / "manifest.yaml").write_text(
        json.dumps(data or manifest_data(), ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def test_loader_parses_yaml_instead_of_only_json_with_a_yaml_extension(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "prompt.md").write_text("Tu es un assistant vocal.\n", encoding="utf-8")
    yaml_text = "\n".join(
        [
            "schema_version: 1",
            "tenant_id: tenant-a",
            "agent_id: agent-a",
            "revision: r1",
            "dids:",
            "  - '+33102030405'",
            "language: fr-FR",
            "prompt_path: prompt.md",
            "prompt_revision: p1",
            "greeting: Bonjour",
            "conversation_mode: freeform",
            "max_concurrent_calls: 10",
            "direction: inbound_only",
            "transport_codec: PCMU",
            "transport_sample_rate_hz: 8000",
            "transcript_retention_days: 7",
            "recording_mode: 'off'",
            "recording_format: wav",
            "recording_retention_days: null",
            "recording_required: false",
            "recording_play_beep: false",
            "",
        ]
    )
    (bundle / "manifest.yaml").write_text(yaml_text, encoding="utf-8")

    assert load_agent_manifest(bundle, host_max_concurrent_calls=10).tenant_id == "tenant-a"


def test_manifest_round_trip_is_strict_and_frozen() -> None:
    manifest = AgentManifestV1.model_validate(manifest_data())

    assert AgentManifestV1.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="frozen"):
        manifest.max_concurrent_calls = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "flow_ref",
        "flows",
        "toolset_id",
        "tools",
        "tool_api_secret",
        "mcp_servers",
    ],
)
def test_manifest_rejects_every_flow_tool_or_mcp_field(forbidden_field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentManifestV1.model_validate(manifest_data(**{forbidden_field: "forbidden"}))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"dids": ["+33102030405", "+33102030405"]}, "duplicate DID"),
        ({"transcript_retention_days": 0}, "transcript_retention_days"),
        ({"recording_mode": "off", "recording_retention_days": 7}, "recording retention"),
        ({"recording_mode": "off", "recording_required": True}, "recording_required"),
        ({"recording_mode": "off", "recording_play_beep": True}, "recording_play_beep"),
        (
            {"recording_mode": "telnyx_dual", "recording_retention_days": None},
            "recording retention",
        ),
        ({"max_concurrent_calls": 0}, "max_concurrent_calls"),
    ],
)
def test_manifest_rejects_invalid_policy_combinations(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AgentManifestV1.model_validate(manifest_data(**updates))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_manifest_does_not_coerce_integer_policy_fields(value: object) -> None:
    for field in ("schema_version", "max_concurrent_calls", "transcript_retention_days"):
        with pytest.raises(ValidationError, match=field):
            AgentManifestV1.model_validate(manifest_data(**{field: value}))


def test_manifest_does_not_coerce_integer_to_boolean_policy() -> None:
    with pytest.raises(ValidationError, match="recording_required"):
        AgentManifestV1.model_validate(manifest_data(recording_required=0))


def test_loader_resolves_prompt_inside_bundle_and_enforces_host_allocation(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path / "bundle")

    manifest = load_agent_manifest(bundle, host_max_concurrent_calls=10)

    assert manifest.prompt_path == (bundle / "prompt.md").resolve()
    with pytest.raises(ValueError, match="host allocation"):
        load_agent_manifest(bundle, host_max_concurrent_calls=9)


@pytest.mark.parametrize(
    "prompt_path", ["../outside.md", "C:/outside.md", "/outside.md", "missing.md"]
)
def test_loader_rejects_prompt_escape_or_missing_file(tmp_path: Path, prompt_path: str) -> None:
    bundle = write_bundle(tmp_path / "bundle", manifest_data(prompt_path=prompt_path))
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="prompt"):
        load_agent_manifest(bundle, host_max_concurrent_calls=10)


def test_loader_rejects_symlink_whose_target_escapes_bundle(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "prompt.md").write_text("outside", encoding="utf-8")
    bundle = write_bundle(
        tmp_path / "bundle", manifest_data(prompt_path="linked-dir/prompt.md")
    )
    link = bundle / "linked-dir"
    if os.name == "nt":
        completed = subprocess.run(  # noqa: S603 - fixed local executable and validated paths
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    else:
        link.symlink_to(outside, target_is_directory=True)

    try:
        with pytest.raises(ValueError, match="prompt"):
            load_agent_manifest(bundle, host_max_concurrent_calls=10)
    finally:
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()


def test_static_first_agent_bundle_matches_v1_policy() -> None:
    bundle = REPOSITORY_ROOT / "agents" / "agent-a"

    manifest = load_agent_manifest(bundle, host_max_concurrent_calls=10)

    assert manifest.conversation_mode == "freeform"
    assert manifest.direction == "inbound_only"
    assert (manifest.transport_codec, manifest.transport_sample_rate_hz) == ("PCMU", 8000)
    assert manifest.max_concurrent_calls == 10
    assert manifest.transcript_retention_days == 7
    assert manifest.recording_mode == "off"
    disclosure = manifest.greeting.casefold()
    for required_disclosure in (
        "projetv0",
        "intelligence artificielle",
        "suivi de votre demande",
        "transcription",
        "7 jours",
        "audio",
        "pas enregistré",
    ):
        assert required_disclosure in disclosure
    assert "flow" not in (bundle / "manifest.yaml").read_text(encoding="utf-8").casefold()
    assert "tool" not in (bundle / "manifest.yaml").read_text(encoding="utf-8").casefold()
