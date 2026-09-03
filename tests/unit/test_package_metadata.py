from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUNTIME_DEPENDENCIES = [
    "aiosqlite==0.22.1",
    "cryptography==50.0.0",
    "fastapi==0.141.1",
    "httpx==0.28.1",
    "loguru==0.7.3",
    "opentelemetry-api==1.44.0",
    "opentelemetry-exporter-otlp-proto-http==1.44.0",
    "opentelemetry-sdk==1.44.0",
    "pipecat-ai[openai,openrouter,silero,websocket]==1.7.0",
    "psycopg[binary,pool]==3.3.4",
    "pydantic==2.13.4",
    "pynacl==1.6.2",
    "pyyaml==6.0.3",
    "requests==2.34.2",
    "telnyx==4.176.0",
    "uvicorn[standard]==0.52.4",
]
EXPECTED_DEVELOPMENT_DEPENDENCIES = [
    "jsonschema==4.25.1",
    "mypy==2.3.1",
    "pipecat-ai[evals]==1.7.0",
    "pytest==9.1.1",
    "pytest-asyncio==1.4.0",
    "respx==0.23.1",
    "ruff==0.16.4",
]
EXPECTED_PRE_IMAGE_ARTIFACTS = [
    "dist/runtime-contract.json",
    "dist/qualified-deployment-profile-v1.schema.json",
    "dist/qualification-candidate-profile-v1.schema.json",
    "dist/qualification-override-v1.schema.json",
    "dist/agent-a-bundle.tar.gz",
    "dist/agent-a-bundle-v1.manifest.json",
]
EXPECTED_RELEASE_HANDOFF_ARTIFACTS = [
    *EXPECTED_PRE_IMAGE_ARTIFACTS,
    "dist/image-reference.txt",
    "dist/sbom.spdx.json",
]
CONTAINER_SMOKE_COMMAND = (
    "PROJETV0_CONTAINER_SMOKE=1 uv run pytest "
    "tests/integration/test_container_smoke.py -q"
)


def load_project_metadata() -> dict[str, object]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject:
        return tomllib.load(pyproject)


def assert_packaging_orchestration(workflow: dict[str, object]) -> None:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    packaging = jobs["packaging-image"]
    assert isinstance(packaging, dict)
    steps = packaging["steps"]
    assert isinstance(steps, list)
    shapes = [
        (
            tuple(sorted(step)),
            step.get("name"),
            step.get("uses"),
            step.get("shell"),
            step.get("env"),
            (
                hashlib.sha256(str(step["run"]).encode()).hexdigest()
                if "run" in step
                else None
            ),
        )
        for step in steps
    ]
    assert shapes == [
        (
            ("uses", "with"),
            None,
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            None,
            None,
            None,
        ),
        (
            ("uses", "with"),
            None,
            "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
            None,
            None,
            None,
        ),
        (
            ("run",),
            None,
            None,
            None,
            None,
            "62d92225bdb1d1703f830f5fb734f8cb1771a59d03fadc351f964d7a27cc97ea",
        ),
        (
            ("run",),
            None,
            None,
            None,
            None,
            "cea4f21eea91021fb825d604fdf96a32a152d60e350b2d1a9bc87fc1010e5204",
        ),
        (
            ("name", "run", "shell"),
            "Export and prove runtime and bundle parity",
            None,
            "bash",
            None,
            "29c0278468b1eca6fdf9588deda638579e49b42c06ea8c9cae1d280526becd56",
        ),
        (
            ("name", "run", "shell"),
            "Validate the six pre-image artifacts",
            None,
            "bash",
            None,
            "efe22a0edbe0bbb223a37fd7f6d11151d1a67cd075ce31003be31f2265198a13",
        ),
        (
            ("name", "uses", "with"),
            "Upload pre-image handoff",
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            None,
            None,
            None,
        ),
        (
            ("name", "run"),
            "Build and run the required offline container smoke",
            None,
            None,
            None,
            "d36650c9d6c9eb4812453b68ddf6e5fd4aaea20e5d62647c57e0bc23a4ac2ef6",
        ),
    ]
    assert steps[0] == {
        "uses": "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "with": {"persist-credentials": "false"},
    }
    assert steps[1] == {
        "uses": "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
        "with": {"version": "0.12.4", "python-version": "3.13.15"},
    }
    assert steps[6] == {
        "name": "Upload pre-image handoff",
        "uses": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "with": {
            "name": "voice-pre-image-${{ github.sha }}",
            "path": "\n".join(EXPECTED_PRE_IMAGE_ARTIFACTS) + "\n",
            "if-no-files-found": "error",
            "retention-days": "14",
        },
    }
    assert steps[7]["run"] == CONTAINER_SMOKE_COMMAND
    assert sum(CONTAINER_SMOKE_COMMAND in str(step.get("run", "")) for step in steps) == 1
    assert all(
        "test_container_smoke.py" not in str(step.get("run", ""))
        for step in steps[:7]
    )


def assert_handoff_orchestration(workflow: dict[str, object]) -> None:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    handoff = jobs["handoff"]
    assert isinstance(handoff, dict)
    steps = handoff["steps"]
    assert isinstance(steps, list)
    shapes = [
        (
            tuple(sorted(step)),
            step.get("name"),
            step.get("uses"),
            step.get("shell"),
            step.get("env"),
            (
                hashlib.sha256(str(step["run"]).encode()).hexdigest()
                if "run" in step
                else None
            ),
        )
        for step in steps
    ]
    assert shapes == [
        (
            ("uses", "with"),
            None,
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            None,
            None,
            None,
        ),
        (
            ("uses", "with"),
            None,
            "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
            None,
            None,
            None,
        ),
        (
            ("name", "run", "shell"),
            "Regenerate the six pre-image artifacts",
            None,
            "bash",
            None,
            "952bacacd4b9e675805ce83055a91d3bd8e6551bfcc2930c3df1c696a5559434",
        ),
        (
            ("env", "name", "run", "shell"),
            "Record the immutable image reference",
            None,
            "bash",
            {"IMAGE_REFERENCE": "${{ needs.release.outputs.image-reference }}"},
            "5613c9b0a38345fadcdbc3fe97d02a3ec17ae5e0742475d6ce81474bd01be455",
        ),
        (
            ("name", "uses", "with"),
            "Download the validated SBOM",
            "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
            None,
            None,
            None,
        ),
        (
            ("name", "run", "shell"),
            "Validate the exact release handoff",
            None,
            "bash",
            None,
            "7524bce0c8eab100b79db50353bdecdeecb1590cc233d4939612fbdbb6d38e4a",
        ),
        (
            ("name", "uses", "with"),
            "Upload the immutable release handoff",
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            None,
            None,
            None,
        ),
    ]
    assert steps[0] == {
        "uses": "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "with": {"ref": "${{ github.sha }}", "persist-credentials": "false"},
    }
    assert steps[1] == {
        "uses": "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
        "with": {"version": "0.12.4", "python-version": "3.13.15"},
    }
    assert steps[4] == {
        "name": "Download the validated SBOM",
        "uses": "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "with": {
            "name": "${{ needs.release.outputs.sbom-artifact }}",
            "path": "dist",
        },
    }
    assert steps[6] == {
        "name": "Upload the immutable release handoff",
        "uses": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "with": {
            "name": "voice-release-handoff-${{ inputs.version }}-${{ github.sha }}",
            "path": "\n".join(EXPECTED_RELEASE_HANDOFF_ARTIFACTS) + "\n",
            "if-no-files-found": "error",
            "overwrite": "false",
            "retention-days": "90",
        },
    }
    orchestration = "\n".join(
        f"{step.get('uses', '')}\n{step.get('run', '')}" for step in steps
    ).casefold()
    for forbidden in (
        "docker build",
        "build-push-action",
        "trivy",
        "grype",
        "syft",
        "cosign",
    ):
        assert forbidden not in orchestration


def test_runtime_uses_python_3_13() -> None:
    assert sys.version_info[:2] == (3, 13)


def test_runtime_package_imports() -> None:
    assert importlib.import_module("projetv0_voice") is not None


def test_production_wiring_module_is_packaged() -> None:
    assert importlib.import_module("projetv0_voice.production_wiring") is not None


def test_module_entrypoint_rejects_forbidden_environment_before_heavy_imports() -> None:
    environment = dict(os.environ)
    environment["VOICE_TELNYX_API_KEY"] = "SYNTHETIC-SENTINEL"

    completed = subprocess.run(
        [sys.executable, "-m", "projetv0_voice.server"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    rendered = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "runtime_environment_forbidden" in rendered
    assert "SYNTHETIC-SENTINEL" not in rendered
    assert "Pipecat" not in rendered
    assert "uvicorn" not in rendered.casefold()


def test_first_heavy_import_observes_exact_uvicorn_stdlib_floors() -> None:
    code = r'''
import builtins
import json
import logging

import projetv0_voice.server as server

server.capture_runtime_environment = lambda _mapping: object()
server.parse_runtime_settings = lambda _capture: object()
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "projetv0_voice.dependency_logging":
        observed = {}
        for logger_name in ("uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
            logger = logging.getLogger(logger_name)
            observed[logger_name] = {
                "handlers": [type(handler).__name__ for handler in logger.handlers],
                "propagate": logger.propagate,
                "disabled": logger.disabled,
                "level": logger.level,
            }
        print(json.dumps(observed, sort_keys=True))
        raise SystemExit(0)
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
server.main({})
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        name: {
            "handlers": ["NullHandler"],
            "propagate": False,
            "disabled": True,
            "level": logging.CRITICAL + 1,
        }
        for name in ("uvicorn.error", "uvicorn.access", "uvicorn.asgi")
    }


def test_pipecat_is_pinned_to_1_7_0() -> None:
    assert version("pipecat-ai") == "1.7.0"


def test_telnyx_is_pinned_to_4_176_0() -> None:
    assert version("telnyx") == "4.176.0"


def test_direct_runtime_dependency_boundary_matches_the_plan() -> None:
    project_metadata = load_project_metadata()
    project = project_metadata["project"]

    assert isinstance(project, dict)
    assert project["dependencies"] == EXPECTED_RUNTIME_DEPENDENCIES


def test_direct_development_dependency_boundary_matches_the_plan() -> None:
    project_metadata = load_project_metadata()
    dependency_groups = project_metadata["dependency-groups"]

    assert isinstance(dependency_groups, dict)
    assert dependency_groups["dev"] == EXPECTED_DEVELOPMENT_DEPENDENCIES


def test_build_backend_requirement_is_exactly_pinned() -> None:
    project_metadata = load_project_metadata()
    build_system = project_metadata["build-system"]

    assert isinstance(build_system, dict)
    assert build_system["requires"] == ["hatchling==1.27.0"]


def test_flows_imports_from_pipecat_main_package() -> None:
    assert importlib.import_module("pipecat.flows") is not None


def test_standalone_flows_distribution_is_not_declared_or_locked() -> None:
    dependency_files = [REPOSITORY_ROOT / "pyproject.toml", REPOSITORY_ROOT / "uv.lock"]
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in dependency_files if path.exists()
    )

    assert "pipecat-ai-flows" not in content.casefold()


def test_production_source_does_not_use_the_development_runner() -> None:
    source_root = REPOSITORY_ROOT / "src"
    production_source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_root.rglob("*.py")
    )

    assert "pipecat.runner.run" not in production_source


def test_voice_runtime_ci_composes_pinned_shared_and_linux_runtime_gates() -> None:
    workflow_path = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow_path.is_file()
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["name"] == "Voice runtime CI"
    assert workflow["on"] == {
        "push": {"branches": ["main", "z/**"]},
        "pull_request": "",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {
        "shared-repository-ci",
        "python-linux",
        "packaging-image",
    }
    assert workflow["jobs"]["shared-repository-ci"] == {
        "uses": (
            "LouisVannobel/projetV0-pipelines/.github/workflows/"
            "reusable-repository-ci.yml@399df8dcb93a28734269ad11b63e847896684487"
        )
    }
    assert_packaging_orchestration(workflow)
    linux = workflow["jobs"]["python-linux"]
    assert linux["runs-on"] == "ubuntu-24.04"
    assert linux["steps"] == [
        {
            "uses": "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "with": {"persist-credentials": "false"},
        },
        {
            "uses": "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
            "with": {"version": "0.12.4", "python-version": "3.13.15"},
        },
        {"run": "uv sync --all-groups --frozen"},
        {"run": "uv run pytest -q"},
        {
            "name": "Run privileged Linux descriptor cases",
            "run": (
                'sudo env "PATH=$PATH" PROJETV0_PRIVILEGED_FILES_GATE=1 '
                'uv run pytest tests/unit/test_runtime_config.py '
                'tests/unit/test_qualified_profile.py '
                'tests/unit/test_production_wiring.py -q -k '
                '"linux_kernel or fifo or socket or grows"'
            ),
        },
        {"run": "uv run ruff check ."},
        {"run": "uv run mypy --strict src"},
    ]

    packaging = workflow["jobs"]["packaging-image"]
    assert packaging["runs-on"] == "ubuntu-24.04"
    assert packaging.get("permissions") == {"contents": "read"}
    steps = packaging["steps"]
    assert steps[:4] == [
        {
            "uses": "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "with": {"persist-credentials": "false"},
        },
        {
            "uses": "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
            "with": {"version": "0.12.4", "python-version": "3.13.15"},
        },
        {"run": "uv lock --check"},
        {"run": "uv sync --all-groups --frozen"},
    ]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "scripts/export_runtime_contract.py" in commands
    assert "tests/contract/test_runtime_contract.py" in commands
    assert (
        "test_bundle_consumer_matches_literal_task11_golden_without_a_producer"
        in commands
    )
    assert "test_linux_kernel_accepts_real_task11_export_without_importing_producer" in commands
    assert (
        "PROJETV0_CONTAINER_SMOKE=1 uv run pytest "
        "tests/integration/test_container_smoke.py -q"
    ) in commands

    uploads = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    assert uploads[0]["uses"] == (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    )
    assert uploads[0]["with"]["path"].splitlines() == EXPECTED_PRE_IMAGE_ARTIFACTS
    assert uploads[0]["with"]["if-no-files-found"] == "error"


def test_manual_release_builds_once_on_default_main_and_assembles_handoff() -> None:
    workflow_path = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"
    assert workflow_path.is_file()
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["on"] == {
        "workflow_dispatch": {
            "inputs": {
                "version": {
                    "description": "Release version",
                    "required": "true",
                    "type": "string",
                }
            }
        }
    }
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"release", "handoff"}

    release = workflow["jobs"]["release"]
    assert release == {
        "if": (
            "github.ref == 'refs/heads/main' && "
            "github.event.repository.default_branch == 'main'"
        ),
        "permissions": {"contents": "read", "packages": "write"},
        "uses": (
            "LouisVannobel/projetV0-pipelines/.github/workflows/"
            "reusable-oci-release.yml@97cf6d2c5348f202c232fd872c4d4592d430297b"
        ),
        "with": {
            "registry-username": "${{ github.actor }}",
            "image": "ghcr.io/louisvannobel/projetv0-voice",
            "version": "${{ inputs.version }}",
            "docker-context": ".",
            "dockerfile": "Dockerfile",
            "platforms": "linux/amd64",
        },
        "secrets": {
            "registry-password": "${{ secrets.GITHUB_TOKEN }}",
            "registry-read-password": "${{ secrets.GITHUB_TOKEN }}",
        },
    }

    assert_handoff_orchestration(workflow)
    handoff = workflow["jobs"]["handoff"]
    assert handoff["needs"] == "release"
    assert handoff["runs-on"] == "ubuntu-24.04"
    assert handoff["permissions"] == {"contents": "read"}
    steps = handoff["steps"]
    assert steps[0] == {
        "uses": "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "with": {"ref": "${{ github.sha }}", "persist-credentials": "false"},
    }
    assert steps[1] == {
        "uses": "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
        "with": {"version": "0.12.4", "python-version": "3.13.15"},
    }
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "uv lock --check" in commands
    assert "uv sync --all-groups --frozen" in commands
    assert "scripts/export_runtime_contract.py" in commands
    assert "dist/image-reference.txt" in commands
    assert "docker build" not in commands
    assert "trivy" not in commands.casefold()

    downloads = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert len(downloads) == 1
    assert downloads[0]["with"] == {
        "name": "${{ needs.release.outputs.sbom-artifact }}",
        "path": "dist",
    }
    uploads = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    assert uploads[0]["with"]["path"].splitlines() == EXPECTED_RELEASE_HANDOFF_ARTIFACTS
    assert uploads[0]["with"]["if-no-files-found"] == "error"
    assert uploads[0]["with"]["overwrite"] == "false"


def test_renovate_keeps_pinned_compatibility_surfaces_separate() -> None:
    config_path = REPOSITORY_ROOT / "renovate.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["$schema"] == "https://docs.renovatebot.com/renovate-schema.json"
    assert config["extends"] == ["config:recommended", "schedule:weekly"]
    assert config["automerge"] is False
    assert config["pinDigests"] is True
    assert config["lockFileMaintenance"] == {"enabled": True}
    assert config["packageRules"] == [
        {
            "description": "Pin declared Python dependencies",
            "matchManagers": ["pep621"],
            "matchDepTypes": [
                "project.dependencies",
                "dependency-groups",
                "build-system.requires",
            ],
            "rangeStrategy": "pin",
        },
        {
            "description": "Group the Pipecat compatibility surface",
            "matchManagers": ["pep621"],
            "matchPackageNames": ["pipecat-ai"],
            "groupName": "Pipecat compatibility surface",
        },
        {
            "description": "Keep the Python base separate",
            "matchPackageNames": ["python"],
            "groupName": "Python base",
        },
        {
            "description": "Keep Telnyx separate",
            "matchManagers": ["pep621"],
            "matchPackageNames": ["telnyx"],
            "groupName": "Telnyx SDK",
        },
    ]


def test_task11_docs_record_artifact_release_and_live_boundaries() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (REPOSITORY_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    combined = readme + "\n" + agents

    for required in (
        "six pre-image artifacts",
        "eight-file release handoff",
        "PROJETV0_CONTAINER_SMOKE=1",
        "main-only manual release",
        "secrets.GITHUB_TOKEN",
        "pipecat-ai[cli]==1.7.0",
        "outside the lock, image, and CI",
        "No live key or provider call belongs to Task 11",
        "Telnyx webhook public key",
        "PostgreSQL DSN",
        "AEAD keyring",
        "Call Control connection ID",
        "benchmark from/DID",
        "WSS/Funnel URL",
        "OTLP endpoint",
        "candidate and qualified profiles",
        "infrastructure readiness",
    ):
        assert required in combined


def test_packaging_contract_rejects_a_premature_duplicate_smoke_step() -> None:
    workflow = yaml.load(
        (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        ),
        Loader=yaml.BaseLoader,
    )
    mutated = copy.deepcopy(workflow)
    mutated["jobs"]["packaging-image"]["steps"].insert(
        2,
        {"name": "Premature container smoke", "run": CONTAINER_SMOKE_COMMAND},
    )

    with pytest.raises(AssertionError):
        assert_packaging_orchestration(mutated)


@pytest.mark.parametrize("mutation", ["extra-action", "alternate-run"])
def test_handoff_contract_rejects_extra_actions_and_alternate_runs(mutation: str) -> None:
    workflow = yaml.load(
        (REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        ),
        Loader=yaml.BaseLoader,
    )
    mutated = copy.deepcopy(workflow)
    steps = mutated["jobs"]["handoff"]["steps"]
    if mutation == "extra-action":
        steps.insert(
            -1,
            {
                "name": "Rebuild image",
                "uses": "docker/build-push-action@0000000000000000000000000000000000000000",
            },
        )
    else:
        record = next(
            step for step in steps if step.get("name") == "Record the immutable image reference"
        )
        record["run"] = "printf '%s\\n' \"$IMAGE_REFERENCE\" > dist/alternate.txt"

    with pytest.raises(AssertionError):
        assert_handoff_orchestration(mutated)
