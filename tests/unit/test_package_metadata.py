from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

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


def load_project_metadata() -> dict[str, object]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject:
        return tomllib.load(pyproject)


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
