from __future__ import annotations

import importlib
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUNTIME_DEPENDENCIES = [
    "aiosqlite==0.22.1",
    "cryptography==50.0.0",
    "fastapi==0.141.1",
    "httpx==0.28.1",
    "opentelemetry-api==1.44.0",
    "opentelemetry-exporter-otlp-proto-http==1.44.0",
    "opentelemetry-sdk==1.44.0",
    "pipecat-ai[openai,openrouter,silero,websocket]==1.7.0",
    "psycopg[binary,pool]==3.3.4",
    "pydantic==2.13.4",
    "pynacl==1.6.2",
    "pyyaml==6.0.3",
    "telnyx==4.176.0",
    "uvicorn[standard]==0.52.4",
]
EXPECTED_DEVELOPMENT_DEPENDENCIES = [
    "mypy==2.3.1",
    "pipecat-ai[evals]==1.7.0",
    "pytest==9.1.1",
    "pytest-asyncio==1.4.0",
    "respx==0.23.1",
    "ruff==0.16.4",
]


def load_project_metadata() -> dict[str, object]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject:
        return tomllib.load(pyproject)


def test_runtime_uses_python_3_13() -> None:
    assert sys.version_info[:2] == (3, 13)


def test_runtime_package_imports() -> None:
    assert importlib.import_module("projetv0_voice") is not None


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
