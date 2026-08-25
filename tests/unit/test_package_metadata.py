from __future__ import annotations

import importlib
import sys
from importlib.metadata import version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_uses_python_3_13() -> None:
    assert sys.version_info[:2] == (3, 13)


def test_runtime_package_imports() -> None:
    assert importlib.import_module("projetv0_voice") is not None


def test_pipecat_is_pinned_to_1_7_0() -> None:
    assert version("pipecat-ai") == "1.7.0"


def test_telnyx_is_pinned_to_4_176_0() -> None:
    assert version("telnyx") == "4.176.0"


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
