from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
PYTHON_IMAGE = (
    "python:3.13.15-slim-bookworm"
    "@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e"
)
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.12.4"
    "@sha256:d0a6eca6c669dc7e9c51218707b8438a3d30402733d739dcc00adb3e213e8f5c"
)
POSTGRES_IMAGE = (
    "postgres:16.15-bookworm"
    "@sha256:bb3e1a57e5407e0a5280b4211980a5e537f4abd234a87014ac979849a78dd825"
)
SMOKE_ENABLED = os.environ.get("PROJETV0_CONTAINER_SMOKE") == "1"
TOKENIZER_FILES = (
    "punkt_tab/english/collocations.tab",
    "punkt_tab/english/sent_starters.txt",
    "punkt_tab/english/abbrev_types.txt",
    "punkt_tab/english/ortho_context.tab",
    "punkt_tab/french/collocations.tab",
    "punkt_tab/french/sent_starters.txt",
    "punkt_tab/french/abbrev_types.txt",
    "punkt_tab/french/ortho_context.tab",
)


@pytest.fixture
def tokenizer_preparation() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "prepare_tokenizers", REPO_ROOT / "scripts/prepare_tokenizers.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tokenizer_archive(
    entries: list[tuple[str, bytes, int]],
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data, mode in entries:
            entry = zipfile.ZipInfo(name)
            entry.create_system = 3
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = mode << 16
            archive.writestr(entry, data)
    return stream.getvalue()


def test_tokenizer_preparation_extracts_only_required_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer_preparation: ModuleType
) -> None:
    entries = [(name, b"fixture", stat.S_IFREG | 0o600) for name in TOKENIZER_FILES]
    entries.append(("punkt_tab/german/abbrev_types.txt", b"unused", stat.S_IFREG | 0o600))
    archive = _tokenizer_archive(entries)
    monkeypatch.setattr(
        tokenizer_preparation, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest()
    )
    destination = tmp_path / "nltk_data"

    tokenizer_preparation.extract_tokenizers(archive, destination)

    assert sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ) == sorted(f"tokenizers/{name}" for name in TOKENIZER_FILES)
    assert all(path.read_bytes() == b"fixture" for path in destination.rglob("*") if path.is_file())


@pytest.mark.parametrize(
    "fault", ["checksum", "missing", "duplicate", "symlink", "archive_bound", "data_bound"]
)
def test_tokenizer_preparation_refuses_invalid_archives_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_preparation: ModuleType,
    fault: str,
) -> None:
    entries = [(name, b"fixture", stat.S_IFREG | 0o600) for name in TOKENIZER_FILES]
    if fault == "missing":
        entries.pop()
    elif fault == "duplicate":
        entries.append(entries[0])
    elif fault == "symlink":
        entries[0] = (entries[0][0], b"/outside", stat.S_IFLNK | 0o777)
    elif fault == "data_bound":
        entries = [(name, b"x" * 1024, mode) for name, _, mode in entries]
    if fault == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive = _tokenizer_archive(entries)
    else:
        archive = _tokenizer_archive(entries)
    if fault == "archive_bound":
        archive = b"x" * 4097
    monkeypatch.setattr(tokenizer_preparation, "MAX_BYTES", 4096)
    monkeypatch.setattr(
        tokenizer_preparation,
        "ARCHIVE_SHA256",
        "0" * 64 if fault == "checksum" else hashlib.sha256(archive).hexdigest(),
    )
    destination = tmp_path / "nltk_data"

    with pytest.raises(ValueError):
        tokenizer_preparation.extract_tokenizers(archive, destination)

    assert not destination.exists()


def test_tokenizer_preparation_refuses_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer_preparation: ModuleType
) -> None:
    archive = _tokenizer_archive(
        [(name, b"fixture", stat.S_IFREG | 0o600) for name in TOKENIZER_FILES]
    )
    monkeypatch.setattr(
        tokenizer_preparation, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest()
    )
    destination = tmp_path / "nltk_data"
    destination.mkdir()
    sentinel = destination / "preserve"
    sentinel.write_bytes(b"owned")

    with pytest.raises(ValueError):
        tokenizer_preparation.extract_tokenizers(archive, destination)

    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"owned"


def _run(
    command: list[str],
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        timeout=timeout,
    )


def _docker(*arguments: str, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    return _run(["docker", *arguments], **kwargs)


def _output(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", errors="replace")


def _assert_image_sentence_tokenization(image: str) -> None:
    program = """
import os
import resource
import socket
import stat
from pathlib import Path

import nltk

assert (os.getuid(), os.getgid()) == (10001, 10001)
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)

def reject_network(*args, **kwargs):
    raise AssertionError("sentence tokenization attempted runtime network access")

socket.socket.connect = reject_network
socket.socket.connect_ex = reject_network
nltk.download = reject_network

for language in ("english", "french"):
    nltk.data.find(f"tokenizers/punkt_tab/{language}/")

root = Path("/opt/projetv0-voice/nltk_data")
assert os.environ.get("NLTK_DATA") == str(root)
expected = {
    f"tokenizers/punkt_tab/{language}/{name}"
    for language in ("english", "french")
    for name in ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab")
}
paths = [root, *root.rglob("*")]
assert {str(path.relative_to(root)) for path in paths if path.is_file()} == expected
for path in paths:
    metadata = path.lstat()
    assert not path.is_symlink()
    assert (metadata.st_uid, metadata.st_gid) == (0, 10001)
    assert stat.S_IMODE(metadata.st_mode) == (0o550 if path.is_dir() else 0o440)

from pipecat.utils.string import match_endofsentence

for language, text, expected_sentences in (
    ("english", "Hello. I can help you.", ["Hello.", "I can help you."]),
    ("french", "Bonjour. Je peux vous aider.", ["Bonjour.", "Je peux vous aider."]),
):
    assert nltk.sent_tokenize(text, language=language) == expected_sentences
    sentences = []
    while text:
        boundary = match_endofsentence(text)
        assert boundary > 0
        sentences.append(text[:boundary])
        text = text[boundary:].lstrip()
    assert sentences == expected_sentences
print("native sentence tokenization passed offline")
"""
    result = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
        "10001:10001",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--ulimit",
        "core=0:0",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--pids-limit",
        "64",
        "--entrypoint",
        "python",
        image,
        "-I",
        "-B",
        "-c",
        program,
        timeout=30,
    )
    assert "native sentence tokenization passed offline" in _output(result)


def _seed_volume(
    image: str,
    volume: str,
    files: dict[str, bytes],
    *,
    uid: int = 0,
    gid: int = 10001,
    directory_mode: int = 0o550,
    file_mode: int = 0o440,
) -> None:
    payload = {
        "files": {
            name: base64.b64encode(content).decode("ascii") for name, content in files.items()
        },
        "uid": uid,
        "gid": gid,
        "directory_mode": directory_mode,
        "file_mode": file_mode,
    }
    program = (
        "import base64,json,os,sys;"
        "from pathlib import Path;"
        "p=json.load(sys.stdin);r=Path('/seed');"
        "[(r/n).parent.mkdir(parents=True,exist_ok=True) for n in p['files']];"
        "[(r/n).write_bytes(base64.b64decode(v)) for n,v in p['files'].items()];"
        "[(os.chown(x,p['uid'],p['gid']),os.chmod(x,p['file_mode'])) "
        "for x in [r/n for n in p['files']]];"
        "[(os.chown(x,p['uid'],p['gid']),os.chmod(x,p['directory_mode'])) "
        "for x in sorted([x for x in r.rglob('*') if x.is_dir()],reverse=True)];"
        "os.chown(r,p['uid'],p['gid']);os.chmod(r,p['directory_mode'])"
    )
    _docker(
        "run",
        "--rm",
        "-i",
        "--user",
        "0:0",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev",
        "--mount",
        f"type=volume,src={volume},dst=/seed",
        "--entrypoint",
        "python",
        image,
        "-c",
        program,
        input_bytes=json.dumps(payload).encode("utf-8"),
    )


def _probe_http(image: str, network: str, url: str) -> tuple[int, bytes]:
    program = (
        "import base64,sys,urllib.error,urllib.request;"
        "u=sys.argv[1];"
        "\ntry:\n r=urllib.request.urlopen(u,timeout=2);s=r.status;b=r.read()"
        "\nexcept urllib.error.HTTPError as e:\n s=e.code;b=e.read()"
        "\nprint(str(s)+'|'+base64.b64encode(b).decode())"
    )
    result = _docker(
        "run",
        "--rm",
        "--network",
        network,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev",
        "--entrypoint",
        "python",
        image,
        "-c",
        program,
        url,
        timeout=20,
    )
    status, encoded = _output(result).strip().rsplit("|", 1)
    return int(status), base64.b64decode(encoded)


def _wait_ready(image: str, network: str) -> bytes:
    deadline = time.monotonic() + 30.0
    last = "runtime did not answer"
    while time.monotonic() < deadline:
        try:
            status, body = _probe_http(
                image,
                network,
                "http://voice-runtime:8080/health/ready",
            )
            if status == 200:
                return body
            last = f"status={status} body={body!r}"
        except (subprocess.SubprocessError, ValueError) as error:
            last = str(error)
        time.sleep(0.25)
    raise AssertionError(last)


def _volume_file_size(image: str, volume: str, path: str) -> int:
    program = (
        "import sys;from pathlib import Path;"
        "p=Path('/capture')/sys.argv[1];print(p.stat().st_size if p.exists() else -1)"
    )
    result = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--mount",
        f"type=volume,src={volume},dst=/capture,readonly",
        "--entrypoint",
        "python",
        image,
        "-c",
        program,
        path,
        timeout=10,
    )
    return int(_output(result).strip())


def _write_volume_marker(image: str, volume: str, path: str) -> None:
    program = "import sys;from pathlib import Path;Path('/capture',sys.argv[1]).write_text('1')"
    _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--mount",
        f"type=volume,src={volume},dst=/capture",
        "--entrypoint",
        "python",
        image,
        "-c",
        program,
        path,
        timeout=10,
    )


def _assert_container_running(container: str, label: str) -> None:
    state = _docker(
        "inspect",
        "--format",
        "{{.State.Running}}",
        container,
        check=False,
        timeout=10,
    )
    if state.returncode == 0 and _output(state).strip() == "true":
        return
    logs = _docker("logs", container, check=False, timeout=10)
    raise AssertionError(f"{label} exited before the smoke completed:\n{_output(logs)}")


def _wait_for_volume_marker(
    image: str,
    volume: str,
    path: str,
    container: str,
    label: str,
) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        marker_size = _volume_file_size(image, volume, path)
        _assert_container_running(container, label)
        if marker_size > 0:
            return
        time.sleep(0.1)
    logs = _docker("logs", container, check=False, timeout=10)
    raise AssertionError(f"{label} did not publish its readiness marker:\n{_output(logs)}")


def test_dockerfile_generates_the_contract_without_a_dist_context() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert f"ARG PYTHON_IMAGE={PYTHON_IMAGE}" in source
    assert f"ARG UV_IMAGE={UV_IMAGE}" in source
    assert source.count("FROM ${UV_IMAGE}") == 1
    assert source.count("FROM ${PYTHON_IMAGE}") == 2
    assert "COPY --from=uv /uv /uvx /bin/" in source
    assert "uv lock --check" in source
    assert "uv sync --locked --no-dev --no-editable" in source
    assert "apt-get install -y --no-install-recommends libgomp1" in source
    assert "rm -rf /var/lib/apt/lists/*" in source
    assert "COPY --from=builder /opt/projetv0-voice/.venv" in source
    assert "COPY scripts/export_runtime_contract.py ./scripts/export_runtime_contract.py" in source
    assert "COPY scripts/prepare_tokenizers.py ./scripts/prepare_tokenizers.py" in source
    assert "COPY agents/agent-a/ ./agents/agent-a/" in source
    assert "COPY deployment-profiles/ ./deployment-profiles/" in source
    assert "python scripts/export_runtime_contract.py" in source
    assert "--output-dir /opt/projetv0-voice/build-artifacts" in source
    assert (
        "COPY --from=builder --chown=0:10001 --chmod=0440 "
        "/opt/projetv0-voice/build-artifacts/runtime-contract.json ./runtime-contract.json"
        in source
    )
    assert "dist/runtime-contract.json" not in source
    assert "USER 10001:10001" in source
    assert 'ENTRYPOINT ["python", "-m", "projetv0_voice.server"]' in source
    assert "HEALTHCHECK" not in source


def test_dockerignore_is_a_minimal_allowlist() -> None:
    rules = [
        line
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert rules == [
        "**",
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!src/",
        "!src/**",
        "!scripts/",
        "!scripts/export_runtime_contract.py",
        "!scripts/prepare_tokenizers.py",
        "!agents/",
        "!agents/agent-a/",
        "!agents/agent-a/**",
        "!deployment-profiles/",
        "!deployment-profiles/qualified-v1.schema.json",
        "!deployment-profiles/qualification-candidate-v1.schema.json",
        "!deployment-profiles/qualification-override-v1.schema.json",
    ]


def test_container_smoke_database_image_is_immutable() -> None:
    assert POSTGRES_IMAGE == (
        "postgres:16.15-bookworm"
        "@sha256:bb3e1a57e5407e0a5280b4211980a5e537f4abd234a87014ac979849a78dd825"
    )


@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set PROJETV0_CONTAINER_SMOKE=1 for the required Docker image gate",
)
def test_container_smoke(tmp_path: Path) -> None:
    docker = shutil.which("docker")
    assert docker is not None, "Docker is required when PROJETV0_CONTAINER_SMOKE=1"
    info = _docker("info", check=False, timeout=30)
    assert info.returncode == 0, (
        "Docker daemon is required when PROJETV0_CONTAINER_SMOKE=1:\n" + _output(info)
    )

    suffix = uuid.uuid4().hex[:12]
    image = f"projetv0-voice-smoke:{suffix}"
    network = f"projetv0-voice-smoke-{suffix}"
    containers = {
        "postgres": f"projetv0-postgres-{suffix}",
        "otlp": f"projetv0-otlp-{suffix}",
        "tripwire": f"projetv0-provider-tripwire-{suffix}",
        "runtime": f"projetv0-runtime-{suffix}",
        "observer": f"projetv0-ready-observer-{suffix}",
    }
    volumes = {
        name: f"projetv0-{name}-{suffix}"
        for name in ("secrets", "agent", "profile", "state", "pgdata", "capture")
    }
    response_bodies: list[bytes] = []
    fake_telnyx = f"TEST-ONLY-NO-LIVE-TELNYX-{suffix}"
    fake_openrouter = f"TEST-ONLY-NO-LIVE-OPENROUTER-{suffix}"
    fake_postgres_password = f"TEST-ONLY-POSTGRES-{suffix}"
    sentinels = (fake_telnyx, fake_openrouter, fake_postgres_password)

    try:
        context = tmp_path / "context"
        context.mkdir()
        for name in ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "README.md"):
            shutil.copy2(REPO_ROOT / name, context / name)
        shutil.copytree(REPO_ROOT / "src", context / "src")
        (context / "scripts").mkdir()
        for script in ("export_runtime_contract.py", "prepare_tokenizers.py"):
            shutil.copy2(REPO_ROOT / "scripts" / script, context / "scripts" / script)
        shutil.copytree(REPO_ROOT / "agents/agent-a", context / "agents/agent-a")
        shutil.copytree(REPO_ROOT / "deployment-profiles", context / "deployment-profiles")
        output_dir = tmp_path / "host-artifacts"
        _run(
            [
                sys.executable,
                str(context / "scripts" / "export_runtime_contract.py"),
                "--repo-root",
                str(context),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert not (context / "dist").exists()
        _docker("build", "--pull", "--tag", image, str(context), timeout=900)

        inspected = json.loads(_output(_docker("image", "inspect", image)))[0]
        config = inspected["Config"]
        assert config["User"] == "10001:10001"
        assert config["Entrypoint"] == ["python", "-m", "projetv0_voice.server"]
        assert config.get("Healthcheck") is None
        assert config["ExposedPorts"] == {"8080/tcp": {}}
        assert not any("VOICE_" in value for value in config.get("Env", []))

        _assert_image_sentence_tokenization(image)

        import_program = (
            "import importlib.util,os,shutil;"
            "from pathlib import Path;"
            "import onnxruntime;"
            "from pipecat.audio.vad.silero import SileroVADAnalyzer;"
            "SileroVADAnalyzer();"
            "assert (os.getuid(),os.getgid())==(10001,10001);"
            "assert importlib.util.find_spec('pip') is None;"
            "assert importlib.util.find_spec('ensurepip') is None;"
            "assert shutil.which('pip') is None;"
            "assert shutil.which('pip3') is None;"
            "assert shutil.which('pip3.13') is None;"
            "assert importlib.util.find_spec('pytest') is None;"
            "assert importlib.util.find_spec('mypy') is None;"
            "assert importlib.util.find_spec('ruff') is None;"
            "assert shutil.which('uv') is None;"
            "assert shutil.which('gcc') is None;"
            "assert not Path('/opt/projetv0-voice/scripts').exists();"
            "assert not Path('/opt/projetv0-voice/agents').exists();"
            "assert not Path('/opt/projetv0-voice/deployment-profiles').exists();"
            "assert not Path('/opt/projetv0-voice/build-artifacts').exists()"
        )
        _docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--entrypoint",
            "python",
            image,
            "-c",
            import_program,
        )

        global_installer_program = (
            "import importlib.util,shutil;"
            "from pathlib import Path;"
            "site_packages=Path('/usr/local/lib/python3.13/site-packages');"
            "assert importlib.util.find_spec('pip') is None;"
            "assert importlib.util.find_spec('ensurepip') is None;"
            "assert shutil.which('pip') is None;"
            "assert shutil.which('pip3') is None;"
            "assert shutil.which('pip3.13') is None;"
            "assert not tuple(site_packages.glob('pip-*.dist-info'));"
            "paths=("
            "'/usr/local/bin/pip',"
            "'/usr/local/bin/pip3',"
            "'/usr/local/bin/pip3.13',"
            "'/usr/local/lib/python3.13/site-packages/pip',"
            "'/usr/local/lib/python3.13/ensurepip'"
            ");"
            "assert all(not Path(path).exists() and not Path(path).is_symlink() for path in paths),"
            "[path for path in paths if Path(path).exists() or Path(path).is_symlink()]"
        )
        _docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--entrypoint",
            "/usr/local/bin/python3",
            image,
            "-I",
            "-c",
            global_installer_program,
        )

        _docker("network", "create", "--internal", network)
        for volume in volumes.values():
            _docker("volume", "create", volume)

        contract = (output_dir / "runtime-contract.json").read_bytes()
        contract_sha256 = hashlib.sha256(contract).hexdigest()
        manifest = json.loads(
            (output_dir / "agent-a-bundle-v1.manifest.json").read_text(encoding="utf-8")
        )
        bundle_sha256 = manifest["bundle_sha256"]
        profile = json.loads(
            (REPO_ROOT / "tests/fixtures/qualified-deployment-profile-v1.json").read_text(
                encoding="utf-8"
            )
        )
        image_reference = f"example.invalid/projetv0-voice@sha256:{'d' * 64}"
        inference_bytes = json.dumps(
            profile["inference"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        inference_sha256 = hashlib.sha256(inference_bytes).hexdigest()
        profile.update(
            runtime_contract_sha256=contract_sha256,
            image_digest=image_reference,
            agent_bundle_sha256=bundle_sha256,
            inference_profile_sha256=inference_sha256,
            telnyx_api_key_sha256=hashlib.sha256(fake_telnyx.encode()).hexdigest(),
        )
        profile_bytes = (
            json.dumps(profile, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        agent_root = context / "agents/agent-a"
        agent_files = {
            record["path"]: agent_root.joinpath(*record["path"].split("/")).read_bytes()
            for record in manifest["files"]
        }
        keyring = json.dumps(
            {
                "schema_version": 1,
                "active_version": 1,
                "keys": [{"version": 1, "aes256_key_hex": "11" * 32}],
            },
            separators=(",", ":"),
        ).encode("ascii")
        webhook_key = base64.b64encode(bytes(range(32)))
        postgres_dsn = (f"postgresql://voice:{fake_postgres_password}@postgres:5432/voice").encode(
            "ascii"
        )
        _seed_volume(image, volumes["agent"], agent_files)
        _seed_volume(image, volumes["profile"], {"qualified.json": profile_bytes})
        _seed_volume(
            image,
            volumes["secrets"],
            {
                "aead_keyring_v1.json": keyring,
                "telnyx-api-key": fake_telnyx.encode("ascii"),
                "telnyx-webhook-public-key": webhook_key,
                "openrouter-api-key": fake_openrouter.encode("ascii"),
                "postgres-dsn": postgres_dsn,
            },
        )
        _seed_volume(
            image,
            volumes["state"],
            {},
            uid=10001,
            gid=10001,
            directory_mode=0o700,
            file_mode=0o600,
        )
        _seed_volume(
            image,
            volumes["capture"],
            {},
            uid=10001,
            gid=10001,
            directory_mode=0o700,
            file_mode=0o600,
        )

        _docker("pull", POSTGRES_IMAGE, timeout=600)
        _docker(
            "run",
            "-d",
            "--name",
            containers["postgres"],
            "--network",
            network,
            "--network-alias",
            "postgres",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--tmpfs",
            "/var/run/postgresql:rw,noexec,nosuid,nodev",
            "--mount",
            f"type=volume,src={volumes['pgdata']},dst=/var/lib/postgresql/data",
            "--env",
            "POSTGRES_USER=voice",
            "--env",
            f"POSTGRES_PASSWORD={fake_postgres_password}",
            "--env",
            "POSTGRES_DB=voice",
            POSTGRES_IMAGE,
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            ready = _docker(
                "exec",
                containers["postgres"],
                "pg_isready",
                "-U",
                "voice",
                "-d",
                "voice",
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise AssertionError(_output(_docker("logs", containers["postgres"])))

        otlp_server = (
            "from http.server import BaseHTTPRequestHandler,HTTPServer;"
            "from pathlib import Path;"
            "\nclass H(BaseHTTPRequestHandler):"
            "\n def do_GET(self): self.send_response(200);self.end_headers()"
            "\n def do_POST(self):"
            "\n  b=self.rfile.read(int(self.headers.get('content-length','0')));"
            "Path('/capture/otlp.bin').write_bytes(b);self.send_response(200);self.end_headers()"
            "\n def log_message(self,*args): pass"
            "\nserver=HTTPServer(('0.0.0.0',4318),H)"
            "\nPath('/capture/otlp-ready').write_text('ready')"
            "\nserver.serve_forever()"
        )
        _docker(
            "run",
            "-d",
            "--name",
            containers["otlp"],
            "--network",
            network,
            "--network-alias",
            "otlp",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--mount",
            f"type=volume,src={volumes['capture']},dst=/capture",
            "--entrypoint",
            "python",
            image,
            "-c",
            otlp_server,
        )
        _wait_for_volume_marker(
            image,
            volumes["capture"],
            "otlp-ready",
            containers["otlp"],
            "OTLP capture",
        )

        tripwire = (
            "import selectors,socket;from pathlib import Path;"
            "selector=selectors.DefaultSelector();capture=Path('/capture')"
            "\ndef observe(key):"
            "\n while True:"
            "\n  try: client,_=key.fileobj.accept()"
            "\n  except BlockingIOError: return"
            "\n  capture.joinpath('provider-connected').write_text(str(key.data));client.close()"
            "\nfor port in (443,8443):"
            "\n sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            "sock.setblocking(False);sock.bind(('0.0.0.0',port));sock.listen();"
            "selector.register(sock,selectors.EVENT_READ,port)"
            "\ncapture.joinpath('provider-tripwire-ready').write_text('ready')"
            "\nwhile True:"
            "\n for key,_ in selector.select(.05): observe(key)"
            "\n if capture.joinpath('provider-drain-request').exists():"
            "\n  for key in selector.get_map().values(): observe(key)"
            "\n  capture.joinpath('provider-drain-ack').write_text('drained')"
        )
        _docker(
            "run",
            "-d",
            "--name",
            containers["tripwire"],
            "--network",
            network,
            "--network-alias",
            "openrouter.ai",
            "--network-alias",
            "api.telnyx.com",
            "--network-alias",
            "media.telnyx.com",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--mount",
            f"type=volume,src={volumes['capture']},dst=/capture",
            "--user",
            "0:0",
            "--entrypoint",
            "python",
            image,
            "-c",
            tripwire,
        )
        _wait_for_volume_marker(
            image,
            volumes["capture"],
            "provider-tripwire-ready",
            containers["tripwire"],
            "provider tripwire",
        )

        environment = {
            "VOICE_RUNTIME_MODE": "strict",
            "VOICE_DEPLOYMENT_ID": "voice-agent-a",
            "VOICE_RUNTIME_CONTRACT_PATH": "/opt/projetv0-voice/runtime-contract.json",
            "VOICE_AGENT_BUNDLE_PATH": "/opt/projetv0-voice/agents/active",
            "VOICE_QUALIFIED_PROFILE_PATH": "/run/projetv0/qualified.json",
            "VOICE_KEYRING_PATH": "/run/secrets/aead_keyring_v1.json",
            "VOICE_SQLITE_PATH": "/var/lib/projetv0-voice/voice.sqlite3",
            "VOICE_RUNTIME_CONTRACT_SHA256": contract_sha256,
            "VOICE_IMAGE_DIGEST": image_reference,
            "VOICE_AGENT_BUNDLE_SHA256": bundle_sha256,
            "VOICE_INFERENCE_PROFILE_SHA256": inference_sha256,
            "VOICE_DEPLOYMENT_MAX_CALLS": "10",
            "VOICE_HANDSHAKE_TIMEOUT_SECONDS": "5",
            "VOICE_CALL_IDLE_TIMEOUT_SECONDS": "30",
            "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS": "5",
            "VOICE_PRE_DRAIN_GRACE_SECONDS": "1",
            "VOICE_UVICORN_GRACE_SECONDS": "5",
            "VOICE_SHUTDOWN_GRACE_SECONDS": "15",
            "VOICE_TELNYX_API_KEY_FILE": "/run/secrets/telnyx-api-key",
            "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE": "/run/secrets/telnyx-webhook-public-key",
            "VOICE_OPENROUTER_API_KEY_FILE": "/run/secrets/openrouter-api-key",
            "VOICE_POSTGRES_DSN_FILE": "/run/secrets/postgres-dsn",
            "VOICE_TELNYX_MEDIA_WSS_URL": "wss://media.telnyx.com:8443/v2",
            "VOICE_OTLP_HTTP_ENDPOINT": "http://otlp:4318/v1/metrics",
            "VOICE_BIND_HOST": "0.0.0.0",
            "VOICE_BIND_PORT": "8080",
        }
        run_arguments = [
            "run",
            "-d",
            "--name",
            containers["runtime"],
            "--network",
            network,
            "--network-alias",
            "voice-runtime",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=volume,src={volumes['state']},dst=/var/lib/projetv0-voice",
            "--mount",
            f"type=volume,src={volumes['secrets']},dst=/run/secrets,readonly",
            "--mount",
            (f"type=volume,src={volumes['agent']},dst=/opt/projetv0-voice/agents/active,readonly"),
            "--mount",
            f"type=volume,src={volumes['profile']},dst=/run/projetv0,readonly",
        ]
        for name, value in environment.items():
            run_arguments.extend(("--env", f"{name}={value}"))
        run_arguments.append(image)
        _docker(*run_arguments)

        body = _wait_ready(image, network)
        response_bodies.append(body)
        assert body == b""
        live_status, live_body = _probe_http(
            image, network, "http://voice-runtime:8080/health/live"
        )
        response_bodies.append(live_body)
        assert live_status == 200 and live_body == b""
        for path in ("/metrics", "/docs", "/redoc", "/openapi.json"):
            absent_status, absent_body = _probe_http(
                image, network, f"http://voice-runtime:8080{path}"
            )
            response_bodies.append(absent_body)
            assert absent_status == 404

        runtime_inspect = json.loads(_output(_docker("inspect", containers["runtime"])))[0]
        assert runtime_inspect["HostConfig"]["ReadonlyRootfs"] is True
        assert runtime_inspect["HostConfig"]["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev"}
        mounts = {mount["Destination"]: mount for mount in runtime_inspect["Mounts"]}
        assert mounts["/var/lib/projetv0-voice"]["RW"] is True
        for destination in (
            "/run/secrets",
            "/opt/projetv0-voice/agents/active",
            "/run/projetv0",
        ):
            assert mounts[destination]["RW"] is False

        denied = _docker(
            "exec",
            containers["runtime"],
            "python",
            "-c",
            "from pathlib import Path;Path('/forbidden').write_text('x')",
            check=False,
        )
        assert denied.returncode != 0
        _docker(
            "exec",
            containers["runtime"],
            "python",
            "-c",
            (
                "from pathlib import Path;"
                "Path('/var/lib/projetv0-voice/write-ok').write_text('x');"
                "Path('/tmp/write-ok').write_text('x')"
            ),
        )
        pid_one = _output(
            _docker(
                "exec",
                containers["runtime"],
                "python",
                "-c",
                "from pathlib import Path;print(Path('/proc/1/cmdline').read_bytes())",
            )
        )
        assert "python\\x00-m\\x00projetv0_voice.server\\x00" in pid_one

        observer = (
            "import time,urllib.error,urllib.request;from pathlib import Path;"
            "end=time.monotonic()+5"
            "\nwhile time.monotonic()<end:"
            "\n try:"
            "\n  status=urllib.request.urlopen("
            "'http://voice-runtime:8080/health/ready',timeout=.2).status;"
            "print(status,flush=True)"
            "\n  if status==200: Path('/capture/observer-ready').write_text('200')"
            "\n except urllib.error.HTTPError as e: print(e.code,flush=True)"
            "\n except Exception: print('closed',flush=True);break"
            "\n time.sleep(.05)"
        )
        _docker(
            "run",
            "-d",
            "--name",
            containers["observer"],
            "--network",
            network,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--mount",
            f"type=volume,src={volumes['capture']},dst=/capture",
            "--entrypoint",
            "python",
            image,
            "-c",
            observer,
        )
        _wait_for_volume_marker(
            image,
            volumes["capture"],
            "observer-ready",
            containers["observer"],
            "ready observer",
        )
        _docker("kill", "--signal", "SIGTERM", containers["runtime"])
        exit_code = int(_output(_docker("wait", containers["runtime"], timeout=30)).strip())
        assert exit_code == 0
        _write_volume_marker(image, volumes["capture"], "provider-drain-request")
        _wait_for_volume_marker(
            image,
            volumes["capture"],
            "provider-drain-ack",
            containers["tripwire"],
            "provider tripwire drain",
        )
        _docker("wait", containers["observer"], timeout=10)
        observed = [
            line.strip()
            for line in _output(_docker("logs", containers["observer"])).splitlines()
            if line.strip() in {"200", "503", "closed"}
        ]
        assert "200" in observed
        if "503" in observed:
            first_draining = observed.index("503")
            assert "200" not in observed[first_draining:]

        logs = _output(_docker("logs", containers["runtime"]))
        combined = logs.encode("utf-8", errors="replace") + b"".join(response_bodies)
        assert all(sentinel.encode() not in combined for sentinel in sentinels)
        _assert_container_running(containers["tripwire"], "provider tripwire")
        assert _volume_file_size(image, volumes["capture"], "provider-connected") == -1
        assert _volume_file_size(image, volumes["capture"], "otlp.bin") > 0
    finally:
        for container in containers.values():
            _docker("rm", "--force", container, check=False, timeout=30)
        _docker("network", "rm", network, check=False, timeout=30)
        for volume in volumes.values():
            _docker("volume", "rm", "--force", volume, check=False, timeout=30)
        _docker("image", "rm", "--force", image, check=False, timeout=60)
