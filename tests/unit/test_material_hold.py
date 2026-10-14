"""Offline stdlib gate tests; sentinel proves ordering, never actual SDK startup.

Filesystem/ownership acceptance requires Linux root launching a real UID10001 child.
Windows skips are expressly not POSIX acceptance evidence.
"""

import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

OPERATOR = Path(__file__).resolve().parents[2] / "scripts/material_handover_1465.py"
IMAGE = "sha256:32515b282f458fee9b549a5cf891258c7a3b2f5b6f81d18b04a7aef751a841f6"
ATTEMPT = "a" * 64
CANDIDATE = "c" * 64
NONCE = "d" * 64  # Synthetic test value only.
NONCE_HASH = hashlib.sha256(NONCE.encode("ascii")).hexdigest()


def test_public_operator_entrypoint_exists():
    # Deliberate initial causal RED: absent requested executable, no fixture import involved.
    assert OPERATOR.is_file(), "native-hold public entrypoint is absent"


@pytest.fixture
def hold_case():
    if sys.platform != "linux" or os.geteuid() != 0:
        pytest.skip("Requires real Linux root and effective UID10001 child; not qualified here")
    assert OPERATOR.is_file(), "native-hold public entrypoint is absent"
    # Root mounts /run as root-owned755 tmpfs; /tmp is unsafe, / is read-only.
    base = Path(tempfile.mkdtemp(prefix="material-hold-", dir="/run"))
    base.chmod(0o755)
    control = base / "control"
    control.mkdir(mode=0o755)
    output = base / "sentinel"
    output.mkdir(mode=0o777)
    output.chmod(0o777)
    executable = base / "material_hold_runner.py"
    shutil.copyfile(OPERATOR, executable)
    executable.chmod(0o644)
    package = base / "projetv0_voice"
    package.mkdir(mode=0o755)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "server.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "marker = Path(os.environ['HOLD_SENTINEL'])\n"
        "with marker.open('a') as stream: stream.write('import\\n')\n"
        "def main(mapping=os.environ):\n"
        "    assert mapping['HOLD_ENV_PROBE'] == 'preserved'\n"
        "    with marker.open('a') as stream: stream.write('main\\n')\n",
        encoding="utf-8",
    )
    children = []

    def launch(expiry_seconds=4, directory=None, uid=10001, overrides=None,
               expiry_text=None, cross_monotonic_limit=False, mode="native-hold",
               extra_arguments=()):
        expiry = expiry_text or (datetime.now(UTC) + timedelta(seconds=expiry_seconds)).isoformat()
        env = dict(os.environ, PYTHONPATH=str(base), HOLD_SENTINEL=str(output / "calls"),
                   HOLD_ENV_PROBE="preserved")

        def drop_uid():
            os.setgroups([])
            os.setgid(10001)
            os.setuid(10001)

        arguments = [str(executable), mode, "--release-directory",
                     str(directory or control), "--attempt-sha256", ATTEMPT,
                     "--candidate-sha256", CANDIDATE, "--nonce-sha256", NONCE_HASH,
                     "--candidate-expiry", expiry]
        arguments.extend(extra_arguments)
        for flag, value in (overrides or {}).items():
            arguments[arguments.index(flag) + 1] = value
        prefix = [sys.executable]
        if cross_monotonic_limit:
            # Only the child's monotonic clock changes: real main/files/validator remain.
            wrapper = (
                "import runpy,sys,time\n"
                "path=sys.argv.pop(1); sys.argv[0]=path\n"
                "module=runpy.run_path(path)\n"
                "ticks=iter([0.0,0.0])\n"
                "time.monotonic=lambda: next(ticks,301.0)\n"
                "raise SystemExit(module['main']())\n"
            )
            prefix += ["-c", wrapper]
        child = subprocess.Popen(
            prefix + arguments, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=drop_uid if uid == 10001 else None,
        )
        children.append(child)
        return child

    def publish(changes=None, raw=None):
        payload = dict(schema_version=1, attempt_sha256=ATTEMPT, candidate_sha256=CANDIDATE,
                       image_digest=IMAGE, nonce=NONCE)
        payload.update(changes or {})
        temporary = control / "pending.json"
        temporary.write_text(raw if raw is not None else json.dumps(payload), encoding="utf-8")
        temporary.chmod(0o644)
        temporary.replace(control / "release.json")

    def finish(child, success=False):
        stdout, stderr = child.communicate(timeout=7)
        assert (child.returncode == 0) is success
        assert NONCE.encode() not in stdout + stderr
        marker = output / "calls"
        if success:
            assert marker.read_text() == "import\nmain\n"
        else:
            assert not marker.exists(), "Unauthorized native import/delegation occurred"

    yield control, output, launch, publish, finish
    for child in children:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=2)
    shutil.rmtree(base)


def wait_until_held(child):
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=3), "No actual secure-directory hold within3s"
        line = child.stdout.readline()
        if line == b"":
            returncode = child.wait(timeout=2)
            # Child input/files are synthetic; bound diagnostics to this test process.
            stderr = child.stderr.read(4096).decode("utf-8", errors="replace")
            pytest.fail(f"Synthetic child exited before hold: returncode={returncode}; "
                        f"stderr={stderr!r}")
        assert line == b"material_startup_held\n", "Unexpected synthetic-child hold diagnostic"
    assert child.poll() is None


def test_empty_hold_then_atomic_valid_release_delegates_once(hold_case):
    _, output, launch, publish, finish = hold_case
    child = launch()
    wait_until_held(child)
    assert not (output / "calls").exists()
    publish()
    finish(child, success=True)


@pytest.mark.parametrize("changes", [
    {"attempt_sha256": "b" * 64}, {"candidate_sha256": "b" * 64},
    {"image_digest": "sha256:" + "b" * 64}, {"nonce": "e" * 64},
    {"nonce": NONCE.upper()}, {"nonce": "d" * 63}, {"nonce": 123},
    {"schema_version": True}, {"schema_version": 1.0}, {"extra": "forbidden"},
])
def test_invalid_values_cannot_import_or_delegate(hold_case, changes):
    _, _, launch, publish, finish = hold_case
    publish(changes)
    finish(launch())


@pytest.mark.parametrize("raw", [
    '{"schema_version":1,"schema_version":1}', '{}', '[]', 'null',
    '{"schema_version":NaN}', '{"schema_version":Infinity}', '{broken',
])
def test_invalid_json_cannot_import_or_delegate(hold_case, raw):
    _, _, launch, publish, finish = hold_case
    publish(raw=raw)
    finish(launch())


@pytest.mark.parametrize("field", ["schema_version", "attempt_sha256", "candidate_sha256",
                                  "image_digest", "nonce"])
def test_each_missing_required_key_refuses(hold_case, field):
    _, _, launch, publish, finish = hold_case
    payload = dict(schema_version=1, attempt_sha256=ATTEMPT, candidate_sha256=CANDIDATE,
                   image_digest=IMAGE, nonce=NONCE)
    del payload[field]
    publish(raw=json.dumps(payload))
    finish(launch())


def test_duplicate_key_in_otherwise_valid_release_refuses(hold_case):
    _, _, launch, publish, finish = hold_case
    payload = dict(schema_version=1, attempt_sha256=ATTEMPT, candidate_sha256=CANDIDATE,
                   image_digest=IMAGE, nonce=NONCE)
    publish(raw=json.dumps(payload)[:-1] + ', "schema_version": 1}')
    finish(launch())


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "oversized", "fifo", "directory",
                                        "writable", "nonroot"])
def test_unsafe_actual_release_file_refuses(hold_case, unsafe):
    control, _, launch, publish, finish = hold_case
    release = control / "release.json"
    if unsafe == "fifo":
        os.mkfifo(release)
    elif unsafe == "directory":
        release.mkdir()
    else:
        publish()
        if unsafe == "symlink":
            release.rename(control / "target")
            release.symlink_to(control / "target")
        elif unsafe == "hardlink":
            os.link(release, control / "other")
        elif unsafe == "oversized":
            valid = release.read_bytes()
            release.write_bytes(valid + b" " * (4097 - len(valid)))
        elif unsafe == "writable":
            release.chmod(0o666)
        elif unsafe == "nonroot":
            os.chown(release, 10001, 10001)
    finish(launch())


@pytest.mark.parametrize("unsafe", ["symlink", "ancestor_symlink", "writable", "nonroot",
                                  "ancestor_writable", "ancestor_nonroot"])
def test_unsafe_control_directory_refuses(hold_case, unsafe):
    control, _, launch, publish, finish = hold_case
    publish()
    directory = control
    if unsafe in {"symlink", "ancestor_symlink"}:
        alias = control.parent / "alias"
        target = control if unsafe == "symlink" else control.parent
        alias.symlink_to(target, target_is_directory=True)
        directory = alias if unsafe == "symlink" else alias / "control"
    elif unsafe == "writable":
        control.chmod(0o777)
    elif unsafe == "ancestor_writable":
        control.parent.chmod(0o777)
    elif unsafe == "ancestor_nonroot":
        os.chown(control.parent, 10001, 10001)
    else:
        os.chown(control, 10001, 10001)
    finish(launch(directory=directory))


def test_directory_replacement_during_hold_refuses(hold_case):
    control, _, launch, publish, finish = hold_case
    child = launch()
    wait_until_held(child)
    control.rename(control.parent / "original")
    control.mkdir(mode=0o755)
    publish()
    finish(child)


@pytest.mark.parametrize("expiry", [-1, 0.4])
def test_original_expiry_blocks_delegation(hold_case, expiry):
    _, _, launch, publish, finish = hold_case
    if expiry < 0:
        publish()
    finish(launch(expiry_seconds=expiry))


def test_sigterm_interrupts_empty_hold_without_delegation(hold_case):
    _, _, launch, _, finish = hold_case
    child = launch()
    wait_until_held(child)
    child.send_signal(signal.SIGTERM)
    finish(child)


def test_independent_300_second_monotonic_cap_refuses_valid_release(hold_case):
    _, _, launch, publish, finish = hold_case
    publish()
    finish(launch(expiry_seconds=3600, cross_monotonic_limit=True))


def test_uid_zero_cannot_delegate_valid_release(hold_case):
    _, _, launch, publish, finish = hold_case
    publish()
    finish(launch(uid=0))


@pytest.mark.parametrize("overrides", [
    {"--attempt-sha256": "a" * 63}, {"--candidate-sha256": "C" * 64},
    {"--nonce-sha256": "not-a-hash"}, {"--release-directory": "relative"},
    {"--release-directory": "/run/../run/control"},
])
def test_invalid_public_arguments_refuse(hold_case, overrides):
    _, _, launch, publish, finish = hold_case
    publish()
    finish(launch(overrides=overrides))


@pytest.mark.parametrize("expiry_text", ["2030-01-01T00:00:00", "2030-01-01T00:00:00+01:00",
                                       "not-a-date"])
def test_non_utc_or_malformed_expiry_refuses(hold_case, expiry_text):
    _, _, launch, publish, finish = hold_case
    publish()
    finish(launch(expiry_text=expiry_text))


@pytest.mark.parametrize("mode,extra_arguments", [
    ("unknown-mode", ()), ("native-hold", ("--unknown-option",)),
])
def test_cli_syntax_refuses(hold_case, mode, extra_arguments):
    _, _, launch, publish, finish = hold_case
    publish()
    finish(launch(mode=mode, extra_arguments=extra_arguments))
