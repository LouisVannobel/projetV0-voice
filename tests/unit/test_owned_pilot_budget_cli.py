"""Real operator/writer shutdown, with only filesystem/provider seams doubled."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import sqlite3
from dataclasses import replace
from decimal import Decimal
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import pytest_asyncio
from pydantic import SecretStr

from projetv0_voice import owned_pilot_budget_cli as cli
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.owned_pilot_budget import OwnedPilotBudgetGate
from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.qualified_profile import (
    QualifiedDeploymentProfileV1,
    canonical_qualified_profile_sha256,
)
from tests.integration.test_fixed_v2_process import KEY, NOW
from tests.unit.test_owned_pilot_budget import budget_policy, observation
from tests.unit.test_production_wiring import _settings


@pytest_asyncio.fixture(autouse=True)
async def no_unretrieved_owner_futures():
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        yield
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled, [item.get("message") for item in unhandled]
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [
    "enroll", "suspend-restored", "extend-sixth-call", "extend-seventh-call",
    "enable-budgeted-calls",
])
@pytest.mark.parametrize("cancel_request,release_late_commit,force_close_timeout", [
    pytest.param(False, True, False, id="late-commit"),
    pytest.param(False, False, False, id="blocked"),
    pytest.param(True, True, False, id="cancel-commit"),
    pytest.param(True, False, False, id="cancel-blocked"),
    pytest.param(True, True, True, id="cancel-commit-close-timeout"),
])
async def test_operator_timeout_joins_writer_without_claiming_non_commit(
    tmp_path,
    monkeypatch,
    action,
    release_late_commit,
    cancel_request,
    force_close_timeout,
):
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/qualified-deployment-profile-v1.json").read_text()
    )
    settings = replace(
        _settings(),
        sqlite_path=tmp_path / "voice.sqlite",
        runtime_contract_sha256=profile.runtime_contract_sha256,
        image_digest=profile.image_digest,
        agent_bundle_sha256=profile.agent_bundle_sha256,
        inference_profile_sha256=profile.inference_profile_sha256,
    )
    or_key, tel_key = SecretStr("offline-or"), SecretStr("offline-tel")
    identity = replace(
        observation().identity,
        openrouter_credential_sha256=hashlib.sha256(b"offline-or").hexdigest(),
        telnyx_credential_sha256=hashlib.sha256(b"offline-tel").hexdigest(),
    )
    policy = budget_policy(
        identity=identity,
        deployment_id=settings.deployment_id,
        runtime_contract_sha256=settings.runtime_contract_sha256,
        image_digest=settings.image_digest,
        agent_bundle_sha256=settings.agent_bundle_sha256,
        inference_profile_sha256=settings.inference_profile_sha256,
        qualified_profile_sha256=canonical_qualified_profile_sha256(profile),
    )
    predecessor = policy
    extensions = {"extend-sixth-call", "extend-seventh-call", "enable-budgeted-calls"}
    if action in extensions:
        if action in {"extend-seventh-call", "enable-budgeted-calls"}:
            predecessor = policy.model_copy(update={
                "total_calls": 7 if action == "enable-budgeted-calls" else 6,
                "image_digest": "fixture@sha256:" + "b" * 64,
                "qualified_profile_sha256": "a" * 64,
            })
        policy = policy.model_copy(update={
            "total_calls": None if action == "enable-budgeted-calls"
            else predecessor.total_calls + 1,
        })
    fact = replace(observation(), identity=identity, openrouter_reported_spend_usd=Decimal("1.40"))
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    manifest = SimpleNamespace(
        agent_id="fixture",
        dids=[identity.did],
        sparra=SimpleNamespace(operation_contract_version=2, connection_id=identity.connection_id),
    )
    factories = SimpleNamespace(
        validate_artifacts=lambda _: None,
        load_manifest=lambda _: manifest,
        load_profile=lambda *_: SimpleNamespace(profile=profile),
        load_keyring=lambda _: keyring,
        read_secret=lambda path: or_key if path == settings.openrouter_api_key_file else tel_key,
    )
    monkeypatch.setattr(cli, "capture_runtime_environment", lambda _: None)
    monkeypatch.setattr(cli, "parse_runtime_settings", lambda _: settings)
    monkeypatch.setattr(cli, "build_production_factories", lambda _: factories)

    async def collect():
        return fact

    monkeypatch.setattr(
        cli,
        "_budget_factory",
        lambda _settings, writer, *_args: OwnedPilotBudgetGate(
            writer, policy, collect, utcnow=lambda: NOW
        ),
    )
    if action in extensions | {"suspend-restored"}:
        seed = PersistenceWriter(
            settings.sqlite_path,
            keyring,
            contract_version=2,
            process_agent_id="fixture",
            process_deployment_id=settings.deployment_id,
            utcnow=lambda: NOW,
        )
        seed.set_owned_budget(predecessor.model_copy(update={"total_calls": 5}))
        owner = asyncio.create_task(seed.run())
        assert await seed.wait_ready()
        assert await seed.owned_budget_request("enroll", observation=fact, predecessor_retired=True)
        await seed.drain(2)
        await owner
        if action in extensions:
            with sqlite3.connect(settings.sqlite_path) as db:
                if action in {"extend-seventh-call", "enable-budgeted-calls"}:
                    from projetv0_voice.persistence.schema import SIXTH_BUDGET_MIGRATION_SQL
                    db.executescript(SIXTH_BUDGET_MIGRATION_SQL)
                    db.execute("PRAGMA user_version=13")
                    if action == "enable-budgeted-calls":
                        from projetv0_voice.persistence.schema import SEVENTH_BUDGET_MIGRATION_SQL
                        db.executescript(SEVENTH_BUDGET_MIGRATION_SQL)
                        db.execute("PRAGMA user_version=14")
                    db.execute("UPDATE owned_pilot_budget SET policy_json=?,policy_sha256=?",
                               (predecessor.model_dump_json(), predecessor.sha256))
                db.execute("UPDATE owned_pilot_budget SET used_calls=?,openrouter_floor='1.10'",
                           (predecessor.total_calls,))

    from tests.integration.test_owned_pilot_sixth_call import all_rows
    before_rows = all_rows(settings.sqlite_path) if settings.sqlite_path.exists() else {}
    before_schema = None
    if before_rows:
        with sqlite3.connect(settings.sqlite_path) as db:
            before_schema = db.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
    old_budget = before_rows.get("owned_pilot_budget", [])
    if action == "enroll":
        new_budget = [("sparra-original-pilot", policy.model_dump_json(), policy.sha256,
                       0, "1.40", "0", 0, None)]
    else:
        row = list(old_budget[0])
        if action == "suspend-restored":
            row[6] = 1
        else:
            row[1], row[2], row[4] = policy.model_dump_json(), policy.sha256, "1.40"
        new_budget = [tuple(row)]

    entered, release = asyncio.Event(), asyncio.Event()
    native_committed, drain_timed_out = asyncio.Event(), asyncio.Event()
    close_safety_seconds = cli._WRITER_CLOSE_SECONDS

    async def pause(name):
        if name == "after_mutation_before_commit":
            entered.set()
            await release.wait()

    instances = []

    def make_writer(path, keys, **kwargs):
        writer = PersistenceWriter(
            path,
            keys,
            **kwargs,
            utcnow=lambda: NOW,
            control_commit_timeout_seconds=0.03,
            failpoint=pause,
        )
        native_drain = writer.drain

        async def drain(timeout_seconds):
            if not release_late_commit:
                # Model the drain deadline at the held mutation, not via fsync timing.
                assert entered.is_set()
                drain_timed_out.set()
                raise TimeoutError("offline mutation drain timeout")
            if force_close_timeout:
                # The transaction is real; lose only its bounded close acknowledgement.
                async with asyncio.timeout(close_safety_seconds):
                    await native_committed.wait()
                drain_timed_out.set()
                raise TimeoutError("offline close acknowledgement timeout")
            try:
                await native_drain(timeout_seconds)
            except TimeoutError:
                drain_timed_out.set()
                raise

        monkeypatch.setattr(writer, "drain", drain)
        instances.append(writer)
        return writer

    monkeypatch.setattr(cli, "PersistenceWriter", make_writer)

    def forbid_parent_exit(_status):
        pytest.fail(
            "operator matrix must not hard-exit pytest; exercise it in the owned subprocess"
        )

    monkeypatch.setattr(cli.os, "_exit", forbid_parent_exit)
    operation = asyncio.create_task(
        cli.operate(action, predecessor_retired=True, expected_policy_sha256=predecessor.sha256)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        writer = instances[0]
        if action == "enroll":
            # Initial V2 schema/contract creation precedes the paused enrollment.
            # This reader sees only that committed baseline, never its pending row.
            before_rows = all_rows(settings.sqlite_path)
            assert before_rows["owned_pilot_budget"] == old_budget
        await asyncio.wait_for(writer.fatal_event.wait(), 2)
        if force_close_timeout:
            native_commit = writer._connection.commit

            async def commit():
                await native_commit()
                native_committed.set()

            monkeypatch.setattr(writer._connection, "commit", commit)
        if cancel_request:
            operation.cancel()
        if release_late_commit:
            release.set()
        done, _ = await asyncio.wait({operation}, timeout=2 * close_safety_seconds + 1)
        assert operation in done, "operator left the degraded writer waiting forever"
        assert drain_timed_out.is_set() is (not release_late_commit or force_close_timeout)
        if cancel_request and release_late_commit and not drain_timed_out.is_set():
            with pytest.raises(asyncio.CancelledError):
                operation.result()
        else:
            with pytest.raises(cli.OwnedPilotBudgetOutcomeUnknown,
                               match="^owned_budget_transition_unknown$"):
                operation.result()
        assert writer._owner_task.done()
        assert writer._closed_event.is_set() and writer._connection is None
        after_rows = all_rows(settings.sqlite_path)
        # Releasing a failpoint is not a COMMIT receipt. A close timeout can
        # interrupt before or after COMMIT, but never leave a mixed authority.
        assert after_rows["owned_pilot_budget"] in (old_budget, new_budget)
        committed = after_rows["owned_pilot_budget"] == new_budget
        if not release_late_commit or not drain_timed_out.is_set():
            assert committed is release_late_commit
        if force_close_timeout:
            assert native_committed.is_set() and drain_timed_out.is_set() and committed
        assert after_rows.keys() == before_rows.keys()
        for table, rows in after_rows.items():
            if table != "owned_pilot_budget":
                assert rows == before_rows.get(table, [])
        with sqlite3.connect(settings.sqlite_path) as db:
            if action in extensions:
                old_version, new_version = {
                    "extend-sixth-call": (11, 13), "extend-seventh-call": (13, 14),
                    "enable-budgeted-calls": (14, 15),
                }[action]
                assert db.execute("PRAGMA user_version").fetchone() == (
                    new_version if committed else old_version,)
            else:
                assert db.execute("PRAGMA user_version").fetchone() == (11,)
        if action in extensions:
            if not committed:
                assert after_rows == before_rows
                with sqlite3.connect(settings.sqlite_path) as db:
                    assert db.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall() == (
                        before_schema
                    )
            restarted = PersistenceWriter(
                settings.sqlite_path,
                keyring,
                contract_version=2,
                process_agent_id="fixture",
                process_deployment_id=settings.deployment_id,
                utcnow=lambda: NOW,
            )
            restarted.set_owned_budget(policy)
            restart_owner = asyncio.create_task(restarted.run())
            try:
                assert await restarted.wait_ready()
                assert bool(
                    await restarted.owned_budget_request(
                        action,
                        observation=fact,
                        predecessor_retired=True,
                        expected_policy_sha256=predecessor.sha256,
                    )
                ) is (not committed)
                assert not await restarted.owned_budget_request(
                    action,
                    observation=fact,
                    predecessor_retired=True,
                    expected_policy_sha256=predecessor.sha256,
                )
                assert await restarted.owned_budget_request("observe", observation=fact)
                state = await restarted.owned_budget_request("state")
                assert state[:3] == (predecessor.total_calls, "1.40", "0")
            finally:
                await restarted.drain(2)
                await restart_owner
    finally:
        release.set()
        async with asyncio.timeout(2 * close_safety_seconds + 1):
            if instances and not instances[0]._closed_event.is_set():
                # Cleanup itself uses the real bounded drain, not the injected failure.
                await PersistenceWriter.drain(instances[0], close_safety_seconds)
            await asyncio.gather(operation, return_exceptions=True)


def test_operator_unknown_outcome_is_not_printed_as_refusal(monkeypatch, capsys):
    import sys

    async def uncertain(*args, **kwargs):
        raise cli.OwnedPilotBudgetOutcomeUnknown()

    monkeypatch.setattr(cli, "operate", uncertain)
    monkeypatch.setattr(sys, "argv", ["owned-pilot-budget", "enroll", "--predecessor-retired"])
    assert cli.main() == 1
    assert capsys.readouterr().out == "owned_budget_transition_unknown\n"


@pytest.mark.parametrize("action", ["enroll", "extend-seventh-call"])
def test_operator_hard_exit_is_bounded_when_real_owner_cannot_join(tmp_path, action):
    import os
    import subprocess
    import sys

    script = r"""
import asyncio
import sqlite3
import sys
from pathlib import Path
from projetv0_voice import owned_pilot_budget_cli as cli
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.writer import PersistenceWriter
from tests.integration.test_fixed_v2_process import KEY, NOW
from tests.unit.test_owned_pilot_budget import budget_policy, observation

async def main():
    action = sys.argv[1]
    predecessor = budget_policy(total_calls=6)
    if action == "extend-seventh-call":
        from projetv0_voice.persistence.schema import SIXTH_AUDIO_BUDGET_SCHEMA_SQL
        with sqlite3.connect("stuck.sqlite") as db:
            db.executescript(SIXTH_AUDIO_BUDGET_SCHEMA_SQL)
            db.execute("INSERT INTO owned_pilot_budget VALUES(?,?,?,?,?,?,?,?)",
                       ("sparra-original-pilot", predecessor.model_dump_json(), predecessor.sha256,
                        6, "1.10", "0.20", 0, None))
    entered = asyncio.Event()
    async def stuck(name):
        if name == "after_mutation_before_commit":
            entered.set()
            # Model the uninterruptible-owner boundary using the native failpoint;
            # all SQLite and transaction ownership remains real.
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
    writer = PersistenceWriter(Path("stuck.sqlite"), CryptoKeyring({1: KEY}, active_version=1),
                               utcnow=lambda: NOW, failpoint=stuck,
                               contract_version=2 if action == "extend-seventh-call" else 1,
                               process_agent_id="fixture", process_deployment_id="fixture")
    writer.set_owned_budget(budget_policy(total_calls=7) if action == "extend-seventh-call"
                            else budget_policy())
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    asyncio.create_task(writer.owned_budget_request(action, observation=observation(),
                        predecessor_retired=True, expected_policy_sha256=predecessor.sha256))
    await entered.wait()
    cli._WRITER_CLOSE_SECONDS = .03
    await cli._close_operator_writer(writer, owner)
    raise AssertionError("unjoined owner returned")

asyncio.run(main())
"""
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script, action],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        env={
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows"),
            "PYTHONPATH": str(root) + os.pathsep + str(root / "src"),
        },
    )
    assert result.returncode == 1
    assert result.stdout == "owned_budget_transition_unknown\n"
    with sqlite3.connect(tmp_path / "stuck.sqlite") as db:
        if action == "extend-seventh-call":
            predecessor = budget_policy(total_calls=6)
            assert db.execute(
                "SELECT used_calls,policy_sha256 FROM owned_pilot_budget"
            ).fetchall() == [(6, predecessor.sha256)]
            assert db.execute("PRAGMA user_version").fetchone() == (13,)
        else:
            assert db.execute("SELECT count(*) FROM owned_pilot_budget").fetchone() == (0,)


@pytest.mark.parametrize(
    "options,message",
    [
        ([], "requires --predecessor-retired"),
        (["--predecessor-retired"], "requires --expected-policy-sha256"),
        (
            ["--predecessor-retired", "--expected-policy-sha256", "F" * 64],
            "requires --expected-policy-sha256",
        ),
    ],
)
@pytest.mark.parametrize("action", [
    "extend-sixth-call", "extend-seventh-call", "enable-budgeted-calls",
])
def test_extension_operator_requires_explicit_flags_before_runtime(
    monkeypatch, capsys, options, message, action
):
    import sys

    monkeypatch.setattr(sys, "argv", ["budget", action, *options])
    monkeypatch.setattr(
        cli,
        "capture_runtime_environment",
        lambda _: pytest.fail("must validate explicit flags before runtime"),
    )
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("action", [
    "extend-sixth-call", "extend-seventh-call", "enable-budgeted-calls",
])
def test_extension_operator_forwards_exact_expected_hash(monkeypatch, capsys, action):
    import sys

    calls = []

    async def operate(action, **values):
        calls.append((action, values))
        return True

    monkeypatch.setattr(cli, "operate", operate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "budget",
            action,
            "--predecessor-retired",
            "--expected-policy-sha256",
            "e" * 64,
        ],
    )
    assert cli.main() == 0
    assert calls == [
        (action, {"predecessor_retired": True, "expected_policy_sha256": "e" * 64})
    ]
    assert capsys.readouterr().out == "owned_budget_transition_accepted\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [
    "extend-seventh-call", "enable-budgeted-calls", "rebind-release",
])
async def test_seventh_operator_rejects_v1_before_writer_or_credentials(monkeypatch, action):
    profile = QualifiedDeploymentProfileV1.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/qualified-deployment-profile-v1.json").read_text()
    )
    manifest = SimpleNamespace(sparra=SimpleNamespace(operation_contract_version=1))
    factories = SimpleNamespace(
        validate_artifacts=lambda _: None,
        load_manifest=lambda _: manifest,
        load_profile=lambda *_: SimpleNamespace(profile=profile),
    )
    monkeypatch.setattr(cli, "capture_runtime_environment", lambda _: None)
    monkeypatch.setattr(cli, "parse_runtime_settings", lambda _: _settings())
    monkeypatch.setattr(cli, "build_production_factories", lambda _: factories)
    monkeypatch.setattr(cli, "PersistenceWriter", lambda *_args, **_kwargs: pytest.fail(
        "seventh V1 must be refused before writer creation"))
    with pytest.raises(RuntimeError, match="^owned_budget_operator_refused$"):
        await cli.operate(action, predecessor_retired=True,
                          expected_policy_sha256="e" * 64, **({
                              "predecessor_qualified_profile_path": "/secure/previous.json",
                              "predecessor_image_digest": "fixture@sha256:" + "a" * 64,
                          } if action == "rebind-release" else {}))


@pytest.mark.parametrize("missing", ["retired", "cas", "profile", "image"])
def test_release_operator_requires_all_explicit_flags_before_runtime(monkeypatch, capsys, missing):
    import sys

    options = {
        "retired": ["--predecessor-retired"],
        "cas": ["--expected-policy-sha256", "e" * 64],
        "profile": ["--predecessor-qualified-profile-path", "/secure/previous-profile.json"],
        "image": ["--predecessor-image-digest", "fixture@sha256:" + "a" * 64],
    }
    monkeypatch.setattr(sys, "argv", ["budget", "rebind-release", *[
        argument for key, arguments in options.items() if key != missing
        for argument in arguments]])
    monkeypatch.setattr(cli, "capture_runtime_environment",
                        lambda _: pytest.fail("explicit release flags must precede runtime"))
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    assert "requires --" in capsys.readouterr().err


def test_release_operator_forwards_the_exact_predecessor_tuple_and_cas(monkeypatch, capsys):
    import sys

    calls = []

    async def operate(action, **values):
        calls.append((action, values))
        return True

    monkeypatch.setattr(cli, "operate", operate)
    monkeypatch.setattr(sys, "argv", ["budget", "rebind-release", "--predecessor-retired",
        "--expected-policy-sha256", "e" * 64, "--predecessor-qualified-profile-path",
        "/secure/previous-profile.json", "--predecessor-image-digest",
        "fixture@sha256:" + "a" * 64])
    assert cli.main() == 0
    assert calls == [("rebind-release", {"predecessor_retired": True,
        "expected_policy_sha256": "e" * 64,
        "predecessor_qualified_profile_path": "/secure/previous-profile.json",
        "predecessor_image_digest": "fixture@sha256:" + "a" * 64})]
    assert capsys.readouterr().out == "owned_budget_transition_accepted\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    None, "previous_image", "previous_runtime", "missing_store", "legacy_store", "malformed_store",
])
async def test_release_operator_loads_both_native_profile_bindings_before_real_cas(
    tmp_path, monkeypatch, invalid,
):
    from projetv0_voice import production_wiring as wiring
    from tests.integration.test_owned_pilot_budgeted_calls import seed_release

    profile = QualifiedDeploymentProfileV1.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/qualified-deployment-profile-v1.json").read_text())
    old_image = "ghcr.io/example/voice@sha256:" + "b" * 64
    old_profile = profile.model_copy(update={"image_digest": old_image,
        **({"runtime_contract_sha256": "f" * 64} if invalid == "previous_runtime" else {})})
    current_path, previous_path = tmp_path / "current.json", tmp_path / "previous.json"
    current_path.write_text(profile.model_dump_json())
    previous_path.write_text(old_profile.model_dump_json())
    settings = replace(_settings(), sqlite_path=tmp_path / "voice.sqlite",
        deployment_id=profile.deployment_id,
        qualified_profile_path=PurePosixPath(current_path.as_posix()),
        runtime_contract_sha256=profile.runtime_contract_sha256,
        image_digest=profile.image_digest, agent_bundle_sha256=profile.agent_bundle_sha256,
        inference_profile_sha256=profile.inference_profile_sha256)
    identity = replace(observation().identity,
        openrouter_credential_sha256=hashlib.sha256(b"offline-or").hexdigest(),
        telnyx_credential_sha256=hashlib.sha256(b"offline-tel").hexdigest())
    policy = budget_policy(total_calls=None, identity=identity,
        deployment_id=settings.deployment_id,
        runtime_contract_sha256=settings.runtime_contract_sha256,
        image_digest=settings.image_digest, agent_bundle_sha256=settings.agent_bundle_sha256,
        inference_profile_sha256=settings.inference_profile_sha256,
        qualified_profile_sha256=canonical_qualified_profile_sha256(profile))
    previous_policy = policy.model_copy(update={"image_digest": old_image,
        "qualified_profile_sha256": canonical_qualified_profile_sha256(old_profile)})
    if invalid not in {"missing_store", "legacy_store"}:
        seed_release(settings.sqlite_path, policy=previous_policy)
    if invalid == "legacy_store":
        from projetv0_voice.persistence.schema import LOCAL_AUDIO_SCHEMA_SQL
        with sqlite3.connect(settings.sqlite_path) as db:
            db.executescript(LOCAL_AUDIO_SCHEMA_SQL)
    if invalid == "malformed_store":
        with sqlite3.connect(settings.sqlite_path) as db:
            db.execute("PRAGMA user_version=16")
    from tests.integration.test_owned_pilot_seventh_call import authority
    before = authority(settings.sqlite_path) if settings.sqlite_path.exists() else None
    frozen = settings.sqlite_path.read_bytes() if before else None
    manifest = SimpleNamespace(agent_id="fixture", dids=[identity.did],
        sparra=SimpleNamespace(operation_contract_version=2, connection_id=identity.connection_id))
    validated = []
    monkeypatch.setattr(wiring, "_validate_artifacts", lambda received: validated.append(received))
    monkeypatch.setattr(wiring, "load_agent_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(wiring, "_root_owned_regular", lambda _: True)
    keyring = CryptoKeyring({1: KEY}, active_version=1)
    monkeypatch.setattr(wiring, "_decode_keyring_secret", lambda _: keyring)
    monkeypatch.setattr(wiring, "read_runtime_secret", lambda path: SecretStr(
        "offline-or" if path == settings.openrouter_api_key_file else "offline-tel"))
    monkeypatch.setattr(cli, "capture_runtime_environment", lambda _: None)
    monkeypatch.setattr(cli, "parse_runtime_settings", lambda _: settings)
    monkeypatch.setattr(cli, "PersistenceWriter", lambda path, keys, **values:
        PersistenceWriter(path, keys, **values, utcnow=lambda: NOW))

    async def collect():
        return replace(observation(), identity=identity)

    monkeypatch.setattr(cli, "_budget_factory", lambda _settings, writer, *_args:
        OwnedPilotBudgetGate(writer, policy, collect, utcnow=lambda: NOW))
    arguments = dict(predecessor_retired=True, expected_policy_sha256=previous_policy.sha256,
        predecessor_qualified_profile_path=str(previous_path),
        predecessor_image_digest="wrong-image" if invalid == "previous_image" else old_image)
    if invalid in {"previous_image", "previous_runtime"}:
        with pytest.raises(ValueError, match="does not match the injected value"):
            await cli.operate("rebind-release", **arguments)
        assert authority(settings.sqlite_path) == before
    elif invalid:
        assert not await cli.operate("rebind-release", **arguments)
        if before:
            assert authority(settings.sqlite_path) == before
            assert settings.sqlite_path.read_bytes() == frozen
        else:
            assert not settings.sqlite_path.exists()
    else:
        assert await cli.operate("rebind-release", **arguments)
        rows, schema, version = authority(settings.sqlite_path)
        assert schema == before[1] and version == before[2] == (15,)
        assert rows["owned_pilot_budget"][0][1:3] == (policy.model_dump_json(), policy.sha256)
        assert rows["owned_pilot_budget"][0][3:] == before[0]["owned_pilot_budget"][0][3:]
    assert validated == [settings, replace(settings,
        qualified_profile_path=PurePosixPath(previous_path.as_posix()),
        image_digest=arguments["predecessor_image_digest"])]
