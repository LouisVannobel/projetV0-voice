"""Real operator/writer shutdown, with only filesystem/provider seams doubled."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
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
@pytest.mark.parametrize("action", ["enroll", "suspend-restored"])
@pytest.mark.parametrize("release_late_commit", [True, False])
async def test_operator_timeout_joins_writer_without_claiming_non_commit(
    tmp_path,
    monkeypatch,
    action,
    release_late_commit,
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
    fact = replace(observation(), identity=identity)
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
    if action == "suspend-restored":
        seed = PersistenceWriter(
            settings.sqlite_path,
            keyring,
            contract_version=2,
            process_agent_id="fixture",
            process_deployment_id=settings.deployment_id,
            utcnow=lambda: NOW,
        )
        seed.set_owned_budget(policy)
        owner = asyncio.create_task(seed.run())
        assert await seed.wait_ready()
        assert await seed.owned_budget_request("enroll", observation=fact, predecessor_retired=True)
        await seed.drain(2)
        await owner

    entered, release = asyncio.Event(), asyncio.Event()

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
        instances.append(writer)
        return writer

    monkeypatch.setattr(cli, "PersistenceWriter", make_writer)
    monkeypatch.setattr(cli, "_WRITER_CLOSE_SECONDS", 0.05)
    operation = asyncio.create_task(cli.operate(action, predecessor_retired=True))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        writer = instances[0]
        await asyncio.wait_for(writer.fatal_event.wait(), 2)
        if release_late_commit:
            release.set()
        done, _ = await asyncio.wait({operation}, timeout=0.5)
        assert operation in done, "operator left the degraded writer waiting forever"
        with pytest.raises(RuntimeError, match="^owned_budget_transition_unknown$"):
            operation.result()
        assert writer._owner_task.done()
        assert writer._closed_event.is_set() and writer._connection is None
        with sqlite3.connect(settings.sqlite_path) as db:
            expected = (
                [(0, 1 if action == "suspend-restored" else 0)]
                if release_late_commit
                else []
                if action == "enroll"
                else [(0, 0)]
            )
            assert (
                db.execute("SELECT used_calls,suspended FROM owned_pilot_budget").fetchall()
                == expected
            )
    finally:
        release.set()
        if instances and not instances[0]._closed_event.is_set():
            await instances[0].drain(2)
        await asyncio.gather(operation, return_exceptions=True)


def test_operator_unknown_outcome_is_not_printed_as_refusal(monkeypatch, capsys):
    import sys

    async def uncertain(*args, **kwargs):
        raise cli.OwnedPilotBudgetOutcomeUnknown()

    monkeypatch.setattr(cli, "operate", uncertain)
    monkeypatch.setattr(sys, "argv", ["owned-pilot-budget", "enroll", "--predecessor-retired"])
    assert cli.main() == 1
    assert capsys.readouterr().out == "owned_budget_transition_unknown\n"


def test_operator_hard_exit_is_bounded_when_real_owner_cannot_join(tmp_path):
    import os
    import subprocess
    import sys

    script = r"""
import asyncio
from pathlib import Path
from projetv0_voice import owned_pilot_budget_cli as cli
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.persistence.writer import PersistenceWriter
from tests.integration.test_fixed_v2_process import KEY, NOW
from tests.unit.test_owned_pilot_budget import budget_policy, observation

async def main():
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
                               utcnow=lambda: NOW, failpoint=stuck)
    writer.set_owned_budget(budget_policy())
    owner = asyncio.create_task(writer.run())
    assert await writer.wait_ready()
    asyncio.create_task(writer.owned_budget_request("enroll", observation=observation(),
                                                   predecessor_retired=True))
    await entered.wait()
    cli._WRITER_CLOSE_SECONDS = .03
    await cli._close_operator_writer(writer, owner)
    raise AssertionError("unjoined owner returned")

asyncio.run(main())
"""
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
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
        assert db.execute("SELECT count(*) FROM owned_pilot_budget").fetchone() == (0,)
