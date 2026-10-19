"""Explicit operator enrollment/suspension. Runtime startup never calls this module."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from projetv0_voice.persistence.writer import PersistenceWriter
from projetv0_voice.production_wiring import _budget_factory, build_production_factories
from projetv0_voice.qualified_profile import (
    QualifiedDeploymentProfileV1,
    canonical_qualified_profile_sha256,
)
from projetv0_voice.runtime_config import capture_runtime_environment, parse_runtime_settings


async def operate(action: str, *, predecessor_retired: bool) -> bool:
    settings = parse_runtime_settings(capture_runtime_environment(os.environ))
    if settings.runtime_mode != "strict":
        raise RuntimeError("owned_budget_operator_refused")
    factories = build_production_factories(settings)
    factories.validate_artifacts(settings)
    manifest = factories.load_manifest(settings)
    selection = factories.load_profile(settings, manifest, datetime.now(UTC))
    if manifest.sparra is None or not isinstance(selection.profile, QualifiedDeploymentProfileV1):
        raise RuntimeError("owned_budget_operator_refused")
    keyring = factories.load_keyring(settings)
    writer = PersistenceWriter(
        Path(str(settings.sqlite_path)),
        keyring,
        contract_version=manifest.sparra.operation_contract_version,
        process_agent_id=manifest.agent_id,
        process_deployment_id=settings.deployment_id,
    )
    or_key = factories.read_secret(settings.openrouter_api_key_file)
    tel_key = factories.read_secret(settings.telnyx_api_key_file)
    gate = _budget_factory(
        settings, writer, or_key, tel_key, lambda: datetime.now(UTC), time.monotonic
    )
    p = gate.policy
    if (
        p.qualified_profile_sha256 != canonical_qualified_profile_sha256(selection.profile)
        or p.runtime_contract_sha256 != settings.runtime_contract_sha256
        or p.agent_bundle_sha256 != settings.agent_bundle_sha256
        or p.inference_profile_sha256 != settings.inference_profile_sha256
        or p.image_digest != settings.image_digest
        or p.deployment_id != settings.deployment_id
        or p.identity.did != manifest.dids[0]
        or p.identity.connection_id != manifest.sparra.connection_id
        or p.identity.openrouter_credential_sha256
        != hashlib.sha256(or_key.get_secret_value().encode()).hexdigest()
        or p.identity.telnyx_credential_sha256
        != hashlib.sha256(tel_key.get_secret_value().encode()).hexdigest()
    ):
        raise RuntimeError("owned_budget_operator_refused")
    owner = asyncio.create_task(writer.run())
    try:
        if not await writer.wait_ready():
            return False
        if action == "suspend-restored":
            return await writer.owned_budget_request("suspend") is True
        if action != "enroll" or not predecessor_retired:
            return False
        fact = await gate.collect()
        return (
            await writer.owned_budget_request("enroll", observation=fact, predecessor_retired=True)
            is True
        )
    finally:
        if not writer.is_degraded:
            await writer.drain(5)
        await owner


def main() -> int:
    parser = argparse.ArgumentParser(description="Original Sparra budget operator transition")
    parser.add_argument("action", choices=("enroll", "suspend-restored"))
    parser.add_argument(
        "--predecessor-retired",
        action="store_true",
        help="Confirm predecessor is fenced and prior exposure is settled",
    )
    args = parser.parse_args()
    if args.action == "enroll" and not args.predecessor_retired:
        parser.error("enroll requires --predecessor-retired")
    try:
        accepted = asyncio.run(operate(args.action, predecessor_retired=args.predecessor_retired))
    except Exception:
        accepted = False
    print("owned_budget_transition_accepted" if accepted else "owned_budget_transition_refused")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
