from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
from pydantic import SecretStr

from projetv0_voice import production_wiring
from tests.unit.test_config import manifest_data, write_bundle
from tests.unit.test_inference_response_bounds import _profile
from tests.unit.test_production_wiring import _settings


@pytest.mark.asyncio
@pytest.mark.parametrize("sparra", [True, False])
async def test_actual_production_factories_apply_audio_limits_only_to_sparra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sparra: bool,
) -> None:
    monkeypatch.setattr(production_wiring, "_validate_artifacts", lambda settings: None)
    extension = {"sparra": {
        "schema_version": 1, "connection_id": "fixture", "original_forward_line_e164": None,
        "qualified_transfer_destination_e164": None,
    }} if sparra else {}
    bundle = write_bundle(tmp_path / "bundle", manifest_data(
        max_concurrent_calls=1, transcript_retention_days=30 if sparra else 7, **extension,
    ))
    settings = _settings(agent_bundle_path=PurePosixPath(str(bundle)))
    factories = production_wiring.build_production_factories(settings)
    factories.validate_artifacts(settings)
    factories.load_manifest(settings)
    inference = factories.inference_factory(SecretStr("offline-unit-key"), _profile(), "fr-FR")
    http = inference.stt_http_client_factory()
    stt = inference.stt_factory(http)
    tts = inference.tts_factory()
    try:
        assert getattr(stt, "_sparra_audio_limits", False) is sparra
        assert getattr(tts, "_enforce_consumption_limits", False) is sparra
        assert stt._client.max_retries == (0 if sparra else 2)
    finally:
        await stt.cleanup()
        await tts.cleanup()
        await http.aclose()
