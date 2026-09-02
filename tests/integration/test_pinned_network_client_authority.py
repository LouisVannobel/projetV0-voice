from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from projetv0_voice.inference.openrouter_tts import OpenRouterTTSService
from projetv0_voice.production_wiring import _inference_factories
from projetv0_voice.qualified_profile import QualifiedDeploymentProfileV1
from projetv0_voice.telnyx.call_control import CallControlClient


def _profile() -> QualifiedDeploymentProfileV1:
    return QualifiedDeploymentProfileV1.model_validate_json(
        Path("tests/fixtures/qualified-deployment-profile-v1.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.asyncio
async def test_authenticated_clients_ignore_proxy_and_ca_environment_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")

    environmental_control = httpx.AsyncClient()
    try:
        assert any(
            type(getattr(transport, "_pool", None)).__name__ == "AsyncHTTPProxy"
            for transport in environmental_control._mounts.values()  # noqa: SLF001
        )
    finally:
        await environmental_control.aclose()

    missing_ca = tmp_path / "missing-ca.pem"
    missing_ca_dir = tmp_path / "missing-ca-dir"
    monkeypatch.setenv("SSL_CERT_FILE", str(missing_ca))
    monkeypatch.setenv("SSL_CERT_DIR", str(missing_ca_dir))
    with pytest.raises(FileNotFoundError):
        httpx.AsyncClient()

    stt_client: httpx.AsyncClient | None = None
    llm: Any | None = None
    tts: OpenRouterTTSService | None = None
    call_control: CallControlClient | None = None
    try:
        factories = _inference_factories(SecretStr("openrouter-test-key"), _profile(), "fr")
        stt_client = factories.stt_http_client_factory()  # type: ignore[assignment]
        stt = factories.stt_factory(stt_client)
        llm = factories.llm_factory()
        tts = factories.tts_factory()
        call_control = CallControlClient(api_key="telnyx-test-key")

        transports = (
            stt._client._client,  # noqa: SLF001
            llm._client._client,  # noqa: SLF001
            tts._client,  # noqa: SLF001
            call_control._client._client,  # noqa: SLF001
        )
        assert stt._client._client is stt_client  # noqa: SLF001
        assert tts._owns_client is True  # noqa: SLF001
        assert all(transport._mounts == {} for transport in transports)  # noqa: SLF001
        assert all(transport._trust_env is False for transport in transports)  # noqa: SLF001

        def pool_limits(transport: Any) -> tuple[int, int, float | None]:
            pool = transport._transport._pool  # noqa: SLF001
            return (
                pool._max_connections,  # noqa: SLF001
                pool._max_keepalive_connections,  # noqa: SLF001
                pool._keepalive_expiry,  # noqa: SLF001
            )

        assert [
            pool_limits(transport) for transport in transports
        ] == [
            (1000, 100, 5.0),
            (1000, 100, None),
            (100, 20, 5.0),
            (100, 20, 5.0),
        ]

        close_counts: dict[int, int] = {}
        original_aclose = httpx.AsyncClient.aclose

        async def counted_aclose(client: httpx.AsyncClient) -> None:
            close_counts[id(client)] = close_counts.get(id(client), 0) + 1
            await original_aclose(client)

        monkeypatch.setattr(httpx.AsyncClient, "aclose", counted_aclose)
        await stt_client.aclose()
        await llm._client.close()  # noqa: SLF001
        await tts.cleanup()
        await tts.cleanup()
        await call_control.aclose()
        await call_control.aclose()

        assert [close_counts.get(id(transport), 0) for transport in transports] == [
            1,
            1,
            1,
            1,
        ]
        stt_client = None
        llm = None
        tts = None
        call_control = None
    finally:
        if call_control is not None:
            with contextlib.suppress(Exception):
                await call_control.aclose()
        if tts is not None:
            with contextlib.suppress(Exception):
                await tts.cleanup()
        if llm is not None:
            with contextlib.suppress(Exception):
                await llm._client.close()  # noqa: SLF001
        if stt_client is not None:
            with contextlib.suppress(Exception):
                await stt_client.aclose()
