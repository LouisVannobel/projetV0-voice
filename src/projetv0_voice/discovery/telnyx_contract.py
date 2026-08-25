"""One-shot live conformance probe for the documented Telnyx handshake."""

from __future__ import annotations

import asyncio
import hmac
import math
from dataclasses import dataclass, field

from starlette.websockets import WebSocket

from projetv0_voice.telnyx.handshake import (
    TOKEN_LOCATOR_ID,
    LeaseAuthority,
    TelnyxHandshakeRejectedError,
    _capture_telnyx_handshake,
)


class TelnyxContractProbeError(RuntimeError):
    """A public constant-safe conformance probe failure."""


class TelnyxContractProbeRejectedError(TelnyxContractProbeError):
    """The documented live contract did not conform."""


class TelnyxContractProbeTimeoutError(TelnyxContractProbeError):
    """The single probe deadline elapsed."""


class TelnyxContractProbeUsedError(TelnyxContractProbeError):
    """The one-shot probe was already consumed."""


@dataclass(frozen=True, slots=True)
class ContractProbeResult:
    token_locator_id: str
    redacted_fixture_bytes: bytes = field(repr=False)
    fixture_sha256: str
    safe_summary: tuple[tuple[str, str | int], ...]

    def __repr__(self) -> str:
        return "ContractProbeResult()"


class TelnyxContractProbe:
    """Consume one synthetic connection and prove the frozen composite contract."""

    def __init__(self, *, lease_authority: LeaseAuthority, timeout_seconds: float) -> None:
        if (
            not isinstance(timeout_seconds, int | float)
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise TelnyxContractProbeError("telnyx_contract_probe_config_invalid")
        self._lease_authority = lease_authority
        self._timeout_seconds = float(timeout_seconds)
        self._used = False

    def __repr__(self) -> str:
        return "TelnyxContractProbe()"

    async def inspect(
        self, websocket: WebSocket, expected_token_digest: bytes
    ) -> ContractProbeResult:
        if self._used:
            raise TelnyxContractProbeUsedError("telnyx_contract_probe_used")
        self._used = True

        call_control_id: str | None = None
        token_digest: bytes | None = None
        abort_required = False
        result: ContractProbeResult | None = None
        timed_out = False
        rejected = False
        unexpected_failure = False
        cleanup_failed = False
        try:
            try:
                if not isinstance(expected_token_digest, bytes) or len(expected_token_digest) != 32:
                    raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
                async with asyncio.timeout(self._timeout_seconds):
                    captured = await _capture_telnyx_handshake(websocket)
                    call_control_id = captured.call_data.call_id
                    token_digest = captured.token_digest
                    if call_control_id is None or not hmac.compare_digest(
                        token_digest, expected_token_digest
                    ):
                        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")

                    abort_required = True
                    lease_claim = await self._lease_authority.claim_once(
                        call_control_id=call_control_id,
                        token_digest=token_digest,
                    )
                    if lease_claim is None:
                        abort_required = False
                        raise TelnyxHandshakeRejectedError("telnyx_handshake_rejected")
                    result = ContractProbeResult(
                        token_locator_id=TOKEN_LOCATOR_ID,
                        redacted_fixture_bytes=captured.redacted_fixture_bytes,
                        fixture_sha256=captured.fixture_sha256,
                        safe_summary=captured.safe_summary,
                    )
            except TimeoutError:
                timed_out = True
            except asyncio.CancelledError:
                raise
            except TelnyxHandshakeRejectedError:
                rejected = True
            except Exception:
                unexpected_failure = True
        finally:
            try:
                if abort_required and call_control_id is not None and token_digest is not None:
                    self._lease_authority.schedule_abort_if_matches(
                        call_control_id=call_control_id,
                        token_digest=token_digest,
                    )
            except Exception:
                cleanup_failed = True

        if cleanup_failed:
            raise TelnyxContractProbeError("telnyx_contract_probe_cleanup_failed")
        if timed_out:
            raise TelnyxContractProbeTimeoutError("telnyx_contract_probe_timeout")
        if rejected:
            raise TelnyxContractProbeRejectedError("telnyx_contract_probe_rejected")
        if unexpected_failure or result is None:
            raise TelnyxContractProbeError("telnyx_contract_probe_failed")
        return result


__all__ = [
    "ContractProbeResult",
    "TelnyxContractProbe",
    "TelnyxContractProbeError",
    "TelnyxContractProbeRejectedError",
    "TelnyxContractProbeTimeoutError",
    "TelnyxContractProbeUsedError",
]
