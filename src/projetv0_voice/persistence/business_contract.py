"""Bounded Sparra result encoding using the existing owned AES-GCM keyring."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Literal
from uuid import UUID

from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import MessageResultEnvelopeV1, MessageResultV1, _text


def validate_turn_text(text: str) -> str:
    """Validate new captured text before encryption; never rewrite a legacy turn."""
    if not isinstance(text, str):
        raise ValueError("turn text must be a string")
    _text(text, 16384)
    if len(text.encode("utf-8")) > 16384:
        raise ValueError("turn text exceeds UTF8 bound")
    return text


def encrypt_message_result(
    result: MessageResultV1,
    *,
    call_id: UUID,
    keyring: CryptoKeyring,
    authenticated_turns: Mapping[UUID, Literal["user", "assistant"]],
) -> MessageResultEnvelopeV1:
    """Caller supplies only authenticated retained turns belonging to this call.

    The single writer must durably freeze this returned envelope before dispatch;
    this encoding boundary does not manufacture provenance or persist/replay it.
    """
    if not isinstance(result, MessageResultV1) or not isinstance(call_id, UUID):
        raise ValueError("result and call identity must be typed")
    if len(authenticated_turns) > 200 or any(
        not isinstance(turn_id, UUID) or role not in {"user", "assistant"}
        for turn_id, role in authenticated_turns.items()
    ):
        raise ValueError("authenticated turn set exceeds the native reader bound")
    if any(authenticated_turns.get(item.turn_id) != item.role for item in result.evidence):
        raise ValueError("result evidence must match authenticated retained turns")
    plaintext = result.model_dump_json().encode("utf-8")
    if len(plaintext) > 8192:
        raise ValueError("result exceeds UTF8 bound")
    encrypted = keyring.encrypt(plaintext, aad=f"result:{call_id}".encode("ascii"))
    return MessageResultEnvelopeV1(
        schema_version=1,
        crypto_version=1,
        key_version=encrypted.key_version,
        nonce_b64=base64.b64encode(encrypted.nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(encrypted.ciphertext).decode("ascii"),
    )
