"""Bounded original-account metadata reads. Facts confer no spending authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    import httpx

_MAX_BODY = 2_097_152
_PRODUCTS = (
    "call-control", "media-streaming", "call-control-features", "recording",
    "sip-trunking", "text-to-speech", "speech-to-text",
)


class BudgetObservationUnavailable(Exception):
    """Constant public failure; provider content must never become diagnostics."""

    def __init__(self) -> None:
        super().__init__("Budget observation unavailable")


@dataclass(frozen=True, repr=False)
class BudgetObservationIdentity:
    openrouter_credential_sha256: str
    openrouter_account_sha256: str
    telnyx_credential_sha256: str
    telnyx_account_sha256: str
    did: str
    connection_id: str
    accounting_start_utc: datetime
    openrouter_expires_at_utc: datetime


@dataclass(frozen=True, repr=False)
class BudgetObservation:
    collection_started_at_utc: datetime
    collection_finished_at_utc: datetime
    identity: BudgetObservationIdentity
    openrouter_reported_spend_usd: Decimal
    openrouter_available_cash_usd: Decimal
    openrouter_limit_usd: Decimal
    openrouter_limit_remaining_usd: Decimal
    openrouter_limit_reset: None
    openrouter_expires_at_utc: datetime
    telnyx_reported_spend_usd: Decimal
    telnyx_available_cash_usd: Decimal
    active_call_control_ids: frozenset[str]
    provenance_sha256: str


def _need(condition: bool) -> None:
    if not condition:
        raise BudgetObservationUnavailable()


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BudgetObservationUnavailable()
    _need(all(isinstance(key, str) for key in value))
    return dict(value)


def _rows(value: object) -> list[object]:
    if not isinstance(value, list):
        raise BudgetObservationUnavailable()
    return list(value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise BudgetObservationUnavailable()
    return value


def _amount(value: object) -> Decimal:
    _need(type(value) in {str, int, Decimal})
    text = str(value)
    _need(len(text) <= 128 and re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text,
                                         flags=re.ASCII) is not None)
    result = Decimal(text)
    parts = result.as_tuple()
    _need(result.is_finite() and len(parts.digits) <= 64)
    # These bounds allow exact addition/subtraction under the private 128-digit context.
    _need(isinstance(parts.exponent, int) and -24 <= parts.exponent <= 24)
    return result


def _utc(value: datetime) -> datetime:
    _need(isinstance(value, datetime) and value.tzinfo is not None
          and value.utcoffset() == timedelta(0))
    return value.astimezone(UTC)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _need(key not in result)
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise BudgetObservationUnavailable()


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


async def collect_budget_observation(
    client: httpx.AsyncClient,
    *,
    identity: BudgetObservationIdentity,
    openrouter_key: SecretStr,
    telnyx_key: SecretStr,
    utcnow: Callable[[], datetime],
    monotonic: Callable[[], float],
) -> BudgetObservation:
    """Read once, GET only, <=15s total, no redirects; caller owns the no-retry client.

    HTTPX transports/hooks are caller-owned and must not retry or log credentials/bodies.
    Cancellation propagates; all ordinary refusals have constant public diagnostics.
    """
    try:
        with localcontext() as context:
            context.prec = 128
            async with asyncio.timeout(15):
                return await _collect(client, identity, openrouter_key, telnyx_key,
                                      utcnow, monotonic)
    except asyncio.CancelledError:
        raise asyncio.CancelledError() from None
    except Exception:
        raise BudgetObservationUnavailable() from None


async def _collect(
    client: httpx.AsyncClient,
    identity: BudgetObservationIdentity,
    openrouter_key: SecretStr,
    telnyx_key: SecretStr,
    utcnow: Callable[[], datetime],
    monotonic: Callable[[], float],
) -> BudgetObservation:
    import httpx

    started = _utc(utcnow())
    start_tick = monotonic()
    _need(math.isfinite(start_tick))
    deadline = start_tick + 15
    accounting_start = _utc(identity.accounting_start_utc)
    expected_expiry = _utc(identity.openrouter_expires_at_utc)
    _need(accounting_start < started < expected_expiry)
    _need(accounting_start.time() == datetime.min.time())
    _need(re.fullmatch(r"\+[1-9][0-9]{1,14}", identity.did) is not None)
    _need(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity.connection_id) is not None)
    _need(re.fullmatch(r"[0-9a-f]{64}", identity.openrouter_account_sha256) is not None)
    _need(re.fullmatch(r"[0-9a-f]{64}", identity.telnyx_account_sha256) is not None)
    for secret, fingerprint in (
        (openrouter_key, identity.openrouter_credential_sha256),
        (telnyx_key, identity.telnyx_credential_sha256),
    ):
        value = secret.get_secret_value()
        _need(bool(value) and value.isascii() and value.strip() == value
              and not any(ord(char) < 33 or ord(char) == 127 for char in value))
        _need(hashlib.sha256(value.encode()).hexdigest() == fingerprint)

    provenance = hashlib.sha256()

    def remaining() -> float:
        tick = monotonic()
        _need(math.isfinite(tick) and start_tick <= tick < deadline)
        return deadline - tick

    async def get(
        provider: str, path: str, params: dict[str, str | int] | None = None,
    ) -> dict[str, object]:
        base, secret = (("https://openrouter.ai", openrouter_key) if provider == "openrouter"
                        else ("https://api.telnyx.com", telnyx_key))
        timeout = remaining()
        async with asyncio.timeout(timeout), client.stream(
            "GET", base + path, params=params,
            headers={"Authorization": "Bearer " + secret.get_secret_value()},
            auth=None, follow_redirects=False, timeout=timeout,
        ) as response:
            _need(response.status_code == 200)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                _need(len(body) + len(chunk) <= _MAX_BODY)
                body.extend(chunk)
                remaining()
        remaining()
        decoded: object = json.loads(body, parse_float=Decimal,
                                     parse_constant=_invalid_constant,
                                     object_pairs_hook=_unique_object)
        result = _object(decoded)
        # Length framing binds the complete ordered request/body sequence, without retaining it.
        url = str(httpx.URL(base + path, params=params)).encode()
        provenance.update(len(url).to_bytes(8, "big") + url + hashlib.sha256(body).digest())
        return result

    def complete_report(report: dict[str, object]) -> list[object]:
        rows = _rows(report["data"])
        meta = _object(report["meta"])
        total_pages, total_results = meta["total_pages"], meta["total_results"]
        _need(type(total_pages) is int and total_pages in {0, 1}
              and type(total_results) is int and total_results == len(rows)
              and len(rows) <= 100 and (total_pages != 0 or not rows))
        if "page_number" in meta:
            _need(type(meta["page_number"]) is int and meta["page_number"] == 1)
        return rows

    router = _object((await get("openrouter", "/api/v1/key"))["data"])
    _need(hashlib.sha256(_text(router["creator_user_id"]).encode()).hexdigest()
          == identity.openrouter_account_sha256)
    limit = _amount(router["limit"])
    _need(limit == 5 and router["limit_reset"] is None)
    expiry = _utc(datetime.fromisoformat(_text(router["expires_at"])))
    _need(expiry == expected_expiry)
    usage, limit_remaining = _amount(router["usage"]), _amount(router["limit_remaining"])
    _need(usage >= 0 and 0 <= limit_remaining <= limit)
    credits = _object((await get("openrouter", "/api/v1/credits"))["data"])
    total_credits, total_usage = _amount(credits["total_credits"]), _amount(credits["total_usage"])
    _need(total_credits >= 0 and total_usage >= 0)
    cash = total_credits - total_usage

    found_number = False

    def number_row(row: dict[str, object]) -> None:
        nonlocal found_number
        _need(not found_number and row["phone_number"] == identity.did
              and row["connection_id"] == identity.connection_id and row["status"] == "active")
        found_number = True

    numbers = await get("telnyx", "/v2/phone_numbers", {
        "filter[phone_number]": identity.did, "page[size]": 100,
    })
    for number in complete_report(numbers):
        number_row(_object(number))
    _need(found_number)

    spent = Decimal(0)

    def usage_row(row: dict[str, object]) -> None:
        nonlocal spent
        _need(row["currency"] == "USD")
        spent += max(Decimal(0), _amount(row["cost"]))

    # Telnyx reports accept at most 31 days. Preserve the original start forever.
    report_start = accounting_start
    while report_start < started:
        report_end = min(report_start + timedelta(days=31), started)
        for product in _PRODUCTS:
            report = await get("telnyx", "/v2/usage_reports", {
                "dimensions": "currency", "metrics": "cost", "product": product,
                "start_date": _stamp(report_start), "end_date": _stamp(report_end),
                "format": "json", "page_size": 100,
            })
            for row in complete_report(report):
                usage_row(_object(row))
        report_start = report_end

    charge_start = accounting_start.date()
    charge_until = (started + timedelta(days=1)).date()
    while charge_start < charge_until:
        charge_end = min(charge_start + timedelta(days=31), charge_until)
        charges = await get("telnyx", "/v2/charges_breakdown", {
            "start_date": charge_start.isoformat(), "end_date": charge_end.isoformat(),
            "format": "json",
        })
        # Original authenticated reader uses top-level fields; generated SDK uses data.
        if "data" in charges:
            _need("currency" not in charges and "results" not in charges)
            charges = _object(charges["data"])
        _need(charges["currency"] == "USD")
        _need(hashlib.sha256(_text(charges["user_id"]).encode()).hexdigest()
              == identity.telnyx_account_sha256)
        for group in _rows(charges["results"]):
            for service in _rows(_object(group)["services"]):
                spent += abs(_amount(_object(service)["cost"]))
        charge_start = charge_end

    balance = _object((await get("telnyx", "/v2/balance"))["data"])
    _need(balance["currency"] == "USD")
    wallet = _amount(balance["balance"])
    active: set[str] = set()

    def active_row(row: dict[str, object]) -> None:
        call_id = _text(row["call_control_id"])
        _need(call_id not in active)
        active.add(call_id)

    calls = await get("telnyx", f"/v2/connections/{identity.connection_id}/active_calls", {
        "page[size]": 100,
    })
    call_rows, meta = _rows(calls["data"]), _object(calls["meta"])
    cursors = _object(meta["cursors"])
    _need(type(meta["total_items"]) is int and meta["total_items"] == len(call_rows)
          and len(call_rows) <= 100 and meta["next"] is None and meta["previous"] is None)
    for direction in ("after", "before"):
        cursor = cursors[direction]
        _need(cursor is None or isinstance(cursor, str) and bool(cursor) and len(cursor) <= 1024)
    for call in call_rows:
        active_row(_object(call))
    remaining()
    finished = _utc(utcnow())
    _need(timedelta(0) <= finished - started <= timedelta(seconds=15) and finished < expiry)
    return BudgetObservation(
        collection_started_at_utc=started, collection_finished_at_utc=finished,
        identity=identity, openrouter_reported_spend_usd=usage,
        openrouter_available_cash_usd=cash, openrouter_limit_usd=limit,
        openrouter_limit_remaining_usd=limit_remaining, openrouter_limit_reset=None,
        openrouter_expires_at_utc=expiry, telnyx_reported_spend_usd=spent,
        telnyx_available_cash_usd=wallet, active_call_control_ids=frozenset(active),
        provenance_sha256=provenance.hexdigest(),
    )
