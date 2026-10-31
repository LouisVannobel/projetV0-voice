"""Offline provider boundary tests; all identities and credentials are synthetic."""

import asyncio
import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr

NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)
EXPIRY = datetime(2026, 12, 1, tzinfo=UTC)
OR_KEY = "synthetic-inference-credential"
TEL_KEY = "synthetic-telephony-credential"
PRODUCTS = (
    "call-control", "media-streaming", "call-control-features", "recording",
    "sip-trunking", "text-to-speech", "speech-to-text",
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def api():
    try:
        return importlib.import_module("projetv0_voice.budget_observation")
    except ModuleNotFoundError:
        pytest.fail("The budget observation adapter is not implemented")


def identity(**changes):
    fields = dict(
        openrouter_credential_sha256=sha(OR_KEY),
        openrouter_account_sha256=sha("router-account"),
        telnyx_credential_sha256=sha(TEL_KEY),
        telnyx_account_sha256=sha("telnyx-account"),
        did="+33123456789", connection_id="123456",
        accounting_start_utc=datetime(2026, 9, 30, tzinfo=UTC),
        openrouter_expires_at_utc=EXPIRY,
    )
    fields.update(changes)
    return api().BudgetObservationIdentity(**fields)


def paged(rows, page=1, pages=1, total=None):
    return {"data": rows, "meta": {"page_number": page, "page_size": 100,
            "total_pages": pages, "total_results": len(rows) if total is None else total}}


def response_for(request):
    path = request.url.path
    if path == "/api/v1/key":
        return {"data": {"creator_user_id": "router-account", "limit": 5,
                "limit_remaining": "4.876543210987654321", "limit_reset": None,
                "usage": "0.123456789012345679", "expires_at": EXPIRY.isoformat(),
                "is_management_key": False}}
    if path == "/api/v1/credits":
        return {"data": {"total_credits": "2.000000000000000001", "total_usage": "0.5"}}
    if path == "/v2/charges_breakdown":
        return {"currency": "USD", "user_id": "telnyx-account",
                "start_date": request.url.params["start_date"],
                "end_date": request.url.params["end_date"],
                "results": [{"services": [{"cost": "-0.2"}, {"cost": "0.1"}]}]}
    if path == "/v2/phone_numbers":
        return paged([{"id": "number-1", "phone_number": "+33123456789",
                       "connection_id": "123456", "status": "active"}])
    if path == "/v2/usage_reports":
        return paged([{"currency": "USD", "cost": "0.01"},
                      {"currency": "USD", "cost": "-9"}])
    if path == "/v2/balance":
        return {"data": {"currency": "USD", "balance": "3.75"}}
    if path == "/v2/connections/123456/active_calls":
        return {"data": [{"call_control_id": "sensitive-call-control-id"}],
                "meta": {"total_items": 1, "cursors": {"after": None, "before": None},
                         "next": None, "previous": None}}
    raise AssertionError(f"Unexpected request path {path}")


async def collect(handler=None, *, expected=None, utcnow=lambda: NOW, monotonic=lambda: 100):
    module = api()
    requests = []

    async def transport(request):
        requests.append(request)
        if handler:
            result = handler(request)
            if result is not None:
                if isinstance(result, httpx.Response):
                    return result
                return httpx.Response(200, json=result)
        return httpx.Response(200, json=response_for(request))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport),
                                follow_redirects=True) as client:
        result = await module.collect_budget_observation(
            client, identity=expected or identity(), openrouter_key=SecretStr(OR_KEY),
            telnyx_key=SecretStr(TEL_KEY), utcnow=utcnow, monotonic=monotonic,
        )
        assert not client.is_closed
    return result, requests


@pytest.mark.asyncio
async def test_exact_facts_and_origin_scoped_gets_from_non_management_key():
    observation, requests = await collect()
    assert observation.openrouter_reported_spend_usd == Decimal("0.123456789012345679")
    assert observation.openrouter_available_cash_usd == Decimal("1.500000000000000001")
    assert observation.openrouter_limit_usd == Decimal(5)
    assert observation.openrouter_limit_remaining_usd == Decimal("4.876543210987654321")
    assert observation.openrouter_limit_reset is None
    assert observation.openrouter_expires_at_utc == EXPIRY
    assert observation.telnyx_reported_spend_usd == Decimal("0.37")
    assert observation.telnyx_available_cash_usd == Decimal("3.75")
    assert observation.active_call_control_ids == frozenset({"sensitive-call-control-id"})
    assert observation.identity == identity()
    assert observation.collection_started_at_utc == NOW
    assert observation.collection_finished_at_utc == NOW
    assert len(observation.provenance_sha256) == 64
    assert len(requests) == 13
    assert {r.url.params["product"] for r in requests
            if r.url.path == "/v2/usage_reports"} == set(PRODUCTS)
    for request in requests:
        assert request.method == "GET"
        assert request.url.scheme == "https"
        expected = OR_KEY if request.url.host == "openrouter.ai" else TEL_KEY
        assert request.url.host in {"openrouter.ai", "api.telnyx.com"}
        assert request.headers["authorization"] == f"Bearer {expected}"
        assert not request.content
        if request.url.path == "/v2/usage_reports":
            assert request.url.params["page_size"] == "100"
            assert "page[number]" not in request.url.params
            assert request.url.params["start_date"] == "2026-09-30T00:00:00Z"
            assert request.url.params["end_date"] == "2026-10-19T12:00:00Z"
            assert request.url.params["dimensions"] == "currency"
            assert request.url.params["metrics"] == "cost"
    assert OR_KEY not in repr(observation)
    assert TEL_KEY not in repr(observation)
    assert "sensitive-call-control-id" not in repr(observation)
    assert "+33123456789" not in repr(observation)
    with pytest.raises(FrozenInstanceError):
        observation.telnyx_available_cash_usd = Decimal(99)


@pytest.mark.asyncio
async def test_freshness_is_acquisition_start_and_digest_changes_with_body():
    clock = iter([NOW, NOW + timedelta(seconds=10)])
    observation, _ = await collect(utcnow=lambda: next(clock))
    assert observation.collection_started_at_utc == NOW
    assert observation.collection_finished_at_utc == NOW + timedelta(seconds=10)
    other, _ = await collect(lambda r: {"data": {"currency": "USD", "balance": "4"}}
                             if r.url.path == "/v2/balance" else None)
    assert observation.provenance_sha256 != other.provenance_sha256


@pytest.mark.asyncio
async def test_complete_active_call_ids_are_kept_without_granting_authority():
    def handler(request):
        if request.url.path == "/v2/connections/123456/active_calls":
            return {"data": [{"call_control_id": "call-1"}, {"call_control_id": "call-2"}],
                    "meta": {"total_items": 2, "cursors": {"after": None, "before": None},
                             "next": None, "previous": None}}
    observation, _ = await collect(handler)
    assert observation.telnyx_reported_spend_usd == Decimal("0.37")
    assert observation.active_call_control_ids == frozenset({"call-1", "call-2"})


@pytest.mark.asyncio
@pytest.mark.parametrize(("path", "field", "bad"), [
    ("/api/v1/key", "creator_user_id", "wrong"),
    ("/api/v1/key", "limit", "6"), ("/api/v1/key", "limit_reset", "daily"),
    ("/api/v1/key", "expires_at", "2027-01-01T00:00:00Z"),
    ("/api/v1/key", "usage", True), ("/api/v1/key", "usage", "NaN"),
    ("/api/v1/key", "usage", "Infinity"), ("/api/v1/key", "usage", "-1"),
    ("/api/v1/key", "usage", "1e999999"),
    ("/api/v1/credits", "total_credits", None),
    ("/api/v1/credits", "total_usage", False),
    ("/v2/charges_breakdown", "currency", "EUR"),
    ("/v2/charges_breakdown", "user_id", "wrong"),
    ("/v2/balance", "currency", "EUR"), ("/v2/balance", "balance", "NaN"),
])
async def test_bad_facts_stop_requests_immediately_with_constant_error(path, field, bad):
    seen_bad = False
    seen = []

    def handler(request):
        nonlocal seen_bad
        seen.append(request.url.path)
        if request.url.path == path:
            seen_bad = True
            body = response_for(request)
            target = body if path == "/v2/charges_breakdown" else body["data"]
            target[field] = bad
            return body

    with pytest.raises(api().BudgetObservationUnavailable) as error:
        await collect(handler)
    assert str(error.value) == "Budget observation unavailable"
    assert OR_KEY not in repr(error.value)
    assert TEL_KEY not in repr(error.value)
    assert seen_bad
    assert seen[-1] == path
    assert seen.count(path) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field,bad", [("phone_number", "+33999999999"),
                                      ("connection_id", "wrong"), ("status", "pending")])
async def test_wrong_did_binding_denies_before_any_later_request(field, bad):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/v2/phone_numbers":
            body = response_for(request)
            body["data"][0][field] = bad
            return body

    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(handler)
    assert seen[-1] == "/v2/phone_numbers"
    assert seen.count("/v2/phone_numbers") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"openrouter_credential_sha256": "0" * 64},
                                    {"telnyx_credential_sha256": "0" * 64}])
async def test_wrong_local_credentials_send_nothing(changes):
    seen = []

    def handler(request):
        seen.append(request)
    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(handler, expected=identity(**changes))
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing_cash", "403", "redirect", "oversized",
                                  "duplicate", "timeout", "transport", "bad_json"])
async def test_unavailable_responses_fail_closed_without_raw_body(kind):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path != "/api/v1/credits":
            return None
        if kind == "missing_cash":
            return {"data": {}}
        if kind == "403":
            return httpx.Response(403, text=OR_KEY)
        if kind == "redirect":
            return httpx.Response(302, headers={"location": "https://other.invalid/stolen"})
        if kind == "oversized":
            return httpx.Response(200, content=b" " * (2_097_152 + 1))
        if kind == "duplicate":
            return httpx.Response(200, content=b'{"data":{},"data":{"total_credits":5}}')
        if kind == "bad_json":
            return httpx.Response(200, text=OR_KEY)
        if kind == "timeout":
            raise httpx.ReadTimeout(OR_KEY)
        raise httpx.ConnectError(TEL_KEY)

    with pytest.raises(api().BudgetObservationUnavailable) as error:
        await collect(handler)
    assert seen == ["/api/v1/key", "/api/v1/credits"]
    assert str(error.value) == "Budget observation unavailable"
    assert error.value.__suppress_context__


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing_page", "next", "previous", "duplicate_call"])
async def test_incomplete_or_inconsistent_pagination_is_refused(kind):
    def handler(request):
        if request.url.path != "/v2/connections/123456/active_calls":
            return None
        rows = [{"call_control_id": "first"}]
        if kind == "duplicate_call":
            rows.append(rows[0])
        return {"data": rows, "meta": {
            "total_items": 2 if kind == "missing_page" else len(rows),
            "cursors": {"after": None, "before": None},
            "next": "opaque" if kind == "next" else None,
            "previous": "opaque" if kind == "previous" else None,
        }}

    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(handler)


@pytest.mark.asyncio
async def test_cancellation_propagates_and_stops_collection():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await collect(handler)
    assert seen == ["/api/v1/key"]


@pytest.mark.asyncio
async def test_total_deadline_and_reversed_utc_are_refused():
    times = iter([0, 16])
    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(monotonic=lambda: next(times))
    times_utc = iter([NOW, NOW - timedelta(seconds=1)])
    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(utcnow=lambda: next(times_utc))


@pytest.mark.asyncio
async def test_json_numeric_tokens_preserve_decimal_precision():
    def handler(request):
        if request.url.path == "/api/v1/credits":
            return httpx.Response(200, content=(
                b'{"data":{"total_credits":2.000000000000000001,"total_usage":0.5}}'))
    observation, _ = await collect(handler)
    assert observation.openrouter_available_cash_usd == Decimal("1.500000000000000001")


@pytest.mark.asyncio
async def test_documented_charges_data_wrapper_has_same_arithmetic():
    observation, _ = await collect(lambda r: {"data": response_for(r)}
                                   if r.url.path == "/v2/charges_breakdown" else None)
    assert observation.telnyx_reported_spend_usd == Decimal("0.37")


@pytest.mark.asyncio
async def test_original_accounting_scope_is_chunked_without_gaps_after_31_days():
    now = datetime(2026, 11, 15, 12, tzinfo=UTC)

    def handler(request):
        if request.url.path == "/v2/charges_breakdown":
            return httpx.Response(200, json=response_for(request), headers={
                "x-ratelimit-remaining": "1", "x-ratelimit-reset": "1",
            })

    observation, requests = await collect(handler, utcnow=lambda: now)
    usage = [r for r in requests if r.url.path == "/v2/usage_reports"
             and r.url.params["product"] == "call-control"]
    assert [(r.url.params["start_date"], r.url.params["end_date"]) for r in usage] == [
        ("2026-09-30T00:00:00Z", "2026-10-31T00:00:00Z"),
        ("2026-10-31T00:00:00Z", "2026-11-15T12:00:00Z"),
    ]
    charges = [r for r in requests if r.url.path == "/v2/charges_breakdown"]
    assert [(r.url.params["start_date"], r.url.params["end_date"]) for r in charges] == [
        ("2026-09-30", "2026-10-31"), ("2026-10-31", "2026-11-16"),
    ]
    assert observation.telnyx_reported_spend_usd == Decimal("0.74")


class CollectionClock:
    def __init__(self):
        self.tick = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.tick

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.tick += seconds


@pytest.mark.asyncio
@pytest.mark.parametrize("header_prefixes", [("x-ratelimit",), ("ratelimit",),
                                           ("x-ratelimit", "ratelimit")])
@pytest.mark.parametrize("body_seconds", [0, 0.4, 1.2])
async def test_charge_ranges_respect_provider_window_from_header_receipt(
    monkeypatch, header_prefixes, body_seconds,
):
    clock = CollectionClock()
    monkeypatch.setattr(api().asyncio, "sleep", clock.sleep)
    charge_ticks = []

    class ChargeBody(httpx.AsyncByteStream):
        def __init__(self, request):
            self.body = json.dumps(response_for(request)).encode()

        async def __aiter__(self):
            clock.tick += body_seconds
            yield self.body

    def handler(request):
        if request.url.path == "/v2/charges_breakdown":
            charge_ticks.append(clock.tick)
            if len(charge_ticks) > 1 and charge_ticks[-1] - charge_ticks[-2] < 1:
                return httpx.Response(429, text="synthetic exhausted charge-report window")
            # The final response needs no metadata because no further range is requested.
            headers = {f"{prefix}-{name}": value
                       for prefix in header_prefixes
                       for name, value in (("limit", "1, 1;w=1"), ("remaining", "0"),
                                           ("reset", "1"))} if len(charge_ticks) == 1 else {}
            return httpx.Response(200, headers=headers, stream=ChargeBody(request))

    now = datetime(2026, 10, 31, 12, tzinfo=UTC)
    observation, requests = await collect(handler, utcnow=lambda: now,
                                          monotonic=clock.monotonic)
    assert charge_ticks == pytest.approx([100, 100 + max(1, body_seconds)])
    assert clock.sleeps == pytest.approx([1 - body_seconds] if body_seconds < 1 else [])
    charges = [r for r in requests if r.url.path == "/v2/charges_breakdown"]
    assert [(r.url.params["start_date"], r.url.params["end_date"]) for r in charges] == [
        ("2026-09-30", "2026-10-31"), ("2026-10-31", "2026-11-01"),
    ]
    assert {r.url.params["product"] for r in requests
            if r.url.path == "/v2/usage_reports"} == set(PRODUCTS)
    assert observation.telnyx_reported_spend_usd == Decimal("0.74")


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {},
    {"x-ratelimit-remaining": "0"},
    {"x-ratelimit-reset": "1"},
    {"x-ratelimit-remaining": "-1", "x-ratelimit-reset": "1"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "-1"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1.5"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "NaN"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "Infinity"},
    {"x-ratelimit-remaining": "0, 1", "x-ratelimit-reset": "1"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1", "ratelimit-reset": "2"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1", "ratelimit-remaining": "1"},
    {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1", "ratelimit-reset": "invalid"},
])
async def test_unsafe_charge_pacing_metadata_refuses_before_next_range(headers):
    charges = []

    def handler(request):
        if request.url.path == "/v2/charges_breakdown":
            charges.append(request)
            return httpx.Response(200, json=response_for(request), headers=headers)

    with pytest.raises(api().BudgetObservationUnavailable) as error:
        await collect(handler, utcnow=lambda: datetime(2026, 10, 31, 12, tzinfo=UTC))
    assert str(error.value) == "Budget observation unavailable"
    assert error.value.__suppress_context__
    assert len(charges) == 1


@pytest.mark.asyncio
async def test_single_charge_range_does_not_require_pacing_metadata():
    def handler(request):
        if request.url.path == "/v2/charges_breakdown":
            return httpx.Response(200, json=response_for(request), headers={
                "x-ratelimit-remaining": "invalid", "x-ratelimit-reset": "invalid",
            })

    observation, requests = await collect(handler)
    assert observation.telnyx_reported_spend_usd == Decimal("0.37")
    assert len([r for r in requests if r.url.path == "/v2/charges_breakdown"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["does_not_fit", "sleep_exhausts_deadline", "cancelled"])
async def test_charge_cooldown_stops_before_second_get_on_deadline_or_cancellation(
    monkeypatch, kind,
):
    clock = CollectionClock()
    charges = []

    async def sleep(seconds):
        await clock.sleep(seconds)
        if kind == "cancelled":
            raise asyncio.CancelledError()
        if kind == "sleep_exhausts_deadline":
            clock.tick = 115

    monkeypatch.setattr(api().asyncio, "sleep", sleep)

    def handler(request):
        if request.url.path == "/v2/charges_breakdown":
            charges.append(request)
            if kind == "does_not_fit":
                clock.tick = 114.5
            return httpx.Response(200, json=response_for(request), headers={
                "x-ratelimit-remaining": "0", "x-ratelimit-reset": "1",
            })

    expected = asyncio.CancelledError if kind == "cancelled" else api().BudgetObservationUnavailable
    with pytest.raises(expected):
        await collect(handler, utcnow=lambda: datetime(2026, 10, 31, 12, tzinfo=UTC),
                      monotonic=clock.monotonic)
    assert len(charges) == 1
    assert clock.sleeps == ([] if kind == "does_not_fit" else [1])


@pytest.mark.asyncio
@pytest.mark.parametrize("wake_tick", [100.999, 101, 101.001])
async def test_charge_cooldown_rechecks_not_before_after_timer_wakes(monkeypatch, wake_tick):
    clock = CollectionClock()
    charge_ticks = []

    async def sleep(seconds):
        await clock.sleep(seconds)
        clock.tick = wake_tick

    monkeypatch.setattr(api().asyncio, "sleep", sleep)

    def handler(request):
        if request.url.path == "/v2/charges_breakdown":
            charge_ticks.append(clock.tick)
            return httpx.Response(200, json=response_for(request), headers={
                "x-ratelimit-remaining": "0", "x-ratelimit-reset": "1",
            })

    now = datetime(2026, 10, 31, 12, tzinfo=UTC)
    if wake_tick < 101:
        with pytest.raises(api().BudgetObservationUnavailable) as error:
            await collect(handler, utcnow=lambda: now, monotonic=clock.monotonic)
        assert str(error.value) == "Budget observation unavailable"
        assert error.value.__suppress_context__
        assert charge_ticks == [100]
    else:
        observation, _ = await collect(handler, utcnow=lambda: now, monotonic=clock.monotonic)
        assert charge_ticks == [100, wake_tick]
        assert observation.telnyx_reported_spend_usd == Decimal("0.74")
    assert clock.sleeps == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v2/usage_reports", "/v2/phone_numbers"])
async def test_unqualified_multiple_pages_are_refused(path):
    def handler(request):
        if request.url.path == path:
            body = response_for(request)
            body["meta"]["total_pages"] = 2
            return body
    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(handler)


@pytest.mark.asyncio
async def test_empty_usage_and_active_call_lists_are_complete():
    def handler(request):
        if request.url.path == "/v2/usage_reports":
            return {"data": [], "meta": {"total_pages": 1, "total_results": 0}}
        if request.url.path == "/v2/connections/123456/active_calls":
            return {"data": [], "meta": {"total_items": 0,
                    "cursors": {"after": None, "before": None},
                    "next": None, "previous": None}}
    observation, _ = await collect(handler)
    assert observation.telnyx_reported_spend_usd == Decimal("0.3")
    assert observation.active_call_control_ids == frozenset()


@pytest.mark.asyncio
async def test_complete_active_page_can_have_boundary_cursors():
    def handler(request):
        if request.url.path == "/v2/connections/123456/active_calls":
            body = response_for(request)
            body["meta"]["cursors"] = {"after": "last-row", "before": "first-row"}
            return body
    observation, _ = await collect(handler)
    assert observation.active_call_control_ids == frozenset({"sensitive-call-control-id"})


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v2/usage_reports", "/v2/charges_breakdown"])
@pytest.mark.parametrize("bad", [True, "NaN", "Infinity", "1e1000"])
async def test_invalid_costs_cannot_become_spending(path, bad):
    def handler(request):
        if request.url.path == path:
            body = response_for(request)
            if path.endswith("usage_reports"):
                body["data"][0]["cost"] = bad
            else:
                body["results"][0]["services"][0]["cost"] = bad
            return body
    with pytest.raises(api().BudgetObservationUnavailable):
        await collect(handler)
