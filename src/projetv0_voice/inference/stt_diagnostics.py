"""Closed, content-free facts for one executing native STT segment (stdlib only)."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal

STT_DIAGNOSTIC_KEY = "projetv0_stt_request_diagnostic"
PAID_CHECKS = frozenset({"not_checked", "allowed", "denied", "check_error"})
RESPONSE_FAMILIES = frozenset({"none", "2xx", "3xx", "4xx", "5xx", "other"})
HTTPX_FAILURES = frozenset({"unobserved", "timeout", "transport_error", "other_request_error"})
# At most 96 diagnostic combinations per existing reason; no content-derived labels.
STT_DIAGNOSTIC_CARDINALITY = 4 * 6 * 4

type PaidCheck = Literal["not_checked", "allowed", "denied", "check_error"]
type HttpxFailure = Literal["unobserved", "timeout", "transport_error", "other_request_error"]


@dataclass(frozen=True, slots=True)
class SttRequestDiagnostic:
    paid_check: str = "not_checked"
    response_family: str = "none"
    httpx_failure: str = "unobserved"


_active: ContextVar[SttRequestDiagnostic | None] = ContextVar(
    "stt_request_diagnostic", default=None
)


@contextmanager
def stt_request_scope() -> Iterator[None]:
    token = _active.set(SttRequestDiagnostic())
    try:
        yield
    finally:
        _active.reset(token)


def observe_paid_check(value: PaidCheck) -> None:
    if (record := _active.get()) is not None:
        _active.set(replace(record, paid_check=value))


def observe_response_status(status: int) -> None:
    if (record := _active.get()) is not None:
        family = f"{status // 100}xx" if 200 <= status < 600 else "other"
        _active.set(replace(record, response_family=family))


def observe_httpx_failure(value: HttpxFailure) -> None:
    if (record := _active.get()) is not None:
        _active.set(replace(record, httpx_failure=value))


def snapshot_stt_request() -> SttRequestDiagnostic | None:
    return _active.get()


def diagnostic_attributes(value: object) -> dict[str, str]:
    """Map absent/malformed snapshots to closed unknowns, never supplied values."""
    if (
        type(value) is not SttRequestDiagnostic
        or type(value.paid_check) is not str or value.paid_check not in PAID_CHECKS
        or type(value.response_family) is not str or value.response_family not in RESPONSE_FAMILIES
        or type(value.httpx_failure) is not str or value.httpx_failure not in HTTPX_FAILURES
    ):
        value = SttRequestDiagnostic()
    return {
        "paid_check": value.paid_check,
        "response_family": value.response_family,
        "httpx_failure": value.httpx_failure,
    }
