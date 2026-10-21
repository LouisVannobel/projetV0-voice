"""Finite original-pilot authority; metadata is evidence, never a bearer grant."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from typing import Literal, Protocol, TypeGuard
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from projetv0_voice.budget_observation import BudgetObservation, BudgetObservationIdentity

ORIGINAL_PILOT_SLOT = "sparra-original-pilot"
RESERVES = (Decimal("3.780408"), Decimal("1.22"))
FORECASTS = (Decimal("0.505068"), Decimal("0.12"))


def _exact_money(value: Decimal) -> bool:
    parts = value.as_tuple()
    return (
        value.is_finite()
        and value >= 0
        and len(parts.digits) <= 64
        and isinstance(parts.exponent, int)
        and -24 <= parts.exponent <= 24
    )


def reported_floors_valid(value: object) -> TypeGuard[tuple[Decimal, Decimal]]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(amount, Decimal) and _exact_money(amount) for amount in value)
    )


class OwnedPilotBudgetPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    policy_id: UUID
    identity: BudgetObservationIdentity = Field(repr=False)
    cost_scope: Literal["original-dedicated-account-key"] = "original-dedicated-account-key"
    total_calls: Literal[5] = 5
    openrouter_anchor_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    telnyx_anchor_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    paid_until: datetime
    qualified_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_digest: str
    deployment_id: str

    @model_validator(mode="after")
    def valid_binding(self) -> OwnedPilotBudgetPolicy:
        values = (
            self.paid_until,
            self.identity.accounting_start_utc,
            self.identity.openrouter_expires_at_utc,
        )
        if (
            any(v.tzinfo is None or v.utcoffset() is None for v in values)
            or self.paid_until <= self.identity.accounting_start_utc
            or self.paid_until > self.identity.openrouter_expires_at_utc
            or not self.image_digest
            or not self.deployment_id
            or not _exact_money(self.openrouter_anchor_usd)
            or not _exact_money(self.telnyx_anchor_usd)
            or any(
                re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (
                    self.identity.openrouter_credential_sha256,
                    self.identity.openrouter_account_sha256,
                    self.identity.telnyx_credential_sha256,
                    self.identity.telnyx_account_sha256,
                )
            )
            or re.fullmatch(r"\+[1-9][0-9]{1,14}", self.identity.did) is None
            or not self.identity.connection_id
        ):
            raise ValueError("owned_budget_policy_invalid")
        return self

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


class OwnedPilotBudgetPermit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    policy_sha256: str
    ordinal: int = Field(ge=1, le=5, strict=True)
    call_id: UUID
    call_control_id: str
    call_leg_id: str | None
    call_session_id: str | None
    generation: UUID
    paid_until: datetime


@dataclass(frozen=True)
class OwnedPilotBudgetDenied:
    """Ordinary admission refusal, not a persistence fault."""


@dataclass(frozen=True, repr=False)
class OwnedPilotBudgetSnapshot:
    """Writer publication; admission closure does not revoke an existing permit."""

    observation: BudgetObservation | None = None
    admission_available: bool = False
    reported_floors: tuple[Decimal, Decimal] | None = None


def observation_valid(
    policy: OwnedPilotBudgetPolicy, fact: BudgetObservation, now: datetime
) -> bool:
    try:
        money = (
            fact.openrouter_reported_spend_usd,
            fact.openrouter_available_cash_usd,
            fact.telnyx_reported_spend_usd,
            fact.telnyx_available_cash_usd,
            fact.openrouter_limit_usd,
            fact.openrouter_limit_remaining_usd,
        )
        return (
            fact.identity == policy.identity
            and 0 <= (now - fact.collection_started_at_utc).total_seconds() <= 60
            and fact.collection_started_at_utc <= fact.collection_finished_at_utc <= now
            and now < policy.paid_until
            and fact.openrouter_expires_at_utc == policy.identity.openrouter_expires_at_utc
            and fact.openrouter_limit_reset is None
            and fact.openrouter_limit_usd == 5
            and all(isinstance(v, Decimal) and _exact_money(v) for v in money)
            and fact.openrouter_reported_spend_usd >= policy.openrouter_anchor_usd
            and fact.telnyx_reported_spend_usd >= policy.telnyx_anchor_usd
        )
    except (ValueError, TypeError, ArithmeticError):
        return False


def affordable(
    policy: OwnedPilotBudgetPolicy, fact: BudgetObservation, count: int,
    *, reported_floors: tuple[Decimal, Decimal],
) -> bool:
    if (not reported_floors_valid(reported_floors)
            or type(count) is not int or not 0 <= count < policy.total_calls):
        return False
    with localcontext() as ctx:
        ctx.prec = 128
        for anchor, spent, cash, reserve, forecast, floor in zip(
            (policy.openrouter_anchor_usd, policy.telnyx_anchor_usd),
            (fact.openrouter_reported_spend_usd, fact.telnyx_reported_spend_usd),
            (fact.openrouter_available_cash_usd, fact.telnyx_available_cash_usd),
            RESERVES,
            FORECASTS,
            reported_floors,
            strict=True,
        ):
            effective = max(floor, spent, anchor + count * forecast)
            remaining_reserve = reserve - count * forecast
            if (effective + remaining_reserve > 5
                    or cash - (effective - spent) < remaining_reserve):
                return False
        gap = max(
            reported_floors[0], fact.openrouter_reported_spend_usd,
            policy.openrouter_anchor_usd + count * FORECASTS[0],
        )
        return (
            fact.openrouter_limit_remaining_usd - (gap - fact.openrouter_reported_spend_usd)
            >= RESERVES[0] - count * FORECASTS[0]
        )


class _BudgetWriter(Protocol):
    def set_owned_budget(
        self, policy: OwnedPilotBudgetPolicy, *, start_envelope_seconds: float = 335
    ) -> None: ...
    async def owned_budget_request(self, action: str, **values: object) -> object: ...
    def owned_budget_snapshots(self) -> tuple[OwnedPilotBudgetSnapshot, ...]: ...


class OwnedPilotBudgetGate:
    """One process consumer of the writer authority and bounded authenticated facts."""

    def __init__(
        self,
        writer: _BudgetWriter,
        policy: OwnedPilotBudgetPolicy,
        collect: Callable[[], Awaitable[BudgetObservation]],
        *,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        start_envelope_seconds: float = 335,
    ) -> None:
        self.policy = policy
        self.writer = writer
        self.collect = collect
        self.utcnow = utcnow
        self.start_envelope_seconds = start_envelope_seconds
        self._lock = asyncio.Lock()
        writer.set_owned_budget(policy, start_envelope_seconds=start_envelope_seconds)

    async def refresh(self) -> bool:
        async with self._lock:
            try:
                fact = await self.collect()
            except asyncio.CancelledError:
                raise
            except Exception:
                fact = None
            try:
                await self.writer.owned_budget_request("observe", observation=fact)
            except asyncio.CancelledError:
                raise
            except Exception:
                return False
            # Result delivery can be arbitrarily late. Never copy its old
            # observation/readiness over the writer's newer committed state.
            return self.ready

    @property
    def ready(self) -> bool:
        return self.can_start()

    def can_start(self) -> bool:
        snapshots = self.writer.owned_budget_snapshots()
        now = self.utcnow()
        return (
            bool(snapshots)
            and all(
                snapshot.admission_available
                and snapshot.observation is not None
                and reported_floors_valid(snapshot.reported_floors)
                and observation_valid(self.policy, snapshot.observation, now)
                and not snapshot.observation.active_call_control_ids
                for snapshot in snapshots
            )
            and (self.policy.paid_until - now).total_seconds() >= self.start_envelope_seconds
        )

    def permits(self, permit: OwnedPilotBudgetPermit | None) -> bool:
        snapshots = self.writer.owned_budget_snapshots()
        return (
            permit is not None
            and permit.policy_sha256 == self.policy.sha256
            and permit.paid_until == self.policy.paid_until
            and bool(snapshots)
            and all(
                snapshot.observation is not None
                and reported_floors_valid(snapshot.reported_floors)
                and observation_valid(self.policy, snapshot.observation, self.utcnow())
                and affordable(self.policy, snapshot.observation, permit.ordinal - 1,
                               reported_floors=snapshot.reported_floors)
                for snapshot in snapshots
            )
        )

    async def dispatch(self, permit: OwnedPilotBudgetPermit) -> bool:
        result = await self.writer.owned_budget_request("dispatch", permit=permit)
        return result is True and self.permits(permit)

    async def settled(self, call_control_id: str) -> None:
        await self.writer.owned_budget_request("settled", call_control_id=call_control_id)
