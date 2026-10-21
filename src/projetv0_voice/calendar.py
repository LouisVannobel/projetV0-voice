"""Closed Calendar bridge DTOs and one native recap playback proof."""

from __future__ import annotations

import asyncio
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from pipecat.frames.frames import Frame, LLMAssistantPushAggregationFrame, TTSSpeakFrame
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_INSTANT = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
)


def calendar_instant(value: object) -> datetime:
    if not isinstance(value, str) or _INSTANT.fullmatch(value) is None:
        raise ValueError("calendar_instant_invalid")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, OverflowError):
        raise ValueError("calendar_instant_invalid") from None


def calendar_uuid(value: object) -> UUID:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("calendar_uuid_invalid")
    return UUID(value)


def calendar_zone(value: str) -> str:
    ZoneInfo(value)
    return value


Instant = Annotated[datetime, BeforeValidator(calendar_instant)]
OpaqueId = Annotated[UUID, BeforeValidator(calendar_uuid)]
Zone = Annotated[str, BeforeValidator(calendar_zone)]


class CalendarDto(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CalendarService(CalendarDto):
    service_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    duration_minutes: int = Field(gt=0, le=1440)


class CalendarCatalog(CalendarDto):
    schema_version: Literal[1]
    call_id: OpaqueId
    policy_revision: int = Field(ge=1)
    timezone: Zone
    services: list[CalendarService] = Field(min_length=1, max_length=100)


class CalendarOffer(CalendarDto):
    offerId: OpaqueId
    serviceId: str = Field(min_length=1, max_length=80)
    serviceName: str = Field(min_length=1, max_length=120)
    timezone: Zone
    start: Instant
    end: Instant
    expiresAt: Instant


class CalendarBooking(CalendarDto):
    serviceId: str = Field(min_length=1, max_length=80)
    serviceName: str = Field(min_length=1, max_length=120)
    timezone: Zone
    start: Instant
    end: Instant
    providerReference: str = Field(min_length=1, max_length=1024)


class CalendarResult(CalendarDto):
    state: Literal[
        "waiting_evidence",
        "pending",
        "running",
        "offered",
        "confirmed",
        "refused",
        "unknown",
        "missing",
    ]
    operationId: OpaqueId
    offers: list[CalendarOffer] | None = Field(default=None, max_length=3)
    booking: CalendarBooking | None = None
    reason: Literal["appointment_to_confirm"] | None = None

    @model_validator(mode="after")
    def result_shape(self) -> CalendarResult:
        if (self.state == "confirmed") != (self.booking is not None):
            raise ValueError("calendar_result_invalid")
        if (self.state == "offered") != (self.offers is not None):
            raise ValueError("calendar_result_invalid")
        intervals: list[CalendarOffer | CalendarBooking] = list(self.offers or [])
        if self.booking is not None:
            intervals.append(self.booking)
        if any(item.start >= item.end for item in intervals):
            raise ValueError("calendar_interval_invalid")
        return self


class FindSlotsArguments(CalendarDto):
    serviceId: str = Field(min_length=1, max_length=80)
    windowStart: Instant
    windowEnd: Instant


class SelectSlotArguments(CalendarDto):
    offerId: OpaqueId


class CalendarConfirmation(CalendarDto):
    recapTurnId: OpaqueId
    confirmationTurnId: OpaqueId
    playbackMark: str = Field(pattern=r"^sparra-calendar-recap-[0-9a-f-]{36}$")
    playbackAcknowledgedAt: Instant


class FindSlotsRequest(FindSlotsArguments):
    kind: Literal["find_slots"]
    policyRevision: int = Field(ge=1)


class BookAppointmentRequest(SelectSlotArguments):
    kind: Literal["book_appointment"]
    confirmation: CalendarConfirmation


def validate_calendar_request(request: dict[str, object]) -> None:
    schema = FindSlotsRequest if request.get("kind") == "find_slots" else BookAppointmentRequest
    schema.model_validate(request)


def valid_calendar_arguments(name: str, arguments: object) -> bool:
    schema: dict[str, type[CalendarDto]] = {
        "find_slots": FindSlotsArguments,
        "book_appointment": SelectSlotArguments,
    }
    if name not in schema:
        return False
    try:
        schema[name].model_validate(arguments)
    except (ValueError, TypeError):
        return False
    return True


class CalendarSink(Protocol):
    async def calendar_catalog(self, call_id: UUID) -> CalendarCatalog | None: ...

    async def calendar_submit(
        self,
        call_id: UUID,
        operation_id: UUID,
        request: dict[str, object],
    ) -> CalendarResult: ...

    async def calendar_result(self, call_id: UUID, operation_id: UUID) -> CalendarResult: ...


async def await_calendar_result(
    sink: CalendarSink,
    call_id: UUID,
    operation_id: UUID,
    request: dict[str, object],
    *,
    wait_seconds: float = 12.0,
    poll_delay: float = 0.25,
) -> CalendarResult | None:
    """One immutable submission; every RPC releases its connection before delay."""
    import copy

    from projetv0_voice.persistence.postgres_sink import (
        OperationSinkCommitAmbiguousError,
        OperationSinkPermanentError,
        OperationSinkTransientError,
    )

    frozen = copy.deepcopy(request)
    canonical = operation_id
    submit = True
    last_unknown: CalendarResult | None = None
    try:
        async with asyncio.timeout(wait_seconds):
            while True:
                try:
                    result = (
                        await sink.calendar_submit(call_id, operation_id, copy.deepcopy(frozen))
                        if submit
                        else await sink.calendar_result(call_id, canonical)
                    )
                    # Once a possible send is observed, only read that operation.
                    # A later response cannot redirect its identity or reopen submit.
                    if last_unknown is None or result.operationId == canonical:
                        canonical = result.operationId
                        if result.state == "unknown":
                            last_unknown = result
                        elif result.state not in {
                            "waiting_evidence", "missing", "pending", "running",
                        }:
                            return result
                        submit = (
                            last_unknown is None and result.state in {"waiting_evidence", "missing"}
                        )
                except (OperationSinkCommitAmbiguousError, OperationSinkTransientError):
                    submit = last_unknown is None
                except OperationSinkPermanentError:
                    return None
                await asyncio.sleep(poll_delay)
    except TimeoutError:
        return last_unknown


_AFFIRMATIVE = re.compile(
    r"(?:oui|ouais|je confirme|d'accord)"
    r"(?: (?:je confirme|parfait|très bien|c'est bon))?"
    r"(?: merci(?: beaucoup)?)?"
)


def is_affirmative(text: str) -> bool:
    # Match the entire utterance: a correction, condition, question or negation
    # must never inherit consent from an affirmative prefix.
    normalized = " ".join(re.sub(r"[,.!]", " ", text.casefold().replace("’", "'")).split())
    return _AFFIRMATIVE.fullmatch(normalized) is not None


def needs_confirmation_clarification(text: str) -> bool:
    normalized = " ".join(text.casefold().split()).rstrip(".!? ")
    return is_affirmative(text.rstrip("? ")) or normalized in {
        "peut-être", "peut être", "je ne sais pas", "allô", "allo",
        "merci", "merci beaucoup", "parfait",
    }


_DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


def appointment_label(offer: CalendarOffer | CalendarBooking) -> str:
    zone = ZoneInfo(offer.timezone)
    start = offer.start.astimezone(zone)
    end = offer.end.astimezone(zone)
    cross_year = start.year != end.year
    changes_clock = start.utcoffset() != end.utcoffset()

    def date_label(value: datetime) -> str:
        label = f"{_DAYS[value.weekday()]} {value.day} {_MONTHS[value.month - 1]}"
        return label + (f" {value.year}" if cross_year else "")

    def time_label(value: datetime) -> str:
        label = f"{value.hour} h" + (f" {value.minute:02d}" if value.minute else "")
        ambiguous = value.replace(fold=0).utcoffset() != value.replace(fold=1).utcoffset()
        if changes_clock or ambiguous:
            if offer.timezone == "Europe/Paris":
                label += " (heure d’été)" if value.dst() else " (heure d’hiver)"
            elif ambiguous:
                label += " (second passage)" if value.fold else " (premier passage)"
        return label

    location = "heure de Paris" if offer.timezone == "Europe/Paris" else "heure de l’établissement"
    label = f"{offer.serviceName}, {date_label(start)}"
    if start.date() == end.date():
        label += f" de {time_label(start)} à {time_label(end)}"
    else:
        label += f" à {time_label(start)}, jusqu’au {date_label(end)} à {time_label(end)}"
    return f"{label}, {location}"


@dataclass(slots=True)
class RecordedTurnReceipt:
    """One-shot writer admission receipt, carried in the native callback task context."""

    ended_at: datetime
    turn_id: UUID | None = None
    expected_text: str | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)

    def settle(self, turn_id: UUID | None) -> None:
        if not self.settled.is_set():
            self.turn_id = turn_id
            self.settled.set()

    async def admitted(self) -> UUID | None:
        try:
            await asyncio.wait_for(self.settled.wait(), 2.0)
        except TimeoutError:
            return None
        return self.turn_id


recorded_turn_context: ContextVar[RecordedTurnReceipt | None] = ContextVar(
    "calendar_recorded_turn",
    default=None,
)


@dataclass(slots=True)
class CalendarRecap:
    offer_id: UUID
    speak_frame: TTSSpeakFrame
    mark_name: str = field(default_factory=lambda: f"sparra-calendar-recap-{uuid4()}")
    aggregation_frame: LLMAssistantPushAggregationFrame = field(
        default_factory=LLMAssistantPushAggregationFrame,
    )
    context_id: str | None = None
    interruption_id: int | None = None
    audio_observed: bool = False
    armed: bool = False
    invalidated: bool = False
    forwarded: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    acknowledged_at: datetime | None = None
    confirmation_start_id: int | None = None
    receipt: RecordedTurnReceipt | None = None


class CalendarPlayback:
    """Only the exact native TTS context and paced carrier ACK can open confirmation."""

    def __init__(self) -> None:
        self.attempt: CalendarRecap | None = None
        self._synthesizing: CalendarRecap | None = None

    def begin(self, offer_id: UUID, text: str) -> CalendarRecap:
        self.invalidate()
        self.attempt = CalendarRecap(offer_id, TTSSpeakFrame(text, append_to_context=True))
        return self.attempt

    def before_tts_frame(self, frame: Frame) -> None:
        attempt = self.attempt
        self._synthesizing = attempt if attempt and frame is attempt.speak_frame else None

    def after_tts_frame(self, frame: Frame) -> None:
        self._synthesizing = None

    def bind_context(self, context_id: str) -> None:
        if self._synthesizing is not None:
            self._synthesizing.context_id = context_id

    def note_audio(self, context_id: str | None) -> None:
        if self.attempt and context_id and context_id == self.attempt.context_id:
            self.attempt.audio_observed = True

    def owns_mark(self, name: str) -> bool:
        return name.startswith("sparra-calendar-recap-")

    def arm_mark(self, name: str) -> bool:
        attempt = self.attempt
        if not attempt or attempt.invalidated or name != attempt.mark_name:
            return False
        attempt.armed = attempt.audio_observed
        return attempt.armed

    def mark_forwarded(self, name: str) -> None:
        attempt = self.attempt
        if attempt and not attempt.invalidated and attempt.armed and name == attempt.mark_name:
            attempt.forwarded.set()

    def accept_mark(self, name: str) -> None:
        attempt = self.attempt
        if (
            attempt
            and not attempt.invalidated
            and name == attempt.mark_name
            and attempt.armed
            and attempt.forwarded.is_set()
            and not attempt.completed.is_set()
        ):
            attempt.acknowledged_at = datetime.now(UTC)
            attempt.completed.set()

    def caller_started(self, started_id: int) -> None:
        attempt = self.attempt
        if attempt is None:
            return
        if (
            attempt.acknowledged_at is None
            or attempt.confirmation_start_id is not None
            or datetime.now(UTC) <= attempt.acknowledged_at
        ):
            self.invalidate()
            # Retain only the rejected attempt for this exact native caller
            # turn's clarification. It can never provide booking authority.
            attempt.confirmation_start_id = started_id
            self.attempt = attempt
        else:
            attempt.confirmation_start_id = started_id

    def interrupted(self, frame_id: int = -1) -> None:
        attempt = self.attempt
        if attempt is None:
            return
        if attempt.invalidated:
            return
        # Only the normal interruption accompanying the exact first post-ACK
        # start preserves completed playback. An independent clear revokes it.
        if attempt.acknowledged_at is not None and attempt.confirmation_start_id is not None:
            if attempt.interruption_id is None:
                attempt.interruption_id = frame_id
            if attempt.interruption_id == frame_id:
                return
        self.invalidate()

    def confirmation_for(self, started_id: int, text: str) -> CalendarRecap | None:
        attempt = self.attempt
        self.attempt = None
        if (
            attempt
            and not attempt.invalidated
            and attempt.acknowledged_at is not None
            and attempt.confirmation_start_id == started_id
            and is_affirmative(text)
        ):
            return attempt
        return None

    def invalidate(self) -> None:
        if self.attempt:
            self.attempt.invalidated = True
            self.attempt.forwarded.set()
            self.attempt.completed.set()
        self.attempt = None
        self._synthesizing = None

    async def wait_for_ack(self, attempt: CalendarRecap, *, ack_timeout: float) -> bool:
        try:
            await asyncio.wait_for(attempt.forwarded.wait(), 20.0)
            await asyncio.wait_for(attempt.completed.wait(), ack_timeout)
        except TimeoutError:
            if self.attempt is attempt:
                self.invalidate()
            return False
        return not attempt.invalidated and attempt.acknowledged_at is not None
