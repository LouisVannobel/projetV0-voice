"""Process-local OpenTelemetry metrics with closed semantic operations."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

import requests
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import AlwaysOffExemplarFilter, MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

from projetv0_voice.observability_bootstrap import (
    ObservabilityBootstrapToken,
    _token_matches_endpoint,
    _valid_endpoint,
)

_PREFIX = "projetv0.voice."
_OTLP_HEADERS = {"User-Agent": "projetv0-voice"}
_EXPORT_TIMEOUT_SECONDS = 2.0
_READER_EXPORT_TIMEOUT_MILLIS = 5000.0
_SHUTDOWN_TIMEOUT_MILLIS = 10000.0
_READER_INTERVAL_MILLIS = 30000.0
_MAX_OTLP_INT = (1 << 63) - 1

_SESSIONS = frozenset({"closed", "failed", "drained"})
_REJECTION_REASONS = frozenset(
    {"capacity", "draining", "qualification", "persistence", "invalid"}
)
_WEBHOOK_CLASSES = frozenset(
    {"initiated", "answered", "terminal", "recording", "unsupported", "invalid"}
)
_RECEIPTS = frozenset({"none", "first", "duplicate"})
_DISPOSITIONS = frozenset(
    {"ok", "bad_request", "forbidden", "too_large", "unavailable", "internal_error"}
)
_ACTIONS = frozenset({"answer", "streaming_start", "hangup"})
_OUTCOMES = frozenset(
    {
        "accepted",
        "rejected",
        "rate_limited",
        "retryable_not_sent",
        "outcome_unknown",
        "internal_error",
    }
)
_LATENCY_KINDS = frozenset({"turn", "first_speech"})
_SERVICES = frozenset({"stt", "llm", "tts"})
_RELAY_STATUSES = frozenset(
    {
        "empty",
        "delivered",
        "retry_scheduled",
        "stale_claim",
        "claim_budget_expired",
        "degraded",
    }
)
_RECORDING_STATUSES = frozenset({"saved", "error", "purged"})


def _closed(value: object, allowed: frozenset[str]) -> bool:
    return type(value) is str and value in allowed


def _nonnegative_number(value: object) -> float | None:
    accepted: int | float
    if type(value) is int or type(value) is float:
        accepted = value
    else:
        return None
    try:
        number = float(accepted)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _positive_number(value: object) -> float | None:
    number = _nonnegative_number(value)
    return number if number is not None and number > 0 else None


def _nonnegative_integer(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= _MAX_OTLP_INT else None


@dataclass(frozen=True, slots=True)
class RuntimePublishedSnapshot:
    """One complete immutable process observation publication."""

    generation: int
    ready: bool
    draining: bool
    writer_queue_depth: int
    writer_queue_oldest_age: float
    writer_quick_check: bool
    outbox_depth: int
    outbox_oldest_age: float
    outbox_bytes: int
    storage_bytes: int
    startup_profile_and_stale_recovery_complete: bool
    admission_open: bool
    qualification_state_valid: bool
    writer_owner_alive_and_ready: bool
    no_writer_fatal_or_degradation: bool
    relay_supervisor_alive: bool
    no_permanent_relay_or_sink_degradation: bool

    def __post_init__(self) -> None:
        integer_values = (
            self.generation,
            self.writer_queue_depth,
            self.outbox_depth,
            self.outbox_bytes,
            self.storage_bytes,
        )
        number_values = (
            self.writer_queue_oldest_age,
            self.outbox_oldest_age,
        )
        boolean_values = (
            self.ready,
            self.draining,
            self.writer_quick_check,
            self.startup_profile_and_stale_recovery_complete,
            self.admission_open,
            self.qualification_state_valid,
            self.writer_owner_alive_and_ready,
            self.no_writer_fatal_or_degradation,
            self.relay_supervisor_alive,
            self.no_permanent_relay_or_sink_degradation,
        )
        if (
            any(_nonnegative_integer(value) is None for value in integer_values)
            or any(_nonnegative_number(value) is None for value in number_values)
            or any(type(value) is not bool for value in boolean_values)
        ):
            raise RuntimeError("runtime_snapshot_invalid") from None


class RuntimePublication:
    """Protect exactly one reference to the latest immutable publication."""

    __slots__ = ("_lock", "_snapshot")

    def __init__(self, snapshot: RuntimePublishedSnapshot) -> None:
        if type(snapshot) is not RuntimePublishedSnapshot:
            raise RuntimeError("runtime_publication_invalid") from None
        self._lock = Lock()
        self._snapshot = snapshot

    def __repr__(self) -> str:
        return "RuntimePublication()"

    def publish(self, snapshot: RuntimePublishedSnapshot) -> None:
        if type(snapshot) is not RuntimePublishedSnapshot:
            raise RuntimeError("runtime_publication_invalid") from None
        with self._lock:
            if snapshot.generation <= self._snapshot.generation:
                raise RuntimeError("runtime_publication_invalid") from None
            self._snapshot = snapshot

    def snapshot(self) -> RuntimePublishedSnapshot:
        with self._lock:
            return self._snapshot


def _initial_runtime_snapshot() -> RuntimePublishedSnapshot:
    return RuntimePublishedSnapshot(
        generation=0,
        ready=False,
        draining=False,
        writer_queue_depth=0,
        writer_queue_oldest_age=0.0,
        writer_quick_check=False,
        outbox_depth=0,
        outbox_oldest_age=0.0,
        outbox_bytes=0,
        storage_bytes=0,
        startup_profile_and_stale_recovery_complete=False,
        admission_open=False,
        qualification_state_valid=False,
        writer_owner_alive_and_ready=False,
        no_writer_fatal_or_degradation=False,
        relay_supervisor_alive=False,
        no_permanent_relay_or_sink_degradation=False,
    )


class _CallMetricLease:
    """Single-finish owner for one call's active, total, and duration metrics."""

    __slots__ = ("_finished", "_owner", "_started_at")

    def __init__(self, owner: RuntimeMetrics, started_at: float | None) -> None:
        self._owner = owner
        self._started_at = started_at
        self._finished = False

    def __repr__(self) -> str:
        return "CallMetricLease()"

    def finish(self, metric_class: object) -> None:
        if self._finished:
            return
        self._finished = True
        self._owner._finish_call_metric(self._started_at, metric_class)  # noqa: SLF001


class RuntimeMetrics:
    """Own one local meter provider and a closed set of voice metrics."""

    def __init__(
        self,
        *,
        provider: Any,
        metric_reader: Any,
        monotonic: Callable[[], object] = time.monotonic,
        publication: RuntimePublication | None = None,
    ) -> None:
        self._provider = provider
        self._metric_reader = metric_reader
        self._failure_code: str | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._monotonic = monotonic
        self._publication = publication or RuntimePublication(_initial_runtime_snapshot())

        self._meter = provider.get_meter("projetv0.voice", None)
        self._calls_active = self._meter.create_up_down_counter(
            _PREFIX + "calls.active", description="", unit=""
        )
        self._calls_total = self._meter.create_counter(
            _PREFIX + "calls.total", description="", unit=""
        )
        self._admission_rejections = self._meter.create_counter(
            _PREFIX + "admission.rejections", description="", unit=""
        )
        self._webhooks_total = self._meter.create_counter(
            _PREFIX + "webhooks.total", description="", unit=""
        )
        self._actions_total = self._meter.create_counter(
            _PREFIX + "actions.total", description="", unit=""
        )
        self._sessions_duration = self._meter.create_histogram(
            _PREFIX + "sessions.duration", description="", unit="s"
        )
        self._user_bot_latency = self._meter.create_histogram(
            _PREFIX + "user_bot_latency", description="", unit="s"
        )
        self._service_ttfb = self._meter.create_histogram(
            _PREFIX + "service_ttfb", description="", unit="s"
        )
        self._disclosure_mark_ack = self._meter.create_histogram(
            _PREFIX + "disclosure.mark_ack", description="", unit="s"
        )
        self._disclosure_timeouts = self._meter.create_counter(
            _PREFIX + "disclosure.timeouts", description="", unit=""
        )
        self._relay_runs = self._meter.create_counter(
            _PREFIX + "relay.runs", description="", unit=""
        )
        self._writer_queue_depth = self._meter.create_observable_gauge(
            _PREFIX + "writer.queue_depth",
            callbacks=(self._observe_writer_queue_depth,),
            description="",
            unit="",
        )
        self._writer_queue_oldest_age = self._meter.create_observable_gauge(
            _PREFIX + "writer.queue_oldest_age",
            callbacks=(self._observe_writer_queue_oldest_age,),
            description="",
            unit="s",
        )
        self._writer_quick_check = self._meter.create_observable_gauge(
            _PREFIX + "writer.quick_check",
            callbacks=(self._observe_writer_quick_check,),
            description="",
            unit="1",
        )
        self._outbox_depth = self._meter.create_observable_gauge(
            _PREFIX + "outbox.depth",
            callbacks=(self._observe_outbox_depth,),
            description="",
            unit="",
        )
        self._outbox_oldest_age = self._meter.create_observable_gauge(
            _PREFIX + "outbox.oldest_age",
            callbacks=(self._observe_outbox_oldest_age,),
            description="",
            unit="s",
        )
        self._outbox_bytes = self._meter.create_observable_gauge(
            _PREFIX + "outbox.bytes",
            callbacks=(self._observe_outbox_bytes,),
            description="",
            unit="By",
        )
        self._transcript_turns_lost = self._meter.create_counter(
            _PREFIX + "transcript.turns_lost", description="", unit=""
        )
        self._recordings_total = self._meter.create_counter(
            _PREFIX + "recordings.total", description="", unit=""
        )
        self._event_loop_lag = self._meter.create_histogram(
            _PREFIX + "runtime.event_loop_lag", description="", unit="s"
        )
        self._ready = self._meter.create_observable_gauge(
            _PREFIX + "ready",
            callbacks=(self._observe_ready,),
            description="",
            unit="1",
        )

    @classmethod
    def in_memory(
        cls,
        *,
        monotonic: Callable[[], object] = time.monotonic,
        publication: RuntimePublication | None = None,
    ) -> RuntimeMetrics:
        """Create an independent local owner without an export thread."""

        reader = InMemoryMetricReader()
        provider = MeterProvider(
            metric_readers=(reader,),
            resource=Resource({"service.name": "projetv0-voice"}),
            exemplar_filter=AlwaysOffExemplarFilter(),
            shutdown_on_exit=False,
        )
        return cls(
            provider=provider,
            metric_reader=reader,
            monotonic=monotonic,
            publication=publication,
        )

    @classmethod
    def production(
        cls,
        token: ObservabilityBootstrapToken,
        *,
        endpoint: str,
    ) -> RuntimeMetrics:
        """Create the fixed local OTLP/HTTP owner after bootstrap validation."""

        if type(token) is not ObservabilityBootstrapToken:
            raise ValueError("observability_bootstrap_token_invalid") from None
        return _build_production(token, endpoint=endpoint)

    @classmethod
    def _from_provider(
        cls,
        provider: Any,
        metric_reader: Any,
        *,
        monotonic: Callable[[], object] = time.monotonic,
        publication: RuntimePublication | None = None,
    ) -> RuntimeMetrics:
        return cls(
            provider=provider,
            metric_reader=metric_reader,
            monotonic=monotonic,
            publication=publication,
        )

    @property
    def failure_code(self) -> str | None:
        return self._failure_code

    def record_admission_rejection(self, reason: object) -> None:
        if not _closed(reason, _REJECTION_REASONS):
            self._latch_failure()
            return
        self._add(self._admission_rejections, {"reason": reason})

    def begin_call(self) -> _CallMetricLease:
        """Begin one call without letting metric or clock faults escape."""

        self._add_value(self._calls_active, 1, {})
        return _CallMetricLease(self, self._sample_monotonic())

    def _finish_call_metric(
        self,
        started_at: float | None,
        metric_class: object,
    ) -> None:
        ended_at = self._sample_monotonic()
        self._add_value(self._calls_active, -1, {})
        if not _closed(metric_class, _SESSIONS):
            self._latch_failure()
            return
        attributes = {"session": metric_class}
        self._add(self._calls_total, attributes)
        if started_at is None or ended_at is None:
            return
        duration = ended_at - started_at
        if not math.isfinite(duration) or duration < 0:
            self._latch_failure()
            return
        self._record(self._sessions_duration, duration, attributes)

    def _sample_monotonic(self) -> float | None:
        try:
            sample = self._monotonic()
        except BaseException:
            self._latch_failure()
            return None
        value = _nonnegative_number(sample)
        if value is None:
            self._latch_failure()
        return value

    def record_webhook(
        self,
        webhook_class: object,
        receipt: object,
        disposition: object,
    ) -> None:
        if not (
            _closed(webhook_class, _WEBHOOK_CLASSES)
            and _closed(receipt, _RECEIPTS)
            and _closed(disposition, _DISPOSITIONS)
        ):
            self._latch_failure()
            return
        self._add(
            self._webhooks_total,
            {
                "webhook_class": webhook_class,
                "receipt": receipt,
                "disposition": disposition,
            },
        )

    def record_action(self, action: object, outcome: object) -> None:
        if not (_closed(action, _ACTIONS) and _closed(outcome, _OUTCOMES)):
            self._latch_failure()
            return
        self._add(self._actions_total, {"action": action, "outcome": outcome})

    def record_user_bot_latency(self, latency_kind: object, seconds: object) -> None:
        value = _nonnegative_number(seconds)
        if not _closed(latency_kind, _LATENCY_KINDS) or value is None:
            self._latch_failure()
            return
        self._record(self._user_bot_latency, value, {"latency_kind": latency_kind})

    def record_service_ttfb(self, service: object, seconds: object) -> None:
        value = _positive_number(seconds)
        if not _closed(service, _SERVICES) or value is None:
            self._latch_failure()
            return
        self._record(self._service_ttfb, value, {"service": service})

    def record_disclosure_ack(self, seconds: object) -> None:
        value = _nonnegative_number(seconds)
        if value is None:
            self._latch_failure()
            return
        self._record(self._disclosure_mark_ack, value, {})

    def record_disclosure_timeout(self) -> None:
        self._add(self._disclosure_timeouts, {})

    def record_relay_run(self, relay_status: object) -> None:
        if not _closed(relay_status, _RELAY_STATUSES):
            self._latch_failure()
            return
        self._add(self._relay_runs, {"relay_status": relay_status})

    def record_transcript_turn_lost(self) -> None:
        self._add(self._transcript_turns_lost, {})

    def record_recording(self, recording_status: object) -> None:
        if not _closed(recording_status, _RECORDING_STATUSES):
            self._latch_failure()
            return
        self._add(self._recordings_total, {"recording_status": recording_status})

    def record_event_loop_lag(self, seconds: object) -> None:
        value = _nonnegative_number(seconds)
        if value is None:
            self._latch_failure()
            return
        self._record(self._event_loop_lag, value, {})

    def update_writer_state(
        self,
        *,
        queue_depth: object,
        oldest_age: object,
        quick_check: object,
    ) -> None:
        depth_value = _nonnegative_integer(queue_depth)
        age_value = _nonnegative_number(oldest_age)
        if depth_value is None or age_value is None or type(quick_check) is not bool:
            self._latch_failure()
            return
        current = self._publication.snapshot()
        self._publication.publish(
            replace(
                current,
                generation=current.generation + 1,
                writer_queue_depth=depth_value,
                writer_queue_oldest_age=age_value,
                writer_quick_check=quick_check,
            )
        )

    def update_outbox_state(
        self,
        *,
        depth: object,
        oldest_age: object,
        bytes_count: object,
    ) -> None:
        depth_value = _nonnegative_integer(depth)
        age_value = _nonnegative_number(oldest_age)
        bytes_value = _nonnegative_integer(bytes_count)
        if depth_value is None or age_value is None or bytes_value is None:
            self._latch_failure()
            return
        current = self._publication.snapshot()
        self._publication.publish(
            replace(
                current,
                generation=current.generation + 1,
                outbox_depth=depth_value,
                outbox_oldest_age=age_value,
                outbox_bytes=bytes_value,
            )
        )

    def update_ready(self, ready: object) -> None:
        if type(ready) is not bool:
            self._latch_failure()
            return
        current = self._publication.snapshot()
        self._publication.publish(
            replace(
                current,
                generation=current.generation + 1,
                ready=ready,
            )
        )

    async def aclose(self) -> None:
        """Run provider shutdown exactly once while shielding shared completion."""

        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._shutdown_provider(),
                name="runtime-metrics-shutdown",
            )
        shutdown_task = self._shutdown_task
        cancellation: asyncio.CancelledError | None = None
        while not shutdown_task.done():
            try:
                await asyncio.shield(shutdown_task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception:
                break
        if cancellation is not None:
            raise cancellation
        shutdown_task.result()

    async def _shutdown_provider(self) -> None:
        failed = False
        try:
            await asyncio.to_thread(
                self._provider.shutdown,
                timeout_millis=_SHUTDOWN_TIMEOUT_MILLIS,
            )
        except Exception:
            failed = True
        if failed:
            raise RuntimeError("metrics_shutdown_failed") from None

    def _add(self, instrument: Any, attributes: dict[str, object]) -> None:
        self._add_value(instrument, 1, attributes)

    def _add_value(
        self,
        instrument: Any,
        value: int,
        attributes: dict[str, object],
    ) -> None:
        try:
            instrument.add(value, attributes)
        except BaseException:
            self._latch_failure()

    def _record(
        self,
        instrument: Any,
        value: float,
        attributes: dict[str, object],
    ) -> None:
        try:
            instrument.record(value, attributes)
        except BaseException:
            self._latch_failure()

    def _latch_failure(self) -> None:
        self._failure_code = "metrics_record_failed"

    def _observe_writer_queue_depth(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._publication.snapshot().writer_queue_depth),)

    def _observe_writer_queue_oldest_age(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._publication.snapshot().writer_queue_oldest_age),)

    def _observe_writer_quick_check(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(int(self._publication.snapshot().writer_quick_check)),)

    def _observe_outbox_depth(self, _options: CallbackOptions) -> Iterable[Observation]:
        return (Observation(self._publication.snapshot().outbox_depth),)

    def _observe_outbox_oldest_age(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._publication.snapshot().outbox_oldest_age),)

    def _observe_outbox_bytes(self, _options: CallbackOptions) -> Iterable[Observation]:
        return (Observation(self._publication.snapshot().outbox_bytes),)

    def _observe_ready(self, _options: CallbackOptions) -> Iterable[Observation]:
        return (Observation(int(self._publication.snapshot().ready)),)


def _build_production(
    token: ObservabilityBootstrapToken,
    *,
    endpoint: object,
    exporter_factory: Callable[..., Any] = OTLPMetricExporter,
    reader_factory: Callable[..., Any] = PeriodicExportingMetricReader,
    provider_factory: Callable[..., Any] = MeterProvider,
    session_factory: Callable[[], requests.Session] = requests.Session,
) -> RuntimeMetrics:
    if type(token) is not ObservabilityBootstrapToken:
        raise ValueError("observability_bootstrap_token_invalid") from None
    if not _valid_endpoint(endpoint):
        raise RuntimeError("observability_endpoint_invalid") from None
    if not _token_matches_endpoint(token, endpoint):
        raise RuntimeError("observability_endpoint_mismatch") from None

    session = session_factory()
    session.trust_env = False
    session.max_redirects = 0
    exporter = exporter_factory(
        endpoint=endpoint,
        headers=dict(_OTLP_HEADERS),
        timeout=_EXPORT_TIMEOUT_SECONDS,
        compression=Compression.NoCompression,
        session=session,
    )
    reader = reader_factory(
        exporter,
        export_interval_millis=_READER_INTERVAL_MILLIS,
        export_timeout_millis=_READER_EXPORT_TIMEOUT_MILLIS,
    )
    provider = provider_factory(
        metric_readers=(reader,),
        resource=Resource({"service.name": "projetv0-voice"}),
        exemplar_filter=AlwaysOffExemplarFilter(),
        shutdown_on_exit=False,
    )
    return RuntimeMetrics(provider=provider, metric_reader=reader)


__all__ = ["RuntimeMetrics", "RuntimePublication", "RuntimePublishedSnapshot"]
