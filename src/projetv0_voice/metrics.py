"""Process-local OpenTelemetry metrics with closed semantic operations."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable, Iterable
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
    _valid_endpoint,
    _validate_observability_mapping,
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


class RuntimeMetrics:
    """Own one local meter provider and a closed set of voice metrics."""

    def __init__(self, *, provider: Any, metric_reader: Any) -> None:
        self._provider = provider
        self._metric_reader = metric_reader
        self._failure_code: str | None = None
        self._shutdown_task: asyncio.Task[None] | None = None

        self._writer_queue_depth_value = 0
        self._writer_queue_oldest_age_value = 0.0
        self._writer_quick_check_value = 0
        self._outbox_depth_value = 0
        self._outbox_oldest_age_value = 0.0
        self._outbox_bytes_value = 0
        self._ready_value = 0

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
    def in_memory(cls) -> RuntimeMetrics:
        """Create an independent local owner without an export thread."""

        reader = InMemoryMetricReader()
        provider = MeterProvider(
            metric_readers=(reader,),
            resource=Resource({"service.name": "projetv0-voice"}),
            exemplar_filter=AlwaysOffExemplarFilter(),
            shutdown_on_exit=False,
        )
        return cls(provider=provider, metric_reader=reader)

    @classmethod
    def production(cls, token: ObservabilityBootstrapToken) -> RuntimeMetrics:
        """Create the fixed local OTLP/HTTP owner after bootstrap validation."""

        if type(token) is not ObservabilityBootstrapToken:
            raise ValueError("observability_bootstrap_token_invalid") from None
        _validate_observability_mapping(os.environ)
        endpoint = os.environ["VOICE_OTLP_HTTP_ENDPOINT"]
        return _build_production(token, endpoint=endpoint)

    @classmethod
    def _from_provider(cls, provider: Any, metric_reader: Any) -> RuntimeMetrics:
        return cls(provider=provider, metric_reader=metric_reader)

    @property
    def failure_code(self) -> str | None:
        return self._failure_code

    def record_admission_rejection(self, reason: object) -> None:
        if not _closed(reason, _REJECTION_REASONS):
            self._latch_failure()
            return
        self._add(self._admission_rejections, {"reason": reason})

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
        self._writer_queue_depth_value = depth_value
        self._writer_queue_oldest_age_value = age_value
        self._writer_quick_check_value = int(quick_check)

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
        self._outbox_depth_value = depth_value
        self._outbox_oldest_age_value = age_value
        self._outbox_bytes_value = bytes_value

    def update_ready(self, ready: object) -> None:
        if type(ready) is not bool:
            self._latch_failure()
            return
        self._ready_value = int(ready)

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
        try:
            instrument.add(1, attributes)
        except Exception:
            self._latch_failure()

    def _record(
        self,
        instrument: Any,
        value: float,
        attributes: dict[str, object],
    ) -> None:
        try:
            instrument.record(value, attributes)
        except Exception:
            self._latch_failure()

    def _latch_failure(self) -> None:
        self._failure_code = "metrics_record_failed"

    def _observe_writer_queue_depth(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._writer_queue_depth_value),)

    def _observe_writer_queue_oldest_age(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._writer_queue_oldest_age_value),)

    def _observe_writer_quick_check(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._writer_quick_check_value),)

    def _observe_outbox_depth(self, _options: CallbackOptions) -> Iterable[Observation]:
        return (Observation(self._outbox_depth_value),)

    def _observe_outbox_oldest_age(
        self, _options: CallbackOptions
    ) -> Iterable[Observation]:
        return (Observation(self._outbox_oldest_age_value),)

    def _observe_outbox_bytes(self, _options: CallbackOptions) -> Iterable[Observation]:
        return (Observation(self._outbox_bytes_value),)

    def _observe_ready(self, _options: CallbackOptions) -> Iterable[Observation]:
        return (Observation(self._ready_value),)


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


__all__ = ["RuntimeMetrics"]
