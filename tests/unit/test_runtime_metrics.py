from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from collections.abc import Callable
from typing import Any

import pytest
import requests
from loguru import logger
from opentelemetry import metrics as global_metrics
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common._internal.metrics_encoder import (
    encode_metrics,
)
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.metrics import AlwaysOffExemplarFilter, MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricExporter,
    MetricExportResult,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, use_span

from projetv0_voice import dependency_logging
from projetv0_voice import metrics as metrics_module
from projetv0_voice.observability_bootstrap import (
    ObservabilityBootstrapToken,
    _validate_observability_mapping,
)
from projetv0_voice.runtime_config import (
    RuntimeSettingsV1,
    capture_runtime_environment,
    parse_runtime_settings,
)
from projetv0_voice.telnyx.call_control import CallControlResult, StreamingStartV1

ENDPOINT = "https://collector.invalid/tenant/v1/metrics"
PREFIX = "projetv0.voice."


def _production_settings(
    *,
    endpoint: str = ENDPOINT,
) -> RuntimeSettingsV1:
    capture = capture_runtime_environment(
        {
            "VOICE_RUNTIME_MODE": "strict",
            "VOICE_DEPLOYMENT_ID": "voice-agent-a",
            "VOICE_RUNTIME_CONTRACT_PATH": "/srv/projetv0/runtime-contract.json",
            "VOICE_AGENT_BUNDLE_PATH": "/srv/projetv0/agent-bundle",
            "VOICE_QUALIFIED_PROFILE_PATH": "/srv/projetv0/qualified.json",
            "VOICE_KEYRING_PATH": "/srv/projetv0/keyring.json",
            "VOICE_SQLITE_PATH": "/var/lib/projetv0/voice.sqlite3",
            "VOICE_RUNTIME_CONTRACT_SHA256": "a" * 64,
            "VOICE_IMAGE_DIGEST": (
                f"ghcr.io/louisvannobel/projetv0-voice@sha256:{'d' * 64}"
            ),
            "VOICE_AGENT_BUNDLE_SHA256": "b" * 64,
            "VOICE_INFERENCE_PROFILE_SHA256": "c" * 64,
            "VOICE_DEPLOYMENT_MAX_CALLS": "10",
            "VOICE_HANDSHAKE_TIMEOUT_SECONDS": "5",
            "VOICE_CALL_IDLE_TIMEOUT_SECONDS": "300",
            "VOICE_CALL_CLEANUP_PHASE_TIMEOUT_SECONDS": "10",
            "VOICE_PRE_DRAIN_GRACE_SECONDS": "15",
            "VOICE_UVICORN_GRACE_SECONDS": "20",
            "VOICE_SHUTDOWN_GRACE_SECONDS": "30",
            "VOICE_TELNYX_API_KEY_FILE": "/run/secrets/telnyx-api-key",
            "VOICE_TELNYX_WEBHOOK_PUBLIC_KEY_FILE": "/run/secrets/telnyx-webhook-key",
            "VOICE_OPENROUTER_API_KEY_FILE": "/run/secrets/openrouter-api-key",
            "VOICE_POSTGRES_DSN_FILE": "/run/secrets/postgres-dsn",
            "VOICE_TELNYX_MEDIA_WSS_URL": "wss://voice.invalid/telnyx/media",
            "VOICE_OTLP_HTTP_ENDPOINT": endpoint,
            "VOICE_BIND_HOST": "127.0.0.1",
            "VOICE_BIND_PORT": "8080",
        }
    )
    settings = parse_runtime_settings(
        capture,
        geteuid=lambda: 10001,
        getegid=lambda: 10001,
    )
    return settings


def _production_token() -> ObservabilityBootstrapToken:
    return _production_settings().observability_token()


EXPECTED_INSTRUMENTS = {
    "calls.active": ("_UpDownCounter", ""),
    "calls.total": ("_Counter", ""),
    "admission.rejections": ("_Counter", ""),
    "webhooks.total": ("_Counter", ""),
    "actions.total": ("_Counter", ""),
    "sessions.duration": ("_Histogram", "s"),
    "user_bot_latency": ("_Histogram", "s"),
    "service_ttfb": ("_Histogram", "s"),
    "stt.failures": ("_Counter", ""),
    "disclosure.mark_ack": ("_Histogram", "s"),
    "disclosure.timeouts": ("_Counter", ""),
    "relay.runs": ("_Counter", ""),
    "writer.queue_depth": ("_ObservableGauge", ""),
    "writer.queue_oldest_age": ("_ObservableGauge", "s"),
    "writer.quick_check": ("_ObservableGauge", "1"),
    "outbox.depth": ("_ObservableGauge", ""),
    "outbox.oldest_age": ("_ObservableGauge", "s"),
    "outbox.bytes": ("_ObservableGauge", "By"),
    "transcript.turns_lost": ("_Counter", ""),
    "recordings.total": ("_Counter", ""),
    "runtime.event_loop_lag": ("_Histogram", "s"),
    "ready": ("_ObservableGauge", "1"),
}


def test_task_10c_l_declares_exact_eleven_non_ingress_metric_owners() -> None:
    from projetv0_voice.lifecycle import NON_INGRESS_METRIC_OWNERS

    assert NON_INGRESS_METRIC_OWNERS == {
        "actions.total": "call_control",
        "relay.runs": "relay_supervisor",
        "writer.queue_depth": "writer_snapshot",
        "writer.queue_oldest_age": "writer_snapshot",
        "writer.quick_check": "writer_snapshot",
        "outbox.depth": "writer_snapshot",
        "outbox.oldest_age": "writer_snapshot",
        "outbox.bytes": "writer_snapshot",
        "recordings.total": "recording_transitions",
        "runtime.event_loop_lag": "lag_supervisor",
        "ready": "runtime_publication",
    }


def test_typed_ingress_observation_finishes_exactly_once() -> None:
    from projetv0_voice.lifecycle import IngressMetricObservation, IngressMetricOutcome

    metrics = metrics_module.RuntimeMetrics.in_memory()
    observation = IngressMetricObservation(metrics)
    outcome = IngressMetricOutcome(
        webhook_class="initiated",
        receipt="none",
        disposition="unavailable",
        admission_rejection="capacity",
    )

    observation.finish(outcome)
    observation.finish(outcome)

    values = _metric_map(metrics)
    webhook = _points(values[PREFIX + "webhooks.total"])
    assert webhook[
        (
            ("disposition", "unavailable"),
            ("receipt", "none"),
            ("webhook_class", "initiated"),
        )
    ].value == 1
    rejection = _points(values[PREFIX + "admission.rejections"])
    assert rejection[(("reason", "capacity"),)].value == 1
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_measured_call_control_counts_one_final_semantic_result() -> None:
    from projetv0_voice.lifecycle import _MeasuredCallControl

    class Control:
        async def answer(self, *_args: object, **_kwargs: object) -> CallControlResult:
            return CallControlResult("accepted")

        async def start_streaming(
            self, *_args: object, **_kwargs: object
        ) -> CallControlResult:
            return CallControlResult("retryable_not_sent")

        async def hangup(self, *_args: object, **_kwargs: object) -> CallControlResult:
            raise RuntimeError("provider-secret")

    metrics = metrics_module.RuntimeMetrics.in_memory()
    measured = _MeasuredCallControl(Control(), metrics)
    request = StreamingStartV1(
        stream_url="wss://voice.invalid/telnyx/media",
        stream_auth_token="A" * 43,
    )

    assert (await measured.answer("control", command_id=object())).outcome == "accepted"
    assert (
        await measured.start_streaming("control", request, command_id=object())
    ).outcome == "retryable_not_sent"
    with pytest.raises(RuntimeError, match="provider-secret"):
        await measured.hangup("control", command_id=object())

    points = _points(_metric_map(metrics)[PREFIX + "actions.total"])
    assert points[(("action", "answer"), ("outcome", "accepted"))].value == 1
    assert points[
        (("action", "streaming_start"), ("outcome", "retryable_not_sent"))
    ].value == 1
    assert points[(("action", "hangup"), ("outcome", "internal_error"))].value == 1
    metrics._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


def _reader(owner: metrics_module.RuntimeMetrics) -> InMemoryMetricReader:
    reader = owner._metric_reader  # noqa: SLF001
    assert isinstance(reader, InMemoryMetricReader)
    return reader


def _metric_map(owner: metrics_module.RuntimeMetrics) -> dict[str, Any]:
    collected = _reader(owner).get_metrics_data()
    assert collected is not None
    return {
        metric.name: metric
        for resource_metrics in collected.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def _point(metric: Any) -> Any:
    points = list(metric.data.data_points)
    assert len(points) == 1
    return points[0]


def _points(metric: Any) -> dict[tuple[tuple[str, object], ...], Any]:
    return {
        tuple(sorted(point.attributes.items())): point
        for point in metric.data.data_points
    }


def test_call_metric_lease_balances_active_and_finishes_exactly_once() -> None:
    samples = iter((10.0, 12.5))
    owner = metrics_module.RuntimeMetrics.in_memory(
        monotonic=lambda: next(samples)
    )

    lease = owner.begin_call()
    active = _metric_map(owner)[PREFIX + "calls.active"]
    assert _point(active).value == 1
    assert repr(lease) == "CallMetricLease()"

    lease.finish("failed")
    lease.finish("drained")

    values = _metric_map(owner)
    assert _point(values[PREFIX + "calls.active"]).value == 0
    total = _points(values[PREFIX + "calls.total"])
    assert total[(("session", "failed"),)].value == 1
    assert (("session", "drained"),) not in total
    duration = _points(values[PREFIX + "sessions.duration"])
    assert duration[(("session", "failed"),)].count == 1
    assert duration[(("session", "failed"),)].sum == 2.5
    owner._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.parametrize("bad_clock", [math.inf, math.nan, -1.0, "bad"])
def test_call_metric_lease_clock_fault_is_nonthrowing_and_skips_duration(
    bad_clock: object,
) -> None:
    owner = metrics_module.RuntimeMetrics.in_memory(monotonic=lambda: bad_clock)

    lease = owner.begin_call()
    lease.finish("closed")

    values = _metric_map(owner)
    assert _point(values[PREFIX + "calls.active"]).value == 0
    assert _points(values[PREFIX + "calls.total"])[
        (("session", "closed"),)
    ].value == 1
    assert PREFIX + "sessions.duration" not in values
    assert owner.failure_code == "metrics_record_failed"
    owner._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


def test_call_metric_finish_instrument_fault_cannot_skip_remaining_emissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = metrics_module.RuntimeMetrics.in_memory(monotonic=iter((1.0, 2.0)).__next__)

    class BrokenCounter:
        def add(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("metric-secret")

    lease = owner.begin_call()
    monkeypatch.setattr(owner, "_calls_total", BrokenCounter())
    lease.finish("drained")

    values = _metric_map(owner)
    assert _point(values[PREFIX + "calls.active"]).value == 0
    assert _points(values[PREFIX + "sessions.duration"])[
        (("session", "drained"),)
    ].sum == 1.0
    assert owner.failure_code == "metrics_record_failed"
    owner._provider.shutdown(timeout_millis=10000.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_two_in_memory_owners_are_local_and_leave_globals_unchanged() -> None:
    meter_global = global_metrics.get_meter_provider()
    tracer_global = trace.get_tracer_provider()

    first = metrics_module.RuntimeMetrics.in_memory()
    second = metrics_module.RuntimeMetrics.in_memory()
    first.record_disclosure_timeout()

    first_data = _metric_map(first)
    second_data = _metric_map(second)
    assert _point(first_data[PREFIX + "disclosure.timeouts"]).value == 1
    assert PREFIX + "disclosure.timeouts" not in second_data
    assert first._provider is not second._provider  # noqa: SLF001
    assert global_metrics.get_meter_provider() is meter_global
    assert trace.get_tracer_provider() is tracer_global

    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_in_memory_provider_has_exact_resource_scope_filter_and_inventory() -> None:
    before_threads = {thread.ident for thread in threading.enumerate()}
    owner = metrics_module.RuntimeMetrics.in_memory()
    provider = owner._provider  # noqa: SLF001
    meter = owner._meter  # noqa: SLF001

    assert type(provider._sdk_config.resource) is Resource  # noqa: SLF001
    assert dict(provider._sdk_config.resource.attributes) == {  # noqa: SLF001
        "service.name": "projetv0-voice"
    }
    assert type(provider._sdk_config.exemplar_filter) is AlwaysOffExemplarFilter  # noqa: SLF001
    assert meter._instrumentation_scope.name == "projetv0.voice"  # noqa: SLF001
    assert meter._instrumentation_scope.version is None  # noqa: SLF001
    instruments = list(meter._instrument_id_instrument.values())  # noqa: SLF001
    assert len(instruments) == 22
    assert {
        instrument.name.removeprefix(PREFIX): (
            type(instrument).__name__,
            instrument.unit,
        )
        for instrument in instruments
    } == EXPECTED_INSTRUMENTS
    assert all(instrument.description == "" for instrument in instruments)
    assert {thread.ident for thread in threading.enumerate()} == before_threads

    await owner.aclose()


def test_closed_enum_catalogs_are_exact_and_complete() -> None:
    expected = {
        "session": frozenset({"closed", "failed", "drained"}),
        "reason": frozenset(
            {"capacity", "draining", "qualification", "persistence", "invalid"}
        ),
        "webhook_class": frozenset(
            {
                "initiated",
                "answered",
                "terminal",
                "recording",
                "unsupported",
                "invalid",
            }
        ),
        "receipt": frozenset({"none", "first", "duplicate"}),
        "disposition": frozenset(
            {
                "ok",
                "bad_request",
                "forbidden",
                "too_large",
                "unavailable",
                "internal_error",
            }
        ),
        "action": frozenset({"answer", "streaming_start", "hangup"}),
        "outcome": frozenset(
            {
                "accepted",
                "rejected",
                "rate_limited",
                "retryable_not_sent",
                "outcome_unknown",
                "internal_error",
            }
        ),
        "latency_kind": frozenset({"turn", "first_speech"}),
        "service": frozenset({"stt", "llm", "tts"}),
        "relay_status": frozenset(
            {
                "empty",
                "delivered",
                "retry_scheduled",
                "stale_claim",
                "claim_budget_expired",
                "degraded",
            }
        ),
        "recording_status": frozenset({"saved", "error", "purged"}),
    }
    actual = {
        "session": metrics_module._SESSIONS,  # noqa: SLF001
        "reason": metrics_module._REJECTION_REASONS,  # noqa: SLF001
        "webhook_class": metrics_module._WEBHOOK_CLASSES,  # noqa: SLF001
        "receipt": metrics_module._RECEIPTS,  # noqa: SLF001
        "disposition": metrics_module._DISPOSITIONS,  # noqa: SLF001
        "action": metrics_module._ACTIONS,  # noqa: SLF001
        "outcome": metrics_module._OUTCOMES,  # noqa: SLF001
        "latency_kind": metrics_module._LATENCY_KINDS,  # noqa: SLF001
        "service": metrics_module._SERVICES,  # noqa: SLF001
        "relay_status": metrics_module._RELAY_STATUSES,  # noqa: SLF001
        "recording_status": metrics_module._RECORDING_STATUSES,  # noqa: SLF001
    }

    assert actual == expected


@pytest.mark.asyncio
async def test_real_owners_disable_exemplars_while_default_negative_control_emits_ids() -> None:
    span = NonRecordingSpan(
        SpanContext(
            trace_id=0x123,
            span_id=0x456,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )

    negative_reader = InMemoryMetricReader()
    negative_provider = MeterProvider(
        metric_readers=(negative_reader,),
        shutdown_on_exit=False,
    )
    negative = negative_provider.get_meter("negative").create_histogram("negative")
    with use_span(span, end_on_exit=False):
        negative.record(0.25)
    negative_data = negative_reader.get_metrics_data()
    assert negative_data is not None
    negative_point = (
        negative_data.resource_metrics[0]
        .scope_metrics[0]
        .metrics[0]
        .data.data_points[0]
    )
    assert [(item.trace_id, item.span_id) for item in negative_point.exemplars] == [
        (0x123, 0x456)
    ]

    owner = metrics_module.RuntimeMetrics.in_memory()
    with use_span(span, end_on_exit=False):
        owner.record_user_bot_latency("turn", 0.25)
    point = _point(_metric_map(owner)[PREFIX + "user_bot_latency"])
    assert point.exemplars == []

    negative_provider.shutdown()
    await owner.aclose()


@pytest.mark.asyncio
async def test_semantic_methods_emit_only_exact_values_and_attribute_keys() -> None:
    owner = metrics_module.RuntimeMetrics.in_memory()
    owner.record_admission_rejection("capacity")
    owner.record_webhook("answered", "first", "ok")
    owner.record_action("streaming_start", "accepted")
    owner.record_user_bot_latency("turn", 0.0)
    owner.record_user_bot_latency("first_speech", 0.5)
    owner.record_service_ttfb("stt", 0.1)
    owner.record_service_ttfb("llm", 0.2)
    owner.record_service_ttfb("tts", 0.3)
    owner.record_disclosure_ack(0.4)
    owner.record_disclosure_timeout()
    owner.record_relay_run("delivered")
    owner.record_transcript_turn_lost()
    owner.record_recording("saved")
    owner.record_event_loop_lag(0.6)
    owner.update_writer_state(queue_depth=2, oldest_age=1.5, quick_check=True)
    owner.update_outbox_state(depth=3, oldest_age=2.5, bytes_count=4096)
    owner.update_ready(True)

    data = _metric_map(owner)
    expected_attributes = {
        "admission.rejections": {"reason": "capacity"},
        "webhooks.total": {
            "webhook_class": "answered",
            "receipt": "first",
            "disposition": "ok",
        },
        "actions.total": {"action": "streaming_start", "outcome": "accepted"},
        "user_bot_latency": {"latency_kind": "turn"},
        "service_ttfb": {"service": "stt"},
        "disclosure.mark_ack": {},
        "disclosure.timeouts": {},
        "relay.runs": {"relay_status": "delivered"},
        "writer.queue_depth": {},
        "writer.queue_oldest_age": {},
        "writer.quick_check": {},
        "outbox.depth": {},
        "outbox.oldest_age": {},
        "outbox.bytes": {},
        "transcript.turns_lost": {},
        "recordings.total": {"recording_status": "saved"},
        "runtime.event_loop_lag": {},
        "ready": {},
    }
    assert set(data) == {PREFIX + name for name in expected_attributes}
    for name, attributes in expected_attributes.items():
        points = list(data[PREFIX + name].data.data_points)
        assert dict(points[0].attributes) == attributes

    latency_points = list(data[PREFIX + "user_bot_latency"].data.data_points)
    assert [(dict(point.attributes), point.count, point.sum) for point in latency_points] == [
        ({"latency_kind": "turn"}, 1, 0.0),
        ({"latency_kind": "first_speech"}, 1, 0.5),
    ]
    ttfb_points = list(data[PREFIX + "service_ttfb"].data.data_points)
    assert [(dict(point.attributes), point.sum) for point in ttfb_points] == [
        ({"service": "stt"}, 0.1),
        ({"service": "llm"}, 0.2),
        ({"service": "tts"}, 0.3),
    ]
    assert _point(data[PREFIX + "writer.queue_depth"]).value == 2
    assert _point(data[PREFIX + "writer.queue_oldest_age"]).value == 1.5
    assert _point(data[PREFIX + "writer.quick_check"]).value == 1
    assert _point(data[PREFIX + "outbox.depth"]).value == 3
    assert _point(data[PREFIX + "outbox.oldest_age"]).value == 2.5
    assert _point(data[PREFIX + "outbox.bytes"]).value == 4096
    assert _point(data[PREFIX + "ready"]).value == 1
    assert owner.failure_code is None

    await owner.aclose()


class _StringSubclass(str):
    pass


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


class _RaisingIntFloat(int):
    def __float__(self) -> float:
        raise ValueError("numeric-hook-must-not-run")


class _RaisingFloatConversion(float):
    def __float__(self) -> float:
        raise TypeError("numeric-hook-must-not-run")


class _RaisingFloatComparison(float):
    def __ge__(self, _other: object) -> bool:
        raise RuntimeError("numeric-hook-must-not-run")


def _observable_snapshot(owner: metrics_module.RuntimeMetrics) -> dict[str, int | float]:
    data = _metric_map(owner)
    names = (
        "writer.queue_depth",
        "writer.queue_oldest_age",
        "writer.quick_check",
        "outbox.depth",
        "outbox.oldest_age",
        "outbox.bytes",
        "ready",
    )
    return {name: _point(data[PREFIX + name]).value for name in names}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(10**10000, id="oversized-int"),
        pytest.param(_IntSubclass(1), id="int-subclass"),
        pytest.param(_FloatSubclass(1.0), id="float-subclass"),
        pytest.param(_RaisingIntFloat(1), id="raising-int-float"),
        pytest.param(_RaisingFloatConversion(1.0), id="raising-float-conversion"),
        pytest.param(_RaisingFloatComparison(1.0), id="raising-float-comparison"),
    ],
)
@pytest.mark.parametrize(
    "consume",
    [
        pytest.param(
            lambda owner, value: owner.record_user_bot_latency("turn", value),
            id="user-bot-latency",
        ),
        pytest.param(
            lambda owner, value: owner.record_service_ttfb("stt", value),
            id="service-ttfb",
        ),
        pytest.param(
            lambda owner, value: owner.record_disclosure_ack(value),
            id="disclosure-ack",
        ),
        pytest.param(
            lambda owner, value: owner.record_event_loop_lag(value),
            id="event-loop-lag",
        ),
        pytest.param(
            lambda owner, value: owner.update_writer_state(
                queue_depth=4,
                oldest_age=value,
                quick_check=False,
            ),
            id="writer-oldest-age",
        ),
        pytest.param(
            lambda owner, value: owner.update_outbox_state(
                depth=5,
                oldest_age=value,
                bytes_count=6,
            ),
            id="outbox-oldest-age",
        ),
    ],
)
async def test_float_numeric_consumers_reject_unsafe_values_atomically(
    consume: Callable[[metrics_module.RuntimeMetrics, object], None],
    invalid: object,
) -> None:
    owner = metrics_module.RuntimeMetrics.in_memory()
    owner.update_writer_state(queue_depth=2, oldest_age=1.5, quick_check=True)
    owner.update_outbox_state(depth=3, oldest_age=2.5, bytes_count=4096)
    owner.update_ready(True)
    before = _observable_snapshot(owner)

    consume(owner, invalid)

    assert owner.failure_code == "metrics_record_failed"
    assert _observable_snapshot(owner) == before
    assert set(_metric_map(owner)) == {
        PREFIX + "writer.queue_depth",
        PREFIX + "writer.queue_oldest_age",
        PREFIX + "writer.quick_check",
        PREFIX + "outbox.depth",
        PREFIX + "outbox.oldest_age",
        PREFIX + "outbox.bytes",
        PREFIX + "ready",
    }
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            lambda owner, value: owner.update_writer_state(
                queue_depth=value,
                oldest_age=1.5,
                quick_check=True,
            ),
            id="writer-queue-depth",
        ),
        pytest.param(
            lambda owner, value: owner.update_outbox_state(
                depth=value,
                oldest_age=2.5,
                bytes_count=4096,
            ),
            id="outbox-depth",
        ),
        pytest.param(
            lambda owner, value: owner.update_outbox_state(
                depth=3,
                oldest_age=2.5,
                bytes_count=value,
            ),
            id="outbox-bytes",
        ),
    ],
)
async def test_integer_gauges_accept_int64_max_and_encode(
    update: Callable[[metrics_module.RuntimeMetrics, object], None],
) -> None:
    owner = metrics_module.RuntimeMetrics.in_memory()

    update(owner, (1 << 63) - 1)

    assert owner.failure_code is None
    collected = _reader(owner).get_metrics_data()
    assert collected is not None
    assert encode_metrics(collected) is not None
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(1 << 63, id="above-int64"),
        pytest.param(10**10000, id="oversized-int"),
    ],
)
@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            lambda owner, value: owner.update_writer_state(
                queue_depth=value,
                oldest_age=9.5,
                quick_check=False,
            ),
            id="writer-queue-depth",
        ),
        pytest.param(
            lambda owner, value: owner.update_outbox_state(
                depth=value,
                oldest_age=9.5,
                bytes_count=99,
            ),
            id="outbox-depth",
        ),
        pytest.param(
            lambda owner, value: owner.update_outbox_state(
                depth=99,
                oldest_age=9.5,
                bytes_count=value,
            ),
            id="outbox-bytes",
        ),
    ],
)
async def test_integer_gauges_reject_out_of_range_atomically_and_still_encode(
    update: Callable[[metrics_module.RuntimeMetrics, object], None],
    invalid: object,
) -> None:
    owner = metrics_module.RuntimeMetrics.in_memory()
    owner.update_writer_state(queue_depth=2, oldest_age=1.5, quick_check=True)
    owner.update_outbox_state(depth=3, oldest_age=2.5, bytes_count=4096)
    owner.update_ready(True)
    before = _observable_snapshot(owner)

    update(owner, invalid)

    assert owner.failure_code == "metrics_record_failed"
    assert _observable_snapshot(owner) == before
    collected = _reader(owner).get_metrics_data()
    assert collected is not None
    assert encode_metrics(collected) is not None
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoke",
    [
        lambda owner: owner.record_admission_rejection("free-form"),
        lambda owner: owner.record_admission_rejection(_StringSubclass("capacity")),
        lambda owner: owner.record_webhook("invalid-class", "first", "ok"),
        lambda owner: owner.record_webhook("answered", "invalid-receipt", "ok"),
        lambda owner: owner.record_webhook("answered", "first", "invalid-disposition"),
        lambda owner: owner.record_action("invalid-action", "accepted"),
        lambda owner: owner.record_action("answer", "invalid-outcome"),
        lambda owner: owner.record_user_bot_latency("invalid-kind", 0.1),
        lambda owner: owner.record_user_bot_latency("turn", -0.1),
        lambda owner: owner.record_user_bot_latency("turn", True),
        lambda owner: owner.record_user_bot_latency("turn", math.nan),
        lambda owner: owner.record_service_ttfb("invalid-service", 0.1),
        lambda owner: owner.record_service_ttfb("stt", 0.0),
        lambda owner: owner.record_service_ttfb("stt", math.inf),
        lambda owner: owner.record_disclosure_ack(-0.1),
        lambda owner: owner.record_relay_run("invalid-relay"),
        lambda owner: owner.record_recording("invalid-recording"),
        lambda owner: owner.record_event_loop_lag(False),
        lambda owner: owner.update_writer_state(
            queue_depth=-1,
            oldest_age=0.0,
            quick_check=True,
        ),
        lambda owner: owner.update_writer_state(
            queue_depth=0,
            oldest_age=math.nan,
            quick_check=True,
        ),
        lambda owner: owner.update_writer_state(
            queue_depth=0,
            oldest_age=0.0,
            quick_check=1,
        ),
        lambda owner: owner.update_outbox_state(
            depth=0,
            oldest_age=0.0,
            bytes_count=True,
        ),
        lambda owner: owner.update_ready(1),
    ],
)
async def test_invalid_semantic_input_latches_only_constant_and_reflects_nothing(
    invoke: Callable[[metrics_module.RuntimeMetrics], None],
) -> None:
    owner = metrics_module.RuntimeMetrics.in_memory()

    invoke(owner)

    assert owner.failure_code == "metrics_record_failed"
    assert set(_metric_map(owner)) == {
        PREFIX + "writer.queue_depth",
        PREFIX + "writer.queue_oldest_age",
        PREFIX + "writer.quick_check",
        PREFIX + "outbox.depth",
        PREFIX + "outbox.oldest_age",
        PREFIX + "outbox.bytes",
        PREFIX + "ready",
    }
    await owner.aclose()


@pytest.mark.asyncio
async def test_instrument_fault_is_fail_open_constant_safe_and_has_no_generic_surface() -> None:
    owner = metrics_module.RuntimeMetrics.in_memory()

    class _FailingHistogram:
        def record(self, _value: object, _attributes: object) -> None:
            raise RuntimeError("metric-instrument-secret")

    owner._user_bot_latency = _FailingHistogram()  # type: ignore[assignment]  # noqa: SLF001
    owner.record_user_bot_latency("turn", 0.1)

    assert owner.failure_code == "metrics_record_failed"
    assert "metric-instrument-secret" not in repr(owner)
    assert not hasattr(owner, "record")
    assert not hasattr(owner, "meter")
    assert not hasattr(owner, "instruments")
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "timeout", "transport", "segment_limit", "text_limit", "text_invalid", "drain_timeout"
])
async def test_stt_failure_counter_accepts_only_fixed_reason_attribute(reason):
    owner = metrics_module.RuntimeMetrics.in_memory()
    try:
        owner.record_stt_failure(reason)
        values = _points(_metric_map(owner)[PREFIX + "stt.failures"])
        assert set(values) == {(("reason", reason),)}
        assert values[(("reason", reason),)].value == 1
        assert owner.failure_code is None
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["provider-private-secret", "stt_failed", None, True, [], 8])
async def test_stt_failure_counter_rejects_raw_or_unknown_values_with_existing_safe_latch(reason):
    owner = metrics_module.RuntimeMetrics.in_memory()
    try:
        owner.record_stt_failure(reason)
        assert owner.failure_code == "metrics_record_failed"
        assert PREFIX + "stt.failures" not in _metric_map(owner)
        assert "provider-private-secret" not in repr(_metric_map(owner))
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_stt_failure_counter_instrument_fault_uses_existing_constant_safe_latch():
    owner = metrics_module.RuntimeMetrics.in_memory()

    class FailedCounter:
        def add(self, _value, _attributes):
            raise RuntimeError("metric-instrument-private-secret")

    owner._stt_failures = FailedCounter()  # noqa: SLF001
    try:
        owner.record_stt_failure("timeout")
        assert owner.failure_code == "metrics_record_failed"
        assert "metric-instrument-private-secret" not in repr(owner)
    finally:
        await owner.aclose()


class _FakeInstrument:
    def add(self, _value: object, _attributes: object = None) -> None:
        return None

    def record(self, _value: object, _attributes: object = None) -> None:
        return None


class _FakeMeter:
    def create_counter(self, *_args: object, **_kwargs: object) -> _FakeInstrument:
        return _FakeInstrument()

    def create_up_down_counter(self, *_args: object, **_kwargs: object) -> _FakeInstrument:
        return _FakeInstrument()

    def create_histogram(self, *_args: object, **_kwargs: object) -> _FakeInstrument:
        return _FakeInstrument()

    def create_observable_gauge(self, *_args: object, **_kwargs: object) -> _FakeInstrument:
        return _FakeInstrument()


class _FakeProvider:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.meter_calls: list[tuple[str, str | None]] = []
        self.shutdown_calls: list[float] = []

    def get_meter(self, name: str, version: str | None = None) -> _FakeMeter:
        self.meter_calls.append((name, version))
        return _FakeMeter()

    def shutdown(self, timeout_millis: float) -> None:
        self.shutdown_calls.append(timeout_millis)


def _published_snapshot(
    generation: int,
    *,
    base: int,
    ready: bool,
) -> metrics_module.RuntimePublishedSnapshot:
    return metrics_module.RuntimePublishedSnapshot(
        generation=generation,
        ready=ready,
        draining=False,
        writer_queue_depth=base,
        writer_queue_oldest_age=float(base) + 0.1,
        writer_quick_check=ready,
        outbox_depth=base + 1,
        outbox_oldest_age=float(base) + 0.2,
        outbox_bytes=base + 2,
        storage_bytes=base + 3,
        startup_profile_and_stale_recovery_complete=ready,
        admission_open=ready,
        qualification_state_valid=ready,
        writer_owner_alive_and_ready=ready,
        no_writer_fatal_or_degradation=ready,
        relay_supervisor_alive=ready,
        no_permanent_relay_or_sink_degradation=ready,
    )


def _observable_callback_values(owner: metrics_module.RuntimeMetrics) -> tuple[object, ...]:
    callbacks = (
        owner._observe_writer_queue_depth,  # noqa: SLF001
        owner._observe_writer_queue_oldest_age,  # noqa: SLF001
        owner._observe_writer_quick_check,  # noqa: SLF001
        owner._observe_outbox_depth,  # noqa: SLF001
        owner._observe_outbox_oldest_age,  # noqa: SLF001
        owner._observe_outbox_bytes,  # noqa: SLF001
        owner._observe_ready,  # noqa: SLF001
    )
    return tuple(next(iter(callback(None))).value for callback in callbacks)  # type: ignore[arg-type]


def test_runtime_publication_is_frozen_and_rejects_nonmonotone_generation() -> None:
    old = _published_snapshot(7, base=10, ready=False)
    publication = metrics_module.RuntimePublication(old)

    assert publication.snapshot() is old
    assert repr(publication) == "RuntimePublication()"
    with pytest.raises((AttributeError, TypeError)):
        old.ready = True
    for generation in (6, 7):
        with pytest.raises(RuntimeError, match="^runtime_publication_invalid$") as caught:
            publication.publish(_published_snapshot(generation, base=20, ready=True))
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    assert publication.snapshot() is old


def test_observable_callbacks_see_only_complete_old_or_new_publication() -> None:
    old = _published_snapshot(1, base=10, ready=False)
    publication = metrics_module.RuntimePublication(old)
    owner = metrics_module.RuntimeMetrics._from_provider(  # noqa: SLF001
        _FakeProvider(),
        metric_reader=None,
        publication=publication,
    )
    first_candidate_scalar = threading.Event()
    finish_candidate = threading.Event()
    candidate: dict[str, object] = {
        "generation": 2,
        "ready": True,
        "draining": False,
        "writer_queue_depth": 20,
    }

    def build_then_publish() -> None:
        first_candidate_scalar.set()
        assert finish_candidate.wait(timeout=5)
        candidate.update(
            {
                "writer_queue_oldest_age": 20.1,
                "writer_quick_check": True,
                "outbox_depth": 21,
                "outbox_oldest_age": 20.2,
                "outbox_bytes": 22,
                "storage_bytes": 23,
                "startup_profile_and_stale_recovery_complete": True,
                "admission_open": True,
                "qualification_state_valid": True,
                "writer_owner_alive_and_ready": True,
                "no_writer_fatal_or_degradation": True,
                "relay_supervisor_alive": True,
                "no_permanent_relay_or_sink_degradation": True,
            }
        )
        publication.publish(metrics_module.RuntimePublishedSnapshot(**candidate))

    publisher = threading.Thread(target=build_then_publish, name="snapshot-publisher")
    publisher.start()
    assert first_candidate_scalar.wait(timeout=5)
    before = _observable_callback_values(owner)
    finish_candidate.set()
    publisher.join(timeout=5)
    assert publisher.is_alive() is False
    after = _observable_callback_values(owner)

    assert before == (10, 10.1, 0, 11, 10.2, 12, 0)
    assert after == (20, 20.1, 1, 21, 20.2, 22, 1)


@pytest.mark.asyncio
async def test_production_builder_passes_exact_http_reader_provider_values() -> None:
    calls: dict[str, Any] = {}
    exporter_result = object()
    reader_result = object()

    def exporter_factory(**kwargs: object) -> object:
        calls["exporter"] = kwargs
        return exporter_result

    def reader_factory(exporter: object, **kwargs: object) -> object:
        calls["reader"] = (exporter, kwargs)
        return reader_result

    def provider_factory(**kwargs: object) -> _FakeProvider:
        provider = _FakeProvider(**kwargs)
        calls["provider"] = provider
        return provider

    token = _production_token()
    owner = metrics_module._build_production(  # noqa: SLF001
        token,
        endpoint=ENDPOINT,
        exporter_factory=exporter_factory,
        reader_factory=reader_factory,
        provider_factory=provider_factory,
        session_factory=requests.Session,
    )

    exporter_args = calls["exporter"]
    assert set(exporter_args) == {
        "endpoint",
        "headers",
        "timeout",
        "compression",
        "session",
    }
    assert exporter_args["endpoint"] == ENDPOINT
    assert exporter_args["headers"] == {"User-Agent": "projetv0-voice"}
    assert exporter_args["headers"] is not metrics_module._OTLP_HEADERS  # noqa: SLF001
    assert exporter_args["timeout"] == 2.0
    assert exporter_args["compression"] is Compression.NoCompression
    session = exporter_args["session"]
    assert isinstance(session, requests.Session)
    assert session.trust_env is False
    assert session.max_redirects == 0
    assert session.proxies == {}
    assert session.cert is None

    exporter, reader_args = calls["reader"]
    assert exporter is exporter_result
    assert reader_args == {
        "export_interval_millis": 30000.0,
        "export_timeout_millis": 5000.0,
    }
    provider = calls["provider"]
    assert provider.meter_calls == [("projetv0.voice", None)]
    assert provider.kwargs["metric_readers"] == (reader_result,)
    assert type(provider.kwargs["resource"]) is Resource
    assert dict(provider.kwargs["resource"].attributes) == {
        "service.name": "projetv0-voice"
    }
    assert type(provider.kwargs["exemplar_filter"]) is AlwaysOffExemplarFilter
    assert provider.kwargs["shutdown_on_exit"] is False
    assert 0 < 2000 <= 5000 < 10000 < 30000

    await owner.aclose()
    assert provider.shutdown_calls == [10000.0]


def test_production_uses_explicit_settings_endpoint_after_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _production_token()
    selected: list[object] = []

    def build(
        received_token: ObservabilityBootstrapToken,
        *,
        endpoint: object,
    ) -> object:
        assert received_token is token
        selected.append(endpoint)
        return object()

    monkeypatch.setattr(metrics_module, "_build_production", build)
    monkeypatch.setitem(
        os.environ,
        "VOICE_OTLP_HTTP_ENDPOINT",
        "https://mutated.invalid/v1/metrics",
    )

    owner = metrics_module.RuntimeMetrics.production(token, endpoint=ENDPOINT)

    assert owner is not None
    assert selected == [ENDPOINT]


def test_production_rejects_token_and_settings_endpoint_mismatch_before_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _production_settings()
    second = _production_settings(endpoint="https://other.invalid/v1/metrics")
    monkeypatch.setattr(
        metrics_module,
        "_build_production",
        lambda *_args, **_kwargs: pytest.fail(
            "cross-settings mismatch reached production construction",
            pytrace=False,
        ),
    )

    with pytest.raises(RuntimeError, match="^observability_endpoint_mismatch$") as caught:
        metrics_module.RuntimeMetrics.production(
            first.observability_token(),
            endpoint=second.otlp_http_endpoint,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="^observability_endpoint_mismatch$"):
        metrics_module._build_production(  # noqa: SLF001
            first.observability_token(),
            endpoint=second.otlp_http_endpoint,
        )


def test_direct_and_guard_only_tokens_cannot_authorize_production_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        ObservabilityBootstrapToken()
    with pytest.raises(TypeError):
        ObservabilityBootstrapToken(ENDPOINT)  # type: ignore[call-arg]

    tokens = (_validate_observability_mapping({"VOICE_OTLP_HTTP_ENDPOINT": ENDPOINT}),)
    monkeypatch.setattr(
        metrics_module,
        "_build_production",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid token reached production construction",
            pytrace=False,
        ),
    )
    for token in tokens:
        with pytest.raises(
            ValueError,
            match="^observability_bootstrap_token_invalid$",
        ):
            metrics_module.RuntimeMetrics.production(token, endpoint=ENDPOINT)

    monkeypatch.undo()
    for token in tokens:
        with pytest.raises(
            ValueError,
            match="^observability_bootstrap_token_invalid$",
        ) as caught:
            metrics_module._build_production(  # noqa: SLF001
                token,
                endpoint=ENDPOINT,
                exporter_factory=lambda **_kwargs: object(),
                reader_factory=lambda *_args, **_kwargs: object(),
                provider_factory=_FakeProvider,
                session_factory=requests.Session,
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


class _LocalExporter(MetricExporter):
    def __init__(self, *, fail: bool = False, **_kwargs: object) -> None:
        super().__init__()
        self.fail = fail
        self.exports = 0
        self.shutdowns = 0

    def export(
        self,
        metrics_data: object,
        timeout_millis: float = 10000,
        **kwargs: object,
    ) -> MetricExportResult:
        del metrics_data, timeout_millis, kwargs
        self.exports += 1
        if self.fail:
            raise RuntimeError("async-export-secret")
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10000) -> bool:
        del timeout_millis
        return True

    def shutdown(self, timeout_millis: float = 30000, **kwargs: object) -> None:
        del timeout_millis, kwargs
        self.shutdowns += 1


@pytest.mark.asyncio
async def test_production_reader_has_real_thread_and_async_export_failure_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = _production_token()
    dependency_logging.configure_dependency_logging(token)
    exporter = _LocalExporter(fail=True)
    before = {thread.ident for thread in threading.enumerate()}
    owner = metrics_module._build_production(  # noqa: SLF001
        token,
        endpoint=ENDPOINT,
        exporter_factory=lambda **_kwargs: exporter,
        reader_factory=PeriodicExportingMetricReader,
        provider_factory=MeterProvider,
        session_factory=requests.Session,
    )
    owner.record_disclosure_timeout()
    created = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in before and thread.name == "OtelPeriodicExportingMetricReader"
    ]
    assert len(created) == 1

    with caplog.at_level(logging.DEBUG):
        await owner.aclose()

    assert exporter.exports == 1
    assert exporter.shutdowns == 1
    assert "async-export-secret" not in caplog.text


@pytest.mark.asyncio
async def test_real_failure_boundary_never_reflects_one_privacy_sentinel(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "".join(("private", "-metric", "-sentinel"))

    def assert_sentinel_absent(*values: object) -> None:
        for value in values:
            if sentinel in str(value) or sentinel in repr(value):
                pytest.fail("privacy sentinel leaked", pytrace=False)

    class _SentinelExporter(_LocalExporter):
        def __init__(self) -> None:
            super().__init__()
            self.metrics_data: object | None = None

        def export(
            self,
            metrics_data: object,
            timeout_millis: float = 10000,
            **kwargs: object,
        ) -> MetricExportResult:
            del timeout_millis, kwargs
            self.metrics_data = metrics_data
            raise RuntimeError(sentinel)

    class _SentinelInstrument:
        def record(self, _value: object, _attributes: object) -> None:
            raise RuntimeError(sentinel)

    class _ShutdownFailure:
        def __init__(self, provider: MeterProvider) -> None:
            self._provider = provider

        def shutdown(self, timeout_millis: float) -> None:
            self._provider.shutdown(timeout_millis=timeout_millis)
            raise RuntimeError(sentinel)

    token = _production_token()
    dependency_logging.configure_dependency_logging(token)
    exporter = _SentinelExporter()
    owner = metrics_module._build_production(  # noqa: SLF001
        token,
        endpoint=ENDPOINT,
        exporter_factory=lambda **_kwargs: exporter,
        reader_factory=PeriodicExportingMetricReader,
        provider_factory=MeterProvider,
        session_factory=requests.Session,
    )
    owner._user_bot_latency = _SentinelInstrument()  # type: ignore[assignment]  # noqa: SLF001
    owner.record_user_bot_latency("turn", 0.25)
    owner.record_disclosure_timeout()
    provider = owner._provider  # noqa: SLF001
    owner._provider = _ShutdownFailure(provider)  # noqa: SLF001
    loguru_messages: list[str] = []
    loguru_sink = logger.add(loguru_messages.append, format="{message}")
    caplog.clear()
    try:
        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(RuntimeError, match="^metrics_shutdown_failed$") as caught,
        ):
            await owner.aclose()
    finally:
        logger.remove(loguru_sink)

    stdout, stderr = capsys.readouterr()
    assert owner.failure_code == "metrics_record_failed"
    assert exporter.metrics_data is not None
    assert_sentinel_absent(stdout, stderr, caplog.text, loguru_messages, owner)

    chain: list[BaseException] = []
    pending: list[BaseException | None] = [caught.value]
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        chain.append(error)
        pending.extend((error.__cause__, error.__context__))
    assert_sentinel_absent(*chain)

    exported = exporter.metrics_data
    assert exported is not None
    serialized_surfaces: list[object] = []
    for resource_metrics in exported.resource_metrics:
        serialized_surfaces.append(dict(resource_metrics.resource.attributes))
        for scope_metrics in resource_metrics.scope_metrics:
            serialized_surfaces.extend(
                (
                    scope_metrics.scope.name,
                    scope_metrics.scope.version,
                    scope_metrics.schema_url,
                )
            )
            for metric in scope_metrics.metrics:
                serialized_surfaces.extend((metric.name, metric.description, metric.unit))
                for point in metric.data.data_points:
                    serialized_surfaces.extend(
                        (
                            dict(point.attributes),
                            getattr(point, "value", None),
                            getattr(point, "sum", None),
                            list(point.exemplars),
                        )
                    )
    assert_sentinel_absent(*serialized_surfaces)


class _BlockingProvider(_FakeProvider):
    def __init__(self, *, failure: bool = False) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.failure = failure

    def shutdown(self, timeout_millis: float) -> None:
        self.shutdown_calls.append(timeout_millis)
        self.started.set()
        assert self.release.wait(timeout=5)
        if self.failure:
            raise RuntimeError("provider-shutdown-secret")


@pytest.mark.asyncio
@pytest.mark.parametrize("callers", [2, 10])
async def test_aclose_concurrent_callers_share_one_provider_shutdown(callers: int) -> None:
    provider = _BlockingProvider()
    owner = metrics_module.RuntimeMetrics._from_provider(provider, metric_reader=None)  # noqa: SLF001
    tasks = [asyncio.create_task(owner.aclose()) for _ in range(callers)]
    assert await asyncio.to_thread(provider.started.wait, 2)
    provider.release.set()

    await asyncio.gather(*tasks)

    assert provider.shutdown_calls == [10000.0]


@pytest.mark.asyncio
async def test_aclose_waits_through_repeated_cancellation_while_loop_heartbeats() -> None:
    provider = _BlockingProvider()
    owner = metrics_module.RuntimeMetrics._from_provider(provider, metric_reader=None)  # noqa: SLF001
    heartbeat = 0
    stop_heartbeat = asyncio.Event()

    async def count_heartbeats() -> None:
        nonlocal heartbeat
        while not stop_heartbeat.is_set():
            heartbeat += 1
            await asyncio.sleep(0)

    beat_task = asyncio.create_task(count_heartbeats())
    close_task = asyncio.create_task(owner.aclose())
    assert await asyncio.to_thread(provider.started.wait, 2)
    close_task.cancel("first-cancel")
    await asyncio.sleep(0)
    close_task.cancel("second-cancel")
    await asyncio.sleep(0.01)
    assert close_task.done() is False
    assert heartbeat > 1
    provider.release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await close_task
    stop_heartbeat.set()
    await beat_task

    assert caught.value.args == ("first-cancel",)
    assert provider.shutdown_calls == [10000.0]


@pytest.mark.asyncio
async def test_aclose_collapses_provider_failure_without_cause() -> None:
    provider = _BlockingProvider(failure=True)
    owner = metrics_module.RuntimeMetrics._from_provider(provider, metric_reader=None)  # noqa: SLF001
    close_task = asyncio.create_task(owner.aclose())
    assert await asyncio.to_thread(provider.started.wait, 2)
    provider.release.set()

    with pytest.raises(RuntimeError, match="^metrics_shutdown_failed$") as caught:
        await close_task

    assert caught.value.__cause__ is None
    assert "provider-shutdown-secret" not in str(caught.value)
    assert provider.shutdown_calls == [10000.0]


def test_production_rejects_non_exact_token_without_importing_endpoint_value() -> None:
    class _TokenSubclass(ObservabilityBootstrapToken):
        pass

    for value in (object(), object.__new__(_TokenSubclass)):
        with pytest.raises(ValueError, match="^observability_bootstrap_token_invalid$"):
            metrics_module._build_production(  # type: ignore[arg-type]  # noqa: SLF001
                value,
                endpoint=ENDPOINT,
            )
