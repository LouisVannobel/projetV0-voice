from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections.abc import Callable
from typing import Any

import pytest
import requests
from opentelemetry import metrics as global_metrics
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.exemplar import AlwaysOffExemplarFilter
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

ENDPOINT = "https://collector.invalid/tenant/v1/metrics"
PREFIX = "projetv0.voice."


EXPECTED_INSTRUMENTS = {
    "calls.active": ("_UpDownCounter", ""),
    "calls.total": ("_Counter", ""),
    "admission.rejections": ("_Counter", ""),
    "webhooks.total": ("_Counter", ""),
    "actions.total": ("_Counter", ""),
    "sessions.duration": ("_Histogram", "s"),
    "user_bot_latency": ("_Histogram", "s"),
    "service_ttfb": ("_Histogram", "s"),
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
    assert len(instruments) == 21
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

    token = _validate_observability_mapping({"VOICE_OTLP_HTTP_ENDPOINT": ENDPOINT})
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
    token = _validate_observability_mapping({"VOICE_OTLP_HTTP_ENDPOINT": ENDPOINT})
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

    for value in (object(), _TokenSubclass()):
        with pytest.raises(ValueError, match="^observability_bootstrap_token_invalid$"):
            metrics_module._build_production(  # type: ignore[arg-type]  # noqa: SLF001
                value,
                endpoint=ENDPOINT,
            )
