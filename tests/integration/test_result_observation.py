from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from openai import DefaultAsyncHttpxClient
from opentelemetry.exporter.otlp.proto.common._internal.metrics_encoder import encode_metrics
from pydantic import SecretStr

from projetv0_voice.inference import services
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.persistence.business_result import RetainedCall, RetainedTurn
from projetv0_voice.pipeline import FirstFailure
from projetv0_voice.qualified_profile import InferenceProfileV1
from projetv0_voice.session import CallSession, _TerminalOutcome

TURN_ID = UUID(int=101)
RESULT = {
    "schema_version": 1, "quality": "partial", "category": "callback",
    "summary": "Rappel demandé.", "next_action": "Rappeler.",
    "contact": {
        "name": None, "callback_e164": None, "preference": None,
        "callback_source": "missing", "callback_confirmed": False,
    },
    "evidence": [{"turn_id": str(TURN_ID), "role": "user"}],
    "request_confirmed": False,
}


class _Writer:
    def __init__(self) -> None:
        self.retained = RetainedCall(
            (RetainedTurn(TURN_ID, 1, "user", "Pouvez-vous me rappeler ?", False),), 0,
        )
        self.frozen = None
        self.facts = None
        self.read_error = None

    async def read_frozen_call_publication(self, _call_id):
        if self.read_error is not None:
            raise self.read_error
        return self.frozen

    async def read_retained_call(self, _call_id):
        return self.retained

    async def read_call_lifecycle(self, _call_id):
        return self.facts


def _session(metrics, inference, *, timeout=1):
    session = CallSession.__new__(CallSession)
    session._identity = SimpleNamespace(
        call_id=UUID(int=1), routing=SimpleNamespace(from_e164=None),
    )
    session._writer = _Writer()
    session._services = SimpleNamespace(llm=SimpleNamespace(run_inference=inference))
    session._runtime_metrics = metrics
    session._no_new_ai = session._result_inference_fenced = False
    session._partial_result = session._result_inference_task = None
    session._terminal_publication = None
    session._cleanup_phase_timeout_seconds = timeout
    session._controller = session._recorder = None
    session._registry_terminalizer = session._end_call_playback = None
    return session


def _assert_observed(metrics, expected, *, duration=None):
    data = metrics._metric_reader.get_metrics_data()
    by_name = {
        m.name: m
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for m in scope.metrics
    }
    total = by_name["projetv0.voice.result.outcomes"].data.data_points
    elapsed = by_name["projetv0.voice.result.duration"].data.data_points
    assert len(total) == len(elapsed) == 1
    assert dict(total[0].attributes) == dict(elapsed[0].attributes) == {"outcome": expected}
    assert total[0].value == elapsed[0].count == 1
    assert math.isfinite(elapsed[0].sum) and elapsed[0].sum >= 0
    if duration is not None:
        assert elapsed[0].sum == duration
    assert not total[0].exemplars and not elapsed[0].exemplars
    encoded = encode_metrics(data).SerializeToString()
    for secret in (b"private-error", b"Pouvez-vous", str(TURN_ID).encode(), b"call-secret"):
        assert secret not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("case, expected", [
    ("valid", "valid"), ("null", "empty"), ("none", "empty"),
    ("no_caller", "empty"), ("schema", "invalid"), ("provenance", "invalid"),
    ("provider", "error"), ("provider_timeout", "error"), ("provider_cancel", "cancelled"),
    ("routing", "not_started"), ("frozen", "frozen_replay"), ("erased", "fenced"),
    ("transfer", "fenced"), ("stopped", "fenced"), ("fence", "fenced"),
    ("late_fence", "fenced"),
])
async def test_native_result_observes_closed_outcome_without_changing_return(case, expected):
    metrics = RuntimeMetrics.in_memory(monotonic=iter((10.0, 10.25)).__next__)
    requests = []

    async def inference(context, **kwargs):
        requests.append((context, kwargs))
        if case == "provider":
            raise RuntimeError("private-error call-secret")
        if case == "provider_timeout":
            raise TimeoutError("private-error")
        if case == "provider_cancel":
            raise asyncio.CancelledError("private-error")
        if case == "late_fence":
            # Revocation after a valid provider response must not count as valid.
            session._result_inference_fenced = True
        if case == "schema":
            return '{"private-error": "call-secret"}'
        if case == "provenance":
            return json.dumps({**RESULT, "evidence": [{"turn_id": str(UUID(int=999)),
                                                       "role": "user"}]})
        return {"null": "null", "none": None}.get(case, json.dumps(RESULT))

    session = _session(metrics, inference)
    if case == "routing":
        session._identity.routing = None
    elif case == "frozen":
        session._writer.frozen = object()
    elif case == "erased":
        session._writer.retained = RetainedCall((), 0, erased=True)
    elif case == "no_caller":
        session._writer.retained = RetainedCall((), 0)
    elif case == "transfer":
        session._writer.facts = SimpleNamespace(transfer_fenced=True)
    elif case == "stopped":
        session._no_new_ai = True
    elif case == "fence":
        session.stop_result_inference()
    try:
        assert await session._prepare_partial_result() is None
        assert (session._partial_result is not None) == (case == "valid")
        if case == "frozen":
            assert session._terminal_publication is session._writer.frozen
        assert len(requests) == (0 if case in {
            "routing", "frozen", "erased", "no_caller", "transfer", "stopped", "fence",
        } else 1)
        if requests:
            assert requests[0][1]["max_tokens"] == 2048
        if session._result_inference_task is not None:
            assert session._result_inference_task.done()
        _assert_observed(metrics, expected, duration=0.25)
    finally:
        await metrics.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cause, expected", [
    ("inner", "inner_timeout"), ("cancel", "cancelled"), ("takeover", "fenced"),
])
async def test_native_result_joins_exact_owned_task_and_identifies_cancellation(cause, expected):
    metrics = RuntimeMetrics.in_memory()
    entered, joined = asyncio.Event(), asyncio.Event()

    async def inference(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            joined.set()

    session = _session(metrics, inference, timeout=0.05 if cause == "inner" else 1)
    preparation = asyncio.create_task(session._prepare_partial_result())
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            owned = session._result_inference_task
            if cause == "cancel":
                preparation.cancel()
            elif cause == "takeover":
                session.stop_result_inference()
            assert await preparation is None
        assert joined.is_set() and owned.done() and owned.cancelled()
        assert session._result_inference_task is owned and session._partial_result is None
        _assert_observed(metrics, expected)
    finally:
        preparation.cancel()
        await asyncio.gather(preparation, return_exceptions=True)
        await metrics.aclose()


@pytest.mark.asyncio
async def test_native_cleanup_outer_deadline_is_observed_but_keeps_first_failure_masked():
    metrics = RuntimeMetrics.in_memory()
    joined = asyncio.Event()

    async def inference(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            joined.set()

    async def nothing(*_args, **_kwargs):
        pass

    session = _session(metrics, inference, timeout=0.1)
    # The outer timeout starts before writer I/O, so the inner handle is later.
    original = session._writer.read_retained_call

    async def slow_read(call_id):
        await asyncio.sleep(0.03)
        return await original(call_id)

    session._writer.read_retained_call = slow_read
    peer = SimpleNamespace(cleanup=nothing)
    session._services.stt = session._services.tts = peer
    session._services.llm.cleanup = nothing
    session._services.aclose = nothing
    session._finish_durable_boundaries = nothing
    first_failure = FirstFailure()
    terminal = _TerminalOutcome("closed")
    try:
        await session._cleanup_owned_state(
            controller=SimpleNamespace(terminalize_and_join=nothing, cleanup_termination=nothing,
                                       disclosure_completed=True),
            recorder=SimpleNamespace(close=lambda: None), first_failure=first_failure,
            transport=SimpleNamespace(input=lambda: peer, output=lambda: peer),
            pipeline=None, runtime=None, runner_task=None, failure_task=None,
            terminal_outcome=terminal, cancel_continuations=False,
        )
        assert joined.is_set() and session._result_inference_task.cancelled()
        assert session._partial_result is None
        assert first_failure.code is None and terminal.reason == "closed"
        _assert_observed(metrics, "outer_timeout")
    finally:
        await metrics.aclose()


@pytest.mark.asyncio
async def test_native_result_read_failure_is_observed_and_still_propagates():
    metrics = RuntimeMetrics.in_memory()
    session = _session(metrics, None)
    session._writer.read_error = RuntimeError("private-error")
    try:
        with pytest.raises(RuntimeError, match="private-error"):
            await session._prepare_partial_result()
        assert session._result_inference_task is None
        _assert_observed(metrics, "error")
    finally:
        await metrics.aclose()


@pytest.mark.asyncio
async def test_native_pipecat_http_result_remains_valid_with_latched_metric_fault(monkeypatch):
    metrics = RuntimeMetrics.in_memory()
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1, "model": "test/llm",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(RESULT),
            }}],
        })

    http = DefaultAsyncHttpxClient(transport=httpx.MockTransport(handle), trust_env=False)
    monkeypatch.setattr(services, "DefaultAsyncHttpxClient", lambda **_kwargs: http)
    profile = InferenceProfileV1.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/inference-profile-v1.json").read_text()
    )
    llm = services.build_llm(profile, SecretStr("offline-key"))
    session = _session(metrics, None, timeout=10)
    session._services.llm = llm
    try:
        await session._prepare_partial_result()
        assert session._partial_result.summary == "Rappel demandé."
        _assert_observed(metrics, "valid")

        def broken(*_args, **_kwargs):
            raise asyncio.CancelledError("private-error")

        monkeypatch.setattr(metrics._result_outcomes, "add", broken)
        monkeypatch.setattr(metrics._result_duration, "record", broken)
        await session._prepare_partial_result()
        assert session._partial_result.summary == "Rappel demandé."
        assert metrics.failure_code == "metrics_record_failed"
        assert session._result_inference_task.done()
        assert len(requests) == 2
        assert all(body["stream"] is False and body["max_completion_tokens"] == 2048
                   for body in requests)
        assert all(body["model"] == "test/llm" for body in requests)
        assert all(body["provider"] == {"allow_fallbacks": True, "sort": "latency"}
                   for body in requests)
    finally:
        await llm._client.close()
        await metrics.aclose()
