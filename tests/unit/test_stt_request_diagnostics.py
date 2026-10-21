from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx
import pytest
from openai import DefaultAsyncHttpxClient
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    FatalErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import QueuedFrameProcessor, run_test
from pydantic import SecretStr

from projetv0_voice.inference.stt_diagnostics import (
    STT_DIAGNOSTIC_KEY,
    diagnostic_attributes,
    snapshot_stt_request,
)
from projetv0_voice.pipeline import InferenceErrorBoundary
from projetv0_voice.production_wiring import _inference_factories
from tests.integration.test_process_session_factory import (
    _claim,
    _factory,
    _grant,
    _handshake,
    _metric_points,
    _real_profile,
    _Registrar,
    _Registry,
)
from tests.unit.test_inference_deadlines import _start_native_stt_worker
from tests.unit.test_stt_consumption_limits import _wav


async def _native_failure(monkeypatch, paid, handler, *, exercise=None, segments=1):
    """Real factory/guard/SDK/adapter/observer; only the call owner is a test shell."""
    from projetv0_voice.metrics import RuntimeMetrics

    original_init = DefaultAsyncHttpxClient.__init__

    def offline_init(self, **kwargs):
        original_init(self, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(DefaultAsyncHttpxClient, "__init__", offline_init)
    native = _inference_factories(
        SecretStr("offline-unit-key"), _real_profile(), "fr-FR", sparra_conversation=True
    )
    claim = _claim()
    owner = RuntimeMetrics.in_memory()
    factory = _factory(
        registry=_Registry(replace(_grant(claim), paid_work_permitted=paid)),
        registrar=_Registrar(), runtime_metrics=owner, events=[],
    )
    factory._stt_http_client_factory = native.stt_http_client_factory
    factory._stt_factory = native.stt_factory
    failures = []
    raw = []
    errors = []
    captured_services = []

    class Capture(BaseObserver):
        async def on_push_frame(self, data):
            if (data.source is captured_services[0].stt
                    and isinstance(data.frame, FatalErrorFrame) and data.first_push):
                raw.append(data.frame)

    class NativeSession:
        def __init__(self, **kwargs):
            self.values = kwargs

        async def run(self, _handshake):
            services = self.values["services"]
            captured_services.append(services)
            try:
                # Resolve the installed SDK's lazy audio resources before starting
                # the worker watchdog; this performs no request or paid check.
                assert services.stt._client.audio.transcriptions is not None
                observer = self.values["observers"]._metrics
                boundary = InferenceErrorBoundary(
                    stt=services.stt, llm=services.llm, tts=services.tts
                )
                if exercise is None:
                    observed = asyncio.Queue()
                    collector = QueuedFrameProcessor(
                        queue=observed, queue_direction=FrameDirection.UPSTREAM
                    )
                    sink = QueuedFrameProcessor(
                        queue=asyncio.Queue(), queue_direction=FrameDirection.DOWNSTREAM
                    )
                    worker, task = await _start_native_stt_worker(
                        collector, boundary, services.stt, sink, observers=[observer, Capture()]
                    )
                    try:
                        await worker.queue_frames([
                            VADUserStartedSpeakingFrame(),
                            InputAudioRawFrame(b"\x01\x00" * 80, 8000, 1),
                            VADUserStoppedSpeakingFrame(),
                        ] * segments)
                        await asyncio.wait_for(task, timeout=5)
                        received = [observed.get_nowait() for _ in range(observed.qsize())]
                        failures.extend(
                            frame for frame in received if isinstance(frame, ErrorFrame)
                        )
                    finally:
                        if not task.done():
                            await worker.queue_frame(CancelFrame())
                        await asyncio.wait_for(task, timeout=5)
                else:
                    frames = await exercise(services.stt)
                    raw.extend(frame for frame in frames if isinstance(frame, FatalErrorFrame))
                    for frame in raw:
                        data = FramePushed(
                            services.stt, services.llm, frame, FrameDirection.UPSTREAM, 1
                        )
                        await observer.on_push_frame(data)
                        await observer.on_push_frame(data)
                        _, upstream = await run_test(
                            boundary, frames_to_send=[frame],
                            frames_to_send_direction=FrameDirection.UPSTREAM,
                            expected_up_frames=[ErrorFrame],
                        )
                        failures.extend(upstream)
            except BaseException as error:
                errors.append(error)
            finally:
                await services.aclose()
                await services.stt.cleanup()
                await services.llm.cleanup()
                await services.tts.cleanup()

    factory._session_factory = NativeSession
    try:
        await factory.run(_handshake(claim))
        if errors:
            raise errors[0]
        assert len(captured_services) == 1
        assert captured_services[0].stt_http_client.is_closed
        return raw, failures, _metric_points(owner, "projetv0.voice.stt.failures")
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_stt_guard_denial_preserves_local_cause_before_transport(monkeypatch):
    checks = 0
    dispatches = 0

    def paid():
        nonlocal checks
        checks += 1
        return False

    async def handler(request):
        nonlocal dispatches
        dispatches += 1
        return httpx.Response(200, json={"text": ""})

    raw, failures, points = await _native_failure(monkeypatch, paid, handler)
    assert checks == 1 and dispatches == 0
    _assert_failure(raw, failures, points, "denied", "none", "unobserved")


def _assert_failure(raw, failures, points, paid, response, failure):
    expected = {"paid_check": paid, "response_family": response, "httpx_failure": failure}
    assert len(raw) == len(failures) == len(points) == 1
    assert failures[0].error == "stt_failed" and failures[0].fatal
    assert failures[0].metadata == {} and failures[0].exception is None
    assert raw[0].error == "openrouter_stt_transport"
    diagnostic = raw[0].metadata[STT_DIAGNOSTIC_KEY]
    assert diagnostic_attributes(diagnostic) == expected
    assert dict(points[0].attributes) == {"reason": "transport", **expected}
    assert points[0].value == 1
    assert snapshot_stt_request() is None


@pytest.mark.asyncio
async def test_actual_guard_check_exception_is_distinct_from_denial(monkeypatch):
    checks = 0

    def paid():
        nonlocal checks
        checks += 1
        raise RuntimeError("canary-private-guard")

    async def forbidden(_request):
        pytest.fail("guard exception reached transport")

    result = await _native_failure(monkeypatch, paid, forbidden)
    assert checks == 1
    _assert_failure(*result, "check_error", "none", "unobserved")
    assert "canary" not in repr(result)


class _BrokenBody(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'{"text":"canary-private-partial'
        raise httpx.ReadTimeout("canary-private-body")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("case,family,failure", [
    ("400", "4xx", "unobserved"), ("503", "5xx", "unobserved"),
    ("timeout", "none", "timeout"), ("connection", "none", "transport_error"),
    ("request", "none", "other_request_error"),
    ("body", "2xx", "timeout"), ("malformed", "2xx", "unobserved"),
])
async def test_actual_sdk_failure_keeps_only_observed_facts(monkeypatch, case, family, failure):
    dispatches = 0
    checks = 0
    stream = _BrokenBody()
    send_stream = []
    from projetv0_voice.production_wiring import _SttHttpClient
    original_send = _SttHttpClient.send

    async def observe_send(self, *args, **kwargs):
        send_stream.append(kwargs.get("stream"))
        return await original_send(self, *args, **kwargs)

    monkeypatch.setattr(_SttHttpClient, "send", observe_send)

    def paid():
        nonlocal checks
        checks += 1
        return True

    async def handler(request):
        nonlocal dispatches
        dispatches += 1
        if case in {"400", "503"}:
            return httpx.Response(int(case), json={"error": {"message": "canary-provider"}})
        if case == "timeout":
            raise httpx.ReadTimeout("canary-private-timeout", request=request)
        if case == "connection":
            raise httpx.ConnectError("canary-private-connection", request=request)
        if case == "request":
            raise httpx.TooManyRedirects("canary-private-request", request=request)
        if case == "body":
            return httpx.Response(200, stream=stream, headers={"content-type": "application/json"})
        return httpx.Response(200, content=b"canary-malformed", headers={
            "content-type": "application/json"
        })

    result = await _native_failure(monkeypatch, paid, handler)
    assert dispatches == checks == 1  # Actual installed SDK never retries the upload.
    assert send_stream == [False]  # Native SDK uses buffered public send, including body reads.
    _assert_failure(*result, "allowed", family, failure)
    assert "canary" not in repr(result)
    if case == "body":
        assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("second", ["denied", "timeout"])
async def test_successful_segment_cannot_supply_later_failure_response(monkeypatch, second):
    checks = 0
    requests = 0

    def paid():
        nonlocal checks
        checks += 1
        return not (checks == 2 and second == "denied")

    async def handler(request):
        nonlocal requests
        requests += 1
        if requests == 2:
            raise httpx.ReadTimeout("canary-second-request", request=request)
        return httpx.Response(200, json={"text": ""})

    async def exercise(stt):
        first = [frame async for frame in stt.run_stt(_wav(80))]
        assert len(first) == 1 and isinstance(first[0], TranscriptionFrame)
        assert snapshot_stt_request() is None
        return [frame async for frame in stt.run_stt(_wav(80))]

    result = await _native_failure(monkeypatch, paid, handler, exercise=exercise)
    assert checks == 2 and requests == (1 if second == "denied" else 2)
    _assert_failure(*result, "denied" if second == "denied" else "allowed", "none",
                    "unobserved" if second == "denied" else "timeout")


@pytest.mark.asyncio
async def test_native_queued_segment_starts_fresh_and_discards_completed_batch(monkeypatch):
    checks = 0
    requests = 0

    def paid():
        nonlocal checks
        checks += 1
        return checks == 1

    async def handler(_request):
        nonlocal requests
        requests += 1
        await asyncio.sleep(0)  # Let native FIFO accept the later segment first.
        return httpx.Response(200, json={"text": ""})

    result = await _native_failure(monkeypatch, paid, handler, segments=3)
    assert checks == 2 and requests == 1  # Queued segment 3 never starts after denial.
    _assert_failure(*result, "denied", "none", "unobserved")


@pytest.mark.asyncio
async def test_concurrent_native_invocations_keep_own_record_and_discard_late_text(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    checks = 0

    def paid():
        nonlocal checks
        checks += 1
        return checks == 1

    async def handler(_request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"text": ""})

    async def exercise(stt):
        async def collect():
            try:
                return [frame async for frame in stt.run_stt(_wav(80))]
            finally:
                assert snapshot_stt_request() is None

        first = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            refused = await collect()
            release.set()
            assert await asyncio.wait_for(first, timeout=2) == []
            assert stt._batch_text == "" and stt._last_transcript is None
            assert [frame async for frame in stt.run_stt(_wav(80))] == []
            return refused
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)

    result = await _native_failure(monkeypatch, paid, handler, exercise=exercise)
    assert checks == 2
    _assert_failure(*result, "denied", "none", "unobserved")


@pytest.mark.asyncio
async def test_native_cancellation_and_generator_close_clear_binding(monkeypatch):
    entered = asyncio.Event()
    never = asyncio.Event()
    mode = "blocked"

    async def handler(request):
        if mode == "blocked":
            entered.set()
            await never.wait()
        raise httpx.ConnectError("canary-close", request=request)

    async def exercise(stt):
        nonlocal mode

        async def collect():
            try:
                return [frame async for frame in stt.run_stt(_wav(80))]
            finally:
                assert snapshot_stt_request() is None

        pending = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        assert snapshot_stt_request() is None
        mode = "failed"
        generator = stt.run_stt(_wav(80))
        try:
            frame = await anext(generator)
            assert isinstance(frame, FatalErrorFrame)
            frozen = frame.metadata[STT_DIAGNOSTIC_KEY]
            assert snapshot_stt_request() is frozen
        finally:
            await generator.aclose()
        assert snapshot_stt_request() is None
        assert diagnostic_attributes(frozen) == {
            "paid_check": "allowed", "response_family": "none", "httpx_failure": "transport_error"
        }
        return [frame]

    result = await _native_failure(monkeypatch, lambda: True, handler, exercise=exercise)
    _assert_failure(*result, "allowed", "none", "transport_error")


@pytest.mark.asyncio
async def test_real_session_stt_failure_cannot_revive_calendar_scope_after_budget_recovery(
    monkeypatch, tmp_path
):
    from decimal import Decimal

    from projetv0_voice.action_authority import ActionAuthority
    from projetv0_voice.session import CallSessionError
    from tests.integration.test_call_session import _native_sparra_opening
    from tests.integration.test_owned_pilot_budget_admission import (
        attach,
        committed,
        registry_case,
        saved_permit,
    )
    from tests.integration.test_voice_action_authority import active, caller
    from tests.test_calendar_tools import CalendarPeer
    from tests.unit.test_owned_pilot_budget import NOW, observation
    from tests.unit.test_sparra_admission import event

    scopes = []
    capture = ActionAuthority.capture

    def capture_actual(self, *args, **kwargs):
        scope = capture(self, *args, **kwargs)
        if scope is not None:
            scopes.append(scope)
        return scope

    monkeypatch.setattr(ActionAuthority, "capture", capture_actual)
    async with registry_case(tmp_path) as budget:
        gate = await attach(budget)
        await committed(budget.registry, budget.writer, event(occurred_at=NOW))
        permit = saved_permit(budget.path)
        assert gate.permits(permit)
        sink = CalendarPeer()
        async with _native_sparra_opening(calendar_sink=sink) as fixture:
            await active(fixture)
            await caller(fixture, "Question synthetique.")
            assert scopes and fixture.session._action_authority.current(scopes[-1])
            old_scope = scopes[-1]
            stt = fixture.session._services.stt
            runtime = fixture.session._active_runtime
            await budget.writer.owned_budget_request("observe", observation=observation(
                openrouter_available_cash_usd=Decimal("0")
            ))
            assert not gate.permits(permit)
            requests = 0

            async def refused(_request):
                nonlocal requests
                requests += 1
                return httpx.Response(400, json={"error": {"message": "canary-cutoff"}})

            # Existing fixture's MockTransport remains the sole provider boundary.
            stt._client._client._transport.handler = refused
            await runtime.worker.queue_frames([
                VADUserStartedSpeakingFrame(), InputAudioRawFrame(b"\x01\x00" * 80, 8000, 1),
                VADUserStoppedSpeakingFrame(),
            ])
            with pytest.raises(CallSessionError, match="^stt_failed$"):
                await asyncio.wait_for(fixture.running, timeout=8)
            assert requests == 1 and fixture.session._terminal_outcome.reason == "stt_failed"
            assert not fixture.session._action_authority.current(old_scope)
            assert not fixture.session._action_authority.claim(old_scope)
            await gate.refresh()
            assert gate.permits(permit)
            # A genuine healthy publication restores funding, never the old call epoch.
            assert not fixture.session._action_authority.current(old_scope)
            assert not fixture.session._action_authority.claim(old_scope)
            assert [frame async for frame in stt.run_stt(_wav(80))] == []
            assert requests == 1
            assert fixture.session._calendar_pending_booking is None
            assert not sink.booked.is_set() and sink.submitted == []


def test_diagnostic_import_is_stdlib_only():
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import projetv0_voice.inference.stt_diagnostics; "
        "assert not {'openai', 'httpx', 'pipecat'} & sys.modules.keys()",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_snapshot_is_frozen_and_status_classification_reads_only_integer():
    from dataclasses import FrozenInstanceError, fields

    from projetv0_voice.inference.stt_diagnostics import (
        SttRequestDiagnostic,
        observe_response_status,
        stt_request_scope,
    )

    assert [field.name for field in fields(SttRequestDiagnostic)] == [
        "paid_check", "response_family", "httpx_failure"
    ]
    with stt_request_scope():
        for status, family in [(200, "2xx"), (302, "3xx"), (400, "4xx"),
                               (503, "5xx"), (199, "other"), (600, "other")]:
            observe_response_status(status)
            snapshot = snapshot_stt_request()
            assert snapshot.response_family == family
        with pytest.raises(FrozenInstanceError):
            snapshot.response_family = "canary"
    assert snapshot_stt_request() is None
