"""Offline native LLM limit, actual call-owner teardown and retained result."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import UUID

import pytest
from openai import OpenAIError
from pipecat.frames.frames import EndFrame, ErrorFrame, LLMContextFrame, LLMFullResponseEndFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pydantic import SecretStr

from projetv0_voice.inference import services
from projetv0_voice.models import RoutingV1
from projetv0_voice.persistence.business_result import RetainedCall, RetainedTurn
from projetv0_voice.session import CallSessionError, ServiceBundle
from tests.integration.test_call_session import (
    NOW,
    _company_snapshot,
    _OfflineTts,
    _OpeningWriter,
    _PassProcessor,
    _profile,
    _session,
    _SttClient,
)


def _retained(turn: int = 101) -> RetainedCall:
    return RetainedCall((RetainedTurn(
        UUID(int=turn), 1, "user", "Pouvez-vous me rappeler ?", False,
    ),), 0)


class _ResultWriter(_OpeningWriter):
    """Existing offline owner fixture with typed retained-call data."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.ack_release.set()
        self.gate_release.set()
        self.retained = _retained()

    async def read_retained_call(self, _call_id: UUID) -> RetainedCall:
        return self.retained


@dataclass
class _Peer:
    base_url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)
    unexpected_ordinary: asyncio.Event = field(default_factory=asyncio.Event)
    final_entered: asyncio.Event = field(default_factory=asyncio.Event)
    final_release: asyncio.Event = field(default_factory=asyncio.Event)
    block_final: bool = False
    finish_reason: str = "stop"
    owner_events: list[str] | None = None
    owner_llm: Any = None
    final_owner_state: list[tuple[bool, bool]] = field(default_factory=list)

    @property
    def ordinary(self) -> list[dict[str, Any]]:
        return [body for body in self.requests if body["stream"]]

    @property
    def final(self) -> list[dict[str, Any]]:
        return [body for body in self.requests if not body["stream"]]


def _result_response(body: dict[str, Any], *, finish_reason: str = "stop") -> dict[str, Any]:
    submitted = json.loads(body["messages"][-1]["content"])
    user = next(turn for turn in submitted["turns"] if turn["role"] == "user")
    result = {
        "schema_version": 1, "quality": "partial", "category": "callback",
        "summary": "Rappel demandé.", "next_action": "Rappeler.",
        "contact": {
            "name": None, "callback_e164": None, "preference": None,
            "callback_source": "missing", "callback_confirmed": False,
        },
        "evidence": [{"turn_id": user["turn_id"], "role": "user"}],
        "request_confirmed": False,
    }
    return {
        "id": "offline-result", "object": "chat.completion", "created": 0,
        "model": body["model"], "choices": [{
            "index": 0, "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": json.dumps({"result": result})},
        }],
    }


@asynccontextmanager
async def _peer(*, block_final: bool = False, finish_reason: str = "stop") -> AsyncIterator[_Peer]:
    peer = _Peer(block_final=block_final, finish_reason=finish_reason)
    connections: set[asyncio.Task[Any]] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            assert headers.startswith(b"POST /v1/chat/completions HTTP/1.1\r\n")
            length = next(
                int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            body = json.loads(await reader.readexactly(length))
            peer.requests.append(body)
            if body["stream"]:
                if len(peer.ordinary) > 54:
                    peer.unexpected_ordinary.set()
                content_type = "text/event-stream"
                payload = b"data: [DONE]\n\n"
            else:
                if peer.owner_events is not None:
                    peer.final_owner_state.append((
                        "pipeline:tts" in peer.owner_events,
                        not peer.owner_llm._client.is_closed(),
                    ))
                peer.final_entered.set()
                if peer.block_final:
                    await peer.final_release.wait()
                content_type = "application/json"
                payload = json.dumps(
                    _result_response(body, finish_reason=peer.finish_reason)
                ).encode()
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # Cancellation can close an accepted offline request's socket.
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    assert server.sockets is not None
    peer.base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
    try:
        yield peer
    finally:
        peer.final_release.set()
        server.close()
        await server.wait_closed()
        if connections:
            await asyncio.wait_for(asyncio.gather(*connections), timeout=2)


def _llm():
    return services.build_llm(
        _profile().inference, SecretStr("owned-offline-key"), sparra_conversation=True,
    )


def _final_method(llm):
    method = getattr(llm, "infer_final_result_once", None)
    assert callable(method), "the actual final-result consumer seam is missing"
    return method


async def _ordinary(llm) -> None:
    stream = await llm.get_chat_completions(LLMContext([
        {"role": "user", "content": "Pouvez-vous me rappeler ?"},
    ]))
    await stream.close()


async def _await_disclosure(writer: _ResultWriter, running: asyncio.Task[Any]) -> None:
    disclosure = asyncio.create_task(writer.disclosure_committed.wait())
    try:
        done, _ = await asyncio.wait(
            (disclosure, running), timeout=3, return_when=asyncio.FIRST_COMPLETED,
        )
        if running in done:
            error = None if running.cancelled() else running.exception()
            detail = type(error).__name__
            if isinstance(error, CallSessionError):
                detail += ":" + str(error)
            pytest.fail(f"call owner exited before disclosure: {detail}")
        assert disclosure in done, "call owner never committed disclosure"
    finally:
        disclosure.cancel()
        await asyncio.gather(disclosure, return_exceptions=True)


def _assert_final_wire(peer: _Peer, turn: int = 101) -> None:
    assert len(peer.final) == 1
    body = peer.final[0]
    ceilings = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
    assert ceilings == [2048]
    assert body["stream"] is False
    schema = body["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["additionalProperties"] is False
    submitted = json.loads(body["messages"][-1]["content"])
    assert submitted["turns"][0]["turn_id"] == str(UUID(int=turn))


@pytest.mark.asyncio
@pytest.mark.parametrize("quota", [False, True], ids=["normal", "quota"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_actual_owner_cleans_then_uses_same_client_for_one_retained_result(
    quota: bool, finish_reason: str, monkeypatch,
) -> None:
    async with _peer(finish_reason=finish_reason) as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        llm = _llm()
        original_client = llm._client
        events: list[str] = []
        writer = _ResultWriter(events)
        stt_http = _SttClient()
        bundle = ServiceBundle(
            stt=_PassProcessor("stt", events), llm=llm, tts=_OfflineTts("tts", events),
            stt_http_client=stt_http, close_timeout_seconds=1,
        )
        session, _, lease, _, _ = _session(
            events=events, echo_ack=True, services_override=bundle, writer_override=writer,
            begin_snapshot=_company_snapshot(False),
        )
        session._identity = replace(session._identity, routing=RoutingV1(
            schema_version=1, direction="incoming", connection_id="offline",
            to_e164="+33123456789", from_e164=None, telnyx_call_control_id="call-control-1",
            telnyx_call_leg_id="call-leg-1", telnyx_call_session_id="call-session-1",
            admitted_at=NOW,
        ))
        call_data = session.handshake.call_data.model_dump(by_alias=True)
        call_data.update({"to": session._identity.routing.to_e164, "from": None})
        session.handshake = replace(
            session.handshake,
            call_data=type(session.handshake.call_data).model_validate(call_data),
        )
        # Use the fixture's existing, bounded cleanup deadline for TCP result work.
        session._cleanup_phase_timeout_seconds = 2
        peer.owner_events, peer.owner_llm = events, llm
        errors: list[ErrorFrame] = []
        response_done = asyncio.Event()

        def response_finished(_llm, frame) -> None:
            if isinstance(frame, LLMFullResponseEndFrame):
                response_done.set()

        llm.add_event_handler("on_after_push_frame", response_finished)
        running = None
        wire_wait = None
        try:
            if quota:
                for _ in range(54):
                    await _ordinary(llm)
                assert len(peer.ordinary) == 54
            running = asyncio.create_task(session.run(session.handshake))
            await _await_disclosure(writer, running)
            await session._controller.join_continuations()
            worker = session._active_runtime.worker

            def record_error(_worker, error: ErrorFrame) -> None:
                errors.append(error)

            worker.add_event_handler("on_pipeline_error", record_error)
            await worker.queue_frame(LLMContextFrame(LLMContext([
                {"role": "user", "content": "Pouvez-vous me rappeler ?"},
            ])))
            if quota:
                wire_wait = asyncio.create_task(peer.unexpected_ordinary.wait())
                done, _ = await asyncio.wait(
                    (running, wire_wait), timeout=3, return_when=asyncio.FIRST_COMPLETED,
                )
                assert wire_wait not in done, "the over-budget native request reached TCP"
                assert running in done, "the native quota error did not terminate its call owner"
                with pytest.raises(CallSessionError):
                    await running
                assert len(errors) == 1
                assert errors[0].fatal and errors[0].error == "llm_failed"
                assert errors[0].processor is None and errors[0].exception is None
                assert lease.calls[0][0] == "failed"
            else:
                await asyncio.wait_for(response_done.wait(), timeout=3)
                await worker.queue_frame(EndFrame())
                await asyncio.wait_for(running, timeout=3)
                assert errors == []
                assert lease.calls[0][0] == "closed"
            _assert_final_wire(peer)
            assert len(peer.ordinary) == (54 if quota else 1)
            assert peer.final_owner_state == [(True, True)]
            assert llm._client is original_client and original_client.is_closed()
            assert stt_http.calls == 1
            assert "recording-cleanup" in events and "lease-terminal" in events
            if finish_reason == "stop":
                assert session._partial_result is not None
                assert session._partial_result.evidence[0].turn_id == UUID(int=101)
                assert session._partial_result.contact.callback_source == "missing"
            else:
                assert session._partial_result is None
            with pytest.raises(OpenAIError, match="llm_output_limit"):
                await _final_method(llm)(writer.retained, None)
            assert len(peer.final) == 1
        finally:
            peer.final_release.set()
            if wire_wait is not None:
                wire_wait.cancel()
                await asyncio.gather(wire_wait, return_exceptions=True)
            if running is not None:
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            await bundle.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["concurrent", "uncertain"])
async def test_real_final_consumer_refuses_second_attempt_while_first_post_is_unknown(
    cancel: bool, monkeypatch,
) -> None:
    async with _peer(block_final=True) as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        llm = _llm()
        first = None
        try:
            await llm.cleanup()
            method = _final_method(llm)
            first = asyncio.create_task(method(_retained(), None))
            await asyncio.wait_for(peer.final_entered.wait(), timeout=3)
            if cancel:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            with pytest.raises(OpenAIError, match="llm_output_limit"):
                await method(_retained(), None)
            assert len(peer.final) == 1
            peer.final_release.set()
            if not cancel:
                result = await asyncio.wait_for(first, timeout=3)
                assert result is not None and result.evidence[0].turn_id == UUID(int=101)
            _assert_final_wire(peer)
        finally:
            peer.final_release.set()
            if first is not None:
                if not first.done():
                    first.cancel()
                await asyncio.gather(first, return_exceptions=True)
            await llm._client.close()


@pytest.mark.asyncio
async def test_second_call_final_reserve_is_independent_and_first_client_closure_does_not_leak(
    monkeypatch,
) -> None:
    async with _peer() as peer:
        monkeypatch.setattr(services, "_OPENROUTER_BASE_URL", peer.base_url)
        first, second = _llm(), _llm()
        try:
            await first.cleanup()
            await second.cleanup()
            one = await _final_method(first)(_retained(101), None)
            assert one is not None and one.evidence[0].turn_id == UUID(int=101)
            await first._client.close()
            assert not second._client.is_closed()
            two = await _final_method(second)(_retained(102), None)
            assert two is not None and two.evidence[0].turn_id == UUID(int=102)
            assert len(peer.final) == 2 and peer.ordinary == []
        finally:
            await first._client.close()
            await second._client.close()
