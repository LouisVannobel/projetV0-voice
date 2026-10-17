"""Synthetic loopback transport checks; no provider execution is exercised."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from uuid import UUID

import pytest
import telnyx

from projetv0_voice.telnyx import call_control

COMMAND = UUID("11111111-1111-4111-8111-111111111111")
CALL = "synthetic-loopback-call"
TOKEN = "synthetic-loopback-token"
STREAM = call_control.StreamingStartV1(
    stream_url="wss://voice.example.test/telnyx/media", stream_auth_token=TOKEN,
)


@asynccontextmanager
async def streaming_loopback(monkeypatch, modes):
    posts, sdk_events, errors = [], [], []
    tasks, writers = set(), set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        writers.add(writer)
        try:
            async with asyncio.timeout(8):
                headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
                path = f"/v2/calls/{CALL}/actions/streaming_start"
                assert headers[0] == f"POST {path} HTTP/1.1"
                fields = dict(line.lower().split(":", 1) for line in headers[1:] if ":" in line)
                size = int(fields["content-length"])
                assert 0 < size < 16384
                body = await reader.readexactly(size)
                posts.append(body)
                mode = modes[min(len(posts) - 1, len(modes) - 1)]
                if mode == "absent":
                    await reader.read()
                    return
                if mode == "delayed":
                    await asyncio.sleep(0.8)
                status = b"400 Bad Request" if mode == "rejected" else b"200 OK"
                response = b'{"errors":[]}' if mode == "rejected" else b'{"data":{"result":"ok"}}'
                writer.write(
                    b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(response)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                )
                if mode == "trickle":
                    for byte in response:
                        writer.write(bytes([byte]))
                        await writer.drain()
                        await asyncio.sleep(0.25)
                else:
                    writer.write(response)
                    await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as error:
            errors.append(error)
        finally:
            try:
                writer.close()
                with suppress(ConnectionError, TimeoutError):
                    async with asyncio.timeout(0.5):
                        await writer.wait_closed()
            finally:
                writers.discard(writer)
                tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    real_sdk = telnyx.AsyncTelnyx
    real_http = telnyx.DefaultAsyncHttpxClient

    class LoopbackHTTP(real_http):
        async def send(self, request, **kwargs):
            assert (request.url.scheme, request.url.host, request.url.port) == (
                "http", "127.0.0.1", port,
            )
            return await super().send(request, **kwargs)

    class ObservedSDK(real_sdk):
        async def request(self, *args, **kwargs):
            number = sum(event == "started" for _, event in sdk_events) + 1
            sdk_events.append((number, "started"))
            try:
                response = await super().request(*args, **kwargs)
            except BaseException as error:
                sdk_events.append((number, type(error).__name__))
                raise
            sdk_events.append((number, response.data.result))
            return response

    def local_sdk(**kwargs):
        assert kwargs["max_retries"] == 0
        kwargs["base_url"] = f"http://127.0.0.1:{port}/v2"
        return ObservedSDK(**kwargs)

    monkeypatch.setattr(telnyx, "AsyncTelnyx", local_sdk)
    monkeypatch.setattr(telnyx, "DefaultAsyncHttpxClient", LoopbackHTTP)
    client = None
    try:
        client = call_control.CallControlClient(api_key="synthetic-loopback-key")
        yield SimpleNamespace(client=client, posts=posts, sdk_events=sdk_events)
    finally:
        try:
            if client is not None:
                await client.aclose()
        finally:
            server.close()
            await server.wait_closed()
            for writer in tuple(writers):
                writer.close()
            pending = tuple(tasks)
            for task in pending:
                task.cancel()
            async with asyncio.timeout(1):
                await asyncio.gather(*pending, return_exceptions=True)
        assert not tasks
        assert not writers
        assert errors == []


@pytest.mark.asyncio
async def test_streaming_receives_first_sdk_response_after_800ms_without_duplicate(monkeypatch):
    async with streaming_loopback(monkeypatch, ("delayed", "accepted")) as local:
        result = await local.client.start_streaming(CALL, STREAM, command_id=COMMAND)

        assert len(local.posts) == 1
        assert local.sdk_events == [(1, "started"), (1, "ok")]
        assert result.outcome == "accepted"
        assert [attempt.category for attempt in result.attempt_observations] == ["accepted"]
        body = json.loads(local.posts[0])
        assert body["command_id"] == str(COMMAND)
        assert body["stream_auth_token"] == TOKEN


@pytest.mark.asyncio
@pytest.mark.parametrize(("second", "outcome"), [("accepted", "accepted"),
                                               ("rejected", "outcome_unknown")])
async def test_streaming_timeout_finishes_before_identical_retry(monkeypatch, second, outcome):
    async with streaming_loopback(monkeypatch, ("absent", second)) as local:
        result = await local.client.start_streaming(CALL, STREAM, command_id=COMMAND)

        assert result.outcome == outcome
        assert len(local.posts) == 2
        assert local.posts[0] == local.posts[1]
        assert local.sdk_events == [
            (1, "started"), (1, "CancelledError"), (2, "started"),
            (2, "ok" if second == "accepted" else "BadRequestError"),
        ]
        assert [attempt.category for attempt in result.attempt_observations] == [
            "deadline", "accepted" if second == "accepted" else "status",
        ]
        assert 1.8 <= result.attempt_observations[0].duration_seconds < 2.5
        body = json.loads(local.posts[0])
        assert body["command_id"] == str(COMMAND)
        assert body["stream_auth_token"] == TOKEN


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["absent", "trickle"])
async def test_streaming_outer_deadline_bounds_two_silent_or_trickling_responses(monkeypatch, mode):
    async with streaming_loopback(monkeypatch, (mode, mode)) as local:
        started = time.monotonic()
        result = await local.client.start_streaming(CALL, STREAM, command_id=COMMAND)
        elapsed = time.monotonic() - started

        assert result.outcome == "outcome_unknown"
        assert 3.8 <= elapsed < 4.6  # Four-second nominal HTTP budget plus scheduling tolerance.
        assert len(local.posts) == 2
        assert local.posts[0] == local.posts[1]
        assert local.sdk_events == [
            (1, "started"), (1, "CancelledError"), (2, "started"), (2, "CancelledError"),
        ]
        assert [attempt.category for attempt in result.attempt_observations] == [
            "deadline", "deadline",
        ]
