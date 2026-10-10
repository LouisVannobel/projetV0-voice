"""Fixture pacing through native PCM/event/SQLite seams; external ACK is separate."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace

import pytest
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from tests.integration.sparra_connected_scenario import Scenario, eventually
from tests.integration.test_local_audio_capture import CALL, accept_local, capture_case


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["native-event", "local-receipt"])
async def test_opposition_fixture_waits_before_next_wire_pcm(tmp_path, monkeypatch, boundary):
    entered, release, overlapped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    resumed = asyncio.Event()
    joined_before_next_wire = []
    case = None
    native_register = AudioBufferProcessor.add_event_handler

    def observe(recorder, name, handler):
        async def held(sender, pcm, rate, channels):
            if boundary == "native-event" and not entered.is_set():
                entered.set()
                await release.wait()
            return await handler(sender, pcm, rate, channels)

        return native_register(recorder, name, held if name == "on_audio_data" else handler)

    async def hold_receipt(name):
        if (
            boundary == "local-receipt" and name == "after_mutation_before_commit"
            and case is not None and case.capture.holder.summary.pending
            and not entered.is_set()
        ):
            entered.set()
            await release.wait()

    class GeneratedWirePeer:
        stream = "capture-stream"

        async def input(self, message):
            # The same native decoder and pipeline receive every generated wire frame.
            if entered.is_set():
                if not release.is_set():
                    overlapped.set()
                else:
                    joined_before_next_wire.append(
                        not case.capture.tap.pending_join
                        and not case.capture.holder.summary.pending
                    )
                    resumed.set()
            frame = await case.serializer.deserialize(json.dumps(message))
            await case.runtime.worker.queue_frame(frame)

    class LocalPrimeScenario:
        # Only admission and the remote active-delivery setup are outside this seam.
        # No external audio ACK is supplied or fabricated by this test.
        async def audio_admit(self):
            await accept_local(case)
            await case.serializer.setup(SimpleNamespace(audio_in_sample_rate=8000))
            return {}

        async def _audio_wait_active_delivery(self):
            assert case.controller.is_active()

    monkeypatch.setattr(AudioBufferProcessor, "add_event_handler", observe)
    async with capture_case(tmp_path, failpoint=hold_receipt) as active_case:
        case = active_case
        probe = LocalPrimeScenario()
        probe.graph = SimpleNamespace(writer=case.writer)
        probe.settings = SimpleNamespace(sqlite_path=case.path)
        probe.call_id = CALL
        probe.audio_capture = case.capture
        probe.audio_seen = set()
        probe.media = GeneratedWirePeer()
        prime = asyncio.create_task(Scenario.audio_opposition_prime(probe))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if boundary == "native-event":
                assert case.capture.tap.pending_join
            else:
                assert case.capture.holder.summary.pending
                assert case.capture.holder.summary.committed_samples == 0
            # This bounded negative observation controls a native completion gate,
            # not production timing. BASE sends more PCM while the gate is closed.
            try:
                await asyncio.wait_for(overlapped.wait(), 0.1)
            except TimeoutError:
                pass
            else:
                await eventually(
                    lambda: case.capture.tap.state == "partial",
                    "controlled_overlap_refuses_actual_capture", 5,
                )
                pytest.fail(f"next wire PCM overlapped held {boundary}; native tap refused partial")
            assert not prime.done() and case.capture.tap.state == "recording"
            release.set()
            await eventually(
                lambda: case.capture.holder.summary.committed_samples >= 8000,
                "controlled_native_samples_committed", 5,
            )
            await asyncio.wait_for(resumed.wait(), 5)
            assert joined_before_next_wire and all(joined_before_next_wire)
            assert case.capture.tap.state == "recording"
            assert not case.capture.holder.summary.partial
        finally:
            release.set()
            prime.cancel()
            with suppress(asyncio.CancelledError):
                await prime
        summary = await case.capture.quiesce()
        assert not summary.pending and not case.capture.tap.pending_join
        assert summary.committed_samples == summary.submitted_samples
        assert not summary.partial and case.controller.is_active()
