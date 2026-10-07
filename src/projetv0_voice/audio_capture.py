"""Call-owned optional capture over the pinned native AudioBufferProcessor."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.base_object import BaseObject

from projetv0_voice.audio_contract import (
    AUDIO_CHUNK_AAD_DOMAIN,
    MAX_AUDIO_AAD_BYTES,
    AudioChunkPayloadV2,
    AudioFinishPayloadV2,
    BeginCallSnapshotV2,
    VoiceOperationV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.crypto import CryptoKeyring
from projetv0_voice.models import validate_deployment_id
from projetv0_voice.persistence.commands import PersistenceError
from projetv0_voice.persistence.writer import PersistenceWriter

CaptureState = Literal["off", "recording", "partial", "stopped"]
AudioFinishReason = Literal["complete", "limit", "failure", "transfer", "interrupted"]
_MONO_BYTES_PER_SECOND = 16000
_MAX_MONO_BYTES = 16000
_NATIVE_THRESHOLD_BYTES = 14400
_MAX_FRAME_BYTES = 1600
_MAX_SAMPLES = 4_800_000


@dataclass(frozen=True, slots=True)
class AudioCaptureSummary:
    submitted_samples: int
    committed_samples: int
    last_sequence: int | None
    reason: Literal["complete", "limit", "failure"]
    partial: bool
    pending: bool


class AudioChunkHolder:
    """Coalesce one bounded native event into the existing writer's audio slot."""

    def __init__(
        self, *, snapshot: BeginCallSnapshotV2, deployment_id: str,
        keyring: CryptoKeyring, writer: PersistenceWriter,
    ) -> None:
        if not isinstance(snapshot, BeginCallSnapshotV2):
            raise ValueError("audio_snapshot_unavailable")
        self._snapshot = snapshot
        self._deployment_id = validate_deployment_id(deployment_id)
        self._keyring, self._writer = keyring, writer
        self._carry = bytearray()
        self._accepted_samples = self._submitted_samples = self._committed_samples = 0
        self._sequence = 0
        self._committed_last_sequence: int | None = None
        self._pending: VoiceOperationV2 | None = None
        self._closed = not snapshot.audio_available or snapshot.recording_policy != "local_30d"
        self._partial = self._closed
        self._reason: Literal["complete", "limit", "failure"] = (
            "failure" if self._closed else "complete"
        )
        self._finished = False

    def ready_for_native_event(self) -> bool:
        return not self._closed and self._pending is None

    def notify_capture_refused(self) -> None:
        # The tap has closed frame admission. Its already admitted native tail
        # may still settle, so this status notification does not close the holder.
        self._partial = True
        if self._reason != "limit":
            self._reason = "failure"

    def _refuse(self) -> bool:
        self._closed = self._partial = True
        if self._reason != "limit":
            self._reason = "failure"
        return False

    def _offer(self, pcm: bytes) -> bool:
        if self._pending is not None or self._sequence >= 600:
            return self._refuse()
        sample_count = len(pcm) // 4
        metadata = {
            "schema_version": 2, "workspace_id": str(self._snapshot.workspace_id),
            "deployment_id": self._deployment_id, "call_id": str(self._snapshot.call_id),
            "recording_id": str(self._snapshot.recording_id), "sequence": self._sequence,
            "sample_count": sample_count, "sample_rate": 8000, "channels": 2,
            "sample_format": "s16le",
            "configuration_revision": self._snapshot.configuration_revision,
            "retention_until": self._snapshot.retention_until.isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "crypto_version": 1, "key_version": self._keyring.active_version,
        }
        aad = AUDIO_CHUNK_AAD_DOMAIN + json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(aad) > MAX_AUDIO_AAD_BYTES:
            return self._refuse()
        encrypted = self._keyring.encrypt(pcm, aad=aad)
        payload = {key: value for key, value in metadata.items()
                   if key not in {"deployment_id", "call_id"}}
        payload.update(
            nonce_b64=base64.b64encode(encrypted.nonce).decode("ascii"),
            ciphertext_b64=base64.b64encode(encrypted.ciphertext).decode("ascii"),
        )
        operation = VoiceOperationV2.model_validate({
            "schema_version": 2, "operation_id": str(uuid4()),
            "deployment_id": self._deployment_id, "call_id": str(self._snapshot.call_id),
            "occurred_at": datetime.now(UTC), "kind": "audio.chunk", "payload": payload,
        })
        if canonical_audio_chunk_aad(operation) != aad:
            return self._refuse()
        if self._writer.offer_audio_chunk(operation) is not True:
            return self._refuse()
        self._pending = operation
        self._submitted_samples += sample_count
        self._sequence += 1
        return True

    def offer_native_pcm(self, pcm: bytes, rate: int, channels: int) -> bool:
        try:
            if (
                self._closed or self._pending is not None or type(pcm) is not bytes
                or rate != 8000 or channels != 2 or not pcm or len(pcm) % 4
                or len(pcm) > 32000 or datetime.now(UTC) >= self._snapshot.retention_until
            ):
                return self._refuse()
            accepted = min(len(pcm), (_MAX_SAMPLES - self._accepted_samples) * 4)
            if accepted <= 0:
                self._reason = "limit"
                return self._refuse()
            view = memoryview(pcm)[:accepted]
            missing = 32000 - len(self._carry)
            self._accepted_samples += accepted // 4
            if len(view) < missing:
                self._carry.extend(view)
            else:
                self._carry.extend(view[:missing])
                chunk = bytes(self._carry)
                self._carry.clear()
                if not self._offer(chunk):
                    return False
                self._carry.extend(view[missing:])
            if self._accepted_samples == _MAX_SAMPLES:
                self._reason = "limit"
                self._closed = self._partial = True
            return True
        except Exception:
            return self._refuse()

    async def after_event_join(self) -> bool:
        operation = self._pending
        if operation is None:
            return True
        try:
            await self._writer.wait_for_audio_commit(operation.operation_id)
        except asyncio.CancelledError:
            self._refuse()
            raise
        except PersistenceError as error:
            if str(error) == "audio_chunk_refused" and self._pending is operation:
                # The actual owner confirmed a fenced rollback, not an unknown
                # commit. No further unsaved tail can pass that terminal fence.
                self._pending = None
                self._carry.clear()
            return self._refuse()
        except Exception:
            return self._refuse()
        if self._pending is operation:
            payload = operation.payload
            if not isinstance(payload, AudioChunkPayloadV2):
                return self._refuse()
            self._committed_samples += payload.sample_count
            self._committed_last_sequence = payload.sequence
            self._pending = None
        return True

    def _summary(self) -> AudioCaptureSummary:
        return AudioCaptureSummary(
            self._submitted_samples, self._committed_samples,
            self._committed_last_sequence,
            self._reason, self._partial, self._pending is not None,
        )

    @property
    def summary(self) -> AudioCaptureSummary:
        return self._summary()

    async def finish(self) -> AudioCaptureSummary:
        self._closed = True
        if self._finished or not await self.after_event_join():
            return self._summary()
        if self._carry:
            tail = bytes(self._carry)
            self._carry.clear()
            try:
                if not self._offer(tail) or not await self.after_event_join():
                    return self._summary()
            except Exception:
                self._refuse()
                return self._summary()
        self._finished = True
        return self._summary()


class BoundedAudioBufferTap(FrameProcessor):
    """Forward every frame and refuse optional capture before native expansion.

    Position/time counters describe only admitted frames and public native events.
    They never read or alter the recorder's private buffers or event/task fields.
    """

    def __init__(
        self,
        *,
        offer_chunk: Callable[[bytes, int, int], bool],
        monotonic: Callable[[], float] = time.monotonic,
        on_event_join: Callable[[], Awaitable[bool]] | None = None,
        ready_for_native_event: Callable[[], bool] | None = None,
        on_capture_refused: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(enable_direct_mode=True)
        self._offer_chunk = offer_chunk
        self._on_event_join = on_event_join
        self._ready_for_native_event = ready_for_native_event
        self._on_capture_refused = on_capture_refused
        self._monotonic = monotonic
        self._recorder = AudioBufferProcessor(
            sample_rate=8000, num_channels=2, buffer_size=_NATIVE_THRESHOLD_BYTES,
            enable_turn_audio=False, auto_start_recording=False, enable_direct_mode=True,
        )
        self._recorder.add_event_handler(
            "on_audio_data", self._receive_audio
        )
        self._state: CaptureState = "off"
        self._started = False
        self._closed = False
        self._stop_called = False
        self._pending_join = False
        self._permit = False
        self._completion: asyncio.Task[bool] | None = None
        self._user_position = self._bot_position = 0
        self._last_user: float | None = None
        self._last_bot: float | None = None
        self._user_speaking = self._bot_speaking = False
        self._total_samples = 0

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def pending_join(self) -> bool:
        return self._pending_join

    # Pipecat 1.7 FrameProcessor narrows BaseObject's TaskManager setup to this public shape.
    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        await super().setup(setup)
        await self._recorder.setup(FrameProcessorSetup(
            clock=setup.clock, task_manager=setup.task_manager,
            pipeline_worker=setup.pipeline_worker, observer=None,
        ))

    async def start_capture(self) -> None:
        if self._started or self._closed:
            return
        self._started = True
        try:
            await self._recorder.start_recording()  # type: ignore[no-untyped-call]
            self._state = "recording"
        except Exception:
            self._refuse_capture()

    def _mark_capture_refused(self) -> None:
        self._closed = True
        self._state = "partial"
        if self._on_capture_refused is not None:
            with suppress(Exception):
                self._on_capture_refused()

    def _refuse_capture(self) -> None:
        self._mark_capture_refused()
        if self._completion is None or self._completion.done():
            self._completion = asyncio.create_task(
                self._stop_and_join(), name="sparra-audio-event-join"
            )

    async def _stop_and_join(self) -> bool:
        self._permit = True
        self._pending_join = True
        try:
            if not self._stop_called:
                await self._recorder.stop_recording()  # type: ignore[no-untyped-call]
                self._stop_called = True
            await BaseObject.cleanup(self._recorder)  # type: ignore[no-untyped-call]
            if self._on_event_join is not None and await self._on_event_join() is not True:
                self._closed = True
                self._state = "partial"
                return False
        except Exception:
            self._closed = True
            self._state = "partial"
            return False
        self._pending_join = False
        self._permit = False
        if self._state != "partial":
            self._state = "stopped"
        return True

    async def _join_event(self) -> bool:
        self._pending_join = True
        try:
            await BaseObject.cleanup(self._recorder)  # type: ignore[no-untyped-call]
        except Exception:
            self._closed = True
            self._state = "partial"
            return False
        self._pending_join = False
        if self._on_event_join is not None:
            # Native events are joined. Subthreshold frames may resume while
            # this same completion awaits the exact local writer receipt.
            self._permit = False
            try:
                if await self._on_event_join() is not True:
                    self._closed = True
                    self._state = "partial"
                    self._pending_join = True
                    return False
            except Exception:
                self._closed = True
                self._state = "partial"
                self._pending_join = True
                return False
        if self._closed:
            return await self._stop_and_join()
        self._permit = False
        return True

    async def stop_capture(self) -> bool:
        self.close_admission()
        return await self.quiesce()

    def close_admission(self, *, partial: bool = False) -> None:
        """Close new dispatch synchronously while retaining already admitted audio."""
        if partial:
            self._mark_capture_refused()
        else:
            self._closed = True
            if self._state != "partial":
                self._state = "stopped"

    async def quiesce(self) -> bool:
        self._closed = True
        task = self._completion
        if task is not None and not task.done() and not await asyncio.shield(task):
            return False
        self._completion = None
        return await self._stop_and_join()

    async def cleanup(self) -> None:
        if not await self.quiesce():
            raise RuntimeError("audio_event_join_pending")
        await self._recorder.cleanup()  # type: ignore[no-untyped-call]
        await super().cleanup()  # type: ignore[no-untyped-call]

    async def _receive_audio(
        self, _recorder: AudioBufferProcessor, pcm: bytes, rate: int, channels: int
    ) -> None:
        try:
            if (
                rate != 8000 or channels != 2 or not pcm or len(pcm) % 4
                or len(pcm) > 32000 or self._total_samples + len(pcm) // 4 > _MAX_SAMPLES
            ):
                self._mark_capture_refused()
                return
            self._total_samples += len(pcm) // 4
            if self._offer_chunk(pcm, rate, channels) is not True:
                self._mark_capture_refused()
        except Exception:
            # Native BaseObject logs callback exceptions, so none may escape here.
            self._mark_capture_refused()

    def _project_audio(
        self, frame: InputAudioRawFrame | OutputAudioRawFrame, now: float
    ) -> tuple[int, int, float | None, float | None] | None:
        size = len(frame.audio)
        if (
            frame.sample_rate != 8000 or frame.num_channels != 1
            or not 0 < size <= _MAX_FRAME_BYTES or size % 2 or not math.isfinite(now)
        ):
            return None
        user, bot = self._user_position, self._bot_position
        user_time, bot_time = self._last_user, self._last_bot
        previous = user_time if isinstance(frame, InputAudioRawFrame) else bot_time
        elapsed = 0.0 if previous is None else now - previous
        if elapsed < 0 or elapsed > 2.0:
            return None
        gap = elapsed - size / _MONO_BYTES_PER_SECOND
        padding = int(gap * _MONO_BYTES_PER_SECOND) if gap > 0.2 else 0
        padding -= padding % 2
        if isinstance(frame, InputAudioRawFrame):
            user += padding
            if not self._bot_speaking:
                bot, bot_time = max(bot, user), now
            user, user_time = user + size, now
        else:
            bot += padding
            if not self._user_speaking:
                user, user_time = max(user, bot), now
            bot, bot_time = bot + size, now
        maximum = max(user, bot)
        if maximum > _MAX_MONO_BYTES or self._total_samples + maximum // 2 > _MAX_SAMPLES:
            return None
        return user, bot, user_time, bot_time

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Observation may await. Admission must use the clock only after it returns.
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._recorder.queue_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop_capture()
        elif not self._closed and self._state == "recording":
            if isinstance(frame, (InputAudioRawFrame, OutputAudioRawFrame)):
                if self._permit:
                    self._refuse_capture()
                else:
                    projected = self._project_audio(frame, self._monotonic())
                    if projected is None or (
                        max(projected[0], projected[1]) >= _NATIVE_THRESHOLD_BYTES
                        and self._ready_for_native_event is not None
                        and self._ready_for_native_event() is not True
                    ):
                        self._refuse_capture()
                    else:
                        self._permit = True
                        await self._recorder.queue_frame(
                            frame, direction
                        )
                        (self._user_position, self._bot_position,
                         self._last_user, self._last_bot) = projected
                        if max(projected[0], projected[1]) >= _NATIVE_THRESHOLD_BYTES:
                            self._user_position = self._bot_position = 0
                            self._last_user = self._last_bot = self._monotonic()
                            self._completion = asyncio.create_task(
                                self._join_event(), name="sparra-audio-event-join"
                            )
                        else:
                            self._permit = False
            elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                    BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
                    self._user_speaking = isinstance(frame, UserStartedSpeakingFrame)
                else:
                    self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
                await self._recorder.queue_frame(frame, direction)
        await self.push_frame(frame, direction)


class LocalAudioCapture:
    """One call-owned native tap and holder; completion is local byte evidence only."""

    def __init__(
        self, *, snapshot: BeginCallSnapshotV2, deployment_id: str,
        generation: UUID, keyring: CryptoKeyring, writer: PersistenceWriter,
    ) -> None:
        if not isinstance(generation, UUID):
            raise ValueError("audio_generation_invalid")
        self.holder = AudioChunkHolder(snapshot=snapshot, deployment_id=deployment_id,
                                       keyring=keyring, writer=writer)
        self._snapshot = snapshot
        self._deployment_id = validate_deployment_id(deployment_id)
        self._generation, self._writer = generation, writer
        self._finish_operation: VoiceOperationV2 | None = None
        self._revoke_operation: VoiceOperationV2 | None = None
        self._finish_pending = self._revoke_pending = False
        self._finish_lock, self._revoke_lock = asyncio.Lock(), asyncio.Lock()
        self._limit_finish_task: asyncio.Task[AudioCaptureSummary] | None = None
        self.tap = BoundedAudioBufferTap(
            offer_chunk=self.holder.offer_native_pcm, on_event_join=self._after_event_join,
            ready_for_native_event=self.holder.ready_for_native_event,
            on_capture_refused=self.holder.notify_capture_refused,
        )
        self._available = snapshot.audio_available and snapshot.recording_policy == "local_30d"
        self._quiesce_lock = asyncio.Lock()
        self._closed = False

    def ready_for_transfer(self) -> bool:
        summary = self.holder.summary
        if not self._closed or self.tap.pending_join or summary.pending:
            return False
        if self._revoke_operation is not None:
            return not self._revoke_pending
        operation = self._finish_operation
        return (
            operation is not None and not self._finish_pending
            and isinstance(operation.payload, AudioFinishPayloadV2)
            and operation.payload.reason == "transfer" and not summary.partial
            and summary.submitted_samples == summary.committed_samples
        )

    async def start(self) -> bool:
        if self._closed or not self._available:
            return False
        await self.tap.start_capture()
        return self.tap.state == "recording"

    def refuse(self) -> None:
        self._closed = True
        if self.tap.state != "stopped":
            self.tap.close_admission(partial=True)

    def close_admission(self) -> None:
        self._closed = True
        self.tap.close_admission()

    async def _after_event_join(self) -> bool:
        settled = await self.holder.after_event_join()
        if self.holder.summary.reason == "limit" and self._limit_finish_task is None:
            self._limit_finish_task = asyncio.create_task(
                self.finish("limit"), name="sparra-audio-limit-finish"
            )
        return settled

    def _terminal_summary(self, summary: AudioCaptureSummary) -> AudioCaptureSummary:
        return AudioCaptureSummary(
            summary.submitted_samples, summary.committed_samples, summary.last_sequence,
            summary.reason, summary.partial,
            summary.pending or self._finish_pending or self._revoke_pending,
        )

    async def _quiesce_capture(self) -> AudioCaptureSummary:
        self.close_admission()
        async with self._quiesce_lock:
            if not await self.tap.quiesce():
                self.holder.notify_capture_refused()
                current = self.holder.summary
                return AudioCaptureSummary(
                    current.submitted_samples, current.committed_samples, current.last_sequence,
                    "failure", True, True,
                )
            return await self.holder.finish()

    async def quiesce(self) -> AudioCaptureSummary:
        task = self._limit_finish_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.shield(task)
        return self._terminal_summary(await self._quiesce_capture())

    def _terminal_operation(
        self, kind: Literal["audio.finish", "audio.revoke"],
        summary: AudioCaptureSummary | None = None, reason: AudioFinishReason = "failure",
    ) -> VoiceOperationV2:
        pin = self._snapshot
        payload: dict[str, object] = {
            "schema_version": 2, "workspace_id": str(pin.workspace_id),
            "recording_id": str(pin.recording_id),
            "configuration_revision": pin.configuration_revision,
            "retention_until": pin.retention_until.isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "reason": "caller_declined" if kind == "audio.revoke" else reason,
        }
        if summary is not None:
            payload.update(
                last_sequence=summary.last_sequence, total_samples=summary.committed_samples
            )
        return VoiceOperationV2.model_validate({
            "schema_version": 2, "operation_id": str(uuid4()),
            "deployment_id": self._deployment_id, "call_id": str(pin.call_id),
            "occurred_at": datetime.now(UTC), "kind": kind, "payload": payload,
        })

    async def finish(self, reason: AudioFinishReason) -> AudioCaptureSummary:
        if reason not in {"complete", "limit", "failure", "transfer", "interrupted"}:
            raise ValueError("audio_finish_reason_invalid")
        self.close_admission()
        async with self._finish_lock:
            summary = await self._quiesce_capture()
            if summary.pending or self._revoke_operation is not None:
                return self._terminal_summary(summary)
            if self._finish_operation is None:
                selected: AudioFinishReason = (
                    "limit" if summary.reason == "limit"
                    else "failure" if reason == "complete" and summary.partial else reason
                )
                self._finish_operation = self._terminal_operation("audio.finish", summary, selected)
            self._finish_pending = True
            try:
                await self._writer.publish_audio_terminal_v2(
                    self._finish_operation, generation=self._generation
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.holder.notify_capture_refused()
                return self._terminal_summary(self.holder.summary)
            self._finish_pending = False
            return self._terminal_summary(summary)

    async def revoke(self) -> AudioCaptureSummary:
        self.refuse()
        if self._revoke_operation is None:
            self._revoke_operation = self._terminal_operation("audio.revoke")
        self._revoke_pending = True
        async with self._revoke_lock:
            try:
                # Writer admission precedes native/receipt quiescence, including unknown audio.
                await self._writer.publish_audio_terminal_v2(
                    self._revoke_operation, generation=self._generation
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.holder.notify_capture_refused()
                return self._terminal_summary(self.holder.summary)
            self._revoke_pending = False
            # Known revoke COMMIT supersedes finish delivery, including a cancelled waiter.
            self._finish_pending = False
            return self._terminal_summary(await self._quiesce_capture())
