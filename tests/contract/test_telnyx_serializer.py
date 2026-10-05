from __future__ import annotations

import base64
import inspect
import json
import struct
from importlib import import_module
from importlib.metadata import version

import pytest
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    ControlFrame,
    DataFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    StartFrame,
    SystemFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.frame_queue import FrameQueue

from projetv0_voice.telnyx.frames import MAX_MARK_NAME_BYTES, TelnyxMarkFrame
from projetv0_voice.telnyx.serializer import ProjetV0TelnyxFrameSerializer, TelnyxSerializerError

serializer_module = import_module("projetv0_voice.telnyx.serializer")


class _UnknownSystemFrame(SystemFrame):
    pass


class _UnknownControlFrame(ControlFrame):
    pass


def _native_serializer(stream_id: str = "stream-one") -> TelnyxFrameSerializer:
    return TelnyxFrameSerializer(
        stream_id=stream_id,
        outbound_encoding="PCMU",
        inbound_encoding="PCMU",
        params=TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=8000,
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
            auto_hang_up=False,
        ),
    )


def _assert_constant_safe(error: TelnyxSerializerError, secret: str) -> None:
    assert str(error) == "telnyx_serializer_invalid"
    assert repr(error) == "TelnyxSerializerError('telnyx_serializer_invalid')"
    if secret:
        assert secret not in str(error)
        assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_pinned_native_types_and_constructor_contract() -> None:
    assert version("pipecat-ai") == "1.7.0"
    assert issubclass(TelnyxMarkFrame, OutputTransportMessageFrame)
    assert issubclass(TelnyxMarkFrame, DataFrame)
    assert not issubclass(TelnyxMarkFrame, SystemFrame)
    assert not issubclass(TelnyxMarkFrame, ControlFrame)
    assert list(inspect.signature(TelnyxFrameSerializer).parameters) == [
        "stream_id",
        "outbound_encoding",
        "inbound_encoding",
        "call_control_id",
        "api_key",
        "params",
    ]

    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one", expected_call_control_id="call-one"
    )

    assert serializer._params.telnyx_sample_rate == 8000
    assert serializer._params.outbound_encoding == "PCMU"
    assert serializer._params.inbound_encoding == "PCMU"
    assert serializer._params.auto_hang_up is False
    assert serializer._call_control_id is None
    assert serializer._api_key is None


@pytest.mark.asyncio
async def test_native_pcmu_audio_dtmf_and_clear_remain_equivalent() -> None:
    native = _native_serializer()
    admission = serializer_module.AudioAdmission()
    admission.bind(lambda: True)
    project = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    start = StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=16000)
    await native.setup(start)
    await project.setup(start)

    pcm = b"".join(struct.pack("<h", sample) for sample in range(-8000, 8000, 100))
    outbound = OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
    assert await project.serialize(outbound) == await native.serialize(outbound)
    assert await project.serialize(InterruptionFrame()) == await native.serialize(
        InterruptionFrame()
    )

    media = json.dumps(
        {
            "event": "media",
            "stream_id": "stream-one",
            "sequence_number": "1",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "0",
                "payload": base64.b64encode(bytes(range(80))).decode("ascii"),
            },
        }
    )
    project_audio = await project.deserialize(media)
    native_audio = await native.deserialize(media)
    assert isinstance(project_audio, InputAudioRawFrame)
    assert isinstance(native_audio, InputAudioRawFrame)
    assert project_audio.audio == native_audio.audio
    assert project_audio.sample_rate == native_audio.sample_rate == 8000
    assert project_audio.num_channels == native_audio.num_channels == 1

    dtmf = json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})
    project_dtmf = await project.deserialize(dtmf)
    native_dtmf = await native.deserialize(dtmf)
    assert isinstance(project_dtmf, InputDTMFFrame)
    assert isinstance(native_dtmf, InputDTMFFrame)
    assert project_dtmf.button == native_dtmf.button == KeypadEntry.FIVE


@pytest.mark.asyncio
async def test_documented_large_media_payload_still_delegates_to_native() -> None:
    native = _native_serializer()
    admission = serializer_module.AudioAdmission()
    admission.bind(lambda: True)
    project = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    start = StartFrame(audio_in_sample_rate=8000)
    await native.setup(start)
    await project.setup(start)
    ulaw = bytes(range(256)) * 200
    media = json.dumps(
        {
            "event": "media",
            "stream_id": "stream-one",
            "sequence_number": "1",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "0",
                "payload": base64.b64encode(ulaw).decode("ascii"),
            },
        }
    )
    assert len(media.encode("utf-8")) > 65_536

    native_frame = await native.deserialize(media)
    project_frame = await project.deserialize(media)

    assert isinstance(native_frame, InputAudioRawFrame)
    assert isinstance(project_frame, InputAudioRawFrame)
    assert project_frame.audio == native_frame.audio
    assert project_frame.sample_rate == native_frame.sample_rate


@pytest.mark.asyncio
async def test_audio_admission_is_closed_until_once_bound_and_fails_closed() -> None:
    admission = serializer_module.AudioAdmission()
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=admission,
    )
    await serializer.setup(StartFrame(audio_in_sample_rate=8000))
    media = json.dumps(
        {
            "event": "media",
            "stream_id": "stream-one",
            "sequence_number": "1",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "0",
                "payload": base64.b64encode(bytes(range(80))).decode("ascii"),
            },
        }
    )

    assert await serializer.deserialize(media) is None

    active = False
    admission.bind(lambda: active)
    assert await serializer.deserialize(media) is None
    active = True
    assert isinstance(await serializer.deserialize(media), InputAudioRawFrame)

    with pytest.raises(
        serializer_module.AudioAdmissionError, match="audio_admission_already_bound"
    ) as duplicate:
        admission.bind(lambda: True)
    assert duplicate.value.__cause__ is None
    assert duplicate.value.__context__ is None

    raising = serializer_module.AudioAdmission()
    raising.bind(lambda: (_ for _ in ()).throw(RuntimeError("provider-secret")))
    guarded = ProjetV0TelnyxFrameSerializer(
        "stream-one",
        expected_call_control_id="call-one",
        audio_admission=raising,
    )
    await guarded.setup(StartFrame(audio_in_sample_rate=8000))
    with pytest.raises(TelnyxSerializerError) as failed:
        await guarded.deserialize(media)
    assert str(failed.value) == "telnyx_audio_admission_failed"
    assert "provider-secret" not in repr(failed.value)
    assert failed.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_outbound_media_is_dropped_before_native_decode_or_audio_admission(
    admitted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = serializer_module.AudioAdmission()

    def admission_predicate() -> bool:
        raise AssertionError("outbound media reached audio admission")

    if admitted:
        admission.bind(admission_predicate)

    async def forbidden_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("outbound media reached the native PCMU converter")

    monkeypatch.setattr("pipecat.serializers.telnyx.ulaw_to_pcm", forbidden_decode)
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one", expected_call_control_id="call-one", audio_admission=admission
    )
    await serializer.setup(StartFrame(audio_in_sample_rate=8000))
    payload = {
        "event": "media",
        "stream_id": "stream-one",
        "sequence_number": "1",
        "media": {
            "track": "outbound",
            "chunk": "1",
            "timestamp": "0",
            "payload": base64.b64encode(bytes(range(80))).decode("ascii"),
        },
    }

    assert await serializer.deserialize(json.dumps(payload)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        "wrong-stream",
        "missing-stream",
        "missing-track",
        "invalid-track",
        "missing-media",
        "list-media",
        "missing-payload",
        "nonstring-payload",
    ],
)
async def test_media_critical_shape_is_validated_even_when_admission_is_closed(
    admitted: bool,
    invalid: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = serializer_module.AudioAdmission()
    if admitted:
        admission.bind(lambda: True)
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one", expected_call_control_id="call-one", audio_admission=admission
    )
    await serializer.setup(StartFrame(audio_in_sample_rate=8000))
    media: dict[str, object] = {
        "track": "inbound",
        "chunk": "1",
        "timestamp": "0",
        "payload": base64.b64encode(bytes(range(80))).decode("ascii"),
    }
    payload: dict[str, object] = {
        "event": "media",
        "stream_id": "stream-one",
        "sequence_number": "1",
        "media": media,
    }
    if invalid == "wrong-stream":
        payload["stream_id"] = "sentinel-wrong-stream"
    elif invalid == "missing-stream":
        del payload["stream_id"]
    elif invalid == "missing-track":
        del media["track"]
    elif invalid == "invalid-track":
        media["track"] = "inbound_track"
    elif invalid == "missing-media":
        del payload["media"]
    elif invalid == "list-media":
        payload["media"] = []
    elif invalid == "missing-payload":
        del media["payload"]
    elif invalid == "nonstring-payload":
        media["payload"] = 123

    async def forbidden_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("invalid media reached the native PCMU converter")

    monkeypatch.setattr("pipecat.serializers.telnyx.ulaw_to_pcm", forbidden_decode)
    with pytest.raises(TelnyxSerializerError) as raised:
        await serializer.deserialize(json.dumps(payload))
    _assert_constant_safe(raised.value, "sentinel-wrong-stream")


@pytest.mark.asyncio
async def test_only_the_project_mark_frame_encodes_the_official_mark_payload() -> None:
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one", expected_call_control_id="call-one"
    )

    mark = TelnyxMarkFrame("disclosure-ready")
    assert mark.message == {
        "event": "mark",
        "mark": {"name": "disclosure-ready"},
    }
    assert await serializer.serialize(mark) == (
        '{"event":"mark","mark":{"name":"disclosure-ready"}}'
    )
    assert (
        await serializer.serialize(
            OutputTransportMessageFrame(
                message={"event": "mark", "mark": {"name": "not-project-owned"}}
            )
        )
        is None
    )
    assert await serializer.serialize(_UnknownSystemFrame()) is None
    assert await serializer.serialize(_UnknownControlFrame()) is None


def test_mark_name_is_nonempty_and_bounded_by_utf8_bytes() -> None:
    TelnyxMarkFrame("e" * MAX_MARK_NAME_BYTES)
    TelnyxMarkFrame("é" * (MAX_MARK_NAME_BYTES // 2))

    with pytest.raises(TelnyxSerializerError) as empty:
        TelnyxMarkFrame("")
    _assert_constant_safe(empty.value, "")

    secret = "é" * (MAX_MARK_NAME_BYTES // 2 + 1)
    with pytest.raises(TelnyxSerializerError) as oversized:
        TelnyxMarkFrame(secret)
    _assert_constant_safe(oversized.value, secret)


@pytest.mark.asyncio
async def test_inbound_extension_events_are_minimal_and_redacted() -> None:
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one", expected_call_control_id="call-one"
    )

    mark = await serializer.deserialize(
        json.dumps(
            {
                "event": "mark",
                "stream_id": "stream-one",
                "sequence_number": "42",
                "mark": {"name": "disclosure-ready", "provider_extra": "discard"},
                "provider_extra": "discard",
            }
        )
    )
    stop = await serializer.deserialize(
        json.dumps(
            {
                "event": "stop",
                "stream_id": "stream-one",
                "sequence_number": "43",
                "stop": {"call_control_id": "call-one"},
            }
        )
    )
    error = await serializer.deserialize(
        json.dumps(
            {
                "event": "error",
                "stream_id": "stream-one",
                "payload": {
                    "code": 100005,
                    "title": "media failure",
                    "detail": "raw-provider-detail-secret",
                },
            }
        )
    )

    assert isinstance(mark, InputTransportMessageFrame)
    assert mark.message == {
        "event": "mark",
        "mark": {"name": "disclosure-ready"},
    }
    assert isinstance(stop, InputTransportMessageFrame)
    assert stop.message == {"event": "stop"}
    assert isinstance(error, InputTransportMessageFrame)
    assert error.message == {
        "event": "error",
        "payload": {"code": 100005, "title": "media failure"},
    }
    assert "raw-provider-detail-secret" not in repr(error)
    assert "call-one" not in repr(stop)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,secret",
    [
        ('{"event":"mark","stream_id":"wrong","mark":{"name":"secret-name"}}', "secret-name"),
        ('{"event":"mark","stream_id":"stream-one","mark":{"name":""}}', "stream-one"),
        (
            '{"event":"error","stream_id":"stream-one","payload":'
            '{"code":100005,"title":"' + "s" * 257 + '"}}',
            "s" * 257,
        ),
        (
            '{"event":"error","stream_id":"stream-one","payload":'
            '{"code":"100005","title":"string-code-secret"}}',
            "string-code-secret",
        ),
        ('{"event":"stop","stream_id":"stream-one",', "stream-one"),
        (
            '{"event":"mark","event":"stop","stream_id":"stream-one",'
            '"mark":{"name":"duplicate-secret"}}',
            "duplicate-secret",
        ),
        ('{"event":"stop","stream_id":"stream-one"}', "stream-one"),
        (
            '{"event":"stop","stream_id":"stream-one","stop":'
            '{"call_control_id":"wrong-call-secret"}}',
            "wrong-call-secret",
        ),
        (
            '{"event":"stop","stream_id":"stream-one","stop":'
            '{"call_control_id":"' + "c" * 1025 + '"}}',
            "c" * 1025,
        ),
    ],
)
async def test_malformed_extension_events_raise_one_constant_safe_error(
    payload: str, secret: str
) -> None:
    serializer = ProjetV0TelnyxFrameSerializer(
        "stream-one", expected_call_control_id="call-one"
    )

    with pytest.raises(TelnyxSerializerError) as raised:
        await serializer.deserialize(payload)

    _assert_constant_safe(raised.value, secret)


@pytest.mark.asyncio
async def test_tts_trailing_audio_stop_and_mark_keep_local_fifo_order() -> None:
    params = TransportParams(
        audio_out_enabled=True,
        audio_out_sample_rate=8000,
        audio_out_channels=1,
        audio_out_10ms_chunks=4,
    )
    transport = BaseOutputTransport(params)
    sender = BaseOutputTransport.MediaSender(
        transport,
        destination=None,
        sample_rate=8000,
        audio_chunk_size=640,
        params=params,
    )
    sender._audio_queue = FrameQueue()
    try:
        trailing = TTSAudioRawFrame(
            audio=b"\x01\x00" * 40,
            sample_rate=8000,
            num_channels=1,
            context_id="greeting",
        )
        stopped = TTSStoppedFrame(context_id="greeting")
        mark = TelnyxMarkFrame("disclosure-ready")

        await sender.handle_audio_frame(trailing)
        await sender.handle_tts_stopped(stopped)
        await sender.handle_sync_frame(mark)

        queued = [sender._audio_queue.get_nowait() for _ in range(3)]
        assert isinstance(queued[0], TTSAudioRawFrame)
        assert len(queued[0].audio) == 640
        assert queued[1] is stopped
        assert queued[2] is mark
    finally:
        sender._executor.shutdown(wait=True)
