"""Owned connected fixture. Native graph/SDKs/ASGI/Pipecat/SQLite/SQL, controlled peers only."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import ipaddress
import json
import socket
import sqlite3
import ssl
import sys
import time
import traceback
from contextlib import ExitStack, asynccontextmanager, closing
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import psycopg
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from nacl.signing import SigningKey
from openai import DefaultAsyncHttpxClient
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr
from websockets.asyncio.client import connect

from projetv0_voice.admission import CallAdmissionRejected
from projetv0_voice.app import create_app
from projetv0_voice.audio_contract import (
    AudioFinishPayloadV2,
    AudioRevokePayloadV2,
    canonical_audio_chunk_aad,
)
from projetv0_voice.config import AgentManifestV1
from projetv0_voice.crypto import EncryptedValue
from projetv0_voice.inference import services as native_services
from projetv0_voice.inference.openrouter_tts import OpenRouterTTSService
from projetv0_voice.lifecycle import (
    RuntimeInferenceFactories,
    RuntimeProductionFactories,
    RuntimeProfileSelection,
    build_production_runtime,
)
from projetv0_voice.metrics import RuntimeMetrics
from projetv0_voice.models import RoutingV1, VoiceOperationV1
from projetv0_voice.persistence.commands import (
    canonical_operation_bytes,
    decode_operation,
    decode_operation_v2,
    operation_aad_from_metadata,
)
from projetv0_voice.persistence.postgres_sink import PsycopgOperationSink
from projetv0_voice.production_wiring import _decode_keyring_value
from projetv0_voice.qualified_profile import (
    InferenceProfileV1,
    QualificationCandidateProfileV1,
    QualifiedDeploymentProfileV1,
    canonical_candidate_profile_sha256,
    canonical_inference_profile_sha256,
)
from projetv0_voice.runtime_config import capture_runtime_environment, parse_runtime_settings
from projetv0_voice.telnyx import call_control as native_control
from projetv0_voice.telnyx.handshake import _redacted_fixture
from projetv0_voice.telnyx.recordings import (
    build_recording_correlation,
    encode_recording_correlation,
)
from projetv0_voice.telnyx.serializer import ProjetV0TelnyxFrameSerializer

_lose_begin_reply = ContextVar("owned_connected_begin_reply", default=False)


class LostBeginReplyPool(AsyncConnectionPool):
    """One lost reply AFTER actual native successful COMMIT, scoped to begin only."""

    lost_once = False

    @asynccontextmanager
    async def connection(self, *args, **kwargs):
        async with super().connection(*args, **kwargs) as connection:
            yield connection
        if _lose_begin_reply.get() and not self.lost_once:
            self.lost_once = True
            raise psycopg.OperationalError("owned_connected_lost_commit_reply")


def now():
    value = datetime.now(UTC)
    return value.replace(microsecond=value.microsecond // 1000 * 1000)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def observe_native_startup(supervisor, patches, diagnostic, report=None):
    """Test-only observation; the original owns scheduling, budgets and exceptions."""
    original = supervisor._startup_await
    phases = {
        "writer_startup_failed",
        "writer_quick_check_failed",
        "qualification_status_failed",
        "operation_sink_open_failed",
        "stale_recovery_failed",
        "runtime_publication_failed",
        "runtime_begin_drain_failed",
    }

    async def observed(awaitable, *, code):
        diagnostic["last_phase"] = (
            code if type(code) is str and code in phases else "other_safe_failure"
        )
        diagnostic["outcome"] = "other"
        if report is not None:
            report(diagnostic["last_phase"])
        try:
            return await original(awaitable, code=code)
        except BaseException:
            diagnostic["outcome"] = (
                "native_timeout"
                if supervisor._startup_phase_timed_out is True
                else "native_exception"
            )
            raise

    patches.enter_context(patch.object(supervisor, "_startup_await", observed))


def safe_startup_diagnostic(error, diagnostic, recovery_case):
    """Closed public fields only: never stringify exceptions, locals or source lines."""
    terminal_codes = {
        "runtime_startup_invalid",
        "stale_recovery_failed",
        "qualification_run_consumed",
        "writer_startup_failed",
        "writer_quick_check_failed",
        "owned_task_registration_failed",
        "runtime_startup_failed",
    }
    phases = {
        "writer_startup_failed",
        "writer_quick_check_failed",
        "qualification_status_failed",
        "operation_sink_open_failed",
        "stale_recovery_failed",
        "runtime_publication_failed",
        "runtime_begin_drain_failed",
    }
    terminal_code = "other_safe_failure"
    if type(error) is RuntimeError and len(error.args) == 1:
        argument = error.args[0]
        if type(argument) is str and argument in terminal_codes:
            terminal_code = argument
    phase = diagnostic.get("last_phase")
    outcome = diagnostic.get("outcome")
    classes = {
        RuntimeError: "RuntimeError",
        AssertionError: "AssertionError",
        TimeoutError: "TimeoutError",
        asyncio.CancelledError: "CancelledError",
    }
    frame_names = {
        "setup",
        "startup",
        "_startup_await",
        "_recover_stale_leases",
        "prepare_before_fifo",
        "maintain_call_content",
        "observed_restore",
        "restore_transfer_fence",
        "_finish_startup_unwind",
    }
    frames = []
    current = error.__traceback__
    for _ in range(64):
        if current is None:
            break
        name = current.tb_frame.f_code.co_name
        if name in frame_names:
            frames.append(name)
        current = current.tb_next
    return {
        "terminal_code": terminal_code,
        "last_phase": phase if type(phase) is str and phase in phases else "other_safe_failure",
        "outcome": outcome
        if type(outcome) is str and outcome in {"native_timeout", "native_exception"}
        else "other",
        "recovery_case": recovery_case
        if type(recovery_case) is str
        and recovery_case in {"held", "backlog", "final-fix", "replay"}
        else "other",
        "error_class": classes.get(type(error), "OtherException"),
        "frames": frames[-8:],
    }


async def eventually(predicate, label, seconds=12):
    try:
        async with asyncio.timeout(seconds):
            while True:
                value = predicate()
                if hasattr(value, "__await__"):
                    value = await value
                if value:
                    return value
                await asyncio.sleep(0.02)
    except TimeoutError:
        raise AssertionError("native_timeout_" + label) from None


class Peers:
    """Controlled wire-level peers, not replaced auth/DTO/SQL/runtime consumers."""

    def __init__(self):
        self.actions = []
        self.stt_text = "Pouvez-vous me rappeler ?"
        self.requests = []
        self.stream = None
        self.inferences = 0
        self.deleted = []
        self.recordings = {}
        self.tool_arguments = None
        self.transfer = None
        self.before_transfer = None

    async def http(self, request):
        path = request.url.path
        if request.url.host == "api.telnyx.eu":
            if path.endswith("/actions/streaming_start"):
                self.stream = json.loads(request.content)
            if path.endswith("/actions/transfer"):
                self.transfer = json.loads(request.content)
                if self.before_transfer is not None:
                    await self.before_transfer(self.transfer)
            if request.method == "DELETE":
                self.deleted.append(path)
                recording = path.rsplit("/", 1)[-1]
                if recording not in self.recordings:
                    return httpx.Response(
                        404,
                        json={"errors": [{"code": "404", "title": "Not found"}]},
                        request=request,
                    )
                del self.recordings[recording]
                return httpx.Response(200, json={"data": {"id": recording}}, request=request)
            self.actions.append(path)
            return httpx.Response(200, json={"data": {"result": "ok"}}, request=request)
        assert request.url.host == "openrouter.ai", "unexpected_fixture_network"
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": self.stt_text}, request=request)
        if path.endswith("/audio/speech"):
            return httpx.Response(
                200, headers={"Content-Type": "audio/pcm"}, content=b"\0\0" * 2400, request=request
            )
        assert path.endswith("/chat/completions"), "unexpected_inference_route"
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("stream"):
            if self.tool_arguments is not None:
                arguments, self.tool_arguments = self.tool_arguments, None
                tool = {
                    "id": "owned-tool-call",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "test/llm",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "owned-human-tool",
                                        "type": "function",
                                        "function": {
                                            "name": "request_human",
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                finish = {
                    "id": "owned-tool-call",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "test/llm",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
                data = (
                    "data: "
                    + json.dumps(tool)
                    + "\n\ndata: "
                    + json.dumps(finish)
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=data,
                    request=request,
                )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b"data: [DONE]\n\n",
                request=request,
            )
        self.inferences += 1
        assert not body.get("tools"), "result_inference_must_be_tool_free"
        facts = json.loads(body["messages"][-1]["content"])
        user = next(turn for turn in facts["turns"] if turn["role"] == "user")
        result = {
            "schema_version": 1,
            "quality": "partial",
            "category": "callback",
            "summary": "Demande de rappel",
            "next_action": "Rappeler",
            "contact": {
                "name": None,
                "callback_e164": None,
                "preference": None,
                "callback_source": "missing",
                "callback_confirmed": False,
            },
            "evidence": [{"turn_id": user["turn_id"], "role": "user"}],
            "request_confirmed": False,
        }
        return httpx.Response(
            200,
            json={
                "id": "fixture-result",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "test/llm",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
            },
            request=request,
        )


class Coordinator:
    def __init__(self, graph):
        self.graph = graph

    async def build_runtime(self, _settings):
        return self.graph

    def publish_runtime(self, _supervisor):
        pass

    def unpublish_runtime(self, _supervisor):
        pass

    def publish_startup_complete(self):
        pass

    def publish_startup_failure(self):
        pass


class MediaPeer:
    """Owned loopback TLS/WSS peer. Public ingress remains operator acceptance."""

    def __init__(self, url, tls, control, stream):
        self.url, self.tls, self.control, self.stream = url, tls, control, stream
        self.messages = []
        self.task = None
        self.socket = None

    async def open(self, token):
        self.socket = await connect(
            self.url + "/telnyx/media",
            ssl=self.tls,
            proxy=None,
            additional_headers={"x-telnyx-streaming-auth-token": token},
            open_timeout=5,
            close_timeout=5,
        )
        self.task = asyncio.create_task(self.observe())
        await self.input(
            {
                "protocol": "Call",
                "version": "1.0.0",
                "event": "connected",
                "connected": {"x-telnyx-streaming-auth-token": token},
            }
        )
        await self.input(
            {
                "event": "start",
                "stream_id": self.stream,
                "sequence_number": "1",
                "start": {
                    "call_control_id": self.control,
                    "from": None,
                    "to": "+33123456789",
                    "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
                },
            }
        )

    async def input(self, message):
        await self.socket.send(json.dumps(message))

    async def observe(self):
        async for message in self.socket:
            value = json.loads(message)
            self.messages.append(value.get("event"))
            if value.get("event") == "mark":
                await self.input({"event": "mark", "stream_id": self.stream, "mark": value["mark"]})

    async def close(self):
        if self.socket is not None:
            await self.socket.close()
        if self.task is not None:
            await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), 15)


class Scenario:
    def __init__(self, request, directory):
        self.request, self.directory = request, Path(directory)
        self.peers = Peers()
        self.signing = SigningKey.generate()
        self.evidence = {"controlled_peers": True, "checks": []}
        self.media = None
        self.graph = None
        self.call_id = None
        self.app = None
        self.phase = "setup"
        self.patches = ExitStack()
        self.startup_diagnostic = {}
        self.lifespan = None
        self.server = None
        self.server_task = None
        self.local_acks = []
        self.remote_acks = []
        self.holder_checks = 0
        self.hold_begin = None
        self.begin_entered = asyncio.Event()
        self.begin_release = asyncio.Event()
        self.begin_cancelled = asyncio.Event()
        self.race_begin_entered = asyncio.Event()
        self.audio_candidate = request.get("audio_candidate") is True
        self.audio_capture = None
        self.audio_pcm_digest = hashlib.sha256()
        self.audio_seen = set()
        self.audio_gate_ingested = {}
        self.audio_gate_acked = {}
        self.audio_ack_entered, self.audio_ack_release = asyncio.Event(), asyncio.Event()
        self.audio_ack_hold = False
        self.audio_ack_refusal = None
        self.audio_ack_attempts = 0
        self.audio_erasure_lease_token = None
        self.audio_cleanup_pending = 0
        self.audio_cleanup_commit_entered = asyncio.Event()
        self.audio_cleanup_commit_release = asyncio.Event()
        self.audio_cleanup_commit_completed = False
        self.audio_cleanup_commit_action = None
        self.audio_cleanup_commit_fence = None
        self.audio_phase_epoch = time.monotonic()

    def candidate_phase(self, phase):
        if self.audio_candidate:
            memory = {}
            if sys.platform == "linux":
                import resource

                memory["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            emit(
                {
                    "phase": phase,
                    "elapsed_ms": int((time.monotonic() - self.audio_phase_epoch) * 1000),
                    **memory,
                }
            )

    def fixture_metrics(self, _cls, _token, **_kwargs):
        self.candidate_phase("metrics-build")
        metrics = RuntimeMetrics.in_memory()
        self.candidate_phase("metrics-built")
        return metrics

    async def setup(self):
        self.candidate_phase("candidate-setup")
        key = SecretStr("owned-" + uuid4().hex)
        inference = InferenceProfileV1.model_validate(
            {
                "schema_version": 1,
                "stt_model": "test/stt",
                "llm_model": "test/llm",
                "tts_model": "test/tts",
                "tts_voice": "fr-test",
                "tts_pcm_sample_rate": 24000,
                "tts_pcm_channels": 1,
                "llm_provider_policy": {"allow_fallbacks": False, "sort": "latency"},
                "tts_provider_options": {},
            }
        )
        env = {
            "VOICE_RUNTIME_MODE": "strict",
            "VOICE_DEPLOYMENT_ID": "fixture-a",
            "VOICE_RUNTIME_CONTRACT_PATH": "/srv/projetv0/runtime-contract.json",
            "VOICE_AGENT_BUNDLE_PATH": "/srv/projetv0/agent-bundle",
            "VOICE_QUALIFIED_PROFILE_PATH": "/srv/projetv0/qualified.json",
            "VOICE_KEYRING_PATH": "/srv/projetv0/keyring.json",
            "VOICE_SQLITE_PATH": "/var/lib/projetv0/voice.sqlite3",
            "VOICE_RUNTIME_CONTRACT_SHA256": "a" * 64,
            "VOICE_IMAGE_DIGEST": "ghcr.io/louisvannobel/projetv0-voice@sha256:" + "d" * 64,
            "VOICE_AGENT_BUNDLE_SHA256": "b" * 64,
            "VOICE_INFERENCE_PROFILE_SHA256": canonical_inference_profile_sha256(inference),
            "VOICE_DEPLOYMENT_MAX_CALLS": "1",
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
            "VOICE_OTLP_HTTP_ENDPOINT": "https://collector.invalid/v1/metrics",
            "VOICE_BIND_HOST": "127.0.0.1",
            "VOICE_BIND_PORT": "8080",
        }
        if self.audio_candidate:
            self.audio_run = uuid4()
            env.pop("VOICE_QUALIFIED_PROFILE_PATH")
            env.update(
                {
                    "VOICE_RUNTIME_MODE": "qualification_candidate",
                    "VOICE_QUALIFICATION_CANDIDATE_PATH": "/srv/projetv0/candidate.json",
                    "VOICE_QUALIFICATION_RUN_ID": str(self.audio_run),
                    "VOICE_BENCHMARK_DID_SHA256": hashlib.sha256(b"+33123456789").hexdigest(),
                }
            )
        parsed_settings = parse_runtime_settings(
            capture_runtime_environment(env), geteuid=lambda: 10001, getegid=lambda: 10001
        )
        self.settings = replace(
            parsed_settings,
            sqlite_path=PurePosixPath((self.directory / "voice.sqlite").as_posix()),
        )
        self.candidate_phase("settings-valid")
        # Relocate only owned fixture storage; retain the native parser-issued token.
        object.__setattr__(
            self.settings, "_observability_token", parsed_settings.observability_token()
        )
        self.manifest = AgentManifestV1.model_validate(
            {
                "schema_version": 1,
                "tenant_id": "fixture",
                "agent_id": "fixture",
                "revision": "fixture",
                "dids": ["+33123456789"],
                "language": "fr",
                "prompt_path": self.directory / "prompt.md",
                "prompt_revision": "fixture",
                "greeting": "Bonjour",
                "conversation_mode": "freeform",
                "max_concurrent_calls": 1,
                "direction": "inbound_only",
                "transport_codec": "PCMU",
                "transport_sample_rate_hz": 8000,
                "transcript_retention_days": 30,
                "recording_mode": "off",
                "recording_format": "wav",
                "recording_retention_days": None,
                "recording_required": False,
                "recording_play_beep": False,
                "sparra": {
                    "schema_version": 1,
                    "connection_id": "connection-a",
                    "original_forward_line_e164": None,
                    "qualified_transfer_destination_e164": "+33102030406",
                },
            }
        )
        if self.audio_candidate:
            workspace = UUID(self.request["workspace_id"])
            self.manifest = AgentManifestV1.model_validate(
                {
                    **self.manifest.model_dump(mode="python"),
                    "tenant_id": str(workspace),
                    "agent_id": "audio-fixture-agent",
                    "sparra": {
                        **self.manifest.sparra.model_dump(mode="python"),
                        "qualified_transfer_destination_e164": (
                            "+33102030406"
                            if self.request.get("audio_transfer_fixture") is True else None
                        ),
                        "operation_contract_version": 2,
                    },
                }
            )
        profile_fields = dict(
            schema_version=1,
            deployment_id="fixture-a",
            runtime_contract_sha256=self.settings.runtime_contract_sha256,
            image_digest=self.settings.image_digest,
            agent_bundle_sha256=self.settings.agent_bundle_sha256,
            inference_profile_sha256=self.settings.inference_profile_sha256,
            inference=inference,
            token_locator_id="telnyx-header-connected-v1",
            telnyx_api_key_sha256=hashlib.sha256(key.get_secret_value().encode()).hexdigest(),
            telnyx_handshake_fixture_sha256=hashlib.sha256(
                _redacted_fixture(token_byte_length=43, from_number=None, to_number="+33123456789")
            ).hexdigest(),
            disclosure_mark_timeout_ms=5000,
            call_lease_ttl_seconds=30,
        )
        if self.audio_candidate:
            # Explicitly unqualified native constructor; the actual graph still
            # enforces candidate mode/bindings/expiry and the one-call limit.
            profile = QualificationCandidateProfileV1.model_validate(
                {
                    **profile_fields,
                    "run_id": self.audio_run,
                    "expires_at": now() + timedelta(minutes=2),
                    "admission_not_before": datetime.now(UTC),
                    "benchmark_did_hash": self.settings.benchmark_did_sha256,
                    "max_concurrent_calls": 1,
                    "disclosure_mark_timeout_ms": 10000,
                }
            )
            self.audio_profile = profile
            self.audio_profile_sha256 = bytes.fromhex(canonical_candidate_profile_sha256(profile))
        else:
            profile = QualifiedDeploymentProfileV1.model_validate(
                {
                    **profile_fields,
                    "telnyx_data_locality": "EU",
                    "qualified_at": now() - timedelta(days=1),
                }
            )
        self.candidate_phase("profile-valid")
        material = {
            str(self.settings.telnyx_api_key_file): key,
            str(self.settings.telnyx_webhook_public_key_file): SecretStr(
                base64.b64encode(bytes(self.signing.verify_key)).decode()
            ),
            str(self.settings.openrouter_api_key_file): key,
            str(self.settings.postgres_dsn_file): SecretStr(self.request["url"]),
        }
        self.patches.enter_context(
            patch.object(
                RuntimeMetrics,
                "production",
                classmethod(self.fixture_metrics),
            )
        )
        self.patches.enter_context(
            patch.object(
                native_services,
                "DefaultAsyncHttpxClient",
                lambda **kwargs: DefaultAsyncHttpxClient(
                    transport=httpx.MockTransport(self.peers.http), **kwargs
                ),
            )
        )
        original_control_http = native_control.telnyx.DefaultAsyncHttpxClient
        self.patches.enter_context(
            patch.object(
                native_control.telnyx,
                "DefaultAsyncHttpxClient",
                lambda **kwargs: original_control_http(
                    transport=httpx.MockTransport(self.peers.http), **kwargs
                ),
            )
        )
        keyring = _decode_keyring_value(
            await asyncio.to_thread(Path(self.request["keyring_path"]).read_text)
        )
        self.candidate_phase("keyring-valid")
        self.begin_identities = []

        def native_sink(dsn):
            self.candidate_phase("sink-build")
            sink = (
                PsycopgOperationSink(dsn.get_secret_value())
                if self.audio_candidate
                else PsycopgOperationSink(dsn.get_secret_value(), pool_factory=LostBeginReplyPool)
            )
            original_begin = sink.begin_call_v2 if self.audio_candidate else sink.begin_call

            async def observed_begin(deployment, call_id, routing):
                self.candidate_phase("begin-rpc-entered")
                self.evidence.pop("begin_failure", None)
                if routing.telnyx_call_control_id == "race-original":
                    self.race_begin_entered.set()
                self.begin_identities.append((call_id, routing.model_dump_json()))
                token = _lose_begin_reply.set(
                    routing.telnyx_call_control_id == "connected-original"
                    and not self.request.get("resume_call_id")
                    and not self.request.get("recovery_case")
                )
                try:
                    try:
                        result = await original_begin(deployment, call_id, routing)
                        self.candidate_phase("begin-rpc-returned")
                    except Exception as error:
                        self.candidate_phase("begin-rpc-failed")
                        self.evidence["begin_failure"] = type(error).__name__
                        raise
                    if routing.telnyx_call_control_id == self.hold_begin:
                        self.begin_entered.set()
                        try:
                            await self.begin_release.wait()
                        finally:
                            self.begin_cancelled.set()
                    return result
                finally:
                    _lose_begin_reply.reset(token)

            if self.audio_candidate:
                sink.begin_call_v2 = observed_begin
                native_ingest = sink.ingest_v2

                async def observed_ingest(operation):
                    await native_ingest(operation)
                    if (
                        operation.kind == "call.upsert" and operation.payload.status == "active"
                        and operation.payload.disclosure_evidence is not None
                        and operation.payload.disclosure_evidence.input_gate_opened_at is not None
                    ):
                        with sqlite3.connect(self.settings.sqlite_path) as database:
                            queued = database.execute(
                                "SELECT queue_id FROM outbox WHERE op_id=?",
                                (str(operation.operation_id),),
                            ).fetchone()
                        assert queued is not None, "native_audio_active_operation_still_owned"
                        self.audio_gate_ingested[queued[0]] = (
                            operation.call_id, operation.operation_id
                        )
                    if (
                        operation.kind == "audio.chunk"
                        and operation.operation_id not in self.audio_seen
                    ):
                        assert len(self.audio_seen) < 200, "native_audio_operation_bound"
                        payload = operation.payload
                        plaintext = bytearray(
                            keyring.decrypt(
                                EncryptedValue(
                                    payload.key_version,
                                    base64.b64decode(payload.nonce_b64),
                                    base64.b64decode(payload.ciphertext_b64),
                                ),
                                aad=canonical_audio_chunk_aad(operation),
                            )
                        )
                        try:
                            assert len(plaintext) <= 32000, "native_audio_observation_bound"
                            self.audio_pcm_digest.update(plaintext)
                        finally:
                            plaintext[:] = b"\0" * len(plaintext)
                        self.audio_seen.add(operation.operation_id)

                sink.ingest_v2 = observed_ingest
            else:
                sink.begin_call = observed_begin
            self.candidate_phase("sink-built")
            return sink

        def native_control_factory(secret, _profile):
            self.candidate_phase("control-build")
            client = native_control.CallControlClient(
                api_key=secret.get_secret_value(), api_region="EU"
            )
            self.candidate_phase("control-built")
            return client

        def native_inference_factory(secret, _profile, language):
            self.candidate_phase("inference-factories-build")
            factories = RuntimeInferenceFactories(
                stt_http_client_factory=lambda: DefaultAsyncHttpxClient(
                    trust_env=False, transport=httpx.MockTransport(self.peers.http)
                ),
                stt_factory=lambda client: native_services.build_stt(
                    inference, secret, language=language, http_client=client
                ),
                llm_factory=lambda: native_services.build_llm(inference, secret),
                tts_factory=lambda: OpenRouterTTSService(
                    profile=inference,
                    api_key=secret,
                    http_client=httpx.AsyncClient(
                        trust_env=False, transport=httpx.MockTransport(self.peers.http)
                    ),
                ),
            )
            self.candidate_phase("inference-factories-built")
            return factories

        factories = RuntimeProductionFactories(
            validate_artifacts=lambda _settings: None,
            load_manifest=lambda _settings: self.manifest,
            load_profile=lambda _settings, _manifest, _time: RuntimeProfileSelection(profile, None),
            read_secret=lambda path: material[str(path)],
            load_keyring=lambda _settings: keyring,
            sink_factory=native_sink,
            call_control_factory=native_control_factory,
            inference_factory=native_inference_factory,
        )

        self.candidate_phase("graph-build")
        self.graph = await build_production_runtime(self.settings, factories=factories)
        self.candidate_phase("graph-built")
        if self.audio_candidate:
            from projetv0_voice import session as native_session
            from projetv0_voice.persistence.writer import LocalCallAdmissionFacts

            native_capture = native_session.LocalAudioCapture
            native_ack = self.graph.writer.ack_outbox

            async def observed_ack(**values):
                result = await native_ack(**values)
                if result.applied:
                    delivered = self.audio_gate_ingested.pop(values["queue_id"], None)
                    if delivered is not None:
                        call_id, operation_id = delivered
                        self.audio_gate_acked[call_id] = operation_id
                return result

            self.graph.writer.ack_outbox = observed_ack

            def observed_capture(**values):
                capture = native_capture(**values)
                self.audio_capture = capture
                return capture

            self.patches.enter_context(
                patch.object(native_session, "LocalAudioCapture", observed_capture)
            )
            native_webhook = self.graph.writer.submit_webhook

            def observed_webhook(**values):
                facts, lease = values.get("admission_facts"), values.get("lease")
                if isinstance(facts, LocalCallAdmissionFacts) and isinstance(lease, dict):
                    emit(
                        {
                            "admission_guard": {
                                "same_call_id": facts.call_id == lease.get("call_id"),
                                "generation_present": facts.admission_generation is not None,
                                "same_admitted_created": facts.admitted_at
                                == lease.get("created_at"),
                                "retention_30d": facts.retention_until
                                == facts.admitted_at + timedelta(days=30),
                            }
                        }
                    )
                return native_webhook(**values)

            self.graph.writer.submit_webhook = observed_webhook
        observe_native_startup(
            self.graph.supervisor,
            self.patches,
            self.startup_diagnostic,
            self.candidate_phase if self.audio_candidate else None,
        )
        self.captures = []
        native_enqueue = (
            self.graph.writer.try_enqueue_turn_v2
            if self.audio_candidate
            else self.graph.writer.try_enqueue_turn
        )

        def observed_enqueue(operation, **kwargs):
            self.captures.append(operation)
            return native_enqueue(operation, **kwargs)

        if self.audio_candidate:
            self.graph.writer.try_enqueue_turn_v2 = observed_enqueue
        else:
            self.graph.writer.try_enqueue_turn = observed_enqueue
        original_resolve = self.graph.registry.resolve_webhook

        async def resolve(event):
            self.candidate_phase("registry-resolve-entered")
            label = (
                event.call_control_id
                if event.call_control_id in {"human-original", "human-target"}
                else "other-owned-event"
            )
            try:
                value = await original_resolve(event)
                self.candidate_phase("registry-resolve-returned")
            except CallAdmissionRejected as error:
                self.candidate_phase("registry-resolve-rejected")
                code = (
                    error.args[0]
                    if error.args
                    and error.args[0]
                    in CallAdmissionRejected._POLICY_CODES | CallAdmissionRejected._EVENT_CODES
                    else "other-safe-rejection"
                )
                self.evidence["resolver"] = {"event": label, "code": code}
                raise
            self.evidence["resolver"] = {"event": label, "code": "resolved"}
            return value

        self.graph.registry.resolve_webhook = resolve
        if self.request.get("resume_call_id"):
            self.call_id = UUID(self.request["resume_call_id"])
            state = json.loads(await asyncio.to_thread((self.directory / "replay.json").read_text))
            self.saved_state = state
            self.captures = [VoiceOperationV1.model_validate(state["capture"])]
            self.recording_state = state["recording_state"]
        original_call_ack = self.graph.sink.ack_call_erasure
        original_recording_ack = self.graph.sink.ack_recording_purge

        async def checked_call_ack(call_id, token, occurred_at, entry_snapshot):
            # These booleans were read synchronously at callback entry. An
            # observer await must not allow cleanup to repair an earlier defect.
            for condition, satisfied in entry_snapshot.items():
                assert satisfied, condition
            entry = self.graph.registry._by_call_id.get(call_id)
            if entry is not None and entry.lease_state != "terminal":
                assert entry.routing is None and entry.begin_snapshot is None, (
                    "native_memory_scrub_before_ack"
                )
                assert entry.begin_future is None and entry.begin_task is None, (
                    "native_begin_holder_scrub_before_ack"
                )
                assert entry.lifecycle_owner is None and entry.session is None, (
                    "native_owner_scrub_before_ack"
                )
                self.evidence["bridge_cache_ack_boundary"] = {
                    "cache_absent": entry.bridge_publication is None,
                    "disclosure_absent": entry.bridge_publication is None
                    or (entry.bridge_publication.payload.disclosure_evidence is None),
                }
                assert entry.bridge_publication is None, "native_bridge_cache_scrub_before_ack"
                assert entry.transfer_facts is None or (
                    entry.transfer_facts.disclosure_evidence is None
                ), "native_bridge_facts_scrub_before_ack"
                self.holder_checks += 1
            assert (await self.graph.writer.read_retained_call(call_id)).erased, (
                "native_content_removed_before_ack"
            )
            if self.audio_candidate and self.audio_ack_hold and call_id == self.call_id:
                def local_audio_content():
                    uri = "file:" + self.settings.sqlite_path.as_posix() + "?mode=ro"
                    with sqlite3.connect(uri, uri=True) as database:
                        pending = database.execute(
                            "SELECT count(*) FROM outbox WHERE call_id=?", (str(call_id),)
                        ).fetchone()[0]
                        rows = database.execute(
                            "SELECT kind,op_id,deployment_id,fingerprint,key_version,nonce,"
                            "ciphertext,acked FROM local_audio_terminal WHERE call_id=? LIMIT 3",
                            (str(call_id),),
                        ).fetchall()
                        pin = database.execute(
                            "SELECT generation FROM local_audio_pin WHERE call_id=?",
                            (str(call_id),),
                        ).fetchone()
                    identity = self.session._identity
                    snapshot = identity.begin_snapshot
                    guard = {
                        "outbox_empty": pending == 0,
                        "terminal_present": bool(rows),
                        "terminal_bounded": len(rows) <= 2,
                        "terminal_known_acked": all(row[7] == 1 for row in rows),
                        "terminal_cipher_present": all(row[6] is not None for row in rows),
                        "terminal_metadata_only": True,
                        "terminal_identity_exact": pin is not None
                        and pin[0] == str(identity.generation.generation),
                        "retention_original_30d": snapshot.retention_until
                        == identity.routing.admitted_at + timedelta(days=30),
                    }
                    for row in rows:
                        assert len(row[6]) <= 65536, "native_audio_terminal_observation_bound"
                        plaintext = bytearray(
                            self.graph.keyring.decrypt(
                                EncryptedValue(row[4], row[5], row[6]),
                                aad=operation_aad_from_metadata(
                                    dict(
                                        schema_version=2,
                                        call_id=str(call_id),
                                        kind=row[0],
                                        operation_id=row[1],
                                        deployment_id=row[2],
                                    )
                                ),
                            )
                        )
                        try:
                            operation = decode_operation_v2(bytes(plaintext))
                            payload = operation.payload
                            guard["terminal_metadata_only"] &= isinstance(
                                payload, (AudioFinishPayloadV2, AudioRevokePayloadV2)
                            )
                            guard["terminal_identity_exact"] &= (
                                operation.call_id == call_id
                                and operation.operation_id == UUID(row[1])
                                and operation.deployment_id == self.settings.deployment_id
                                and isinstance(
                                    payload, (AudioFinishPayloadV2, AudioRevokePayloadV2)
                                )
                                and payload.workspace_id == snapshot.workspace_id
                                and payload.recording_id == snapshot.recording_id
                                and payload.configuration_revision
                                == snapshot.configuration_revision
                                and payload.retention_until == snapshot.retention_until
                                and hashlib.sha256(canonical_operation_bytes(operation)).digest()
                                == row[3]
                            )
                        finally:
                            plaintext[:] = b"\0" * len(plaintext)
                    emit({"audio_terminal_guard": guard})
                    return pending, len(rows), guard

                pending, count, terminal_guard = await asyncio.to_thread(local_audio_content)
                assert pending == 0 and 1 <= count <= 2 and all(terminal_guard.values()), (
                    "native_audio_content_removed_known_metadata_before_ack"
                )
                self.audio_ack_entered.set()
                self.candidate_phase("audio-ack-held-entered")
                await self.audio_ack_release.wait()
            if getattr(self, "erasure_turn", None) is not None and (
                call_id == self.erasure_turn.call_id
            ):
                assert await self.queued_operation(self.erasure_turn) is None, (
                    "native_erasure_queued_turn_absent_before_ack"
                )
                assert await self.graph.writer.read_frozen_call_publication(call_id) is None, (
                    "native_erasure_frozen_publication_absent_before_ack"
                )
                self.evidence["erasure_queue"]["removed_before_ack"] = True
            await original_call_ack(call_id, token, occurred_at)
            if not self.audio_candidate:
                async with await psycopg.AsyncConnection.connect(
                    self.request["url"], prepare_threshold=None
                ) as connection:
                    row = await (
                        await connection.execute(
                            "SELECT voice.ack_call_erasure_v1(%s,%s,%s) IS NULL",
                            (call_id, token, occurred_at),
                        )
                    ).fetchone()
                assert row == (True,), "native_call_ack_sql_null"
            self.local_acks.append(call_id)

        async def recording_ack(recording_id, token, outcome, occurred_at):
            if outcome in {"deleted", "not_found"}:
                assert not self.peers.recordings, "native_provider_bytes_gone_before_ack"
            await original_recording_ack(recording_id, token, outcome, occurred_at)
            async with await psycopg.AsyncConnection.connect(
                self.request["url"], prepare_threshold=None
            ) as connection:
                row = await (
                    await connection.execute(
                        "SELECT voice.ack_recording_purge_v1(%s,%s,%s,%s) IS NULL",
                        (recording_id, token, outcome, occurred_at),
                    )
                ).fetchone()
            assert row == (True,), "native_recording_ack_sql_null"
            self.remote_acks.append(recording_id)

        async def call_ack(call_id, token, occurred_at):
            entry_snapshot = {}
            if self.audio_candidate and self.audio_ack_hold and call_id == self.call_id:
                self.audio_ack_attempts += 1
                entry = self.graph.registry._by_call_id.get(call_id)
                live_entry = entry is not None and entry.lease_state != "terminal"
                capture = self.audio_capture
                # Capture and native holders are sampled before the first await.
                entry_snapshot = {
                    "native_memory_scrub_before_ack": not live_entry
                    or (entry.routing is None and entry.begin_snapshot is None),
                    "native_begin_holder_scrub_before_ack": not live_entry
                    or (entry.begin_future is None and entry.begin_task is None),
                    "native_owner_scrub_before_ack": not live_entry
                    or (entry.lifecycle_owner is None and entry.session is None),
                    "native_bridge_cache_scrub_before_ack": not live_entry
                    or entry.bridge_publication is None,
                    "native_bridge_facts_scrub_before_ack": not live_entry
                    or entry.transfer_facts is None
                    or entry.transfer_facts.disclosure_evidence is None,
                    "native_capture_owned_before_ack": capture is not None,
                    "native_capture_terminal_before_ack": capture is not None
                    and capture.tap.state in {"partial", "stopped"},
                    "native_capture_event_joined_before_ack": capture is not None
                    and not capture.tap.pending_join,
                    "native_capture_receipts_joined_before_ack": capture is not None
                    and not capture.holder.summary.pending,
                    "native_cleanup_commit_joined_before_ack": self.audio_cleanup_pending == 0
                    and self.audio_cleanup_commit_completed,
                }
            self.candidate_phase("audio-ack-callback-entered")
            try:
                await checked_call_ack(call_id, token, occurred_at, entry_snapshot)
            except Exception as error:
                if self.audio_candidate:
                    conditions = {
                        "native_memory_scrub_before_ack",
                        "native_begin_holder_scrub_before_ack",
                        "native_owner_scrub_before_ack",
                        "native_bridge_cache_scrub_before_ack",
                        "native_bridge_facts_scrub_before_ack",
                        "native_content_removed_before_ack",
                        "native_capture_owned_before_ack",
                        "native_capture_terminal_before_ack",
                        "native_capture_event_joined_before_ack",
                        "native_capture_receipts_joined_before_ack",
                        "native_cleanup_commit_joined_before_ack",
                        "native_audio_ciphertext_removed_before_ack",
                        "native_audio_content_removed_known_metadata_before_ack",
                        "native_audio_terminal_observation_bound",
                    }
                    condition = (
                        error.args[0] if isinstance(error, AssertionError) and error.args else ""
                    )
                    error_class = type(error).__name__
                    classes = {
                        "AssertionError",
                        "RuntimeError",
                        "PersistenceError",
                        "CommandSerializationError",
                        "OperationSinkContractError",
                        "OperationSinkPermanentError",
                        "TimeoutError",
                    }
                    self.audio_ack_refusal = self.audio_ack_refusal or (
                        condition if condition in conditions else "native_ack_failure"
                    )
                    emit(
                        {
                            "audio_ack_refusal": {
                                "condition": condition
                                if condition in conditions
                                else "native_ack_failure",
                                "error_class": error_class
                                if error_class in classes
                                else "OtherException",
                            }
                        }
                    )
                raise

        self.graph.sink.ack_call_erasure = call_ack
        self.graph.sink.ack_recording_purge = recording_ack
        if self.audio_candidate:
            native_leases = self.graph.sink.lease_call_erasures
            native_stop = self.graph.session_factory.erase_call_by_id
            native_erase = self.graph.writer.erase_call_content
            native_content = self.graph.writer._apply_content_command

            async def observed_content(values):
                result = await native_content(values)
                if (
                    values.get("action") in {"erase", "audio_erase_complete"}
                    and values.get("call_id") == self.call_id
                    and self.audio_erasure_lease_token is not None
                    and values.get("lease_token") == self.audio_erasure_lease_token
                    and self.audio_ack_hold
                    and isinstance(result, datetime)
                ):
                    # The native branch has mutated its actual owner transaction.
                    # Hold its real COMMIT, whether erase completed directly or
                    # native archive cleanup required audio_erase_complete.
                    connection = self.graph.writer._require_owner_connection()
                    cursor = await connection.execute(
                        "SELECT lease_token,lease_cleaned_at,lease_acked,lease_settled "
                        "FROM sparra_content_fences WHERE call_id=?",
                        (str(self.call_id),),
                    )
                    fence = await cursor.fetchone()
                    await cursor.close()
                    self.audio_cleanup_commit_fence = (
                        fence is not None
                        and fence[0] == str(self.audio_erasure_lease_token)
                        and fence[1] is not None
                        and fence[2:] == (0, 0)
                    )
                    self.audio_cleanup_commit_action = values["action"]
                    self.audio_cleanup_commit_entered.set()
                    await self.audio_cleanup_commit_release.wait()
                return result

            async def observed_leases(*args):
                try:
                    leases = await native_leases(*args)
                except Exception:
                    self.candidate_phase("audio-erase-lease-failed")
                    raise
                for lease in leases:
                    if lease.call_id == self.call_id:
                        self.audio_erasure_lease_token = lease.lease_token
                        self.candidate_phase("audio-erase-lease-acquired")
                return leases

            async def observed_stop(call_id):
                self.candidate_phase("audio-erase-stop-start")
                try:
                    await native_stop(call_id)
                    self.candidate_phase("audio-erase-stop-completed")
                except Exception:
                    self.candidate_phase("audio-erase-stop-failed")
                    raise

            async def observed_erase(*args, **kwargs):
                self.candidate_phase("audio-erase-writer-start")
                owned = (args[0] if args else kwargs.get("call_id")) == self.call_id
                if owned:
                    self.audio_cleanup_pending += 1
                try:
                    result = await native_erase(*args, **kwargs)
                    if (
                        owned
                        and kwargs.get("lease_token") == self.audio_erasure_lease_token
                        and self.audio_cleanup_commit_entered.is_set()
                    ):
                        # Native erase returns after its actual writer COMMIT.
                        self.audio_cleanup_commit_completed = True
                    self.candidate_phase("audio-erase-writer-completed")
                    return result
                except Exception:
                    self.candidate_phase("audio-erase-writer-failed")
                    raise
                finally:
                    if owned:
                        self.audio_cleanup_pending -= 1

            self.graph.sink.lease_call_erasures = observed_leases
            self.graph.session_factory.erase_call_by_id = observed_stop
            self.graph.writer.erase_call_content = observed_erase
            self.graph.writer._apply_content_command = observed_content
        self.restore_verified = False
        if self.request.get("recovery_case"):
            self.recovery_case = self.request["recovery_case"]
            self.case_state = json.loads(
                await asyncio.to_thread((self.directory / (self.recovery_case + ".json")).read_text)
            )
            self.call_id = UUID(self.case_state["call_id"])
            if self.recovery_case == "final-fix":
                self.evidence.update(self.case_state["evidence"])
            original_restore = self.graph.registry.restore_transfer_fence

            async def observed_restore(stale):
                if stale.call_id == self.call_id:
                    if self.recovery_case == "final-fix":
                        assert stale.lifecycle.content_erased, "native_live_bridge_erasure_restored"
                    else:
                        assert not stale.lifecycle.content_erased, (
                            "native_unknown_lease_not_fake_erased"
                        )
                    assert stale.lifecycle.local_closing_at is not None, (
                        "native_unknown_lease_minimal_departure"
                    )
                    assert stale.lifecycle.admission_generation == UUID(
                        self.case_state["generation"]
                    ), "native_restart_original_generation"
                await original_restore(stale)
                if stale.call_id == self.call_id:
                    entry = self.graph.registry._by_call_id[self.call_id]
                    assert entry.no_new_ai and entry.generation == UUID(
                        self.case_state["generation"]
                    ), "native_restart_no_ai_original_generation"
                    assert entry.begin_snapshot is None and entry.routing is None, (
                        "native_restart_no_content_holders"
                    )
                    assert not any(path.endswith("/hangup") for path in self.peers.actions), (
                        "native_restart_no_original_hangup"
                    )
                    self.restore_verified = True

            self.graph.registry.restore_transfer_fence = observed_restore
        self.app = create_app(self.settings, Coordinator(self.graph))
        self.lifespan = self.app.router.lifespan_context(self.app)
        self.candidate_phase("lifespan-start")
        await self.lifespan.__aenter__()
        self.candidate_phase("network-start")
        await self.start_network()
        async with await psycopg.AsyncConnection.connect(
            self.request["url"], prepare_threshold=None
        ) as connection:
            identity = await (
                await connection.execute("SELECT session_user,current_user")
            ).fetchone()
        assert identity == ("sparra_voice_a", "sparra_voice_a"), "native_session_user"
        self.evidence["checks"].append("native-session-user-pgbouncer")
        self.candidate_phase("scenario-ready")

    async def start_network(self):
        def certificate():
            key = ec.generate_private_key(ec.SECP256R1())
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
            cert = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now() - timedelta(minutes=1))
                .not_valid_after(now() + timedelta(hours=1))
                .add_extension(
                    x509.SubjectAlternativeName(
                        [
                            x509.DNSName("localhost"),
                            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                        ]
                    ),
                    critical=False,
                )
                .sign(key, hashes.SHA256())
            )
            keyfile, certfile = (
                self.directory / "owned-tls-key.pem",
                self.directory / "owned-tls-cert.pem",
            )
            keyfile.write_bytes(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            return keyfile, certfile

        keyfile, certfile = await asyncio.to_thread(certificate)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.setblocking(False)
        self.tls = ssl.create_default_context(cafile=str(certfile))
        self.base_url = f"https://127.0.0.1:{port}"
        self.media_url = f"wss://127.0.0.1:{port}"
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.observed_asgi if self.audio_candidate else self.app,
                interface="asgi3" if self.audio_candidate else "auto",
                host="127.0.0.1",
                port=port,
                lifespan="off",
                ssl_keyfile=str(keyfile),
                ssl_certfile=str(certfile),
                access_log=False,
                log_level="error",
            )
        )
        self.server_task = asyncio.create_task(self.server.serve(sockets=[listener]))
        await eventually(lambda: self.server.started, "native_owned_tls_server")
        self.http = httpx.AsyncClient(base_url=self.base_url, verify=self.tls, trust_env=False)

    async def observed_asgi(self, scope, receive, send):
        observed = scope.get("type") == "http" and scope.get("path") == "/telnyx/events"
        if observed:
            self.candidate_phase("asgi-webhook-entered")
        try:
            await self.app(scope, receive, send)
        finally:
            if observed:
                self.candidate_phase("asgi-webhook-retired")

    async def event(self, kind, control="connected-original", expected_status=200, **updates):
        payload = {
            "call_control_id": control,
            "call_leg_id": control + "-leg",
            "call_session_id": control + "-session",
            "connection_id": "connection-a",
            "to": "+33123456789",
            "from": "anonymous",
        }
        if kind == "call.initiated":
            payload.update(direction="incoming", state="parked")
        elif kind == "call.answered":
            payload.update(state="answered")
        payload.update(updates)
        body = json.dumps(
            {
                "data": {
                    "id": str(uuid4()),
                    "event_type": kind,
                    "occurred_at": now().isoformat(),
                    "payload": payload,
                }
            }
        ).encode()
        timestamp = str(int(time.time()))
        signature = base64.b64encode(
            self.signing.sign(timestamp.encode() + b"|" + body).signature
        ).decode()
        response = await self.http.post(
            "/telnyx/events",
            content=body,
            headers={"Telnyx-Signature-Ed25519": signature, "Telnyx-Timestamp": timestamp},
        )
        self.last_event = (
            body,
            {"Telnyx-Signature-Ed25519": signature, "Telnyx-Timestamp": timestamp},
        )
        assert response.status_code == expected_status, (
            "native_signed_event_"
            + kind
            + "_control_"
            + control
            + "_"
            + str(response.status_code)
            + "_drain_"
            + str(self.graph.registry._draining)
            + "_slots_"
            + str(await self.graph.registry.live_call_count())
            + "_writer_"
            + str(self.graph.writer.is_degraded)
            + "_fault_"
            + str(
                None
                if self.graph.writer.fatal_fault is None
                else self.graph.writer.fatal_fault.code
            )
            + "_begin_"
            + str(self.evidence.get("begin_failure"))
            + "_resolver_"
            + str(self.evidence.get("resolver"))
        )
        await self.graph.supervisor.webhook_finalizers.join_until_empty(
            asyncio.get_running_loop().time() + 10
        )

    async def open_call(self, control):
        self.evidence.pop("begin_failure", None)
        self.peers.stream = None
        self.control = control
        self.candidate_phase("webhook-initiated-send")
        await self.event("call.initiated", control)
        self.candidate_phase("webhook-initiated-accepted")
        snapshot = await eventually(
            lambda: self.graph.registry.snapshot(control), "native_admission"
        )
        self.call_id = snapshot.call_id
        self.candidate_phase("registry-admitted")
        await self.event("call.answered", control)
        self.candidate_phase("webhook-answered-accepted")
        return await self.attach_media(control)

    async def attach_media(self, control):
        self.candidate_phase("stream-command-wait")
        await eventually(lambda: self.peers.stream, "provider_streaming_command")
        self.candidate_phase("stream-command-seen")
        token = self.peers.stream["stream_auth_token"]
        self.media = MediaPeer(self.media_url, self.tls, control, control + "-stream")
        await self.media.open(token)
        self.candidate_phase("media-connected")
        seen_session = False
        seen_mark = False

        async def active_session():
            nonlocal seen_session, seen_mark
            if self.audio_candidate and not seen_mark and "mark" in self.media.messages:
                seen_mark = True
                self.candidate_phase("disclosure-mark-echoed")
            for entry in self.graph.registry._by_control.values():
                if entry.call_control_id != control:
                    continue
                owner = entry.lifecycle_owner
                session = None if owner is None else owner._session
                if self.audio_candidate and session is not None and not seen_session:
                    seen_session = True
                    self.candidate_phase("session-constructed")
                if (
                    session is not None
                    and session._controller is not None
                    and (
                        session._controller.is_active()
                        or self.audio_candidate
                        and session._controller.state.name == "WAITING_CHOICE"
                    )
                ):
                    return session

        self.session = await eventually(active_session, "native_disclosure_gate", 20)
        self.candidate_phase("controller-choice-ready")
        snapshot = self.session._identity.begin_snapshot
        local_audio = (
            self.audio_candidate and snapshot.audio_available
            and snapshot.recording_policy == "local_30d"
        )
        if local_audio:
            assert self.audio_capture is not None, "native_audio_capture_constructor"
            assert self.audio_capture.holder.summary.committed_samples == 0, (
                "native_audio_no_prechoice_capture"
            )
            assert "mark" in self.media.messages, "native_audio_disclosure_mark_before_choice"
            self.candidate_phase("dtmf-one-send")
            await self.media.input(
                {
                    "event": "dtmf",
                    "stream_id": self.media.stream,
                    "sequence_number": "2",
                    "occurred_at": now().isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "dtmf": {"digit": getattr(self, "audio_caller_choice", "1")},
                }
            )
            self.candidate_phase("dtmf-one-sent")
            await eventually(self.session._controller.is_active, "native_audio_caller_one_gate", 5)
            self.candidate_phase("input-gate-active")
        elif self.audio_candidate:
            assert self.audio_capture is None, "native_audio_off_has_no_capture"
            assert self.session._controller.is_active(), "native_audio_off_gate_active"
        assert self.session._identity.routing.from_e164 is None
        assert self.session._identity.begin_snapshot.knowledge.business_name == (
            "Native capture fixture" if self.audio_candidate else "Garage connecté"
        )
        assert "mark" in self.media.messages
        self.evidence["checks"].extend(
            [
                "signed-admission",
                "native-wss-handshake-pipecat",
                "owned-loopback-tls-wss",
                "actual-disclosure-mark-gate",
                "masked-caller-null",
            ]
        )
        return {
            "call_id": str(self.call_id),
            "revision": self.session._identity.begin_snapshot.configuration_revision,
        }

    async def admit(self):
        result = await self.open_call("connected-original")
        assert self.begin_identities[0] == self.begin_identities[1], (
            "native_unknown_commit_same_identity"
        )
        self.evidence["checks"].append("actual-commit-lost-reply-same-pin-before-answer")
        return result

    async def captured(self, text):
        self.peers.stt_text = text
        frames = [frame async for frame in self.session._services.stt.run_stt(b"\0\0" * 800)]
        transcript = next(frame for frame in frames if isinstance(frame, TranscriptionFrame))
        processors = self.session._active_runtime.pipeline._processors
        aggregator = next(
            processor
            for processor in processors
            if processor.__class__.__name__ == "LLMUserAggregator"
        )
        self.aggregator = aggregator
        # Controlled turn-stop timing; native aggregation produces the content/event/context.
        # Acoustic accuracy/real speaker/latency acceptance remains an operator gate.
        await aggregator.process_frame(transcript, FrameDirection.DOWNSTREAM)
        await aggregator._maybe_emit_user_turn_stopped()
        tasks = [task for _name, task in aggregator._event_tasks]
        await asyncio.gather(*tasks)
        return await self.graph.writer.read_retained_call(self.call_id)

    async def mapping(self):
        retained = await self.graph.writer.read_retained_call(self.call_id)
        ids = {turn.turn_id for turn in retained.turns}
        return {
            str(operation.payload.turn_id): operation.payload.model_dump(mode="json")
            for operation in self.captures
            if operation.payload.turn_id in ids
        }, retained.loss_count

    async def overflow(self):
        for _ in range(23):
            await self.captured("Rappelez-moi " + "a" * (16384 - 13))
        mapping, loss = await self.mapping()
        assert loss == 0
        sample = next(iter(mapping.values()))

        # Native metadata/envelope dictates candidate size, not a duplicated budget formula.
        def candidate(length):
            value = dict(
                sample,
                turn_id=str(uuid4()),
                turn_no=24,
                nonce_b64="A" * len(sample["nonce_b64"]),
                ciphertext_b64=base64.b64encode(b"\0" * (length + 16)).decode(),
                started_at=now().isoformat().replace("+00:00", "Z"),
                ended_at=now().isoformat().replace("+00:00", "Z"),
            )
            return {**mapping, value["turn_id"]: value}

        lower, upper = 1, 16384
        while lower <= upper:
            middle = (lower + upper) // 2
            if len(json.dumps(candidate(middle), ensure_ascii=False).encode()) <= 524288:
                lower = middle + 1
            else:
                upper = middle - 1
        assert upper > 0, "native_fitting_candidate"
        exact = upper
        await self.captured("a" * exact)
        mapping, loss = await self.mapping()
        assert loss == 0, "native_exact_fit_retained"
        before = dict(mapping)
        retained = await self.captured("a")
        mapping, loss = await self.mapping()
        assert loss == 1 and mapping == before, "native_one_more_dropped"
        compact_candidate = {
            **mapping,
            str(uuid4()): dict(
                sample,
                turn_id=str(uuid4()),
                turn_no=25,
                ciphertext_b64=base64.b64encode(b"\0" * 17).decode(),
                started_at=now().isoformat().replace("+00:00", "Z"),
                ended_at=now().isoformat().replace("+00:00", "Z"),
            ),
        }
        spaced = len(json.dumps(compact_candidate, ensure_ascii=False).encode())
        compact = len(
            json.dumps(compact_candidate, ensure_ascii=False, separators=(",", ":")).encode()
        )
        assert compact <= 524288 < spaced, "compact_fits_jsonb_does_not"
        assert retained.loss_count == loss and not self.graph.writer.is_degraded, (
            "native_durable_loss_and_writer_health"
        )
        async with await psycopg.AsyncConnection.connect(
            self.request["url"], prepare_threshold=None
        ) as connection:
            size = await (
                await connection.execute(
                    "SELECT octet_length(%s::jsonb::text)", [json.dumps(mapping)]
                )
            ).fetchone()
        assert size[0] == len(json.dumps(mapping, ensure_ascii=False).encode()), (
            "native_postgresql_jsonb_parity"
        )
        self.map = mapping
        self.evidence["checks"].extend(
            [
                "less-than-200-aggregate-overflow",
                "exact-fit-one-more",
                "compact-vs-jsonb-overhead",
                "native-jsonb-byte-parity",
            ]
        )
        return {
            "retained": len(mapping),
            "loss": loss,
            "map_bytes": size[0],
            "compact_bytes": compact,
            "candidate_bytes": spaced,
        }

    async def finish_replay(self):
        await self.event("call.hangup")
        await self.media.close()

        async def frozen():
            return await self.graph.writer.read_frozen_call_publication(self.call_id)

        operation = await eventually(frozen, "actual_finalizer_publication", 20)
        assert operation.payload.message_result is not None, "actual_result_inference"
        before = operation.model_dump_json()
        await self.graph.sink.ingest(operation)
        await self.graph.sink.ingest(operation)
        current = await self.graph.writer.read_frozen_call_publication(self.call_id)
        assert before == current.model_dump_json()
        assert self.peers.inferences == 1

        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_complete_fifo_delivery", 20)
        correlation = build_recording_correlation(
            self.session._identity, retention_days=30, required=False
        )
        self.recording_state = encode_recording_correlation(correlation).get_secret_value()
        state = {
            "call_id": str(self.call_id),
            "terminal_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "capture": self.captures[0].model_dump(mode="json"),
            "retained_ids": sorted(
                str(turn.turn_id)
                for turn in (await self.graph.writer.read_retained_call(self.call_id)).turns
            ),
            "recording_state": self.recording_state,
            "sqlite_ciphertext_sha256": await self.ciphertext_digest(),
        }
        await asyncio.to_thread(
            (self.directory / "replay.json").write_text, json.dumps(state), encoding="utf-8"
        )
        self.evidence["checks"].append("actual-finalizer-stable-replay")
        return {"stable": True}

    async def replay(self):
        operation = await self.graph.writer.read_frozen_call_publication(self.call_id)
        assert operation is not None, "native_restart_frozen_publication"
        assert (
            hashlib.sha256(operation.model_dump_json().encode()).hexdigest()
            == self.saved_state["terminal_sha256"]
        ), "native_crash_envelope_operation_stable"
        retained = await self.graph.writer.read_retained_call(self.call_id)
        assert (
            sorted(str(turn.turn_id) for turn in retained.turns) == self.saved_state["retained_ids"]
        ), "native_crash_retained_decision_stable"
        assert await self.ciphertext_digest() == self.saved_state["sqlite_ciphertext_sha256"], (
            "native_crash_sqlite_ciphertext_stable"
        )
        await self.graph.sink.ingest(operation)
        async with await psycopg.AsyncConnection.connect(
            self.request["url"], prepare_threshold=None
        ) as connection:
            receipt = await (
                await connection.execute(
                    "SELECT voice.ingest_operation_v1(%s::jsonb)", [operation.model_dump_json()]
                )
            ).fetchone()
        assert receipt[0]["status"] == "duplicate", "native_crash_duplicate_receipt"
        self.evidence["checks"].append("actual-process-crash-frozen-replay")
        return {"stable": True}

    async def ciphertext_digest(self):
        def inspect_owned_read_only():
            digest = hashlib.sha256()
            uri = "file:" + (self.directory / "voice.sqlite").as_posix() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as inspection:
                for table in ("sparra_turn_decisions", "sparra_publications"):
                    for row in inspection.execute(
                        "SELECT * FROM " + table + " WHERE call_id=? ORDER BY 1,2",
                        (str(self.call_id),),
                    ):
                        for value in row:
                            digest.update(
                                value if isinstance(value, bytes) else str(value).encode()
                            )
            return digest.hexdigest()

        # Fixture-only read-only inspection; no second producer connection or writer.
        await self.graph.writer.read_retained_call(self.call_id)
        return await asyncio.to_thread(inspect_owned_read_only)

    async def queued_operation(self, operation):
        def inspect_owned_read_only():
            uri = "file:" + (self.directory / "voice.sqlite").as_posix() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as inspection:
                row = inspection.execute(
                    "SELECT schema_version,op_id,deployment_id,call_id,kind,"
                    "key_version,nonce,ciphertext FROM outbox WHERE op_id=?",
                    (str(operation.operation_id),),
                ).fetchone()
            if row is None:
                return None
            metadata = dict(
                zip(
                    ("schema_version", "operation_id", "deployment_id", "call_id", "kind"),
                    row[:5],
                    strict=True,
                )
            )
            plaintext = self.graph.keyring.decrypt(
                EncryptedValue(row[5], row[6], row[7]),
                aad=operation_aad_from_metadata(metadata),
            )
            queued_bytes = canonical_operation_bytes(decode_operation(plaintext))
            assert queued_bytes == canonical_operation_bytes(operation), (
                "native_queued_operation_bytes_immutable"
            )
            return {
                "operation_sha256": hashlib.sha256(queued_bytes).hexdigest(),
                "ciphertext_sha256": hashlib.sha256(row[6] + row[7]).hexdigest(),
            }

        return await asyncio.to_thread(inspect_owned_read_only)

    async def queue_erasure_race(self):
        # The replayed original stays finalized. A distinct admitted owner produces this turn.
        await self.event("call.hangup", "connected-original")

        async def released():
            return await self.graph.registry.live_call_count() == 0

        await eventually(released, "native_replayed_original_capacity_released")
        await self.open_call("erasure-original")

        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_new_erasure_admission_delivered")
        await self.pause_delivery()
        await self.captured("Une nouvelle demande capturée sera effacée avec son enregistrement.")
        self.erasure_turn = self.captures[-1]
        assert self.erasure_turn.call_id == self.call_id, "native_new_erasure_call_identity"
        assert self.erasure_turn.operation_id != self.captures[0].operation_id, (
            "native_erasure_turn_is_new_capture"
        )
        await self.graph.writer.read_retained_call(self.call_id)
        queued = await self.queued_operation(self.erasure_turn)
        assert queued is not None, "native_erasure_turn_actual_outbox_exists"
        self.recording_state = encode_recording_correlation(
            build_recording_correlation(self.session._identity, retention_days=30, required=False)
        ).get_secret_value()
        await self.event("call.hangup", "erasure-original")
        await self.media.close()

        async def frozen():
            return await self.graph.writer.read_frozen_call_publication(self.call_id)

        publication = await eventually(frozen, "native_erasure_call_finalizer_frozen", 20)
        assert publication.payload.message_result is not None, (
            "native_erasure_actual_partial_result"
        )
        assert await self.queued_operation(self.erasure_turn) == queued, (
            "native_erasure_queued_turn_immutable_through_finalization"
        )
        assert await self.queued_operation(publication) is not None, (
            "native_erasure_actual_frozen_publication_queued"
        )
        self.evidence["erasure_queue"] = {
            **queued,
            "call_id": str(self.call_id),
            "operation_id": str(self.erasure_turn.operation_id),
            "finalizer_operation_sha256": hashlib.sha256(
                canonical_operation_bytes(publication)
            ).hexdigest(),
            "removed_before_ack": False,
        }
        self.peers.recordings["owned-erasure-recording"] = b"owned synthetic recording"
        await self.event(
            "call.recording.saved",
            "erasure-original",
            recording_id="owned-erasure-recording",
            client_state=self.recording_state,
            recording_started_at=now().isoformat(),
            recording_ended_at=now().isoformat(),
            channels="dual",
        )
        body, headers = self.last_event
        duplicate = await self.http.post("/telnyx/events", content=body, headers=headers)
        assert duplicate.status_code == 200, "native_duplicate_recording_callback"
        await self.event(
            "call.recording.saved",
            "erasure-original",
            expected_status=500,
            recording_id="owned-wrong-correlation",
            client_state=self.recording_state,
            call_session_id="foreign-session",
            recording_started_at=now().isoformat(),
            recording_ended_at=now().isoformat(),
            channels="dual",
        )
        await eventually(released, "native_new_erasure_signed_hangup_capacity_released")
        await self.event("call.initiated", "unrelated-next")
        self.next_call_id = (await self.graph.registry.snapshot("unrelated-next")).call_id
        assert await self.queued_operation(self.erasure_turn) == queued, (
            "native_erasure_turn_present_before_owner_erase"
        )
        self.evidence["checks"].extend(
            [
                "signed-recording-first-duplicate",
                "signed-recording-invalid-correlation-refused",
                "genuine-recording-identity-queued-with-erased-turn",
            ]
        )
        return {"call_id": str(self.call_id), "queue_witness": self.evidence["erasure_queue"]}

    async def cleanup(self):
        async def erased():
            facts = await self.graph.writer.read_call_lifecycle(self.call_id)
            return facts is not None and facts.content_erased

        self.delivery_release.set()
        await eventually(erased, "native_local_cleanup")
        await eventually(lambda: self.remote_acks, "native_recording_cleanup_ack")

        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_unrelated_fifo_continuation")
        assert self.call_id in self.local_acks and not self.peers.recordings, (
            "native_joined_actual_cleanup"
        )
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_cleanup_never_hangs_up"
        )
        # A late exact native producer command is refused by actual SQL, not reported duplicate.
        from projetv0_voice.persistence.postgres_sink import OperationSinkErasedError

        try:
            await self.graph.sink.ingest(self.erasure_turn)
        except OperationSinkErasedError:
            self.evidence["checks"].append("native-late-pv301-refusal")
        else:
            raise AssertionError("native_late_content_refused")
        self.evidence["checks"].append("actual-local-cleanup")
        return {
            "cleaned": True,
            "recording_ack": bool(self.remote_acks),
            "no_hangup": True,
            "ack_before_scrub": False,
            "queue_witness": self.evidence["erasure_queue"],
        }

    async def pause_delivery(self):
        self.delivery_release = asyncio.Event()
        entered = asyncio.Event()
        original_prepare = self.graph.relay._before_fifo
        if not hasattr(self, "final_fix_native_prepare"):
            self.final_fix_native_prepare = original_prepare

        async def pause():
            entered.set()
            await self.delivery_release.wait()
            await original_prepare()

        self.graph.relay._before_fifo = pause
        await asyncio.wait_for(entered.wait(), 2)

    async def prepare_inflight(self):
        await self.event("call.hangup", "unrelated-next")
        self.hold_begin = "inflight-original"
        self.inflight = asyncio.create_task(self.event("call.initiated", self.hold_begin))
        await asyncio.wait_for(self.begin_entered.wait(), 4)
        entry = self.graph.registry._by_control[self.hold_begin]
        self.call_id = entry.call_id
        assert entry.begin_task is not None and not entry.begin_task.done(), (
            "native_inflight_begin_holder"
        )
        return {"call_id": str(self.call_id)}

    async def inflight_cleanup(self):
        await eventually(
            lambda: self.call_id in self.local_acks, "native_inflight_memory_cleanup_ack"
        )
        assert self.begin_cancelled.is_set() and self.holder_checks > 0, (
            "native_begin_cancelled_and_scrubbed_before_ack"
        )
        assert not any(
            path.endswith("/inflight-original/actions/answer") for path in self.peers.actions
        ), "native_inflight_erase_no_answer"
        await asyncio.gather(self.inflight, return_exceptions=True)
        await self.event("call.hangup", self.hold_begin)
        self.evidence["checks"].append("inflight-begin-memory-holder-scrub-before-native-ack")
        return {
            "cleaned": True,
            "no_hangup": not any(path.endswith("/hangup") for path in self.peers.actions),
            "ack_before_scrub": False,
        }

    async def save_case(self, name, control):
        entry = self.graph.registry._by_control[control]
        self.call_id = entry.call_id
        state = {
            "call_id": str(entry.call_id),
            "generation": str(entry.generation),
            "control": control,
        }
        await asyncio.to_thread(
            (self.directory / (name + ".json")).write_text, json.dumps(state), encoding="utf-8"
        )
        return state

    async def prepare_held(self):
        await self.event("call.initiated", "held-original")
        state = await self.save_case("held", "held-original")
        await self.pause_delivery()
        return {"call_id": state["call_id"]}

    async def hold_lease(self):
        leases = await self.graph.sink.lease_call_erasures("task5-held", 30, 100)
        assert self.call_id in {lease.call_id for lease in leases}, (
            "native_held_lease_actually_granted"
        )
        assert not (await self.graph.writer.read_retained_call(self.call_id)).erased, (
            "native_crash_before_local_fence"
        )
        return {}

    async def prepare_backlog(self):
        await self.event("call.hangup", "held-original")
        await self.pause_delivery()
        ids = []
        from projetv0_voice.persistence.postgres_sink import OperationSinkCommitAmbiguousError

        for _ in range(100):
            call_id = uuid4()
            routing = RoutingV1(
                schema_version=1,
                direction="incoming",
                connection_id="connection-a",
                to_e164="+33123456789",
                from_e164=None,
                telnyx_call_control_id="backlog-" + str(call_id),
                telnyx_call_leg_id=None,
                telnyx_call_session_id=None,
                admitted_at=now(),
            )
            try:
                await self.graph.sink.begin_call("fixture-a", call_id, routing)
            except OperationSinkCommitAmbiguousError:
                await self.graph.sink.begin_call("fixture-a", call_id, routing)
            ids.append(str(call_id))
        await self.event("call.initiated", "backlog-original")
        state = await self.save_case("backlog", "backlog-original")
        return {"call_id": state["call_id"], "call_ids": ids}

    async def held_backlog(self):
        assert self.restore_verified, "native_real_lease_exclusion_restart_observed"
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_unknown_lease_no_hangup"
        )
        if self.recovery_case == "held":
            leases = await self.graph.sink.lease_call_erasures("task5-exclusion", 1, 100)
            assert self.call_id not in {lease.call_id for lease in leases}, (
                "native_sql_held_lease_excluded"
            )
            label = "held-native-lease"
        else:
            label = "backlog-101"
        self.evidence["checks"].append(label)
        return {"checks": [label], "no_hangup": True}

    async def prepare_claim(self):
        await self.event("call.hangup", "backlog-original")
        await self.open_call("claim-original")

        async def empty():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(empty, "native_claim_fixture_prior_delivery")
        await self.pause_delivery()
        await self.captured("Un rappel distinct pour le test de propriété de la queue.")
        target = self.captures[-1]
        entered, self.dispatch_release = asyncio.Event(), asyncio.Event()
        original_ingest = self.graph.sink.ingest
        original_erase = self.graph.writer.erase_call_content
        self.protected_claim = False
        self.rejected_dispatch = False

        async def ingest(operation):
            if operation.operation_id == target.operation_id:
                entered.set()
                await self.dispatch_release.wait()
                replaced = await self.graph.writer.read_relay_batch(
                    batch_size=1, now=now() + timedelta(seconds=40), lease_seconds=30
                )
                assert (
                    len(replaced) == 1 and replaced[0].operation.operation_id == target.operation_id
                ), "native_actual_claim_replaced"
                self.replaced_claim = replaced[0]
                try:
                    return await original_ingest(operation)
                except Exception as error:
                    from projetv0_voice.persistence.postgres_sink import OperationSinkErasedError

                    assert isinstance(error, OperationSinkErasedError), (
                        "native_actual_pv301_dispatch"
                    )
                    self.rejected_dispatch = True
                    raise
            return await original_ingest(operation)

        async def erase(*args, **kwargs):
            expected = kwargs.get("expected_item")
            result = await original_erase(*args, **kwargs)
            if expected is not None and expected.operation.operation_id == target.operation_id:
                assert expected.queue_id == self.replaced_claim.queue_id, (
                    "native_same_queue_identity"
                )
                assert expected.claim_attempt + 1 == self.replaced_claim.claim_attempt, (
                    "native_replacement_claim_attempt"
                )
                assert (
                    result is None
                    and not (await self.graph.writer.read_retained_call(self.call_id)).erased
                ), "native_production_stop_does_not_bypass_expected_claim"
                self.protected_claim = True
            return result

        self.graph.sink.ingest = ingest
        self.graph.writer.erase_call_content = erase
        self.delivery_release.set()
        await asyncio.wait_for(entered.wait(), 5)
        return {"call_id": str(self.call_id)}

    async def claim_refusal(self):
        self.dispatch_release.set()
        await eventually(lambda: self.protected_claim, "native_expected_claim_protected", 20)
        assert self.rejected_dispatch, "native_pv301_not_synthetic"
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_pv301_stop_no_original_hangup"
        )
        await self.event("call.hangup", "claim-original")
        await self.media.close()

        async def released():
            return await self.graph.registry.live_call_count() == 0

        await eventually(released, "native_original_signed_hangup_capacity_released", 20)
        self.evidence["checks"].append("real-pv301-production-stop-replaced-claim-preserved")
        return {"checks": ["expected-claim-replaced"], "no_hangup": True}

    async def human_takeover(self):
        await self.open_call("human-original")
        self.human_call_id = self.call_id
        assert self.session._identity.begin_snapshot.transfer_destination == "+33102030406", (
            "native_human_pin_has_qualified_destination"
        )
        self.peers.tool_arguments = {"to": "+33111111111"}
        await self.captured(
            "Je souhaite un interlocuteur, mais ce numéro est une donnée non fiable."
        )
        await eventually(
            lambda: any(
                message.get("role") == "tool"
                and "unavailable_collect_message" in str(message.get("content"))
                for message in self.aggregator.context.get_messages()
            ),
            "native_untrusted_tool_argument_refused",
        )
        assert self.peers.transfer is None, "native_llm_cannot_choose_destination"
        self.peers.tool_arguments = {}
        await self.captured("Pouvez-vous me passer la ligne qualifiée ?")
        await eventually(lambda: self.peers.transfer, "native_qualified_transfer_command")
        transfer = self.peers.transfer
        assert transfer["to"] == "+33102030406" and transfer["timeout_secs"] == 20, (
            "native_operator_qualified_destination_only"
        )
        assert type(transfer.get("time_limit_secs")) is int and (
            30 <= transfer["time_limit_secs"] <= 278
        ), "native_finite_answered_target_leg_limit"
        assert not self.session.no_new_ai, "native_command_ack_not_connected"
        facts = await self.graph.writer.read_call_lifecycle(self.call_id)
        assert facts.transfer_command_id == UUID(transfer["command_id"]), (
            "native_transfer_intent_committed_before_provider"
        )
        fields = {
            "to": "+33102030406",
            "from": "+33123456789",
            "call_session_id": "human-original-session",
            "client_state": transfer["target_leg_client_state"],
        }
        await self.event("call.initiated", "human-target", direction="outgoing", **fields)
        await self.event("call.answered", "human-target", **fields)
        await self.event("call.bridged", "human-target", call_leg_id="wrong-target-leg", **fields)
        assert not self.session.no_new_ai, "native_wrong_leg_not_connected"
        requests_before = self.peers.inferences
        await self.event("call.bridged", "human-target", **fields)
        await eventually(lambda: self.session.no_new_ai, "native_correlated_bridge_stops_ai")
        await self.media.close()
        retained = await self.graph.writer.read_retained_call(self.call_id)
        self.session._recorder.record_user("Late data cannot rearm capture.", now().isoformat())
        assert await self.graph.writer.read_retained_call(self.call_id) == retained, (
            "native_human_connection_stops_new_capture"
        )
        await self.session._prepare_partial_result()
        assert self.peers.inferences == requests_before, "native_no_inference_after_human"
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_human_bridge_no_original_hangup"
        )

        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_human_closing_delivery")
        self.evidence["checks"].extend(
            [
                "untrusted-tool-target-refused",
                "command-ack-wrong-leg-not-connected",
                "qualified-bridge-no-new-ai-no-inferred-end",
            ]
        )
        return {
            "call_id": str(self.call_id),
            "checks": ["qualified-bridge-no-new-ai", "untrusted-tool-target-refused"],
            "no_hangup": True,
        }

    async def final_fix_recording(self):
        await self.pause_delivery()
        facts = await self.graph.writer.read_call_lifecycle(self.human_call_id)
        assert facts.disclosure_evidence.completed_at is not None, (
            "native_actual_completed_disclosure"
        )
        self.final_fix_eligible = self.human_call_id
        self.final_fix_late = next(c for c in self.captures if c.call_id == self.human_call_id)
        self.final_fix_retention = facts.retention_until
        self.final_fix_generation = self.graph.registry._by_call_id[self.human_call_id].generation
        correlation = encode_recording_correlation(
            build_recording_correlation(self.session._identity, retention_days=30, required=False)
        ).get_secret_value()
        self.peers.recordings["owned-final-fix-recording"] = b"owned synthetic recording"
        await self.event(
            "call.recording.saved",
            "human-original",
            recording_id="owned-final-fix-recording",
            client_state=correlation,
            recording_started_at=now().isoformat(),
            recording_ended_at=now().isoformat(),
            channels="dual",
        )
        return {}

    async def final_fix_bridge_erased(self):
        try:
            await self.final_fix_native_prepare()
        finally:
            self.delivery_release.set()
        await eventually(
            lambda: self.human_call_id in self.local_acks, "native_bridge_real_local_ack", 20
        )
        await eventually(lambda: self.remote_acks, "native_bridge_real_recording_ack", 20)
        entry = self.graph.registry._by_call_id[self.human_call_id]
        assert entry.bridge_publication is None, "native_bridge_cache_absent_after_ack"
        assert entry.transfer_facts.disclosure_evidence is None, (
            "native_bridge_evidence_absent_after_ack"
        )
        assert entry.generation == self.final_fix_generation, (
            "native_bridge_original_generation_preserved"
        )
        facts = await self.graph.writer.read_call_lifecycle(self.human_call_id)
        assert facts.content_erased and facts.bridge_operation_id is not None, (
            "native_bridge_durable_minimal_fact"
        )
        assert await self.graph.registry.live_call_count() == 1, "native_live_bridge_capacity_held"
        requests = self.peers.inferences
        fields = dict(
            to="+33102030406",
            call_session_id="human-original-session",
            client_state=self.peers.transfer["target_leg_client_state"],
        )
        await self.event("call.bridged", "human-target", **fields)
        body, headers = self.last_event
        assert (
            await self.http.post("/telnyx/events", content=body, headers=headers)
        ).status_code == 200
        self.session._recorder.record_user("Late erased content", now().isoformat())
        await self.session._prepare_partial_result()
        assert self.peers.inferences == requests, "native_erased_bridge_no_new_ai"
        assert (await self.graph.writer.read_retained_call(self.human_call_id)).erased
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_erased_bridge_no_hangup"
        )
        self.evidence["checks"].extend(
            [
                "bridge-real-owner-erasure-real-acks",
                "bridge-cache-disclosure-absent-before-ack",
                "duplicate-target-no-content-no-ai",
            ]
        )
        return {"cleaned": True, "recording_ack": True, "no_hangup": True}

    async def final_fix_original_end(self):
        original_observer = self.graph.writer._record_original_end

        async def historical_missing_observer(*args):
            # Model only the pre-field metadata gap. The real signed event,
            # receipt, original terminal transition and reconciliation commit.
            return None

        self.graph.writer._record_original_end = historical_missing_observer
        try:
            await self.event("call.hangup", "human-original")
        finally:
            self.graph.writer._record_original_end = original_observer
        assert await self.graph.registry.live_call_count() == 0, (
            "native_signed_original_end_releases_capacity"
        )
        assert (
            await self.graph.writer.read_call_lifecycle(self.final_fix_eligible)
        ).original_ended_at is None, "native_historical_end_fact_missing"
        return {}

    async def final_fix_original_end_observer(self):
        call_id = self.final_fix_eligible
        await self.graph.writer.cleanup_local_state(
            now=self.final_fix_retention + timedelta(seconds=901)
        )
        assert await self.graph.writer.read_call_lifecycle(call_id) is not None, (
            "native_unknown_terminal_end_not_collected"
        )
        await self.event("call.hangup", "human-original", call_leg_id="wrong-original-leg")
        await self.event("call.hangup", "human-target", call_session_id="human-original-session")
        assert (await self.graph.writer.read_call_lifecycle(call_id)).original_ended_at is None, (
            "native_ignored_hangups_do_not_mint_original_end"
        )
        await self.event("call.hangup", "human-original")
        body, headers = self.last_event
        occurred_at = datetime.fromisoformat(json.loads(body)["data"]["occurred_at"])
        assert (await self.graph.writer.read_call_lifecycle(call_id)).original_ended_at == (
            occurred_at
        ), "native_existing_terminal_original_end_fact"
        assert (
            await self.http.post("/telnyx/events", content=body, headers=headers)
        ).status_code == 200
        assert (
            await self.graph.writer.read_call_lifecycle(call_id)
        ).original_ended_at == occurred_at
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_end_observer_never_dispatches_hangup"
        )
        self.evidence["checks"].extend(
            [
                "actual-terminal-with-historical-missing-end-fact-preserved",
                "signed-wrong-leg-target-cannot-stamp-original-end",
                "signed-matching-existing-terminal-and-duplicate-end-observed",
            ]
        )
        return {"stable": True}

    async def final_fix_prepare_unsettled(self):
        await self.open_call("unsettled-original")
        await self.captured("Une demande avec nettoyage natif encore en attente.")
        self.final_fix_unsettled_call = self.call_id
        self.final_fix_unsettled_retention = (
            await self.graph.writer.read_call_lifecycle(self.call_id)
        ).retention_until
        await self.pause_delivery()
        correlation = encode_recording_correlation(
            build_recording_correlation(self.session._identity, retention_days=30, required=False)
        ).get_secret_value()
        self.peers.recordings["owned-unsettled-recording"] = b"owned synthetic recording"
        await self.event(
            "call.recording.saved",
            self.control,
            recording_id="owned-unsettled-recording",
            client_state=correlation,
            recording_started_at=now().isoformat(),
            recording_ended_at=now().isoformat(),
            channels="dual",
        )
        await self.event("call.hangup", self.control)
        await self.media.close()
        await self.open_call("unrelated-final-fix")
        await self.captured("Une capture indépendante reste dans sa propre file.")
        self.final_fix_unrelated = self.call_id
        self.final_fix_unrelated_turn = self.captures[-1]
        assert await self.queued_operation(self.final_fix_unrelated_turn) is not None
        return {"call_id": str(self.final_fix_unsettled_call)}

    async def final_fix_unsettled(self):
        call_id = self.final_fix_unsettled_call
        facts = await self.graph.writer.read_call_lifecycle(call_id)
        clock = facts.retention_until + timedelta(seconds=901)
        ack_entered, ack_release = asyncio.Event(), asyncio.Event()
        recording_entered, recording_release = asyncio.Event(), asyncio.Event()
        original_finish = self.graph.writer.finish_erasure_ack
        original_ingest = self.graph.sink.ingest
        maintenance = None

        async def finish(call, *args, **kwargs):
            if call == call_id:
                ack_entered.set()
                await ack_release.wait()
            return await original_finish(call, *args, **kwargs)

        async def ingest(operation):
            if operation.call_id == call_id and operation.kind == "recording.upsert":
                recording_entered.set()
                await recording_release.wait()
            return await original_ingest(operation)

        self.graph.writer.finish_erasure_ack = finish
        self.graph.sink.ingest = ingest
        try:
            maintenance = asyncio.create_task(self.final_fix_native_prepare())
            await asyncio.wait_for(ack_entered.wait(), 10)
            assert call_id in self.local_acks, "native_unsettled_actual_remote_ack_received"
            await self.graph.writer.cleanup_local_state(now=clock)
            assert await self.graph.writer.read_call_lifecycle(call_id) is not None, (
                "native_gc_preserves_unsettled_ack"
            )
            queued = await self.queued_operation(self.final_fix_unrelated_turn)
            assert queued is not None, "native_gc_preserves_unrelated_queued_capture"
            ack_release.set()
            await asyncio.wait_for(recording_entered.wait(), 10)
            await self.graph.writer.cleanup_local_state(now=clock)
            assert await self.graph.writer.read_call_lifecycle(call_id) is not None, (
                "native_gc_preserves_claimed_recording_handoff"
            )
            assert await self.queued_operation(self.final_fix_unrelated_turn) == queued, (
                "native_gc_unrelated_bytes_unchanged"
            )
            recording_release.set()
            await maintenance
        finally:
            ack_release.set()
            recording_release.set()
            self.delivery_release.set()
            if maintenance is not None:
                await maintenance
            self.graph.writer.finish_erasure_ack = original_finish
            self.graph.sink.ingest = original_ingest
            self.delivery_release.set()
        await eventually(
            lambda: "owned-unsettled-recording" not in self.peers.recordings,
            "native_unsettled_real_recording_purge",
            20,
        )
        await self.event("call.hangup", "unrelated-final-fix")
        await self.media.close()
        self.evidence["checks"].extend(
            [
                "real-ack-awaiting-local-settlement-preserved",
                "real-claimed-recording-awaiting-handoff-preserved",
                "unrelated-queued-capture-bytes-preserved",
            ]
        )
        return {"stable": True}

    async def final_fix_prepare_bridge_race(self):
        await self.open_call("human-race-original")
        await self.captured("Une demande observée avant le pont et l'effacement.")
        self.peers.transfer = None
        generation = await self.graph.registry.generation_handle(self.control)
        self.final_fix_transfer = asyncio.create_task(self.graph.registry.request_human(generation))
        await eventually(lambda: self.peers.transfer, "native_race_real_transfer")
        fields = dict(
            to="+33102030406",
            call_session_id="human-race-original-session",
            client_state=self.peers.transfer["target_leg_client_state"],
        )
        await self.event("call.initiated", "human-race-target", direction="outgoing", **fields)
        await self.event("call.answered", "human-race-target", **fields)
        self.final_fix_read_entered, self.final_fix_read_release = asyncio.Event(), asyncio.Event()
        original_read = self.graph.writer.read_call_lifecycle

        async def read(call_id):
            result = await original_read(call_id)
            if call_id == self.call_id and not self.final_fix_read_entered.is_set():
                assert result.disclosure_evidence.completed_at is not None, (
                    "native_race_actual_disclosure"
                )
                self.final_fix_read_entered.set()
                await self.final_fix_read_release.wait()
            return result

        self.graph.writer.read_call_lifecycle = read
        self.final_fix_bridge = asyncio.create_task(
            self.event("call.bridged", "human-race-target", **fields)
        )
        await asyncio.wait_for(self.final_fix_read_entered.wait(), 10)
        return {"call_id": str(self.call_id)}

    async def final_fix_intent_departure(self):
        await self.open_call("intent-departure-original")
        entry = self.graph.registry._by_call_id[self.call_id]
        observed = await self.graph.writer.read_call_lifecycle(self.call_id)
        assert observed.disclosure_evidence.completed_at is not None
        original_transfer = self.graph.registry._transfer_owned
        original_commit = self.graph.writer.commit_transfer_intent
        entered, release = asyncio.Event(), asyncio.Event()
        requested = None
        self.peers.transfer = None

        async def holder_variant(owner, destination):
            # Fresh production intent facts have no evidence. This supported
            # holder-shape compatibility regression copies ONLY actual native
            # dated evidence before the dispatcher captures its reservation.
            owner.transfer_facts = replace(
                owner.transfer_facts, disclosure_evidence=observed.disclosure_evidence
            )
            return await original_transfer(owner, destination)

        async def committed_intent(facts):
            await original_commit(facts)
            entered.set()
            await release.wait()

        self.graph.registry._transfer_owned = holder_variant
        self.graph.writer.commit_transfer_intent = committed_intent
        try:
            requested = asyncio.create_task(
                self.graph.registry.request_human(
                    await self.graph.registry.generation_handle(self.control)
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            await self.media.close()
            await eventually(
                lambda: entry.lifecycle_owner is None and entry.session is None,
                "native_intent_normal_ai_departure",
                20,
            )
            release.set()
            await requested
            assert self.peers.transfer is not None, (
                "native_completed_disclosure_intent_dispatches_after_normal_departure"
            )
            assert not any(path.endswith("/hangup") for path in self.peers.actions)
            assert await self.graph.registry.live_call_count() == 1
        finally:
            release.set()
            if requested is not None:
                await requested
            self.graph.registry._transfer_owned = original_transfer
            self.graph.writer.commit_transfer_intent = original_commit
        await self.event("call.hangup", self.control)
        assert await self.graph.registry.live_call_count() == 0
        self.evidence["checks"].append(
            "actual-disclosure-holder-shape-committed-intent-normal-departure-sdk-dispatch"
        )
        return {"stable": True, "no_hangup": True}

    async def final_fix_finish_bridge_race(self):
        await eventually(
            lambda: self.call_id in self.local_acks, "native_inflight_bridge_actual_ack", 20
        )
        self.final_fix_read_release.set()
        await self.final_fix_bridge
        await self.final_fix_transfer
        await self.media.close()
        entry = self.graph.registry._by_call_id[self.call_id]
        assert entry.bridge_publication is None, "native_inflight_bridge_does_not_repopulate_cache"
        assert entry.transfer_facts.disclosure_evidence is None, (
            "native_inflight_bridge_no_disclosure_holder"
        )
        assert await self.graph.writer.read_frozen_call_publication(self.call_id) is None
        facts = await self.graph.writer.read_call_lifecycle(self.call_id)
        assert facts.content_erased and facts.qualified_line_bridged_at is not None
        assert facts.content_departed_generation == entry.generation, (
            "native_inflight_bridge_preserves_departure"
        )
        assert await self.graph.registry.live_call_count() == 1, (
            "native_inflight_bridge_keeps_live_phone"
        )
        self.final_fix_live = self.call_id
        state = dict(
            call_id=str(entry.call_id),
            generation=str(entry.generation),
            eligible_call_id=str(self.final_fix_eligible),
            retention=max(self.final_fix_retention, self.final_fix_unsettled_retention).isoformat(),
            late=self.final_fix_late.model_dump(mode="json"),
            additional_eligible=str(self.final_fix_unsettled_call),
            unrelated=str(self.final_fix_unrelated),
            evidence=self.evidence,
        )
        await asyncio.to_thread((self.directory / "final-fix.json").write_text, json.dumps(state))
        self.evidence["checks"].append("actual-bridge-read-crosses-real-owner-ack-no-resurrection")
        return {"cleaned": True, "no_hangup": True}

    async def final_fix_gc_before(self):
        await self.graph.writer.cleanup_local_state(
            now=self.final_fix_retention + timedelta(seconds=899)
        )
        assert await self.graph.writer.read_call_lifecycle(self.final_fix_eligible) is not None
        self.evidence["checks"].append("original-retention-plus-899-not-collected")
        return {"stable": True}

    async def final_fix_gc_after(self):
        eligible = UUID(self.case_state["eligible_call_id"])
        clock = datetime.fromisoformat(self.case_state["retention"]) + timedelta(seconds=901)
        from projetv0_voice.persistence.relay import maintain_call_content

        await maintain_call_content(
            self.graph.writer,
            self.graph.sink,
            self.graph.registry.stop_call_content,
            utcnow=lambda: clock,
            timeout_seconds=10,
        )
        assert await self.graph.writer.read_call_lifecycle(eligible) is None, (
            "native_terminal_metadata_finite_gc"
        )
        assert (
            await self.graph.writer.read_call_lifecycle(
                UUID(self.case_state["additional_eligible"])
            )
            is None
        ), "native_settled_recording_metadata_finite_gc"
        assert (
            await self.graph.writer.read_call_lifecycle(UUID(self.case_state["unrelated"]))
            is not None
        ), "native_gc_preserves_unknown_remote_ack"

        def inspect():
            uri = "file:" + (self.directory / "voice.sqlite").as_posix() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                return {
                    table: connection.execute(
                        "SELECT count(*) FROM " + table + " WHERE call_id=?", (str(eligible),)
                    ).fetchone()[0]
                    for table in (
                        "call_leases",
                        "sparra_content_fences",
                        "sparra_turn_decisions",
                        "sparra_publications",
                        "outbox",
                    )
                }

        counts = await asyncio.to_thread(inspect)
        assert all(value == 0 for value in counts.values()), (
            "native_gc_no_identifying_local_lifecycle"
        )
        late = VoiceOperationV1.model_validate(self.case_state["late"])
        for _ in range(2):
            assert self.graph.writer.try_enqueue_turn(late)
            assert self.graph.writer.try_enqueue_capture_loss(eligible, late.payload.turn_id)
            retained = await self.graph.writer.read_retained_call(eligible)
            assert retained.erased and not retained.turns and retained.loss_count == 0, (
                "native_gc_late_capture_fail_closed"
            )
        assert await self.graph.writer.oldest_outbox_created_at() is None
        entry = self.graph.registry._by_call_id[self.call_id]
        assert entry.generation == UUID(self.case_state["generation"])
        assert entry.bridge_publication is None and entry.no_new_ai
        assert await self.graph.registry.live_call_count() == 1, (
            "native_gc_keeps_live_human_phone_fence"
        )
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_gc_never_infers_original_end"
        )
        self.evidence["terminal_gc"] = {
            "counts": counts,
            "boundary_seconds": 901,
            "live_bridge_preserved": True,
            "late_duplicate_denied": True,
        }
        self.evidence["checks"].append(
            "restart-original-retention-plus-900-real-ack-recording-handoff-gc"
        )
        await self.event("call.hangup", "human-race-original")
        assert await self.graph.registry.live_call_count() == 0
        return {"cleaned": True, "stable": True, "no_hangup": True}

    async def human_hangup(self):
        await self.event("call.hangup", "human-original")
        return {}

    async def prepare_begin_race(self):
        self.race_admission = asyncio.create_task(self.event("call.initiated", "race-original"))
        await asyncio.wait_for(self.race_begin_entered.wait(), 3)
        assert not any("race-original/actions/answer" in path for path in self.peers.actions), (
            "native_begin_lock_no_unsourced_answer"
        )
        return {}

    async def finish_begin_race(self):
        await self.race_admission
        entry = self.graph.registry._by_control["race-original"]
        revision = entry.begin_snapshot.configuration_revision
        self.evidence["checks"].append("native-workspace-lock-before-answer-save-begin-race")
        self.control, self.call_id = "race-original", entry.call_id
        self.peers.stream = None
        await self.event("call.answered", self.control)
        await self.attach_media(self.control)
        await self.captured("Un rappel indépendant de la capture qui expirera ensuite.")

        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_unrelated_before_new_frozen_publication")
        await self.pause_delivery()
        original_clock = self.graph.writer._utcnow
        self.graph.writer._utcnow = lambda: now() + timedelta(seconds=901)
        try:
            await self.event("call.hangup", self.control)
            await self.media.close()

            async def frozen():
                return await self.graph.writer.read_frozen_call_publication(entry.call_id)

            self.expiry_unrelated = await eventually(
                frozen, "native_fresh_unrelated_finalizer_publication", 20
            )
        finally:
            self.graph.writer._utcnow = original_clock
        self.expiry_unrelated_sha = hashlib.sha256(
            self.expiry_unrelated.model_dump_json().encode()
        ).hexdigest()
        return {"revision": revision, "call_id": str(entry.call_id)}

    async def prepare_expiry(self):
        await self.open_call("expired-original")
        await self.captured("Une capture réellement passée par le service natif avant expiration.")
        return {"call_id": str(self.call_id)}

    async def expiry_cleanup(self):
        fifo_now = now() + timedelta(seconds=901)
        oldest = await self.graph.writer.oldest_outbox_created_at()
        assert (fifo_now - oldest).total_seconds() >= 901, "native_expired_fifo_age_witness"

        def unrelated_created_at():
            uri = "file:" + (self.directory / "voice.sqlite").as_posix() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as inspection:
                row = inspection.execute(
                    "SELECT created_at FROM outbox WHERE op_id=?",
                    (str(self.expiry_unrelated.operation_id),),
                ).fetchone()
            assert row is not None, "native_unrelated_frozen_operation_queued_once"
            return datetime.fromisoformat(row[0])

        unrelated_age = (fifo_now - await asyncio.to_thread(unrelated_created_at)).total_seconds()
        assert 0 <= unrelated_age <= 900, "native_unrelated_fifo_age_witness"
        self.evidence["expiry_fifo"] = {
            "expired_age_seconds": (fifo_now - oldest).total_seconds(),
            "unrelated_age_seconds": unrelated_age,
            "unrelated_operation_sha256": self.expiry_unrelated_sha,
            "scope": "relay-fifo-clock-only-readiness-unchanged",
        }
        self.graph.relay._utcnow = lambda: now() + timedelta(seconds=901)
        self.delivery_release.set()
        await eventually(
            lambda: self.call_id in self.local_acks, "native_expiry_actual_cleanup_ack", 20
        )
        assert (await self.graph.writer.read_retained_call(self.call_id)).erased, (
            "native_expiry_erased_actual_content"
        )
        assert not self.graph.relay._degraded and not self.graph.writer.is_degraded, (
            "native_expired_fifo_does_not_degrade_unrelated"
        )

        async def empty():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(empty, "native_expired_fifo_unrelated_delivered")
        frozen = await self.graph.writer.read_frozen_call_publication(self.expiry_unrelated.call_id)
        assert hashlib.sha256(frozen.model_dump_json().encode()).hexdigest() == (
            self.expiry_unrelated_sha
        ), "native_unrelated_frozen_payload_unchanged"
        await self.graph.sink.ingest(frozen)
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_expiry_no_telephone_end_inference"
        )
        await self.event("call.hangup", "expired-original")
        self.evidence["checks"].append(
            "native-expired-901s-fifo-before-age-gate-unrelated-continuation"
        )
        return {"cleaned": True, "no_hangup": True}

    async def close(self):
        self.candidate_phase("native-close-entered")
        self.audio_ack_release.set()
        self.audio_cleanup_commit_release.set()
        if hasattr(self, "final_fix_read_release"):
            self.final_fix_read_release.set()
        if hasattr(self, "final_fix_bridge"):
            await asyncio.gather(self.final_fix_bridge, return_exceptions=True)
        if self.media is not None:
            self.candidate_phase("native-close-media-start")
            await self.media.close()
            self.candidate_phase("native-close-media-joined")
        if hasattr(self, "http"):
            self.candidate_phase("native-close-http-start")
            await self.http.aclose()
            self.candidate_phase("native-close-http-joined")
        if self.server is not None:
            if self.audio_candidate:
                terminalizer = getattr(
                    getattr(self, "session", None), "_registry_terminalizer", None
                )
                owner = getattr(terminalizer, "_owner", None)
                owner_task = getattr(owner, "_task", None)
                stacks = []
                connection_states = []
                for connection in tuple(self.server.server_state.connections)[:8]:
                    protocol = type(connection).__name__
                    transport = connection.transport
                    connection_states.append(
                        {
                            "protocol": protocol
                            if protocol
                            in {
                                "H11Protocol",
                                "HttpToolsProtocol",
                                "WebSocketProtocol",
                                "WebSocketsSansIOProtocol",
                                "WSProtocol",
                            }
                            else "other",
                            "closing": transport.is_closing(),
                            "write_buffer_bytes": transport.get_write_buffer_size(),
                            "tls": transport.get_extra_info("ssl_object") is not None,
                        }
                    )
                allowed_files = {
                    "server.py",
                    "websockets_impl.py",
                    "websockets_sansio_impl.py",
                    "wsproto_impl.py",
                    "h11_impl.py",
                    "httptools_impl.py",
                    "proxy_headers.py",
                    "app.py",
                    "applications.py",
                    "routing.py",
                    "errors.py",
                    "exceptions.py",
                    "base.py",
                    "session.py",
                    "session_factory.py",
                    "handshake.py",
                    "sparra_connected_scenario.py",
                    "locks.py",
                }
                for task in tuple(self.server.server_state.tasks)[:8]:
                    coroutine = task.get_coro()
                    frames = []
                    for _ in range(16):
                        if not inspect.iscoroutine(coroutine):
                            break
                        code, frame = coroutine.cr_code, coroutine.cr_frame
                        filename = Path(code.co_filename).name
                        frames.append(
                            {
                                "file": filename if filename in allowed_files else "native-code",
                                "function": code.co_name
                                if code.co_name.isidentifier() and len(code.co_name) <= 128
                                else "native-code",
                                "line": frame.f_lineno if frame is not None else 0,
                            }
                        )
                        coroutine = coroutine.cr_await
                    stacks.append({"done": task.done(), "frames": frames})
                phase = getattr(owner, "_phase", "absent")
                emit(
                    {
                        "server_join_guard": {
                            "connections": len(self.server.server_state.connections),
                            "tasks": len(self.server.server_state.tasks),
                            "owner_present": owner is not None,
                            "owner_closed": owner is not None and owner._closed.is_set(),
                            "owner_task_done": owner_task is not None and owner_task.done(),
                            "owner_phase": phase
                            if phase
                            in {
                                "absent",
                                "gated",
                                "constructing",
                                "preactivated",
                                "finishing",
                                "done",
                            }
                            else "other",
                            "stacks": stacks,
                            "connection_states": connection_states,
                        }
                    }
                )
            self.server.should_exit = True
        if self.server_task is not None:
            self.candidate_phase("native-close-server-start")
            await asyncio.wait_for(self.server_task, 15)
            self.candidate_phase("native-close-server-joined")
        if self.lifespan is not None:
            self.candidate_phase("native-close-runtime-start")
            await self.lifespan.__aexit__(None, None, None)
            self.candidate_phase("native-close-runtime-joined")
        self.patches.close()
        self.candidate_phase("native-close-completed")
        if self.audio_candidate and self.request.get("fixture_close_failure"):
            # CLI refusal witness after the real close, not a runtime/ACK substitute.
            raise RuntimeError("native_fixture_close_failure")
        assert self.audio_ack_refusal is None, "native_audio_ack_refusal_latched"

    async def audio_admit(self):
        self.candidate_phase("audio-admit-entered")
        assert self.audio_candidate, "native_audio_candidate_mode"
        admitted = await self.open_call("audio-original")
        snapshot = self.session._identity.begin_snapshot
        assert self.graph.writer.contract_version == 2, "native_audio_fixed2_writer"
        assert self.manifest.agent_id != self.settings.deployment_id, (
            "native_audio_distinct_process_ids"
        )
        assert snapshot.audio_available and snapshot.recording_id is not None, (
            "native_audio_actual_begin_available"
        )
        self.evidence["checks"].extend(
            [
                "native-unqualified-audio-candidate",
                "native-fixed2-distinct-process-ids",
                "native-actual-begin-snapshot",
                "native-disclosure-mark-then-dtmf-one",
            ]
        )
        self.candidate_phase("audio-admit-completed")
        return {
            **admitted,
            "recording_id": str(snapshot.recording_id),
            "retention_until": snapshot.retention_until.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
        }

    async def _audio_wait_active_delivery(self):
        async def delivered():
            return (
                self.call_id in self.audio_gate_acked
                and await self.graph.writer.oldest_outbox_created_at() is None
            )

        await eventually(delivered, "native_audio_active_operation_ack_joined", 5)

    async def audio_off_admit(self):
        assert self.audio_candidate, "native_audio_candidate_mode"
        admitted = await self.open_call("audio-off-original")
        snapshot = self.session._identity.begin_snapshot
        assert snapshot.recording_policy == "off", "native_audio_owner_off_policy"
        assert not snapshot.audio_available and snapshot.recording_id is None, (
            "native_audio_off_has_no_identity"
        )
        assert self.audio_capture is None, "native_audio_off_has_no_capture"
        await self._audio_wait_active_delivery()
        self.audio_original_pin = snapshot
        return {
            **admitted,
            "recording_id": None,
            "recording_policy": snapshot.recording_policy,
            "audio_available": snapshot.audio_available,
            "retention_until": snapshot.retention_until.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
        }

    async def audio_off_replay(self):
        snapshot = self.session._identity.begin_snapshot
        replay = await self.graph.sink.begin_call_v2(
            self.settings.deployment_id, self.call_id, self.session._identity.routing
        )
        assert snapshot == self.audio_original_pin == replay, "native_audio_off_pin_immutable"
        assert self.audio_capture is None, "native_audio_off_stays_without_capture"
        for sequence in range(3, 19):
            await self.media.input({
                "event": "media", "stream_id": self.media.stream,
                "sequence_number": str(sequence),
                "media": {"payload": base64.b64encode(b"\x9e" * 800).decode(), "track": "inbound"},
            })
        await self.graph.writer.read_retained_call(self.call_id)
        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_audio_off_control_delivery", 5)
        assert not self.audio_seen, "native_audio_off_no_chunk_delivery"
        assert self.session._controller.is_active(), "native_audio_off_conversation_live"
        assert await self.graph.registry.live_call_count() == 1, "native_audio_off_phone_live"
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_audio_off_no_phone_hangup"
        )
        with sqlite3.connect(self.settings.sqlite_path) as database:
            terminals = database.execute(
                "SELECT count(*) FROM local_audio_terminal WHERE call_id=?", (str(self.call_id),)
            ).fetchone()
        assert terminals == (0,), "native_audio_off_no_terminal_operation"
        return {"original_pin": True, "capture_owned": False, "audio_chunks": 0, "phone_live": True}

    async def audio_decline_admit(self):
        assert self.audio_candidate, "native_audio_candidate_mode"
        self.audio_caller_choice = "2"
        admitted = await self.open_call("audio-declined-original")
        snapshot = self.session._identity.begin_snapshot
        assert snapshot.audio_available and snapshot.recording_id is not None, (
            "native_audio_decline_offer_available"
        )
        with sqlite3.connect(self.settings.sqlite_path) as database:
            choice = database.execute(
                "SELECT choice_state FROM local_audio_pin WHERE call_id=?", (str(self.call_id),)
            ).fetchone()
        assert choice == ("off",), "native_audio_caller_two_must_commit_off"
        assert self.audio_capture is not None, "native_audio_decline_capture_owned"
        assert self.audio_capture.holder.summary.committed_samples == 0, (
            "native_audio_decline_no_pcm"
        )
        await self._audio_wait_active_delivery()
        return {
            **admitted,
            "recording_id": str(snapshot.recording_id),
            "retention_until": snapshot.retention_until.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
        }

    async def audio_decline_check(self):
        async def delivered():
            with sqlite3.connect(self.settings.sqlite_path) as database:
                terminal = database.execute(
                    "SELECT acked FROM local_audio_terminal "
                    "WHERE call_id=? AND kind='audio.revoke'",
                    (str(self.call_id),),
                ).fetchone()
            return terminal == (1,) and await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_audio_caller_two_revoke_ack", 5)
        assert (
            not self.audio_capture.tap.pending_join
            and not self.audio_capture.holder.summary.pending
        ), "native_audio_caller_two_capture_joined"
        assert self.audio_capture.holder.summary.committed_samples == 0 and not self.audio_seen, (
            "native_audio_caller_two_no_pcm_or_chunks"
        )
        assert self.session._controller.is_active(), "native_audio_caller_two_conversation_live"
        assert await self.graph.registry.live_call_count() == 1, (
            "native_audio_caller_two_phone_live"
        )
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_audio_caller_two_no_phone_hangup"
        )
        return {"choice_off": True, "audio_chunks": 0, "phone_live": True, "capture_joined": True}

    def _audio_transfer_boundary(self):
        capture = self.audio_capture
        snapshot = self.session._identity.begin_snapshot
        summary = capture.holder.summary
        with sqlite3.connect(self.settings.sqlite_path) as database:
            row = database.execute(
                "SELECT op_id,deployment_id,fingerprint,key_version,nonce,ciphertext "
                "FROM local_audio_terminal WHERE call_id=? AND kind='audio.finish'",
                (str(self.call_id),),
            ).fetchone()
        finished = False
        if row is not None:
            plaintext = bytearray(self.graph.keyring.decrypt(
                EncryptedValue(row[3], row[4], row[5]),
                aad=operation_aad_from_metadata(dict(
                    schema_version=2, call_id=str(self.call_id), kind="audio.finish",
                    operation_id=row[0], deployment_id=row[1],
                )),
            ))
            try:
                operation = decode_operation_v2(bytes(plaintext))
                payload = operation.payload
                finished = (
                    isinstance(payload, AudioFinishPayloadV2)
                    and operation.call_id == self.call_id
                    and operation.operation_id == UUID(row[0])
                    and operation.deployment_id == self.settings.deployment_id
                    and payload.workspace_id == snapshot.workspace_id
                    and payload.recording_id == snapshot.recording_id
                    and payload.configuration_revision == snapshot.configuration_revision
                    and payload.retention_until == snapshot.retention_until
                    and payload.reason == "transfer"
                    and payload.last_sequence == summary.last_sequence
                    and payload.total_samples == summary.committed_samples
                    and hashlib.sha256(canonical_operation_bytes(operation)).digest() == row[2]
                )
            finally:
                plaintext[:] = b"\0" * len(plaintext)
        return {
            "admission_closed": capture.tap.state in {"stopped", "partial"},
            "event_joined": not capture.tap.pending_join,
            "receipt_joined": not summary.pending,
            "tail_committed": summary.committed_samples >= self.audio_transfer_before_tail + 800,
            "submitted_committed": summary.submitted_samples == summary.committed_samples,
            "finish_transfer_committed": finished,
            "original_retention": snapshot.retention_until
            == self.session._identity.routing.admitted_at + timedelta(days=30),
        }

    async def audio_transfer_boundary(self):
        assert self.request.get("audio_transfer_fixture") is True, (
            "native_transfer_fixed_fixture_opt_in"
        )
        admitted = await self.audio_admit()
        await self._audio_wait_active_delivery()
        assert self.session._identity.begin_snapshot.transfer_destination == "+33102030406", (
            "native_transfer_actual_owner_and_manifest_target"
        )
        sequence = 2
        payload = base64.b64encode(b"\x9e" * 800).decode()
        processed = asyncio.Event()
        pending_sequence = pending_frame_id = None
        native_deserialize = ProjetV0TelnyxFrameSerializer.deserialize

        async def observed_deserialize(serializer, data):
            nonlocal pending_frame_id
            frame = await native_deserialize(serializer, data)
            if not isinstance(frame, InputAudioRawFrame):
                return frame
            message = json.loads(data)
            if (
                message.get("event") == "media"
                and message.get("stream_id") == self.media.stream
                and message.get("sequence_number") == pending_sequence
                and pending_frame_id is None
            ):
                pending_frame_id = frame.id
            return frame

        def after_push(_tap, frame):
            if isinstance(frame, InputAudioRawFrame) and frame.id == pending_frame_id:
                processed.set()

        tap = self.audio_capture.tap
        tap.add_event_handler("on_after_push_frame", after_push)
        try:
            with patch.object(
                ProjetV0TelnyxFrameSerializer, "deserialize", observed_deserialize
            ):
                for sequence in range(3, 35):
                    if self.audio_seen:
                        break
                    assert self.audio_capture.tap.state == "recording", (
                        "native_transfer_capture_active_before_request"
                    )
                    processed.clear()
                    pending_sequence, pending_frame_id = str(sequence), None
                    await self.media.input({
                        "event": "media", "stream_id": self.media.stream,
                        "sequence_number": pending_sequence,
                        "media": {"payload": payload, "track": "inbound"},
                    })
                    await asyncio.wait_for(processed.wait(), 5)
                    await asyncio.sleep(0.005)
                    # Wire send is not native frame/event/SQLite receipt completion.
                    await eventually(
                        lambda: not tap.pending_join
                        and not self.audio_capture.holder.summary.pending,
                        "native_transfer_peer_event_and_receipt_joined", 5,
                    )
        finally:
            tap.remove_event_handler("on_after_push_frame", after_push)
            pending_sequence = pending_frame_id = None
            processed.clear()

        async def chunk_delivered():
            return (
                bool(self.audio_seen)
                and await self.graph.writer.oldest_outbox_created_at() is None
            )

        await eventually(chunk_delivered, "native_transfer_real_chunk_ack", 5)
        await eventually(lambda: not self.audio_capture.tap.pending_join
                         and not self.audio_capture.holder.summary.pending,
                         "native_transfer_prime_event_and_receipt_joined", 5)
        assert self.audio_capture.holder.summary.committed_samples >= 8000, (
            "native_transfer_real_committed_chunk"
        )
        self.audio_transfer_before_tail = self.audio_capture.holder.summary.committed_samples
        assert self.audio_capture.tap.state == "recording", (
            "native_transfer_capture_active_before_tail"
        )
        tail_processed = asyncio.Event()
        native_frame = self.audio_capture.tap.process_frame

        async def observed_frame(frame, direction):
            await native_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                tail_processed.set()

        before_intent = None
        sdk_entry = None
        intent_command = None
        native_intent = self.graph.writer.commit_transfer_intent

        async def observed_intent(facts):
            nonlocal before_intent, intent_command
            before_intent = self._audio_transfer_boundary()
            await native_intent(facts)
            intent_command = str(facts.transfer_command_id)

        async def observed_sdk(transfer):
            nonlocal sdk_entry
            sdk_entry = {
                **self._audio_transfer_boundary(),
                "intent_committed": intent_command == transfer["command_id"],
                "fixed_target": transfer["to"] == "+33102030406",
                "finite_target_limit": type(transfer.get("time_limit_secs")) is int
                    and 30 <= transfer["time_limit_secs"] <= 278,
            }

        with patch.object(self.audio_capture.tap, "process_frame", observed_frame), patch.object(
            self.graph.writer, "commit_transfer_intent", observed_intent
        ), patch.object(self.peers, "before_transfer", observed_sdk):
            await self.media.input({
                "event": "media", "stream_id": self.media.stream,
                "sequence_number": str(sequence + 1),
                "media": {"payload": payload, "track": "inbound"},
            })
            await asyncio.wait_for(tail_processed.wait(), 5)
            self.peers.tool_arguments = {}
            await self.captured("Pouvez-vous me passer la ligne qualifiée ?")
            await eventually(lambda: sdk_entry is not None, "native_transfer_actual_sdk_entry", 5)
            await eventually(lambda: any(
                message.get("role") == "tool" and "ringing" in str(message.get("content"))
                for message in self.aggregator.context.get_messages()
            ), "native_transfer_native_tool_result_after_sdk", 5)

        # Assert outside SDK MockTransport: an assertion there becomes a provider
        # exception and could be swallowed as outcome_unknown by native control.
        assert before_intent is not None and sdk_entry is not None, (
            "native_transfer_observations_present"
        )
        self.evidence["transfer_boundary"] = {
            "before_intent": before_intent, "sdk_entry": sdk_entry
        }
        emit({"transfer_boundary": self.evidence["transfer_boundary"]})
        assert all(before_intent.values()), "native_transfer_audio_join_before_intent"
        assert all(sdk_entry.values()), "native_transfer_audio_join_before_sdk_dispatch"
        assert await self.graph.registry.live_call_count() == 1, (
            "native_transfer_original_phone_live"
        )
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_transfer_no_original_hangup"
        )
        assert not any(path.endswith("/record_start") for path in self.peers.actions), (
            "native_transfer_no_provider_recording"
        )
        entry = self.graph.registry._by_call_id.get(self.call_id)
        assert entry is not None, "native_transfer_original_entry_present"
        self.audio_transfer_reader_entry = entry
        self.audio_transfer_reader_generation = entry.generation
        self.audio_transfer_reader_actions = tuple(self.peers.actions)
        self.evidence["checks"].append("native-transfer-real-tool-tail-joined-before-intent-and-sdk")
        return {**admitted, "capture_joined": True, "phone_live": True,
                "checks": ["native-transfer-real-tool-tail-joined-before-intent-and-sdk"]}

    async def audio_transfer_reader_live(self):
        """Observe the original native call after the App's compiled Range reads."""
        assert self.request.get("audio_transfer_fixture") is True, (
            "native_reader_transfer_fixture_opt_in"
        )
        entry = self.graph.registry._by_call_id.get(self.call_id)
        assert entry is self.audio_transfer_reader_entry, (
            "native_reader_original_registry_entry_preserved"
        )
        assert entry.generation == self.audio_transfer_reader_generation, (
            "native_reader_original_registry_generation_preserved"
        )
        assert entry.call_id == self.call_id and entry.session is self.session, (
            "native_reader_original_call_and_session_preserved"
        )
        assert not entry.capacity_released and entry.terminal_event is None, (
            "native_reader_original_phone_has_no_terminal_event"
        )
        live_call_count = await self.graph.registry.live_call_count()
        assert live_call_count == 1, "native_reader_original_phone_capacity_held"
        facts = await self.graph.writer.read_call_lifecycle(self.call_id)
        assert facts is not None and entry.transfer_facts is not None, (
            "native_reader_actual_transfer_lifecycle_present"
        )
        bridge_seen = (
            facts.qualified_line_bridged_at is not None
            or facts.bridge_operation_id is not None
            or entry.transfer_facts.qualified_line_bridged_at is not None
            or entry.transfer_facts.bridge_operation_id is not None
            or entry.bridge_publication is not None
        )
        original_end_seen = facts.original_ended_at is not None
        actions = tuple(self.peers.actions)
        result = {
            "call_id": str(self.call_id),
            "live_call_count": live_call_count,
            "answer_actions": sum(path.endswith("/actions/answer") for path in actions),
            "transfer_actions": sum(path.endswith("/actions/transfer") for path in actions),
            "hangup_actions": sum(path.endswith("/actions/hangup") for path in actions),
            "bridge_seen": bridge_seen,
            "original_end_seen": original_end_seen,
            "provider_actions_unchanged": actions == self.audio_transfer_reader_actions,
        }
        assert not bridge_seen and not original_end_seen, (
            "native_reader_does_not_infer_bridge_or_original_end"
        )
        assert result["provider_actions_unchanged"], (
            "native_reader_does_not_dispatch_new_phone_action"
        )
        self.evidence["audio_transfer_reader"] = result
        self.evidence["checks"].append("native-transfer-live-phone-after-compiled-ranges")
        return result

    async def audio_complete(self):
        return await self.audio_finish(normal_completion=True)

    async def audio_opposition_prime(self):
        admitted = await self.audio_admit()
        await self._audio_wait_active_delivery()
        self.audio_opposition_sequence = 2
        self.audio_opposition_chunk_ids = set()
        chunk_acked = asyncio.Event()
        native_ack = self.graph.writer.ack_outbox
        processed = asyncio.Event()
        pending_sequence = pending_frame_id = None
        native_deserialize = ProjetV0TelnyxFrameSerializer.deserialize

        async def observed_deserialize(serializer, data):
            nonlocal pending_frame_id
            frame = await native_deserialize(serializer, data)
            if not isinstance(frame, InputAudioRawFrame):
                return frame
            message = json.loads(data)
            if (
                message.get("event") == "media"
                and message.get("stream_id") == self.media.stream
                and message.get("sequence_number") == pending_sequence
                and pending_frame_id is None
            ):
                pending_frame_id = frame.id
            return frame

        def after_push(_tap, frame):
            if isinstance(frame, InputAudioRawFrame) and frame.id == pending_frame_id:
                processed.set()

        tap = self.audio_capture.tap

        async def observed_chunk_ack(**values):
            with sqlite3.connect(self.settings.sqlite_path) as database:
                queued = database.execute(
                    "SELECT op_id FROM outbox WHERE queue_id=? "
                    "AND kind='audio.chunk' AND call_id=?",
                    (values["queue_id"], str(self.call_id)),
                ).fetchone()
            result = await native_ack(**values)
            if result.applied and queued is not None:
                assert len(self.audio_opposition_chunk_ids) < 8, (
                    "native_opposition_ack_bound"
                )
                self.audio_opposition_chunk_ids.add(queued[0])
                chunk_acked.set()
            return result

        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        tap.add_event_handler("on_after_push_frame", after_push)
        with patch.object(self.graph.writer, "ack_outbox", observed_chunk_ack), patch.object(
            ProjetV0TelnyxFrameSerializer, "deserialize", observed_deserialize
        ):
            try:
                for sequence in range(3, 35):
                    if chunk_acked.is_set():
                        break
                    assert self.audio_capture.tap.state == "recording", (
                        "native_opposition_capture_active"
                    )
                    self.audio_opposition_sequence = sequence
                    processed.clear()
                    pending_sequence, pending_frame_id = str(sequence), None
                    await self.media.input({
                        "event": "media", "stream_id": self.media.stream,
                        "sequence_number": pending_sequence,
                        "media": {
                            "payload": base64.b64encode(b"\x9e" * 800).decode(),
                            "track": "inbound",
                        },
                    })
                    await asyncio.wait_for(processed.wait(), 5)
                    # Existing bounded synthetic wire pacing; success needs ACK.
                    await asyncio.sleep(0.005)
                    # Match audio_finish: wire send is not frame/event/receipt completion.
                    await eventually(
                        lambda: not tap.pending_join
                        and not self.audio_capture.holder.summary.pending,
                        "native_opposition_peer_event_and_receipt_joined", 5,
                    )
                await asyncio.wait_for(chunk_acked.wait(), 5)
                await eventually(
                    delivered, "native_opposition_chunk_delivery_joined", 5
                )
            finally:
                tap.remove_event_handler("on_after_push_frame", after_push)
                pending_sequence = pending_frame_id = None
                processed.clear()
                chunk_acked.clear()
        assert self.audio_opposition_chunk_ids <= {
            str(operation_id) for operation_id in self.audio_seen
        }, "native_opposition_ack_after_real_ingest"
        assert self.audio_capture.holder.summary.committed_samples >= 8000, (
            "native_opposition_at_least_one_real_chunk"
        )
        return {
            **admitted,
            "total_samples": self.audio_capture.holder.summary.committed_samples,
            "audio_chunks": len(self.audio_opposition_chunk_ids),
        }

    async def audio_opposition_revoke(self):
        self.audio_opposition_sequence += 1
        await self.media.input({
            "event": "dtmf", "stream_id": self.media.stream,
            "sequence_number": str(self.audio_opposition_sequence),
            "occurred_at": now().isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "dtmf": {"digit": "2"},
        })

        def choice_off():
            with sqlite3.connect(self.settings.sqlite_path) as database:
                choice = database.execute(
                    "SELECT choice_state FROM local_audio_pin WHERE call_id=?",
                    (str(self.call_id),),
                ).fetchone()
            return choice == ("off",)

        await eventually(choice_off, "native_opposition_caller_two_must_commit_off", 5)

        async def delivered():
            with sqlite3.connect(self.settings.sqlite_path) as database:
                terminal = database.execute(
                    "SELECT acked FROM local_audio_terminal "
                    "WHERE call_id=? AND kind='audio.revoke'", (str(self.call_id),)
                ).fetchone()
            return (
                terminal == (1,)
                and await self.graph.writer.oldest_outbox_created_at() is None
            )

        await eventually(delivered, "native_opposition_revoke_ack_joined", 5)
        assert (
            not self.audio_capture.tap.pending_join
            and not self.audio_capture.holder.summary.pending
        ), "native_opposition_capture_joined"
        assert self.session._controller.is_active(), (
            "native_opposition_conversation_live"
        )
        assert await self.graph.registry.live_call_count() == 1, (
            "native_opposition_phone_live"
        )
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_opposition_no_phone_hangup"
        )
        self.audio_opposition_closed_samples = (
            self.audio_capture.holder.summary.committed_samples
        )
        self.audio_opposition_closed_chunks = len(self.audio_seen)
        with sqlite3.connect(self.settings.sqlite_path) as database:
            self.audio_opposition_terminal = database.execute(
                "SELECT op_id,fingerprint,acked FROM local_audio_terminal "
                "WHERE call_id=? AND kind='audio.revoke'", (str(self.call_id),)
            ).fetchone()
        assert self.audio_opposition_terminal is not None, (
            "native_opposition_terminal_known"
        )
        return {"choice_off": True, "capture_joined": True, "phone_live": True}

    async def audio_opposition_late(self):
        late_sequence = self.audio_opposition_sequence + 1
        sequences = {
            str(value) for value in range(late_sequence + 1, late_sequence + 9)
        }
        decoded, processed = {}, set()
        wire_done, keypad_done = asyncio.Event(), asyncio.Event()
        native_deserialize = ProjetV0TelnyxFrameSerializer.deserialize
        native_process = self.audio_capture.tap.process_frame
        native_keypad = self.session._controller.accept_dtmf

        async def observed_deserialize(serializer, data):
            frame = await native_deserialize(serializer, data)
            message = json.loads(data)
            if (
                message.get("event") == "media"
                and message.get("stream_id") == self.media.stream
                and message.get("sequence_number") in sequences
            ):
                assert isinstance(frame, InputAudioRawFrame), (
                    "native_late_wire_real_pcm_frame"
                )
                assert len(decoded) < 8, "native_late_wire_frame_bound"
                assert frame.id not in decoded, "native_late_wire_unique_frame"
                decoded[frame.id] = message["sequence_number"]
            return frame

        async def observed_process(frame, direction):
            await native_process(frame, direction)
            if (
                isinstance(frame, InputAudioRawFrame)
                and direction is FrameDirection.DOWNSTREAM
                and frame.id in decoded
            ):
                assert frame.id not in processed, "native_late_pcm_processed_once"
                processed.add(frame.id)
                if len(processed) == 8:
                    wire_done.set()

        async def observed_keypad(frame):
            handled = await native_keypad(frame)
            if frame.button.value == "1" and frame.sequence_number == late_sequence:
                assert handled is True and frame.occurred_at is not None, (
                    "native_late_caller_one_handled"
                )
                keypad_done.set()
            return handled

        with ExitStack() as observers:
            observers.enter_context(patch.object(
                ProjetV0TelnyxFrameSerializer, "deserialize", observed_deserialize
            ))
            observers.enter_context(patch.object(
                self.audio_capture.tap, "process_frame", observed_process
            ))
            observers.enter_context(patch.object(
                self.session._controller, "accept_dtmf", observed_keypad
            ))
            try:
                await self.media.input({
                    "event": "dtmf", "stream_id": self.media.stream,
                    "sequence_number": str(late_sequence),
                    "occurred_at": now().isoformat(timespec="milliseconds").replace(
                        "+00:00", "Z"
                    ),
                    "dtmf": {"digit": "1"},
                })
                for sequence in range(late_sequence + 1, late_sequence + 9):
                    await self.media.input({
                        "event": "media", "stream_id": self.media.stream,
                        "sequence_number": str(sequence),
                        "media": {
                            "payload": base64.b64encode(b"\x8e" * 800).decode(),
                            "track": "inbound",
                        },
                    })
                await asyncio.wait_for(
                    asyncio.gather(wire_done.wait(), keypad_done.wait()), 2
                )
                assert (
                    set(decoded.values()) == sequences and processed == set(decoded)
                ), "native_late_wire_all_exact_frames_processed"
                processed_count = len(processed)
            finally:
                wire_done.clear()
                keypad_done.clear()
                decoded.clear()
                processed.clear()
        with sqlite3.connect(self.settings.sqlite_path) as database:
            choice = database.execute(
                "SELECT choice_state,denied_at IS NOT NULL FROM local_audio_pin "
                "WHERE call_id=?",
                (str(self.call_id),),
            ).fetchone()
            terminal = database.execute(
                "SELECT op_id,fingerprint,acked FROM local_audio_terminal "
                "WHERE call_id=? AND kind='audio.revoke'", (str(self.call_id),)
            ).fetchone()
        assert choice == ("off", 1), "native_late_choice_stays_off"
        assert terminal == self.audio_opposition_terminal, (
            "native_late_terminal_unchanged"
        )
        assert self.audio_capture.holder.summary.committed_samples == (
            self.audio_opposition_closed_samples
        ), "native_late_pcm_cannot_rearm_samples"
        assert len(self.audio_seen) == self.audio_opposition_closed_chunks, (
            "native_late_pcm_cannot_publish_chunks"
        )
        async def delivered():
            return await self.graph.writer.oldest_outbox_created_at() is None

        await eventually(delivered, "native_late_audio_delivery_joined", 5)
        assert (
            not self.audio_capture.tap.pending_join
            and not self.audio_capture.holder.summary.pending
        ), "native_late_audio_capture_still_joined"
        assert self.session._controller.is_active(), (
            "native_late_audio_conversation_live"
        )
        assert await self.graph.registry.live_call_count() == 1, (
            "native_late_audio_phone_live"
        )
        assert not any(path.endswith("/hangup") for path in self.peers.actions), (
            "native_late_audio_no_phone_hangup"
        )
        return {"late_dtmf_handled": True, "late_pcm_processed": processed_count,
                "choice_off": True, "capture_joined": True, "phone_live": True}

    async def audio_finish(self, *, normal_completion=False):
        self.candidate_phase("audio-finish-entered")
        assert self.audio_candidate and self.audio_capture is not None, "native_audio_capture_owned"
        # Controlled wire PCM traverses the actual serializer, STT passthrough,
        # output tap, native recorder event and one-command durable writer.
        # Synthetic audio duration is not wall-clock/carrier/latency evidence.
        # PCMU expands to s16le: 800 encoded samples become 1600 decoded
        # mono bytes, the native tap's supported per-frame bound.
        payload = base64.b64encode(b"\x9e" * 800).decode()
        processed = asyncio.Event()
        pending_sequence = pending_frame_id = None
        native_deserialize = ProjetV0TelnyxFrameSerializer.deserialize

        async def observed_deserialize(serializer, data):
            nonlocal pending_frame_id
            frame = await native_deserialize(serializer, data)
            if not isinstance(frame, InputAudioRawFrame):
                return frame
            message = json.loads(data)
            if (
                message.get("event") == "media"
                and message.get("stream_id") == self.media.stream
                and message.get("sequence_number") == pending_sequence
                and pending_frame_id is None
            ):
                pending_frame_id = frame.id
            return frame

        def after_push(_tap, frame):
            if isinstance(frame, InputAudioRawFrame) and frame.id == pending_frame_id:
                processed.set()

        tap = self.audio_capture.tap
        tap.add_event_handler("on_after_push_frame", after_push)
        try:
            with patch.object(
                ProjetV0TelnyxFrameSerializer, "deserialize", observed_deserialize
            ):
                for sequence in range(3, 803):
                    if self.audio_capture.holder.summary.committed_samples >= 512_000:
                        break
                    assert tap.state == "recording", "native_audio_capture_not_refused"
                    processed.clear()
                    pending_sequence, pending_frame_id = str(sequence), None
                    await self.media.input(
                        {
                            "event": "media",
                            "stream_id": self.media.stream,
                            "sequence_number": pending_sequence,
                            "media": {"payload": payload, "track": "inbound"},
                        }
                    )
                    await asyncio.wait_for(processed.wait(), 5)
                    # A wire send is not native frame/event/receipt completion.
                    # Keep the synthetic yield and let the actual owner join its
                    # native event and exact SQLite receipt before the next send.
                    await asyncio.sleep(0.005)
                    await eventually(
                        lambda: not tap.pending_join
                        and not self.audio_capture.holder.summary.pending,
                        "native_audio_peer_event_and_receipt_joined",
                        5,
                    )
        finally:
            tap.remove_event_handler("on_after_push_frame", after_push)
            pending_sequence = pending_frame_id = None
            processed.clear()
        assert self.audio_capture.holder.summary.committed_samples >= 512_000, (
            "native_audio_actual_sample_threshold"
        )
        if normal_completion:
            # Pipecat's public graceful stop queues EndFrame; CallSession owns
            # controller/capture completion and the native terminal publication.
            # Do not inject a provider hangup or call capture.finish ourselves.
            worker = self.session._active_runtime.worker
            owner = self.session._registry_terminalizer._owner
            assert owner is not None and owner._task is not None, "native_audio_normal_owner"
            assert not worker.has_finished(), "native_audio_normal_worker_active"
            assert not any(path.endswith("/hangup") for path in self.peers.actions), (
                "native_audio_normal_no_prior_hangup"
            )
            await worker.stop_when_done()
            await eventually(
                lambda: owner._closed.is_set() and owner._task.done(),
                "native_audio_normal_owner_joined",
                5,
            )
            assert worker.has_finished(), "native_audio_normal_worker_joined"
            assert self.session._terminal_outcome.reason == "closed", (
                "native_audio_normal_session_closed"
            )
            assert not self.audio_capture.holder.summary.partial, "native_audio_normal_not_partial"
            assert self.audio_capture.holder.summary.reason == "complete", (
                "native_audio_normal_capture_complete"
            )
        else:
            await self.event("call.hangup", "audio-original")
            self.candidate_phase("audio-hangup-accepted")
        self.candidate_phase("audio-media-close-start")
        await self.media.close()
        self.candidate_phase("audio-media-close-completed")
        capture_state = self.audio_capture.tap.state
        assert capture_state in {"off", "recording", "partial", "stopped"}, (
            "native_audio_capture_state_known"
        )
        self.candidate_phase("audio-capture-state-" + capture_state)
        self.candidate_phase(
            "audio-capture-event-pending"
            if self.audio_capture.tap.pending_join
            else "audio-capture-event-joined"
        )
        self.candidate_phase(
            "audio-capture-receipt-pending"
            if self.audio_capture.holder.summary.pending
            else "audio-capture-receipt-joined"
        )
        # A provider hangup drains the actual session conservatively. Its native
        # partial state remains partial after the event and receipt are joined.
        # The separate normal action requires graceful, non-partial completion.
        await eventually(
            lambda: (
                (
                    self.audio_capture.tap.state == "stopped"
                    if normal_completion
                    else self.audio_capture.tap.state in {"partial", "stopped"}
                )
                and not self.audio_capture.tap.pending_join
                and not self.audio_capture.holder.summary.pending
            ),
            "native_audio_normal_capture_joined"
            if normal_completion
            else "native_audio_partial_capture_joined",
            5,
        )
        self.candidate_phase("audio-capture-stopped")
        terminal_observation = None

        async def finished_delivered():
            nonlocal terminal_observation
            with sqlite3.connect(self.settings.sqlite_path) as database:
                row = database.execute(
                    "SELECT acked FROM local_audio_terminal "
                    "WHERE call_id=? AND kind='audio.finish'",
                    (str(self.call_id),),
                ).fetchone()
            observation = "absent" if row is None else "acked" if row[0] == 1 else "pending"
            if observation != terminal_observation:
                terminal_observation = observation
                self.candidate_phase("audio-terminal-" + observation)
            empty = await self.graph.writer.oldest_outbox_created_at() is None
            return row is not None and row[0] == 1 and empty

        await eventually(finished_delivered, "native_audio_actual_terminal_delivery", 6)
        summary = self.audio_capture.holder.summary
        assert not summary.pending and not self.audio_capture.tap.pending_join, (
            "native_audio_capture_joined"
        )
        assert self.audio_seen, "native_audio_actual_chunks_ingested"
        self.evidence["checks"].append(
            "native-normal-endframe-capture-real-pgbouncer-delivery"
            if normal_completion
            else "native-partial-hangup-capture-real-pgbouncer-delivery"
        )
        self.evidence["capture"] = {
            "committed_samples": summary.committed_samples,
            "last_sequence": summary.last_sequence,
            "operations": len(self.audio_seen),
            "scope": "synthetic-wire-media-normal-endframe"
            if normal_completion
            else "synthetic-wire-media-partial-provider-hangup",
        }
        self.candidate_phase("audio-finish-completed")
        return {
            "call_id": str(self.call_id),
            "total_samples": summary.committed_samples,
            "pcm_sha256": self.audio_pcm_digest.hexdigest(),
        }

    async def audio_hold_ack(self):
        assert self.audio_candidate, "native_audio_candidate_mode"
        self.audio_ack_hold = True
        return {"ack_held": True}

    async def audio_erasure_held(self):
        assert self.audio_ack_hold, "native_audio_erasure_hold_armed"
        await asyncio.wait_for(self.audio_cleanup_commit_entered.wait(), 6)
        assert self.audio_cleanup_pending > 0 and not self.audio_cleanup_commit_completed, (
            "native_audio_cleanup_completion_still_held"
        )
        assert self.audio_cleanup_commit_fence is True, (
            "native_audio_exact_owner_fence_mutation_before_commit"
        )
        assert self.audio_ack_attempts == 0 and self.call_id not in self.local_acks, (
            "native_audio_no_callback_before_cleanup_commit"
        )

        def completion_fence():
            uri = "file:" + self.settings.sqlite_path.as_posix() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as database:
                return database.execute(
                    "SELECT lease_token,lease_cleaned_at,lease_acked,lease_settled "
                    "FROM sparra_content_fences WHERE call_id=?",
                    (str(self.call_id),),
                ).fetchone()

        fence = await asyncio.to_thread(completion_fence)
        assert (
            self.audio_erasure_lease_token is not None
            and (
                fence is None
                or fence[0] != str(self.audio_erasure_lease_token)
                or fence[1] is None
            )
        ), (
            "native_audio_durable_cleanup_commit_not_completed"
        )
        assert self.audio_ack_attempts == 0 and self.audio_ack_refusal is None, (
            "native_audio_no_callback_while_cleanup_commit_held"
        )
        self.audio_cleanup_commit_release.set()
        await asyncio.wait_for(self.audio_ack_entered.wait(), 6)
        assert self.audio_ack_refusal is None, "native_audio_ack_refusal_latched"
        assert self.call_id not in self.local_acks, "native_audio_no_optimistic_ack"
        assert (await self.graph.writer.read_retained_call(self.call_id)).erased, (
            "native_audio_local_erasure_before_ack"
        )
        assert any(
            call == self.call_id
            for call, _token, _at in await self.graph.writer.pending_erasure_acks()
        ), "native_audio_exact_pending_ack_durable"
        return {
            "writer_cleaned": True,
            "ack_held": True,
            "cleanup_commit_held": True,
            "cleanup_command": self.audio_cleanup_commit_action,
        }

    async def audio_release_ack(self):
        assert self.audio_ack_refusal is None, "native_audio_ack_refusal_latched"
        self.audio_ack_release.set()
        await eventually(
            lambda: self.call_id in self.local_acks, "native_audio_real_erasure_ack", 5
        )

        async def ack_joined():
            return not any(
                call == self.call_id
                for call, _token, _at in await self.graph.writer.pending_erasure_acks()
            )

        await eventually(ack_joined, "native_audio_pending_ack_joined", 5)
        assert self.audio_ack_refusal is None, "native_audio_ack_refusal_latched"
        assert not self.peers.recordings, "native_audio_no_provider_archive"
        self.evidence["checks"].append("native-capture-erasure-real-ack-joined")
        return {"cleaned": True, "ack_before_scrub": False}


def candidate_consumption_receipt(path, run_id):
    uri = "file:" + path.as_posix() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as database:
        return database.execute(
            "SELECT run_id,consumed_at,profile_sha256,total_calls,used_calls "
            "FROM qualification_runs WHERE run_id=?",
            (str(run_id),),
        ).fetchone()


async def connected(request):
    directory = Path(request["state_path"])
    assert directory.parent.resolve() == Path(request["keyring_path"]).parent.resolve()
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    scenario = Scenario(request, directory)
    stopped = False
    try:
        await scenario.setup()
        emit({"ready": True, **({"candidate": True} if scenario.audio_candidate else {})})
        while line := await asyncio.to_thread(sys.stdin.readline):
            command = json.loads(line)
            scenario.phase = command["action"]
            if scenario.phase == "stop":
                if scenario.audio_candidate and command.get("fixture_close_failure"):
                    scenario.request["fixture_close_failure"] = True
                stopped = True
                break
            if scenario.phase == "audio-next-candidate":
                # Reconstruct only the owned fixture, after its actual close.
                # Native one-use flags and the consumed SQLite row stay intact.
                assert scenario.audio_candidate, "native_next_candidate_mode"
                assert (
                    scenario.audio_original_pin.recording_policy == "off"
                    and scenario.audio_original_pin.call_id == scenario.call_id
                ), "native_next_candidate_original_off_pin"
                previous_run = scenario.audio_run
                previous_profile = scenario.audio_profile
                previous_profile_sha256 = scenario.audio_profile_sha256
                assert await scenario.graph.writer.qualification_run_consumed(
                    previous_run, total_calls=previous_profile.total_calls,
                    profile_sha256=previous_profile_sha256,
                ), "native_next_candidate_previous_consumed"
                sqlite_path = scenario.settings.sqlite_path
                previous_receipt = await asyncio.to_thread(
                    candidate_consumption_receipt, sqlite_path, previous_run
                )
                assert previous_receipt is not None, "native_next_candidate_receipt_present"
                await scenario.close()
                previous_closed = (
                    scenario.graph.supervisor._closed
                    and scenario.graph.supervisor._writer_task.done()
                    and scenario.graph.writer._closed_event.is_set()
                    and scenario.graph.writer.fatal_fault is None
                    and scenario.app.state.runtime_graph is None
                    and scenario.server_task.done()
                    and not scenario.server.server_state.connections
                    and not scenario.server.server_state.tasks
                    and scenario.http.is_closed
                    and scenario.media.task.done()
                )
                assert previous_closed, "native_next_candidate_previous_joined"
                # Assign before setup: failure/EOF closes the new actual owner.
                scenario = Scenario(dict(request), directory)
                await scenario.setup()
                assert scenario.audio_run != previous_run, "native_next_candidate_fresh_run"
                previous_consumed = await scenario.graph.writer.qualification_run_consumed(
                    previous_run, total_calls=previous_profile.total_calls,
                    profile_sha256=previous_profile_sha256,
                )
                current_consumed = await scenario.graph.writer.qualification_run_consumed(
                    scenario.audio_run, total_calls=scenario.audio_profile.total_calls,
                    profile_sha256=scenario.audio_profile_sha256,
                )
                previous_run_preserved = previous_receipt == await asyncio.to_thread(
                    candidate_consumption_receipt, scenario.settings.sqlite_path, previous_run
                )
                assert (
                    previous_consumed and not current_consumed and previous_run_preserved
                ), "native_next_candidate_consumption_preserved"
                result = {
                    "previous_run_id": str(previous_run),
                    "run_id": str(scenario.audio_run),
                    "previous_consumed": previous_consumed,
                    "current_consumed": current_consumed,
                    "previous_run_preserved": previous_run_preserved,
                    "previous_closed": previous_closed,
                }
            else:
                result = await getattr(scenario, scenario.phase.replace("-", "_"))()
            await asyncio.to_thread(
                Path(request["evidence_path"], "native.json").write_text,
                json.dumps(scenario.evidence, indent=2),
                encoding="utf-8",
            )
            emit(result)
    except Exception as error:
        # Never serialize exception arguments/requests/credentials; source location only.
        scenario.evidence["startup_diagnostic"] = safe_startup_diagnostic(
            error, scenario.startup_diagnostic, scenario.request.get("recovery_case")
        )
        if scenario.graph is not None:
            entry = scenario.graph.registry._by_call_id.get(scenario.call_id)
            scenario.evidence["failure_state"] = {
                "writer_fault": None
                if scenario.graph.writer.fatal_fault is None
                else scenario.graph.writer.fatal_fault.code,
                "registry_draining": scenario.graph.registry._draining,
                "relay_degraded": scenario.graph.relay._degraded,
                "entry_present": entry is not None,
                "content_stop_started": entry is not None and entry.content_stop_task is not None,
                "content_stop_done": entry is not None
                and entry.content_stop_task is not None
                and entry.content_stop_task.done(),
                "cache_present": entry is not None and entry.bridge_publication is not None,
            }
        await asyncio.to_thread(
            Path(request["evidence_path"], "native.json").write_text,
            json.dumps(scenario.evidence, indent=2),
            encoding="utf-8",
        )
        frame = traceback.extract_tb(error.__traceback__)[-1]
        label = (
            error.args[0]
            if isinstance(error, AssertionError)
            and error.args
            and isinstance(error.args[0], str)
            and error.args[0].startswith("native_")
            else ""
        )
        emit(
            {
                "error": type(error).__name__,
                "where": f"{scenario.phase}:{frame.name}:{frame.lineno}:{label}",
            }
        )
    finally:
        await scenario.close()
    if stopped:
        emit({"stopped": True})
