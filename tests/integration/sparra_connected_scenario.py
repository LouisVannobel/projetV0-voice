"""Owned connected fixture. Native graph/SDKs/ASGI/Pipecat/SQLite/SQL, controlled peers only."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import socket
import sqlite3
import ssl
import sys
import time
import traceback
from contextlib import ExitStack, asynccontextmanager
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
from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr
from websockets.asyncio.client import connect

from projetv0_voice.admission import CallAdmissionRejected
from projetv0_voice.app import create_app
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
    operation_aad_from_metadata,
)
from projetv0_voice.persistence.postgres_sink import PsycopgOperationSink
from projetv0_voice.production_wiring import _decode_keyring_value
from projetv0_voice.qualified_profile import (
    InferenceProfileV1,
    QualifiedDeploymentProfileV1,
    canonical_inference_profile_sha256,
)
from projetv0_voice.runtime_config import capture_runtime_environment, parse_runtime_settings
from projetv0_voice.telnyx import call_control as native_control
from projetv0_voice.telnyx.handshake import _redacted_fixture
from projetv0_voice.telnyx.recordings import (
    build_recording_correlation,
    encode_recording_correlation,
)

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


def observe_native_startup(supervisor, patches, diagnostic):
    """Test-only observation; the original owns scheduling, budgets and exceptions."""
    original = supervisor._startup_await
    phases = {
        "writer_startup_failed", "writer_quick_check_failed", "qualification_status_failed",
        "operation_sink_open_failed", "stale_recovery_failed", "runtime_publication_failed",
        "runtime_begin_drain_failed",
    }

    async def observed(awaitable, *, code):
        diagnostic["last_phase"] = (
            code if type(code) is str and code in phases else "other_safe_failure"
        )
        diagnostic["outcome"] = "other"
        try:
            return await original(awaitable, code=code)
        except BaseException:
            diagnostic["outcome"] = (
                "native_timeout" if supervisor._startup_phase_timed_out is True
                else "native_exception"
            )
            raise

    patches.enter_context(patch.object(supervisor, "_startup_await", observed))


def safe_startup_diagnostic(error, diagnostic, recovery_case):
    """Closed public fields only: never stringify exceptions, locals or source lines."""
    terminal_codes = {
        "runtime_startup_invalid", "stale_recovery_failed", "qualification_run_consumed",
        "writer_startup_failed", "writer_quick_check_failed", "owned_task_registration_failed",
        "runtime_startup_failed",
    }
    phases = {
        "writer_startup_failed", "writer_quick_check_failed", "qualification_status_failed",
        "operation_sink_open_failed", "stale_recovery_failed", "runtime_publication_failed",
        "runtime_begin_drain_failed",
    }
    terminal_code = "other_safe_failure"
    if type(error) is RuntimeError and len(error.args) == 1:
        argument = error.args[0]
        if type(argument) is str and argument in terminal_codes:
            terminal_code = argument
    phase = diagnostic.get("last_phase")
    outcome = diagnostic.get("outcome")
    classes = {RuntimeError: "RuntimeError", AssertionError: "AssertionError",
               TimeoutError: "TimeoutError", asyncio.CancelledError: "CancelledError"}
    frame_names = {
        "setup", "startup", "_startup_await", "_recover_stale_leases", "prepare_before_fifo",
        "maintain_call_content", "observed_restore", "restore_transfer_fence",
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
        "outcome": outcome if type(outcome) is str
        and outcome in {"native_timeout", "native_exception"} else "other",
        "recovery_case": recovery_case if type(recovery_case) is str
        and recovery_case in {"held", "backlog", "final-fix", "replay"} else "other",
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

    async def http(self, request):
        path = request.url.path
        if request.url.host == "api.telnyx.eu":
            if path.endswith("/actions/streaming_start"):
                self.stream = json.loads(request.content)
            if path.endswith("/actions/transfer"):
                self.transfer = json.loads(request.content)
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

    async def setup(self):
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
        parsed_settings = parse_runtime_settings(
            capture_runtime_environment(env), geteuid=lambda: 10001, getegid=lambda: 10001
        )
        self.settings = replace(
            parsed_settings,
            sqlite_path=PurePosixPath((self.directory / "voice.sqlite").as_posix()),
        )
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
        profile = QualifiedDeploymentProfileV1(
            schema_version=1,
            deployment_id="fixture-a",
            runtime_contract_sha256=self.settings.runtime_contract_sha256,
            image_digest=self.settings.image_digest,
            agent_bundle_sha256=self.settings.agent_bundle_sha256,
            inference_profile_sha256=self.settings.inference_profile_sha256,
            inference=inference,
            token_locator_id="telnyx-header-connected-v1",
            telnyx_api_key_sha256=hashlib.sha256(key.get_secret_value().encode()).hexdigest(),
            telnyx_data_locality="EU",
            telnyx_handshake_fixture_sha256=hashlib.sha256(
                _redacted_fixture(token_byte_length=43, from_number=None, to_number="+33123456789")
            ).hexdigest(),
            disclosure_mark_timeout_ms=5000,
            call_lease_ttl_seconds=30,
            qualified_at=now() - timedelta(days=1),
        )
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
                classmethod(lambda _cls, _token, **_kwargs: RuntimeMetrics.in_memory()),
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
        self.begin_identities = []

        def native_sink(dsn):
            sink = PsycopgOperationSink(dsn.get_secret_value(), pool_factory=LostBeginReplyPool)
            original_begin = sink.begin_call

            async def observed_begin(deployment, call_id, routing):
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
                    except Exception as error:
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

            sink.begin_call = observed_begin
            return sink

        factories = RuntimeProductionFactories(
            validate_artifacts=lambda _settings: None,
            load_manifest=lambda _settings: self.manifest,
            load_profile=lambda _settings, _manifest, _time: RuntimeProfileSelection(profile, None),
            read_secret=lambda path: material[str(path)],
            load_keyring=lambda _settings: keyring,
            sink_factory=native_sink,
            call_control_factory=lambda secret, _profile: native_control.CallControlClient(
                api_key=secret.get_secret_value(), api_region="EU"
            ),
            inference_factory=lambda secret, _profile, language: RuntimeInferenceFactories(
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
            ),
        )

        self.graph = await build_production_runtime(self.settings, factories=factories)
        observe_native_startup(self.graph.supervisor, self.patches, self.startup_diagnostic)
        self.captures = []
        native_enqueue = self.graph.writer.try_enqueue_turn

        def observed_enqueue(operation, **kwargs):
            self.captures.append(operation)
            return native_enqueue(operation, **kwargs)

        self.graph.writer.try_enqueue_turn = observed_enqueue
        original_resolve = self.graph.registry.resolve_webhook

        async def resolve(event):
            label = (
                event.call_control_id
                if event.call_control_id in {"human-original", "human-target"}
                else "other-owned-event"
            )
            try:
                value = await original_resolve(event)
            except CallAdmissionRejected as error:
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

        async def call_ack(call_id, token, occurred_at):
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
                    "disclosure_absent": entry.bridge_publication is None or (
                        entry.bridge_publication.payload.disclosure_evidence is None
                    ),
                }
                assert entry.bridge_publication is None, "native_bridge_cache_scrub_before_ack"
                assert entry.transfer_facts is None or (
                    entry.transfer_facts.disclosure_evidence is None
                ), "native_bridge_facts_scrub_before_ack"
                self.holder_checks += 1
            assert (await self.graph.writer.read_retained_call(call_id)).erased, (
                "native_content_removed_before_ack"
            )
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

        self.graph.sink.ack_call_erasure = call_ack
        self.graph.sink.ack_recording_purge = recording_ack
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
        await self.lifespan.__aenter__()
        await self.start_network()
        async with await psycopg.AsyncConnection.connect(
            self.request["url"], prepare_threshold=None
        ) as connection:
            identity = await (
                await connection.execute("SELECT session_user,current_user")
            ).fetchone()
        assert identity == ("sparra_voice_a", "sparra_voice_a"), "native_session_user"
        self.evidence["checks"].append("native-session-user-pgbouncer")

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
                self.app,
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
        await self.event("call.initiated", control)
        snapshot = await eventually(
            lambda: self.graph.registry.snapshot(control), "native_admission"
        )
        self.call_id = snapshot.call_id
        await self.event("call.answered", control)
        return await self.attach_media(control)

    async def attach_media(self, control):
        await eventually(lambda: self.peers.stream, "provider_streaming_command")
        token = self.peers.stream["stream_auth_token"]
        self.media = MediaPeer(self.media_url, self.tls, control, control + "-stream")
        await self.media.open(token)

        async def active_session():
            for entry in self.graph.registry._by_control.values():
                if entry.call_control_id != control:
                    continue
                owner = entry.lifecycle_owner
                session = None if owner is None else owner._session
                if (
                    session is not None
                    and session._controller is not None
                    and session._controller.is_active()
                ):
                    return session

        self.session = await eventually(active_session, "native_disclosure_gate", 20)
        assert self.session._identity.routing.from_e164 is None
        assert self.session._identity.begin_snapshot.knowledge.business_name == "Garage connecté"
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
            retention=max(
                self.final_fix_retention, self.final_fix_unsettled_retention
            ).isoformat(),
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
        if hasattr(self, "final_fix_read_release"):
            self.final_fix_read_release.set()
        if hasattr(self, "final_fix_bridge"):
            await asyncio.gather(self.final_fix_bridge, return_exceptions=True)
        if self.media is not None:
            await self.media.close()
        if self.server is not None:
            self.server.should_exit = True
        if self.server_task is not None:
            await asyncio.wait_for(self.server_task, 15)
        if self.lifespan is not None:
            await self.lifespan.__aexit__(None, None, None)
        if hasattr(self, "http"):
            await self.http.aclose()
        self.patches.close()


async def connected(request):
    directory = Path(request["state_path"])
    assert directory.parent.resolve() == Path(request["keyring_path"]).parent.resolve()
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    scenario = Scenario(request, directory)
    try:
        await scenario.setup()
        emit({"ready": True})
        while line := await asyncio.to_thread(sys.stdin.readline):
            command = json.loads(line)
            scenario.phase = command["action"]
            if scenario.phase == "stop":
                emit({"stopped": True})
                break
            method = getattr(scenario, scenario.phase.replace("-", "_"))
            result = await method()
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
            json.dumps(scenario.evidence, indent=2), encoding="utf-8",
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
