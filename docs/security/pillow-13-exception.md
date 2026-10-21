# Temporary Pillow 13 security exception

Owner: Sparra Voice maintainers. Review/expiry: **2026-11-01**.

This operational exception retains official, unmodified
`pipecat-ai[openai,openrouter,silero,websocket]==1.12.0` and development
`pipecat-ai[evals]==1.12.0`, Python `3.13.15`, and uv `0.12.4`, while resolving
`Pillow==13.0.0` and `certifi==2026.10.19`. It selects the published fixes for the
HIGH/CRITICAL Pillow and certifi findings in the refreshed repository and image
security scans. No advisory or security gate is suppressed.

Pipecat's published wheel still declares `Pillow>=11.1.0,<13`. The version-scoped
`tool.uv.override-dependencies` table changes only that named dependency of
`pipecat-ai==1.12.0`; pinned uv `0.12.4` resolves and records this scope in
`uv.lock`. Every other package object and direct pin is retained. This is **not
upstream Pipecat support or dependency consistency**. The actual installed
checker `uv pip check` exits **1** with this sole expected incompatibility:

```text
The package `pipecat-ai` requires `pillow>=11.1.0,<13`, but `13.0.0` is installed
```

An additional mismatch blocks this exception. Existing CI/image checks remain
mandatory and unchanged; this document does not waive a checker or gate.

This exception covers the audio-only Telnyx phone pipeline. Caller messages
become PCMU/PCM audio, DTMF or bounded mark/stop/error frames; unknown events do
not become image frames. STT returns text, the OpenRouter LLM emits text/tool
calls, TTS yields validated PCM audio, and registered tools return JSON plus program
speech. Transport video input/output remain disabled. The actual imported
Pipecat PIL consumers are `llm_context` and `base_output`; their image-message
JPEG conversion and video resizing branches are not enabled by this consumer.
They use `Image.frombytes`, `save`, `resize`, and `tobytes`, rather than the
documented Pillow 13 removals (I-mode PNG behavior, `fromarray` mode,
`get_child_images`, `Image._show`, and removed ImageCms fields).

Local Windows evidence: official CPython 3.13 Pillow wheel installed frozen;
`PIL.Image` and native `PIL._imaging` load alongside actual server/runtime
imports. The bounded 199 existing HTTP/audio/serializer/handshake/native worker
tests, Ruff and mypy (46 source files) pass. No general image-service suite or
live provider call is claimed. Official non-yanked CPython 3.13 Linux/amd64
manylinux wheels are available; Linux execution, complete exact-source hosted
CI/startup/packaging, immutable-image security and application qualification
remain required before promotion. Earlier failed scans remain failed.

Remove this override when a published Pipecat release admits a patched Pillow,
or an official patched Pillow satisfies the published bound; then resolve
normally and requalify. Reconsider the exception whenever this tuple changes
or image/video processing is enabled. Expiry requires explicit review; it does
not authorize a silent rollback to a vulnerable release.
