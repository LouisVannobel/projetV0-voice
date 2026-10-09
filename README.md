# projetV0 Voice

Reusable self-hosted Pipecat voice runtime for projetV0 SaaS deployments.

The repository currently contains the pinned Python package baseline. Runtime behavior is added task-by-task against the architecture and contract tests; the Pipecat development runner is intentionally not part of production source.

## Development

```powershell
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Python 3.13.15 and all direct dependencies are pinned. Default development and test commands must not contact live providers.

## Packaging and release

The exporter creates exactly six pre-image artifacts under ignored `dist/`: the
runtime contract, three deployment-profile schemas, the agent tarball, and its
external bundle manifest. CI validates and uploads those six files, then builds
the image through the explicitly enabled offline gate:

```powershell
$env:PROJETV0_CONTAINER_SMOKE = '1'
uv run pytest tests/integration/test_container_smoke.py -q
```

Without `PROJETV0_CONTAINER_SMOKE=1`, the ordinary offline test suite skips the
container smoke. With it, Docker is mandatory and an unavailable daemon is a
failure.

Publication is a main-only manual release. It can run only when both the
dispatch ref and the repository default branch are `main`. The reusable release
uses job-scoped `secrets.GITHUB_TOKEN` for GHCR and builds, scans, and promotes
one immutable `linux/amd64` digest. It does not deploy. The post-release job
regenerates the six files, adds the returned image reference and validated SPDX
SBOM, and uploads one eight-file release handoff without rebuilding or
rescanning the image.

The Pipecat Context Hub is optional workstation documentation tooling:

```powershell
uv tool install "pipecat-ai[cli]==1.12.0" --with pipecat-ai-context-hub
pipecat context-hub install --client codex
pipecat context-hub refresh --framework-version v1.12.0
```

It remains outside the lock, image, and CI and does not replace the pinned 1.12.0
contract fixtures.

## Live qualification boundary

No live key or provider call belongs to Task 11. A later, explicitly authorized
qualification must inject the Telnyx and OpenRouter credentials through
root-owned secret files and also supply the Telnyx webhook public key,
PostgreSQL DSN, AEAD keyring, Call Control connection ID, benchmark from/DID,
WSS/Funnel URL, OTLP endpoint, image/runtime/bundle/inference digests, candidate
and qualified profiles, and infrastructure readiness. None of those live inputs
belongs in commands, reports, fixtures, GitHub, Docker layers, or environment
variable values.

The qualified V1 `telnyx_data_locality: "EU"` field is an operator attestation
for EU Voice API and media routing. It does not attest Telnyx CDR/MDR storage
location or external inference residency. Its existing schema and admission
guards remain in force.

For a fresh Sparra V2 qualification candidate, the operator must stop and close
the old graph and finish its bounded joins before minting `admission_not_before`
once from the current UTC clock, with full microsecond precision. Save it in the
immutable new candidate profile before starting the new run. Preserve that exact
floor, run identity and profile fingerprint on restart; do not derive a new floor
at startup, backdate it, or regenerate a remaining allowance from an old run.
The floor must precede `expires_at`. A new initiation must have a verified body
timestamp at or after the floor, and the current clock must have reached it.
Existing same-store duplicate receipts, terminal cleanup and media authority keep
their existing rules. An out-of-order answered placeholder consumes no call unit;
it cannot make a pre-floor initiation create a durable admission. This timestamp
gate does not prove global carrier identity or deduplication across separate stores.
Legacy candidate profiles may omit the field with unchanged canonical fingerprints;
native Sparra operation-contract V2 candidate composition requires it.

Linked Sparra calls use Pipecat's native WebSocket `session_timeout=300` to bound
each session to five minutes. The existing session timeout handler closes audio
admission and uses the call owner's terminal cleanup and hangup command. Generic
voice deployments keep the native default of no session timeout. This duration
bound is not a monetary spending cap.
