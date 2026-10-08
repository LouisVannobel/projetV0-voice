# Agent instructions

Use the repository-pinned Python and dependencies. Canonical local checks are:

```powershell
uv lock --check
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Prefer supported Pipecat primitives and documented upstream interfaces. Add custom code only for projetV0-specific contracts, policy, security, lifecycle, or persistence.

Default tests must be offline. Never put live Telnyx, OpenRouter, PostgreSQL, or other credentials in source, tests, fixtures, logs, commits, or task reports. Do not use live services unless a qualification task explicitly authorizes them.

## Packaging and release boundaries

- `dist/` stays ignored. CI exports and validates exactly six pre-image artifacts; the manual post-release job adds `image-reference.txt` and `sbom.spdx.json` to form one eight-file release handoff.
- Ordinary `uv run pytest -q` stays offline and skips the container smoke. Run it as an image gate only with `PROJETV0_CONTAINER_SMOKE=1`; Docker must then be available and failure must not become a skip.
- The release is a main-only manual release guarded by both a `main` dispatch ref and `main` as the repository default branch. Use only job-scoped `secrets.GITHUB_TOKEN`; never add a PAT or live provider secret.
- Pipecat Context Hub is optional developer documentation tooling installed from `pipecat-ai[cli]==1.12.0`; keep it outside the lock, image, and CI.
- No live key or provider call belongs to Task 11. Post-Task-11 qualification additionally requires the Telnyx webhook public key, PostgreSQL DSN, AEAD keyring, Call Control connection ID, benchmark from/DID, WSS/Funnel URL, OTLP endpoint, image/runtime/bundle/inference digests, candidate and qualified profiles, and infrastructure readiness through a separately authorized operator gate.
