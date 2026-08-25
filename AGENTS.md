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
