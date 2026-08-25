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
