ARG PYTHON_IMAGE=python:3.13.15-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.4@sha256:d0a6eca6c669dc7e9c51218707b8438a3d30402733d739dcc00adb3e213e8f5c

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /opt/projetv0-voice

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv lock --check \
    && uv sync --locked --no-dev --no-editable \
    && rm -rf /root/.cache/uv

FROM ${PYTHON_IMAGE} AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/projetv0-voice

ENV PATH="/opt/projetv0-voice/.venv/bin:$PATH"

COPY --from=builder /opt/projetv0-voice/.venv /opt/projetv0-voice/.venv
COPY --chown=0:10001 --chmod=0440 dist/runtime-contract.json ./runtime-contract.json

EXPOSE 8080

USER 10001:10001

ENTRYPOINT ["python", "-m", "projetv0_voice.server"]
