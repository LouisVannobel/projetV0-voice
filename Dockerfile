ARG PYTHON_IMAGE=python:3.13.15-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.4@sha256:d0a6eca6c669dc7e9c51218707b8438a3d30402733d739dcc00adb3e213e8f5c

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /opt/projetv0-voice

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY scripts/export_runtime_contract.py ./scripts/export_runtime_contract.py
COPY scripts/prepare_tokenizers.py ./scripts/prepare_tokenizers.py
COPY agents/agent-a/ ./agents/agent-a/
COPY deployment-profiles/ ./deployment-profiles/

RUN uv lock --check \
    && uv sync --locked --no-dev --no-editable \
    && python scripts/export_runtime_contract.py \
        --repo-root /opt/projetv0-voice \
        --output-dir /opt/projetv0-voice/build-artifacts \
    && python scripts/prepare_tokenizers.py /opt/projetv0-voice/nltk_data \
    && rm -rf /root/.cache/uv

FROM ${PYTHON_IMAGE} AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get install -y --no-install-recommends --only-upgrade libpcre2-8-0=10.42-1+deb12u2 perl-base=5.36.0-7+deb12u4 \
    && rm -rf /var/lib/apt/lists/*

# The final runtime never installs packages; remove base-image installers and vendored payloads.
RUN PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m pip uninstall --yes pip \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.13 \
    && rm -rf /usr/local/lib/python3.13/ensurepip

WORKDIR /opt/projetv0-voice

ENV PATH="/opt/projetv0-voice/.venv/bin:$PATH"
ENV NLTK_DATA="/opt/projetv0-voice/nltk_data"

COPY --from=builder /opt/projetv0-voice/.venv /opt/projetv0-voice/.venv
COPY --from=builder --chown=0:10001 --chmod=0440 /opt/projetv0-voice/build-artifacts/runtime-contract.json ./runtime-contract.json
COPY --from=builder --chown=0:10001 /opt/projetv0-voice/nltk_data ./nltk_data

RUN find ./nltk_data -type d -exec chmod 0550 {} + \
    && find ./nltk_data -type f -exec chmod 0440 {} +

EXPOSE 8080

USER 10001:10001

ENTRYPOINT ["python", "-m", "projetv0_voice.server"]
