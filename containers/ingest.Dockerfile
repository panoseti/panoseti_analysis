# PANOSETI ingest image — built as Docker, run as Apptainer on Expanse.
# Build context MUST be the repo root (the pypff submodule is in-repo):
#   docker build -t panoseti-analysis-ingest:0.1.0 -f containers/ingest.Dockerfile .
FROM python:3.14-slim

# procps: Nextflow task metrics; git: uv may resolve VCS metadata for the submodule.
RUN apt-get update && apt-get install -y --no-install-recommends procps git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Workspace root + the pypff submodule (a workspace member) + our package and shims.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY pypff/ ./pypff/
COPY src/ ./src/
COPY bin/ ./bin/

# Install the workspace (panoseti_analysis + pypff[zarr] + deps) into /app/.venv.
RUN uv sync --no-dev --frozen && chmod +x bin/*

# pa-* resolve from bin/ shims and/or the venv console scripts.
ENV PATH="/app/bin:/app/.venv/bin:$PATH"
ENV PYTHONNOUSERSITE=1
