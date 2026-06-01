# PANOSETI ML image — Ray + PyTorch + CUDA
# Build context MUST be the repo root:
#   docker build -t panoseti-analysis-ml:0.1.0 -f containers/ml.Dockerfile .
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

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

# Install CPU-only torch first (already bundled in the pytorch base image);
# install workspace without dev deps and without reinstalling torch.
RUN uv sync --no-dev --frozen --no-install-package torch && chmod +x bin/*

# pa-* resolve from bin/ shims and/or the venv console scripts.
ENV PATH="/app/bin:/app/.venv/bin:$PATH"
ENV PYTHONNOUSERSITE=1
