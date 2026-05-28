# Build from the parent directory so both pypff and the pipeline are in context:
#   docker build -t panoseti-zarr-pipeline:0.3.0 -f panoseti_zarr_seqera/Dockerfile .
#
# Run a single calibration step manually:
#   docker run --rm -v $PWD:/data panoseti-zarr-pipeline:0.3.0 \
#       calibrate_ph /data/L0/store.zarr /data/L1/store_L1.zarr

FROM python:3.13-slim

# procps provides `ps`, required by Nextflow for task metrics collection
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy pypff source (required as editable dep)
COPY pypff/ /pypff/

# Copy pipeline package
COPY panoseti_zarr_seqera/pyproject.toml panoseti_zarr_seqera/uv.lock ./
COPY panoseti_zarr_seqera/src/ src/
COPY panoseti_zarr_seqera/bin/ bin/

# Install all dependencies using uv (respects uv.lock)
RUN uv sync --no-dev

# Make bin/ scripts executable and expose on PATH
RUN chmod +x bin/*
ENV PATH="/app/bin:/app/.venv/bin:$PATH"

LABEL org.opencontainers.image.title="panoseti-zarr-pipeline" \
      org.opencontainers.image.version="0.3.0" \
      org.opencontainers.image.description="PFF → Zarr v3 conversion + calibration"
