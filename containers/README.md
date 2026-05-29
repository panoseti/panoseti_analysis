# Containers

One CPU `ingest` image for Chunk 1 (GPU/Ray images come in a later chunk).

## Build (from the repo root — the build context needs the in-repo pypff submodule)

```bash
git submodule update --init --recursive
docker build -t panoseti-analysis-ingest:0.1.0 -f containers/ingest.Dockerfile .
```

Run locally with Nextflow:

```bash
nextflow run . -profile test,docker --outdir results_smoke \
  -process.container panoseti-analysis-ingest:0.1.0
```

(Or set `process.container` in `conf/docker.config`.)

## Run as Apptainer on Expanse

Build Docker on a machine with Docker, push to a registry, then on Expanse pull once into a
shared cache so concurrent SLURM tasks don't each pull (risk #2):

```bash
# one-time prefetch on Expanse
export APPTAINER_CACHEDIR=/expanse/lustre/scratch/$USER/.apptainer_cache
apptainer pull panoseti-analysis-ingest_0.1.0.sif docker://<registry>/panoseti-analysis-ingest:0.1.0
```

Pin the image (by tag or digest) in `conf/hpc_slurm.config`'s `process.container`, and the
`hpc_slurm` profile (Apptainer enabled, autoMounts) runs every process inside it.

> Note: this image has not yet been built/validated in CI (Docker build is heavy); the
> Dockerfile is provided and reviewed. Build + a `-profile test,docker` smoke is a follow-up.
